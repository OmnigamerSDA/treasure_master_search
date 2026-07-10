-- =====================================================================
-- tm_map_engine_w.vhd -- W-lane lockstep map engine (the gap-killer).
-- ---------------------------------------------------------------------
-- Direct FPGA transliteration of the production GPU/CPU inner loop:
--   CUDA tm_cuda_raceway.cuh (ILP=6) and CPU tm_avx512_r512s_8 (_x8/_x10):
--   W candidates advance in lockstep through one map -- shared nibble selector
--   and RNG window, but each lane has its OWN state and OWN offset (rng_offset
--   on GPU, local_pos on CPU) and dispatches its own alg per inner step.
--
-- vs the per-candidate map_engine + FSM turnaround, this removes the inter-
-- candidate gap: all W lanes run every clock. GROUP_UNROLL chains G inner
-- steps/clock; LANES (=W) is the orthogonal parallelism knob. On FPGA the
-- lanes are physical (W datapaths), bounded by the W window reads -- which is
-- why each lane does exactly ONE window read feeding the whole op-pair blend
-- tree (the blmerge2 "single-load" / register-pruning lesson).
--
-- Differences from GPU/CPU that FAVOR FPGA:
--   * no __shfl_sync / LDS+barrier for the alg_id source (flat state -> wiring)
--   * no register-file ceiling on W (area/read-ports bound instead)
--   * butterflies {2,5} fold into the blend tree (no branch) via apply_map_blend
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.tm_pkg.all;

entity tm_map_engine_w is
  generic (
    LANES        : integer := 8;
    GROUP_UNROLL : integer := 4   -- must divide 16
  );
  port (
    clk       : in  std_logic;
    rst       : in  std_logic;
    start     : in  std_logic;
    state_in  : in  state_vec_t(0 to LANES-1);
    win       : in  win_arr_t;
    nibble    : in  std_logic_vector(15 downto 0);
    busy      : out std_logic;
    done      : out std_logic;
    state_out : out state_vec_t(0 to LANES-1)
  );
end entity;

architecture rtl of tm_map_engine_w is
  -- per-lane RNG offset as (f,c) counters: off = 128*f + c = {f,c} concat
  -- (c<=15<128, no adder); see tm_map_engine for the triangular-set rationale.
  type fcv_t is array (0 to LANES-1) of integer range 0 to INNER+1;
  signal st_v   : state_vec_t(0 to LANES-1);
  signal foff_v : fcv_t;
  signal coff_v : fcv_t;
  signal nib    : std_logic_vector(15 downto 0);
  signal icnt   : integer range 0 to INNER := 0;
  signal active : std_logic := '0';
begin
  busy      <= active;
  state_out <= st_v;

  process(clk)
    variable vst   : state_vec_t(0 to LANES-1);
    variable vf    : fcv_t;
    variable vc    : fcv_t;
    variable v_nib : std_logic_vector(15 downto 0);
    variable v_i   : integer range 0 to INNER;
    variable aid   : std_logic_vector(2 downto 0);
    variable adv   : integer;
    variable carry : std_logic;
    variable rr    : tworow_t;                    -- per-lane shared row slice
  begin
    if rising_edge(clk) then
      if rst = '1' then
        active <= '0'; done <= '0'; icnt <= 0;
      else
        done <= '0';
        if active = '0' then
          if start = '1' then
            st_v   <= state_in;
            for l in 0 to LANES-1 loop foff_v(l) <= 0; coff_v(l) <= 0; end loop;
            nib    <= nibble;
            icnt   <= 0;
            active <= '1';
          end if;
        else
          vst   := st_v;
          vf    := foff_v;
          vc    := coff_v;
          v_nib := nib;
          v_i   := icnt;
          for g in 0 to GROUP_UNROLL-1 loop
            for l in 0 to LANES-1 loop
              aid      := pick_alg(vst(l), v_i, v_nib(15));      -- lane's own byte
              rr       := rng_rows(win, vf(l));                  -- ONE row-select/lane
              carry    := rr(vc(l))(7);                          -- shared window
              vst(l)   := apply_map_blend(vst(l), rng_vec_r(rr, vc(l)), rng_vec_fwd_r(rr, vc(l)), aid, carry);
              adv      := rng_advance(aid);
              if    adv = 128 then vf(l) := vf(l) + 1;
              elsif adv = 1   then vc(l) := vc(l) + 1;
              end if;
            end loop;
            v_i   := v_i + 1;
            v_nib := v_nib(14 downto 0) & '0';                   -- shared nibble
          end loop;
          st_v   <= vst;
          foff_v <= vf;
          coff_v <= vc;
          nib    <= v_nib;
          if v_i >= INNER then
            active <= '0'; done <= '1'; icnt <= 0;
          else
            icnt <= v_i;
          end if;
        end if;
      end if;
    end if;
  end process;
end architecture;
