-- =====================================================================
-- tm_map_engine.vhd -- one map (16 inner steps) over a RESIDENT RNG window.
-- ---------------------------------------------------------------------
-- The core datapath unit. Given a 128-byte state, the map's 2 KB RNG window
-- (resident -> reads are combinational muxes, NOT BRAM, so no II=2 penalty),
-- and the 16-bit nibble selector, it runs the 16 data-dependent inner steps
-- and returns the new state.
--
-- GROUP_UNROLL = G chains G inner steps combinationally per clock (the RNG
-- reads are combinational into `win`, which is what makes >1 step/clock
-- possible). Cycles per map = 16/G. G must divide 16 (1,2,4,8,16). G is the
-- knob the P&R Fmax-vs-utilization sweep tunes; G=16 ("1 clock per map") is
-- the deep end and almost certainly past the Fmax knee.
--
-- Reuses apply_map / rng_vec / pick_alg / rng_advance from tm_pkg, so it is
-- bit-identical to the reference run_one_map modulo the alg2/alg5 PARITY GATE.
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.tm_pkg.all;

entity tm_map_engine is
  generic (
    GROUP_UNROLL : integer := 4;      -- inner steps per clock; must divide 16
    USE_BLEND    : boolean := false    -- false = apply_map (flat 8:1, the golden
                                       -- reference); true = apply_map_blend
                                       -- (shallower op-pair tree -> higher Fmax).
                                       -- Proven bit-identical by tb_apply_blend.
  );
  port (
    clk       : in  std_logic;
    rst       : in  std_logic;
    start     : in  std_logic;                       -- pulse: begin this map
    state_in  : in  state_t;
    win       : in  win_arr_t;                       -- resident map RNG window
    nibble    : in  std_logic_vector(15 downto 0);   -- map nibble selector
    busy      : out std_logic;
    done      : out std_logic;                       -- 1-cycle pulse
    state_out : out state_t
  );
end entity;

architecture rtl of tm_map_engine is
  signal st     : state_t;
  -- RNG offset as the (f,c) counter pair: off = 128*f + c, f = #byte-algs,
  -- c = #carry-algs(alg2/5). c<=15<128 so off = {f, c} is concatenation, no
  -- adder; the touched set is the bounded triangular <=136-row superset.
  signal foff   : integer range 0 to INNER := 0;     -- f: +128 advances
  signal coff   : integer range 0 to INNER := 0;     -- c: +1 advances
  signal nib    : std_logic_vector(15 downto 0);
  signal icnt   : integer range 0 to INNER := 0;     -- inner step index
  signal active : std_logic := '0';
begin
  busy      <= active;
  state_out <= st;

  process(clk)
    variable v_st  : state_t;
    variable v_f   : integer range 0 to INNER+1;
    variable v_c   : integer range 0 to INNER+1;
    variable v_nib : std_logic_vector(15 downto 0);
    variable v_i   : integer range 0 to INNER;
    variable aid   : std_logic_vector(2 downto 0);
    variable adv   : integer;
    variable carry : std_logic;
    variable rr    : tworow_t;                    -- shared per-step row slice
  begin
    if rising_edge(clk) then
      if rst = '1' then
        active <= '0';
        done   <= '0';
        icnt   <= 0;
      else
        done <= '0';
        if active = '0' then
          if start = '1' then
            st     <= state_in;
            foff   <= 0;
            coff   <= 0;
            nib    <= nibble;
            icnt   <= 0;
            active <= '1';
          end if;
        else
          -- chain GROUP_UNROLL inner steps combinationally this clock.
          v_st  := st;
          v_f   := foff;
          v_c   := coff;
          v_nib := nib;
          v_i   := icnt;
          for g in 0 to GROUP_UNROLL-1 loop
            aid   := pick_alg(v_st, v_i, v_nib(15));
            rr    := rng_rows(win, v_f);             -- ONE 16:1 row-select/step
            carry := rr(v_c)(7);                     -- alg2/alg5 boundary bit
            if USE_BLEND then
              v_st := apply_map_blend(v_st, rng_vec_r(rr, v_c), rng_vec_fwd_r(rr, v_c), aid, carry);
            else
              v_st := apply_map(v_st, rng_vec_r(rr, v_c), rng_vec_fwd_r(rr, v_c), aid, carry);
            end if;
            adv   := rng_advance(aid);
            if    adv = 128 then v_f := v_f + 1;     -- byte-alg
            elsif adv = 1   then v_c := v_c + 1;     -- alg2/alg5
            end if;                                  -- alg7: no advance
            v_nib := v_nib(14 downto 0) & '0';
            v_i   := v_i + 1;
          end loop;
          st   <= v_st;
          foff <= v_f;
          coff <= v_c;
          nib  <= v_nib;

          if v_i >= INNER then
            active <= '0';
            done   <= '1';
            icnt   <= 0;
          else
            icnt <= v_i;
          end if;
        end if;
      end if;
    end if;
  end process;

end architecture;
