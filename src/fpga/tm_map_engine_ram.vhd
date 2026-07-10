-- =====================================================================
-- tm_map_engine_ram.vhd -- tm_map_engine with the window held in an internal
-- runtime-writable DISTRIBUTED RAM instead of a 16384-bit input port.
-- ---------------------------------------------------------------------
-- Identical FSM + datapath to tm_map_engine; the only change is the window
-- source: 16 rows x 1024 bits in a distributed RAM (ram_style="distributed"),
-- loaded once per key/map through wr_en/wr_row/wr_data (the "preprocessing
-- fixes a table, not a bitstream" model), read ASYNCHRONOUSLY by (f,c) so the
-- GROUP_UNROLL combinational chain is preserved. Parity-locked to tm_8 in
-- tb_map_parity_ram (same map_vectors.txt as tb_map_parity).
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.tm_pkg.all;

entity tm_map_engine_ram is
  generic (
    GROUP_UNROLL : integer := 4;
    USE_BLEND    : boolean := false
  );
  port (
    clk       : in  std_logic;
    rst       : in  std_logic;
    start     : in  std_logic;
    state_in  : in  state_t;
    nibble    : in  std_logic_vector(15 downto 0);
    -- window LOAD interface: write one 128-byte row (r=0..15) per cycle
    wr_en     : in  std_logic;
    wr_row    : in  std_logic_vector(3 downto 0);
    wr_data   : in  slv1024;
    busy      : out std_logic;
    done      : out std_logic;
    state_out : out state_t
  );
end entity;

architecture rtl of tm_map_engine_ram is
  type rows_t is array (0 to 15) of slv1024;
  signal ram : rows_t := (others => (others => '0'));
  attribute ram_style : string;
  attribute ram_style of ram : signal is "distributed";

  signal st     : state_t;
  signal foff   : integer range 0 to INNER := 0;
  signal coff   : integer range 0 to INNER := 0;
  signal nib    : std_logic_vector(15 downto 0);
  signal icnt   : integer range 0 to INNER := 0;
  signal active : std_logic := '0';
begin
  busy      <= active;
  state_out <= st;

  -- preprocessing load: synchronous write, distributed RAM
  process(clk) begin
    if rising_edge(clk) then
      if wr_en = '1' then
        ram(to_integer(unsigned(wr_row))) <= wr_data;
      end if;
    end if;
  end process;

  process(clk)
    variable v_st  : state_t;
    variable v_f   : integer range 0 to INNER+1;
    variable v_c   : integer range 0 to INNER+1;
    variable v_nib : std_logic_vector(15 downto 0);
    variable v_i   : integer range 0 to INNER;
    variable aid   : std_logic_vector(2 downto 0);
    variable adv   : integer;
    variable carry : std_logic;
    variable rr    : tworow_t;
    variable rowf, rowf1 : slv1024;
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
          v_st  := st;
          v_f   := foff;
          v_c   := coff;
          v_nib := nib;
          v_i   := icnt;
          for g in 0 to GROUP_UNROLL-1 loop
            aid   := pick_alg(v_st, v_i, v_nib(15));
            -- async RAM read of rows f and f+1 (16:1 by f), then align by c
            rowf  := ram(v_f mod 16);
            rowf1 := ram((v_f + 1) mod 16);
            for j in 0 to NUM_BYTES-1 loop
              rr(j)             := rowf (8*j+7 downto 8*j);
              rr(NUM_BYTES + j) := rowf1(8*j+7 downto 8*j);
            end loop;
            carry := rr(v_c)(7);                     -- alg2/alg5 boundary bit
            if USE_BLEND then
              v_st := apply_map_blend(v_st, rng_vec_r(rr, v_c), rng_vec_fwd_r(rr, v_c), aid, carry);
            else
              v_st := apply_map(v_st, rng_vec_r(rr, v_c), rng_vec_fwd_r(rr, v_c), aid, carry);
            end if;
            adv   := rng_advance(aid);
            if    adv = 128 then v_f := v_f + 1;
            elsif adv = 1   then v_c := v_c + 1;
            end if;
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
