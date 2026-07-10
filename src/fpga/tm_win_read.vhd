-- =====================================================================
-- tm_win_read.vhd -- PROTOTYPE: window storage + (f,c) read, two styles.
-- ---------------------------------------------------------------------
-- Isolates the "fixed-key window as a runtime-writable table" question. The
-- window is organized as 16 rows x 128 bytes (1024-bit words). A preprocessing
-- LOAD writes the 16 rows (wr_en/wr_row/wr_data); the datapath READS the two
-- rows f, f+1 asynchronously and aligns by c -- exactly rng_rows + rng_vec_r/
-- fwd_r/carry from tm_pkg, but with the window held INTERNALLY instead of an
-- input port. Async read keeps the read combinational (chainable under G-unroll).
--
--   USE_RAM = false : 16x1024 register array + 16:1 read mux  (models TODAY,
--                     where the engine muxes a 16384-bit window bus)
--   USE_RAM = true  : same array tagged ram_style="distributed" -> SLICEM
--                     LUTRAM: the per-key table is RAM CONTENTS, written at
--                     load, no bitstream reload, no re-synthesis.
--
-- The ONLY difference between the two is the ram_style attribute. Value-parity
-- vs the tm_pkg reference reads is checked in tb_win_read for BOTH styles.
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.tm_pkg.all;

entity tm_win_read is
  generic ( USE_RAM : boolean := false );
  port (
    clk       : in  std_logic;
    -- preprocessing load: one 128-byte row per write (row r = 0..15)
    wr_en     : in  std_logic;
    wr_row    : in  std_logic_vector(3 downto 0);
    wr_data   : in  slv1024;
    -- data-dependent read by the (f,c) offset pair
    f         : in  std_logic_vector(3 downto 0);
    c         : in  std_logic_vector(3 downto 0);
    rev_out   : out slv1024;     -- reversed vector (alg0/1/3/4)
    fwd_out   : out slv1024;     -- forward  vector (alg6)
    carry_out : out std_logic    -- alg2/alg5 boundary bit
  );
end entity;

architecture rtl of tm_win_read is
  type rows_t is array (0 to 15) of slv1024;
  signal row_f_s, row_fp1_s : slv1024;

  function fp1(x : std_logic_vector(3 downto 0)) return integer is
  begin
    return (to_integer(unsigned(x)) + 1) mod 16;   -- f=15 -> row 0 (unused slice)
  end function;
begin

  -- ---- storage + async row read: RAM vs register-array, same behavior ----
  g_ram : if USE_RAM generate
    signal ram : rows_t := (others => (others => '0'));
    attribute ram_style : string;
    attribute ram_style of ram : signal is "distributed";
  begin
    process(clk) begin
      if rising_edge(clk) then
        if wr_en = '1' then ram(to_integer(unsigned(wr_row))) <= wr_data; end if;
      end if;
    end process;
    row_f_s   <= ram(to_integer(unsigned(f)));
    row_fp1_s <= ram(fp1(f));
  end generate;

  g_reg : if not USE_RAM generate
    signal regs : rows_t := (others => (others => '0'));
    attribute ram_style : string;
    attribute ram_style of regs : signal is "registers";   -- force FF storage
  begin
    process(clk) begin
      if rising_edge(clk) then
        if wr_en = '1' then regs(to_integer(unsigned(wr_row))) <= wr_data; end if;
      end if;
    end process;
    row_f_s   <= regs(to_integer(unsigned(f)));
    row_fp1_s <= regs(fp1(f));
  end generate;

  -- ---- common align (combinational): reversed / forward / carry ----
  process(all)
    variable rr : tworow_t;
    variable ci : integer range 0 to 15;
  begin
    for j in 0 to NUM_BYTES-1 loop
      rr(j)             := row_f_s(8*j+7 downto 8*j);
      rr(NUM_BYTES + j) := row_fp1_s(8*j+7 downto 8*j);
    end loop;
    ci := to_integer(unsigned(c));
    for i in 0 to NUM_BYTES-1 loop
      rev_out(8*i+7 downto 8*i) <= rr(ci + 127 - i);   -- rng_vec_r
      fwd_out(8*i+7 downto 8*i) <= rr(ci + i);         -- rng_vec_fwd_r
    end loop;
    carry_out <= rr(ci)(7);                            -- rng_carry_bit
  end process;

end architecture;
