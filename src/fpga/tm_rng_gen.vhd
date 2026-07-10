-- =====================================================================
-- tm_rng_gen.vhd -- table-free per-map RNG window generator.
-- ---------------------------------------------------------------------
-- Walks run_rng (combinational rng_step, NO 128 KB ROM) from a map's start
-- seed and fills a WIN_LEN-byte window. UNROLL_K rng_steps are chained
-- combinationally per clock, so a window fills in WIN_LEN/UNROLL_K cycles.
--
-- This is the unit both memory strategies use:
--   * candidate-major: generate all 27 windows once per key into a (shared)
--     store, reuse across every candidate (more memory, ~free reuse);
--   * map-major batched: regenerate ONE window per map per candidate-batch
--     into a small resident buffer (limited memory, modest recompute).
-- See README "Why not NES-scant memory?" for the tradeoff.
--
-- UNROLL_K must divide WIN_LEN (use 1,2,4,8,16,...). Validated against the
-- reference run_rng stream in tb_rng_gen.vhd.
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.tm_pkg.all;

entity tm_rng_gen is
  generic (
    UNROLL_K : integer := 8
  );
  port (
    clk      : in  std_logic;
    rst      : in  std_logic;
    start    : in  std_logic;                       -- pulse to begin a fill
    seed_in  : in  std_logic_vector(15 downto 0);   -- map start seed
    busy     : out std_logic;
    done     : out std_logic;                       -- 1-cycle pulse when full
    win_out  : out win_arr_t                        -- the filled window
  );
end entity;

architecture rtl of tm_rng_gen is
  signal seed_r : std_logic_vector(15 downto 0);
  signal pos    : integer range 0 to WIN_LEN := 0;
  signal win_r  : win_arr_t := (others => (others => '0'));
  signal active : std_logic := '0';
begin

  busy    <= active;
  win_out <= win_r;

  process(clk)
    variable sd  : std_logic_vector(15 downto 0);
    variable ns  : std_logic_vector(15 downto 0);
    variable ob  : std_logic_vector(7 downto 0);
  begin
    if rising_edge(clk) then
      if rst = '1' then
        active <= '0';
        done   <= '0';
        pos    <= 0;
      else
        done <= '0';
        if active = '0' then
          if start = '1' then
            seed_r <= seed_in;
            pos    <= 0;
            active <= '1';
          end if;
        else
          -- chain UNROLL_K run_rng steps this cycle, writing K bytes.
          sd := seed_r;
          for k in 0 to UNROLL_K-1 loop
            rng_step(sd, ns, ob);
            win_r(pos + k) <= ob;       -- pos is K-aligned (K | WIN_LEN)
            sd := ns;
          end loop;
          seed_r <= sd;

          if pos + UNROLL_K >= WIN_LEN then
            active <= '0';
            done   <= '1';
            pos    <= 0;
          else
            pos <= pos + UNROLL_K;
          end if;
        end if;
      end if;
    end if;
  end process;

end architecture;
