-- =====================================================================
-- tb_screen_top.vhd -- integration smoke for the limited-memory pipeline.
-- Drives one candidate with stub schedule (seeds/nibbles), zero expansion
-- and zero world data, and checks the pipeline walks expand->27 maps->
-- checksum and pulses done. Proves rng_gen + map_engine + FSM compose and
-- run end to end. (Value-parity vs the software screen is the documented
-- gate; needs a real key_schedule + world data + reference flag.)
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.tm_pkg.all;

entity tb_screen_top is
  generic (UNROLL_K : integer := 8; GROUP_UNROLL : integer := 4);
end entity;

architecture sim of tb_screen_top is
  signal clk   : std_logic := '0';
  signal rst   : std_logic := '1';
  signal start : std_logic := '0';
  signal key   : std_logic_vector(31 downto 0) := x"2ca5b42d";
  signal data  : std_logic_vector(31 downto 0) := x"00000001";
  signal seeds : seed_arr_t;
  signal nibs  : nib_arr_t;
  signal expv  : state_t := (others => (others => '0'));
  signal wx    : slv1024 := (others => '0');
  signal wm    : slv1024 := (others => '0');
  signal busy  : std_logic;
  signal done  : std_logic;
  signal flag  : std_logic_vector(7 downto 0);
  signal cyc   : integer := 0;
begin
  clk <= not clk after 5 ns;

  dut : entity work.tm_screen_top
    generic map (UNROLL_K => UNROLL_K, GROUP_UNROLL => GROUP_UNROLL)
    port map (clk=>clk, rst=>rst, start=>start, key=>key, data=>data,
              seeds=>seeds, nibbles=>nibs, exp_vec=>expv,
              world_xor=>wx, world_mask=>wm,
              busy=>busy, done=>done, screen_flag=>flag);

  -- stub schedule: distinct per-map seeds, nonzero nibbles to exercise dispatch
  sched : process
  begin
    for m in 0 to NUM_MAPS-1 loop
      seeds(m) <= std_logic_vector(to_unsigned((m*2654 + 305) mod 65536, 16));
      nibs(m)  <= std_logic_vector(to_unsigned((m*40503) mod 65536, 16));
    end loop;
    wait;
  end process;

  cnt : process(clk)
  begin
    if rising_edge(clk) then
      if busy = '1' then cyc <= cyc + 1; end if;
    end if;
  end process;

  stim : process
  begin
    wait for 23 ns; rst <= '0';
    wait until rising_edge(clk);
    start <= '1'; wait until rising_edge(clk); start <= '0';

    wait until done = '1' for 200 us;
    assert done = '1' report "FAIL: pipeline did not finish" severity failure;
    report "PASS: screen_top end-to-end done. busy_cycles=" & integer'image(cyc)
         & " flag=0x" & to_hstring(flag)
         & " (UNROLL_K=" & integer'image(UNROLL_K)
         & " GROUP_UNROLL=" & integer'image(GROUP_UNROLL) & ")"
      severity note;
    wait;
  end process;

end architecture;
