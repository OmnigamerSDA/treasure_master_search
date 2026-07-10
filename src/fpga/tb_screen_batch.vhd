-- =====================================================================
-- tb_screen_batch.vhd -- throughput demonstrator for the map-major batch top.
-- Drives one batch and reports busy cycles + per-candidate amortized cost,
-- which should be ~compute-bound (27*(16/G+gap)) with generation overlapped,
-- NOT the ~3648-7185 of the one-shot per-candidate regen.
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.tm_pkg.all;

entity tb_screen_batch is
  generic (BATCH : integer := 64; UNROLL_K : integer := 16; GROUP_UNROLL : integer := 4);
end entity;

architecture sim of tb_screen_batch is
  signal clk  : std_logic := '0';
  signal rst  : std_logic := '1';
  signal start: std_logic := '0';
  signal key  : std_logic_vector(31 downto 0) := x"2ca5b42d";
  signal dbase: std_logic_vector(31 downto 0) := x"00000000";
  signal seeds: seed_arr_t;
  signal nibs : nib_arr_t;
  signal expv : state_t := (others => (others => '0'));
  signal wx   : slv1024 := (others => '0');
  signal wm   : slv1024 := (others => '0');
  signal busy : std_logic;
  signal done : std_logic;
  signal flags: byte_vec_t(0 to BATCH-1);
  signal cyc  : integer := 0;
begin
  clk <= not clk after 5 ns;

  dut : entity work.tm_screen_batch
    generic map (BATCH=>BATCH, UNROLL_K=>UNROLL_K, GROUP_UNROLL=>GROUP_UNROLL)
    port map (clk=>clk, rst=>rst, start=>start, key=>key, data_base=>dbase,
              seeds=>seeds, nibbles=>nibs, exp_vec=>expv, world_xor=>wx, world_mask=>wm,
              busy=>busy, done=>done, flags=>flags);

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

    wait until done = '1' for 2 ms;
    assert done = '1' report "FAIL: batch did not finish" severity failure;
    report "PASS: batch done. BATCH=" & integer'image(BATCH)
         & " busy_cycles=" & integer'image(cyc)
         & " per_candidate=" & integer'image(cyc / BATCH)
         & " (UNROLL_K=" & integer'image(UNROLL_K)
         & " GROUP_UNROLL=" & integer'image(GROUP_UNROLL) & ")"
      severity note;
    wait;
  end process;

end architecture;
