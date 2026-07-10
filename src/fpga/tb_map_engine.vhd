-- =====================================================================
-- tb_map_engine.vhd -- functional-transparency + timing check for GROUP_UNROLL.
-- ---------------------------------------------------------------------
-- GROUP_UNROLL only changes the per-clock schedule, never the result. Two
-- engines (G=1 and G=GCHK) fed identical (state, window, nibble) must produce
-- bit-identical state_out; the G=1 engine takes 16 cycles, the G=GCHK engine
-- 16/GCHK. This validates the unroll without needing run_one_map ref vectors
-- (the datapath value-correctness is gated separately by the alg2/5 parity TB).
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.tm_pkg.all;

entity tb_map_engine is
  generic (GCHK : integer := 4);
end entity;

architecture sim of tb_map_engine is
  signal clk    : std_logic := '0';
  signal rst    : std_logic := '1';
  signal start  : std_logic := '0';
  signal st_in  : state_t;
  signal win    : win_arr_t;
  signal nib    : std_logic_vector(15 downto 0) := x"A5A5";

  signal busy1, done1, busyG, doneG : std_logic;
  signal out1, outG : state_t;

  signal c1, cG : integer := 0;  -- cycle counts while busy
begin
  clk <= not clk after 5 ns;

  e1 : entity work.tm_map_engine generic map (GROUP_UNROLL => 1)
    port map (clk=>clk, rst=>rst, start=>start, state_in=>st_in, win=>win,
              nibble=>nib, busy=>busy1, done=>done1, state_out=>out1);

  eG : entity work.tm_map_engine generic map (GROUP_UNROLL => GCHK)
    port map (clk=>clk, rst=>rst, start=>start, state_in=>st_in, win=>win,
              nibble=>nib, busy=>busyG, done=>doneG, state_out=>outG);

  -- deterministic stimulus that exercises the data-dependent alg dispatch
  init : process
  begin
    for i in 0 to NUM_BYTES-1 loop
      st_in(i) <= std_logic_vector(to_unsigned((i*7 + 1) mod 256, 8));
    end loop;
    for i in 0 to WIN_LEN-1 loop
      win(i) <= std_logic_vector(to_unsigned((i*131 + 17) mod 256, 8));
    end loop;
    wait;
  end process;

  cnt : process(clk)
  begin
    if rising_edge(clk) then
      if busy1 = '1' then c1 <= c1 + 1; end if;
      if busyG = '1' then cG <= cG + 1; end if;
    end if;
  end process;

  stim : process
    variable mism : integer := 0;
  begin
    wait for 23 ns; rst <= '0';
    wait until rising_edge(clk);
    start <= '1'; wait until rising_edge(clk); start <= '0';

    -- 16 inner steps max -> both engines finish well within 1 us; state_out
    -- latches and holds after each engine's done pulse.
    wait for 1 us;
    assert c1 = INNER
      report "FAIL: G=1 took " & integer'image(c1) & " cycles, expected "
           & integer'image(INNER) severity failure;
    assert cG = INNER/GCHK
      report "FAIL: G=" & integer'image(GCHK) & " took " & integer'image(cG)
           & " cycles, expected " & integer'image(INNER/GCHK) severity failure;

    for i in 0 to NUM_BYTES-1 loop
      if out1(i) /= outG(i) then mism := mism + 1; end if;
    end loop;

    assert mism = 0
      report "FAIL: G=1 vs G=" & integer'image(GCHK)
           & " differ in " & integer'image(mism) & " bytes" severity failure;

    report "PASS: map_engine G=1 == G=" & integer'image(GCHK)
         & " (out identical); cycles G1=" & integer'image(c1)
         & " G" & integer'image(GCHK) & "=" & integer'image(cG)
      severity note;
    wait;
  end process;

end architecture;
