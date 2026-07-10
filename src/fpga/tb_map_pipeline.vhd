-- tb_map_pipeline.vhd -- verify the map pipeline: (1) output == sequential
-- stage-by-stage application, (2) II = one candidate per ~16/GROUP_UNROLL cyc.
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library work;
use work.tm_pkg.all;

entity tb_map_pipeline is
  generic (N_STAGES : integer := 4; GU : integer := 4);
end entity;

architecture sim of tb_map_pipeline is
  signal clk : std_logic := '0';
  signal rst : std_logic := '1';
  signal in_valid : std_logic := '0';
  signal in_state : state_t;
  signal wins : win_vec_t(0 to N_STAGES-1);
  signal nibs : slv16_vec_t(0 to N_STAGES-1);
  signal tick : std_logic;
  signal out_valid : std_logic;
  signal out_state : state_t;

  -- reference single engine
  signal r_start : std_logic := '0';
  signal r_in    : state_t;
  signal r_win   : win_arr_t;
  signal r_nib   : std_logic_vector(15 downto 0);
  signal r_busy, r_done : std_logic;
  signal r_out   : state_t;

  signal cand0 : state_t;
  signal ii_meas : time := 0 ns;
  signal captured : std_logic := '0';
  signal cap_out : state_t;
begin
  clk <= not clk after 5 ns;

  dut : entity work.tm_map_pipeline
    generic map (N_STAGES=>N_STAGES, GROUP_UNROLL=>GU)
    port map (clk=>clk, rst=>rst, in_valid=>in_valid, in_state=>in_state,
              wins=>wins, nibs=>nibs, tick=>tick, out_valid=>out_valid, out_state=>out_state);

  ref : entity work.tm_map_engine
    generic map (GROUP_UNROLL=>GU)
    port map (clk=>clk, rst=>rst, start=>r_start, state_in=>r_in, win=>r_win,
              nibble=>r_nib, busy=>r_busy, done=>r_done, state_out=>r_out);

  init : process
  begin
    for s in 0 to N_STAGES-1 loop
      for i in 0 to WIN_LEN-1 loop
        wins(s)(i) <= std_logic_vector(to_unsigned((i*131 + 17 + s*61) mod 256, 8));
      end loop;
      nibs(s) <= std_logic_vector(to_unsigned((s*40503 + 12345) mod 65536, 16));
    end loop;
    for i in 0 to NUM_BYTES-1 loop
      cand0(i) <= std_logic_vector(to_unsigned((i*7 + 1) mod 256, 8));
    end loop;
    wait;
  end process;

  -- measure II between tick pulses
  iimeas : process(clk)
    variable prev : time := 0 ns;
  begin
    if rising_edge(clk) then
      if tick = '1' then
        if prev /= 0 ns then ii_meas <= now - prev; end if;
        prev := now;
      end if;
    end if;
  end process;

  -- capture the first real candidate as it exits (out_valid is a 1-cyc pulse)
  cap : process(clk)
  begin
    if rising_edge(clk) then
      if out_valid = '1' and captured = '0' then
        cap_out  <= out_state;
        captured <= '1';
      end if;
    end if;
  end process;

  stim : process
    variable ref_s : state_t;
    variable mism  : integer := 0;
  begin
    wait for 23 ns; rst <= '0';
    -- wait for first tick (pipeline primed)
    wait until tick = '1';
    wait until rising_edge(clk);

    -- inject ONE real candidate at the next shift
    in_state <= cand0; in_valid <= '1';
    wait until tick = '1';
    wait until rising_edge(clk);
    in_valid <= '0';

    -- compute reference: cand0 through stages 0..N-1 sequentially
    r_in <= cand0;
    for s in 0 to N_STAGES-1 loop
      r_win <= wins(s); r_nib <= nibs(s);
      wait until rising_edge(clk);
      r_start <= '1'; wait until rising_edge(clk); r_start <= '0';
      wait until r_done = '1' for 2 us;
      wait until rising_edge(clk);
      if s < N_STAGES-1 then r_in <= r_out; end if;
    end loop;
    for i in 0 to NUM_BYTES-1 loop ref_s(i) := r_out(i); end loop;

    -- the injected candidate was captured concurrently as it exited
    wait until captured = '1' for 5 us;
    assert captured = '1' report "FAIL: no valid output" severity failure;

    for i in 0 to NUM_BYTES-1 loop
      if cap_out(i) /= ref_s(i) then mism := mism + 1; end if;
    end loop;

    assert mism = 0
      report "FAIL: pipeline out != sequential ref in " & integer'image(mism) & " bytes"
      severity failure;
    report "PASS: map_pipeline (" & integer'image(N_STAGES) & " stages) out == sequential ref; "
         & "II = " & time'image(ii_meas) & " (= (16/" & integer'image(GU) & ")+1 cycles @100MHz)"
      severity note;
    wait;
  end process;
end architecture;
