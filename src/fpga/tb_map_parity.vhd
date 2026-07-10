-- =====================================================================
-- tb_map_parity.vhd -- VALUE-PARITY GATE.
-- Reads ref/map_vectors.txt (run_one_map outputs from the canonical scalar
-- kernel tm_8) and checks, for each vector, that:
--     tm_map_engine( state_in, rng_gen(seed), nibble ) == state_out
-- bit for bit. This closes the alg2/alg5 derivation, the window-offset
-- indexing, and the byte order against the repo ground truth.
--
--   (build ref/dump_map first -> map_vectors.txt)
--   ghdl -a --std=08 tm_pkg.vhd tm_rng_gen.vhd tm_map_engine.vhd tb_map_parity.vhd
--   ghdl -e --std=08 tb_map_parity
--   ghdl -r --std=08 tb_map_parity --stop-time=2ms
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

library work;
use work.tm_pkg.all;

entity tb_map_parity is
  generic (PATH : string := "ref/map_vectors.txt";
           UNROLL_K : integer := 16; GROUP_UNROLL : integer := 4);
end entity;

architecture sim of tb_map_parity is
  signal clk : std_logic := '0';
  signal rst : std_logic := '1';

  signal gen_start, gen_busy, gen_done : std_logic := '0';
  signal gen_seed : std_logic_vector(15 downto 0);
  signal win_sig  : win_arr_t;

  signal eng_start, eng_busy, eng_done : std_logic := '0';
  signal eng_sin  : state_t;
  signal eng_nib  : std_logic_vector(15 downto 0);
  signal eng_out  : state_t;

  function hexdig(c : character) return integer is
  begin
    case c is
      when '0' to '9' => return character'pos(c) - character'pos('0');
      when 'A' to 'F' => return character'pos(c) - character'pos('A') + 10;
      when 'a' to 'f' => return character'pos(c) - character'pos('a') + 10;
      when others     => return 0;
    end case;
  end function;
begin
  clk <= not clk after 5 ns;

  gen : entity work.tm_rng_gen generic map (UNROLL_K=>UNROLL_K)
    port map (clk=>clk, rst=>rst, start=>gen_start, seed_in=>gen_seed,
              busy=>gen_busy, done=>gen_done, win_out=>win_sig);

  eng : entity work.tm_map_engine generic map (GROUP_UNROLL=>GROUP_UNROLL)
    port map (clk=>clk, rst=>rst, start=>eng_start, state_in=>eng_sin, win=>win_sig,
              nibble=>eng_nib, busy=>eng_busy, done=>eng_done, state_out=>eng_out);

  stim : process
    file     fh    : text;
    variable ln    : line;
    variable fst   : file_open_status;
    variable seed16, nib16 : std_logic_vector(15 downto 0);
    variable s256  : string(1 to 256);
    variable sin, sout : state_t;
    variable nvec, nbad : integer := 0;
    variable bad_v : integer := 0;
  begin
    wait for 23 ns; rst <= '0';
    wait until rising_edge(clk);

    file_open(fst, fh, PATH, read_mode);
    assert fst = open_ok
      report "cannot open " & PATH & " (build ref/dump_map first)" severity failure;

    while not endfile(fh) loop
      -- line 1: SEED NIBBLE (hex), whitespace-separated
      readline(fh, ln); hread(ln, seed16); hread(ln, nib16);
      -- line 2: state_in (256 hex, byte 0 first)
      readline(fh, ln); read(ln, s256);
      for k in 0 to NUM_BYTES-1 loop
        sin(k) := std_logic_vector(to_unsigned(hexdig(s256(2*k+1))*16 + hexdig(s256(2*k+2)), 8));
      end loop;
      -- line 3: expected state_out
      readline(fh, ln); read(ln, s256);
      for k in 0 to NUM_BYTES-1 loop
        sout(k) := std_logic_vector(to_unsigned(hexdig(s256(2*k+1))*16 + hexdig(s256(2*k+2)), 8));
      end loop;

      -- regenerate the per-map window from seed
      gen_seed <= seed16;
      gen_start <= '1'; wait until rising_edge(clk) and gen_busy = '1'; gen_start <= '0';
      wait until gen_done = '1'; wait until rising_edge(clk);

      -- run one map
      eng_sin <= sin; eng_nib <= nib16;
      eng_start <= '1'; wait until rising_edge(clk) and eng_busy = '1'; eng_start <= '0';
      wait until eng_done = '1'; wait until rising_edge(clk);

      bad_v := 0;
      for k in 0 to NUM_BYTES-1 loop
        if eng_out(k) /= sout(k) then bad_v := bad_v + 1; end if;
      end loop;
      if bad_v /= 0 then
        nbad := nbad + 1;
        if nbad <= 4 then
          report "MISMATCH vec=" & integer'image(nvec) & " seed=" & to_hstring(seed16)
               & " nib=" & to_hstring(nib16) & " bad_bytes=" & integer'image(bad_v)
            severity warning;
        end if;
      end if;
      nvec := nvec + 1;
    end loop;
    file_close(fh);

    assert nbad = 0
      report "FAIL: " & integer'image(nbad) & "/" & integer'image(nvec)
           & " map vectors mismatched" severity failure;
    report "PASS: tm_map_engine == tm_8 run_one_map for all "
         & integer'image(nvec) & " vectors (alg2/5 + window + byte order LOCKED)"
      severity note;
    wait;
  end process;
end architecture;
