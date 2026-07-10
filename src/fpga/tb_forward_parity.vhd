-- =====================================================================
-- tb_forward_parity.vhd -- END-TO-END forward parity gate.
-- Reproduces expand + all 27 maps (rng_gen windows generated once per key,
-- then map_engine x27 per candidate) and checks the full 128-byte post-maps
-- state against tm_8 (ref/forward_vectors.txt). This locks the COMPLETE
-- forward chain to ground truth, not just a single map.
--
--   (build ref/dump_forward -> forward_vectors.txt)
--   ghdl -a --std=08 tm_pkg.vhd tm_rng_gen.vhd tm_map_engine.vhd tb_forward_parity.vhd
--   ghdl -e --std=08 tb_forward_parity
--   ghdl -r --std=08 tb_forward_parity --stop-time=5ms
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

library work;
use work.tm_pkg.all;

entity tb_forward_parity is
  generic (PATH : string := "ref/forward_vectors.txt";
           UNROLL_K : integer := 16; GROUP_UNROLL : integer := 4);
end entity;

architecture sim of tb_forward_parity is
  signal clk : std_logic := '0';
  signal rst : std_logic := '1';

  signal gen_start, gen_busy, gen_done : std_logic := '0';
  signal gen_seed : std_logic_vector(15 downto 0);
  signal gen_win  : win_arr_t;

  signal eng_start, eng_busy, eng_done : std_logic := '0';
  signal eng_sin, eng_out : state_t;
  signal cur_win : win_arr_t;
  signal cur_nib : std_logic_vector(15 downto 0);

  signal win_store : win_vec_t(0 to NUM_MAPS-1);
  signal seeds     : seed_arr_t;
  signal nibs      : nib_arr_t;
  signal exp_vec   : state_t;
  signal key_r     : std_logic_vector(31 downto 0);
  signal m_idx     : integer range 0 to NUM_MAPS-1 := 0;

  function hexdig(c : character) return integer is
  begin
    case c is
      when '0' to '9' => return character'pos(c) - character'pos('0');
      when 'A' to 'F' => return character'pos(c) - character'pos('A') + 10;
      when 'a' to 'f' => return character'pos(c) - character'pos('a') + 10;
      when others     => return 0;
    end case;
  end function;

  function init_state(k, d : std_logic_vector(31 downto 0)) return state_t is
    variable s : state_t; variable w : std_logic_vector(31 downto 0);
  begin
    for lane in 0 to 31 loop
      if (lane mod 2) = 0 then w := k; else w := d; end if;
      s(lane*4+0):=w(31 downto 24); s(lane*4+1):=w(23 downto 16);
      s(lane*4+2):=w(15 downto 8);  s(lane*4+3):=w(7 downto 0);
    end loop;
    return s;
  end function;
begin
  clk <= not clk after 5 ns;

  gen : entity work.tm_rng_gen generic map (UNROLL_K=>UNROLL_K)
    port map (clk=>clk, rst=>rst, start=>gen_start, seed_in=>gen_seed,
              busy=>gen_busy, done=>gen_done, win_out=>gen_win);

  eng : entity work.tm_map_engine generic map (GROUP_UNROLL=>GROUP_UNROLL)
    port map (clk=>clk, rst=>rst, start=>eng_start, state_in=>eng_sin, win=>cur_win,
              nibble=>cur_nib, busy=>eng_busy, done=>eng_done, state_out=>eng_out);

  cur_win <= win_store(m_idx);
  cur_nib <= nibs(m_idx);

  stim : process
    file     fh   : text;
    variable ln   : line;
    variable fst  : file_open_status;
    variable nmaps, ndata, bad_v, nbad : integer := 0;
    variable s16  : std_logic_vector(15 downto 0);
    variable d32  : std_logic_vector(31 downto 0);
    variable s256 : string(1 to 256);
    variable st, expst : state_t;
  begin
    wait for 23 ns; rst <= '0';
    wait until rising_edge(clk);

    file_open(fst, fh, PATH, read_mode);
    assert fst = open_ok report "cannot open " & PATH severity failure;

    -- header: KEY NMAPS
    readline(fh, ln); hread(ln, d32); read(ln, nmaps);
    key_r <= d32;
    assert nmaps = NUM_MAPS report "NMAPS mismatch" severity failure;

    -- schedule
    for m in 0 to NUM_MAPS-1 loop
      readline(fh, ln); hread(ln, s16); seeds(m) <= s16;
      hread(ln, s16); nibs(m) <= s16;
    end loop;

    -- expansion vector
    readline(fh, ln); read(ln, s256);
    for k in 0 to NUM_BYTES-1 loop
      exp_vec(k) <= std_logic_vector(to_unsigned(hexdig(s256(2*k+1))*16 + hexdig(s256(2*k+2)), 8));
    end loop;
    wait until rising_edge(clk);   -- let seeds/exp settle

    -- generate the 27 windows ONCE (fixed key)
    for m in 0 to NUM_MAPS-1 loop
      gen_seed <= seeds(m);
      gen_start <= '1'; wait until rising_edge(clk) and gen_busy = '1'; gen_start <= '0';
      wait until gen_done = '1'; wait until rising_edge(clk);
      win_store(m) <= gen_win; wait until rising_edge(clk);
    end loop;

    -- data vectors
    readline(fh, ln); read(ln, ndata);
    for v in 0 to ndata-1 loop
      readline(fh, ln); hread(ln, d32);
      readline(fh, ln); read(ln, s256);
      for k in 0 to NUM_BYTES-1 loop
        expst(k) := std_logic_vector(to_unsigned(hexdig(s256(2*k+1))*16 + hexdig(s256(2*k+2)), 8));
      end loop;

      -- expand
      st := add_vec(init_state(key_r, d32), exp_vec);

      -- 27 maps, reusing the stored windows
      for m in 0 to NUM_MAPS-1 loop
        m_idx   <= m; wait until rising_edge(clk);   -- cur_win/cur_nib settle
        eng_sin <= st;
        eng_start <= '1'; wait until rising_edge(clk) and eng_busy = '1'; eng_start <= '0';
        wait until eng_done = '1'; wait until rising_edge(clk);
        st := eng_out;
      end loop;

      bad_v := 0;
      for k in 0 to NUM_BYTES-1 loop
        if st(k) /= expst(k) then bad_v := bad_v + 1; end if;
      end loop;
      if bad_v /= 0 then
        nbad := nbad + 1;
        if nbad <= 4 then
          report "MISMATCH data=" & to_hstring(d32) & " bad_bytes=" & integer'image(bad_v)
            severity warning;
        end if;
      end if;
    end loop;
    file_close(fh);

    assert nbad = 0
      report "FAIL: " & integer'image(nbad) & "/" & integer'image(ndata)
           & " forward vectors mismatched" severity failure;
    report "PASS: expand+27maps == tm_8 for all " & integer'image(ndata)
         & " data vectors (FULL FORWARD CHAIN LOCKED)" severity note;
    wait;
  end process;
end architecture;
