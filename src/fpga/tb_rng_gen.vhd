-- =====================================================================
-- tb_rng_gen.vhd -- validate tm_rng_gen against the reference run_rng stream.
-- Fills a window from seed 0x1234 and checks all 2048 bytes against
-- ref/rng_stream_1234.hex (produced by dump_rng_table). Also a cheap proof
-- that UNROLL_K chaining matches the 1-step-at-a-time reference.
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

library work;
use work.tm_pkg.all;

entity tb_rng_gen is
  generic (HEX_PATH : string := "ref/rng_stream_1234.hex";
           UNROLL_K : integer := 8);
end entity;

architecture sim of tb_rng_gen is
  signal clk     : std_logic := '0';
  signal rst     : std_logic := '1';
  signal start   : std_logic := '0';
  signal seed_in : std_logic_vector(15 downto 0) := x"1234";
  signal busy    : std_logic;
  signal done    : std_logic;
  signal win     : win_arr_t;

  function hx2(s : string) return std_logic_vector is
    variable v : std_logic_vector(7 downto 0) := (others => '0');
    variable d : integer;
  begin
    for i in 1 to 2 loop
      case s(i) is
        when '0' to '9' => d := character'pos(s(i)) - character'pos('0');
        when 'A' to 'F' => d := character'pos(s(i)) - character'pos('A') + 10;
        when 'a' to 'f' => d := character'pos(s(i)) - character'pos('a') + 10;
        when others     => d := 0;
      end case;
      v := v(3 downto 0) & std_logic_vector(to_unsigned(d, 4));
    end loop;
    return v;
  end function;
begin

  clk <= not clk after 5 ns;

  dut : entity work.tm_rng_gen
    generic map (UNROLL_K => UNROLL_K)
    port map (clk => clk, rst => rst, start => start, seed_in => seed_in,
              busy => busy, done => done, win_out => win);

  process
    file     fh   : text;
    variable ln   : line;
    variable str  : string(1 to 2);
    variable fst  : file_open_status;
    variable exp  : byte_t;
    variable errs : integer := 0;
  begin
    wait for 23 ns; rst <= '0';
    wait until rising_edge(clk);
    start <= '1'; wait until rising_edge(clk); start <= '0';

    wait until done = '1' for 60 us;  -- K=1 needs ~20us (2048 cycles)
    assert done = '1' report "FAIL: gen did not finish" severity failure;
    wait until rising_edge(clk);   -- let win_out settle

    file_open(fst, fh, HEX_PATH, read_mode);
    assert fst = open_ok
      report "cannot open " & HEX_PATH & " (build ref/dump_rng_table first)"
      severity failure;
    for n in 0 to WIN_LEN-1 loop
      readline(fh, ln); read(ln, str);
      exp := hx2(str);
      if win(n) /= exp then
        if errs < 8 then
          report "stream mismatch n=" & integer'image(n)
               & " got=" & to_hstring(win(n)) & " exp=" & to_hstring(exp)
            severity warning;
        end if;
        errs := errs + 1;
      end if;
    end loop;
    file_close(fh);

    assert errs = 0
      report "FAIL: rng_gen mismatched in " & integer'image(errs) & " bytes"
      severity failure;
    report "PASS: rng_gen window matches run_rng reference for all "
         & integer'image(WIN_LEN) & " bytes (UNROLL_K=" & integer'image(UNROLL_K) & ")"
      severity note;
    wait;
  end process;

end architecture;
