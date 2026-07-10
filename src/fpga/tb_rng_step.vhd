-- =====================================================================
-- tb_rng_step.vhd -- EXHAUSTIVE validation of the table-free rng_step.
-- ---------------------------------------------------------------------
-- Reads ref/rng_table.hex (the ACTUAL repo rng_table, dumped from
-- src/common/rng.cpp) and checks, for all 65536 seeds, that:
--    rng_step(seed).nseed == rng_table[seed]
--    rng_step(seed).obyte == ((rng_table[seed]>>8) xor rng_table[seed]) & 0xFF
-- This proves the 128 KB ROM can be replaced by combinational logic.
--
--   (build ref first)
--   cd ref && g++ -O2 -I../../common dump_rng_table.cpp ../../common/rng.cpp \
--             -o dump_rng_table && ./dump_rng_table && cd ..
--   ghdl -a --std=08 tm_pkg.vhd tb_rng_step.vhd
--   ghdl -e --std=08 tb_rng_step
--   ghdl -r --std=08 tb_rng_step
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

library work;
use work.tm_pkg.all;

entity tb_rng_step is
  generic (HEX_PATH : string := "ref/rng_table.hex");
end entity;

architecture sim of tb_rng_step is

  -- hex string (4 chars) -> 16-bit
  function hx(s : string) return std_logic_vector is
    variable v : std_logic_vector(15 downto 0) := (others => '0');
    variable d : integer;
  begin
    for i in 1 to 4 loop
      case s(i) is
        when '0' to '9' => d := character'pos(s(i)) - character'pos('0');
        when 'A' to 'F' => d := character'pos(s(i)) - character'pos('A') + 10;
        when 'a' to 'f' => d := character'pos(s(i)) - character'pos('a') + 10;
        when others     => d := 0;
      end case;
      v := v(11 downto 0) & std_logic_vector(to_unsigned(d, 4));
    end loop;
    return v;
  end function;

begin

  check : process
    file     fh    : text;
    variable ln    : line;
    variable str   : string(1 to 4);
    variable fst   : file_open_status;
    variable expect: std_logic_vector(15 downto 0);
    variable nseed : std_logic_vector(15 downto 0);
    variable obyte : std_logic_vector(7 downto 0);
    variable exp_b : std_logic_vector(7 downto 0);
    variable errs  : integer := 0;
  begin
    file_open(fst, fh, HEX_PATH, read_mode);
    assert fst = open_ok
      report "cannot open " & HEX_PATH & " (build ref/dump_rng_table first)"
      severity failure;

    for s in 0 to 65535 loop
      readline(fh, ln);
      read(ln, str);
      expect := hx(str);

      rng_step(std_logic_vector(to_unsigned(s, 16)), nseed, obyte);
      exp_b := expect(15 downto 8) xor expect(7 downto 0);

      if nseed /= expect then
        if errs < 8 then
          report "nseed mismatch seed=" & integer'image(s)
               & " got=" & to_hstring(nseed) & " exp=" & to_hstring(expect)
            severity warning;
        end if;
        errs := errs + 1;
      end if;
      if obyte /= exp_b then
        if errs < 8 then
          report "obyte mismatch seed=" & integer'image(s)
               & " got=" & to_hstring(obyte) & " exp=" & to_hstring(exp_b)
            severity warning;
        end if;
        errs := errs + 1;
      end if;
    end loop;
    file_close(fh);

    assert errs = 0
      report "FAIL: rng_step mismatched in " & integer'image(errs) & " checks"
      severity failure;
    report "PASS: rng_step matches rng_table for all 65536 seeds (nseed+obyte)"
      severity note;
    wait;
  end process;

end architecture;
