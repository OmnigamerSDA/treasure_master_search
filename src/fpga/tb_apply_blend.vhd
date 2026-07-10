-- =====================================================================
-- tb_apply_blend.vhd -- prove the op-pair blend-tree datapath is bit-identical
-- to the case-based apply_map, for all 8 alg_ids over many (state, rng, carry)
-- vectors. Closes the refactor (A): apply_map_blend == apply_map.
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.tm_pkg.all;

entity tb_apply_blend is end entity;

architecture sim of tb_apply_blend is
begin
  check : process
    variable s, r, o_case, o_bl : state_t;
    variable carry : std_logic;
    variable seedA, seedB : integer;
    variable errs : integer := 0;
    variable aid  : std_logic_vector(2 downto 0);
  begin
    -- deterministic pseudo-random vectors, 16-bit LCG (products stay < 2^31)
    for trial in 0 to 63 loop
      seedA := (trial*7919 + 12345) mod 65536;
      seedB := (trial*25171 + 7) mod 65536;
      for i in 0 to NUM_BYTES-1 loop
        seedA := (seedA*25173 + 13849) mod 65536;
        seedB := (seedB*4129  + 1)     mod 65536;
        s(i)  := std_logic_vector(to_unsigned(seedA mod 256, 8));
        r(i)  := std_logic_vector(to_unsigned((seedB / 256) mod 256, 8));
      end loop;
      carry := '0';
      if (trial mod 2) = 1 then carry := '1'; end if;

      for a in 0 to 7 loop
        aid    := std_logic_vector(to_unsigned(a, 3));
        -- pass r as both reversed and forward operands (equivalence check only)
        o_case := apply_map(s, r, r, aid, carry);
        o_bl   := apply_map_blend(s, r, r, aid, carry);
        for i in 0 to NUM_BYTES-1 loop
          if o_case(i) /= o_bl(i) then errs := errs + 1; end if;
        end loop;
      end loop;
    end loop;

    assert errs = 0
      report "FAIL: apply_map_blend differs from apply_map in "
           & integer'image(errs) & " bytes" severity failure;
    report "PASS: apply_map_blend == apply_map for all 8 algs x 64 vectors"
      severity note;
    wait;
  end process;
end architecture;
