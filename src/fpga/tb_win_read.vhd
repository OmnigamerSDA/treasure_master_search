-- tb_win_read.vhd -- value-parity for the RAM/reg window read prototype.
-- Loads a known window into BOTH styles and checks rev/fwd/carry == the tm_pkg
-- reference (rng_vec / rng_vec_fwd / rng_carry_bit on the flat window) for every
-- valid (f,c) with f+c<=15. Those pkg functions are already parity-locked to
-- tm_8 (tb_map_parity), so matching them == bit-exact vs the golden kernel.
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library work;
use work.tm_pkg.all;

entity tb_win_read is
end entity;

architecture sim of tb_win_read is
  signal clk : std_logic := '0';
  signal wr_en : std_logic := '0';
  signal wr_row, f, c : std_logic_vector(3 downto 0) := (others => '0');
  signal wr_data : slv1024 := (others => '0');
  signal rev_r, fwd_r, rev_m, fwd_m : slv1024;
  signal carry_r, carry_m : std_logic;

  -- reference window (flat) + its row-packed form
  signal win : win_arr_t;
begin
  clk <= not clk after 5 ns;

  dut_reg : entity work.tm_win_read generic map (USE_RAM => false)
    port map (clk=>clk, wr_en=>wr_en, wr_row=>wr_row, wr_data=>wr_data,
              f=>f, c=>c, rev_out=>rev_r, fwd_out=>fwd_r, carry_out=>carry_r);
  dut_ram : entity work.tm_win_read generic map (USE_RAM => true)
    port map (clk=>clk, wr_en=>wr_en, wr_row=>wr_row, wr_data=>wr_data,
              f=>f, c=>c, rev_out=>rev_m, fwd_out=>fwd_m, carry_out=>carry_m);

  stim : process
    variable bad : integer := 0;
    variable rv, fv : state_t;
    variable cy : std_logic;
    variable rowd : slv1024;
  begin
    -- build a pseudo-random window
    for k in 0 to WIN_LEN-1 loop
      win(k) <= std_logic_vector(to_unsigned((k*151 + 29) mod 256, 8));
    end loop;
    wait until rising_edge(clk);

    -- LOAD 16 rows into both DUTs
    for r in 0 to 15 loop
      for j in 0 to NUM_BYTES-1 loop
        rowd(8*j+7 downto 8*j) := win(128*r + j);
      end loop;
      wr_data <= rowd; wr_row <= std_logic_vector(to_unsigned(r,4)); wr_en <= '1';
      wait until rising_edge(clk);
    end loop;
    wr_en <= '0';
    wait until rising_edge(clk);

    -- check every valid (f,c): f+c <= 15
    for ff in 0 to 15 loop
      for cc in 0 to 15-ff loop
        f <= std_logic_vector(to_unsigned(ff,4));
        c <= std_logic_vector(to_unsigned(cc,4));
        wait for 2 ns;                       -- settle async read
        rv := rng_vec(win, ff, cc);
        fv := rng_vec_fwd(win, ff, cc);
        cy := rng_carry_bit(win, ff, cc);
        for i in 0 to NUM_BYTES-1 loop
          if rev_r(8*i+7 downto 8*i) /= rv(i) then bad := bad+1; end if;
          if rev_m(8*i+7 downto 8*i) /= rv(i) then bad := bad+1; end if;
          if fwd_r(8*i+7 downto 8*i) /= fv(i) then bad := bad+1; end if;
          if fwd_m(8*i+7 downto 8*i) /= fv(i) then bad := bad+1; end if;
        end loop;
        if carry_r /= cy then bad := bad+1; end if;
        if carry_m /= cy then bad := bad+1; end if;
      end loop;
    end loop;

    assert bad = 0
      report "FAIL: win_read mismatch, bad=" & integer'image(bad) severity failure;
    report "PASS: tm_win_read reg AND ram == tm_pkg reference for all valid (f,c)"
      severity note;
    wait;
  end process;
end architecture;
