-- =====================================================================
-- tm_screen_w.vhd -- screening top built on the W-lane lockstep engine.
-- ---------------------------------------------------------------------
-- Processes W candidates map-major with ONE engine run per map (all W lanes
-- in parallel), double-buffered window generation. This removes the per-
-- candidate start/done turnaround that bottlenecked tm_screen_batch: per map
-- is 16/GROUP_UNROLL engine cycles + a window swap, shared across all W
-- candidates, so per-candidate ~= 27*(16/G + swap)/W -> the compute bound.
--
-- Knobs: LANES (W, parallelism / area), GROUP_UNROLL (G, depth/clock),
-- UNROLL_K (generation rate, kept overlapped). key_schedule is an input.
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.tm_pkg.all;

entity tm_screen_w is
  generic (
    LANES        : integer := 8;
    UNROLL_K     : integer := 16;
    GROUP_UNROLL : integer := 4;
    CSUM_HI_POS  : integer := 127 - (16#72# - 1);
    CSUM_LO_POS  : integer := 127 - (16#72# - 2)
  );
  port (
    clk        : in  std_logic;
    rst        : in  std_logic;
    start      : in  std_logic;
    key        : in  std_logic_vector(31 downto 0);
    data_base  : in  std_logic_vector(31 downto 0);
    seeds      : in  seed_arr_t;
    nibbles    : in  nib_arr_t;
    exp_vec    : in  state_t;
    world_xor  : in  slv1024;
    world_mask : in  slv1024;
    busy       : out std_logic;
    done       : out std_logic;
    flags      : out byte_vec_t(0 to LANES-1)
  );
end entity;

architecture rtl of tm_screen_w is
  type fsm_t is (S_IDLE, S_LOAD, S_PRIME, S_MAPSTART, S_RUN, S_MAPEND, S_CHECK, S_DONE);
  signal fsm : fsm_t := S_IDLE;

  signal cand  : state_vec_t(0 to LANES-1);
  signal c_idx : integer range 0 to LANES := 0;
  signal m_idx : integer range 0 to NUM_MAPS := 0;
  signal sel_b : std_logic := '0';

  signal genA_start, genA_busy, genA_done, genA_rdy : std_logic := '0';
  signal genB_start, genB_busy, genB_done, genB_rdy : std_logic := '0';
  signal genA_seed, genB_seed : std_logic_vector(15 downto 0);
  signal winA, winB : win_arr_t;
  signal win_act : win_arr_t;

  signal eng_start, eng_busy, eng_done : std_logic := '0';
  signal eng_out : state_vec_t(0 to LANES-1);
  signal eng_nib : std_logic_vector(15 downto 0);

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

  function csum_flag(st : state_t; wx, wm : slv1024; hi, lo : integer) return byte_t is
    variable acc : unsigned(15 downto 0) := (others=>'0'); variable dh, dl : byte_t;
  begin
    for i in 0 to NUM_BYTES-1 loop
      acc := acc + resize(unsigned((st(i) xor wx(8*i+7 downto 8*i)) and wm(8*i+7 downto 8*i)), 16);
    end loop;
    dh := st(hi) xor wx(8*hi+7 downto 8*hi);
    dl := st(lo) xor wx(8*lo+7 downto 8*lo);
    if std_logic_vector(acc) = (dh & dl) then return x"08"; else return x"00"; end if;
  end function;
begin
  busy     <= '0' when fsm = S_IDLE else '1';
  genA_seed <= seeds(m_idx) when fsm = S_PRIME else
               seeds(m_idx+1) when m_idx < NUM_MAPS-1 else (others=>'0');
  genB_seed <= seeds(m_idx+1) when m_idx < NUM_MAPS-1 else (others=>'0');
  win_act  <= winA when sel_b = '1' else winB;
  eng_nib  <= nibbles(m_idx) when m_idx < NUM_MAPS else (others => '0');

  genA : entity work.tm_rng_gen generic map (UNROLL_K=>UNROLL_K)
    port map (clk=>clk, rst=>rst, start=>genA_start, seed_in=>genA_seed,
              busy=>genA_busy, done=>genA_done, win_out=>winA);
  genB : entity work.tm_rng_gen generic map (UNROLL_K=>UNROLL_K)
    port map (clk=>clk, rst=>rst, start=>genB_start, seed_in=>genB_seed,
              busy=>genB_busy, done=>genB_done, win_out=>winB);

  eng : entity work.tm_map_engine_w
    generic map (LANES=>LANES, GROUP_UNROLL=>GROUP_UNROLL)
    port map (clk=>clk, rst=>rst, start=>eng_start, state_in=>cand, win=>win_act,
              nibble=>eng_nib,
              busy=>eng_busy, done=>eng_done, state_out=>eng_out);

  process(clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        fsm <= S_IDLE; done <= '0';
        genA_start<='0'; genB_start<='0'; eng_start<='0';
      else
        done <= '0'; genA_start<='0'; genB_start<='0';
        if genA_done='1' then genA_rdy<='1'; end if;
        if genB_done='1' then genB_rdy<='1'; end if;

        case fsm is
          when S_IDLE =>
            if start='1' then c_idx<=0; m_idx<=0; sel_b<='0'; fsm<=S_LOAD; end if;

          when S_LOAD =>
            cand(c_idx) <= add_vec(
              init_state(key, std_logic_vector(unsigned(data_base) + c_idx)), exp_vec);
            if c_idx = LANES-1 then
              c_idx<=0; genA_start<='1'; genA_rdy<='0'; fsm<=S_PRIME;
            else c_idx<=c_idx+1; end if;

          when S_PRIME =>
            if genA_rdy='1' then sel_b<='1'; fsm<=S_MAPSTART; end if;

          when S_MAPSTART =>
            if m_idx < NUM_MAPS-1 then
              if sel_b='1' then genB_start<='1'; genB_rdy<='0';
              else genA_start<='1'; genA_rdy<='0'; end if;
            end if;
            eng_start <= '1';            -- one run for ALL W lanes
            fsm <= S_RUN;

          when S_RUN =>
            if eng_busy='1' then eng_start<='0'; end if;
            if eng_done='1' then
              cand <= eng_out;
              fsm  <= S_MAPEND;
            end if;

          when S_MAPEND =>
            if m_idx = NUM_MAPS-1 then
              c_idx<=0; fsm<=S_CHECK;
            elsif (sel_b='1' and genB_rdy='1') or (sel_b='0' and genA_rdy='1') then
              sel_b<= not sel_b; m_idx<=m_idx+1; fsm<=S_MAPSTART;
            end if;

          when S_CHECK =>
            flags(c_idx) <= csum_flag(cand(c_idx), world_xor, world_mask, CSUM_HI_POS, CSUM_LO_POS);
            if c_idx = LANES-1 then fsm<=S_DONE; else c_idx<=c_idx+1; end if;

          when S_DONE =>
            done <= '1'; fsm <= S_IDLE;
        end case;
      end if;
    end if;
  end process;
end architecture;
