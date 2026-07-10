-- =====================================================================
-- tm_screen_batch.vhd -- map-major batched screen (throughput demonstrator).
-- ---------------------------------------------------------------------
-- Processes a batch of BATCH candidates map-major so the per-key, per-map RNG
-- window is generated ONCE and reused across the whole batch, and the
-- generation overlaps the batch compute (double-buffered: gen map m+1 while
-- the engine runs the batch on map m). This is what turns the per-candidate
-- ~3648-7185 cycles of the one-shot demonstrator into the COMPUTE bound:
--
--   cyc/candidate  ~=  27 * (16/GROUP_UNROLL + gap)        [gen hidden]
--
-- with the generator non-bottleneck as long as 2048/UNROLL_K <= BATCH*16/G
-- (it overlaps the c-loop). The 1 B/clk serial chain never gates throughput.
--
-- ARCHITECTURE NOTE / OPEN FOR REVIEW: this is ONE top-level proposal. It
-- still sequences candidates through a single map_engine with a ~1-cycle gap
-- between them (a start/done turnaround), which is the main remaining
-- sequential overhead. The higher-quality form interleaves candidates through
-- the inner-step datapath at II=1 (no gap) and/or widens combinational depth
-- per clock (GROUP_UNROLL) so fewer, deeper clocks do the work. Treat the
-- map_engine + rng_gen as the settled units and this sequencer as the part to
-- second-opinion.
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.tm_pkg.all;

entity tm_screen_batch is
  generic (
    BATCH        : integer := 64;
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
    data_base  : in  std_logic_vector(31 downto 0);   -- candidate c = data_base + c
    seeds      : in  seed_arr_t;
    nibbles    : in  nib_arr_t;
    exp_vec    : in  state_t;
    world_xor  : in  slv1024;
    world_mask : in  slv1024;
    busy       : out std_logic;
    done       : out std_logic;
    flags      : out byte_vec_t(0 to BATCH-1)
  );
end entity;

architecture rtl of tm_screen_batch is

  type fsm_t is (S_IDLE, S_LOAD, S_PRIME, S_MAPSTART, S_RUN, S_MAPEND, S_CHECK, S_DONE);
  signal fsm : fsm_t := S_IDLE;

  signal cand   : state_vec_t(0 to BATCH-1);
  signal c_idx  : integer range 0 to BATCH := 0;
  signal m_idx  : integer range 0 to NUM_MAPS := 0;

  signal win_act  : win_arr_t;            -- window the engine is using
  signal sel_b    : std_logic := '0';     -- which gen bank is "next"

  -- two generators (double buffer): genA fills bank A, genB fills bank B.
  signal genA_start, genA_busy, genA_done : std_logic := '0';
  signal genB_start, genB_busy, genB_done : std_logic := '0';
  signal genA_rdy, genB_rdy : std_logic := '0';   -- sticky: window filled
  signal genA_seed, genB_seed : std_logic_vector(15 downto 0);
  signal winA, winB : win_arr_t;

  -- engine
  signal eng_start : std_logic := '0';
  signal eng_busy  : std_logic;
  signal eng_done  : std_logic;
  signal eng_in    : state_t;
  signal eng_out   : state_t;
  signal eng_nib   : std_logic_vector(15 downto 0);

  function init_state(k : std_logic_vector(31 downto 0);
                      d : std_logic_vector(31 downto 0)) return state_t is
    variable s : state_t;
    variable w : std_logic_vector(31 downto 0);
  begin
    for lane in 0 to 31 loop
      if (lane mod 2) = 0 then w := k; else w := d; end if;
      s(lane*4+0) := w(31 downto 24); s(lane*4+1) := w(23 downto 16);
      s(lane*4+2) := w(15 downto  8); s(lane*4+3) := w( 7 downto  0);
    end loop;
    return s;
  end function;

  function csum_flag(st : state_t; wx, wm : slv1024;
                     hi, lo : integer) return byte_t is
    variable acc : unsigned(15 downto 0) := (others => '0');
    variable d   : byte_t;
    variable dh, dl : byte_t;
  begin
    for i in 0 to NUM_BYTES-1 loop
      d   := st(i) xor wx(8*i+7 downto 8*i);
      acc := acc + resize(unsigned(d and wm(8*i+7 downto 8*i)), 16);
    end loop;
    dh := st(hi) xor wx(8*hi+7 downto 8*hi);
    dl := st(lo) xor wx(8*lo+7 downto 8*lo);
    if std_logic_vector(acc) = (dh & dl) then return x"08"; else return x"00"; end if;
  end function;

begin

  busy    <= '0' when fsm = S_IDLE else '1';
  eng_in  <= cand(c_idx) when c_idx < BATCH else cand(0);
  eng_nib <= nibbles(m_idx) when m_idx < NUM_MAPS else (others => '0');

  -- gen seeds: each gen fills the map it is assigned. The "next" bank is sel_b.
  genA_seed <= seeds(m_idx) when (sel_b = '0' and fsm = S_PRIME) else
               seeds(m_idx+1) when (m_idx < NUM_MAPS-1) else (others => '0');
  genB_seed <= seeds(m_idx+1) when (m_idx < NUM_MAPS-1) else (others => '0');

  win_act <= winA when sel_b = '1' else winB;   -- active = the NOT-next bank

  genA : entity work.tm_rng_gen generic map (UNROLL_K => UNROLL_K)
    port map (clk=>clk, rst=>rst, start=>genA_start, seed_in=>genA_seed,
              busy=>genA_busy, done=>genA_done, win_out=>winA);
  genB : entity work.tm_rng_gen generic map (UNROLL_K => UNROLL_K)
    port map (clk=>clk, rst=>rst, start=>genB_start, seed_in=>genB_seed,
              busy=>genB_busy, done=>genB_done, win_out=>winB);

  eng : entity work.tm_map_engine generic map (GROUP_UNROLL => GROUP_UNROLL)
    port map (clk=>clk, rst=>rst, start=>eng_start, state_in=>eng_in,
              win=>win_act, nibble=>eng_nib,
              busy=>eng_busy, done=>eng_done, state_out=>eng_out);

  process(clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        fsm <= S_IDLE; done <= '0';
        genA_start <= '0'; genB_start <= '0'; eng_start <= '0';
      else
        done <= '0';
        genA_start <= '0'; genB_start <= '0';

        -- latch generator completion (done is a 1-cycle pulse; the c-loop is
        -- much longer than a fill, so S_MAPEND must poll a sticky flag).
        if genA_done = '1' then genA_rdy <= '1'; end if;
        if genB_done = '1' then genB_rdy <= '1'; end if;

        case fsm is

          when S_IDLE =>
            if start = '1' then
              c_idx <= 0; m_idx <= 0; sel_b <= '0';
              fsm <= S_LOAD;
            end if;

          -- load + expand the whole batch (one candidate/clock).
          when S_LOAD =>
            cand(c_idx) <= add_vec(
              init_state(key, std_logic_vector(unsigned(data_base) + c_idx)),
              exp_vec);
            if c_idx = BATCH-1 then
              c_idx <= 0;
              genA_start <= '1';        -- prime bank A with map 0
              genA_rdy   <= '0';
              fsm <= S_PRIME;
            else
              c_idx <= c_idx + 1;
            end if;

          -- wait for map-0 window, place it in bank A, mark B as "next".
          when S_PRIME =>
            if genA_rdy = '1' then
              sel_b <= '1';             -- active = A (sel_b=1 -> win_act=winA)
              fsm   <= S_MAPSTART;
            end if;

          -- kick the "next" bank generator for map m+1 (overlaps the c-loop),
          -- then start streaming the batch through the active window.
          when S_MAPSTART =>
            if m_idx < NUM_MAPS-1 then
              if sel_b = '1' then genB_start <= '1'; genB_rdy <= '0';  -- next bank = B
              else                genA_start <= '1'; genA_rdy <= '0';  -- next bank = A
              end if;
            end if;
            c_idx     <= 0;
            eng_start <= '1';           -- start candidate 0
            fsm       <= S_RUN;

          -- tight candidate loop: when the engine finishes one candidate,
          -- latch it and immediately start the next (gap ~= 1 clk).
          when S_RUN =>
            if eng_busy = '1' then eng_start <= '0'; end if;
            if eng_done = '1' then
              cand(c_idx) <= eng_out;
              if c_idx = BATCH-1 then
                fsm <= S_MAPEND;
              else
                c_idx     <= c_idx + 1;
                eng_start <= '1';        -- next candidate
              end if;
            end if;

          -- batch done with map m; wait for the overlapped next window, swap.
          when S_MAPEND =>
            if m_idx = NUM_MAPS-1 then
              c_idx <= 0;
              fsm   <= S_CHECK;
            elsif (sel_b = '1' and genB_rdy = '1') or
                  (sel_b = '0' and genA_rdy = '1') then
              sel_b <= not sel_b;        -- swap active/next banks
              m_idx <= m_idx + 1;
              fsm   <= S_MAPSTART;
            end if;

          when S_CHECK =>
            flags(c_idx) <= csum_flag(cand(c_idx), world_xor, world_mask,
                                      CSUM_HI_POS, CSUM_LO_POS);
            if c_idx = BATCH-1 then fsm <= S_DONE;
            else                    c_idx <= c_idx + 1; end if;

          when S_DONE =>
            done <= '1';
            fsm  <= S_IDLE;

        end case;
      end if;
    end if;
  end process;

end architecture;
