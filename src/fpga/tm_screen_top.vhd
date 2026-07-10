-- =====================================================================
-- tm_screen_top.vhd -- limited-memory screening pipeline (one candidate).
-- ---------------------------------------------------------------------
-- Composes the verified pieces into an end-to-end forward screen with the
-- NES-spirit memory model the discussion converged on:
--
--   expand -> { for each map: regenerate that map's 2 KB window with
--               tm_rng_gen, then run it through tm_map_engine } -> checksum
--
-- Memory resident at any time: ONE ~2 KB RNG window (inside tm_rng_gen) +
-- the 128-byte candidate state. NO 59 KB offset-stream store, NO 128 KB
-- table -- the window is rate-matching buffer, regenerated from the map's
-- start seed via the combinational rng_step.
--
-- DEMONSTRATOR SCOPE: this processes ONE candidate and regenerates each
-- window for that candidate, so it is RNG-bound (the honest limited-memory
-- baseline). The throughput-optimal form keeps the SAME window and reuses it
-- across a BATCH of B candidates (map-major), amortizing regeneration over B
-- and adding only B x 128 B of batch state -- see README. The datapath unit
-- (tm_map_engine) and generator (tm_rng_gen) are unchanged in that form; only
-- the sequencer wraps a batch loop. Double-buffering (gen map m+1 while the
-- engine runs map m) hides the regen latency and is a second rng_gen + a
-- ping-pong select.
--
-- key_schedule (seeds + nibbles, 108 bytes) is the separate pure-key block;
-- here it is an input (host- or on-chip-provided).
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.tm_pkg.all;

entity tm_screen_top is
  generic (
    UNROLL_K     : integer := 8;   -- rng_gen steps/clock
    GROUP_UNROLL : integer := 4;   -- map_engine inner steps/clock (divides 16)
    CODE_LENGTH  : integer := 16#72#;
    CSUM_HI_POS  : integer := 127 - (16#72# - 1);   -- byte 14
    CSUM_LO_POS  : integer := 127 - (16#72# - 2)    -- byte 15
  );
  port (
    clk        : in  std_logic;
    rst        : in  std_logic;
    start      : in  std_logic;
    key        : in  std_logic_vector(31 downto 0);
    data       : in  std_logic_vector(31 downto 0);
    -- per-key schedule (from key_schedule block) + expansion vector
    seeds      : in  seed_arr_t;
    nibbles    : in  nib_arr_t;
    exp_vec    : in  state_t;                 -- (key>>16) expansion vector
    world_xor  : in  slv1024;
    world_mask : in  slv1024;
    busy       : out std_logic;
    done       : out std_logic;
    screen_flag: out std_logic_vector(7 downto 0)
  );
end entity;

architecture rtl of tm_screen_top is

  type fsm_t is (S_IDLE, S_EXPAND, S_GEN_REQ, S_GEN_WAIT,
                 S_RUN_REQ, S_RUN_WAIT, S_CHECK, S_DONE);
  signal fsm : fsm_t := S_IDLE;

  signal st       : state_t;
  signal map_idx  : integer range 0 to NUM_MAPS-1 := 0;

  -- generator
  signal gen_start : std_logic := '0';
  signal gen_seed  : std_logic_vector(15 downto 0);
  signal gen_busy  : std_logic;
  signal gen_done  : std_logic;
  signal win_sig   : win_arr_t;

  -- map engine
  signal eng_start : std_logic := '0';
  signal eng_nib   : std_logic_vector(15 downto 0);
  signal eng_busy  : std_logic;
  signal eng_done  : std_logic;
  signal eng_out   : state_t;

  function init_state(k : std_logic_vector(31 downto 0);
                      d : std_logic_vector(31 downto 0)) return state_t is
    variable s : state_t;
    variable w : std_logic_vector(31 downto 0);
  begin
    for lane in 0 to 31 loop
      if (lane mod 2) = 0 then w := k; else w := d; end if;
      s(lane*4 + 0) := w(31 downto 24);
      s(lane*4 + 1) := w(23 downto 16);
      s(lane*4 + 2) := w(15 downto  8);
      s(lane*4 + 3) := w( 7 downto  0);
    end loop;
    return s;
  end function;

  function masked_sum(dec : state_t; mask : slv1024) return std_logic_vector is
    variable acc : unsigned(15 downto 0) := (others => '0');
    variable m   : std_logic_vector(7 downto 0);
  begin
    for i in 0 to NUM_BYTES-1 loop
      m   := dec(i) and mask(8*i+7 downto 8*i);
      acc := acc + resize(unsigned(m), 16);
    end loop;
    return std_logic_vector(acc);
  end function;

begin

  busy     <= '0' when fsm = S_IDLE else '1';
  gen_seed <= seeds(map_idx);                -- combinational: map_idx stable
  eng_nib  <= nibbles(map_idx);

  gen : entity work.tm_rng_gen
    generic map (UNROLL_K => UNROLL_K)
    port map (clk=>clk, rst=>rst, start=>gen_start, seed_in=>gen_seed,
              busy=>gen_busy, done=>gen_done, win_out=>win_sig);

  eng : entity work.tm_map_engine
    generic map (GROUP_UNROLL => GROUP_UNROLL)
    port map (clk=>clk, rst=>rst, start=>eng_start, state_in=>st,
              win=>win_sig, nibble=>eng_nib,
              busy=>eng_busy, done=>eng_done, state_out=>eng_out);

  process(clk)
    variable dec   : state_t;
    variable sum   : std_logic_vector(15 downto 0);
    variable csval : std_logic_vector(15 downto 0);
  begin
    if rising_edge(clk) then
      if rst = '1' then
        fsm         <= S_IDLE;
        done        <= '0';
        gen_start   <= '0';
        eng_start   <= '0';
        screen_flag <= (others => '0');
      else
        done <= '0';
        case fsm is

          when S_IDLE =>
            if start = '1' then
              st      <= init_state(key, data);
              map_idx <= 0;
              fsm     <= S_EXPAND;
            end if;

          when S_EXPAND =>
            st  <= add_vec(st, exp_vec);
            fsm <= S_GEN_REQ;

          -- request a window; hold start until the generator acknowledges
          -- (busy), then drop it. Robust req/ack -- no pulse-timing race.
          when S_GEN_REQ =>
            gen_start <= '1';
            if gen_busy = '1' then
              gen_start <= '0';
              fsm       <= S_GEN_WAIT;
            end if;

          when S_GEN_WAIT =>
            if gen_done = '1' then
              fsm <= S_RUN_REQ;       -- win_sig now valid for map_idx
            end if;

          when S_RUN_REQ =>
            eng_start <= '1';
            if eng_busy = '1' then
              eng_start <= '0';
              fsm       <= S_RUN_WAIT;
            end if;

          when S_RUN_WAIT =>
            if eng_done = '1' then
              st <= eng_out;
              if map_idx = NUM_MAPS-1 then
                fsm <= S_CHECK;
              else
                map_idx <= map_idx + 1;
                fsm     <= S_GEN_REQ;
              end if;
            end if;

          when S_CHECK =>
            for i in 0 to NUM_BYTES-1 loop
              dec(i) := st(i) xor world_xor(8*i+7 downto 8*i);
            end loop;
            sum   := masked_sum(dec, world_mask);
            csval := dec(CSUM_HI_POS) & dec(CSUM_LO_POS);
            if sum = csval then screen_flag <= x"08"; else screen_flag <= x"00"; end if;
            fsm <= S_DONE;

          when S_DONE =>
            done <= '1';
            fsm  <= S_IDLE;

        end case;
      end if;
    end if;
  end process;

end architecture;
