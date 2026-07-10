-- =====================================================================
-- tm_map_pipeline.vhd -- partially-unrolled map pipeline (throughput = clock).
-- ---------------------------------------------------------------------
-- N_STAGES map_engines in series; candidates flow through one stage per map.
-- Throughput becomes a pure function of clock: a candidate exits every
-- II = (16/GROUP_UNROLL)+1 cycles (the engine's per-map latency), regardless
-- of the 27-map depth -- the depth is pipelined away. N_STAGES candidates are
-- in flight; there is NO per-map FSM turnaround (stages just advance in
-- lockstep on the shared engine-done tick).
--
-- Fixed-key enabler: each stage's window + nibble are per-key CONSTANTS
-- (`wins(i)`, `nibs(i)`), wired in -- no "which map" mux, no shared-store read
-- port contention. Generate once per key, hold for the whole ~4B sweep.
--
-- For the full screen, N_STAGES = NUM_MAPS (27). Real builds can also make each
-- stage a tm_map_engine_w (W lanes) to multiply throughput by W; here each
-- stage is single-lane to keep the pipeline mechanics clear.
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.tm_pkg.all;

entity tm_map_pipeline is
  generic (
    N_STAGES     : integer := 4;
    GROUP_UNROLL : integer := 4;
    USE_BLEND    : boolean := true     -- throughput vehicle -> shallower blend
                                       -- datapath by default (higher Fmax knee).
  );
  port (
    clk       : in  std_logic;
    rst       : in  std_logic;
    in_valid  : in  std_logic;
    in_state  : in  state_t;
    wins      : in  win_vec_t(0 to N_STAGES-1);
    nibs      : in  slv16_vec_t(0 to N_STAGES-1);
    tick      : out std_logic;                 -- pulses once per II (debug)
    out_valid : out std_logic;
    out_state : out state_t
  );
end entity;

architecture rtl of tm_map_pipeline is
  signal stage_in  : state_vec_t(0 to N_STAGES-1);
  signal eng_out   : state_vec_t(0 to N_STAGES-1);
  signal eng_start : std_logic := '0';
  signal eng_busy  : std_logic_vector(0 to N_STAGES-1);
  signal eng_done  : std_logic_vector(0 to N_STAGES-1);
  signal pipe_valid: std_logic_vector(0 to N_STAGES-1) := (others => '0');
  signal primed    : std_logic := '0';
begin

  -- All stages share one start/done cadence (identical 16/G timing).
  gen_stages : for i in 0 to N_STAGES-1 generate
    eng_i : entity work.tm_map_engine
      generic map (GROUP_UNROLL => GROUP_UNROLL, USE_BLEND => USE_BLEND)
      port map (clk=>clk, rst=>rst, start=>eng_start,
                state_in=>stage_in(i), win=>wins(i), nibble=>nibs(i),
                busy=>eng_busy(i), done=>eng_done(i), state_out=>eng_out(i));
  end generate;

  tick <= eng_done(0);

  process(clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        eng_start  <= '1';                  -- kick the first cycle
        primed     <= '0';
        pipe_valid <= (others => '0');
        out_valid  <= '0';
      else
        eng_start <= '0';
        out_valid <= '0';

        if primed = '0' then
          -- first launch after reset (start was held high one cycle)
          primed <= '1';
        elsif eng_done(0) = '1' then
          -- one map completed in every stage -> shift the pipeline and relaunch
          out_state <= eng_out(N_STAGES-1);
          out_valid <= pipe_valid(N_STAGES-1);
          for i in N_STAGES-1 downto 1 loop
            stage_in(i)   <= eng_out(i-1);
            pipe_valid(i) <= pipe_valid(i-1);
          end loop;
          stage_in(0)   <= in_state;
          pipe_valid(0) <= in_valid;
          eng_start     <= '1';             -- relaunch all stages
        end if;
      end if;
    end if;
  end process;

end architecture;
