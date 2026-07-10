-- =====================================================================
-- tm_pkg.vhd  -- Treasure Master forward screen: shared types + datapath
-- ---------------------------------------------------------------------
-- VHDL-2008. This package holds the *map datapath* as a pure function so
-- the core (tm_screen_core.vhd) is just an FSM + memory ports wrapped
-- around it. The datapath encodes the key architectural claim from the
-- feasibility study:
--
--   * State is a flat 128-byte (1024-bit) register, NOT split across
--     32 lanes. The cross-lane carry that forces a work-group barrier and
--     a precomputed alg2/alg5 carry table on the GPU is, here, one wire.
--   * The 8 algorithms are FOLDED into one shared datapath selected by a
--     3-bit alg_id used as *control*, not 8 computed results + an 8:1 mux.
--   * Shifts / rotates / masks / bit-placement are interconnect (zero
--     logic): alg0/alg6/alg2/alg5/alg7 cost ~no LUTs; only alg1/alg4
--     need the shared 8-bit adder bank.
--
-- Reference semantics: src/bruteforce/tm_opencl_32_8_test/tm.cl
--   run_alg() (cases 0..7) and run_one_map().
--
-- PARITY GATE: the alg2/alg5 cross-byte wiring below was derived by hand
-- from the uint32 reference in tm_process (even byte -> shift-right form,
-- odd byte -> shift-left form, neighbour = byte i+1, boundary i=127 from
-- the RNG carry bit). It MUST be confirmed bit-exact against the software
-- reference with the existing `--parity 64` harness before trusting any
-- timing. Everything else (alg0/1/3/4/6/7, expand, checksum) is a direct
-- transcription and is low-risk.
-- =====================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package tm_pkg is

  constant NUM_BYTES : integer := 128;   -- working-state width in bytes
  constant NUM_MAPS  : integer := 27;    -- schedule length
  constant INNER     : integer := 16;    -- inner steps per map

  subtype byte_t is std_logic_vector(7 downto 0);
  type    state_t is array (0 to NUM_BYTES-1) of byte_t;

  subtype slv1024 is std_logic_vector(NUM_BYTES*8-1 downto 0);

  -- Per-map resident RNG stream window. The generator walks run_rng from the
  -- map's start seed and fills WIN_LEN consecutive output bytes; the core
  -- indexes a 128-byte sliding window into it. Max offset reached in a map is
  -- 15*128 = 1920, and a byte-alg there reads up to offset+127 = 2047, so
  -- WIN_LEN = 2048 covers every reachable read.
  constant WIN_LEN : integer := 2048;
  type win_arr_t is array (0 to WIN_LEN-1) of byte_t;

  -- Per-key schedule (from the separate key_schedule block): one start seed
  -- and one 16-bit nibble selector per map. 27 x 4 bytes = 108 bytes total.
  type seed_arr_t is array (0 to NUM_MAPS-1) of std_logic_vector(15 downto 0);
  type nib_arr_t  is array (0 to NUM_MAPS-1) of std_logic_vector(15 downto 0);

  -- Unconstrained batch types (for the map-major batched top).
  type state_vec_t is array (natural range <>) of state_t;
  type byte_vec_t  is array (natural range <>) of byte_t;

  -- Per-stage arrays for the unrolled map pipeline (each stage holds a
  -- per-key-CONSTANT window + nibble; fixed key => constants, not muxed).
  type win_vec_t   is array (natural range <>) of win_arr_t;
  type slv16_vec_t is array (natural range <>) of std_logic_vector(15 downto 0);

  -- Bounds-safe stream read. WIN_LEN is a power of two (2^11), so the wrap is a
  -- free bit-mask -- NOT a magnitude comparator. The f+c<=15 map invariant keeps
  -- every REACHED read in [0, WIN_LEN-1], so masking is bit-identical to a clamp
  -- on all reachable indices while costing zero LUTs (drops the compare).
  function win_byte(w : win_arr_t; idx : integer) return byte_t;

  -- Two-row window slice: the ONE 16:1 row-select (rows f and f+1) that every
  -- per-step read shares. Extract once per (f), then reversed / forward / carry
  -- all index into it -- this is the "single window read" the datapath wants,
  -- made structural instead of hoping synthesis CSEs three separate rng_vec
  -- calls across the win_byte boundary. c<=15 so only rows f,f+1 are touched.
  -- (An explicit two-`win_row` 16:1 form was prototyped + measured 2026-07-08:
  --  netlist-identical at G=1, ~6% LUT worse at G=2 on Kintex-7 -- Vivado
  --  already infers the 16:1 from this flat index, so the flat form is kept.)
  type tworow_t is array (0 to 2*NUM_BYTES-1) of byte_t;
  function rng_rows(w : win_arr_t; f : integer) return tworow_t;

  -- Aligned reads off a pre-extracted two-row slice (no further row-select):
  --   rng_vec_r     (reversed, alg0/1/3/4):  r(i)  = rr[c + 127 - i]
  --   rng_vec_fwd_r (forward,  alg6):        r6(i) = rr[c + i]
  --   carry (alg2/alg5)                       = rr[c](7)      (row f, byte c)
  function rng_vec_r(rr : tworow_t; c : integer) return state_t;
  function rng_vec_fwd_r(rr : tworow_t; c : integer) return state_t;

  -- RNG vector reads, addressed by the (f,c) offset pair (off = 128*f + c).
  -- Because c <= 15 < 128, the 128-byte window straddles only rows f and f+1,
  -- so the read is a 16:1 ROW-SELECT by f + a small (<=15) align by c -- NOT a
  -- 2048-wide barrel. f selects the row, c/i address within two rows. These are
  -- kept for direct callers/testbenches; they now compose rng_rows + *_r so the
  -- extraction logic lives in ONE place.
  --   rng_vec     (reversed, alg0/1/3/4):  r(i)  = stream[128f + c + 127 - i]
  --   rng_vec_fwd (forward,  alg6):        r6(i) = stream[128f + c + i]
  function rng_vec(w : win_arr_t; f : integer; c : integer) return state_t;
  function rng_vec_fwd(w : win_arr_t; f : integer; c : integer) return state_t;

  -- alg2/alg5 carry bit = bit7 of stream[128f + c] = row f, byte c.
  function rng_carry_bit(w : win_arr_t; f : integer; c : integer) return std_logic;

  -- RNG offset advance per algorithm (matches the offset-stream kernel):
  --   alg 0,1,3,4,6 consume a full 128-byte vector -> +128
  --   alg 2,5       consume one carry bit           -> +1
  --   alg 7         consumes nothing                -> +0
  function rng_advance(alg_id : std_logic_vector(2 downto 0)) return integer;

  -- pack/unpack between the byte-array view and a flat 1024-bit memory word.
  -- Byte i occupies bits [8*i+7 : 8*i]; host BRAM contents must match.
  function to_state(v : slv1024) return state_t;
  function to_slv  (s : state_t)  return slv1024;

  -- alg_id for inner step i: byte i of state, high nibble if nib_top='1',
  -- then bits [3:1] (matches run_one_map).
  function pick_alg(s : state_t; i : integer; nib_top : std_logic)
    return std_logic_vector;

  -- The whole map step: apply one algorithm (alg_id) to all 128 bytes.
  --   s     : current state
  --   r      : RNG vector for this step (regular bytes; alg0/alg6 derive
  --            their single bit from r(i)(7), so only ONE stream is stored)
  --   carry  : alg2/alg5 boundary bit (RNG-sourced) for byte 127
  -- r  = reversed window vector (alg0/1/3/4), r6 = forward window vector (alg6).
  function apply_map(s     : state_t;
                     r     : state_t;
                     r6    : state_t;
                     alg_id : std_logic_vector(2 downto 0);
                     carry  : std_logic) return state_t;

  -- Op-pair blend-tree form (port of the natmap blmerge structure): instead of
  -- a flat 8:1 select per byte, compute 4 op-pair candidates and a 2-level
  -- blend -- {0,6} share a shifter, {1,4} an adder, {3,7} an xor, {2,5} the
  -- cross-byte rotate. Shallower/smaller select -> shorter alg-step path ->
  -- higher Fmax / more GROUP_UNROLL. Proven bit-identical to apply_map in
  -- tb_apply_blend. ONE rng vector feeds all candidates (blmerge2 single-read).
  function apply_map_blend(s     : state_t;
                           r     : state_t;
                           r6    : state_t;
                           alg_id : std_logic_vector(2 downto 0);
                           carry  : std_logic) return state_t;

  -- Per-byte add used by expansion (and structurally identical to alg1).
  function add_vec(s : state_t; e : state_t) return state_t;

  -- One run_rng step, derived in closed form from generate_rng_table -- NO
  -- table. seed = A(15:8) & B(7:0). Returns the next seed and the output byte
  -- ((result>>8) xor result). ~4 byte add-with-carry ops; a few dozen LUTs.
  -- Validated exhaustively (all 65536 seeds) against the repo rng_table in
  -- tb_rng_step.vhd. This is what lets the generator drop the 128 KB ROM and
  -- unroll the run_rng chain.
  procedure rng_step(seed  : in  std_logic_vector(15 downto 0);
                     nseed : out std_logic_vector(15 downto 0);
                     obyte : out std_logic_vector(7 downto 0));

end package tm_pkg;

package body tm_pkg is

  function rng_advance(alg_id : std_logic_vector(2 downto 0)) return integer is
  begin
    case alg_id is
      when "010" | "101" => return 1;    -- alg2 / alg5
      when "111"         => return 0;    -- alg7
      when others        => return 128;  -- alg0/1/3/4/6
    end case;
  end function;

  function to_state(v : slv1024) return state_t is
    variable s : state_t;
  begin
    for i in 0 to NUM_BYTES-1 loop
      s(i) := v(8*i+7 downto 8*i);
    end loop;
    return s;
  end function;

  function to_slv(s : state_t) return slv1024 is
    variable v : slv1024;
  begin
    for i in 0 to NUM_BYTES-1 loop
      v(8*i+7 downto 8*i) := s(i);
    end loop;
    return v;
  end function;

  function add_vec(s : state_t; e : state_t) return state_t is
    variable res : state_t;
  begin
    for i in 0 to NUM_BYTES-1 loop
      res(i) := std_logic_vector(unsigned(s(i)) + unsigned(e(i)));
    end loop;
    return res;
  end function;

  function win_byte(w : win_arr_t; idx : integer) return byte_t is
  begin
    -- WIN_LEN = 2^11; `mod` by a power of two is a bit-mask (0 LUTs of compare).
    -- VHDL `mod` result takes the sign of the (positive) divisor, so this is in
    -- [0, WIN_LEN-1] even for a stray negative idx -- no clamp comparator needed.
    return w(idx mod WIN_LEN);
  end function;

  -- Extract the two candidate rows (f and f+1) ONCE via a 16:1 select on f.
  -- rr(0..127) = row f, rr(128..255) = row f+1. Every per-step read (reversed,
  -- forward, carry) then indexes this same slice -> one row-select, not three.
  function rng_rows(w : win_arr_t; f : integer) return tworow_t is
    variable rr : tworow_t;
  begin
    for j in 0 to 2*NUM_BYTES-1 loop
      rr(j) := win_byte(w, 128*f + j);           -- 16:1 on f (shared)
    end loop;
    return rr;
  end function;

  -- Reversed read off the pre-extracted slice. c<=15, i in [0,127] => the
  -- position c+127-i lies in [c, c+127] <= 142 < 256, so it never leaves rr.
  function rng_vec_r(rr : tworow_t; c : integer) return state_t is
    variable r : state_t;
  begin
    for i in 0 to NUM_BYTES-1 loop
      r(i) := rr(c + 127 - i);                   -- reversed window position
    end loop;
    return r;
  end function;

  -- Forward read (alg6) off the same slice; position c+i in [c, c+127] <= 142.
  function rng_vec_fwd_r(rr : tworow_t; c : integer) return state_t is
    variable r : state_t;
  begin
    for i in 0 to NUM_BYTES-1 loop
      r(i) := rr(c + i);                          -- forward window position
    end loop;
    return r;
  end function;

  -- Compatibility wrappers: same values as before, now composed from the shared
  -- extraction so the row-select logic exists in exactly one place.
  function rng_vec(w : win_arr_t; f : integer; c : integer) return state_t is
  begin
    return rng_vec_r(rng_rows(w, f), c);
  end function;

  function rng_vec_fwd(w : win_arr_t; f : integer; c : integer) return state_t is
  begin
    return rng_vec_fwd_r(rng_rows(w, f), c);
  end function;

  function rng_carry_bit(w : win_arr_t; f : integer; c : integer) return std_logic is
  begin
    return rng_rows(w, f)(c)(7);                  -- row f, byte c, MSB
  end function;

  function pick_alg(s : state_t; i : integer; nib_top : std_logic)
    return std_logic_vector is
    variable b : std_logic_vector(7 downto 0);
  begin
    b := s(i);
    if nib_top = '1' then
      b := "0000" & b(7 downto 4);
    end if;
    return b(3 downto 1);
  end function;

  function apply_map_blend(s     : state_t;
                           r     : state_t;
                           r6    : state_t;
                           alg_id : std_logic_vector(2 downto 0);
                           carry  : std_logic) return state_t is
    variable res     : state_t;
    variable x, rb, r6b : std_logic_vector(7 downto 0);
    variable shifted, smask, inj, addend, xop : std_logic_vector(7 downto 0);
    variable c_sh, c_add, c_x, c_rot          : std_logic_vector(7 downto 0);
    variable cin4       : unsigned(0 downto 0);
    variable n_hi, n_lo : std_logic;
    variable is6, is4, is7, is5         : boolean;
    variable is_rot, is_xor, is_add     : boolean;
  begin
    is6    := (alg_id = "110");
    is4    := (alg_id = "100");
    is7    := (alg_id = "111");
    is5    := (alg_id = "101");
    is_rot := (alg_id = "010") or (alg_id = "101");
    is_xor := (alg_id = "011") or (alg_id = "111");
    is_add := (alg_id = "001") or (alg_id = "100");

    for i in 0 to NUM_BYTES-1 loop
      x   := s(i);
      rb  := r(i);
      r6b := r6(i);
      if i < NUM_BYTES-1 then n_hi := s(i+1)(7); n_lo := s(i+1)(0);
      else                    n_hi := carry;     n_lo := carry; end if;

      -- {0,6}: one shifter; alg0 bit from reversed r, alg6 bit from forward r6
      if is6 then shifted := '0' & x(7 downto 1); smask := x"7F"; inj := r6b and x"80";
      else        shifted := x(6 downto 0) & '0'; smask := x"FE"; inj := "0000000" & rb(7);
      end if;
      c_sh := (shifted and smask) or inj;

      -- {1,4}: ONE adder. alg1 = x + r; alg4 = x - r = x + ~r + 1, folded into
      -- the shared adder's carry-in (cin4) instead of a second increment.
      if is4 then addend := not rb; cin4 := "1"; else addend := rb; cin4 := "0"; end if;
      c_add := std_logic_vector(unsigned(x) + unsigned(addend) + cin4);

      -- {3,7}: one xor, operand r (alg3) or 0xFF (alg7)
      if is7 then xop := x"FF"; else xop := rb; end if;
      c_x := x xor xop;

      -- {2,5}: cross-byte rotate (wiring); is5 swaps even/odd direction
      if (i mod 2) = 0 then
        if is5 then c_rot := x(6 downto 0) & n_lo; else c_rot := n_hi & x(7 downto 1); end if;
      else
        if is5 then c_rot := n_hi & x(7 downto 1); else c_rot := x(6 downto 0) & n_lo; end if;
      end if;

      -- 2-level blend (4:1): rot / xor / add / shift
      if    is_rot then res(i) := c_rot;
      elsif is_xor then res(i) := c_x;
      elsif is_add then res(i) := c_add;
      else              res(i) := c_sh;
      end if;
    end loop;
    return res;
  end function;

  procedure rng_step(seed  : in  std_logic_vector(15 downto 0);
                     nseed : out std_logic_vector(15 downto 0);
                     obyte : out std_logic_vector(7 downto 0)) is
    variable A, B : unsigned(8 downto 0);   -- 9-bit: bit 8 = carry-out
    variable c    : unsigned(8 downto 0);
  begin
    A := '0' & unsigned(seed(15 downto 8));
    B := '0' & unsigned(seed( 7 downto 0));
    B := ('0' & B(7 downto 0)) + A;                            -- B = B + A (carry discarded)
    A := ('0' & A(7 downto 0)) + to_unsigned(16#89#, 9);       -- A = A + 0x89
    c := "00000000" & A(8);
    B := ('0' & B(7 downto 0)) + to_unsigned(16#2A#, 9) + c;   -- B = B + 0x2A + carry
    c := "00000000" & B(8);
    A := ('0' & A(7 downto 0)) + to_unsigned(16#21#, 9) + c;   -- A = A + 0x21 + carry
    c := "00000000" & A(8);
    B := ('0' & B(7 downto 0)) + to_unsigned(16#43#, 9) + c;   -- B = B + 0x43 + carry
    nseed := std_logic_vector(A(7 downto 0)) & std_logic_vector(B(7 downto 0));
    obyte := std_logic_vector(A(7 downto 0) xor B(7 downto 0));
  end procedure;

  function apply_map(s     : state_t;
                     r     : state_t;
                     r6    : state_t;
                     alg_id : std_logic_vector(2 downto 0);
                     carry  : std_logic) return state_t is
    variable res  : state_t;
    variable x    : unsigned(7 downto 0);
    variable rb   : unsigned(7 downto 0);
    variable n_hi : std_logic;   -- neighbour byte (i+1) bit 7
    variable n_lo : std_logic;   -- neighbour byte (i+1) bit 0
  begin
    for i in 0 to NUM_BYTES-1 loop
      x  := unsigned(s(i));
      rb := unsigned(r(i));

      -- neighbour bits for the cross-byte shift algs (pure routing).
      -- Boundary byte 127 takes the RNG-sourced carry instead of i+1.
      if i < NUM_BYTES-1 then
        n_hi := s(i+1)(7);
        n_lo := s(i+1)(0);
      else
        n_hi := carry;
        n_lo := carry;
      end if;

      case alg_id is
        -- alg0: (x<<1)&0xFE | rng_bit7   -> wiring + 1 RNG bit
        when "000" => res(i) := s(i)(6 downto 0) & r(i)(7);

        -- alg1: x + rng   (shared adder bank)
        when "001" => res(i) := std_logic_vector(x + rb);

        -- alg3: x XOR rng   (wiring)
        when "011" => res(i) := s(i) xor r(i);

        -- alg4: x + (~rng) + 1 = x - rng   (same adder, invert operand + cin)
        when "100" => res(i) := std_logic_vector(x + (not rb) + 1);

        -- alg6: (x>>1)&0x7F | rng_bit7  (FORWARD vector r6, see rng_vec_fwd)
        when "110" => res(i) := r6(i)(7) & s(i)(7 downto 1);

        -- alg7: ~x   (wiring)
        when "111" => res(i) := not s(i);

        -- alg2: 1-bit shift across the byte array (PARITY-GATE wiring).
        --   even byte -> {neighbour[7], self[7:1]}
        --   odd  byte -> {self[6:0], neighbour[0]}   (i=127 uses carry)
        when "010" =>
          if (i mod 2) = 0 then
            res(i) := n_hi & s(i)(7 downto 1);
          else
            res(i) := s(i)(6 downto 0) & n_lo;
          end if;

        -- alg5: mirror of alg2 (PARITY-GATE wiring).
        --   even byte -> {self[6:0], neighbour[0]}
        --   odd  byte -> {neighbour[7], self[7:1]}   (i=127 uses carry)
        when others =>  -- "101"
          if (i mod 2) = 0 then
            res(i) := s(i)(6 downto 0) & n_lo;
          else
            res(i) := n_hi & s(i)(7 downto 1);
          end if;
      end case;
    end loop;
    return res;
  end function;

end package body tm_pkg;
