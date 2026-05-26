# Treasure Master NES Password Conversion Algorithm

*Derived from 6502 CPU execution traces, including the original
`keysched.log` work and the later `tm_initial.log` / `tm_sched.log`
confirmation traces.*

## Overview

The NES password system converts a 24-character password into an 8-byte key+data pair through a multi-step process involving 5-bit packing, checksum validation, a bit rotation, and RNG expansion. The packed bytes are NOT key+data directly — a rotation step transforms them first.

**Critical implementation detail:** The rotation at step 4 is a **121-bit** rotation (120-bit buffer + 1-bit carry), not a pure 120-bit circular rotation. The NES carry flag starts at 0 (from `CPY #$20`) and persists across loop iterations. The inverse requires trying both final-carry values (0 and 1) and validating which produces carry=0 at the initial state.

## Conversion Flow

### Step 1: Character Lookup ($B0BE-$B0CB)

Each password character's tile value (from `$016A+Y`) is searched in a 32-entry lookup table at `$B145`. The position X where the match is found (0-31) is the 5-bit code.

Character set (32 values, 5 bits each):
```
 0-9  : digits '0'-'9'
10-30 : consonants B,C,D,F,G,H,J,K,L,M,N,P,Q,R,S,T,V,W,X,Y,Z
31    : '!'
```

### Step 2: Bit Packing ($B0CB-$B0DE)

The position X is shifted left 3 times (`ASL x3`), putting the 5-bit code into the high bits of A. Then 5 iterations of ROL cascade through the 15-byte buffer at `$0200-$020E` (X=0x0E down to X=0x00), shifting in one bit per iteration MSB-first.

```
24 characters x 5 bits = 120 bits = 15 bytes
```

The current `password_codec.py` `password_to_bytes()` correctly implements this step. The known password `3HDJL9DNQV2WYTV4S91RXR86` produces packed bytes:
```
1B D9 09 25 94 B6 85 BE E7 44 C2 43 7E 5D 06
```

### Step 3: First Checksum ($FD18, X=0)

Called at `$B0E7` with X=0. The $FD18 subroutine validates the packed buffer using parameterized tables:

| Parameter | Table | X=0 value | Meaning |
|-----------|-------|-----------|---------|
| Pointer lo | $FD50 | 0x00 | Buffer at $0200 |
| Pointer hi | $FD57 | 0x02 | |
| Byte count | $FD5E | 0x0D (13) | Sum 13 bytes (0-12) |
| Check hi | $FD65 | 0xFF (nonzero) | Check BOTH lo and hi bytes |

Algorithm:
```
$42 = sum(buffer[0..12]) & 0xFF     (accumulator lo)
$43 = sum(buffer[0..12]) >> 8       (accumulator hi, via carry)
Validate: buffer[13] == $42 AND buffer[14] == $43
```

For the known password: `sum(1B+D9+09+25+94+B6+85+BE+E7+44+C2+43+7E) = 0x065D`
→ byte[13]=0x5D, byte[14]=0x06 ✓

If this check fails, carry is set and processing jumps to error at $B107.

### Step 4: Bit Rotation ($B0ED-$B0FD) — CRITICAL STEP

After the first checksum passes, the NES performs a **120-bit circular left rotation** of the entire 15-byte buffer:

```asm
B0ED: LDY $0200       ; Y = byte[0] of packed buffer (rotation count)
B0F0: CPY #$20        ; if byte[0] >= 32, ERROR (password rejected)
B0F4: LDX #$0E        ; inner loop: X = 14 down to 0
B0F6: ROL $0200,X     ; ROL cascade $020E→$020D→...→$0200
B0F9: DEX
B0FA: BPL $B0F6
B0FC: DEY             ; outer loop: repeat Y times
B0FD: BNE $B0F4
```

This rotates the buffer left by `byte[0]` bit positions. The carry flag persists across outer loop iterations — it starts at 0 (from `CPY`) and after each inner cascade equals the old MSB of byte[0]. This makes the rotation a **121-bit** circular shift (120-bit buffer + 1-bit carry), not a pure 120-bit rotation.

For the known password: byte[0] = 0x1B = 27, so the buffer is rotated left 27 bits. The final carry after 27 steps is 1.

**Constraint**: `byte[0] < 0x20` (32) is required, or the password is rejected.

**Inverse recovery**: Since the final carry is unknown when reversing, both carry=0 and carry=1 must be tried. The correct one produces initial carry=0 after undoing all steps. For some buffers, both carries yield initial carry=0 — validity must be confirmed by checking `pre[0] == rot` and the pre-rotation checksum.

After rotation, the buffer transforms from:
```
Pre:  1B D9 09 25 94 B6 85 BE E7 44 C2 43 7E 5D 06
Post: 2C A5 B4 2D F7 3A 26 12 [rotated checksum bytes...]
      ^^^^^^^^^^^^ ^^^^^^^^^^^^
      key          data
```

### Step 5: Second Checksum ($FD18, X=1)

Called at `$B101` with X=1. Same subroutine, different parameters:

| Parameter | Table | X=1 value | Meaning |
|-----------|-------|-----------|---------|
| Pointer lo | $FD50 | 0x00 | Buffer at $0200 (post-rotation!) |
| Pointer hi | $FD57 | 0x02 | |
| Byte count | $FD5E | 0x08 (8) | Sum 8 bytes (0-7) |
| Check hi | $FD65 | 0x00 (zero) | Check ONLY lo byte |

Algorithm:
```
$42 = sum(buffer[0..7]) & 0xFF
Validate: buffer[8] == $42    (hi byte NOT checked)
```

For the known password (post-rotation): `sum(2C+A5+B4+2D+F7+3A+26+12) = 0x031B`
→ byte[8] = 0x1B ✓ (only low byte checked)

### Step 6: Key+Data Extraction

Post-rotation bytes 0-7 are key+data directly:
```
$0200: 2C A5 B4 2D F7 3A 26 12
       ^^^^^^^^^^^^ ^^^^^^^^^^^^
       key          data
```

### Step 7: Copy to $0191 ($9588)

At `$9588`, bytes `$0200[0:3]` are copied to `$0191-$0194`:
```asm
9588: LDX #$03
958A: LDA $0200,X
958D: STA $0191,X
9590: DEX
9591: BPL $958A
```
Confirmed: `$0191-$0194 = 2C A5 B4 2D`

`docs/tm_initial.log` confirms this same entry point after the first write of
the expanded bytes to PPU `$2700`. The trace reaches `$8011: JSR $9588`, then
`$9588: LDX #$03`, `$958A: LDA $0200,X`, and `$958D: STA $0191,X`. That log
ends immediately after the first store to `$0194`, so use `tm_sched.log` for
the schedule advance itself.

### Step 7b: Live key schedule state at $0191

`docs/tm_sched.log` confirms `$0191-$0194` is the live rolling 4-byte key
schedule state, not only an immutable saved key copy.

The first observed advance starts with the initial key:

```text
$0191-$0194 = 2C A5 B4 2D
```

The map index in `$F3` is `0x00`. The selector at `$827E-$8288` reads the
schedule byte selected by `(map >> 4)`, derives the algorithm nibble, adds
`0x27`, and dispatches through `$EE6B`:

```asm
827E: LDA $0191,X    ; X = (map >> 4), here 0
8281: LSR
8282: LSR
8283: AND #$07       ; algorithm = ($0191 >> 2) & 7 = 3
8285: CLC
8286: ADC #$27
8288: JSR $EE6B
```

For map `0x00`, algorithm `3` is selected. In the repo model this is
`algorithm_2` followed by `algorithm_1`.

The first primitive is reached at `$A084: JSR $B7A1`. The subroutine
`$B7A1-$B7BB` implements the algorithm-2 byte permutation/rotation, using
`$F3` as the map addend:

```asm
B7A1: LDA $0191
B7A4: CLC
B7A5: ADC $F3
B7A7: PHA
B7A8: LDX $0192
B7AB: LDY $0193
B7AE: LDA $0194
B7B1: STA $0191
B7B4: PLA
B7B5: STA $0194
B7B8: STX $0193
B7BB: STY $0192
```

For map `0x00`, this changes:

```text
2C A5 B4 2D -> 2D B4 A5 2C
```

The second primitive loop at `$A087-$A093` implements algorithm 1, a rolling
add from byte 3 down to byte 0, starting from `$F3` with carry set:

```asm
A087: LDA $F3
A089: LDY #$03
A08B: SEC
A08C: ADC $0191,Y
A08F: STA $0191,Y
A092: DEY
A093: BPL $A08C
```

It advances the state to:

```text
2D B4 A5 2C -> B4 86 D2 2D
```

The post-advance state is immediately consumed at `$82A3-$82B7`:

```asm
82A3: LDA $0191      ; B4
82A6: STA $0436      ; rng1
82A9: LDA $0192      ; 86
82AC: STA $0437      ; rng2
82AF: LDA $0193      ; D2
82B2: STA $42        ; nibble selector low
82B4: LDA $0194      ; 2D
82B7: STA $43        ; nibble selector high
```

Important byte order:

```text
RAM $0191-$0194:    rng1 rng2 nibble_lo nibble_hi
Model/blob order:   rng1 rng2 nibble_hi nibble_lo
```

So the first generated entry is:

```text
RAM state:          B4 86 D2 2D
seed:               B486
nibble selector:    2DD2
entry/blob bytes:   B4 86 2D D2
```

Useful emulator breakpoints/watchpoints:

```text
PC $827E    algorithm selector reads schedule state
PC $8288    dispatch to selected schedule algorithm
PC $A084    algorithm-3 wrapper for the first map
PC $B7A1    algorithm-2 primitive
PC $A087    algorithm-1 primitive
PC $A08F    in-place schedule writes
PC $82A3    post-advance schedule consumed as rng/selector
read/write $0191-$0194
```

### Step 8: RNG Expansion ($B12C)

Seeds the RNG with `$0200[0]` as rngA and `$0200[1]` as rngB (`$0436/$0437`), then expands to 128 bytes:

```asm
B11C: LDA $0200     → STA $0436    ; rngA = key_hi = 0x2C
B122: LDA $0201     → STA $0437    ; rngB = key_lo = 0xA5
B128: LDX #$00, LDY #$08
B12C: JSR $F1DA                     ; run_rng → A = rngA ^ rngB
B12F: CLC
B130: ADC $0200,X                   ; A += source_byte[X]
B133: STA $0200,Y                   ; store at destination[Y]
B136: INX, INY
B138: BPL $B12C                     ; loop Y=8..127 (X=0..119)
```

This is the same as `generate_expansion_values_8` in the C++ codebase. Each new byte = `run_rng() + buffer[Y-8]`, creating a feedback expansion. Bytes 0-7 (key+data) are preserved; bytes 8-127 are the expanded working code.

## Confirmed Values

### Carnival Broadcast Password
```
Password:  3HDJL9DNQV2WYTV4S91RXR86
Key:       0x2CA5B42D
Data:      0xF73A2612
```

The data value `0xF73A2612` is `kBaselineStart` in `test_cpu.cpp` — the original developer used the real carnival data value as the test constant throughout the codebase.

## Byte Layout Summary

### Pre-Rotation (after 5-bit packing, before ROL cascade)
```
Byte  | Content
------|------------------------------------------
0     | Rotation count (must be < 0x20)
1-12  | Scrambled payload
13    | sum(bytes 0-12) & 0xFF
14    | sum(bytes 0-12) >> 8
```

### Post-Rotation (after ROL cascade by byte[0] positions)
```
Byte  | Content
------|------------------------------------------
0-3   | Key (big-endian uint32)
4-7   | Data (big-endian uint32)
8     | sum(bytes 0-7) & 0xFF (low byte only)
9-14  | Rotated remnants of pre-rotation checksums
```

## Checksum Constraint Summary

Two independent checksums protect the password:

1. **Pre-rotation checksum** (16-bit, both bytes checked):
   `byte[13:14] = sum(byte[0:12])` as (lo, hi)

2. **Post-rotation checksum** (8-bit, low byte only):
   `byte[8] = sum(byte[0:7]) & 0xFF`

3. **Rotation constraint**: pre-rotation `byte[0] < 0x20`

## Free Bits for TAS Optimization

For a given key+data (8 bytes fixed), the degrees of freedom are:

- Post-rotation bytes 9-14 (6 bytes = 48 bits) are NOT directly checked by the second checksum
- However, they ARE constrained by the pre-rotation checksum (bytes 13-14 must equal sum of 0-12)
- The rotation amount couples the pre- and post-rotation buffers via the 121-bit carry chain
- Post-rotation byte 8 is fully determined: `sum(key+data) & 0xFF`
- Which free bytes affect `pre[0]` depends on the rotation amount (typically bytes 11-12 for rot ~27)

**Optimization approach** (two-phase search per rotation):
1. **Phase 1**: Identify which free post-rotation bytes affect `pre[0]`. Vary those jointly (up to 2 bytes, 65536 combinations) to find values where `pre[0] == rot`.
2. **Phase 2**: For each phase-1 hit, vary 1-2 additional free bytes (up to 65536 combinations) to satisfy the pre-rotation checksum `pre[13:14] == sum(pre[0:12])`.
3. Score each valid password by cursor travel distance on the 32-position ring.

The 226 known carnival collision data values provide 226 independent starting points, each potentially yielding a different optimal password. Initial results show **25+ steps saved** versus the broadcast password (129 vs 154 travel).

### Cursor Travel Scoring

The password entry screen uses a 32-position linear ring:
```
0 1 2 3 4 5 6 7 8 9 B C D F G H J K L M N P Q R S T V W X Y Z !
```
The cursor starts at position 0 ('0'). Distance between positions a and b = `min(|a-b|, 32-|a-b|)`. Total travel = sum of distances between consecutive selections.

## NES ROM Tables Referenced

### $B145: Character Tile Lookup (32 bytes)

Maps 5-bit password code (index) to NES tile ID. Verified against CPU trace (7/32 entries) and full ROM dump.

```
00 01 02 03 04 05 06 07 08 09 18 12 11 14 17 1A
1D 20 22 1E 1B 23 0A 13 0E 16 15 0D 0F 19 0C 26
```

| Code | Tile | Char | Code | Tile | Char | Code | Tile | Char | Code | Tile | Char |
|------|------|------|------|------|------|------|------|------|------|------|------|
| 0 | 0x00 | 0 | 8 | 0x08 | 8 | 16 | 0x1D | J | 24 | 0x0E | S |
| 1 | 0x01 | 1 | 9 | 0x09 | 9 | 17 | 0x20 | K | 25 | 0x16 | T |
| 2 | 0x02 | 2 | 10 | 0x18 | B | 18 | 0x22 | L | 26 | 0x15 | V |
| 3 | 0x03 | 3 | 11 | 0x12 | C | 19 | 0x1E | M | 27 | 0x0D | W |
| 4 | 0x04 | 4 | 12 | 0x11 | D | 20 | 0x1B | N | 28 | 0x0F | X |
| 5 | 0x05 | 5 | 13 | 0x14 | F | 21 | 0x23 | P | 29 | 0x19 | Y |
| 6 | 0x06 | 6 | 14 | 0x17 | G | 22 | 0x0A | Q | 30 | 0x0C | Z |
| 7 | 0x07 | 7 | 15 | 0x1A | H | 23 | 0x13 | R | 31 | 0x26 | ! |

### $FD50-$FD6B: Checksum Parameter Tables (7 entries each)

```
$FD50 (ptr_lo):   00 00 A5 A5 6A 95 00
$FD57 (ptr_hi):   02 02 06 06 01 01 00
$FD5E (count):    0D 08 70 51 23 0D F8
$FD65 (check_hi): FF 00 FF (+ 4 more entries not yet dumped)
```

Only X=0 (pre-rotation) and X=1 (post-rotation) are used for password validation. Other entries point to different buffers ($06A5, $016A, $0195) for game-internal checksums.
