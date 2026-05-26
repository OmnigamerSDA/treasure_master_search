# Decryption Execution Trace Reference

Date: 2026-04-15

## Scope

This document describes the actual decryption and dispatch flow as observed in the earlier emulator traces, primarily:

- `docs/tm_decrypt.log`
- `docs/tm_other_hyp.log`
- `docs/tm_initial.log`
- `docs/tm_sched.log`

The goal here is not to restate the wiki hypothesis, but to record what the traces show the game really doing.

## High-level pipeline

The observed runtime pipeline is:

1. Refill the 128-byte working/XOR buffer at `$0200-$027F` from PPU memory via `$2006/$2007`.
2. Enter fixed-bank decrypt/dispatch code at `$FC52`.
3. Select a ciphertext source pointer and payload length from small tables indexed by `$46`.
4. XOR-decrypt the selected payload directly into `$06A5+`.
5. Run a checksum helper over the decrypted plaintext.
6. If checksum passes, branch to `$FCA2` and `JMP $06A5`.
7. If checksum fails, fall through to the alternate/failure dispatcher at `$FC93-$FC9F`.

Only the success path is fully captured in the provided traces. The failure dispatcher addresses are known from notes and partial trace evidence, but not fully walked in the same logs.

## Stage 1: refill `$0200-$027F` from PPU

The trace shows a helper reached through `$EE98` jumping to `$BA59`, where the game sets the PPU read address to `$2700` and copies 128 bytes into RAM:

```asm
BA59: A9 00      LDA #$00
BA5B: A2 27      LDX #$27
BA5D: 8E 06 20   STX $2006
BA60: 8D 06 20   STA $2006      ; PPU address = $2700
BA63: A2 7F      LDX #$7F
BA65: AD 07 20   LDA $2007      ; dummy read
BA68: AD 07 20   LDA $2007
BA6B: 9D 00 02   STA $0200,X
BA6E: CA         DEX
BA6F: 10 F7      BPL $BA68
```

What this means:

- The final decrypt-time XOR partner is the RAM image at `$0200-$027F`.
- That RAM image is populated by reading back a PPU buffer starting at `$2700`.
- The copy is descending in RAM address: first byte read after the dummy read lands at `$027F`, final byte lands at `$0200`.

The earlier decrypt traces only showed the readback path. `docs/tm_initial.log`
now captures the corresponding initial population of PPU `$2700-$277F`.

## Stage 1a: initial population of PPU `$2700-$277F`

After the password bytes are validated and expanded into `$0200-$027F`, the
initial trace reaches `$800D: JSR $EE54`, dispatches through `$EE98`, and jumps
to `$A7B6`.

The PPU write routine is:

```asm
A7B6: A9 00      LDA #$00
A7B8: A2 27      LDX #$27
A7BA: 8E 06 20   STX $2006
A7BD: 8D 06 20   STA $2006      ; PPU address = $2700
A7C0: A2 7F      LDX #$7F
A7C2: BD 00 02   LDA $0200,X
A7C5: 8D 07 20   STA $2007
A7C8: CA         DEX
A7C9: 10 F7      BPL $A7C2
A7CB: 60         RTS
```

The source is `$0200-$027F` in descending order. The first write after address
setup comes from `$027F`; the final write comes from `$0200`.

Useful breakpoints:

```text
PC $A7B6    PPU save routine entry
PC $A7BA    PPUADDR high-byte setup for $2700
PC $A7C5    every byte written through $2007
PPU write $2700-$277F, if the emulator supports resolved PPU watchpoints
```

Trace-note caveat: FCEUX-style lines such as `STA $2007 = #$05` show the
previous/read value associated with the operand, not the byte being written.
Use the A register loaded by the immediately preceding `LDA $0200,X`.

## Runtime key schedule state at `$0191-$0194`

`docs/tm_sched.log` confirms that `$0191-$0194` is the live rolling key schedule
state. It is copied from the password-derived key by `$9588`, then advanced in
place as maps execute.

The first observed schedule advance starts from:

```text
$0191-$0194 = 2C A5 B4 2D
```

The algorithm selector is:

```asm
827E: BD 91 01   LDA $0191,X    ; X = (map >> 4)
8281: 4A         LSR
8282: 4A         LSR
8283: 29 07      AND #$07
8285: 18         CLC
8286: 69 27      ADC #$27
8288: 20 6B EE   JSR $EE6B
```

For map `$00`, this selects algorithm `3`, matching the C++ schedule model's
`algorithm_2` then `algorithm_1`.

The trace then executes:

```asm
A084: 20 A1 B7   JSR $B7A1      ; algorithm_2 primitive
B7A1: AD 91 01   LDA $0191
...
B7B1: 8D 91 01   STA $0191
B7B5: 8D 94 01   STA $0194
B7B8: 8E 93 01   STX $0193
B7BB: 8C 92 01   STY $0192

A087: A5 F3      LDA $F3        ; algorithm_1 primitive
A089: A0 03      LDY #$03
A08B: 38         SEC
A08C: 79 91 01   ADC $0191,Y
A08F: 99 91 01   STA $0191,Y
A092: 88         DEY
A093: 10 F7      BPL $A08C
```

The first advance is:

```text
initial key:        2C A5 B4 2D
after algorithm_2:  2D B4 A5 2C
after algorithm_1:  B4 86 D2 2D
```

The advanced state is immediately loaded into the RNG seed and nibble selector:

```asm
82A3: AD 91 01   LDA $0191      ; B4
82A6: 8D 36 04   STA $0436
82A9: AD 92 01   LDA $0192      ; 86
82AC: 8D 37 04   STA $0437
82AF: AD 93 01   LDA $0193      ; D2
82B2: 85 42      STA $42
82B4: AD 94 01   LDA $0194      ; 2D
82B7: 85 43      STA $43
```

The in-RAM byte order is:

```text
$0191-$0194 = rng1 rng2 nibble_lo nibble_hi
```

The forward/GPU schedule-entry byte order is:

```text
rng1 rng2 nibble_hi nibble_lo
```

For the first carnival entry, that means:

```text
RAM state:          B4 86 D2 2D
seed:               B486
nibble selector:    2DD2
entry/blob bytes:   B4 86 2D D2
```

For future traces, the most useful watchpoints are:

```text
PC $827E
PC $8288
PC $A084
PC $B7A1
PC $A087
PC $A08F
PC $82A3
read/write $0191-$0194
```

## Stage 2: fixed-bank decrypt/dispatch at `$FC52`

The fixed-bank dispatcher sets the destination pointer to `$06A5`:

```asm
FC57: A9 A5      LDA #$A5
FC59: 85 6A      STA $6A
FC5B: A9 06      LDA #$06
FC5D: 85 6B      STA $6B        ; destination = $06A5
```

It then selects source pointer and length using tables indexed by `$46`:

```asm
FC65: A6 46      LDX $46
FC67: BD A5 FC   LDA $FCA5,X
FC6A: 85 40      STA $40
FC6C: BD A7 FC   LDA $FCA7,X
FC6F: 85 41      STA $41
FC71: BD AB FC   LDA $FCAB,X
FC74: 85 42      STA $42
```

In the carnival trace:

- `$46 = 0`
- `$40/$41 = $BA/$B8`, so source pointer is `$B8BA`
- `$42 = $72`

The notes already identify the length table as starting:

- `$FCAB: 72 53 ...`

which matches the known carnival length `0x72` and the other-world length `0x53`.

## Stage 3: XOR-decrypt loop

The trace-backed decrypt loop is:

```asm
FC76: A2 7F      LDX #$7F
FC78: A0 00      LDY #$00
FC7A: B1 40      LDA ($40),Y
FC7C: 5D 00 02   EOR $0200,X
FC7F: 99 A5 06   STA $06A5,Y
FC82: CA         DEX
FC83: C8         INY
FC84: C4 42      CPY $42
FC86: D0 F2      BNE $FC7A
```

Semantically:

```text
for y in 0 .. length-1:
    plaintext[y] = ciphertext[source + y] XOR workbuf[0x0200 + (0x7F - y)]
    store to 0x06A5 + y
```

So:

- ciphertext is consumed in ascending order
- the `$0200` work buffer is consumed in descending order
- plaintext is written in ascending order starting at `$06A5`

For carnival:

- destination range is `$06A5-$0716`

For the other-world length `0x53`:

- destination range would be `$06A5-$06F7`

## Stage 4: checksum helper

After decrypting, the game calls a checksum helper through `$EE98`; the relevant inner loop is at `$FD28+`.

The helper accumulates a 16-bit sum into `$42/$43` over the decrypted plaintext bytes up to, but not including, the final checksum bytes:

```asm
FD28: B1 40      LDA ($40),Y
FD2A: 18         CLC
FD2B: 65 42      ADC $42
FD2D: 85 42      STA $42
FD2F: 90 02      BCC $FD33
FD31: E6 43      INC $43
FD33: C8         INY
FD34: 98         TYA
FD35: DD 5E FD   CMP $FD5E,X
FD38: D0 EE      BNE $FD28
```

In the carnival trace:

- the stop value used by this helper is `$70`
- after the sum, it compares against the final two decrypted bytes at `$0715/$0716`

```asm
FD3A: A5 42      LDA $42
FD3C: D1 40      CMP ($40),Y    ; low checksum byte
FD3E: D0 0E      BNE $FD4E
FD40: BD 65 FD   LDA $FD65,X
FD43: F0 07      BEQ $FD4C
FD45: C8         INY
FD46: A5 43      LDA $43
FD48: D1 40      CMP ($40),Y    ; high checksum byte
FD4A: D0 02      BNE $FD4E
FD4C: 18         CLC
FD4D: 60         RTS
```

This matches the working checksum rule already used elsewhere in the project:

- sum plaintext bytes `0 .. length-3`
- compare the 16-bit result against the final two plaintext bytes

For carnival (`length = 0x72`):

- checksum bytes are at offsets `0x70/0x71`, i.e. `$0715/$0716`

For other-world (`length = 0x53`):

- checksum bytes should be at offsets `0x51/0x52`, i.e. `$06F6/$06F7`

The other-world hypothesis trace also shows decrypt writes landing at `$06F6/$06F7`, which is consistent with this layout.

## Stage 5: success branch into decrypted code

On checksum success, the helper returns with carry clear. The caller immediately uses that carry state:

```asm
FC91: 90 0F      BCC $FCA2
FCA2: 4C A5 06   JMP $06A5
```

This is the success handoff into the decrypted loader code.

The carnival trace then enters:

```asm
06A5: A2 05
06A7: EC 51 04
...
```

The other-world hypothesis trace enters:

```asm
06A5: A0 05
06A7: 4C 95 85
...
```

So the fixed decrypt/dispatch logic is the same; only the decrypted payload differs.

## Stage 6: failure path and alternate selection

The success traces do not fully walk the failure path, but the notes identify the fallback dispatcher:

```asm
FC93: A6 46      LDX $46
FC95: BD FCAD,X  LDA ...
FC98: 85 42      STA $42
FC9A: BD FCAF,X  LDA ...
FC9D: 85 43      STA $43
FC9F: 6C 42 00   JMP ($0042)
```

The existing notes state:

- when `$46 = 0`, failure routes to code that increments `$46` and retries
- when `$46 = 1`, it ends

This is enough to explain the observed behavior where a failed carnival checksum can fall through to an alternate code path rather than immediately abort.

## Concrete carnival values

From the successful carnival trace:

- PPU refill source: `$2700+`
- work buffer after refill: `$0200-$027F`
- decrypt destination: `$06A5-$0716`
- ciphertext source: `$B8BA`
- payload length: `$72`
- checksum bytes: `$0715/$0716 = C9 2D`

## Concrete other-world implications

From the successful hypothesis trace:

- decrypt destination is still `$06A5+`
- the tested other-world init bytes enter at `$06A5`
- with length `0x53`, the implied checksum positions are `$06F6/$06F7`

This means any debugger-assisted checksum bypass for the other-world hypothesis should target:

- computed checksum in `$42/$43`
- expected plaintext checksum bytes at `$06F6/$06F7`

## Practical summary

The trace-backed runtime model is:

```text
PPU $2700+ -> refill $0200-$027F
select source/length via tables indexed by $46
decrypt: plaintext[y] = source[y] XOR workbuf[0x7F-y]
write plaintext to $06A5+
checksum plaintext against final two plaintext bytes
if pass: JMP $06A5
if fail: go through $FC93-$FC9F alternate/fallback dispatcher
```

That is the reference execution model to use when reasoning about provenance, debugger pokes, checksum forcing, and alternative payload selection.
