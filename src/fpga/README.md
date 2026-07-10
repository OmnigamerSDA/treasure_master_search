# FPGA forward screening core — RTL (GHDL-verified)

Feasibility RTL for a Treasure Master forward screener on FPGA. The goal is a
device that, given a key, screens its data candidates with **mostly logic and
limited memory** — no 8 MB RNG tables, no 128 KB `rng_table` ROM, no host-built
offset stream. All datapath/RNG pieces are simulated and checked under GHDL;
the remaining gates (value-parity vs. the software screen, P&R numbers) are
called out explicitly.

## Closeout verdict (2026-07-08)

This RTL is useful as a checked architectural probe, but it is **not a GPU-class
throughput path** for the diffuse forward-screen workload.

Best measured unit after the RAM-window/blend work is `tm_map_engine_ram`,
`GROUP_UNROLL=1`, `USE_BLEND=true`: **~23.1 K LUT, 850 LUTRAM, 1.1 K FF,
post-route Fmax ~158 MHz** on Kintex-7. That is only
`158 MHz / (27 maps * 16 steps) ~= 0.366 M candidates/s` per engine before final
screen overhead. `GROUP_UNROLL=2` roughly doubles area and drops Fmax to
~96 MHz, so throughput/area gets worse. The prior "G=16 / 500 MHz" pipeline
figures were pre-P&R upper bounds and should not be used for planning.

The structural reason is the algorithm, not just this RTL:

- The key schedule/control surface is reduced: each inner step selects an
  algorithm from one state byte/nibble plus the schedule bit.
- The selected primitive still mutates the **full 128-byte / 1024-bit state**.
  Later dispatch bytes are read from that already-mutated state, so the full
  state must be carried through each semantic step unless a separate dependency
  proof eliminates most bytes for the final predicate.
- A fully unrolled one-candidate/clock path would need roughly
  `27 * 16 = 432` dependent transform stages, each carrying 1024 state bits and
  performing a 128-byte transform. It also creates hundreds of concurrent
  window/RNG read sites or heavy table replication. This can buy throughput, but
  not sublinear area growth; it is a large FPGA occupancy trade, not a hidden
  AES-like compact pipeline.

Conclusion: do not spend more effort on ordinary FPGA tuning for raw
candidate-rate. The only reasons to keep this tree alive are (1) correctness
reference value, (2) a very specific perf/W deployment where low absolute
throughput is acceptable, or (3) a future algorithmic dependency collapse that
proves most of the 128-byte state is dead for the target predicate. Without (3),
GPU/CPU remain the right production path.

## Architecture

```
 key ─▶ [key_schedule]──27×(seed,nibble)──┐         (separate pure-key block; here an input)
                                          ▼
 seed ─▶ [tm_rng_gen]  walks run_rng via combinational rng_step (NO ROM),
          │            unrolled K-deep, fills ONE 2 KB map window
          ▼
       [2 KB window] ──resident──▶ [tm_map_engine] 16 inner steps over the
                                    window, GROUP_UNROLL G steps/clock,
                                    folded alg datapath (alg_id = control)
                                          │
                       expand ▶ 27 maps ▶ │ ▶ decrypt + checksum ▶ flag
                                    [tm_screen_top]
```

- **No table.** `rng_table[seed]` is a closed-form ~4 byte-add function
  (`rng_step` in `tm_pkg`), proven equal to the real table for all 65,536 seeds.
- **No large memory.** The only resident RNG is **one 2 KB map window**, a
  rate-matching buffer between the serial `run_rng` producer (1 byte/clk, or
  K/clk unrolled) and the 128-byte/clk parallel datapath. See *Memory* below.
- **Folded datapath.** The 8 algorithms are one configurable per-byte unit
  selected by `alg_id`; shifts/rotates/masks (alg0/2/5/6/7) are wiring, only
  alg1/alg4 use the shared adder. The alg2/alg5 cross-byte carry is one routed
  bit — the thing that costs the GPU a barrier + carry table.

## Files & verification status

| File | Role | Verified |
|---|---|---|
| `tm_pkg.vhd` | types, `rng_step`, `apply_map`, **`apply_map_blend`**, `rng_vec`, `pick_alg` | via TBs below |
| `tm_rng_gen.vhd` | table-free per-map window generator, UNROLL_K | ✅ vs `run_rng` ref, K=1/8/16 |
| `tm_map_engine.vhd` | 16 inner steps over resident window, GROUP_UNROLL (1 lane) | ✅ G=1..16 transparency |
| `tm_map_engine_w.vhd` | **W-lane lockstep** engine (the GPU/CPU ILP pattern) | ✅ W lanes == W single-lane |
| `tm_screen_top.vhd` | expand → 27 maps → checksum, one-candidate | ✅ end-to-end smoke |
| `tm_screen_batch.vhd` | map-major batched (double-buffered gen) | ✅ 166/85 cyc/cand |
| `tm_screen_w.vhd` | W-lane top (one engine run/map) | ✅ (gen-bound — see Finding) |
| `tb_rng_step.vhd` | **exhaustive** 65,536-seed check vs real `rng_table` | ✅ PASS |
| `tb_rng_gen.vhd` | window vs reference `run_rng` stream | ✅ PASS |
| `tb_map_engine.vhd` | G=1 vs G=k bit-identical + cycle counts | ✅ PASS |
| `tb_apply_blend.vhd` | `apply_map_blend` == `apply_map`, all 8 algs × 64 vecs | ✅ PASS |
| `tb_map_engine_w.vhd` | W-lane == W single-lane (independence) | ✅ PASS |
| `tb_screen_top/batch/w.vhd` | integration / throughput demonstrators | ✅ PASS |
| `ref/dump_rng_table.cpp` | dumps real table + a `run_rng` stream (ground truth) | — |

### Datapath: op-pair blend tree (from the natmap blmerge)
`apply_map_blend` builds the 8-way dispatch as **4 op-pair candidates + a 2-level blend** — `{0,6}` one shifter, `{1,4}` one adder, `{3,7}` one xor, `{2,5}` the cross-byte rotate — instead of a flat 8:1 select. Shallower/smaller → shorter alg-step path → higher Fmax / more GROUP_UNROLL. Proven bit-identical to `apply_map`. The CPU's motivation (branch-misprediction) and the GPU's `__shfl_sync` for the alg-id source are **absent on FPGA** (flat state = wiring); the structure transfers, the cost doesn't.

### W-lane engine (CUDA ILP / CPU x8–x10 transliterated)
`tm_map_engine_w` runs W candidates lockstep through a map — shared window + nibble, per-lane state + offset + dispatch — exactly the CUDA `tm_cuda_raceway.cuh` ILP loop and the CPU `_x8`/`_x10` path, minus the cross-lane shuffle/barrier. It removes the per-candidate turnaround that capped `tm_screen_batch`.

### Fixed-key model (W4B) and the unrolled pipeline -- superseded
The operating space fixes the key across the whole ~4B data sweep (any new key
is a different context). So the 27 windows are generated **once per key** and
shared by every replicant over billions of candidates → **generation ≈ 0 per
candidate** (the earlier "gen-bound 448" was a demonstrator artifact of
regenerating per tiny batch). Crucially, fixed key makes each map's window a
**per-key operand**. Earlier notes treated that as enough to unlock a favorable
unrolled pipeline; P&R did not support that conclusion.

`tm_map_pipeline.vhd` lays the 27 maps as a pipeline (`N_STAGES` map_engines in
series), each stage holding its per-key-constant window/nibble. Candidates flow
through; the 27-deep latency is pipelined away. Verified: pipelined output ==
sequential stage-by-stage, and **II is a small constant independent of depth**:

| GROUP_UNROLL | II (cycles) | old optimistic throughput/pipeline @500 MHz, ×1 lane |
|---:|---:|---:|
| 4  | 6 | ~83 M/s |
| 16 | 3 | ~167 M/s |

(II ≈ 16/G + 2; the +2 is the engine's activate+restart dead cycles, tightenable
to 16/G.) These figures were useful as a functional/cadence sanity check, but
they are **not achievable planning numbers**. Routed Kintex-7 timing landed near
150 MHz at G=1 and below 100 MHz at G=2, while area scaled close to the number
of step bodies. A full pipeline still has to carry the 1024-bit state through
every semantic stage and provide per-stage window reads; it removes control
turnaround, not the actual work.

> **Correction (2026-07-07 review):** throughput-per-area is *not* ~equal to
> iterative once **register-window FF** dominates. G-deep combinational chaining
> forces each *live* window into FF/LUTRAM (BRAM's read latency can't feed the
> chain), so the full 27-stage pipeline holds all 27 windows resident
> (~442 Kb FF) while a map-major iterative engine keeps only ~2 (ping-pong). The
> *datapath* work is equal; the 27× window-FF replication is pure overhead that
> makes full unroll **worse** throughput/area on all but the largest parts
> (Agilex 5 / UltraScale+, where windows are noise). See the review section below.

> **Final correction (2026-07-08 P&R):** even after replacing the window FF/mux
> structure with internal LUTRAM, the step path is route-dominated and G>1 loses
> throughput/area. The unrolled pipeline is therefore a ceiling experiment only,
> not the recommended implementation.

✅ means GHDL analyze+elaborate+run PASS. The later `tb_map_parity` /
`tb_map_parity_ram` checks close the map-level value-parity gate against the
scalar `run_one_map` reference.

## Memory: why not NES-scant, and why ~2 KB is the floor

The NES ran the whole algorithm on a 6502 with scant RAM. We can too — the
*algorithm* needs only ~128 B state + a 16-bit seed + `rng_step`. What forces a
buffer is a **producer/consumer rate mismatch we create by parallelizing**:
`run_rng` is an irreducibly serial chain (1 byte/clk; K-deep unroll costs depth),
while the 128-wide datapath consumes 128 bytes/clk. The NES had no mismatch (a
slow serial consumer), so it needed no buffer; our wide parallel consumer must
read from one.

The buffer is small and bounded by **one map's window (~2 KB)**, not the
algorithm. The 59 KB "store all 27 windows" figure was a *candidate-major*
caching choice. Because each window is **key-global** (depends only on the map
seed, not the candidate), the throughput-optimal form is **map-major over a
batch of B candidates**: push the batch through map *m* (window resident, reused
B times), then map *m+1*. Footprint ≈ 2 KB window + B×128 B state (~10 KB at
B=64); regeneration amortizes over B.

`tm_screen_top` is the **one-candidate demonstrator**: it regenerates each
window for a single candidate, so it is deliberately RNG-bound — the honest
limited-memory baseline. The measured cost makes the case for batching concrete:

| config | busy cycles / candidate | breakdown |
|---|---:|---|
| UNROLL_K=8,  G=4  | 7185 | ~256 gen + 4 compute per map (×27) → **~98% RNG gen** |
| UNROLL_K=16, G=16 | 3648 | ~128 gen + 1 compute per map (×27) → **~95% RNG gen** |

Batching over B=64 amortizes the gen term: e.g. K=16/G=16 → ~128/64 + 1 ≈ 3
cyc/map → **~80 cyc/candidate** (compute-bound), vs 3648 regenerating per
candidate. That ~45× gap *is* the rate-mismatch lesson, measured.

## GROUP_UNROLL knob & the Fmax knee

`GROUP_UNROLL` chains G inner steps combinationally per clock (G | 16). The
`tb_map_engine` check proves G is **functionally transparent** (G=1..16 give
bit-identical output) and cuts cycles/map to 16/G. G=16 is "1 clock per map".
Device throughput ≈ (datapaths × Fmax)/work, so G helps only until logic depth
drops Fmax — **target the Fmax knee (likely G≈2–6), not G=16**. Finding that
knee is the P&R sweep.

## RNG representation: two structural reductions (both banked + parity-locked)

1. **One stream, not three (~1/3 footprint).** `alg0`/`alg6` values are just
   **bit 7 (the MSB)** of the regular `run_rng` byte (`alg0` at the reversed
   index, `alg6` at the forward/mirrored index — `rng_vec` vs `rng_vec_fwd`),
   and `alg2`/`alg5` carries are the same MSB at the offset. So the only RNG
   asset is the **regular stream**; alg0/2/5/6 are `bit7` wiring (1 bit/byte,
   not a byte). `tm_rng_gen` stores only that stream. (On CPU this was L1-
   resident already so it was left as 3 tables; on FPGA it cuts BRAM ~3× and
   makes those algs' RNG injection a single-bit fan-out.)

2. **Offset = `(f,c)` counters, not an 11-bit adder.** Every touched offset in a
   map is `128·f + c` with `f` = #byte-algs, `c` = #carry-algs(alg2/5),
   `f+c ≤ 15`. Since `c ≤ 15 < 128`, `off = {f[3:0], c[6:0]}` is a **pure
   concatenation (no adder)**, `off>>7=f` / `off&127=c` decode free, and the
   touched set is a bounded triangular **≤136-row** superset — so the windowed
   read is a **16:1 row-select (f) + ≤15-byte align (c)**, not a 2048-wide
   barrel. `tm_map_engine` tracks `(foff,coff)` as two small counters.
   (From the late-branchless RNG discovery; on CPU it was a sub-L1 non-win,
   here it cheapens the per-step RNG read + offset logic.) Both reductions are
   verified bit-exact end-to-end by `tb_map_parity` + `tb_forward_parity`.

## Finding: two orthogonal axes — the gap AND gen amortization

The throughput demonstrators isolated two independent bottlenecks:

1. **Per-candidate turnaround (the gap).** `tm_screen_batch` ran one `map_engine`
   per candidate with a start/done restart → ~35–68% overhead (85 cyc/cand at
   G=16 vs 27 ideal). **Fixed by the W-lane engine** (lockstep, no restart).
2. **Generation amortization.** `tm_screen_w` (W=8, regenerate window per map)
   is **gen-bound**: 8 lanes do 4 compute cyc/map but the window regen is
   128 cyc/map (K=16) → 448 cyc/cand. Raising only the gen rate K confirms it:

   | K | 16 | 64 | 256 | 2048 |
   |---|---:|---:|---:|---:|
   | cyc/cand | 448 | 124 | 43 | 29 |

   So **the W-lane engine removes the gap but does nothing for gen amortization**
   — a separate axis. Reaching the compute bound (~13.5 cyc/cand at W=8/G=4)
   requires amortizing the per-key window generation, two ways:
   - **Store all 27 windows** (~59 KB, shared), generate once per key → gen ≈ 0
     per candidate over the (billions-deep) data range; candidate-major. **Most
     memory-efficient for a flat screener** — this *corrects* the earlier
     "regenerate ~2 KB beats 59 KB" claim: for throughput you must amortize gen,
     and batch-regen needs B≫256 resident states (≫59 KB) to do so.
   - **Or make gen cheap** (high K + a faster generation clock domain, the
     producer/consumer rate-match): then small-batch regen at ~2 KB becomes
     viable (the K-sweep above is exactly this lever in sim).

   Net: gen amortization is a real top-level decision (store vs fast-gen), and
   the residual ~29 cyc/cand at K=2048 is the per-map FSM turnaround. This was
   the next thing to pipeline before P&R showed the raw lane is not competitive.

## Design review adjustments (2026-07-07)

A datapath/area review produced the following. **Applied** items are in the RTL
and re-verified bit-exact (GHDL 4.1.0: full 8/8 suite PASS, plus `tb_map_engine`
G ∈ {1,2,8,16} and `tb_map_pipeline` N=6, GU ∈ {2,8,16}). The architectural
items listed afterward are retained for context, but are superseded by the
2026-07-08 routed closeout.

### Applied (bit-identical, verified)

1. **One shared RNG row-extraction (`rng_rows`).** The per-step read was three
   separate calls (`rng_vec` + `rng_vec_fwd` + `rng_carry_bit`), each
   re-deriving the same rows f/f+1 — i.e. it relied on the synthesizer to CSE a
   16:1 row-select across the `win_byte` boundary. Now the engines extract the
   two-row slice **once** per step (per lane) into `tworow_t` and the reversed /
   forward / carry reads all index that slice (`rng_vec_r`, `rng_vec_fwd_r`,
   `rr(c)(7)`). The "each lane does exactly ONE window read" claim is now
   *structural*, not aspirational. `rng_vec`/`rng_vec_fwd`/`rng_carry_bit` are
   kept as compat wrappers composed from the shared path (one place for the
   logic). This is the biggest LUT/Fmax lever and it multiplies by W×G.
2. **Power-of-two masked window read.** `win_byte` now returns `w(idx mod WIN_LEN)`
   (WIN_LEN = 2¹¹, so a free bit-mask) instead of a `< 0 / > WIN_LEN-1` clamp —
   dropping a magnitude comparator from every one of the 256 row selects. The
   `f+c ≤ 15` map invariant bounds every *reached* index to ≤ 2047, so the mask
   is bit-identical to the clamp on all reachable reads (confirmed by the parity
   TBs on real seed/nibble/state vectors).
3. **alg4 folded into the shared adder carry-in.** In `apply_map_blend` the
   `{1,4}` op-pair was `x + (~r + 1)` — a second 8-bit increment. It is now one
   adder with `cin = is4` (`x + (is4?~r:r) + cin`), halving the alg1/alg4 add
   cost (128 × G of them). `tb_apply_blend` re-confirms all 8 algs × 64 vecs.
4. **`USE_BLEND` generic → the pipeline uses the shallow blend datapath.** The
   throughput vehicle (`tm_map_pipeline`) previously instantiated `tm_map_engine`
   with the **flat 8:1** `apply_map`, contradicting the whole "shallower path →
   higher Fmax knee" argument. `tm_map_engine` now takes `USE_BLEND` (default
   `false` = `apply_map`, the golden reference; single-lane parity gate
   unchanged) and the pipeline sets it `true` by default. Bit-identity is
   transitive: `blend == apply_map` (`tb_apply_blend`) and `apply_map == tm_8`
   (`tb_map_parity`), and `tb_map_pipeline` now runs the blend path end-to-end.

### Superseded architectural ideas

A. **Free-running pipeline stage (drop II from ~16/G+2 to 16/G).** Each stage
   still carries the engine's `start/done` FSM turnaround. Replace it with a
   flow-through stage (mod-16/G counter, state registered every 16/G, valid
   propagated — no handshake). This remains a real local improvement, but after
   routed timing it is only a modest multiplier on a lane that is already far
   below GPU throughput.
B. **Generate all 27 windows once per key; map-major ping-pong.** Fixed key ⇒
   windows are per-key constants: prefill all 27 into a BRAM-backed store at
   key-load (steady-state gen ≈ 0 over billions of candidates), and DMA the
   current map's window into a **single** register-window (ping-pong with the
   next map) so only ~2 live windows exist instead of 27. Resolves the gen-bound
   `tm_screen_w`/`tm_screen_batch` tops and cuts window-FF ~13×, but not the
   full-state step cost.
C. **W-lane shared-window read-port banking (W ≳ 8).** All W lanes read the
   shared window combinationally at their own (f,c); past W≈8 the shared-net
   fan-out caps Fmax. Bank/replicate the register-window per lane-group and/or
   retime the row-select into a pipeline register — trade FF for Fmax. This is
   a scaling study only if FPGA work is reopened for a power-constrained target.

### Unrolled vs. iterative — final verdict

The post-route result rejects both the old full-unroll optimism and the idea
that small iterative cleanup can close the gap. Full unroll buys zero control
turnaround, but it still replicates hundreds of full-state transform stages and
window read sites. Iteration keeps area bounded, but per-lane throughput is only
~0.366 M candidates/s. A W-widened, map-major partial unroll remains the best
shape **if** a niche FPGA deployment must be built, but it is not a production
throughput strategy against the current GPU/CPU paths.

## First synthesis estimates (2026-07-08)

Real OOC numbers, resource-only (no place+route yet, so **no Fmax**). Recipe:
Vivado 2026.1 `synth_design -mode out_of_context` + `report_utilization` — the
quickest reliable estimate; OOC makes the wide top ports internal so there is no
I/O-fit failure and no virtual-pin bookkeeping. (Quartus Lite 25.1 A&S on
Cyclone V is the fallback but slower, and its "ALMs needed" is a pessimistic
pre-fit figure.) **Device note:** the shipped `syn/*.tcl` target parts/editions
not necessarily installed — `xcku5p`/Agilex 5 need UltraScale+ device support /
Quartus Pro; **`xc7k160t` (Kintex-7) and Cyclone V** are the free/installed
proxies used here.

**`tm_map_engine`, Kintex-7 `xc7k160tffg676-2`, post-synth (Vivado):**

| GROUP_UNROLL | LUT (logic) | % dev | FF | F7+F8 muxes | CARRY4 | cyc/map |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16,723 | 16.5% | 1,111 | 4,504 | 512 | 16 |
| 2 | 32,048 | 31.6% | 1,100 | 7,473 | 1,024 | 8 |
| 4 | 63,899 | 63.0% | 1,118 | 15,233 | 2,048 | 4 |

What it says: **LUT/CARRY4/mux scale ~linearly with G** (G is combinational
copies of the step; CARRY4 = 128 byte-adders × G exactly), while **FF is flat at
~1.1 K** = just the 128-byte state (confirms the window is an external port).
**0 DSP, 0 BRAM.** One G=4 engine already fills ~63% of a mid Kintex-7, so area
is tighter than the "mostly wiring" framing — the ~4.5 K→15 K F7/F8 muxes are the
window read, and they are **inherent to the runtime two-16:1 structure** (16:1
row-select by f × 16:1 align by c), not an RTL artifact. Throughput/area for the
iterative engine is thus ~G-invariant on area; G's payoff is entirely on the
(unmeasured) Fmax side.

### Falsified: the structural row-select prototype
Hypothesis: expressing the read as an explicit `[16][128]` `win_row` 16:1 select
(vs the flat `win(128*f + j)` index) would cut the window-read muxes. **Measured
and rejected (bit-identical via the parity gate, then re-synthesized):**
netlist-**identical** at G=1, **~6 % LUT worse** at G=2 (34,051 vs 32,048) and
+2.2 K muxes at G=4. Vivado already infers the 16:1 from the flat index, so the
explicit form only removes optimizer freedom. The flat form is kept. (It *might*
still help a 4-LUT family — PolarFire/Spartan — which wasn't available to test;
revisit only if a Libero/ISE flow is on hand.) **The real area lever is not the
RTL form but fixed-key constant-window specialization** (fold the selects when
the window is a per-key constant), which is an architectural change, not a
rewrite. The follow-up P&R below provided the real Fmax and rejected deeper G.

### Confirmed: window as a runtime-writable distributed-RAM table
The fixed-key window benefit is realizable **without any bitstream bake** — as
RAM *contents* written by a per-key preprocessing load, structure fixed in the
bitstream. Prototype `tm_win_read.vhd` isolates the window storage + `(f,c)` read
+ align in two styles (parity-locked to `tm_pkg` for all valid `(f,c)` via
`tb_win_read`), synthesized on Kintex-7:

| window style | LUT logic | LUT mem | FF | F7+F8 muxes |
|---|---:|---:|---:|---:|
| forced registers (`ram_style="registers"`) — today's FF storage | 9,556 | 0 | **16,384** | 3,432 |
| **distributed RAM** (`ram_style="distributed"`, async read) | 2,167 | 764 | **0** | **0** |

The RAM form **eliminates all 16,384 window FF** (→ 764 LUTRAM cells), cuts logic
LUTs ~77 %, and removes the read-mux trees (the async RAM read replaces them).
Because it is a real RAM with a write port, the per-key table is loaded at
key-time and the fabric structure is shared by every replication — the
"preprocessing fixes a table, not a bitstream" model. For a pure-data operand
like the window this captures essentially the *entire* fixed-key benefit; true
compile-time folding buys ~nothing more (the `(f,c)` address is runtime
regardless), so no partial-reconfig / LUT-INIT patching is warranted. Notes:
- **Async-read distributed RAM keeps the read combinational**, so the `G`-unroll
  chain survives on Xilinx SLICEM. (Intel MLAB may need a registered read →
  pipeline it.) `G>1` and `W` lanes issue multiple reads/cycle → replicate the
  RAM per concurrent read (the read-port limit), but LUTRAM replication at
  ~764/read beats FF storage + mux by a wide margin.
- **Biggest payoff is the 27-stage pipeline:** 27 × 16,384 = ~442 K window FF
  today → **0 FF** in LUTRAM (~20 K mem-LUT) or a handful of BRAM36. This is the
  single largest area item from the review, and it dissolves.
- Completed follow-up: `tm_map_engine_ram` integrates this RAM window and was
  synthesized/routed below. The read-path saving did not change the throughput
  conclusion.

### Integrated RAM engine + measured Fmax + rate refresh (2026-07-08)
`tm_map_engine_ram` = `tm_map_engine` with the window in the internal
distributed RAM (loaded via `wr_*`), parity-locked to `tm_8` (`tb_map_parity_ram`,
128/128 vectors). Synth + **place-and-route** on Kintex-7:

| engine | G | LUT | LUTRAM | FF | CARRY4 | post-route |
|---|---:|---:|---:|---:|---:|---|
| free-window baseline, flat | 1 | 17,130 | 0 | 1,182 | 512 | **Fmax 161 MHz** |
| RAM window, flat | 1 | 25,595 | 850 | 1,092 | 512 | **Fmax 153 MHz** |
| RAM window, blend | 1 | 23,084 | 850 | 1,072 | 256 | **Fmax 158 MHz** |
| RAM window, blend | 2 | 46,180 | 1,700 | 1,124 | 512 | **Fmax 96 MHz** |

**The RAM window is a FF↔LUT trade, not a free win.** It removes ~16 K window FF
per window but adds ~6 K logic LUT + LUTRAM for the read. So:
- **Single iterative/batch engine** (1–2 resident windows): FF was never the
  bottleneck, and RAM costs *more* LUT → keep the window in FF there.
- **Full 27-stage pipeline / high-W** (27 windows = ~442 K FF, *infeasible* on a
  mid part): RAM/LUTRAM is the enabler. If a ceiling pipeline is ever reopened,
  prefer **BRAM** for the windows — the design uses 0 BRAM otherwise, and
  27×16 Kb ≈ 13 BRAM36 with a 1-cycle read the pipeline stages already absorb.

**Measured Fmax corrects earlier optimism, and picks the G knee.** The window
read (LUTRAM + c-align) sits in the critical path, so one inner step is ~6.3 ns →
**158 MHz at G=1**, and G=2 falls to **96 MHz**. The old "~500 MHz /
~167 M-per-pipeline at G=16" figures predated any real timing and were ~7–10×
optimistic. `Fmax·G` rises with G (151→196) but **unit area rises faster**
(23.1 K→46.2 K LUT), so **throughput/area is highest at G=1** (15.9 vs 8.9 for
G=2). **G=1 is the operating point** — deeper G loses on this design.

**Refreshed achievable rate** — total ≈ (usable LUT / LUT-per-lane) · Fmax · G /
432, with 432 = 27 maps × 16/G inner steps (pipelined ideal; the FSM engine is
~12 % under this from per-map turnaround). At the measured G=1 point
(0.366 M cand/s per 23.1 K-LUT lane, ~75 % LUT util):

| device (LUT6) | engines | M cand/s (G=1) |
|---|---:|---:|
| Kintex-7 xc7k160t (101 K) | 3 | 1.1 |
| Kintex-7 xc7k410t (254 K) | 8 | 2.9 |
| Kintex US+ xcku15p (523 K) | 16 | 5.9 |
| Virtex US+ xcvu13p (1.73 M) | 56 | 20.5 |
| Versal ~(2.0 M) | 65 | 23.8 |

**Verdict vs GPU.** Diffuse-segment GPU bar is ~92 M/s (RTX 5090) to ~252 M/s
(CUDA HM). So this design is **~4–12× below the GPU on raw candidate-rate** even
on the largest FPGA — Fmax (~158 MHz) and per-lane area (~23 K LUT) cap the
parallelism. Perf/watt is the only possible competitive regime, and it is too
deployment-specific to justify more RTL work without a real board/power target.
Open levers (all modest): the **free-running pipeline** (+~12 %) and **BRAM
windows** to free LUTs for more lanes; the G-knee is already spent (G=1
optimal). Net: FPGA is **not** a throughput-ceiling path for this workload.

## Archive checklist

No ordinary FPGA implementation work is recommended after the 2026-07-08 P&R
closeout. Keep these only as reference/cleanup tasks:

1. **Preserve value parity.** `tb_map_parity` and `tb_map_parity_ram` check
   `tm_map_engine` / `tm_map_engine_ram` against scalar `run_one_map` vectors;
   keep those tests with the RTL as the correctness anchor.
2. **Do not build the old batched/top-level roadmap for throughput.** A
   map-major sequencer, key-schedule block, other-world comparator, and
   expansion-vector generator would make the RTL more complete, but not change
   the throughput conclusion.
3. **Only reopen FPGA work for a new dependency-collapse proof.** The relevant
   proof would show that the final predicate does not require carrying most of
   the 128-byte state through most maps. Without that, full unroll and hybrid
   acceleration remain area-linear variants of the measured lane.

## Build / simulate (GHDL 5.0.1, VHDL-2008)

```
# reference vectors (ground truth from the real repo RNG)
cd ref && g++ -O2 -I../../common dump_rng_table.cpp ../../common/rng.cpp -o dump_rng_table && ./dump_rng_table && cd ..

# exhaustive rng_step (table elimination)
ghdl -a --std=08 tm_pkg.vhd tb_rng_step.vhd && ghdl -e --std=08 tb_rng_step && ghdl -r --std=08 tb_rng_step

# generator vs reference stream
ghdl -a --std=08 tm_rng_gen.vhd tb_rng_gen.vhd && ghdl -e --std=08 tb_rng_gen && ghdl -r --std=08 tb_rng_gen --stop-time=70us

# GROUP_UNROLL transparency (try -gGCHK=1/2/4/8/16)
ghdl -a --std=08 tm_map_engine.vhd tb_map_engine.vhd && ghdl -e --std=08 tb_map_engine && ghdl -r --std=08 tb_map_engine -gGCHK=4 --stop-time=2us

# end-to-end pipeline
ghdl -a --std=08 tm_screen_top.vhd tb_screen_top.vhd && ghdl -e --std=08 tb_screen_top && ghdl -r --std=08 tb_screen_top --stop-time=300us
```
