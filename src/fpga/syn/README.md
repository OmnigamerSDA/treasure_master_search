# Synthesis / P&R prep — flows, device fit, and the sweep

The RTL is **vendor-neutral VHDL-2008** (no primitives, no IP), so the same
sources synthesize on every target; only the constraint + flow scripts differ.
All design files are listed in `sources.txt` (testbenches and `ref/` dumpers are
sim-only and excluded). Functional + ground-truth parity is already locked in
sim (`../README.md`); this directory is for getting **real LUT/FF/carry/Fmax**.

## What the logic actually demands (drives device choice)

Per active map-step the datapath is:
- **128 × 8-bit adders** (alg1/alg4 — the *only* arithmetic) → carry-chain bound.
- **4:1 op-pair blend** per byte + **16:1 RNG row-select** per byte (the `(f,c)`
  read) → **mux/LUT-bound**; wider LUTs pack these in fewer levels.
- **1024-bit state registers** × lanes × pipeline-stages → **FF-bound** at scale.
- `rng_gen`: a few `run_rng` adders × `UNROLL_K` → tiny.
- Memory: one ~2 KB window/engine (distributed RAM or registers); **no DSP, no
  multiply, minimal BRAM**.

So the figure of merit is **LUT-input size (mux packing) + carry speed + FF
density + Fmax** — *not* DSP/BRAM. A DSP-rich part wins nothing here.

## Device recommendations (your spectrum)

| Family | Tool | Fit for this design | Use it for |
|---|---|---|---|
| **Agilex 5** (7 nm, Intel) | Quartus Pro | **Best fit.** 8-input fracturable ALMs pack the 4:1/16:1 muxes + blend in the fewest levels; fast carry; high FF density + Fmax. | **Throughput ceiling** (max N×Fmax) |
| **UltraScale+** (16 nm, Xilinx) | Vivado | **Co-best.** 6-LUT + CARRY8, URAM for shared windows, most mature flow; very high density/Fmax. | **Throughput ceiling**, shared-window store |
| **PolarFire** (28 nm, Microchip) | Libero | Mid density, **flash → low static power**; 4-input LUTs ⇒ wide muxes cost more levels (this is where the `(f,c)` row-select restructure helps most). | **Perf/watt**, deployable many-board |
| **Lattice Avant-E** (16 nm) | Radiant | Newer low-power mid-range; decent Fmax for the class. | Perf/watt, low-power mid |
| **Kintex-7** (28 nm) | Vivado | Solid mid 7-series, CARRY4, ~300-400 MHz. | Low-cost mid proof point |
| **Spartan-7** (28 nm) | Vivado | Low-cost modern; small ⇒ few cores. | Low-cost / edge proof |
| **Spartan-6** (45 nm) | **ISE only** | Legacy toolchain, slow, small, partial VHDL-2008. **Avoid unless a fixed board mandates it.** | only if forced |

**Recommendation for "initial numbers":** synth on **Agilex 5 + UltraScale+**
first — they set the throughput ceiling (whether FPGA rivals the ~420 M/s CUDA /
~250 M/s diffuse bar). Add **PolarFire** for the perf/watt story (the real lever
vs a 300-450 W GPU). Kintex-7/Spartan-7 give the low-cost data point. Skip
Spartan-6 unless required (and expect VHDL-93 edits + ISE).

Two design notes that matter across families:
- **No DSP demand** → don't pay for DSP-heavy parts; balance LUT/FF.
- The mux-heavy read is why the `(f,c)` **row-select (16:1) beats the 2048-wide
  barrel** — and it helps *most* on the 4-LUT families (PolarFire/Spartan),
  where wide muxes are expensive. That restructure is already in the RTL.

## Flows

**Vivado** (UltraScale+, Kintex-7, Spartan-7):
```
vivado -mode batch -source vivado_ooc.tcl -tclargs tm_map_engine xcku5p-ffvb676-2-e 2.0 "GROUP_UNROLL=8"
```
Full OOC synth+place+route; prints WNS/Fmax, writes `*_util.rpt` / `*_timing.rpt`.

**Quartus Pro** (Agilex 5):
```
quartus_sh -t quartus_ooc.tcl tm_map_engine A5ED065BB32AE6SR0 2.0 "GROUP_UNROLL=8"
```
map+fit+sta; read `*_q.fit.rpt` (ALM/reg) and `*_q.sta.rpt` (Fmax).

**Libero (PolarFire)** / **Radiant (Lattice Avant)**: import `sources.txt` as
VHDL-2008, add `constraints/clk.sdc`, set top + generics, run synth→P&R→timing.
(Both use Synplify Pro; no batch tcl shipped here because their project APIs
differ — the RTL + SDC are the portable parts.)

**ISE (Spartan-6, legacy):** create an XST project from `sources.txt`; ISE's
VHDL-2008 support is partial — expect to relax a few 2008-isms (e.g. unconstrained
array ports) to VHDL-93. Lowest priority.

## The sweep (what produces the verdict)

For each device, OOC each unit and sweep the knobs:
- `tm_map_engine`  — `GROUP_UNROLL ∈ {1,2,4,8,16}` → per-core LUT/FF + Fmax(G); find the **Fmax knee**.
- `tm_rng_gen`     — `UNROLL_K ∈ {1,2,4,8,16}` → generator area/Fmax (bytes/clk).
- `tm_map_engine_w`— `LANES ∈ {2,4,8,16}` × G → parallel-lane area/Fmax; watch the window read-ports.
- `tm_map_pipeline`— `N_STAGES=27`, sweep G → the unrolled forward (throughput = Fmax/II).

Then the feasibility number per device:
```
throughput ≈ (instances that fit) × Fmax / cycles_per_candidate
```
and compare to the diffuse-segment GPU bar (CUDA ~252 M/s, OpenCL ~158 M/s;
`docs/raceway_precert_hm_20260618.md`). Report **Fmax + LUT/FF/carry per unit**
and the utilization at which Fmax starts dropping — that knee is the answer.
