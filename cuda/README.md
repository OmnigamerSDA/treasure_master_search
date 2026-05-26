# Treasure Master — CUDA Forward Search

A CUDA implementation of the forward decryption/state-check loop for the
1991 NES game *Treasure Master*. Given a 4-byte key, the kernel sweeps the
2^32 starting-data space, running the game's custom 27-step algorithm
forward and checking which results match either the carnival-world or
other-world target state.

This directory is the CUDA implementation in the consolidated Treasure
Master forward-search release. It is self-contained enough to publish as a
standalone CUDA package, but the preferred public layout also includes the
CPU and OpenCL implementations beside it.

## What the code does

For each candidate `(key, data)`:

1. Run the **key schedule** to derive 27 per-step RNG seeds and algorithm
   selectors.
2. Run the **forward algorithm**: 27 sequential steps that mix the 32-byte
   working state with RNG-derived values. Each step is one of 8 operations
   (alg 0–7) determined by the schedule.
3. Compute the **checksum screen**: a fast filter checking whether the
   final state could match either world's checksum.
4. For checksum survivors, the host emits the candidate for full opcode
   validation.

The 2^32 key space is searched by fixing a key and sweeping the data axis
(or vice versa). The kernel is the data-sweep variant: one fixed key per
launch, batched data values across threads.

## Performance

Two screen-kernel paths are available:

* **Baseline** (default): universal RNG tables, ILP4. Stable, broadly portable.
* **Offset-stream + ILP** (`--screen-offsets [--ilp 4|6|8]`): per-key
  precomputed offset streams, configurable ILP. **~1.4× faster on
  sm_120 Blackwell.** Costs ~22 MB of extra device memory per key.

Measured on a clean GPU (no contention). 2^24-candidate workunit, fixed
key `0x2CA5B42D`:

| GPU                                       | Path                          | Screen rate    | Wall rate       |
|-------------------------------------------|-------------------------------|----------------|-----------------|
| NVIDIA RTX 5090                            | baseline                      | 102.6 M cand/s | 82.9 M cand/s  |
| NVIDIA RTX 5090                            | `--screen-offsets --ilp 6`    | **141.2 M cand/s** | **108.2 M cand/s** |
| NVIDIA RTX 5090                            | `--screen-offsets --ilp 4`    | 139.5 M cand/s | 105.2 M cand/s |
| NVIDIA RTX 5090                            | `--screen-offsets --ilp 8`    | 136.6 M cand/s | 107.1 M cand/s |
| NVIDIA RTX PRO 6000 Blackwell Max-Q WS Ed. | baseline                      | 74.8 M cand/s  | 72.8 M cand/s  |

Survivor counts are byte-identical across all four configurations
(deterministic kernel). ILP6 is the empirical sweet spot on Blackwell;
ILP4 and ILP8 are exposed for tuning on other GPU families where the
optimum may shift (smaller register file → prefer ILP4; larger →
ILP8 may win).

At the measured screen-kernel rates, a full `2^32` single-key sweep on
RTX 5090 is about **30-33 s** with `--screen-offsets --ilp 6`, or about
**42 s** with the baseline path. The same short 2^24 benchmark reports
wall-rate projections of about **40 s** and **52 s** respectively because
startup, precompute, launch, copy, and reporting overhead are a larger
fraction of a short run.

Survivor count for `0x2CA5B42D` across the full 2^32:

- 152,049 checksum survivors total
- 92,538 carnival
- 59,511 other-world

### Kernel design (the wins that mattered)

**Baseline kernel** (`tm_checksum_screen_cuda`):

1. Removed block-wide synchronization from the hot schedule loop.
2. Rewrote `alg2` and `alg5` to use warp shuffles instead of shared memory.
3. Moved from 1 warp/block to **4 warps/block** (128 threads).
4. Parallelized checksum reduction across the warp.
5. Packed other-world and checksum-mask data into `uint32_t` `__constant__`
   tables.
6. Interleaved 4 candidates per warp inside the screen kernel (**ILP4**).
   2× helped; 4× helped more; 8× regressed.

**Offset-stream + ILP kernel** (`tm_checksum_screen_offset_store_ilp{4,6,8}_cuda`,
added May 2026):

7. **Per-key offset-stream RNG**: host precomputes `(regular / alg0 / alg6)`
   byte streams and `(alg2 / alg5)` carry uint32 streams indexed by
   `(schedule_step, rng_offset_within_step)`. The kernel does one indexed
   read per alg call instead of walking `rng_seed` via the per-step
   `rng_forward_1 / rng_forward_128` tables — fewer instructions per alg
   and less dependency between successive steps.
8. **PreIDs**: precompute all ILP candidates' `algorithm_id` from a single
   `__shfl_sync` before dispatching the alg-apply, letting the compiler
   hoist the LDGs as a block.
9. **Packed flag store**: when ILP is divisible by 4, lane 0 packs the
   flags into a `uint32_t` and writes once instead of 4 byte stores.
10. ILP6 → I-cache sweet spot: smaller inner-loop code than ILP8 (the
    `no_instruction` warp-issue stall drops ~34 %), avoids the
    register-pressure hit of ILP12+.

What didn't help (verified with Nsight Compute):

- `__restrict__` on hot-path pointers.
- `__constant__` for the schedule blob.
- `__ldg()` qualifier (SASS-identical on sm_120 unified L1/RO cache).
- `__constant__` for the alg-0/6 RNG tables (32 unique addresses/warp
  serialize 32× through constant cache).
- Manual software prefetch (NVCC was already overlapping LDGs across
  ILP candidates; restructuring the loop into prefetch/apply phases
  inflates the body too much for the unroll heuristic).
- Forcing `#pragma unroll` on the schedule loop (5× slowdown from
  I-cache thrash — NVCC's default heuristic is correct here; do NOT
  add the pragma).
- Small loop-unroll tweaks beyond the above.

The remaining bottleneck is the dependent table-load chain in
`run_alg_offset()` — see the inline notes in `src/tm_cuda.cu`.

## Build

### Prerequisites

- **CUDA Toolkit 12.x or 13.x**. Install from
  https://developer.nvidia.com/cuda-downloads or your distro's package
  manager. The default Makefile looks in `/usr/local/cuda` — set
  `CUDA_PATH` if yours is elsewhere.
- **A C++17-capable host compiler** (g++ 9+, clang 10+).
- **An NVIDIA GPU** with compute capability matching the `GENCODE`
  setting (default `sm_120` covers Blackwell).

### Build commands

Default (targets sm_120 — RTX 50-series, RTX PRO 6000 Blackwell, B100):
```sh
make
```

For other GPU families, override `GENCODE`:
```sh
# Ada Lovelace (RTX 4090, RTX 4080, L40, RTX 5000 Ada, etc.)
make GENCODE='-gencode arch=compute_89,code=sm_89'

# Ampere (A100, A40, RTX 3090, RTX 30-series)
make GENCODE='-gencode arch=compute_80,code=sm_80'

# Turing (T4, RTX 2080 Ti, Quadro RTX)
make GENCODE='-gencode arch=compute_75,code=sm_75'
```

Use a non-standard CUDA install:
```sh
make CUDA_PATH=/opt/cuda-13.2
```

Output:
- `tm_cuda` — the host binary.
- `tm_cuda.fatbin` — the kernel (loaded at runtime by the host).

### Tuning for other GPUs

Two launch-shape constants were hand-tuned on Blackwell hardware. On
smaller / older GPUs the optimal point is often different (lower
register file per SM, less occupancy headroom). Both are exposed as
preprocessor macros:

| Macro                      | Default | What it controls                          |
|----------------------------|---------|-------------------------------------------|
| `TM_WARPS_PER_BLOCK`       | `4`     | Warps per block (128 threads at default)  |
| `TM_CANDIDATES_PER_WARP`   | `4`     | ILP factor inside the screen kernel       |

Override at build time. The full `CXXFLAGS` value must be repeated
because `make` overrides — not appends — when set on the command line:

```sh
# Conservative — try on older Turing / smaller Ampere
make CXXFLAGS='-std=c++17 -O2 -m64 -DTM_WARPS_PER_BLOCK=2 -DTM_CANDIDATES_PER_WARP=2 -I$(CUDA_PATH)/include -I./src'

# Or just ILP1 if 2 still regresses
make CXXFLAGS='-std=c++17 -O2 -m64 -DTM_CANDIDATES_PER_WARP=1 -I$(CUDA_PATH)/include -I./src'
```

Empirical findings from the original tuning pass:
- Going from 1 warp/block to 4 warps/block was a major win on Blackwell.
- ILP1 → ILP2 was a major win.
- ILP2 → ILP4 was a smaller but real win.
- ILP4 → ILP8 *regressed* throughput (register pressure exceeds occupancy gain).

Smaller GPUs hit that regression earlier. If your throughput numbers are
significantly below what you'd expect from your card's memory bandwidth,
try lowering ILP first, then warps-per-block.

### Other arch features

The kernel uses only warp-shuffle intrinsics (`__shfl_sync`,
`__shfl_down_sync`, `__shfl_xor_sync`) and `__constant__` memory — all
present since sm_30 (Kepler, 2012). No Tensor Cores, no Hopper/Blackwell
features (TMA, thread block clusters, `wgmma`), no inline PTX, no
`__CUDA_ARCH__` guards. The kernel source is broadly portable; only the
default `GENCODE` is Blackwell-specific.

## Run

### 1. Identify your GPU's device index

```sh
nvidia-smi -L
# Example output:
# GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-...)
# GPU 1: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (UUID: GPU-...)
```

Take the `GPU N` number for the device you want and pass it to `--device`.
The host prints the resolved device name in its startup banner so you can
double-check. For a single-GPU system, use `--device 0`.

### 2. Parity check (recommended first run)

Verify CUDA matches the CPU reference for 64 candidates:
```sh
./tm_cuda --device 0 --parity 64 --batch_size 64 --workunit_size 64 --warmup_batches 0
```
Expect: `PASS — all 64 candidates match CPU (baseline kernel)`.

To also verify the offset-store path agrees with the baseline:
```sh
./tm_cuda --device 0 --parity 4096 --batch_size 4096 --workunit_size 4096 \
            --warmup_batches 0 --screen-offsets --ilp 6
```
Expect both lines: `PASS — all 4096 candidates match CPU (baseline kernel)`
and `PASS — all 4096 flag bytes match`.

### 3. Short throughput benchmark (~150 ms on a Blackwell GPU)

Baseline kernel:
```sh
./tm_cuda --device 0 --key_id 0x2CA5B42D --range_start 0 \
            --workunit_size 16777216 --batch_size 1048576 --warmup_batches 1
```

Fast kernel (offset-stream + ILP6, ~1.4× faster):
```sh
./tm_cuda --device 0 --key_id 0x2CA5B42D --range_start 0 \
            --workunit_size 16777216 --batch_size 1048576 --warmup_batches 1 \
            --screen-offsets --ilp 6
```

### 4. Full 2^32 sweep for one fixed key (~30-33 s screen time on RTX 5090)

```sh
./tm_cuda --device 0 --key_id 0x2CA5B42D --range_start 0 \
            --workunit_size 4294967296 --batch_size 1048576 --warmup_batches 1 \
            --screen-offsets --ilp 6
```

The `--screen-offsets` path uses ~22 MB of extra GPU memory per key (the
precomputed offset streams). It must be re-uploaded if the key changes;
for single-key workunits this is amortized once at startup.

### Note on device numbering

The `--device` index passed to the host is the CUDA enumeration index
and **may not match `nvidia-smi` ordering** under some driver / topology
configurations. Always check the device name printed in the startup
banner before drawing conclusions from a benchmark run.

**Range bounds**: `--range_start` and `--workunit_size` accept 64-bit
values on the host side, but the kernel is 32-bit candidate-indexed.
`--range_start + --workunit_size` must stay within 2^32.

## Layout

```
src/
  main.cpp           CUDA host: argument parsing, kernel launch, validation
  tm_cuda.cu         CUDA kernel: schedule + alg loop + checksum screen
  key_schedule.{cpp,h}  Forward key-schedule generator
  rng_obj.{cpp,h}    PRNG class — produces all RNG tables fed to the kernel
  alignment2.{cpp,h} Memory-alignment helpers
  data_sizes.h       Compile-time size constants
Makefile             Build driver (configurable CUDA_PATH and GENCODE)
README.md            This file
```

The host computes all RNG tables once on the CPU (via `rng_obj`) and
uploads them to device memory before the sweep starts. The kernel itself
just consumes those tables.

## Profiling

Nsight Systems (replace `0` with your device index):
```sh
nsys profile --trace=cuda,osrt --sample=none -o nsys_run \
    ./tm_cuda --device 0 --key_id 0x2CA5B42D --range_start 0 \
                --workunit_size 16777216 --batch_size 1048576 --warmup_batches 1
```

Nsight Compute (source-correlated, one kernel launch):
```sh
ncu --set basic --target-processes all \
    --kernel-name regex:tm_checksum_screen_cuda \
    --launch-skip 1 --launch-count 1 \
    ./tm_cuda --device 0 --key_id 0x2CA5B42D --range_start 0 \
                --workunit_size 1048576 --batch_size 1048576 --warmup_batches 1
```

NVIDIA's profiling counters must be enabled for the running user. On
Linux: `NVreg_RestrictProfilingToAdminUsers=0` in
`/proc/driver/nvidia/params` (reboot required if it doesn't take effect).

## Algorithm reference

A 6502 disassembly of the unlock path and a step-by-step walkthrough of
the algorithm are published in the consolidated repository's `wiki/` and
`docs/` directories. This directory focuses on the CUDA implementation;
the algorithm itself is also documented in the host `key_schedule.cpp`
and `tm_cuda.cu` sources.

## License

MIT — see [LICENSE](LICENSE) for the full text.

## Background

*Treasure Master* shipped with a hidden "Prize World" reachable only by
entering a 24-character password released on MTV in April 1992. Years
later, reverse-engineering revealed a **second** Prize World ("Bonus
World 2") that was never publicly unlocked — its password requires a
key search over 2^32 candidates plus opcode-validity verification. This
kernel is one tool in that search.
