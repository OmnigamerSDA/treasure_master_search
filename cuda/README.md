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

## Production engine: the bounded-wave raceway

The **production forward engine (2026-06-16) is the bounded-wave raceway**
(`src/tm_cuda_raceway.cuh`) — best across **both throughput and memory**, the
default for any system. It originates the MAP1 frontier into an LLC-/VRAM-sized
wave and drains it through persistent per-boundary fingerprint caps; it is
**FN-safe** (finds every hit). Run it:

```sh
./tm_cuda --device D --key_id 0x... --workunit_size 4294967296 \
  --raceway-direct-wave-continue-batch auto
```

Tune it per device once (sweeps span-ILP × cap-bits, auto-applied on later runs):

```sh
./tm_cuda --device D --calibrate-raceway
```

The flat **checksum screen** and **on-GPU compaction** paths below are retained
as **research / A-B** comparisons (the screen is also the bit-exact parity
reference and the exact per-value hit tally).

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

## Raceway performance (production)

Full-key `2^32`, FN-safe, flat memory (set by the cap, not the window):

| GPU | Raceway throughput |
|---|---:|
| RTX 5090 | ~310 M/s typical (population harmonic mean); ~224–261 M/s on the diffuse long pole |
| RTX PRO 6000 Blackwell Max-Q | ~0.8× the 5090 (clock-bound) |

Tune once per device with `tm_cuda --calibrate-raceway` (sweeps span-ILP ×
cap-bits, records `engine=raceway …` to `tm_compaction.conf`); production raceway
runs auto-apply the calibrated span-ILP. The raceway never misses a hit; for a
**bit-exact dedup count** use the research screen/compaction paths below, which
it is validated against.

## Research baseline (screen & compaction)

A flat **checksum screen** and an **on-GPU VRAM compaction** path are retained
**purely as a stability baseline and the bit-exact parity reference** — use the
raceway (or any dedup architecture) whenever possible; these are not the
production engine. They are deterministic and survivor-count-identical to the
exact reference. Flags: `--screen-offsets [--ilp 4|6|8]` (flat screen),
`--compaction` / `--calibrate` (compaction A-B). Implementation + tuning notes
live in `src/tm_cuda_screen.cuh` and `src/tm_cuda_dedup.cuh`; broader methodology
in `docs/gpu_forward_benchmark_notes.md`.

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

The default launch shape targets Blackwell. The research screen kernel exposes
`-DTM_WARPS_PER_BLOCK` / `-DTM_CANDIDATES_PER_WARP` for older/smaller GPUs; the
production raceway is tuned per device by `--calibrate-raceway` instead.

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

### Production run: the bounded-wave raceway

One-time per device, tune the raceway geometry (sweeps span-ILP × cap-bits,
writes `engine=raceway …` to `tm_compaction.conf`):
```sh
./tm_cuda --device 0 --calibrate-raceway
```

Full `2^32` single-key sweep (FN-safe; auto-applies the calibrated span-ILP;
`auto` sizes the wave/cap to free VRAM):
```sh
./tm_cuda --device 0 --key_id 0x2CA5B42D \
            --workunit_size 4294967296 --raceway-direct-wave-continue-batch auto
```

The raceway runs MAP1 certified-shed pre-exclusion by default where the selected
production launch shape has a mask-aware bridge. For certified keys this reduces
the work to the logical support axis before MAP1; for zero-cert keys it is a
no-op. Use `--no-precert` to force the contiguous raw data sweep, or explicit
`--precert` to require a supported certifier-aware shape. The bridge currently
expects `--range_start 0` and a workunit divisible by `2^certified_bits`.

The steps below (parity, screen/compaction benchmarks, `--calibrate`) drive the
**research / A-B** paths — useful for validation and for a bit-exact dedup count,
but not the production engine.

### Validation (parity vs the CPU reference)

```sh
./tm_cuda --device 0 --parity 64 --batch_size 64 --workunit_size 64 --warmup_batches 0
```
Expect `PASS`. The flat screen is the parity reference; the raceway is FN-safe
(finds every hit) and validated against it. Research baseline runs
(`--screen-offsets`, `--compaction`, `--calibrate`) are documented in the source.

### Note on device numbering

The `--device` index passed to the host is the CUDA enumeration index
and **may not match `nvidia-smi` ordering** under some driver / topology
configurations. Always check the device name printed in the startup
banner before drawing conclusions from a benchmark run.

**Range bounds**: `--range_start` and `--workunit_size` accept 64-bit
values on the host side, but the kernel is 32-bit candidate-indexed.
`--range_start + --workunit_size` must stay within 2^32.

**Precertifier**: `--precert` / `--no-precert` controls the default per-key
MAP1 certified-shed pre-exclusion. It is enabled by default for supported
production raceway launches and disabled with `--no-precert`.

## Layout

```
src/
  main.cpp                CUDA host: argument parsing, kernel launch, calibration,
                          compaction pipeline, validation
  tm_cuda.cu              CUDA kernel compilation unit (thin stub — includes below)
  tm_cuda_primitives.cuh  Device helpers: __constant__ tables, run_alg, schedule
                          runners, HLL support, screen_candidate
  tm_cuda_raceway.cuh     PRODUCTION engine: bounded-wave raceway (cap-span +
                          wave-local compaction) kernels and fixed-capacity caps
  tm_cuda_screen.cuh      research: screen/HLL/dump/materialize kernels (also the
                          bit-exact parity reference)
  tm_cuda_dedup.cuh       research: on-GPU VRAM compaction kernels (run_span_dedup,
                          compact_survivors_ordered, span geometries for --calibrate)
  wide_merge_sort.cu      cub radix-sort / RLE host+device object (wide-merge path)
  key_schedule.{cpp,h}    Forward key-schedule generator
  rng_obj.{cpp,h}         PRNG class — produces all RNG tables fed to the kernel
  alignment2.{cpp,h}      Memory-alignment helpers
  data_sizes.h            Compile-time size constants
Makefile                  Build driver (configurable CUDA_PATH and GENCODE)
README.md                 This file
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
and `tm_cuda_primitives.cuh` / `tm_cuda_screen.cuh` sources.

## License

MIT — see [LICENSE](LICENSE) for the full text.

## Background

*Treasure Master* shipped with a hidden "Prize World" reachable only by
entering a 24-character password released on MTV in April 1992. Years
later, reverse-engineering revealed a **second** Prize World ("Bonus
World 2") that was never publicly unlocked — its password requires a
key search over 2^32 candidates plus opcode-validity verification. This
kernel is one tool in that search.
