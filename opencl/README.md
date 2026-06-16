# Treasure Master — OpenCL Forward Search

An OpenCL implementation of the forward decryption/state-check loop for
the 1991 NES game *Treasure Master*. Given a 4-byte key, the kernels
sweep the 2^32 starting-data space, running the game's custom 27-step
algorithm forward and checking whether the result matches either the
carnival-world or other-world target.

This directory is the OpenCL implementation in the consolidated Treasure
Master forward-search release. CUDA is faster on NVIDIA hardware, but this
OpenCL build remains useful for **AMD GPUs, Intel GPUs, Apple Silicon, and
any non-NVIDIA accelerator**.

## Production engine: the bounded-wave raceway

The **production forward engine (2026-06-16) is the bounded-wave raceway** — best
across **both throughput and memory**, the default for any system. This OpenCL
port runs at ~70% of the CUDA raceway and is the recommended path for non-NVIDIA
devices. Run it via `--raceway-direct-offset` (with `--raceway-cap-bits/-ways`,
`--raceway-direct-wave-span-ilp`, `--raceway-cap-boundaries`). Per-launch work is
**watchdog-safe** (scaled by compute-unit count, so small integrated GPUs do not
trip a GPU-recovery). The flat **checksum screen** and **compaction** paths below
are retained as **research / A-B** comparisons (the screen is the parity reference).

## What the code does

For each candidate `(key, data)`:

1. Run the **key schedule** on the host to derive 27 per-step RNG seeds
   and algorithm selectors.
2. Upload all RNG tables to device memory.
3. Run the kernel: 27 sequential algorithm steps (alg 0–7) over the 32-
   byte working state, with cross-lane carries for the rotate variants.
4. Run the **checksum-screen kernel** to filter survivors.
5. Optionally run `tm_materialize_survivors` to extract candidates.

## Kernels in `src/tm.cl`

| Kernel                      | What it does                                  |
|-----------------------------|-----------------------------------------------|
| `tm_process`                | Single-step algorithm dispatch (alg 0..7)     |
| `test_expand`               | IV expansion (8 B → 128 B via RNG accumulation) |
| `test_alg`                  | Standalone algorithm test harness             |
| `full_process`              | End-to-end (expand + schedule + alg loop)     |
| `tm_stats`                  | Machine-flag stats accumulator                |
| `tm_checksum_screen`        | Baseline checksum-only screen (1 candidate / work-group) |
| `tm_checksum_screen_offset_ilp6` | Fast screen: offset-stream + ILP6 + preids (~2.3× faster) |
| `tm_materialize_survivors`       | Emit survivor records to host buffer        |

Algorithms 0, 1, 3, 4, 6, 7 are byte-parallel. Algorithms 2 and 5 (rotate
left / rotate right by 1) need cross-lane carries — the last lane reads
a precomputed carry from `alg2_values` / `alg5_values`.

## Performance (research / A-B paths)

Besides the production raceway above, three screen/compaction paths are available
for comparison and as the bit-exact parity reference:

* **Baseline** (default fallback): universal RNG tables, 1 candidate per work-group.
  Stable, broadly portable. The original implementation.
* **Offset-stream + ILP6** (`--ilp6`): per-key precomputed offset streams,
  6 candidates per work-group, all 6 algorithm-IDs precomputed before the
  alg-apply phase. **~2.3× faster on the GPUs measured.** Costs ~22 MB of
  extra device memory per key.
* **On-GPU VRAM compaction** (`--compaction`, or auto-selected after `--calibrate`):
  `run_span_dedup` + `compact_survivors_ordered` + `resolve_flags`. Keeps survivors
  in device memory between spans and shrinks the effective work-group grid, recovering
  idle occupancy. **~1.4–1.8× over baseline on high-R keys.** Use `--calibrate` to
  measure the optimal span geometry for your GPU and key mix.

Measured on a clean GPU (no contention). 2^24-candidate workunit, key
`0x2CA5B42D`:

| GPU                                       | Path        | Screen rate    | Wall rate     |
|-------------------------------------------|-------------|----------------|---------------|
| NVIDIA RTX 5090                            | baseline    | 37.0 M cand/s  | 33.4 M cand/s |
| NVIDIA RTX 5090                            | `--ilp6`    | **86.5 M cand/s** | **74.9 M cand/s** |
| NVIDIA RTX PRO 6000 Blackwell Max-Q WS Ed. | baseline    | 32.6 M cand/s  | 30.5 M cand/s |
| NVIDIA RTX PRO 6000 Blackwell Max-Q WS Ed. | `--ilp6`    | **72.2 M cand/s** | ~62 M cand/s |

Survivor counts are byte-identical between the two paths (parity-verified
across 5 keys at 2^24 candidates each). The `--ilp6` path closes most of
the historical OpenCL-vs-CUDA gap: the CUDA implementation still wins on
NVIDIA hardware, but the OpenCL path is now within ~1.5× rather than
~3.5× on the RTX 5090 measurement.

The 21-key ledger sweep (16 shards of 2^28 each, sequential) from the
parent research project ran on this OpenCL path and matched later CUDA
re-verification on the same survivor sets — i.e. the OpenCL kernel is
slower but correct.

At the measured screen-kernel rates, a full `2^32` single-key sweep is
roughly **50 s** on RTX 5090 with `--ilp6`, **1.9 min** on RTX 5090
baseline, and **60 s** on RTX PRO 6000 Blackwell Max-Q with `--ilp6`.
The short-run wall-rate projections are about **57 s**, **2.1 min**, and
**1.2 min** respectively because startup, precompute, launch, copy, and
reporting overhead are a larger fraction of a small benchmark.

To run your own benchmark, see the [Run](#run) section below.

If you collect numbers worth sharing for AMD / Intel / Apple Silicon
hardware, please open a PR adding them here.

## Build

### Prerequisites

- **An OpenCL 1.2+ runtime and headers** (`<CL/cl.h>`).
- **A C++11 compiler** (the default is `g++`; override with `make CXX=clang++`).
- **A GPU with a working OpenCL ICD installed** — any NVIDIA, AMD,
  Intel, or Apple GPU with vendor drivers should expose one.

### Build commands

**Linux (Debian / Ubuntu)** — installs both headers and the unversioned
`-lOpenCL` link:
```sh
sudo apt install opencl-headers ocl-icd-opencl-dev
make
```

**Linux (Fedora / RHEL)**:
```sh
sudo dnf install opencl-headers ocl-icd-devel
make
```

**Linux with only CUDA installed** (no system OpenCL SDK): the CUDA
toolkit ships a libOpenCL ICD loader as `libOpenCL.so.1` (no unversioned
symlink, so `-lOpenCL` won't resolve). It does **not** ship the headers
in current versions — get those separately from the
[Khronos OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers)
source tree, or install `opencl-headers` from your distro. Then:
```sh
make OPENCL_HEADERS=/path/to/OpenCL-Headers \
     OPENCL_LIB=/usr/local/cuda/targets/x86_64-linux/lib/libOpenCL.so.1
```

**macOS** (OpenCL framework ships with the OS):
```sh
make OPENCL_LIB='-framework OpenCL'
```

**Windows** (MSYS2 / Cygwin with GPU vendor's OpenCL SDK installed):
```sh
make
```

### Build output

- `tm_opencl_forward` — the host binary.
- The kernel `src/tm.cl` is loaded at runtime. The binary searches `./tm.cl`
  then `src/tm.cl` relative to the working directory, so running it from the
  package root works without copying anything. To run from elsewhere, either
  `cd` into the package directory or copy `src/tm.cl` next to the binary.

## Run

### 1. Identify your platform and device indices

OpenCL distinguishes **platforms** (vendor ICDs) from **devices** (the
GPUs themselves). Most systems have one platform and one device, so the
common case is `--platform 0 --device 0`.

To enumerate what's available, install `clinfo` and run it:
```sh
sudo apt install clinfo   # or dnf install clinfo
clinfo -l
# Example output:
# Platform #0: NVIDIA CUDA
#  `-- Device #0: NVIDIA GeForce RTX 5090
#  `-- Device #1: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
# Platform #1: Intel(R) OpenCL HD Graphics
#  `-- Device #0: Intel(R) UHD Graphics 770
```

The host's startup banner also prints the resolved platform and device
names — verify before benchmarking.

### 2. Smoke test

Confirm the kernel runs end-to-end on a 64K-candidate sweep:
```sh
./tm_opencl_forward --platform 0 --device 0 \
                    --key_id 0x2CA5B42D --range_start 0 \
                    --workunit_size 65536 --batch_size 65536 \
                    --warmup_batches 0
```
You should see a `wall_rate` reported and survivor counts. For this
specific small range and key, expect `checksum survivors: 0`.

### 3. Short throughput benchmark

Baseline kernel:
```sh
./tm_opencl_forward --platform 0 --device 0 \
                    --key_id 0x2CA5B42D --range_start 0 \
                    --workunit_size 16777216 --batch_size 1048576 \
                    --warmup_batches 1
```

Fast kernel (offset-stream + ILP6, ~2.3× faster):
```sh
./tm_opencl_forward --platform 0 --device 0 \
                    --key_id 0x2CA5B42D --range_start 0 \
                    --workunit_size 16777216 --batch_size 1048576 \
                    --warmup_batches 1 --ilp6
```

### 4. Calibrate compaction geometry for your GPU (recommended once per machine)

```sh
./tm_opencl_forward --platform 0 --device 0 --calibrate
```

Sweeps the pre-compiled span geometries and writes the winner to
`tm_compaction.conf`, keyed by device fingerprint. Subsequent runs load this
config and auto-select `--compaction` when it outperforms the screen path.

### 5. Larger sweep, closer to steady-state

```sh
./tm_opencl_forward --platform 0 --device 0 \
                    --key_id 0x2CA5B42D --range_start 0 \
                    --workunit_size 268435456 --batch_size 1048576 \
                    --warmup_batches 1 --ilp6
```

### 6. Parity check (verify the two kernels agree)

```sh
./tm_opencl_forward --platform 0 --device 0 \
                    --key_id 0x2CA5B42D --range_start 0 \
                    --workunit_size 16777216 --parity 16777216
```
Expect: `PASS - all 16777216 flag bytes match` and a `speedup` value
~2.0–2.4× depending on GPU.

### Full CLI

| Flag                | Purpose                                                      |
|---------------------|--------------------------------------------------------------|
| `--platform <idx>`  | OpenCL platform index (from `clinfo -l`)                     |
| `--device <idx>`    | Device index within that platform                            |
| `--key_id <hex>`    | The 32-bit key to fix while sweeping the data axis           |
| `--range_start <N>` | First data index to test (0-based)                           |
| `--workunit_size <N>` | Number of data indices to sweep                            |
| `--batch_size <N>`  | Kernel batch size (default 2^20 candidates)                  |
| `--warmup_batches <N>` | Number of warm-up batches before timing                   |
| `--output_csv <path>` | Optional CSV output of survivors                           |
| `--ilp6`            | Use the offset-stream + ILP6 screen kernel (~2.3× faster).   |
|                     | Uses ~22 MB extra device memory per key.                     |
| `--compaction`      | Use the on-GPU VRAM compaction path (auto-selected after      |
|                     | `--calibrate` if it wins on your device).                    |
| `--calibrate`       | Sweep span geometries, measure vs screen, write              |
|                     | `tm_compaction.conf`. Exits after calibration.               |
| `--parity <count>`  | Run both screen kernels over `<count>` candidates, compare   |
|                     | every flag byte, and report speedup + PASS/FAIL. Exits.      |

## Layout

```
src/
  main.cpp           OpenCL host: device init, kernel build, calibration,
                     compaction pipeline, dispatch
  tm.cl              OpenCL kernels: screen, ILP6, HLL, dump, materialize,
                     and on-GPU compaction (run_span_dedup,
                     compact_survivors_ordered, resolve_flags)
  key_schedule.{cpp,h}  Forward key-schedule generator (C-style)
  rng.{cpp,h}        PRNG functions — produces RNG tables for the kernel
  data_sizes.h       Compile-time size constants
Makefile             Build driver
README.md            This file
```

## Compared to the CUDA implementation

The CUDA implementation reaches ~140 M cand/s on an RTX 5090 (using its
`--screen-offsets --ilp 6` path) by exploiting NVIDIA-specific features
that OpenCL 1.2 does not expose:

- Warp shuffles (`__shfl_sync`, `__shfl_down_sync`) for register-resident
  inter-lane state — no need for shared memory or barriers in the inner
  loop. The OpenCL build relies on `__local` memory + `barrier()`.
- `__vadd4` / `__vsub4` byte-SIMD intrinsics for alg-1 / alg-4. The
  OpenCL build emulates these with masked 16-bit arithmetic.
- 4-warps-per-block scheduling at register pressure tuned for sm_120.

The OpenCL build **does** implement the portable subset of the
advancements:

- Offset-stream RNG (`--ilp6` path): host pre-walks `rng_seed` per
  schedule step and ships a 27 × 2048 × 128-byte stream per stream type,
  removing the runtime `rng_forward_1 / rng_forward_128` walk.
- ILP6: 6 candidates per work-group, sharing the 27-step schedule cost.
- PreIDs: all 6 algorithm-IDs precomputed before the alg-apply phase.

The net effect on RTX 5090 is ~86 M cand/s for the `--ilp6` OpenCL path
vs ~37 M cand/s for the baseline — i.e. the OpenCL gap to CUDA is now
~1.5× rather than ~3.5×. Further closing the gap would require either
OpenCL 2.0+ subgroup extensions (where supported) or a vendor-specific
intrinsic path. PRs welcome.

## License

MIT — see [LICENSE](LICENSE) for the full text.

## Background

*Treasure Master* (NES, 1991) ships with a hidden "Prize World" unlocked
via a 24-character password released on MTV in April 1992. Reverse
engineering later revealed a **second**, never-unlocked Prize World whose
password requires a brute-force key search over 2^32 candidates. This
OpenCL build is one front of that effort, targeting non-NVIDIA hardware.
