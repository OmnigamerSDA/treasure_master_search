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
across **both throughput and memory**, the default for any system. On an RTX 5090
this OpenCL port reaches ~64% of the CUDA default-precert represented-throughput
HM, and it remains the recommended path for non-NVIDIA devices. Run it via
`--raceway-wave-cap-mark` (with `--raceway-cap-bits/-ways`,
`--raceway-cap-ilp`, `--raceway-cap-boundaries`). Per-launch work is
**watchdog-safe** (scaled by compute-unit count, so small integrated GPUs do not
trip a GPU-recovery). The flat **checksum screen** and **compaction** paths below
are retained as **research / A-B** comparisons (the screen is the parity reference).

Supported raceway launches run the MAP1 certified-shed pre-exclusion by default:
if a key has certified shed bits, the host scans only the logical support axis
and fixes the shed bits before MAP1. Disable with `--no-precert`; explicit
`--precert` requires `--range_start 0` and a workunit divisible by
`2^certified_bits`.

## What the code does

For each candidate `(key, data)`:

1. Run the **key schedule** on the host to derive 27 per-step RNG seeds
   and algorithm selectors.
2. Upload all RNG tables to device memory.
3. Run the forward maps: 27 sequential algorithm steps (alg 0–7) over the
   32-byte working state, with cross-lane carries for the rotate variants.
4. **Production raceway:** drain the frontier through per-boundary fingerprint
   caps (FN-safe dedup), screening survivors at the end. (Research baseline:
   the flat checksum-screen kernel tests every candidate instead.)
5. Optionally run `tm_materialize_survivors` to extract candidates.

## Kernels in `src/tm.cl`

| Kernel                      | What it does                                  |
|-----------------------------|-----------------------------------------------|
| **`raceway_boundary_cap_state_offset`** | **PRODUCTION** cap-span originate (carries boundary state) |
| **`raceway_span_state_liveidx_cap_offset`** | **PRODUCTION** per-boundary cap-drain span |
| **`raceway_boundary_cap_mark_offset`** | **PRODUCTION** direct offset-stream cap mark pass |
| `tm_checksum_screen_offset_ilp6` | research: offset-stream + ILP6 flat screen (parity reference) |
| `tm_checksum_screen`        | research: baseline checksum-only screen        |
| `tm_materialize_survivors`  | Emit survivor records to host buffer          |
| `tm_stats` / `tm_process` / `test_expand` / `test_alg` / `full_process` | dev/test harness helpers |

Algorithms 0, 1, 3, 4, 6, 7 are byte-parallel. Algorithms 2 and 5 (rotate
left / rotate right by 1) need cross-lane carries — the last lane reads
a precomputed carry from `alg2_values` / `alg5_values`.

## Raceway performance (production)

FN-safe, flat memory (set by the cap, not the window). Per-launch work is scaled
by compute-unit count so small integrated GPUs do not trip a GPU-recovery watchdog.
Regime-dependent: collapse keys are fast, the diffuse keys are the long pole.

| Device | Raceway throughput |
|---|---:|
| NVIDIA RTX 5090 | ~264 M represented/s default-precert HM (8-key W256M; 2026-06-18) |
| AMD RX 7800 XT (RDNA3, 30 CU, 16 GB, fp64 cap) | ~58 M/s cap-span aggregate; ~40 M/s diffuse, ~120 M/s collapse (≈0.21× a 5090's OpenCL raceway) |
| AMD Ryzen iGPU (gfx1036, 1 CU) | ~2.2–2.8 M/s — runs the full fp64-cap pipeline; viable as a floor / CI smoke target |

Parity PASS on all of the above (`--parity`). For a bit-exact dedup count use the
research screen/compaction paths below (the raceway never misses a hit). The AMD
cap path auto-selects fp64 when `cl_khr_int64_base_atomics` is present, else a
portable fp32 cap (`--raceway-cap-fp32`).
The default-precert headline uses represented candidates per second: certified
keys keep one logical representative per shed group and multiply the logical
scan by `2^certified_bits`.

## Research baseline (screen & compaction)

A flat **checksum screen** and an **on-GPU compaction** path are retained
**purely as a stability baseline and the bit-exact parity reference** — use the
raceway (or any dedup architecture) whenever possible; these are not the
production engine. Deterministic, survivor-count-identical to the exact
reference. Flags: `--ilp6` (flat offset-stream screen), `--compaction` /
`--calibrate` (compaction A-B). Methodology: `docs/gpu_forward_benchmark_notes.md`.

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

### Production run: the bounded-wave raceway

Full `2^32` single-key sweep (FN-safe; the recommended path on non-NVIDIA GPUs):
```sh
./tm_opencl_forward --platform 0 --device 0 --key_id 0x2CA5B42D \
    --workunit_size 4294967296 --raceway-wave-cap-mark
```
Tune cap/ILP with `--raceway-cap-bits/-ways`, `--raceway-cap-ilp`, and
`--raceway-cap-boundaries`. OpenCL chunks the saved-state wave buffers to the
device allocation limit while keeping cap tables persistent, so a large workunit
does not require one full-workunit state buffer. The steps below (smoke test,
screen/compaction benchmarks, `--calibrate`) drive the **research / A-B** paths
— useful for validation and a bit-exact dedup count, not the production engine.

### Validation

```sh
./tm_opencl_forward --platform 0 --device 0 --key_id 0x2CA5B42D \
    --workunit_size 1048576 --parity 4096
```
Expect `PASS - all flag bytes match`. The flat screen is the parity reference;
the raceway is FN-safe and validated against it. Research baseline runs
(`--ilp6`, `--compaction`, `--calibrate`) are described in the CLI table below.

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
| `--precert` / `--no-precert` | Enable/disable default MAP1 certified-shed pre-exclusion for supported raceway launches |
| **`--raceway-wave-cap-mark`** | **PRODUCTION** bounded-wave raceway (state-saving cap spans + wave compaction) |
| `--raceway-cap-mark` | Diagnostic direct offset-stream boundary-cap mark pass |
| `--raceway-cap-bits <B>` / `--raceway-cap-ways <W>` | raceway cap size (FN-safe over-keep)    |
| `--raceway-cap-ilp <N>` | cap-span phase ILP                              |
| `--raceway-cap-boundaries <L>` | completed-map drain cadence, e.g. `2,5,10,16`         |
| `--raceway-cap-fp32` | force the portable fp32 cap (else fp64 if int64 atomics present) |
| `--ilp6`            | research: offset-stream + ILP6 flat screen (parity reference; +~22 MB/key) |
| `--compaction` / `--calibrate` | research: on-GPU compaction A-B + per-device geometry sweep |
| `--parity <count>`  | Compare screen kernels over `<count>` candidates flag-by-flag; PASS/FAIL. Exits. |

## Layout

```
src/
  main.cpp           OpenCL host: device init, kernel build, calibration, dispatch
  opencl_opcodes.h   6502 opcode tables + screen-flag bits (host validation)
  opencl_validate.h  host-side machine-code validation (reverse_offset, check_machine_code)
  tm.cl              OpenCL kernels: PRODUCTION raceway (raceway_boundary_cap_*,
                     raceway_span_state_*) + research screen/ILP6/HLL/compaction
  key_schedule.{cpp,h}  Forward key-schedule generator (C-style)
  rng.{cpp,h}        PRNG functions — produces RNG tables for the kernel
  rng_obj.{cpp,h}    Object-style PRNG used by the MAP1 certifier
  alignment2.{cpp,h} Memory-alignment helpers
  map1_certifier.h   Shared certified-shed MAP1 pre-exclusion helper
  data_sizes.h       Compile-time size constants
Makefile             Build driver
README.md            This file
```

## Compared to the CUDA implementation

This OpenCL port runs the same bounded-wave **raceway** architecture as CUDA. On
the same RTX 5090 it reaches **~64% of the CUDA default-precert represented HM**;
on AMD RDNA3 (RX 7800 XT) older no-precert W16M measurements were **≈0.21× a
5090's OpenCL raceway** and **≈0.14× the 5090 CUDA raceway**. The 7800 XT trails
the 5090 by ~2.85× in raw FP32/bandwidth,
so it realizes ~60% of the per-FLOP efficiency the 5090 gets on the *same* OpenCL
code — a ~40% portability gap.

The gap is structural to OpenCL 1.2, which lacks the NVIDIA-specific levers CUDA
uses (warp shuffles for register-resident cross-lane state, `__vadd4`/`__vsub4`
byte-SIMD, sm_120-tuned scheduling); the OpenCL build emulates these with `__local`
memory + `barrier()` and masked 16-bit arithmetic. Identified AMD headroom to close
the ~40% gap: RDNA3 subgroup/DPP (`ds_permute`) cross-lane ops instead of the LDS
round-trip, and a benign-race plain cap store instead of fp64 atomics (the raceway is
already FN-safe, so a relaxed cap store is tolerable). PRs welcome.

## License

MIT — see [LICENSE](LICENSE) for the full text.

## Background

*Treasure Master* (NES, 1991) ships with a hidden "Prize World" unlocked
via a 24-character password released on MTV in April 1992. Reverse
engineering later revealed a **second**, never-unlocked Prize World whose
password requires a brute-force key search over 2^32 candidates. This
OpenCL build is one front of that effort, targeting non-NVIDIA hardware.
