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
port reaches ~70% of CUDA on the same NVIDIA hardware and is the recommended
portable path for non-NVIDIA devices. Run it via `--raceway-direct-offset` (with `--raceway-cap-bits/-ways`,
`--raceway-direct-wave-span-ilp`, `--raceway-cap-boundaries`). Per-launch work is
**watchdog-safe** (scaled by compute-unit count, so small integrated GPUs do not
trip a GPU-recovery). The flat **checksum screen** and **compaction** paths below
are retained as **research / A-B** comparisons (the screen is the parity reference).

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

| Device | Raceway throughput (W16M, cadence 2,5,10,16) |
|---|---:|
| NVIDIA (same GPU) | ~70% of the CUDA raceway |
| AMD RX 7800 XT (RDNA3, 30 CU, 16 GB, fp64 cap) | 58.3 M/s cap-span HM at W16M; tuned warm runs reach 77.0 M/s mid-key at W64M and ~45 M/s on diffuse long-pole keys |
| AMD Ryzen iGPU (gfx1036, 1 CU) | ~2.2–2.8 M/s — runs the full fp64-cap pipeline; viable as a floor / CI smoke target |

Parity PASS on all of the above (`--parity`). The RX 7800 XT W16M representative
set measured 58.3 M/s cap-span HM and 49.0 M/s full-pipeline HM; the later AMD
wave-size sweep found the best warm operating points at ILP 8, cadence
`2,5,10,16`, W8M for diffuse keys and larger waves for mid/collapse keys. Discard
the first run of a fresh process before benchmarking because AMD clocks ramp from
idle. For a bit-exact dedup count use the research screen/compaction paths below
(the raceway never misses a hit). The AMD cap path auto-selects fp64 when
`cl_khr_int64_base_atomics` is present, else a portable fp32 cap
(`--raceway-cap-fp32`).

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

One W16M workunit (FN-safe; the recommended path on non-NVIDIA GPUs):
```sh
./tm_opencl_forward --platform 0 --device 0 --key_id 0x2CA5B42D \
    --range_start 0 --workunit_size 16777216 --raceway-direct-offset
```
The OpenCL host takes 32-bit `--range_start` and `--workunit_size` values. For a
full data-axis sweep, run multiple workunits:
```sh
for start in $(seq 0 16777216 4278190080); do
  ./tm_opencl_forward --platform 0 --device 0 --key_id 0x2CA5B42D \
      --range_start "$start" --workunit_size 16777216 --raceway-direct-offset
done
```
Tune cap/ILP with `--raceway-cap-bits/-ways`, `--raceway-direct-wave-span-ilp`,
`--raceway-cap-boundaries`. The steps below (smoke test, screen/compaction
benchmarks, `--calibrate`) drive the **research / A-B** paths — useful for
validation and a bit-exact dedup count, not the production engine.

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
| **`--raceway-direct-offset`** | **PRODUCTION** bounded-wave raceway (cap-span + wave compaction) |
| `--raceway-cap-bits <B>` / `--raceway-cap-ways <W>` | raceway cap size (FN-safe over-keep)    |
| `--raceway-direct-wave-span-ilp <N>` | cap-span phase ILP                              |
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
  data_sizes.h       Compile-time size constants
Makefile             Build driver
README.md            This file
```

## Compared to the CUDA implementation

This OpenCL port runs the same bounded-wave **raceway** architecture as CUDA. On
the same NVIDIA GPU it reaches **~70% of the CUDA raceway**; on AMD RDNA3 (RX 7800
XT) it is **≈0.21× a 5090's OpenCL raceway** and **≈0.14× the 5090 CUDA raceway**
(the production ceiling). The 7800 XT trails the 5090 by ~2.85× in raw FP32/bandwidth,
so it realizes ~60% of the per-FLOP efficiency the 5090 gets on the *same* OpenCL
code — a ~40% portability gap.

The gap is structural to OpenCL 1.2, which lacks the NVIDIA-specific levers CUDA
uses (warp shuffles for register-resident cross-lane state, `__vadd4`/`__vsub4`
byte-SIMD, sm_120-tuned scheduling); the OpenCL build emulates these with `__local`
memory + `barrier()` and masked 16-bit arithmetic. On the tested RX 7800 XT,
subgroup/DPP-style cross-lane experiments regressed and benign-race plain cap
stores were neutral; the measured ceiling is vector-memory-op throughput from
divergent RNG-table reads, not LDS, atomics, or occupancy. The useful AMD tuning
knob found so far is wave size: W8M for diffuse keys, larger waves for mid/collapse
when VRAM allows. PRs welcome.

## License

MIT — see [LICENSE](LICENSE) for the full text.

## Background

*Treasure Master* (NES, 1991) ships with a hidden "Prize World" unlocked
via a 24-character password released on MTV in April 1992. Reverse
engineering later revealed a **second**, never-unlocked Prize World whose
password requires a brute-force key search over 2^32 candidates. This
OpenCL build is one front of that effort, targeting non-NVIDIA hardware.
