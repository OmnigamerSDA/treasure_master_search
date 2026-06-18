# Forward Search Implementations

This directory contains the CPU, CUDA, and OpenCL forward verifiers for the
Treasure Master Bonus World 2 search. All three implementations evaluate the
same transform: fix a 32-bit key, sweep data values, run the key schedule and
forward maps, then screen the final state against the target worlds.

## Build Targets

From the repository root:

```sh
make raceway    # PRODUCTION CPU engine (bounded-wave forward dedup)
make cpu        # research: native AVX/SIMD throughput benchmark
make cpu-dedup  # research: state-dedup characterization tools
make cuda
make opencl
```

Or build individual CPU tools in place:

```sh
cd src/bruteforce/bench_cpu && make
cd src/bruteforce/state_dedup_screen_bench && make
```

## Layout

```text
cpu/                         Scalar + SIMD forward kernels
cpu_raceway/                 PRODUCTION CPU engine — bounded-wave forward dedup
bench_cpu/                   research: native CPU throughput benchmark
state_dedup_speedup_bench/   research: flat dedup speedup smoke/parity tool
state_dedup_matrix_bench/    research: representative-key/window matrix tool
state_dedup_screen_bench/    research: flat/no-origin dedup plus full flag screening
state_dedup_origin_bench/    research: origin-tracking dedup benchmark
```

The consolidated public repo exposes the cleaned GPU releases as top-level
`cuda/` and `opencl/` directories. In the development tree those packages are
staged under `release_staging/tm_cuda/` and `release_staging/opencl_public/`.

## CPU

**Production CPU engine (2026-06-16): the bounded-wave raceway** (`cpu_raceway/`) — best
across BOTH throughput and memory, the default for any host. It auto-selects the right ISA
build (AVX-512 natmap / AVX2 legacy) and a host-appropriate wave/cap:

```sh
cd src/bruteforce/cpu_raceway && ./raceway_autoconfig.sh --build
./cpu_raceway <key> 0 <threads>        # 0 = full 2^32 window (needs PRODUCER_CAP=1)
```

### Thread scaling

Raceway throughput vs thread count, AVX-512 build on a Ryzen 9 9900X (12 physical
cores / 24 SMT), W16M, cap-on, harmonic mean across the closer/mid/diffuse key
classes:

| Threads | Raceway throughput (HM) | vs 1 thread |
|---|---:|---:|
| 1                | 2.43 M/s  | 1.0×  |
| 8                | 15.3 M/s  | 6.3×  |
| 12 (all physical)| 21.1 M/s  | 8.7×  |
| 24 (SMT on)      | 27.5 M/s  | 11.3× |

≈72% multi-core efficiency out to all 12 physical cores; SMT (12→24 threads) adds
a further ~30%. Per-thread throughput tapers at high counts from shared-L3
pressure, so the cap (which bounds the working set) is the main lever there.
Per-key spread at 24 threads: closer ≈114, mid ≈32, diffuse ≈14 M/s (diffuse is
the long pole). The AVX2 build runs ≈0.7× the AVX-512 build. Use
`raceway_autoconfig.sh` to pick the pin set and cap for a given host.

The CPU raceway enables MAP1 certified-shed pre-exclusion by default. Certified
keys scan only the logical support axis before MAP1; zero-cert keys are a
no-op. On the 2026-06-18 8-key W256M default-precert set, the 24-thread AVX-512
build measured 22.9 M represented candidates/s HM, with certified collapsers at
1.34 B/s HM and no-cert diffuse keys at 13.5 M/s HM. Use `PRECERT=0` for raw
contiguous-window parity tests.

### Research / characterization tools

`bench_cpu` (non-dedup AVX/SIMD throughput) and the `state_dedup_*_bench` family
(flat/origin dedup + flag screening) are **baselines only** — use the raceway for
real searches. They remain useful for kernel A-B throughput, dedup-collapse
studies, and as the exact-screen parity reference. Examples:

```sh
./bench_cpu/bench_cpu --impl avx2_r256_map_8 --threads 1 --workunit_size 1048576
./state_dedup_screen_bench/state_dedup_screen_bench \
  --key 0x2CA5B42D --window 4096 --windows-per-key 16 --threads 4
```

Their tuning knobs (`--first-dedup-maps`, `--dedup-every-maps`, `--window`,
origin tracking) are documented in each tool's `--help`.

## GPU (CUDA / OpenCL)

The GPU engines live in `cuda/` and `opencl/` with their own READMEs. Both run
the **bounded-wave raceway** as the production engine (CUDA fastest on NVIDIA;
OpenCL measured ~64% of CUDA on the same RTX 5090 default-precert HM); the flat screen / compaction paths
are research/baseline only. Supported production raceway launches enable the
same MAP1 certified-shed pre-exclusion by default; disable with `--no-precert`.
Build with `make cuda` / `make opencl` from the repo root, then see the package
README for raceway run + per-device calibration.

## Notes

- `docs/forward_release_candidate_20260525.md` contains the current CPU/GPU
  release-candidate profiling summary.
- `docs/gpu_forward_benchmark_notes.md` contains the CUDA tuning history.
- The public package is Makefile-first. Development-only legacy workers,
  reverse search, and CNF/SAT tooling are intentionally omitted.
