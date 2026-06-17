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

The tools below (`bench_cpu`, `state_dedup_*_bench`) are **research / characterization**
harnesses (throughput, dedup-collapse, parity), not the production search path:

```sh
./src/bruteforce/bench_cpu/bench_cpu \
  --impl avx2_r256_map_8 --threads 1 --workunit_size 1048576
```

Recent release-candidate profiling on a Ryzen 9 9900X measured roughly
`0.55 M candidates/s/thread` for the non-dedup AVX2 map screen path. Scaling remains useful
through high concurrency, but per-thread throughput falls after about 8 to 12 threads from
shared cache/table pressure.

The other CPU tools here — `bench_cpu` (non-dedup AVX/SIMD throughput) and the
`state_dedup_*_bench` family (flat/origin dedup + flag screening) — are
**research / characterization baselines only**: use the raceway (any dedup
architecture) for real searches. They remain useful for kernel A-B throughput,
dedup-collapse studies, and as the exact-screen parity reference. Example:

```sh
./state_dedup_screen_bench/state_dedup_screen_bench \
  --key 0x2CA5B42D --window 4096 --windows-per-key 16 --threads 4
```

Their tuning knobs (`--first-dedup-maps`, `--dedup-every-maps`, `--window`,
origin tracking) are documented in each tool's `--help`.

## GPU (CUDA / OpenCL)

The GPU engines live in `cuda/` and `opencl/` with their own READMEs. Both run
the **bounded-wave raceway** as the production engine (CUDA fastest on NVIDIA;
OpenCL ~70% of CUDA for non-NVIDIA devices); the flat screen / compaction paths
are research/baseline only. Build with `make cuda` / `make opencl` from the repo
root, then see the package README for raceway run + per-device calibration.

## Notes

- `docs/forward_release_candidate_20260525.md` contains the current CPU/GPU
  release-candidate profiling summary.
- `docs/gpu_forward_benchmark_notes.md` contains the CUDA tuning history.
- The public package is Makefile-first. Development-only legacy workers,
  reverse search, and CNF/SAT tooling are intentionally omitted.
