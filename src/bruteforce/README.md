# Forward Search Implementations

This directory contains the CPU, CUDA, and OpenCL forward verifiers for the
Treasure Master Bonus World 2 search. All three implementations evaluate the
same transform: fix a 32-bit key, sweep data values, run the key schedule and
forward maps, then screen the final state against the target worlds.

## Build Targets

From the repository root:

```sh
make cpu
make cpu-dedup
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
bench_cpu/                   Native CPU throughput benchmark
state_dedup_speedup_bench/   Flat dedup speedup smoke/parity tool
state_dedup_matrix_bench/    Representative-key/window matrix tool
state_dedup_screen_bench/    Flat/no-origin dedup plus full flag screening
state_dedup_origin_bench/    Origin-tracking dedup benchmark
```

The consolidated public repo exposes the cleaned GPU releases as top-level
`cuda/` and `opencl/` directories. In the development tree those packages are
staged under `release_staging/tm_cuda/` and `release_staging/opencl_public/`.

## CPU

Primary current CPU implementation:

```sh
./src/bruteforce/bench_cpu/bench_cpu \
  --impl avx2_r256s_8 --threads 1 --workunit_size 1048576
```

Recent release-candidate profiling on a Ryzen 9 9900X measured roughly
`0.52 M candidates/s/thread` for the production non-dedup AVX2 screen path.
Scaling remains useful through high concurrency, but per-thread throughput
falls after about 8 to 12 threads from shared cache/table pressure.

The CPU state-dedup route is useful when many data values converge to the
same final states. The fast production shape is:

```sh
./src/bruteforce/state_dedup_screen_bench/state_dedup_screen_bench \
  --key 0x2CA5B42D --window 4096 --windows-per-key 16 --threads 4
```

This tool runs flat/no-origin dedup, screens unique final states for checksum
and machine-code flags, and reports both unique-state counts and original
multiplicity counts. Use `--strict-invalid-mask` to control which flags
disqualify a strict `ALL_ENTRIES_VALID` result. The default rejects
unofficial NOPs, illegal opcodes, and JAM.

Origin tracking is available separately:

```sh
./src/bruteforce/state_dedup_origin_bench/state_dedup_origin_bench \
  --key 0x2CA5B42D --window 4096
```

Recommended production pattern: run no-origin dedup for bulk screening, then
rerun only rare strict-passing windows with origin tracking or a direct scan
to recover exact data offsets.

## CUDA

CUDA is the fastest route on NVIDIA hardware.

Build:

```sh
cd cuda
make CUDA_ARCHS="89 120"
```

Parity check:

```sh
./tm_cuda --device 0 --parity 64 --batch_size 64 \
  --workunit_size 64 --warmup_batches 0
```

Short benchmark:

```sh
./tm_cuda --device 0 --key_id 0x2CA5B42D --range_start 0 \
  --workunit_size 16777216 --batch_size 1048576 --warmup_batches 1 \
  --screen-offsets --ilp 8
```

The current production screen path is the offset-stream kernel. Recent
measurements were about `132 M candidates/s` on RTX 5090 and about
`105 M candidates/s` on RTX PRO 6000 Blackwell Max-Q.

## OpenCL

OpenCL is the portable GPU route for AMD, Intel, Apple, and non-CUDA systems.

Build:

```sh
cd opencl
make OPENCL_HEADERS=/path/to/OpenCL-Headers OPENCL_LIB=/path/to/libOpenCL.so
```

Smoke test:

```sh
./tm_opencl_forward --platform 0 --device 0 \
  --key_id 0x2CA5B42D --range_start 0 \
  --workunit_size 65536 --batch_size 65536 --warmup_batches 0
```

Fast OpenCL path:

```sh
./tm_opencl_forward --platform 0 --device 0 \
  --key_id 0x2CA5B42D --range_start 0 \
  --workunit_size 16777216 --batch_size 1048576 \
  --warmup_batches 1 --ilp6
```

## Notes

- `docs/forward_release_candidate_20260525.md` contains the current CPU/GPU
  release-candidate profiling summary.
- `docs/gpu_forward_benchmark_notes.md` contains the CUDA tuning history.
- The public package is Makefile-first. Development-only legacy workers,
  reverse search, and CNF/SAT tooling are intentionally omitted.
