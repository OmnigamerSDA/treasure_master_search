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
  --impl avx2_r256_map_8 --threads 1 --workunit_size 1048576
```

Recent release-candidate profiling on a Ryzen 9 9900X measured roughly
`0.55 M candidates/s/thread` for the production non-dedup AVX2 map screen
path. Scaling remains useful through high concurrency, but per-thread
throughput falls after about 8 to 12 threads from shared cache/table pressure.

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
unofficial NOPs, illegal opcodes, and JAM. The current flat-dedup winner is
the AVX2 r256s kernel, not the map kernel.

**Default merge policy:** `--first-dedup-maps 1 --dedup-every-maps 4`, i.e.
dedup once after MAP1 (captures the ~52% entry-0 collapse on its own merge
step), then merge after every 4 maps for the rest of the schedule (7 merges
total). This is the universal bathtub-bottom policy: it wins on both the
r256s and map_8 kernels and at both small (4096) and large (65536) windows
across thread counts from 1 to 24. The choice fell out of three concurrent
changes that all needed to land for f1k4 to dominate:

1. The 4-way parallel FNV fingerprint (cuts hash time ~20%).
2. The skip-expand-dedup default (saves an initial merge that does nothing
   useful for non-wrapping windows).
3. The i-cache pressure fix in `tm_avx2_r256_map_8::_run_maps_fixed`:
   removing the multi-map unrolled cases (count=3..9) and letting count > 1
   fall through to the loop dropped L1i misses 100× on map mode, lifted
   per-state IPC, and re-balanced the bathtub bottom from f1k5 onto f1k4.

Tuning knobs:

- `--dedup-every-maps 5` (f1k5) trades 1 fewer merge for ~10% more map work
  per call. Was the deep-concurrency winner under the pre-fix map_8 i-cache
  pressure, but is now slightly behind f1k4 everywhere. Kept as a knob for
  scenarios where L3 contention is unusually high (e.g. many co-resident
  workloads competing for the shared cache).
- `--dedup-every-maps 3` (f1k3) gives marginally tighter dedup. At large
  windows with low task count per thread it occasionally matches or slightly
  beats f1k4; the gap is within run-to-run noise.
- `--skip-final-hash` skips the post-last-map merge for callers that don't
  need a fully-dedup'd output (parity-clean; emits the un-deduplicated
  post-last-map state set). Sub-noise at production loads; documented for
  reference.

On a 512-key representative sample (512 keys × 16 windows/key, window=4096,
12 Ryzen 9 9900X threads, the documented production shape), flat/no-origin
dedup plus full flag screening with the **current default policy** measures
**5.93 M candidates/s total** (0.495 M/s/thread). At threads=24 the same
default measures **9.93 M/s**. The previous K=1 default measured 5.43 M/s
at threads=12 and 8.67 M/s at threads=24 — the policy change plus the
4-way parallel FNV fingerprint plus the skip-expand-dedup default plus the
i-cache fix combine to **+9-14% throughput uplift** over the previous
default, with no API or invocation change required for callers using the
new default.

Window size is the primary throughput/coverage tradeoff. `--window 4096` is
the conservative general-purpose setting. A fixed-work sweep over 64 sampled
CNF-forward keys found the best observed 12-thread throughput at
`--window 65536`, around 9.0-9.5 M candidates/s total, but larger windows did
not keep improving. Use 4096 for the most tested baseline; use 65536 as the
first larger-window trial when CPU throughput matters and the workload is
similar to the sampled candidate-key distribution.

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
