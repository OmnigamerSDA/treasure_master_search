# Treasure Master Forward Search

Forward-search implementations for the hidden Bonus World 2 unlock code in
the NES game *Treasure Master*.

This repository packages the actively maintained forward verifiers in one
place:

- **CPU**: scalar/SIMD implementations, including the current AVX2 path and
  state-dedup experiments.
- **CUDA**: the fastest NVIDIA GPU verifier, including the offset-stream
  screen path.
- **OpenCL**: portable GPU verifier for AMD, Intel, Apple, and non-CUDA
  systems.

The three paths all solve the same task: for a fixed 32-bit key, sweep data
values, run the game's forward transform, and screen the final state for the
target worlds.

## Quick Build

From the repository root:

```sh
make cpu        # native CPU AVX/SIMD benchmark
make cpu-dedup  # CPU state-dedup characterization tools
make cuda       # CUDA forward verifier
make opencl     # OpenCL forward verifier
```

Each implementation also has its own Makefile: CPU tools under
`src/bruteforce/`, CUDA under `cuda/`, and OpenCL under `opencl/`.

## First Runs

CPU:

```sh
./src/bruteforce/bench_cpu/bench_cpu \
  --impl avx2_r256_map_8 --threads 1 --workunit_size 1048576
```

CPU flat-dedup screen:

```sh
./src/bruteforce/state_dedup_screen_bench/state_dedup_screen_bench \
  --key 0x2CA5B42D --window 4096 --windows-per-key 16 --threads 4
```

CUDA parity check:

```sh
./cuda/tm_cuda \
  --device 0 --parity 64 --batch_size 64 --workunit_size 64 \
  --warmup_batches 0
```

OpenCL smoke test:

```sh
./opencl/tm_opencl_forward \
  --platform 0 --device 0 --key_id 0x2CA5B42D --range_start 0 \
  --workunit_size 65536 --batch_size 65536 --warmup_batches 0
```

## Reference Performance

Typical production use fixes one 32-bit key and sweeps the full `2^32`
data space. For that use case, the screen-kernel rate is the best
steady-state comparison point. Short-run wall rates include startup,
precompute, launch, copy, and reporting overhead, so they are useful for
small local checks but can understate full-key production throughput.

| Path | Hardware / mode | Steady screen rate | Full-key screen time | Short-run wall rate |
|---|---|---:|---:|---:|
| CUDA | RTX 5090, `--screen-offsets --ilp 6` | 132.4-141.2 M/s | ~30-33 s | 108.2 M/s |
| CUDA | RTX 5090, baseline | 102.6 M/s | ~42 s | 82.9 M/s |
| OpenCL | RTX 5090, `--ilp6` | 86.5 M/s | ~50 s | 74.9 M/s |
| OpenCL | RTX 5090, baseline | 37.0 M/s | ~1.9 min | 33.4 M/s |
| OpenCL | RTX PRO 6000 Blackwell Max-Q, `--ilp6` | 72.2 M/s | ~60 s | ~62 M/s |
| CPU AVX2 flat dedup | Ryzen 9 9900X, 12 threads, window 4096 | 6.373 M/s (~0.531/thread) | distribution-dependent | same |
| CPU AVX2 flat dedup | Ryzen 9 9900X, 12 threads, window 65536 | ~9.0-9.5 M/s (~0.75-0.79/thread) | distribution-dependent | same |
| CPU AVX2 map | Ryzen 9 9900X, 24 threads, non-dedup | ~7.2 M/s | ~9.9 min | same |
| CPU AVX2 map | Ryzen 9 9900X, 12 threads, non-dedup | ~5.1 M/s | ~14.1 min | same |
| CPU AVX2 map | Ryzen 9 9900X, 1 thread, non-dedup | ~0.55 M/s | ~2.2 h | same |

CPU state dedup is key-distribution dependent. On a 512-key representative
sample from prior forward-search candidates, flat/no-origin dedup plus full
flag screening measured 6.373 M/s on 12 threads at window 4096. This is the
main CPU path to try for candidate routing because it avoids replaying
duplicate final states. Use it as a routing/screening tool, not as a universal
replacement for the GPU full-key sweep. The flat dedup path uses the AVX2
r256s kernel; `--dedup-every-maps 2` can help modestly on larger windows/high
thread counts, while the default of `1` is the conservative setting.

Window size is the main CPU dedup tuning knob. `--window 4096` is the
conservative default used for broad representative testing. In a fixed-work
sweep over 64 sampled CNF-forward keys, larger windows improved throughput up
to `--window 65536`, which measured about 9.0-9.5 M/s total on 12 Ryzen 9
9900X threads. Larger windows were not monotonically better: 131072 remained
strong, while 262144 and above slowed back down. For general operation, start
with 4096 when you want the most trusted behavior, and try 65536 when you want
maximum CPU throughput on a representative batch for the same workload.

## Layout

```text
src/
  bruteforce/
    bench_cpu/                 CPU SIMD throughput benchmark
    cpu/                       Scalar, SSE, AVX, AVX2, AVX-512 kernels
    cpu_raceway/               Bounded-wave forward dedup (single-host full-2^32 search)
    state_dedup_*_bench/       CPU state-dedup tools
  common/                      RNG, key schedule, target data, shared types
cuda/                          Public-clean CUDA forward verifier
opencl/                        Public-clean OpenCL forward verifier
docs/                          Technical notes and release-candidate results
scripts/                       Validation and export helpers
```

The release-candidate performance summary is in
`docs/forward_release_candidate_20260525.md`. CUDA tuning details are in
`docs/gpu_forward_benchmark_notes.md`.

## Current Routing Guidance

Use CUDA when an NVIDIA GPU is available. Use OpenCL for non-NVIDIA GPUs or
when CUDA is not available. Use the CPU path for portability, validation, and
for the state-dedup route on keys where CPU convergence is favorable.

For CPU dedup, the practical production split is:

1. Run flat/no-origin dedup and screen unique final states.
2. Treat `ALL_ENTRIES_VALID` as a minimum signal, then apply policy-specific
   invalid-flag filters.
3. Re-run only rare passing windows with origin tracking or direct non-dedup
   scanning to recover exact `key,data,flags` records.

This avoids paying origin-list cost across normal traffic while preserving
exact candidate emission when a real window appears.

## Background

*Treasure Master* was released for the NES in 1991 and was tied to an MTV
contest in 1992. The known contest code unlocks one prize world; the game
also contains a second hidden prize world whose unlock code has not been
publicly recovered. The code is represented by an 8-byte `(key, data)` pair
encoded as a 24-character password. This repository focuses on the forward
search needed to verify candidate pairs.

This work builds on earlier public groundwork from
[`micro500/treasure-master-hack`](https://github.com/micro500/treasure-master-hack),
which documented and implemented a brute-force approach to the same second
prize world password problem.
