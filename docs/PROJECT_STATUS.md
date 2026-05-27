# Project Status

> Snapshot of the public-release subset. The dev branch is more current;
> this file is updated each public sync.

**Last public sync:** release candidate refresh pending, 2026-05-25

## Where this sits

Treasure Master Bonus World 2 has never been unlocked. The 24-character
password decodes to an 8-byte (key, data) pair; the key is injective in
2^32, and the data axis collapses heavily, so key recovery is sufficient.
This public repository is scoped to the actively maintained forward-search
implementations: CPU SIMD/state-dedup, CUDA, and OpenCL. Reverse constraint
propagation, CNF/SAT tooling, daily research notes, and legacy worker
packages remain in the development repo.

## What is public

| Lane             | Status                                                                 |
|------------------|------------------------------------------------------------------------|
| Forward CPU SIMD | Full ladder, AVX2 native screen, state-dedup primitive                 |
| Forward CUDA     | Offset-stream screen, 132M cand/s on RTX 5090                          |
| Forward OpenCL   | Offset-stream ILP6 path; portable GPU fallback                        |

## What stays in the dev repo

- Modified SAT solver forks (under separate per-solver repos)
- IPASIR wrappers and solver-binding scripts
- Hypothesis-tracking pipeline (`run_hypothesis_pipeline.py` and friends)
- Daily research notes (byte-permutation rounds, path closures, etc.)
- Heavy benchmark corpora and reverse-record dumps
- Reverse engines and CNF/SAT generation
- Legacy BOINC/worker packages

## Measured forward rates

CPU SIMD (Ryzen 9 9900X, GCC `-O3 -march=native`, key `0x2CA5B42D`,
1M candidates; see `docs/forward_release_candidate_20260525.md`):

| Variant            | Rate (M cand/s) |
|--------------------|-----------------:|
| `tm_8` (scalar)    | 0.148            |
| `tm_8` nway        | 0.191            |
| `tm_avx_r128s_8`   | 0.304            |
| `tm_avx_r256s_8`   | 0.260            |
| `tm_avx2_r256s_8`  | 0.523            |
| `tm_avx512_r512s_8`| 0.327            |

`src/bruteforce/bench_cpu` also has an explicit `make pgo` target; on this
host the PGO+hugepage AVX2 screen measured 0.556 M/s/thread on a 4M-candidate
run.

State dedup: `src/common/state_dedup.h`, origin tracking in
`src/common/state_dedup_origins.h`, with benchmarks under
`src/bruteforce/state_dedup_*_bench`. Current flat/no-origin dedup can screen
unique final states for checksum and machine-code flags, then re-run only rare
strict-passing windows with origins to recover exact data values.

The flat-dedup default merge policy is `--first-dedup-maps 1
--dedup-every-maps 4` ("f1k4") — dedup once after MAP1 to capture the ~52%
entry-0 collapse, then merge after every 4 maps (7 merges total). This is
the universal bathtub-bottom policy: it wins on both the r256s and map_8
kernels and at both small (4096) and large (65536) windows from 1 to 24
threads. Aggregate uplift over the previous K=1 default at the documented
production shape (window=4096, threads=12-24) is **+9-14%**.

The investigation behind this choice — geometric per-map collapse curve,
per-stage threshold rule, concurrency-scaling sweep, and the i-cache
pressure fix in `tm_avx2_r256_map_8::_run_maps_fixed` that was load-bearing
for the f1k4 verdict — is in
`docs/hybrid_dedup_architecture_notes_20260527.md`.

CUDA offset-stream screen:

| GPU                                 | Screen rate | Full-key screen time |
|-------------------------------------|-------------|----------------------|
| RTX 5090                            | 132.4-141.2 M/s | ~30-33 s          |
| RTX PRO 6000 Blackwell Max-Q        | 105.1 M/s   | ~41 s               |

OpenCL offset-stream ILP6 screen:

| GPU                                 | Screen rate | Full-key screen time | Short-run wall rate |
|-------------------------------------|-------------|----------------------|---------------------|
| RTX 5090                            | 86.5 M/s    | ~50 s                | 74.9 M/s            |
| RTX PRO 6000 Blackwell Max-Q        | 72.2 M/s    | ~60 s                | ~62 M/s             |

## Building

- Forward CUDA: `cd cuda && make all`
- Forward CPU bench: `cd src/bruteforce/bench_cpu && make all`
- CPU dedup benches: `make cpu-dedup`
- Forward OpenCL: `cd opencl && make all`

See each subdirectory's Makefile.

## Foundational docs

- `docs/password_conversion_algorithm.md` — codec details
- `docs/decryption_execution_trace_reference.md` — algorithm walkthrough
- `docs/forward_release_candidate_20260525.md` — current forward RC note
- `docs/gpu_forward_benchmark_notes.md` — CUDA optimization log
