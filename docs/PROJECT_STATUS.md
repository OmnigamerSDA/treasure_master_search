# Project Status

> Snapshot of the public-release subset. The dev branch is more current;
> this file is updated each public sync.

**Last public sync:** raceway-production refresh, 2026-06-16

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
| Forward CPU      | **Bounded-wave raceway (production)** + AVX/SIMD screen baseline        |
| Forward CUDA     | **Raceway (production)**, per-device `--calibrate-raceway`; screen baseline |
| Forward OpenCL   | **Raceway (production)**, portable AMD/Intel/Apple path; screen baseline |

The **bounded-wave raceway** is the production engine on every backend (best
across throughput AND memory; FN-safe). The flat checksum screen and
compaction/state-dedup paths are kept only as a stability baseline and the
bit-exact parity reference.

## What stays in the dev repo

- Modified SAT solver forks (under separate per-solver repos)
- IPASIR wrappers and solver-binding scripts
- Hypothesis-tracking pipeline (`run_hypothesis_pipeline.py` and friends)
- Daily research notes (byte-permutation rounds, path closures, etc.)
- Heavy benchmark corpora and reverse-record dumps
- Reverse engines and CNF/SAT generation
- Legacy BOINC/worker packages

## Measured forward rates

Production raceway, FN-safe (flat memory set by the cap):

| Backend | Hardware | Raceway throughput |
|---|---|---:|
| CUDA   | RTX 5090 | ~310 M/s typical (population HM); ~224-261 M/s diffuse long pole |
| CUDA   | RTX PRO 6000 Blackwell Max-Q | ~0.8x the 5090 (clock-bound) |
| OpenCL | NVIDIA via OpenCL | ~70% of the CUDA raceway on the same GPU |
| OpenCL | AMD RX 7800 XT | 58.3 M/s W16M cap-span HM; tuned mid-key runs reach ~70-77 M/s |
| CPU    | Ryzen 9 9900X, AVX-512, 24t | 27.49 M/s HM (113.79 collapse / 32.30 mid / 14.41 diffuse) |

Per-device GPU tuning: `tm_cuda --calibrate-raceway` (sweeps span-ILP x cap-bits,
auto-applied). The CPU raceway auto-selects its build/wave/cap via
`src/bruteforce/cpu_raceway/raceway_autoconfig.sh`.

**Research baseline (not production):** a flat AVX/SIMD/CUDA/OpenCL checksum
screen and the state-dedup/compaction benches remain as a stability baseline and
the bit-exact parity reference. The raceway never misses a hit; use the screen
only for an exact dedup count or A-B validation.

## Building

- Forward CPU raceway (production): `make raceway`
- Forward CUDA: `cd cuda && make all`  (raceway + research paths)
- Forward OpenCL: `cd opencl && make all`  (raceway + research paths)
- Research CPU baselines: `make cpu` (bench) / `make cpu-dedup` (dedup benches)

See each subdirectory's Makefile.

## Foundational docs

- `docs/password_conversion_algorithm.md` — codec details
- `docs/decryption_execution_trace_reference.md` — algorithm walkthrough
- `docs/forward_release_candidate_20260525.md` — historical screen/dedup RC note superseded by the raceway
- `docs/gpu_forward_benchmark_notes.md` — historical GPU screen-benchmark notes
