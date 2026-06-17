# Forward Release Candidate Notes (2026-05-25)

> **HISTORICAL SNAPSHOT (2026-05-25) — SUPERSEDED.** This pre-dates the bounded-wave
> raceway, which is now the production engine on every backend (best across throughput
> AND memory; FN-safe). The "production screen" / offset-stream and state-dedup paths
> described below are now the **research / baseline + parity reference**, not the default.
> For current status and rates see `docs/PROJECT_STATUS.md`; for operating the engines see
> the CPU/CUDA/OpenCL READMEs. The measurements below remain valid as a
> dated record of the screen/dedup work.

This note captures the forward-search state after the CUDA offset-stream work
and the follow-up CPU forward profiling pass. It is intended as the handoff
record for a release-candidate snapshot that can later be set aside in a
separate public repo.

Host for CPU measurements: AMD Ryzen 9 9900X, 12 cores / 24 threads, 64 MB
L3, GCC with `-O3 -march=native -mtune=native`.

## CPU Forward Profiling

Update from the 2026-05-26 AVX2 map-kernel followup: for plain non-dedup
`run_bruteforce_data` screening, `tm_avx2_r256_map_8` should be included in
comparisons and is the current AVX2 contender after the dispatch-switch fix.
The `0.67+ M/s/thread` rows below are state-dedup/windowed measurements, not a
separate plain forward implementation.

Final 2026-05-26 cleanup note: `tm_avx2_r256_map_8` is the recommended
plain/non-dedup CPU sweep path. The state-dedup winner remains the flat AVX2
r256s path; the map kernel was exposed in the dedup matrix for comparison but
did not beat r256s there. The remaining plausible CPU-side improvement with
more than low-single-digit upside is boundary/interleaving work, which is
invasive enough that it should not block this release candidate.

Hardware-counter profiling was run with `perf_event_paranoid=1`, using
`perf stat -d` and sampled `perf record` on the AVX2 native screen path.

The concrete CPU win found in this pass was that `tm_avx2_r256s_8` had the
fastest forward core, but did not have the native SIMD checksum/decrypt
screening override present in the older AVX 256-bit implementation. The
benchmark therefore measured AVX2 through a generic `fetch_data` +
scalar-checksum path.

Promoted change:

- `tm_avx2_r256s_8` now implements native SIMD decrypt, checksum, checksum
  fetch, and `run_bruteforce_data`.
- `bench_cpu` now uses native screening for SIMD implementations that expose
  `run_bruteforce_data`.

Single-thread benchmark, key `0x2CA5B42D`, range start `0`,
`workunit_size=1048576`, `warmup=262144`:

| Implementation | M candidates/s/thread |
|---|---:|
| `tm_8` scalar | 0.149 |
| `tm_8` nway | 0.190 |
| `tm_avx_r128s_8` | 0.305 |
| `tm_avx_r256s_8` | 0.261 |
| `tm_avx2_r256s_8` | **0.523** |
| `tm_avx512_r512s_8` | 0.327 |

Before this CPU pass, the same AVX2 bench path measured about 0.367 M/s on
the same 1M range. Native AVX2 screening plus Linux hugepage advice for large
RNG tables lifts the non-dedup CPU screener to about **1.43x** over that
starting point on this host.

AVX2 thread scaling, same key/range family:

| Threads | Total M/s | M/s/thread |
|---:|---:|---:|
| 1 | 0.465 | 0.465 |
| 2 | 0.924 | 0.462 |
| 4 | 1.773 | 0.443 |
| 6 | 2.625 | 0.437 |
| 8 | 3.418 | 0.427 |
| 12 | 4.123 | 0.344 |
| 24 | 6.549 | 0.273 |

Perf summary for `tm_avx2_r256s_8`, 4M timed candidates:

| Counter | Value |
|---|---:|
| Rate | 0.518-0.523 M/s/thread |
| IPC | 1.47 |
| Branch misses | 7.33% |
| L1-dcache load misses | 41.1% |
| dTLB load misses | 0.23-0.52% |
| Generic cache misses | 4.14% |

Interpretation: the hot path is the inlined AVX2 `run_bruteforce_data`
loop. It is not DRAM-bound; outer-cache misses are low, while L1 miss rate is
high from the RNG table stream. Large-table hugepage advice is useful because
it removes most dTLB pressure; remaining stalls are a mix of RNG-table
latency, dynamic algorithm dispatch, and dependent SIMD work inside the
algorithm transforms.

Individual follow-up experiments:

| Experiment | Result | Decision |
|---|---:|---|
| PGO build target | 0.563 M/s/thread on range 0; 0.349 M/s/thread on high-hit range; 0.556 M/s/thread on 4M range | Positive; available via `make pgo` in `src/bruteforce/bench_cpu` |
| Linux `MADV_HUGEPAGE` for large aligned allocations | 0.523 M/s/thread; dTLB miss rate fell from 2.92% to 0.52% | Promoted |
| Direct AVX2 checksum-byte extraction | 0.459 M/s/thread on 4M run | Negative/noisy; reverted |
| AVX2 map-mode port + switch dispatch | about 0.55 M/s/thread on plain 4M sweeps | Promoted for plain non-dedup CPU sweeps |
| Dedup every-K-maps merge knob | K=2 modestly helped large-window/high-concurrency runs; K>=4 was negative | Keep as `--dedup-every-maps`, default `1` |
| Dedup prefetch and alg0/alg6 bit packing | neutral or slower | Reverted from release code |

Correctness checks performed:

- AVX2 state and checksum parity against scalar on representative data values,
  including the known carnival data `0xF73A2612`.
- Survivor-count parity against scalar on two 1M ranges:
  - range `0`: scalar 4, AVX2 4
  - range `0xF1468000`: scalar 43, AVX2 43
- `state_dedup_speedup_bench` smoke over four keys and windows 16/256:
  all parity cells matched.

## State Dedup Status

The dedup primitive remains the main CPU-side structural win:

- source: `src/common/state_dedup.h`
- origin-tracking source: `src/common/state_dedup_origins.h`
- benchmark: `src/bruteforce/state_dedup_speedup_bench`
- origin benchmark: `src/bruteforce/state_dedup_origin_bench`
- shape: open-addressed flat hash keyed by internal 128-byte state, with
  multiplicity-preserving frontier merge at each schedule boundary
- origin shape: each unique final state carries a linked list of data offsets,
  so a production screen can test one state and emit all producing data values

Follow-up profiling found one obvious AVX2 dedup issue: `run_one_map` built a
temporary one-entry `key_schedule` for every frontier state. Dedup calls this
at every boundary, so this was replaced with a direct single-entry AVX2 path.

Measured effective single-thread dedup rates after the direct `run_one_map`
fix:

| Key | Window | Baseline SIMD M/s | Dedup SIMD M/s | Dedup speedup |
|---|---:|---:|---:|---:|
| `0x2CA5B42D` | 4096 | 0.444 | 1.089 | 2.45x |
| `0xF6C9E358` | 4096 | 0.320 | 1.206 | 3.77x |
| `0x12345678` | 4096 | 0.206 | 0.781 | 3.79x |
| `0xDEADBEEF` | 4096 | 0.278 | 1.681 | 6.05x |

Origin-tracking dedup, window 4096, all origins verified:

| Key | Origin dedup M/s/thread | Final unique states | Max origins/state |
|---|---:|---:|---:|
| `0x2CA5B42D` | 1.047 | 761 | 280 |
| `0xF6C9E358` | 1.170 | 544 | 88 |
| `0x12345678` | 0.788 | 1023 | 128 |
| `0xDEADBEEF` | 1.643 | 465 | 64 |

Perf summary for origin-tracking dedup on `0x2CA5B42D`, window 4096,
300 repeats:

| Counter | Value |
|---|---:|
| Rate | 0.979 M/s/thread |
| IPC | 1.43 |
| Branch misses | 9.15% |
| L1-dcache load misses | 3.73% |

Sampled `perf record` put about 67% of cycles in AVX2 `run_one_map` and
about 22% in the origin-dedup driver. That supports the current direction:
dedup is now real production machinery rather than just a measurement POC,
but its next gains would likely come from batching/fusing per-boundary map
execution rather than hash-table micro-tuning alone.

Rejected follow-up: a CPU analogue of the GPU offset stream was tested as a
temporary AVX2 origin-dedup path. Each map entry has 153 reachable RNG seeds
(`f^offset(seed)` for offsets formed by 16 choices of 0, 1, or 128), so a
compact cache of all AVX2 RNG rows would still be about 68.5 KiB per entry
before lookup metadata. A narrower version kept the universal SIMD row tables
and precomputed only the seed-by-offset stream plus selector nibbles. It
parity-verified, but direction was mixed: at window 4096 it measured about
`+2%` to `+6%`, while larger windows ranged from about `+4%` on
`0x12345678` to about `-3%` on `0xDEADBEEF`. The perf split stayed effectively
unchanged at about 70% in `run_one_map`, so this was not kept for the release
candidate.

Prior 477-key characterization on the same host showed median combined
SIMD+dedup speedups of about 5x to 6x versus scalar baseline at practical
window sizes. This pass did not find a larger post-dedup CPU forward
opportunity than the native AVX2 screening gap above.

Representative-key smoke was moved away from hand-picked examples. The
pipeline database
`common/results/bonus2_loader_policy_longrun_20260509/pipeline.sqlite`
contains 6232 CNF SAT rows and 5598 distinct `sat_key_hex` values; these are
all CUDA false positives, but they are the right production-routing sample
because they are the keys CNF sends to forward verification.

Tooling:

- `scripts/export_cnf_fp_keys.py` exports distinct `sat_task.sat_key_hex`
  values from a pipeline DB to a key CSV.
- `src/common/key_file.h` lets dedup benches read one-key-per-line files and
  collected CSVs with a `key_hex` column.

Representative AVX2 flat-dedup smoke, 128 random CNF-FP keys, parity matched:

| Window | Median M/s/thread | Median dedup speedup | Median final unique/window | Best-window count |
|---:|---:|---:|---:|---:|
| 256 | 0.515 | 1.846x | 0.414 | 24 |
| 1024 | 0.604 | 2.181x | 0.329 | 26 |
| 4096 | 0.624 | 2.340x | 0.318 | 78 |

Larger-window AVX2 flat-dedup smoke, 64 random CNF-FP keys, parity matched:

| Window | Median M/s/thread | Median dedup speedup | Median final unique/window | Best-window count |
|---:|---:|---:|---:|---:|
| 4096 | 0.683 | 2.435x | 0.284 | 17 |
| 8192 | 0.705 | 2.412x | 0.279 | 7 |
| 16384 | 0.706 | 2.564x | 0.267 | 40 |

The flat dedup driver also exposes `--dedup-every-maps N`, which merges the
frontier every N schedule entries instead of after every entry. Testing showed
K=2 can give a small throughput bump on larger windows at higher concurrency,
but K>=4 loses too much collision collapse. Keep the default at K=1 and treat
K=2 as a workload/hardware tuning option.

Origin-tracking smoke on 16 random CNF-FP keys, all origins verified:

| Window | Median M/s/thread | Median final unique/window | Best-window count |
|---:|---:|---:|---:|
| 4096 | 0.774 | 0.236 | 5 |
| 8192 | 0.797 | 0.230 | 3 |
| 16384 | 0.673 | 0.262 | 8 |

Production-shaped origin-dedup concurrency smoke, 64 random CNF-FP keys,
window 4096, 20 repeats/key, origin verification disabled for timing:

| Threads | Total M/s | M/s/thread |
|---:|---:|---:|
| 1 | 0.596 | 0.596 |
| 2 | 1.164 | 0.582 |
| 4 | 2.254 | 0.564 |
| 6 | 3.271 | 0.545 |
| 8 | 4.202 | 0.525 |
| 12 | 5.748 | 0.479 |
| 16 | 6.667 | 0.417 |
| 24 | 8.153 | 0.340 |

On the same 64-key sample at window 4096, flat/no-origin dedup measured
0.625 aggregate M/s/thread and origin-tracking dedup measured 0.598 aggregate
M/s/thread. That puts origin tracing overhead at about 4.3% for the
representative sample. A sampled single-thread origin run put 83.7% of
cycles in AVX2 `run_one_map` and 15.2% in the dedup/origin driver, so origin
bookkeeping is visible but not currently a large enough gap to justify a more
complex origin representation for the release candidate.

Flat/no-origin dedup can still screen unique final states for the full
machine-code flag suite. What it lacks is only the origin list needed to emit
the exact producing data values. `state_dedup_screen_bench` measures that
route directly: flat dedup, checksum screen, and `check_machine_code` over
unique final states, while preserving multiplicity counts.

Representative flat/no-origin screen sweep, 512 random CNF-FP keys,
67,108,864 original candidates per row, 12 threads:

| Window | Windows/key | Total M/s | M/s/thread | Unique states | Checksum unique | Checksum origins | All-entries unique | All-entries origins | All-entries windows |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 32 | 6.561 | 0.547 | 22,338,135 | 684 | 1,581 | 2 | 3 | 2 |
| 8192 | 16 | 6.869 | 0.572 | 21,038,540 | 638 | 1,581 | 2 | 3 | 2 |
| 16384 | 8 | 7.334 | 0.611 | 19,321,421 | 598 | 1,581 | 2 | 3 | 2 |

Superwindow no-origin screen sweep, 64 random CNF-FP keys, same first
1,048,576 data values per key for every row, 67,108,864 original candidates
per row, 12 threads:

| Window | Windows/key | Total M/s | M/s/thread | Unique states | Unique/candidate | Checksum unique | Checksum windows | All-entries windows |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 256 | 6.905 | 0.575 | 21,297,469 | 0.317 | 722 | 683 | 0 |
| 16384 | 64 | 7.576 | 0.631 | 18,798,581 | 0.280 | 628 | 554 | 0 |
| 65536 | 16 | 9.450 | 0.787 | 13,173,192 | 0.196 | 442 | 319 | 0 |
| 131072 | 8 | 8.997 | 0.750 | 12,081,964 | 0.180 | 396 | 233 | 0 |
| 262144 | 4 | 8.320 | 0.693 | 11,485,476 | 0.171 | 374 | 157 | 0 |
| 524288 | 2 | 7.670 | 0.639 | 11,185,543 | 0.167 | 362 | 97 | 0 |
| 1048576 | 1 | 6.782 | 0.565 | 10,917,315 | 0.163 | 351 | 55 | 0 |

This shows real cross-window collapse: the same candidate span drops from
31.7% unique final states at 4096 to 19.6% at 65536 and 16.3% at 1048576.
Throughput peaks around 65536 in this no-origin run; beyond that, larger
tables continue reducing duplicate state work but lose enough locality and
scheduling granularity to fall back toward the default-window rate.

Longer default-window screen sweep, 512 random CNF-FP keys, window 4096,
128 windows/key, 268,435,456 original candidates, 12 threads:

| Metric | Value |
|---|---:|
| Total rate | 6.373 M/s |
| Rate/thread | 0.531 M/s/thread |
| Unique final states | 89,996,048 |
| Checksum unique states | 2,835 |
| Checksum origins | 8,177 |
| Checksum windows | 2,755 / 65,536 |
| First-entry unique states | 423 |
| First-entry origins | 1,289 |
| All-entries unique states | 3 |
| All-entries origins | 4 |
| All-entries windows | 3 / 65,536 |
| Strict invalid mask | `0xe0` (`UNOFFICIAL_NOPS | ILLEGAL_OPCODES | JAM`) |
| Strict all-entries unique states | 0 |
| Strict all-entries origins | 0 |
| Strict all-entries windows | 0 / 65,536 |

Interpretation: a no-origin first pass is appropriate for the bulk route.
Checksum passes are common enough that replaying all checksum windows would be
noticeable. Raw `ALL_ENTRIES_VALID` is only a minimum signal; production
filtering should also reject policy-disqualifying flags such as unofficial
NOPs, illegal opcodes, and JAM. With that default strict mask, the longer
default-window sweep had no windows needing origin recovery. That makes a
hybrid route practical: flat/no-origin dedup for normal processing, then
re-run only rare strict-passing windows with origin tracking or a direct
non-dedup scan to recover exact `key,data,flags` records.

Interpretation: dynamic window sizing is worth considering, but not as a
release-candidate default. `4096` remains the conservative trusted default:
it is good across the representative distribution and keeps memory, origin
lists, latency, and scheduling granularity bounded. The no-origin superwindow
probe found a stronger flat-screen knee around `65536`, so a future adaptive
router should first run a cheap probe window, estimate collapse rate and
origin fanout, and only promote a key to a larger window when the probe
predicts a clear margin over the default. Before changing production defaults,
the `65536` candidate should be retested on a broader key sample and with the
origin-replay path included for strict-passing windows.

## GPU Forward Status

The CUDA production screen path is now the offset-stream implementation:

- kernel: `tm_checksum_screen_offset_store_ilp6_preids_cuda`
- host flag: `--screen-offsets` (ILP defaults to 6; public binary also accepts `--ilp 4|6|8`)
- key/schedule preprocessing buffer: 21,676,032 bytes
- source: `src/bruteforce/test_cuda/main.cpp`,
  `src/bruteforce/test_cuda/tm_cuda_screen.cuh` (split from `tm_cuda.cu` 2026-05-30)

Representative rates (2026-06-01, ILP6, production fatbin = CUDA 13.3 + compaction ACF):

| GPU | Screen rate | Compaction (high-R key) |
|---|---:|---:|
| RTX 5090 | ~139–144 M/s | ~256–503 M/s (R-dependent) |
| RTX PRO 6000 Blackwell Max-Q | ~104–106 M/s | ~115–175 M/s (R-dependent) |

The production fatbin is compiled with `--apply-controls tools/ciq_results/compaction_best_acf.bin`
(CUDA 13.3).  The Makefile still defaults to CUDA 13.2 (safe fallback); re-apply the ACF after
any `make` rebuild.  See `docs/compileiq_acf_tuning_20260601.md` for the full ACF analysis.

The on-GPU survivor-compaction path (`--compaction-sweep`) is the recommended production sweep
mode for keys with R ≥ 1.3 (auto-selected by `tm_compaction.conf` calibration).  The flat ilp6
screen remains the fallback for low-R keys and hardware without calibration data.

## Release Candidate Boundary

Keep in the RC:

- CPU forward SIMD ladder and `bench_cpu`
- `state_dedup.h` and `state_dedup_speedup_bench`
- CUDA offset-stream screen path and existing benchmark variants
- OpenCL forward path as the heterogeneous fallback
- public sync tooling: `.releaseinclude`, `.releaseexclude`,
  `scripts/sync_public.sh`
- foundational docs, fixtures, and password codec

Exclude or defer:

- solver forks and active solver-binding scripts
- large profiler output directories and benchmark corpora
- CUDA/OpenCL build artifacts under `release_staging/`
- GPU dedup as an automatic default route

Before cutting the separate repo:

1. Run `scripts/sync_public.sh` as a dry run and review missing allowlist
   entries.
2. Build `src/bruteforce/bench_cpu`, `src/bruteforce/test_cuda`, and the
   OpenCL target on the destination checkout.
3. Re-run the AVX2 survivor parity smoke above and CUDA parity smoke:
   `./test_cuda --parity 64 --workunit_size 64 --batch_size 64`.
4. Remove generated binaries and profiler exports before the initial public
   commit.
