# Treasure Master Forward Search

Forward-search implementations for the hidden Bonus World 2 unlock code in
the NES game *Treasure Master*.

All paths solve the same task: for a fixed 32-bit key, sweep the `2^32` data
space, run the game's forward transform (key schedule → 27 maps), and screen
the final state for the target worlds.

## Production engine: the bounded-wave raceway

The **production forward engine is the bounded-wave raceway** — the default on
both CPU and GPU, best across **both throughput and memory**. It originates the
post-MAP1 frontier into an LLC-/VRAM-sized *wave* and drains it through
persistent per-boundary fingerprint caps; it is **FN-safe** (finds every hit)
and its memory is flat in the window size (set by the cap, not the sweep).

The flat **checksum screen** and **on-GPU/in-RAM dedup/compaction** paths are
retained as **research / A-B comparisons** and as the bit-exact parity
reference, but are no longer the default.

- **CUDA** — fastest; the raceway on NVIDIA GPUs.
- **OpenCL** — portable raceway for AMD/Intel/Apple/other non-NVIDIA GPUs (~264 M represented/s on an RTX 5090, 2026-06-18).
- **CPU** — the raceway for hosts without a GPU (AVX-512 / AVX2), auto-configured per host.

## Quick Build

```sh
make raceway    # PRODUCTION bounded-wave raceway (CPU)
make cuda       # CUDA forward verifier (raceway + research paths)
make opencl     # OpenCL forward verifier (raceway + research paths)
make cpu        # research: native CPU AVX/SIMD throughput benchmark
make cpu-dedup  # research: CPU state-dedup characterization tools
```

Each implementation also has its own Makefile: CPU under `src/bruteforce/`,
CUDA under `cuda/`, OpenCL under `opencl/`.

## First Runs

CPU raceway (auto-selects the AVX-512/AVX2 build + wave/cap for this host):

```sh
cd src/bruteforce/cpu_raceway && ./raceway_autoconfig.sh --build
./cpu_raceway 0x2CA5B42D 16777216 <threads>     # window 0 = full 2^32 (needs PRODUCER_CAP=1)
```

CUDA raceway (per-device span-ILP auto-applied after a one-time `--calibrate-raceway`):

```sh
./cuda/tm_cuda --device 0 --key_id 0x2CA5B42D \
  --workunit_size 4294967296 --raceway-direct-wave-continue-batch auto
```

OpenCL raceway (non-NVIDIA devices):

```sh
./opencl/tm_opencl_forward --device 0 --key_id 0x2CA5B42D \
  --workunit_size 4294967296 --raceway-wave-cap-mark
```

Supported production raceway launches run a per-key MAP1 certifier by default.
When the key has certified-shed data bits, the launcher scans only the logical
support axis and fixes the shed bits before MAP1, avoiding the excluded input
work entirely. Disable it with `--no-precert` on CUDA/OpenCL or `PRECERT=0` on
the CPU raceway. If a key has no certified bits, the path is a no-op.

## Clearing a keyspace (the operator)

`scripts/cert_tier_ops.py` is a durable, resumable orchestrator for clearing a
whole key/data space **tier by tier**: it leases bounded batches from a SQLite
queue, runs one worker per GPU, checkpoints completed keys, forwards the
other-world/dual checksum survivors to the CPU receiver
(`inspect_bonus2_survivors`) for machine-code verification, and records genuine
hits as **alerts**. The operational improvements are on by default: hit
verification runs in the **background** (overlapping the next batch's GPU
compute), batch relaunch is **event-driven** (no polling), and an optional
persistent-worker **daemon** feeds successive batches to one resident worker per
device (no per-batch CUDA restart).

### Try it now: the cert16 sample

A ready-to-run 64-key sample queue ships under `examples/cert16_sample/`
(certified tier 16, so each key forwards a `2^16` window — the whole sample
clears in seconds). Use it to validate a fresh build end to end:

```sh
make cuda        # builds cuda/tm_cuda (the GPU worker; supports --raceway-daemon)
make receiver    # builds src/bruteforce/inspect_bonus2_survivors (the CPU verifier)

python3 scripts/cert_tier_ops.py run \
  --db examples/cert16_sample/cert16_queue.db --out-dir /tmp/cert16_run \
  --engine raceway --devices 0 --loops 0 \
  --binary cuda/tm_cuda --inspect-binary src/bruteforce/inspect_bonus2_survivors
```

A clean run marks all **64 keys done with 0 alerts** (cert16 is already cleared —
no valid password). Check state any time with
`cert_tier_ops.py status --db examples/cert16_sample/cert16_queue.db`. The
shipped DB stays pending until you run it; copy it first if you want to re-run.

### Set up your own queue

The certifier (`src/common/map1_certifier.h`) proves, per key, how many data
bits are *shed* at MAP1 (identical final state regardless of those bits), so a
key with `c` certified bits needs only `2^(32-c)` representatives forwarded.
Clearing is organized by that certified-bit **tier**.

1. **`rank_array.bin`** — one `uint8` per 32-bit key holding its certified
   shed-bit count. This is a one-time full-keyspace certifier scan and is the
   input to sharding (it is not shipped — the 4 GiB array is regenerated from the
   certifier for the target problem).
2. **Shards** — `build_cert_tier_shards.py` scans `rank_array.bin` and emits
   balanced per-device key shards for a tier:
   ```sh
   python3 scripts/build_cert_tier_shards.py --rank-array <rank_array.bin> \
     --min-bits 6 --max-bits-exclusive 8 --shards 2 --format bin32u8 \
     --out-dir common/results/my_tier_sweep
   ```
3. **Init the queue** — one shard per device:
   ```sh
   python3 scripts/cert_tier_ops.py init \
     --db common/results/my_tier_ops/queue.sqlite \
     --shard-dir common/results/my_tier_sweep --min-bits 6 --devices 0,1
   ```
4. **Run + watch** — either the interactive one-view console, or two shells:
   ```sh
   # interactive: launch the run AND show the live dashboard (p=pause r=resume q=quit d=detach)
   python3 scripts/cert_tier_ops.py operate \
     --db common/results/my_tier_ops/queue.sqlite --out-dir common/results/my_tier_ops \
     --engine raceway --devices 0,1 --loops 0 \
     --binary cuda/tm_cuda --inspect-binary src/bruteforce/inspect_bonus2_survivors
   # or: `run` in one shell (worker loop) + `monitor` in another (read-only dashboard)
   ```

Per-tier wave geometry (window / drain boundaries / cap-ways) is problem- and
hardware-specific, so nothing is baked in — pass your own `--raceway-*` flags, or
plug in a validated per-tier table via a `cert_tier_geometry_override.py` hook
(see `_load_tier_geometry_override`). For the big low-cert tiers add `--daemon`
(one resident worker per device). `--sync-verify` forces blocking verification;
`request-stop` / `clear-stop` pause and resume; `reset-running` recovers leases
after a hard crash. `--resume` re-launches with the most-recently-used
`run`/`operate` flags (saved under `~/.cert_tier_ops/`), so a recurring launch
needs no flags — e.g. `cert_tier_ops.py operate --resume`; any flag you pass
alongside it overrides the saved value. `cert_tier_ops.py --help` and each
subcommand's `--help` document the full surface.

## Reference Performance

Raceway, full-key `2^32` sweep, FN-safe. GPU numbers are key-class dependent
(diffuse/large-frontier keys are the long pole); CPU is a harmonic mean across
a representative key mix.

| Path | Hardware | Raceway throughput |
|---|---|---:|
| CUDA | RTX 5090 | ~480 M represented/s default-precert HM (8-key W256M); ~290 M/s diffuse HM (2026-07-15) |
| CUDA | RTX PRO 6000 Blackwell Max-Q | ~344 M represented/s (same 8-key W256M, clock-bound; 2026-07-15) |
| OpenCL | NVIDIA RTX 5090 | ~264 M represented/s default-precert HM (same 8-key W256M; 2026-06-18) |
| OpenCL | AMD RX 7800 XT (RDNA3) | ~70 M/s (updated OpenCL raceway) |
| OpenCL | AMD Ryzen iGPU (1 CU) | runs the full pipeline (portability floor / CI-smoke) |
| CPU (AVX-512) | Ryzen 9 9900X, 24 threads | ~22.9 M represented/s default-precert HM (same 8-key W256M); ~13.5 M/s diffuse HM (2026-06-18) |
| CPU (AVX2) | host without AVX-512 | ~0.7× the AVX-512 raceway |

Per-device GPU tuning: `tm_cuda --calibrate-raceway` sweeps span-ILP × cap-bits
and records the result; production raceway runs auto-apply it. The research
screen/compaction rates and methodology are in
`docs/gpu_forward_benchmark_notes.md`.

Default-precert represented-throughput details are in
`docs/raceway_precert_hm_20260618.md`.

## Layout

```text
src/
  bruteforce/
    cpu_raceway/               PRODUCTION CPU engine — bounded-wave forward dedup
    cpu/                       Scalar, SSE, AVX, AVX2, AVX-512 kernels
    bench_cpu/                 research: CPU SIMD throughput benchmark
    state_dedup_*_bench/       research: CPU state-dedup characterization tools
  common/                      RNG, key schedule, target data, shared types
  fpga/                        FPGA forward implementation — VHDL map/screen engines,
                               testbenches, and out-of-context synthesis scripts
cuda/                          CUDA forward verifier (raceway + research paths)
opencl/                        OpenCL forward verifier (raceway + research paths)
docs/                          Technical notes and results
```

## Choosing a path

- **NVIDIA GPU present:** CUDA raceway (`--raceway-direct-wave-continue-batch auto`).
- **Non-NVIDIA GPU:** OpenCL raceway (`--raceway-wave-cap-mark`).
- **No GPU / distributed CPU hosts:** the CPU raceway (`cpu_raceway/`), auto-configured per host.
- **Custom hardware / FPGA:** the VHDL map and screen engines under `src/fpga/` (map/RNG
  pipeline + screen, with self-checking testbenches and OOC synthesis scripts).
- **Need a bit-exact dedup count or per-value accounting:** the screen/compaction
  (BFS) paths, which the raceway is validated against. The raceway itself never
  misses a hit; it just isn't an exact-dedup count.

## Background

*Treasure Master* was released for the NES in 1991 and was tied to an MTV
contest in 1992. The known contest code unlocks one prize world; the game also
contains a second hidden prize world whose unlock code has not been publicly
recovered. The code is an 8-byte `(key, data)` pair encoded as a 24-character
password. This repository focuses on the forward search needed to verify
candidate pairs.

This work builds on earlier public groundwork from
[`micro500/treasure-master-hack`](https://github.com/micro500/treasure-master-hack),
which documented and implemented a brute-force approach to the same second
prize world password problem.
