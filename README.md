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
- **OpenCL** — portable raceway for AMD/Intel/Apple/other non-NVIDIA GPUs.
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
PRODUCER_CAP=1 PCAP_BITS=24 ./cpu_raceway 0x2CA5B42D 0 <threads>
```

CUDA raceway (per-device span-ILP auto-applied after a one-time `--calibrate-raceway`):

```sh
./cuda/tm_cuda --device 0 --key_id 0x2CA5B42D \
  --workunit_size 4294967296 --raceway-direct-wave-continue-batch auto
```

OpenCL raceway (non-NVIDIA devices):

```sh
./opencl/tm_opencl_forward --device 0 --key_id 0x2CA5B42D \
  --range_start 0 --workunit_size 16777216 --raceway-direct-offset
```

## Reference Performance

Production raceway, FN-safe. GPU numbers are key-class dependent
(diffuse/large-frontier keys are the long pole); CPU is a harmonic mean across
a representative key mix.

| Path | Hardware | Raceway throughput |
|---|---|---:|
| CUDA | RTX 5090 | ~310 M/s typical (population HM); ~224–261 M/s on the diffuse long pole |
| CUDA | RTX PRO 6000 Blackwell Max-Q | ~0.8× the 5090 (clock-bound) |
| OpenCL | NVIDIA (same GPU) | ~70% of the CUDA raceway |
| OpenCL | AMD RX 7800 XT (RDNA3) | 58.3 M/s W16M cap-span HM; tuned single-key runs reach ~70–77 M/s on mid keys and ~45 M/s on diffuse keys |
| OpenCL | AMD Ryzen iGPU (1 CU) | ~2.2–2.8 M/s — runs the full pipeline (floor / CI-smoke) |
| CPU (AVX-512) | Ryzen 9 9900X, 24 threads | 27.49 M/s HM (113.79 collapse / 32.30 mid / 14.41 diffuse) |
| CPU (AVX2) | host without AVX-512 | ~0.71–0.76× the AVX-512 raceway on the 9900X reference host |

Per-device GPU tuning: `tm_cuda --calibrate-raceway` sweeps span-ILP × cap-bits
and records the result; production raceway runs auto-apply it. The research
screen/compaction rates and methodology are in
`docs/gpu_forward_benchmark_notes.md`.

## Layout

```text
src/
  bruteforce/
    cpu_raceway/               PRODUCTION CPU engine — bounded-wave forward dedup
    cpu/                       Scalar, SSE, AVX, AVX2, AVX-512 kernels
    bench_cpu/                 research: CPU SIMD throughput benchmark
    state_dedup_*_bench/       research: CPU state-dedup characterization tools
  common/                      RNG, key schedule, target data, shared types
cuda/                          CUDA forward verifier (raceway + research paths)
opencl/                        OpenCL forward verifier (raceway + research paths)
docs/                          Technical notes and results
```

## Choosing a path

- **NVIDIA GPU present:** CUDA raceway (`--raceway-direct-wave-continue-batch auto`).
- **Non-NVIDIA GPU:** OpenCL raceway (`--raceway-direct-offset`).
- **No GPU / distributed CPU hosts:** the CPU raceway (`cpu_raceway/`), auto-configured per host.
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
