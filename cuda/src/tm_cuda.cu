// Treasure Master — CUDA kernel compilation unit.
//
// This is the single NVCC compilation unit for all GPU kernels.
// Device code is organised across three headers (included below) so that each
// logical layer can be read and reviewed independently:
//
//   tm_cuda_primitives.cuh  — __constant__ tables, per-lane helpers, algorithm
//                             implementations, schedule runners, screen_candidate,
//                             map-RNG infrastructure, and HLL support
//
//   tm_cuda_screen.cuh      — checksum-screen, HLL, and materialize kernels
//                             (the production forward-search path and its
//                             diagnostic / experimental variants)
//
//   tm_cuda_dedup.cuh       — RESEARCH/A-B: state-dedup POC and on-GPU VRAM compaction
//                             kernels (Phase 1–2 within-block dedup and the 2026-05-29
//                             span-based compaction architecture)
//
//   tm_cuda_raceway.cuh     — PRODUCTION engine: bounded-wave raceway (cap-span +
//                             wave-local compaction) kernels and fixed-capacity caps
//
// PRODUCTION ENGINE = raceway (tm_cuda_raceway.cuh): best across BOTH throughput and
// memory, the default for any system (2026-06-16). The flat ilp6 offset screen
// (tm_checksum_screen_offset_store_ilp6_preids_cuda, tm_cuda_screen.cuh) and the on-GPU
// compaction engine (tm_cuda_dedup.cuh) are now RESEARCH / A-B paths, not the default.
// Per-device tuning:
//   test_cuda --calibrate-raceway   PRODUCTION: sweep raceway span-ILP x cap-bits, write
//                                   engine=raceway (auto-applied by production raceway runs)
//   test_cuda --calibrate           research: screen-vs-compaction geometry/tile sweep
//
// Measured full 2^32 sweep, wall (key 0x2CA5B42D, 500 W, 2026-05-30):
//   RTX 5090:            ilp6 screen ~130 M/s (~33 s/key);  compaction ~146 M/s (~29.5 s)
//   RTX PRO 6000 Max-Q:  ilp6 screen ~104 M/s (~41 s/key);  compaction ~115 M/s (~37.4 s)
//   (compaction speedup is key/region-dependent: ~on-par on typical keys, up to ~1.75x high-R.)
//
// Build (default):       make                          (CUDA_ARCHS=120, Blackwell sm_120 only)
// Build (distribution):  make CUDA_ARCHS="86 89 120"    (Ampere+Ada+Blackwell native SASS + PTX;
//                        a 120-only fatbin will NOT load on Ada/Ampere — PTX can't JIT downward)

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cstdint>

#include "tm_cuda_primitives.cuh"
#include "tm_cuda_screen.cuh"
#include "tm_cuda_dedup.cuh"
#include "tm_cuda_raceway.cuh"
