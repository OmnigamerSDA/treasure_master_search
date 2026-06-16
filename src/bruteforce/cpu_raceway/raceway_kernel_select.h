#ifndef RACEWAY_KERNEL_SELECT_H
#define RACEWAY_KERNEL_SELECT_H
// raceway_kernel_select.h — ISA / kernel-variant selection for cpu_raceway.
//
// ORDERED INCLUDE of the cpu_raceway translation unit (not a standalone header):
// include it right after state_dedup.h / key_schedule.h / rng_obj.h. Selects the
// forward kernel (`Kern`) and the interleave width (`g_ilp`) from the build macros
// UNIVERSAL / AVX2_NATMAP / RACEWAY_NM_ILP. See cpu_raceway.cpp header block and
// docs/cpu_raceway_poc_results_20260614.md §23-§24 for the width rationale.

#ifdef UNIVERSAL
#include "tm_avx512_r512s_8.h"
using Kern = tm_avx512_r512s_8;     // universal: x12 (interleave sweet spot, branched) or x8 (branchless)
static int g_ilp = 12;              // RUNTIME on universal: 12 for x12-branched, 8 for x8-branchless
#elif defined(AVX2_NATMAP)
#include "tm_avx2_r256_map_8.h"
using Kern = tm_avx2_r256_map_8;    // AVX2 natmap port (older-ISA distribution; no AVX-512 required)
// AVX2 has 16 YMM (4/state); the kernel's x8-signature methods process all 8 harness lanes as
// AVX2_RACEWAY_W-wide register-resident sweeps (default x2 — with the branchless deep dispatch x2 ties x4
// with zero spills; -DAVX2_RACEWAY_W=4 for the full-file point). g_ilp is fixed at 8 so the originate/deep
// paths and RACEWAY_NM_DEEP if-constexpr instantiate exactly the x8 shape. Deep dispatch defaults to
// blmerge (branchless) — DEEP_DISP=branched falls back to the 8-way switch (~23% mispredict).
static constexpr int g_ilp = 8;
#else
#include "tm_avx512_r512_map_8.h"
using Kern = tm_avx512_r512_map_8;  // natmap x8 sweet spot
// Interleave width: COMPILE-TIME constant (-DRACEWAY_NM_ILP=N) so each build keeps a deterministic drain-frame
// codegen (the documented -O2/-O3 fragility constraint — a runtime g_ilp would perturb the frame). if-constexpr
// in RACEWAY_NM_DEEP emits exactly one call shape.
// PRODUCTION DEFAULT = 10 (W6, 2026-06-15): x10-blmerge is the spill-free register-file ceiling (32/32 ZMM,
// 0 spills) and the population sweet spot (+2.9% harmonic-mean over x8 across a 13-key sample; wins closer/mid,
// neutral on the binding diffuse key). x8 reachable via -DRACEWAY_NM_ILP=8; x12 (-DRACEWAY_NM_ILP=12) builds via
// the mask-rematerialization lever but is net-negative at population scale (spills). x10/x12 run blmerge when
// g_deep_disp==2 (default), branched otherwise; x8 uses the W5 DEEP_DISP path.
#ifdef RACEWAY_NM_ILP
static constexpr int g_ilp = RACEWAY_NM_ILP;
#else
static constexpr int g_ilp = 10;
#endif
#endif

#endif // RACEWAY_KERNEL_SELECT_H
