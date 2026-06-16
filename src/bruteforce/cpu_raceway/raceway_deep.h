#ifndef RACEWAY_DEEP_H
#define RACEWAY_DEEP_H
// raceway_deep.h — deep-drain kernel dispatch for cpu_raceway.
//
// ORDERED INCLUDE of the cpu_raceway translation unit (not a standalone header):
// include it after raceway_cap.h (needs g_deep_disp / g_ilp / Kern). Defines the
// member-function-pointer dispatch (DeepKFn / deep_kfn) and the RACEWAY_NM_DEEP
// macro that emits exactly one call shape per build. The dispatch is invoked INLINE
// at each drain site — never through a wrapper function — to preserve the proven
// drain-frame codegen (the -O2/-O3 fragility; see cpu_raceway.cpp + the o2_o3 doc).

// W5 dispatch: select branched vs branchless alg-dispatch via a MEMBER-FUNCTION POINTER, called inline at
// the drain site (a wrapper *function* calling the AVX-512 kernel from its own -O2 frame corrupts spilled
// args — the documented fragility; the inline indirect call keeps the proven-safe drain-frame shape). All
// x8 variants share this signature on BOTH kernels. natmap: x8 family. universal: x8 family incl. blmerge2
// (the universal-only cmov-pointer-select fix for its L1-hostile tables); branched=0 uses x12 (the
// universal interleave sweet spot) handled separately in the drain.
typedef void (Kern::*DeepKFn)(const key_schedule&, std::size_t, std::size_t,
  const uint8*,const uint8*,const uint8*,const uint8*,const uint8*,const uint8*,const uint8*,const uint8*,
  uint8*,uint8*,uint8*,uint8*,uint8*,uint8*,uint8*,uint8*);
static inline DeepKFn deep_kfn(){
  switch(g_deep_disp){
#ifdef UNIVERSAL
    case 4: return &Kern::run_maps_range_x8_blmerge2;   // universal-only
#else
    case 4: return &Kern::run_maps_range_x8_blmerge;    // natmap has no blmerge2 -> fall back to blmerge
#endif
    case 3: return &Kern::run_maps_range_x8_blall;
    case 2: return &Kern::run_maps_range_x8_blmerge;
    case 1: return &Kern::run_maps_range_x8_blarith;
    default:return &Kern::run_maps_range_x8;            // x8 branched (both kernels)
  }
}

#ifdef AVX2_NATMAP
// AVX2: only the x8-signature deep path exists (g_ilp is fixed at 8; the kernel does the AVX2-native
// x2/x4 sweep internally). A dedicated macro avoids naming the AVX-512-only x10/x12/x14_arr methods —
// g++ name-looks-up every if-constexpr branch in this non-template context, so they'd have to exist.
#define RACEWAY_NM_DEEP(BEG,END) \
  (tm.*deep_kfn())(s,BEG,END, in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7], out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7])
#elif !defined(UNIVERSAL)
// W6: natmap interleave dispatch. g_ilp is a compile-time constant so if-constexpr instantiates exactly ONE
// call shape per build (deterministic drain frame). ILP 8 = the W5 DEEP_DISP path (blmerge default); the
// others are BRANCHED. Called INLINE at each drain site (never via a wrapper fn — the documented fragility).
#define RACEWAY_NM_DEEP(BEG,END) \
  do { \
    if constexpr (g_ilp==8)       (tm.*deep_kfn())(s,BEG,END, in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7], out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]); \
    else if constexpr (g_ilp==1)  tm.run_maps_range_x1(s,BEG,END, in[0], out[0], 0); \
    else if constexpr (g_ilp==4)  tm.run_maps_range_x4(s,BEG,END, in[0],in[1],in[2],in[3], out[0],out[1],out[2],out[3]); \
    else if constexpr (g_ilp==10) { if(g_deep_disp==2) tm.run_maps_range_x10_blmerge_arr(s,BEG,END, &in[0][0], &out[0][0]); else tm.run_maps_range_x10_arr(s,BEG,END, &in[0][0], &out[0][0]); } \
    else if constexpr (g_ilp==12) { if(g_deep_disp==2) tm.run_maps_range_x12_blmerge_arr(s,BEG,END, &in[0][0], &out[0][0]); else tm.run_maps_range_x12_arr(s,BEG,END, &in[0][0], &out[0][0]); } \
    else if constexpr (g_ilp==14) tm.run_maps_range_x14_arr(s,BEG,END, &in[0][0], &out[0][0]); \
  } while(0)
#endif

#endif // RACEWAY_DEEP_H
