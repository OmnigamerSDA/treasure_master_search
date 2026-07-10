// cpu_raceway — minimal bounded-wave CPU raceway forward (POC, Phase 1).
//
// A standalone CPU forward architecture adapting the GPU bounded-wave raceway
// (`bounded persistent cap-drain + wave-local compaction`) to the CPU's two
// shared-resource walls: cross-core memory bandwidth and per-core vector-ALU
// ports. See docs/cpu_raceway_poc_design_20260613.md (the spec) and
// docs/w4b_bfs_gpu_techniques_reference.md.
//
// Shape (per worker thread, sharding the data axis):
//   ORIGINATE  : stream the shard, MAP1 (run_maps_range_x8 over [0,1)), dedup MAP1
//                reps with a strong-fingerprint set (HSet, fp128 default), emit
//                distinct reps into a bounded WAVE buffer (sized <= LLC).
//   DEEP FORWARD: drain each full wave through the K-cadence span schedule, all
//                reps advancing LOCKSTEP x8 (natmap) / x12 (universal) within a
//                span — bit-identical to streamhyb's deep_il8. The cap probe
//                happens BETWEEN spans only, so the AVX-512 interleave lever is
//                fully intact.
//     cap-off (RACEWAY_CAP=0): EXACT FlatTable boundary dedup == streamhyb
//                deep_il8. This is the correctness ceiling / parity baseline.
//     cap-on  (RACEWAY_CAP=1): a persistent, LLC-resident, single-probe benign-
//                race SharedCap per boundary depth. A resident hit DROPS the rep
//                (skips its remaining maps); a miss is KEPT and carried (state) to
//                the next span. FN-only: pressure/races over-keep, never false-drop.
//   FINALIZE   : kept finals are deduped exactly (count mode, per-thread FlatTable)
//                then unioned across threads -> UNION_final, OR screened (natmap).
//
// Parity contract:
//   cap-off UNION_final == streamhyb_hash UNION_final   (exact, integer-equal)
//   cap-on  UNION_final == cap-off UNION_final          (final union is exact; the
//                cap drops only true dups, so the distinct count is preserved
//                modulo a negligible strong64 birthday collision on the kept set)
//   The cap's COST is over-keep: cap-on intermediate frontier and total map-evals
//   are >= cap-off. The bet (measured at high T, not 1T) is that the fixed LLC-
//   resident cap relieves the bandwidth wall and the extra evals fill SMT slack.
//
// ---------------------------------------------------------------------------
// OPERATING INSTRUCTIONS (current; see docs/cpu_raceway_poc_results_20260614.md
// and docs/avx2_raceway_port_benchmark_20260615.md for full results)
//
//   Build (from src/bruteforce/w4b_dedup_probe/):
//     make CXX=clang++-21 cpu_raceway        # AVX-512 natmap  (PRODUCTION primary)
//     make                 cpu_raceway_u     # AVX-512 universal (g++; -DUNIVERSAL)
//     make CXX=clang++-21 cpu_raceway_avx2   # AVX2 natmap  (older-ISA / "legacy" distribution)
//   clang-21 is the recommended natmap compiler (better codegen; +14-22% vs g++ on the
//   deep-bound keys). All targets pass -fno-builtin-memset (roots out the -O3 vectorized-
//   memset heap-corruption SIGSEGV; see docs/cpu_raceway_o2_o3_fragility_20260614.md).
//
//   Production defaults (no flags needed): cap-on, deep dispatch = blmerge (branchless,
//   the §16/W5 win), MAP1 originate stays BRANCHED (predictable dispatch — blmerge there
//   is -42% on closer keys). Interleave width:
//     AVX-512 natmap   : x10  (-DRACEWAY_NM_ILP=8 reverts to x8; §23/W6)
//     AVX-512 universal: x12 branched / x8 branchless
//     AVX2 natmap      : x2   (16 YMM/4 = x4 ceiling; with blmerge x2==x4, x2 has zero
//                              spills — set via -DAVX2_RACEWAY_W=4 for the x4 point)
//
//   Run:   ./cpu_raceway <key> <window> <T> [K] [count|screen] [wave_N] [cap_bits] [cap_ways] [fp64|fp96|fp128]
//          window=0 => full 2^32.  Size wave_N so the wave (2 x wave_N x 128 B) fits the LLC
//          (§4: e.g. 131072 ~ 32 MiB/buf, good for a >=32 MiB-L3 host); full-2^32 needs
//          PRODUCER_CAP=1 or the MAP1 set OOMs.
//   Env :  RACEWAY_CAP=0|1 (default 1)  DEEP_DISP=branched|blmerge (default blmerge)
//          PRODUCER_CAP=exact|shadow|1  PCAP_BITS/PCAP_WAYS  CONT=carry|recompute
//          SHARD_CHUNK=N (0=static)  WIN_POLICY=linear|squeeze|... (linear is best, §11/§23)
//   Pin :  pin to PHYSICAL cores (one logical CPU per core; SMT-off T) then add the siblings
//          for the SMT-on T. The sibling layout is host-specific (e.g. logical 0,1 = core0 vs
//          0,N = core0) — check /sys/devices/system/cpu/cpu0/topology/thread_siblings_list, or
//          let ./raceway_autoconfig.sh (this dir) pick the build + a sensible pin set for this host.
//
//   Source layout (this TU is split into ordered includes for readability — see each header's
//   banner; they are NOT standalone headers, they assume this file's include order):
//     raceway_kernel_select.h  ISA/kernel typedef (Kern) + interleave width (g_ilp)
//     raceway_cap.h            SharedCap boundary caps, W1 producer cap, run-config, MAP1 HSet
//     raceway_deep.h           deep-drain dispatch (DeepKFn / deep_kfn / RACEWAY_NM_DEEP)
#include "state_dedup.h"
#include "key_schedule.h"
#include "rng_obj.h"
#include "raceway_kernel_select.h"
#include "strong_hash.h"
#include "window_policy.h"
#include "map1_certifier.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <sys/resource.h>
using namespace state_dedup;

// ---- config (set in main, read by workers) --------------------------------
static std::uint32_t g_key = 0;
static std::uint64_t g_window = 0;       // 0 => full 2^32
static int           g_T = 1;
static int           g_K = 5;
static bool          g_screen = false;   // screen mode (natmap only) vs count mode
static std::uint32_t g_wave_N = 1u << 18;// reps per bounded wave (default 256K ~ 32MB x2 buf)
static bool          g_cap_on = true;
static std::uint32_t g_cap_bits = 22;    // SharedCap hot-tier total slots = 2^cap_bits (8B each)
static std::uint32_t g_cap_ways = 4;
enum class FpMode { fp64, fp96, fp128 };
static FpMode        g_fp_mode = FpMode::fp128;  // MAP1 dedup fingerprint width
static std::uint32_t g_win_polmask = 0;
static std::size_t   g_ne = 0;           // schedule length (== entries.size(), e.g. 27)
static bool          g_pre_active = false;
static std::uint32_t g_pre_shed_mask = 0;
static std::uint32_t g_pre_support_mask = 0xFFFFFFFFu;
static std::uint32_t g_pre_bits = 0;
static std::uint64_t g_pre_source_mult = 1;
static std::uint64_t g_represented_eff = 0;

// TILED enumeration (TILE_BITS=m): remap loop index d -> data so the m squeeze-selected (locally-
// collapsible) bits are the WITHIN-TILE axis (vary fastest), and the complement bits index the tile
// (vary slowest). At full 2^32 this is a pure REORDER of the whole keyspace (same final set, apples-to-
// apples vs linear) that brings co-collapsing reps close in time -> the persistent caps (Pool B) catch
// them while resident (the wave-locality / C* capture law) -> better drain population. The single-window
// WIN_POLICY (g_win_polmask) only SELECTS a subset; tiling SWEEPS the whole space tile by tile.
static std::uint32_t g_tile_bits = 0;        // 0 => tiling off
static std::uint32_t g_tile_inner_mask = 0;  // m squeeze-selected bit positions (within-tile axis)
static std::uint32_t g_tile_outer_mask = 0;  // complement (tile index axis)

static inline std::uint32_t win_map_data(std::uint64_t d){
  if(g_pre_active) return tm_window_policy::deposit_bits32((std::uint32_t)d, g_pre_support_mask);
  if(g_tile_bits){
    const std::uint32_t inner=(std::uint32_t)d & ((1u<<g_tile_bits)-1u);
    const std::uint32_t tile =(std::uint32_t)(d>>g_tile_bits);
    return tm_window_policy::deposit_bits32(inner, g_tile_inner_mask)
         | tm_window_policy::deposit_bits32(tile,  g_tile_outer_mask);
  }
  return g_win_polmask ? tm_window_policy::deposit_bits32((std::uint32_t)d, g_win_polmask)
                       : (std::uint32_t)d;
}

static std::vector<std::uint8_t> schedule_blob_from_key_schedule(const key_schedule& schedule){
  std::vector<std::uint8_t> blob;
  blob.reserve(schedule.entries.size() * 4u);
  for(const auto& e : schedule.entries){
    blob.push_back(e.rng1);
    blob.push_back(e.rng2);
    blob.push_back((std::uint8_t)((e.nibble_selector >> 8) & 0xFFu));
    blob.push_back((std::uint8_t)(e.nibble_selector & 0xFFu));
  }
  return blob;
}

#include "raceway_cap.h"

#include "raceway_deep.h"

// ===========================================================================
// Wave drain — EXACT (cap-off): the streamhyb deep_il8 boundary dedup. The honest
// correctness/throughput ceiling. Frontier shrinks by exact dedup at each span.
// ===========================================================================

// NOINLINE: keep these AVX-512 + large-stack drain bodies OFF the worker frame. Inlining one into
// worker bloats/misaligns its frame and trips the clang-21 _dl_runtime_resolve_xsavec SIGSEGV at -O3
// (same fragility as streamhyb's deep_traj_group; a cold concurrent kernel-init race).
__attribute__((noinline)) static void drain_wave_exact(Kern& tm, const key_schedule& s, FlatTable& o, FlatTable& sc, FlatTable& fin){
  const std::size_t ne=g_ne;
#ifdef UNIVERSAL
  alignas(64) std::uint8_t in[12][128], out[12][128];
#else
  static constexpr int NB = (g_ilp>12 ? 16 : 12);
  alignas(64) std::uint8_t in[NB][128], out[NB][128];
#endif
  for(std::size_t gb=1; gb<ne;){
    std::size_t ge=std::min(ne, gb+(std::size_t)g_K);
    prepare_map_group(tm,s,gb,ge);
    sc.reset((std::uint32_t)o.pool.size());
    const std::size_t np=o.pool.size();
    g_deep_map_evals += np*(ge-gb); account_span(np);
    for(std::size_t pi=0; pi<np; pi+=g_ilp){ std::size_t n=std::min((std::size_t)g_ilp, np-pi);
      for(int k=0;k<g_ilp;k++){ const std::size_t idx=pi+((std::size_t)k<n?k:0); std::memcpy(in[k], o.pool[idx].state.data(),128); }
#ifdef UNIVERSAL
      if(g_deep_disp==0)  // x12 branched (universal interleave sweet spot)
        tm.run_maps_range_x12(s,gb,ge, in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7],in[8],in[9],in[10],in[11],
                              out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11]);
      else                // x8 branchless (blmerge/blmerge2/...); g_ilp=8 so only in[0..7] are filled
        (tm.*deep_kfn())(s,gb,ge, in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7],
                         out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
#else
      RACEWAY_NM_DEEP(gb,ge);   // W6 interleave sweep (ILP 8 = W5 DEEP_DISP path)
#endif
      for(std::size_t k=0;k<n;k++) sc.insert(out[k],1u);
    }
    std::swap(o,sc); gb=ge;
  }
  // o.pool == finals; exact union into the per-thread final table.
  g_finals_kept += o.pool.size();
  if(g_screen){
    for(std::size_t i=0;i<o.pool.size();i++){ uint8 fl=0; if(tm.screen_state_raw(o.pool[i].state.data(),fl)) g_screen_hits++; }
  } else {
    for(std::size_t i=0;i<o.pool.size();i++) fin.insert(o.pool[i].state.data(),1u);
  }
}

// ===========================================================================
// Wave drain — CAP (cap-on): carry-state bounded wave. Survivors advance lockstep;
// the per-boundary SharedCap drops residents (skip remaining maps). FN-only.
// in_buf holds the live frontier; we run it -> out_buf, then compact kept states
// back into in_buf for the next span (wave-local re-densification).
// ===========================================================================
__attribute__((noinline)) static void drain_wave_cap(Kern& tm, const key_schedule& s,
                           std::vector<std::uint8_t>& in_buf, std::vector<std::uint8_t>& out_buf,
                           std::size_t nw, FlatTable& fin){
  const std::size_t ne=g_ne;
#ifdef UNIVERSAL
  alignas(64) std::uint8_t in[12][128], out[12][128];
#else
  static constexpr int NB = (g_ilp>12 ? 16 : 12);
  alignas(64) std::uint8_t in[NB][128], out[NB][128];
#endif
  for(std::size_t gb=1; gb<ne && nw>0;){
    std::size_t ge=std::min(ne, gb+(std::size_t)g_K);
    prepare_map_group(tm,s,gb,ge);
    g_deep_map_evals += nw*(ge-gb); account_span(nw);
    const bool probe = (ge<ne);                 // never drain at the final boundary (no maps saved)
    const int  cslot = probe ? boundary_index(ge) : -1;
    SharedCap* cap = (cslot>=0) ? &g_caps[cslot] : nullptr;
    std::size_t w=0, probes=0, drops=0;
    for(std::size_t pi=0; pi<nw; pi+=g_ilp){ std::size_t n=std::min((std::size_t)g_ilp, nw-pi);
      for(int k=0;k<g_ilp;k++){ const std::size_t idx=pi+((std::size_t)k<n?k:0); std::memcpy(in[k], &in_buf[idx*128],128); }
#ifdef UNIVERSAL
      if(g_deep_disp==0)  // x12 branched (universal interleave sweet spot)
        tm.run_maps_range_x12(s,gb,ge, in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7],in[8],in[9],in[10],in[11],
                              out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11]);
      else                // x8 branchless (blmerge/blmerge2/...); g_ilp=8 so only in[0..7] are filled
        (tm.*deep_kfn())(s,gb,ge, in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7],
                         out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
#else
      RACEWAY_NM_DEEP(gb,ge);   // W6 interleave sweep (ILP 8 = W5 DEEP_DISP path)
#endif
      for(std::size_t k=0;k<n;k++){
        bool keep=true;
        if(cap){ probes++; keep = cap->screen(w4b::strong64(out[k])); }   // false => resident dup => drop
        if(keep){ std::memcpy(&in_buf[w*128], out[k], 128); w++; }
        else drops++;
      }
    }
    if(cap){ g_cap_probes += probes; g_drops[cslot] += drops; }
    nw=w; gb=ge;
  }
  // in_buf[0..nw) == surviving finals; exact union (or screen) — drops were FN-only.
  g_finals_kept += nw;
  if(g_screen){
    for(std::size_t i=0;i<nw;i++){ uint8 fl=0; if(tm.screen_state_raw(&in_buf[i*128],fl)) g_screen_hits++; }
  } else {
    for(std::size_t i=0;i<nw;i++) fin.insert(&in_buf[i*128],1u);
  }
}

// ===========================================================================
// Wave drain — CAP + RECOMPUTE continuation. Survivors are DATA VALUES (4 B). At each boundary b the
// state is re-derived from scratch: expand(d) then maps [0,b). 32x less wave/carry memory than carry-
// state, at triangular compute cost (each boundary redoes the whole [0,b) prefix). Same FN-only cap.
// ===========================================================================
__attribute__((noinline)) static void drain_wave_cap_recompute(Kern& tm, const key_schedule& s,
                                     std::vector<std::uint32_t>& dvals, std::size_t nw, FlatTable& fin){
  const std::size_t ne=g_ne;
#ifdef UNIVERSAL
  alignas(64) std::uint8_t in[12][128], out[12][128];
#else
  static constexpr int NB = (g_ilp>12 ? 16 : 12);
  alignas(64) std::uint8_t in[NB][128], out[NB][128];
#endif
  // boundary list: intermediate cap depths, then the final ne (recompute full, screen/insert).
  std::vector<std::size_t> bs = g_cap_ge; bs.push_back(ne);
  for(std::size_t bi=0; bi<bs.size() && nw>0; bi++){
    const std::size_t b = bs[bi];
    prepare_map_group(tm,s,0,b);
    g_deep_map_evals += nw*b; account_span(nw);  // full [0,b) recompute (incl. MAP1) — the triangular cost
    const bool probe = (b<ne);
    const int  cslot = probe ? boundary_index(b) : -1;
    SharedCap* cap = (cslot>=0) ? &g_caps[cslot] : nullptr;
    const bool last = (b==ne);
    std::size_t w=0, probes=0, drops=0;
    for(std::size_t pi=0; pi<nw; pi+=g_ilp){ std::size_t n=std::min((std::size_t)g_ilp, nw-pi);
      for(int k=0;k<g_ilp;k++){ const std::size_t idx=pi+((std::size_t)k<n?k:0); tm.expand(g_key, dvals[idx]); std::memcpy(in[k], tm.state_raw(),128); }
#ifdef UNIVERSAL
      if(g_deep_disp==0)  // x12 branched
        tm.run_maps_range_x12(s,0,b, in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7],in[8],in[9],in[10],in[11],
                              out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11]);
      else                // x8 branchless
        (tm.*deep_kfn())(s,0,b, in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7],
                         out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
#else
      RACEWAY_NM_DEEP(0,b);   // W6 interleave sweep (ILP 8 = W5 DEEP_DISP path)
#endif
      for(std::size_t k=0;k<n;k++){
        const std::size_t idx=pi+k;
        if(last){
          if(g_screen){ uint8 fl=0; if(tm.screen_state_raw(out[k],fl)) g_screen_hits++; }
          else fin.insert(out[k],1u);
        } else {
          bool keep=true;
          if(cap){ probes++; keep = cap->screen(w4b::strong64(out[k])); }
          if(keep) dvals[w++]=dvals[idx];        // keep the DATA VALUE (4 B compaction)
          else drops++;
        }
      }
    }
    if(cap){ g_cap_probes += probes; g_drops[cslot] += drops; }
    if(!last) nw=w;
  }
  g_finals_kept += nw;   // survivors entering the final span (== finals screened/inserted)
}

// ---- worker: originate this shard, fill bounded waves, drain each --------------
static void worker(int id, std::uint64_t lo, std::uint64_t hi){
  const auto wt0=std::chrono::steady_clock::now();
  RNG rng; Kern tm(&rng); key_schedule s(g_key, key_schedule::ALL_MAPS);
  tm.bind_dedup_schedule(s); tm.bind_maps_range(s,0,1);
  HSet hs; if(g_pcap_mode!=PCap::production) hs.init(g_fp_mode);   // production: cap replaces HSet (bounded RSS)
  FlatTable& fin = g_thread_final[id]; fin.set_dynamic(true); fin.reset(1u<<16);
  FlatTable o, sc, scratch;            // exact-path tables (reused)
  std::vector<std::uint8_t> in_buf, out_buf;     // carry-path wave buffers (128 B states)
  std::vector<std::uint32_t> dwave;              // recompute-path wave (4 B data values)
  std::size_t local_f1=0;

  alignas(64) std::uint8_t pin[8][128], pout[8][128];

  const bool recompute = g_cap_on && (g_cont==Cont::recompute);
  auto flush_exact=[&](){ if(o.pool.empty()) return; drain_wave_exact(tm,s,o,sc,fin); o.reset(g_wave_N); };
  std::size_t cap_nw=0;
  auto flush_cap=[&](){ if(cap_nw==0) return; drain_wave_cap(tm,s,in_buf,out_buf,cap_nw,fin); cap_nw=0; };
  auto flush_recompute=[&](){ if(dwave.empty()) return; drain_wave_cap_recompute(tm,s,dwave,dwave.size(),fin); dwave.clear(); };

  if(g_cap_on){
    if(recompute) dwave.reserve(g_wave_N);                                            // 4 B/rep, 32x smaller
    else { in_buf.resize((std::size_t)g_wave_N*128); (void)out_buf; }   // out_buf is unused by drain_wave_cap (it compacts in place) — don't allocate it
  } else { o.set_dynamic(false); sc.set_dynamic(false); o.reset(g_wave_N); }

  auto stream_rep=[&](const std::uint8_t* st, std::uint32_t dval){
    local_f1++;
    if(g_cap_on){
      if(recompute){ dwave.push_back(dval); if(dwave.size()==g_wave_N) flush_recompute(); }
      else { std::memcpy(&in_buf[cap_nw*128], st, 128); if(++cap_nw==g_wave_N) flush_cap(); }
    } else { o.insert(st,1u); if(o.pool.size()>=g_wave_N) flush_exact(); }
  };
  // Producer-cap routing: route high-shed MAP1 reps into the cap, bypass low-shed (stream un-hashed).
  const bool routing = (g_pcap_route_tau>=0.0) && (g_pcap_mode!=PCap::exact);
  const float rtau = (float)g_pcap_route_tau;
  std::size_t l_routed=0, l_bypass=0, l_capkeep=0;
  float rsc[8];

  // Originate one data-axis range [lo2,hi2): MAP1 + dedup + stream into the bounded wave.
  // Waves persist ACROSS originate() calls (drained only when full / at worker end), so the chunk
  // size only affects load-balance granularity, not wave fullness.
  auto originate=[&](std::uint64_t lo2, std::uint64_t hi2){
    std::uint32_t dvk[8];
    std::uint64_t d=lo2;
    for(; d+8<=hi2; d+=8){
      for(int k=0;k<8;k++){ dvk[k]=win_map_data(d+k); tm.expand(g_key, dvk[k]); std::memcpy(pin[k], tm.state_raw(),128); }
      if(routing){
        tm.run_map1_range_x8_scores(s, pin[0],pin[1],pin[2],pin[3],pin[4],pin[5],pin[6],pin[7],
                                       pout[0],pout[1],pout[2],pout[3],pout[4],pout[5],pout[6],pout[7], rsc);
        for(int k=0;k<8;k++){
          const bool route = rsc[k] >= rtau;
          bool keep;
          if(g_pcap_mode==PCap::production){
            if(route){ l_routed++; keep=g_pcap.screen(w4b::strong64(pout[k])); } else { l_bypass++; keep=true; }
          } else { // shadow: exact HSet is truth; cap shadows the routed subset only
            keep=map1_dedup(hs,pout[k]);
            if(route){ l_routed++; if(g_pcap.screen(w4b::strong64(pout[k]))) l_capkeep++; } else l_bypass++;
          }
          if(keep) stream_rep(pout[k], dvk[k]);
        }
      } else {
        // MAP1 originate stays BRANCHED (do NOT swap to blmerge): for closer/mid keys MAP1
        // dominates total work, but its dispatch over collapsing data is highly predictable, so
        // branched is near-free while blmerge's 3-candidate inflation is pure overhead (measured
        // -42% @closer key). Branchless only pays where dispatch is unpredictable (the deep path).
        tm.run_maps_range_x8(s,0,1, pin[0],pin[1],pin[2],pin[3],pin[4],pin[5],pin[6],pin[7],
                                    pout[0],pout[1],pout[2],pout[3],pout[4],pout[5],pout[6],pout[7]);
        for(int k=0;k<8;k++){ if(producer_keep(hs,pout[k])) stream_rep(pout[k], dvk[k]); }   // false => dup, drop
      }
    }
    for(; d<hi2; d++){   // scalar tail (<8 reps): always route into cap (negligible count)
      const std::uint32_t dv=win_map_data(d); tm.expand(g_key, dv); tm.run_maps_range(s,0,1);
      if(producer_keep(hs,tm.state_raw())) stream_rep(tm.state_raw(), dv);
    }
  };

  // SINGLE originate() call site (two inline copies of this AVX-512 body bloat the natmap worker
  // frame past the clang -O2 stack-misalign threshold -> SIGSEGV). DYNAMIC: pull data-axis chunks
  // from a shared atomic counter (load-balances per-region MAP1-frontier-density variance, the ~20%
  // T16 imbalance); the wave/HSet persist across pulls. STATIC (SHARD_CHUNK=0): run [lo,hi) once.
  std::uint64_t a=lo, b=hi; bool first=true;
  for(;;){
    if(g_dynamic_shard){ std::uint64_t c=g_work_next.fetch_add(g_shard_chunk, std::memory_order_relaxed);
      if(c>=g_eff) break; a=c; b=std::min(c+g_shard_chunk, g_eff); }
    else { if(!first) break; first=false; }
    originate(a,b);
  }
  if(g_cap_on){ if(recompute) flush_recompute(); else flush_cap(); } else flush_exact();
  g_f1 += local_f1;
  g_pcap_routed += l_routed; g_pcap_bypassed += l_bypass; g_pcap_capkeep += l_capkeep;
  g_thread_secs[id]=std::chrono::duration<double>(std::chrono::steady_clock::now()-wt0).count();
  g_thread_streamed[id]=local_f1;
}

static const char* fp_name(FpMode m){ return m==FpMode::fp128?"128":m==FpMode::fp96?"96":"64"; }

int main(int c,char**v){
  if(c<4){
    std::fprintf(stderr,"usage: %s key window T [K] [count|screen] [wave_N] [cap_bits] [cap_ways] [fp64|fp96|fp128]\n",v[0]);
    std::fprintf(stderr,"  PRODUCTION CPU forward engine (bounded-wave raceway). window 0 = full 2^32 (needs PRODUCER_CAP=1).\n");
    std::fprintf(stderr,"  ./raceway_autoconfig.sh --build picks the host's ISA build + wave/cap. Env: PRECERT=0 disables the\n");
    std::fprintf(stderr,"  default certifier pre-exclusion; RACEWAY_CAP, DEEP_DISP, PRODUCER_CAP/PCAP_BITS, CONT, SHARD_CHUNK,\n");
    std::fprintf(stderr,"  WIN_POLICY (see the cpu_raceway.cpp header block).\n");
    return 2; }
  g_key=std::stoul(v[1],0,0); g_window=std::stoull(v[2],0,0); g_T=std::stoi(v[3]);
  if(c>4) g_K=std::stoi(v[4]);
  if(c>5) g_screen=(std::string(v[5])=="screen");
  if(c>6) g_wave_N=std::stoul(v[6],0,0);
  if(c>7) g_cap_bits=std::stoul(v[7],0,0);
  if(c>8) g_cap_ways=std::stoul(v[8],0,0);
  if(c>9){ std::string fp=v[9]; g_fp_mode=fp=="fp64"?FpMode::fp64:fp=="fp96"?FpMode::fp96:FpMode::fp128; }
  if(const char* e=std::getenv("RACEWAY_CAP")) g_cap_on=(std::string(e)!="0");
  if(const char* e=std::getenv("PRODUCER_CAP")){ std::string p=e;
    g_pcap_mode = (p=="1"||p=="production")?PCap::production : (p=="shadow")?PCap::shadow : PCap::exact; }
  if(const char* e=std::getenv("PCAP_BITS")) g_pcap_bits=std::stoul(e);
  if(const char* e=std::getenv("PCAP_WAYS")) g_pcap_ways=std::stoul(e);
  if(const char* e=std::getenv("PCAP_ROUTE_TAU")) g_pcap_route_tau=std::stod(e);
  if(const char* e=std::getenv("CONT")) g_cont=(std::string(e)=="recompute")?Cont::recompute:Cont::carry;
  if(g_cont==Cont::recompute && !g_cap_on){ std::fprintf(stderr,"CONT=recompute requires RACEWAY_CAP=1 (cap-path only); ignoring\n"); g_cont=Cont::carry; }
  // Shard default: dynamic IFF a shared producer cap is active. With per-thread EXACT MAP1 dedup,
  // static contiguous shards exploit local MAP1 collapse (fewer reps/thread) and beat dynamic by ~7-8%
  // on mid/diffuse; with the shared cap that locality penalty vanishes and dynamic's balance wins
  // (+2-18%). SHARD_CHUNK env overrides (0 => static).
  g_dynamic_shard = (g_pcap_mode != PCap::exact);
  if(const char* e=std::getenv("SHARD_CHUNK")){ g_shard_chunk=std::stoull(e); g_dynamic_shard=(g_shard_chunk!=0); }
#ifdef UNIVERSAL
  g_deep_disp = 0;   // universal default: x12 branched (interleave sweet spot, the production-best at T16)
#endif
  if(const char* e=std::getenv("DEEP_DISP")){ std::string d=e;
    g_deep_disp = d=="blmerge2"?4 : d=="blall"?3 : d=="blmerge"?2 : d=="blarith"?1 : 0; }
#ifdef UNIVERSAL
  if(g_deep_disp!=0) g_ilp=8;   // universal branchless variants are x8 (drops the x12 interleave)
#endif
  if(g_wave_N==0) throw std::runtime_error("wave_N must be > 0");
#ifdef UNIVERSAL
  if(g_screen){ std::fprintf(stderr,"screen mode unsupported on universal build (screen_state_raw is a stub); use count\n"); return 2; }
#endif
  const std::uint64_t represented_eff = g_window ? g_window : ((std::uint64_t)1<<32);
  std::uint64_t eff = represented_eff;
  g_represented_eff = represented_eff;
  {
    const char* precert_env = std::getenv("PRECERT");
    const bool precert_explicit = precert_env != nullptr;
    const bool precert_requested = precert_env == nullptr || std::string(precert_env) != "0";
    if(precert_requested){
      key_schedule ps(g_key, key_schedule::ALL_MAPS);
      g_pre_shed_mask = tm_map1_certifier::certified_shed_mask_from_schedule_blob(
        g_key, schedule_blob_from_key_schedule(ps));
      g_pre_bits = (std::uint32_t)__builtin_popcount(g_pre_shed_mask);
      g_pre_support_mask = ~g_pre_shed_mask;
      g_pre_source_mult = 1ull << g_pre_bits;
      if(g_pre_bits){
        const bool compatible =
          ((represented_eff % g_pre_source_mult) == 0ull) &&
          !std::getenv("WIN_POLICY") &&
          !std::getenv("TILE_BITS");
        if(!compatible){
          if((represented_eff % g_pre_source_mult) != 0ull && precert_explicit)
            throw std::runtime_error("PRECERT requires window divisible by 2^certified_bits");
          if((std::getenv("WIN_POLICY") || std::getenv("TILE_BITS")) && precert_explicit)
            throw std::runtime_error("PRECERT v1 disables WIN_POLICY/TILE_BITS; run without those env vars");
        } else {
          eff = represented_eff / g_pre_source_mult;
          if(eff == 0 || eff > ((std::uint64_t)1<<32))
            throw std::runtime_error("PRECERT logical support-axis window is out of range");
          g_pre_active = true;
          std::fprintf(stderr,"PRECERT active shed_bits=%u shed_mask=0x%08x support_mask=0x%08x represented=%llu logical=%llu source_mult=%llu\n",
            g_pre_bits, g_pre_shed_mask, g_pre_support_mask,
            (unsigned long long)represented_eff, (unsigned long long)eff, (unsigned long long)g_pre_source_mult);
        }
      } else {
        if(precert_explicit) std::fprintf(stderr,"PRECERT requested: shed_bits=0 mask=0x%08x (no-op)\n", g_pre_shed_mask);
      }
    }
  }
  g_eff = eff;

  // Window data-ordering policy (squeeze/backfill/...); identity at full 2^32 or unset.
  if(const char* e=std::getenv("WIN_POLICY")){ std::string p=e;
    if(!p.empty() && p!="linear" && p!="lowbits"){
      if((eff&(eff-1))!=0) throw std::runtime_error("WIN_POLICY requires W a power of two");
      std::uint32_t wbits=0; while(((std::uint64_t)1<<wbits)<eff) wbits++;
      g_win_polmask=tm_window_policy::make_bit_mask(tm_window_policy::parse_policy(p), wbits);
      std::fprintf(stderr,"WIN_POLICY=%s wbits=%u mask=0x%08x\n",p.c_str(),wbits,g_win_polmask);
    }
  }

  // TILED enumeration (TILE_BITS=m): inner m bits = squeeze-selected within-tile axis, rest = tile index.
  // The inner mask uses WIN_POLICY if set, else squeeze (the policy we are proving out). At full 2^32 this
  // reorders the whole keyspace; for a sub-window it sweeps the window as 2^(wbits-m) tiles of 2^m.
  if(const char* e=std::getenv("TILE_BITS")){
    g_tile_bits=std::stoul(e);
    if(g_tile_bits){
      std::uint32_t wbits=32; if(g_window){ wbits=0; while(((std::uint64_t)1<<wbits)<eff) wbits++; }
      if(g_tile_bits>=wbits) throw std::runtime_error("TILE_BITS must be < window bits");
      const std::string pol = std::getenv("WIN_POLICY") ? std::getenv("WIN_POLICY") : "squeeze";
      g_tile_inner_mask = tm_window_policy::make_bit_mask(tm_window_policy::parse_policy(pol), g_tile_bits);
      // outer = the wbits-wide axis minus the inner positions (tile index occupies the complement).
      const std::uint32_t axis = (wbits>=32)? 0xFFFFFFFFu : ((1u<<wbits)-1u);
      g_tile_outer_mask = axis & ~g_tile_inner_mask;
      g_win_polmask = 0;   // tiling supersedes the single-window policy mask
      std::fprintf(stderr,"TILE_BITS=%u pol=%s inner=0x%08x outer=0x%08x (tiles=2^%u of 2^%u)\n",
                   g_tile_bits, pol.c_str(), g_tile_inner_mask, g_tile_outer_mask, wbits-g_tile_bits, g_tile_bits);
    }
  }

  // Schedule length + boundary cadence (== streamhyb deep_il8 grouping: 1,1+K,...,ne).
  { key_schedule s(g_key, key_schedule::ALL_MAPS); g_ne=s.entries.size(); }
  for(std::size_t gb=1; gb<g_ne;){ std::size_t ge=std::min(g_ne, gb+(std::size_t)g_K);
    if(ge<g_ne) g_cap_ge.push_back(ge);    // intermediate boundaries only (no final-map drain)
    gb=ge; }
  if(g_cap_on){
    g_caps.resize(g_cap_ge.size());
    ScreenCapConfig cfg; cfg.enabled=true; cfg.cap_bits=g_cap_bits; cfg.cap_ways=g_cap_ways;
    for(auto& cap:g_caps) cap.init(cfg);
    g_drops=new std::atomic<std::size_t>[g_cap_ge.size()]; for(std::size_t i=0;i<g_cap_ge.size();i++) g_drops[i].store(0);
  }
  if(g_pcap_mode!=PCap::exact){
    ScreenCapConfig pcfg; pcfg.enabled=true; pcfg.cap_bits=g_pcap_bits; pcfg.cap_ways=g_pcap_ways;
    g_pcap.init(pcfg);
  }
  g_thread_final.resize(g_T);
  g_thread_secs.assign(g_T, 0.0); g_thread_streamed.assign(g_T, 0);

  // Warm one TM single-threaded (lazy RNG-table init) before workers spawn — the
  // static-init race guarded in map1par / map1_screen_capped.
  { RNG rng; Kern tm(&rng); key_schedule s(g_key, key_schedule::ALL_MAPS); tm.bind_dedup_schedule(s); tm.bind_maps_range(s,0,1); }

  auto t0=std::chrono::steady_clock::now();
  std::vector<std::thread> th;
  for(int i=0;i<g_T;i++){ std::uint64_t lo=eff*i/g_T, hi=eff*(i+1)/g_T; th.emplace_back(worker,i,lo,hi); }
  for(auto&t:th) t.join();
  double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();

  // Final union across per-thread finals -> exact distinct count.
  std::size_t uni=0; double uni_lf=0.0;
  if(!g_screen){
    FlatTable U; U.set_dynamic(true); U.reset(1u<<16);
    for(int i=0;i<g_T;i++) for(std::size_t pi=0;pi<g_thread_final[i].pool.size();pi++) U.insert(g_thread_final[i].pool[pi].state.data(),1u);
    uni=U.pool.size();
    uni_lf = (U.mask+1u) ? (double)U.pool.size()/(double)(U.mask+1u) : 0.0;
  }

  struct rusage ru; getrusage(RUSAGE_SELF,&ru);
  std::size_t total_drops=0; for(std::size_t i=0;i<g_cap_ge.size();i++) total_drops += g_cap_on? g_drops[i].load():0;
  const char* pcn = g_pcap_mode==PCap::production?"production":g_pcap_mode==PCap::shadow?"shadow":"exact";
  std::printf("key=0x%08x W=%llu T=%d K=%d wave_N=%u cap=%s cap_bits=%u cap_ways=%u fp=%s mode=%s pcap=%s cont=%s ne=%zu\n",
    g_key,(unsigned long long)g_window,g_T,g_K,g_wave_N, g_cap_on?"on":"off", g_cap_bits,g_cap_ways,fp_name(g_fp_mode), g_screen?"screen":"count", pcn,
    g_cont==Cont::recompute?"recompute":"carry", g_ne);
  if(g_dynamic_shard) std::printf("  shard=dynamic chunk=%llu", (unsigned long long)g_shard_chunk);
  else std::printf("  shard=static");
#ifdef UNIVERSAL
  { const char* dn[]={"x12","blarith","blmerge","blall","blmerge2"}; std::printf("  deep_disp=%s ilp=%d\n", dn[g_deep_disp], g_ilp); }
#else
  { const char* dn[]={"branched","blarith","blmerge","blall","blmerge2->blmerge"}; std::printf("  deep_disp=%s\n", dn[g_deep_disp]); }
#endif
  if(g_tile_bits) std::printf("  tiled: tile_bits=%u inner=0x%08x outer=0x%08x partition=%s\n",
    g_tile_bits, g_tile_inner_mask, g_tile_outer_mask,
    ((g_tile_inner_mask&g_tile_outer_mask)==0 && (g_tile_inner_mask|g_tile_outer_mask)==(g_window?0u:0xFFFFFFFFu))?"OK(bijection)":"partial(subwindow)");
  if(g_pre_active) std::printf("  precert: shed_bits=%u shed_mask=0x%08x support_mask=0x%08x represented_window=%llu logical_window=%llu source_mult=%llu\n",
    g_pre_bits, g_pre_shed_mask, g_pre_support_mask,
    (unsigned long long)g_represented_eff, (unsigned long long)g_eff, (unsigned long long)g_pre_source_mult);
  std::printf("  streamed=%zu  deep_map_evals=%zu  cap_probes=%zu  finals_kept=%zu  UNION_final=%zu  screen_hits=%llu\n",
    g_f1.load(), g_deep_map_evals.load(), g_cap_probes.load(), g_finals_kept.load(), uni, g_screen_hits.load());
  const std::size_t rtd=g_pcap_routed.load(), byp=g_pcap_bypassed.load();
  const bool routing = (g_pcap_route_tau>=0.0) && (g_pcap_mode!=PCap::exact);
  if(g_pcap_mode==PCap::shadow){
    const std::size_t f1=g_f1.load(), ck=g_pcap_capkeep.load();
    // production-equivalent streamed: bypassed reps (all stream) + cap-kept among routed.
    const std::size_t eqstream = routing ? (byp+ck) : ck;
    std::printf("  pcap(shadow): exact_F1=%zu prod_equiv_stream=%zu overkeep=%.3fx  pcap_MB=%.1f (bits=%u ways=%u)\n",
      f1, eqstream, f1?(double)eqstream/(double)f1:0.0, g_pcap.table_bytes()/1048576.0, g_pcap_bits, g_pcap_ways);
  } else if(g_pcap_mode==PCap::production){
    std::printf("  pcap(production): streamed=%zu (= distinct + over-keep; UNION is the exact truth)  pcap_MB=%.1f (bits=%u ways=%u)\n",
      g_f1.load(), g_pcap.table_bytes()/1048576.0, g_pcap_bits, g_pcap_ways);
  }
  if(routing){
    const std::size_t tot=rtd+byp;
    std::printf("  pcap(route): tau=%.1f routed=%zu (%.1f%%) bypassed=%zu (%.1f%%)  -> cap holds only the high-shed subset\n",
      g_pcap_route_tau, rtd, tot?100.0*rtd/tot:0.0, byp, tot?100.0*byp/tot:0.0);
  }
  if(g_cap_on){
    std::printf("  drops: total=%zu  by_boundary:", total_drops);
    for(std::size_t i=0;i<g_cap_ge.size();i++) std::printf(" ge%zu=%zu", g_cap_ge[i], g_drops[i].load());
    std::printf("  cap_MB(per_boundary)=%.1f total=%.1f\n",
      g_caps.empty()?0.0:g_caps[0].table_bytes()/1048576.0,
      (double)g_cap_ge.size()*(g_caps.empty()?0.0:g_caps[0].table_bytes())/1048576.0);
  }
  // ---- instrumentation: SIMD lane-fill, thread imbalance, table utilization ----
  {
    const std::size_t lu=g_lanes_used.load(), ld=g_lanes_dispatched.load();
    std::printf("  lanes: fill=%.4f (used=%zu disp=%zu)  spans=%zu underfull[nw<%d]=%zu\n",
      ld?(double)lu/(double)ld:0.0, lu, ld, g_spans.load(), g_ilp, g_spans_underfull.load());
    double tmin=1e30,tmax=0,tsum=0; std::size_t smin=(std::size_t)-1,smax=0;
    for(int i=0;i<g_T;i++){ double s=g_thread_secs[i]; if(s<tmin)tmin=s; if(s>tmax)tmax=s; tsum+=s;
      std::size_t w=g_thread_streamed[i]; if(w<smin)smin=w; if(w>smax)smax=w; }
    const double tmean=g_T?tsum/g_T:0.0;
    std::printf("  threads: secs min=%.2f mean=%.2f max=%.2f imbalance[max/mean]=%.3f  streamed min=%zu max=%zu spread[max/min]=%.3f\n",
      tmin,tmean,tmax, tmean?tmax/tmean:1.0, smin,smax, smin?(double)smax/(double)smin:0.0);
    if(g_pcap_mode!=PCap::exact){ const std::size_t o=g_pcap.hot_occupied(), sl=g_pcap.hot_slots();
      std::printf("  pcap   LF=%.3f (occ=%zu/%zu)\n", sl?(double)o/(double)sl:0.0, o, sl); }
    if(g_cap_on) for(std::size_t i=0;i<g_caps.size();i++){ const std::size_t o=g_caps[i].hot_occupied(), sl=g_caps[i].hot_slots();
      std::printf("  deepcap ge%-2zu LF=%.3f (occ=%zu/%zu)\n", g_cap_ge[i], sl?(double)o/(double)sl:0.0, o, sl); }
    if(!g_screen) std::printf("  union  LF=%.3f\n", uni_lf);
  }
  std::printf("  wall=%.2fs  cps=%.2f M/s", wall, eff/(wall*1e6));
  if(g_pre_active) std::printf("  represented_cps=%.2f M/s", g_represented_eff/(wall*1e6));
  std::printf("  peakRSS=%.2f GB\n", ru.ru_maxrss/1048576.0);
  if(g_cap_on && g_drops) delete[] g_drops;
  return 0;
}
