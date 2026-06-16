#ifndef RACEWAY_CAP_H
#define RACEWAY_CAP_H
// raceway_cap.h — bounded-cap + MAP1-dedup run state for cpu_raceway.
//
// ORDERED INCLUDE of the cpu_raceway translation unit (not a standalone header):
// include it after the config globals (FpMode, g_fp_mode, win_map_data) and the
// kernel selection. Declares the persistent per-boundary SharedCaps (Pool B), the
// W1 producer cap (Pool A), the continuation / deep-dispatch / sharding run-config,
// the global counters + lane-fill instrumentation, and the strong-fingerprint MAP1
// dedup set (HSet) with map1_dedup / producer_keep. All FN-only by construction.

// One SharedCap per intermediate boundary depth (ge value, 1<ge<ne). Persistent
// across all waves and threads for the whole run (the cross-wave drain). Global-
// shared is the correct topology on a single-LLC host; per-CCD/sharded is a
// Phase-2 A/B for split-L3 hosts.
static std::vector<SharedCap>      g_caps;       // index = boundary slot (see boundary_index)
static std::vector<std::size_t>    g_cap_ge;     // ge depth for each cap slot
static std::atomic<std::size_t>*   g_drops = nullptr;   // drops[cap_slot]
static int boundary_index(std::size_t ge){
  for(std::size_t i=0;i<g_cap_ge.size();i++) if(g_cap_ge[i]==ge) return (int)i;
  return -1;
}

// ---- W1: producer cap (Pool A as a fixed LLC-resident cap) ------------------
// The MAP1 dedup set is the W-growing RSS sink (exact HSet ~ F1 fingerprints; ~270M
// @2^32 — infeasible on a modest-L3 host). W1 fronts/replaces it with one global-
// shared fixed SharedCap (Pool A), so the MAP1 dedup working set is flat in W. The
// deep boundary caps (g_caps) are the Pool B analog (persistent cross-wave deep
// recapture) and are already built. FN-only: a cap miss streams the rep (over-keep,
// merged downstream by the exact final union / re-caught by Pool B), never a lost
// distinct. Modes: exact (HSet, the baseline), production (cap IS the dedup -> bounded
// RSS), shadow (HSet is truth + cap shadows to report the over-keep curve).
enum class PCap { exact, production, shadow };
static PCap g_pcap_mode = PCap::exact;
static SharedCap g_pcap;                          // global-shared MAP1 producer cap
static std::uint32_t g_pcap_bits=22, g_pcap_ways=4;
static std::atomic<std::size_t> g_pcap_capkeep{0};// shadow: count the cap WOULD keep among ROUTED reps
// Producer-cap ROUTING dial (GPU map1-frontier-route-tau): route only high-shed MAP1 reps into the
// cap; low-shed reps BYPASS un-hashed (streamed directly). Shed proxy = MAP1 alg0 score from
// run_map1_range_x8_scores (the same producer score streamhyb's route_tau uses). FN-safe: a bypassed
// rep is kept (over-keep if it was a dup), never dropped. The bet (GPU: 2-4x table shrink @tau 4/5 on
// diffuse): low-shed reps are mostly unique, so bypassing them frees the cap of their entries (smaller
// cap / fewer probes) for little over-keep. <0 => routing off (route every rep into the cap).
static double g_pcap_route_tau = -1.0;
static std::atomic<std::size_t> g_pcap_routed{0}, g_pcap_bypassed{0};

// Continuation mode (cap-on path). carry: survivors carry their 128 B boundary state through
// compaction; each span resumes from it (the natural lockstep shape; 128 B/rep carry+gather traffic).
// recompute: survivors carry only the 4 B data value; each span RE-derives state by expand(d)+maps[0,ge)
// from scratch — 32x less wave/carry memory (4 B vs 128 B/rep), but TRIANGULAR compute (span k redoes
// maps [0,ge_k) not just [gb,ge_k)). The recompute-vs-carry crossover is a port-vs-bandwidth trade
// (design §3.4): recompute favored on a BW-bound / tight-carry host, carry on a compute-bound one.
enum class Cont { carry, recompute };
static Cont g_cont = Cont::carry;

// W5: deep-drain alg-dispatch mode (natmap). The profile (§15) shows run_maps_range_x8 is 81% of cycles at
// 27.5% branch-mispredict — the alg-dispatch branches. Branchless variants remove them (blmerge is
// spill-free, -56% mispredicts, +13-32% cps to T4). The raceway thesis: drops cut deep
// map-evals, so the residual branchless dispatch (port-bound, ~0 SMT scaling) has the slack to win at scale.
// DEEP_DISP=branched|blarith|blmerge|blall. (natmap only; universal keeps x12.)
// DEFAULT = blmerge (2): W5 (§16) showed it wins at every thread count (closer +22/+11/+21% T1/8/16,
// diffuse +30/+9/+2%), parity-exact, branch-mispredict 27.8%->7.6%. Override with DEEP_DISP=branched.
static int g_deep_disp = 2;   // 0=branched 1=blarith 2=blmerge 3=blall

// Dynamic data-axis sharding: workers pull SHARD_CHUNK-sized data ranges from a shared atomic counter
// instead of one fixed contiguous shard each, balancing the per-region MAP1-frontier-density variance
// (the ~20% T16 wall imbalance the instrumentation found). SHARD_CHUNK=0 => static contiguous shards.
static bool g_dynamic_shard = true;
static std::uint64_t g_shard_chunk = 262144;   // data values per pull (>> wave fill is fine; persists across pulls)
static std::atomic<std::uint64_t> g_work_next{0};
static std::uint64_t g_eff = 0;

// ---- global counters / accounting -----------------------------------------
static std::atomic<std::size_t> g_f1{0};            // distinct MAP1 reps (summed over shards; T-dependent)
static std::atomic<std::size_t> g_deep_map_evals{0};// sum over spans of reps_entering*(ge-gb) — the port-cycle proxy
static std::atomic<std::size_t> g_cap_probes{0};
static std::atomic<std::size_t> g_finals_kept{0};   // reps reaching the final span (pre-union)
static std::atomic<unsigned long long> g_screen_hits{0};
static std::vector<FlatTable>   g_thread_final;     // per-thread exact final union (count mode)
// ---- instrumentation: SIMD lane-fill + per-thread imbalance ----------------
static std::atomic<std::size_t> g_lanes_used{0};        // sum of actual reps run across all span-batches
static std::atomic<std::size_t> g_lanes_dispatched{0};  // sum of g_ilp*batches (incl. padding lanes)
static std::atomic<std::size_t> g_spans{0};             // total span dispatches
static std::atomic<std::size_t> g_spans_underfull{0};   // spans entered with nw < g_ilp (lanes not even one full batch)
static std::vector<double>      g_thread_secs;          // per-thread wall (origination+drain)
static std::vector<std::size_t> g_thread_streamed;      // per-thread reps streamed (work proxy)
// Account one span's lane utilization (called once per span with the entering survivor count).
static inline void account_span(std::size_t nw){
  const std::size_t batches=(nw+(std::size_t)g_ilp-1)/(std::size_t)g_ilp;
  g_lanes_used += nw; g_lanes_dispatched += batches*(std::size_t)g_ilp; g_spans++;
  if(nw < (std::size_t)g_ilp) g_spans_underfull++;
}

// Strong-fingerprint MAP1 dedup set (fingerprints only — avoids materializing the
// full 128B MAP1 frontier). Identical shape to streamhyb's HSet so cap-off F1 and
// UNION match exactly at T1.
struct HSet {
  std::vector<std::uint64_t> a,b64; std::vector<std::uint32_t> b32;
  std::size_t mask=0,n=0; FpMode mode=FpMode::fp128;
  void init(FpMode m){ mode=m; a.assign(1u<<16,0); if(mode==FpMode::fp128) b64.assign(1u<<16,0); if(mode==FpMode::fp96) b32.assign(1u<<16,0); mask=(1u<<16)-1; }
  inline bool occ(std::size_t i) const { return a[i]||(mode==FpMode::fp128&&b64[i])||(mode==FpMode::fp96&&b32[i]); }
  void grow(){ std::size_t ns=(mask+1)<<1; std::vector<std::uint64_t> na(ns,0),nb64; std::vector<std::uint32_t> nb32;
    if(mode==FpMode::fp128) nb64.assign(ns,0); if(mode==FpMode::fp96) nb32.assign(ns,0); std::size_t nm=ns-1;
    for(std::size_t i=0;i<=mask;i++) if(occ(i)){ std::size_t j=(std::size_t)a[i]&nm;
      while(na[j]||(mode==FpMode::fp128&&nb64[j])||(mode==FpMode::fp96&&nb32[j])) j=(j+1)&nm;
      na[j]=a[i]; if(mode==FpMode::fp128) nb64[j]=b64[i]; if(mode==FpMode::fp96) nb32[j]=b32[i]; }
    a.swap(na); if(mode==FpMode::fp128) b64.swap(nb64); if(mode==FpMode::fp96) b32.swap(nb32); mask=nm; }
  inline bool insert(std::uint64_t f0,std::uint64_t f1=0){
    const std::size_t cap=mask+1; const std::size_t max_n=mode==FpMode::fp128?(cap>>1):((cap*3)>>2);
    const std::uint32_t f1_32=(std::uint32_t)f1;
    if(n>=max_n) grow(); if(!f0&&(mode==FpMode::fp64||!f1||(mode==FpMode::fp96&&!f1_32))) f0=1;
    std::size_t j=(std::size_t)f0&mask; while(occ(j)){
      if(a[j]==f0&&(mode==FpMode::fp64||(mode==FpMode::fp128&&b64[j]==f1)||(mode==FpMode::fp96&&b32[j]==f1_32))) return false;
      j=(j+1)&mask; }
    a[j]=f0; if(mode==FpMode::fp128) b64[j]=f1; if(mode==FpMode::fp96) b32[j]=f1_32; n++; return true; }
};

static inline bool map1_dedup(HSet& hs, const std::uint8_t* st){
  std::uint64_t h0,h1=0;
  if(g_fp_mode==FpMode::fp64) h0=w4b::strong64(st);
  else w4b::strong128(st,h0,h1);
  return hs.insert(h0,h1);
}

// MAP1 keep decision per producer-cap mode. Returns true => stream this rep downstream.
//   exact      : strong128 HSet (the baseline; per-thread exact dedup).
//   production : global-shared SharedCap (strong64) IS the dedup -> bounded MAP1 RSS,
//                FN-only over-keep. Also dedups CROSS-shard (shared), which the per-thread
//                HSet does not, cutting redundant deep work as a bonus.
//   shadow     : HSet decides (truth, == distinct), cap shadows -> over-keep diagnostic.
static inline bool producer_keep(HSet& hs, const std::uint8_t* st){
  switch(g_pcap_mode){
    case PCap::production: return g_pcap.screen(w4b::strong64(st));
    case PCap::shadow: {
      const bool ek = map1_dedup(hs, st);
      if(g_pcap.screen(w4b::strong64(st))) g_pcap_capkeep.fetch_add(1, std::memory_order_relaxed);
      return ek;
    }
    case PCap::exact: default: return map1_dedup(hs, st);
  }
}

#endif // RACEWAY_CAP_H
