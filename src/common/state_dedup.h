#ifndef STATE_DEDUP_H
#define STATE_DEDUP_H

// Production state-dedup primitive: open-addressed flat hash + templated
// forward-block driver for processing a contiguous range of data values
// against a key's forward schedule with per-boundary state-merge.
//
// Speedup characterization (Zen 5 9900X, 477 SAT-queued keys, base
// 0x12345600, single-thread, dedup_expanded_states=true legacy default):
//
//                    W=16   W=256  W=1024  W=4096
//   on tm_8:         1.12x  1.85x  2.18x   2.35x   (vs no-dedup baseline)
//   on tm_avx2_r256s_8 (combined SIMD + dedup, vs scalar baseline):
//                    ~2.9x  ~5.6x  ~5.3x   ~6.1x  (median)
//
// All cells parity-verified: dedup'd multiset of final states equals the
// baseline multiset.
//
// Default policy now skips the post-expand merge: expand(key, data) is
// injective in `data` for any fixed `key` (data bytes are deterministically
// shuffled into specific byte positions and then have key-derived RNG
// material added with carry — both bijective for fixed key), so distinct
// data values in a window produce distinct expanded states. Hashing them
// is therefore pure overhead. Set dedup_expanded_states=true on
// forward_block_with_dedup to restore the legacy behaviour for comparison.
//
// Requirements on the TM impl (compile-time, not virtual):
//   void expand(uint32 key, uint32 data);
//   void run_one_map(const key_schedule::key_schedule_entry&);
//   const uint8_t* state_raw() const;
//   void load_state_raw(const uint8_t* src);
//
// The state_raw / load_state_raw pair lets the table hash and replay the
// impl's *internal* layout directly, bypassing the shuffle that
// fetch_data/load_data would impose on SIMD impls.
//
// Extension surface: this header currently tracks per-state multiplicity
// only. A BOINC-shaped consumer that needs to emit per-data-value results
// will want an origin-tracking variant (state -> list of producing data
// values). That's a separate type with the same probe machinery; see
// state_dedup_origins.h (to be added) for the planned shape.

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(__SSE__) || defined(_MSC_VER)
#include <xmmintrin.h>  // _mm_prefetch (batched-insert prefetch pipeline)
#endif

#include "data_sizes.h"
#include "key_schedule.h"
#include "routing.h"
#include "rng_obj.h"
#include "strong_hash.h"
#include "window_policy.h"

namespace state_dedup
{

constexpr std::size_t STATE_BYTES = 128;
constexpr std::uint64_t FULL_U32_WINDOW = 1ull << 32;
using RoutedState = tm_routing::RoutedState;

inline bool is_full_u32_window(std::uint32_t window)
{
    return window == 0;
}

inline std::uint64_t effective_window_count(std::uint32_t window)
{
    return is_full_u32_window(window) ? FULL_U32_WINDOW : static_cast<std::uint64_t>(window);
}

template <typename Fn>
inline void for_each_window_index(std::uint32_t window, Fn&& fn)
{
    if (is_full_u32_window(window))
    {
        std::uint32_t i = 0;
        do
        {
            fn(i);
            ++i;
        } while (i != 0);
        return;
    }

    for (std::uint32_t i = 0; i < window; i++) fn(i);
}

#ifdef STATE_DEDUP_TRACE
// Instrumentation-only insert-stream trace (compiled out unless STATE_DEDUP_TRACE
// is defined). Records, in insert order across all boundaries, each probe's
// fingerprint and whether it merged into an existing entry (is_dup=1) or created
// a new one (is_dup=0). Used by the front-cache-locality probe to simulate a
// small direct-mapped pre-table cache over the real merge stream. Never compiled
// into the production header.
struct InsertTraceRec { std::uint64_t fp; std::uint8_t is_dup; };
inline std::vector<InsertTraceRec>* g_insert_trace = nullptr;
#endif

// Open-addressed flat hash table for dedup'ing 128-byte states.
//
// Layout: a slot array (fingerprint + pool index) plus a contiguous entry
// pool (state bytes + multiplicity). Linear probing at load factor 0.5
// (table sized to next-power-of-two of 2x expected count). Allocator-free
// past reset() — both vectors are reused across boundary advances.
//
// The 64-bit fingerprint is FNV-1a over the 128-byte state followed by a
// murmur3 finalizer (improved avalanche -> fewer probe clusters). Full
// 128-byte memcmp only on fingerprint match; at our scales (frontier <
// 100K) the false-positive rate is ~2^-44 so the memcmp is genuinely rare.
struct FlatTable
{
    // 8-byte slot (was 16): probing touches half the cache lines per cluster and
    // the slot array is denser in L2/L3 — the random slot-probe is the dominant
    // hash-side memory cost once the per-boundary zero-fill is gone (see reset()).
    //   pool_index : full 32-bit index into the entry pool.
    //   fp16       : low 16 bits of the fingerprint — a fast-reject tag. A 16-bit
    //                tag leaves ~1/65536 spurious matches per probed slot, all
    //                caught by the 128-byte memcmp, so correctness is unaffected
    //                and the extra memcmp rate is negligible (~2e-5 / insert).
    //   epoch_tag  : low 16 bits of the table epoch; slot live iff == table tag.
    struct Slot
    {
        std::uint32_t pool_index;
        std::uint16_t fp16;
        std::uint16_t epoch_tag;
    };

    struct Entry
    {
        std::array<std::uint8_t, STATE_BYTES> state;
        std::uint32_t multiplicity;
    };

    struct InsertResult
    {
        std::uint32_t pool_index;
        bool inserted;
    };

    // Tier-1 front cache (optional; default disabled). A small, L2/per-CCD-L3-
    // resident direct-mapped cache of {full fp, pool index} indexed by a HIGH
    // fingerprint slice (decorrelated from the tier-2 low-bit index). A hit skips
    // the tier-2 random slot probe entirely. Tier 2 (slots) stays the source of
    // truth — a fc miss or fp/state mismatch falls through to the normal probe —
    // so the merged multiset is identical (parity-exact). The win is on the
    // post-closure first-merge duplicate flood (≈95% hits into a small hot set):
    // on split-L3 / no-V-cache hosts those would otherwise be cross-CCD/DRAM
    // probes. See docs/cpu_dedup_forward_interleave_findings_20260601.md §25.
    struct FCEntry
    {
        std::uint64_t fp;
        std::uint32_t pool_index;
        std::uint16_t epoch_tag;
    };

    std::vector<Slot> slots;
    std::vector<Entry> pool;
    std::vector<FCEntry> fc;          // tier-1 front cache (empty => disabled)
    std::uint32_t mask = 0;
    std::uint32_t fc_mask = 0;        // 0 => front cache disabled
    std::uint32_t epoch = 0;          // low 16 bits tag slots + fc; bumped per reset()
    bool dyn = false;                 // dynamic sizing: table tracks the frontier, not the window

    // Dynamic-sizing floors (only used when dyn==true): the table starts here and
    // grows+rehashes to the actual occupancy. The symbolic MAP1 predictor cannot
    // pre-size it (it is blind to the cross-page/concrete collapse that drives
    // closure — confirmed: returns "no collapse" even for strongly-closing keys),
    // so we size empirically from observed occupancy instead. See findings §30.
    static constexpr std::uint32_t kDynSlotFloor  = 1u << 14;  // 16K slots initial
    static constexpr std::uint32_t kDynPoolFloor  = 1u << 14;  // 16K entries initial reserve

    // Enable the tier-1 front cache with 2^bits entries. Call once before use;
    // survives reset() (epoch tag invalidates stale entries for free).
    inline void enable_front_cache(unsigned bits)
    {
        fc.assign(static_cast<std::size_t>(1) << bits, FCEntry{0, 0, 0});
        fc_mask = (1u << bits) - 1u;
    }

    // Dynamic table sizing: instead of pre-sizing slots to 2*window and reserving
    // window pool entries (which makes RSS scale with the WINDOW even though the
    // post-MAP1 frontier — for a closing key — is orders of magnitude smaller), the
    // table starts small and grows+rehashes to track the actual occupancy. The
    // window's duplicate inserts collapse onto existing entries and never grow it,
    // so a closing key holds ~2*frontier slots regardless of window. See
    // docs/cpu_dedup_forward_interleave_findings_20260601.md §28. Default off.
    inline void set_dynamic(bool d) { dyn = d; }

    // Double the slot array and re-insert every live pool entry. Called from
    // insert_fp when occupancy crosses load factor 0.5. Pool indices are stable
    // (pool is untouched) so the front cache stays valid across a grow.
    inline void grow_and_rehash()
    {
        const std::uint32_t newsize = (mask + 1u) << 1u;
        slots.assign(newsize, Slot{0, 0, 0});
        mask = newsize - 1u;
        const std::uint16_t et = static_cast<std::uint16_t>(epoch);
        const std::uint32_t n = static_cast<std::uint32_t>(pool.size());
        for (std::uint32_t pi = 0; pi < n; pi++)
        {
            const std::uint64_t fp = fingerprint(pool[pi].state.data());
            std::uint32_t idx = static_cast<std::uint32_t>(fp) & mask;
            while (slots[idx].epoch_tag == et) idx = (idx + 1u) & mask;
            slots[idx].epoch_tag = et;
            slots[idx].fp16 = static_cast<std::uint16_t>(fp);
            slots[idx].pool_index = pi;
        }
    }

    // Begin a new table generation. Instead of zero-filling the slot array every
    // boundary (~1-2 MB of pure-overhead writes per merge at large frontiers — a
    // shared-L3/DRAM bandwidth cost that dominates the dedup side at high thread
    // counts), we bump an `epoch` counter: a slot is empty iff its 16-bit tag !=
    // the table's. The slot array is only ever grown (never shrunk/re-zeroed), so
    // after the first window every reset() is a single increment. The 16-bit tag
    // wraps every 65536 resets (~2400 windows); on wrap we re-stale once, an
    // amortized-negligible full pass.
    // Size the table. Static mode: slots = next_pow2(2*expected_count) (the window
    // upper-bounds the frontier, no growth). Dynamic mode: ignore expected_count
    // for sizing — start at a small floor (or reuse an already-grown array; slots
    // only grow, never shrink) and let insert_fp grow+rehash to the actual
    // occupancy, so a closing key's table stays ~2*frontier regardless of window.
    inline void reset(std::uint32_t expected_count)
    {
        std::uint32_t size;
        if (dyn)
        {
            size = static_cast<std::uint32_t>(slots.size());
            if (size < kDynSlotFloor) size = kDynSlotFloor;
        }
        else
        {
            size = 16;
            while (size < expected_count * 2u) size <<= 1u;
        }
        mask = size - 1u;
        if (slots.size() < size)
        {
            // Grow (and implicitly clear): new slots get tag 0, stale vs >=1.
            slots.assign(size, Slot{0, 0, 0});
            epoch = 1;
        }
        else
        {
            ++epoch;
            if (static_cast<std::uint16_t>(epoch) == 0)
            {
                // 16-bit tag wrapped: re-stale everything, then skip tag 0.
                for (Slot& s : slots) s.epoch_tag = 0;
                for (FCEntry& e : fc) e.epoch_tag = 0;
                ++epoch;
            }
        }
        pool.clear();
        pool.reserve(dyn ? (expected_count == 0 ? kDynPoolFloor : std::min<std::uint32_t>(expected_count, kDynPoolFloor))
                         : expected_count);
    }

    // Reset only the entry pool, skipping the slot-table zero-fill. Used by
    // forward_block_with_dedup's skip-final-hash path where the slot table is
    // never touched. Avoids the (typically) 512KB slot-array clear per call.
    inline void reset_pool_only(std::uint32_t expected_count)
    {
        pool.clear();
        pool.reserve(expected_count);
        mask = 0;
        // No slot probing happens on this path (states are appended to pool
        // directly), so the epoch is left untouched.
    }

    // Four parallel FNV-1a streams over 16x uint64_t + murmur3 finalizer.
    // The original single-accumulator FNV serializes 16 multiplies (~3 cycle
    // latency each on Zen 5 → ~48-cycle dependency chain per state). Splitting
    // into 4 independent streams keeps each chain at 4 multiplies (~12 cycles),
    // with the streams interleaving on the multiplier port. Mixing them via
    // XOR into the murmur3 finalizer preserves the avalanche property at the
    // fingerprint level: any input-bit flip hits one stream, and the finalizer
    // diffuses the change across all 64 output bits.
    //
    // Result is never 0 (remapped to 1) so 0 stays usable as the empty marker.
    static inline std::uint64_t fingerprint(const std::uint8_t* state)
    {
        constexpr std::uint64_t seed  = 0xcbf29ce484222325ull;
        constexpr std::uint64_t prime = 0x100000001b3ull;
        const std::uint64_t* w = reinterpret_cast<const std::uint64_t*>(state);

        std::uint64_t h0 = seed, h1 = seed, h2 = seed, h3 = seed;
        h0 ^= w[ 0]; h0 *= prime;  h1 ^= w[ 1]; h1 *= prime;
        h2 ^= w[ 2]; h2 *= prime;  h3 ^= w[ 3]; h3 *= prime;
        h0 ^= w[ 4]; h0 *= prime;  h1 ^= w[ 5]; h1 *= prime;
        h2 ^= w[ 6]; h2 *= prime;  h3 ^= w[ 7]; h3 *= prime;
        h0 ^= w[ 8]; h0 *= prime;  h1 ^= w[ 9]; h1 *= prime;
        h2 ^= w[10]; h2 *= prime;  h3 ^= w[11]; h3 *= prime;
        h0 ^= w[12]; h0 *= prime;  h1 ^= w[13]; h1 *= prime;
        h2 ^= w[14]; h2 *= prime;  h3 ^= w[15]; h3 *= prime;

        std::uint64_t h = h0 ^ h1 ^ h2 ^ h3;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdull;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ull;
        h ^= h >> 33;
        return h == 0 ? 1ull : h;
    }

    // Insert a state with the given multiplicity. On duplicate, the existing
    // entry's multiplicity is incremented by `mult`.
    inline void insert(const std::uint8_t* state, std::uint32_t mult)
    {
        insert_fp(state, fingerprint(state), mult);
    }

    // As insert(), but with a precomputed fingerprint. Lets the batched-insert
    // pipeline compute and prefetch all fingerprints in a group before probing,
    // overlapping the random slot-probe latency. See batched_run_insert.
    inline void insert_fp(const std::uint8_t* state, std::uint64_t fp, std::uint32_t mult)
    {
        (void)insert_fp_result(state, fp, mult);
    }

    // As insert_fp(), but returns the stable pool index touched by the insert
    // and whether this call created a new pool entry. This lets routed callers
    // maintain side metadata without adding fields to the hot Entry layout.
    inline InsertResult insert_fp_result(const std::uint8_t* state, std::uint64_t fp, std::uint32_t mult)
    {
        const std::uint16_t et = static_cast<std::uint16_t>(epoch);
        const std::uint16_t fp16 = static_cast<std::uint16_t>(fp);

        // Tier-1 front-cache probe (when enabled): a confirmed hit increments the
        // pool entry directly and skips the tier-2 slot scan. A miss or mismatch
        // falls through to the authoritative tier-2 probe below.
        std::uint32_t fi = 0;
        if (fc_mask != 0)
        {
            fi = (static_cast<std::uint32_t>(fp >> 24)) & fc_mask;
            FCEntry& fe = fc[fi];
            if (fe.epoch_tag == et && fe.fp == fp)
            {
                Entry& entry = pool[fe.pool_index];
                if (std::memcmp(entry.state.data(), state, STATE_BYTES) == 0)
                {
                    entry.multiplicity += mult;
#ifdef STATE_DEDUP_TRACE
                    if (g_insert_trace) g_insert_trace->push_back({fp, 1});
#endif
                    return InsertResult{fe.pool_index, false};
                }
            }
        }

        // Dynamic sizing: grow+rehash before this insert can push occupancy past
        // load factor 0.5. Duplicate inserts (resolved above / below) don't reach
        // a state that grows the table beyond the live-entry count, so the window's
        // duplicate flood never inflates a closing key's table past ~2*frontier.
        if (dyn && pool.size() >= ((static_cast<std::size_t>(mask) + 1u) >> 1))
            grow_and_rehash();

        std::uint32_t idx = static_cast<std::uint32_t>(fp) & mask;
        while (true)
        {
            Slot& slot = slots[idx];
            if (slot.epoch_tag != et)
            {
                slot.epoch_tag = et;
                slot.fp16 = fp16;
                const std::uint32_t pidx = static_cast<std::uint32_t>(pool.size());
                slot.pool_index = pidx;
                pool.emplace_back();
                std::memcpy(pool.back().state.data(), state, STATE_BYTES);
                pool.back().multiplicity = mult;
                if (fc_mask != 0) fc[fi] = FCEntry{fp, pidx, et};
#ifdef STATE_DEDUP_TRACE
                if (g_insert_trace) g_insert_trace->push_back({fp, 0});
#endif
                return InsertResult{pidx, true};
            }
            if (slot.fp16 == fp16)
            {
                Entry& entry = pool[slot.pool_index];
                if (std::memcmp(entry.state.data(), state, STATE_BYTES) == 0)
                {
                    entry.multiplicity += mult;
                    if (fc_mask != 0) fc[fi] = FCEntry{fp, slot.pool_index, et};
#ifdef STATE_DEDUP_TRACE
                    if (g_insert_trace) g_insert_trace->push_back({fp, 1});
#endif
                    return InsertResult{slot.pool_index, false};
                }
            }
            idx = (idx + 1u) & mask;
        }
    }
};

inline void append_pool_state(FlatTable& table, const std::uint8_t* state, std::uint32_t mult)
{
    table.pool.emplace_back();
    std::memcpy(table.pool.back().state.data(), state, STATE_BYTES);
    table.pool.back().multiplicity = mult;
}

// Batched, prefetched insert. run_one(j) produces the j-th state into
// tm.state_raw(); mult_of(j) gives its multiplicity. For each group of up to
// BH items we run the producer, snapshot the states, then compute all
// fingerprints + prefetch all initial slots before resolving any probe — which
// overlaps the random slot-probe L2/L3 latency that dominates hash time at
// large frontiers. Inserts are reordered only within a group, so the resulting
// multiset is identical to per-state insertion (parity-preserving). Returns the
// nanoseconds spent in the insert phase (fingerprint + prefetch + probe),
// matching forward_block_with_dedup's hash_ns accounting.
// time_hash: when false (production, stats not requested) the per-batch
// clock::now() pair is skipped entirely — profiling showed the unconditional
// boundary timing cost ~2% in __vdso_clock_gettime. The timing is pure
// instrumentation for BoundaryRecord; skipping it cannot change the output.
template <typename TM, typename RunOne, typename MultOf>
inline std::uint64_t batched_run_insert(
    TM& tm, FlatTable& dst, std::uint32_t count, RunOne run_one, MultOf mult_of,
    bool time_hash = true)
{
    using clock = std::chrono::steady_clock;
    constexpr std::uint32_t BH = 8;  // batch / prefetch depth
    alignas(32) std::uint8_t states[BH * STATE_BYTES];
    std::uint32_t mults[BH];
    std::uint64_t fps[BH];
    std::uint64_t hash_ns = 0;
    for (std::uint32_t i = 0; i < count; )
    {
        const std::uint32_t b = std::min<std::uint32_t>(BH, count - i);
        for (std::uint32_t k = 0; k < b; k++)
        {
            run_one(i + k);
            std::memcpy(states + k * STATE_BYTES, tm.state_raw(), STATE_BYTES);
            mults[k] = mult_of(i + k);
        }
        clock::time_point h0;
        if (time_hash) h0 = clock::now();
        for (std::uint32_t k = 0; k < b; k++)
        {
            fps[k] = FlatTable::fingerprint(states + k * STATE_BYTES);
#if defined(__SSE__) || defined(_MSC_VER)
            _mm_prefetch(reinterpret_cast<const char*>(
                &dst.slots[static_cast<std::uint32_t>(fps[k]) & dst.mask]), _MM_HINT_T0);
#endif
        }
        for (std::uint32_t k = 0; k < b; k++)
            dst.insert_fp(states + k * STATE_BYTES, fps[k], mults[k]);
        if (time_hash)
            hash_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - h0).count());
        i += b;
    }
    return hash_ns;
}

// Routed variant of batched_run_insert. run_one(j) produces the j-th state,
// score_of(j) returns the confidence score for that just-produced state, and
// mult_of(j) gives its multiplicity. States with score >= tau are inserted into
// dst; lower-score states are appended to pass_buf as un-hashed carries. The
// hash_ns return measures only fingerprint/prefetch/probe time for hashed reps.
template <typename TM, typename RunOne, typename MultOf, typename ScoreOf>
inline std::pair<std::uint64_t, std::size_t> batched_run_insert_routed(
    TM& tm, FlatTable& dst, std::vector<RoutedState>& pass_buf,
    std::uint32_t count, RunOne run_one, MultOf mult_of, ScoreOf score_of,
    float tau, bool time_hash = true)
{
    using clock = std::chrono::steady_clock;
    constexpr std::uint32_t BH = 8;
    alignas(32) std::uint8_t states[BH * STATE_BYTES];
    std::uint32_t mults[BH];
    float scores[BH];
    std::uint64_t fps[BH];
    std::uint64_t hash_ns = 0;
    std::size_t passed = 0;

    for (std::uint32_t i = 0; i < count; )
    {
        const std::uint32_t b = std::min<std::uint32_t>(BH, count - i);
        for (std::uint32_t k = 0; k < b; k++)
        {
            run_one(i + k);
            scores[k] = static_cast<float>(score_of(i + k));
            std::memcpy(states + k * STATE_BYTES, tm.state_raw(), STATE_BYTES);
            mults[k] = mult_of(i + k);
        }

        clock::time_point h0;
        if (time_hash) h0 = clock::now();
        for (std::uint32_t k = 0; k < b; k++)
        {
            if (scores[k] < tau) continue;
            fps[k] = FlatTable::fingerprint(states + k * STATE_BYTES);
#if defined(__SSE__) || defined(_MSC_VER)
            _mm_prefetch(reinterpret_cast<const char*>(
                &dst.slots[static_cast<std::uint32_t>(fps[k]) & dst.mask]), _MM_HINT_T0);
#endif
        }
        for (std::uint32_t k = 0; k < b; k++)
        {
            if (scores[k] >= tau)
            {
                dst.insert_fp(states + k * STATE_BYTES, fps[k], mults[k]);
            }
            else
            {
                RoutedState carry;
                std::memcpy(carry.state.data(), states + k * STATE_BYTES, STATE_BYTES);
                carry.multiplicity = mults[k];
                pass_buf.push_back(carry);
                ++passed;
            }
        }
        if (time_hash)
            hash_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - h0).count());
        i += b;
    }

    return std::make_pair(hash_ns, passed);
}

// Metadata-aware routed variant. The hash side writes one RoutedState sidecar
// per FlatTable pool entry in hashed_meta; pass-through states are appended to
// pass_buf with their metadata preserved. meta_of(j) supplies the incoming
// metadata/history for the source rep, and alg0_of(j) supplies the current
// stage's alg0 count for the just-produced state. A state routed to hash gets
// a saturating hash_checks increment before insertion. On duplicate merge,
// merge_meta(dst_meta, src_meta) controls how competing histories collapse.
template <typename TM, typename RunOne, typename MultOf, typename ScoreOf,
          typename MetaOf, typename Alg0Of, typename MergeMeta>
inline std::pair<std::uint64_t, std::size_t> batched_run_insert_routed_meta(
    TM& tm, FlatTable& dst, std::vector<RoutedState>& hashed_meta,
    std::vector<RoutedState>& pass_buf, std::uint32_t count,
    RunOne run_one, MultOf mult_of, ScoreOf score_of, MetaOf meta_of,
    Alg0Of alg0_of, float tau, MergeMeta merge_meta, bool time_hash = true)
{
    using clock = std::chrono::steady_clock;
    constexpr std::uint32_t BH = 8;
    alignas(32) std::uint8_t states[BH * STATE_BYTES];
    std::uint32_t mults[BH];
    float scores[BH];
    float alg0s[BH];
    RoutedState metas[BH];
    std::uint64_t fps[BH];
    std::uint64_t hash_ns = 0;
    std::size_t passed = 0;

    for (std::uint32_t i = 0; i < count; )
    {
        const std::uint32_t b = std::min<std::uint32_t>(BH, count - i);
        for (std::uint32_t k = 0; k < b; k++)
        {
            const std::uint32_t j = i + k;
            run_one(j);
            scores[k] = static_cast<float>(score_of(j));
            alg0s[k] = static_cast<float>(alg0_of(j));
            metas[k] = meta_of(j);
            std::memcpy(states + k * STATE_BYTES, tm.state_raw(), STATE_BYTES);
            mults[k] = mult_of(j);
        }

        clock::time_point h0;
        if (time_hash) h0 = clock::now();
        for (std::uint32_t k = 0; k < b; k++)
        {
            if (scores[k] < tau) continue;
            fps[k] = FlatTable::fingerprint(states + k * STATE_BYTES);
#if defined(__SSE__) || defined(_MSC_VER)
            _mm_prefetch(reinterpret_cast<const char*>(
                &dst.slots[static_cast<std::uint32_t>(fps[k]) & dst.mask]), _MM_HINT_T0);
#endif
        }
        for (std::uint32_t k = 0; k < b; k++)
        {
            RoutedState meta = metas[k];
            std::memcpy(meta.state.data(), states + k * STATE_BYTES, STATE_BYTES);
            meta.current_alg0 = alg0s[k];
            meta.multiplicity = mults[k];

            if (scores[k] >= tau)
            {
                meta = tm_routing::mark_hash_checked(meta, alg0s[k]);
                const FlatTable::InsertResult ir =
                    dst.insert_fp_result(states + k * STATE_BYTES, fps[k], mults[k]);
                if (hashed_meta.size() < dst.pool.size())
                    hashed_meta.resize(dst.pool.size());
                if (ir.inserted)
                    hashed_meta[ir.pool_index] = meta;
                else
                    merge_meta(hashed_meta[ir.pool_index], meta);
            }
            else
            {
                pass_buf.push_back(meta);
                ++passed;
            }
        }
        if (time_hash)
            hash_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - h0).count());
        i += b;
    }

    return std::make_pair(hash_ns, passed);
}

template <typename TM, typename RunOne, typename MultOf, typename ScoreOf,
          typename MetaOf, typename Alg0Of>
inline std::pair<std::uint64_t, std::size_t> batched_run_insert_routed_meta(
    TM& tm, FlatTable& dst, std::vector<RoutedState>& hashed_meta,
    std::vector<RoutedState>& pass_buf, std::uint32_t count,
    RunOne run_one, MultOf mult_of, ScoreOf score_of, MetaOf meta_of,
    Alg0Of alg0_of, float tau, bool time_hash = true)
{
    return batched_run_insert_routed_meta(
        tm, dst, hashed_meta, pass_buf, count, run_one, mult_of, score_of,
        meta_of, alg0_of, tau, tm_routing::merge_hash_metadata_max, time_hash);
}

template <typename TM, typename RunOne, typename MultOf>
inline std::uint64_t batched_run_insert_full_u32(
    TM& tm, FlatTable& dst, RunOne run_one, MultOf mult_of,
    bool time_hash = true)
{
    using clock = std::chrono::steady_clock;
    constexpr std::uint32_t BH = 8;
    alignas(32) std::uint8_t states[BH * STATE_BYTES];
    std::uint32_t mults[BH];
    std::uint64_t fps[BH];
    std::uint64_t hash_ns = 0;

    std::uint32_t i = 0;
    do
    {
        std::uint32_t b = 0;
        do
        {
            run_one(i);
            std::memcpy(states + b * STATE_BYTES, tm.state_raw(), STATE_BYTES);
            mults[b] = mult_of(i);
            ++i;
            ++b;
        } while (b < BH && i != 0);

        clock::time_point h0;
        if (time_hash) h0 = clock::now();
        for (std::uint32_t k = 0; k < b; k++)
        {
            fps[k] = FlatTable::fingerprint(states + k * STATE_BYTES);
#if defined(__SSE__) || defined(_MSC_VER)
            _mm_prefetch(reinterpret_cast<const char*>(
                &dst.slots[static_cast<std::uint32_t>(fps[k]) & dst.mask]), _MM_HINT_T0);
#endif
        }
        for (std::uint32_t k = 0; k < b; k++)
            dst.insert_fp(states + k * STATE_BYTES, fps[k], mults[k]);
        if (time_hash)
            hash_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - h0).count());
    } while (i != 0);

    return hash_ns;
}

template <typename TM>
inline void run_map_group(
    TM& tm,
    const key_schedule& schedule,
    std::size_t begin,
    std::size_t end)
{
    if constexpr (requires(TM& t, const key_schedule& s, std::size_t b, std::size_t e)
                  { t.run_maps_range(s, b, e); })
    {
        tm.run_maps_range(schedule, begin, end);
    }
    else
    {
        for (std::size_t map_idx = begin; map_idx < end; map_idx++)
        {
            tm.run_one_map(schedule.entries[map_idx]);
        }
    }
}

template <typename TM>
inline void prepare_map_group(
    TM& tm,
    const key_schedule& schedule,
    std::size_t begin,
    std::size_t end)
{
    if constexpr (requires(TM& t, const key_schedule& s, std::size_t b, std::size_t e)
                  { t.bind_maps_range(s, b, e); })
    {
        tm.bind_maps_range(schedule, begin, end);
    }
}

template <typename TM>
inline int state_physical_index_for_alg_select(TM& tm, int logical_index)
{
    if constexpr (requires { TM::DEDUP_STATE_SHUFFLE; })
    {
        if constexpr (TM::DEDUP_STATE_SHUFFLE == 512)
            return (logical_index & 1) * 64 + ((logical_index >> 1) & 63);
        else if constexpr (TM::DEDUP_STATE_SHUFFLE == 256)
            return (logical_index / 64) * 64 + (logical_index & 1) * 32 + ((logical_index >> 1) & 31);
        else
            return logical_index;
    }
    else if constexpr (requires(TM& t) { t.shuffle(logical_index); })
        return tm.shuffle(logical_index);
    else
        return logical_index;
}

template <typename TM>
inline std::uint8_t raw_alg0_count_for_entry(
    TM& tm,
    const std::uint8_t* state,
    const key_schedule::key_schedule_entry& entry)
{
    std::uint16_t nibble_selector = entry.nibble_selector;
    std::uint8_t count = 0u;
    for (int i = 0; i < 16; i++)
    {
        const bool high_nibble = ((nibble_selector >> 15) & 1u) != 0;
        nibble_selector = static_cast<std::uint16_t>(nibble_selector << 1);
        std::uint8_t current_byte = state[state_physical_index_for_alg_select(tm, i)];
        if (high_nibble) current_byte = static_cast<std::uint8_t>(current_byte >> 4);
        const std::uint8_t alg_id = static_cast<std::uint8_t>((current_byte >> 1) & 0x07u);
        if (alg_id == 0u) ++count;
    }
    return count;
}

template <typename TM>
inline float run_one_map_scored(
    TM& tm,
    const key_schedule::key_schedule_entry& entry,
    bool use_shed_proxy)
{
    std::uint16_t rng_seed = static_cast<std::uint16_t>((entry.rng1 << 8) | entry.rng2);
    std::uint16_t nibble_selector = entry.nibble_selector;
    int op[16];
    std::uint8_t alg0_count = 0u;

    for (int i = 0; i < 16; i++)
    {
        const bool high_nibble = ((nibble_selector >> 15) & 1u) != 0;
        nibble_selector = static_cast<std::uint16_t>(nibble_selector << 1);
        std::uint8_t current_byte = tm.state_raw()[state_physical_index_for_alg_select(tm, i)];
        if (high_nibble) current_byte = static_cast<std::uint8_t>(current_byte >> 4);
        const int alg_id = static_cast<int>((current_byte >> 1) & 0x07u);
        op[i] = alg_id;
        if (alg_id == 0) ++alg0_count;
        tm.run_alg(alg_id, &rng_seed, 1);
    }

    return use_shed_proxy ? tm_routing::shed_proxy_map1(op) : static_cast<float>(alg0_count);
}

struct RoutingConfig
{
    bool enabled = false;
    float tau_base = 3.0f;
    float tau_slope = 0.25f;
    float tau_geom = 1.0f;
    std::uint32_t tau_start_entry = 0u;
    bool route_initial = true;
    bool route_initial_proxy = true;
    bool route_final = false;
    bool score_during_map = true;
};

inline float routing_tau_for_entry(const RoutingConfig& routing, std::uint32_t entry_begin)
{
    const std::uint32_t delta = entry_begin > routing.tau_start_entry
        ? entry_begin - routing.tau_start_entry
        : 0u;
    double tau = static_cast<double>(routing.tau_base);
    if (routing.tau_geom != 1.0f)
        tau *= std::pow(static_cast<double>(routing.tau_geom), static_cast<double>(delta));
    tau += static_cast<double>(routing.tau_slope) * static_cast<double>(delta);
    return static_cast<float>(tau < 0.0 ? 0.0 : tau);
}

inline bool routing_single_map_enabled(
    const RoutingConfig* routing,
    std::size_t entry_begin,
    std::size_t entry_end,
    bool is_initial,
    bool is_last)
{
    return routing != nullptr &&
        routing->enabled &&
        entry_end == entry_begin + 1u &&
        (!is_initial || routing->route_initial) &&
        (!is_last || routing->route_final);
}

// Per-boundary stats captured by forward_block_with_dedup when a non-null
// BoundaryStats* is passed. One BoundaryRecord is appended per merge group
// (including the initial expand-only frontier build when dedup_expanded_states
// is true). Wall times are the elapsed nanoseconds spent inside that group's
// map work versus its hash-merge work — they exclude any
// bind_maps_range / table-prep cost so the map/hash split stays comparable
// across impls.
struct BoundaryRecord
{
    std::uint32_t entry_begin;     // first schedule entry in the group (or UINT32_MAX for the expand-only initial frontier)
    std::uint32_t entry_end;       // one past last schedule entry in the group
    std::uint64_t frontier_in;     // unique states entering the group; 2^32 is represented exactly
    std::uint64_t frontier_out;    // unique states emitted after merge
    std::uint64_t map_ns;          // time spent running maps over the in-frontier
    std::uint64_t hash_ns;         // time spent inserting/merging into out-frontier
    std::uint64_t routed_hashed = 0;
    std::uint64_t routed_passed = 0;
    float routing_tau = 0.0f;
};

struct BoundaryStats
{
    std::vector<BoundaryRecord> records;
    inline void clear() { records.clear(); }
};

struct Map1SourcePrefilterConfig
{
    bool enabled = false;
    bool global_nibble_gate = true;
    std::uint16_t sample_nibble_mask = 0xFFFFu;
    std::uint32_t sample_max_classes = 1;
};

struct DataEnumerationConfig
{
    bool enabled = false;
    std::uint32_t variable_mask = 0xffffffffu;
    std::uint32_t fixed_value = 0u;
    std::uint32_t source_multiplicity = 1u;
};

inline std::uint32_t data_value_for_index(
    const DataEnumerationConfig* data_enum,
    std::uint32_t data_start,
    std::uint32_t data_stride,
    std::uint32_t i)
{
    const std::uint32_t logical = data_start + i * data_stride;
    if (data_enum != nullptr && data_enum->enabled)
    {
        return data_enum->fixed_value |
            tm_window_policy::deposit_bits32(logical, data_enum->variable_mask);
    }
    return logical;
}

inline std::uint32_t source_multiplicity_for_enum(const DataEnumerationConfig* data_enum)
{
    return (data_enum != nullptr && data_enum->enabled) ? data_enum->source_multiplicity : 1u;
}

enum class SourceTriBit : std::uint8_t
{
    Zero = 0,
    One = 1,
    Dep = 2,
};

struct SourceBit
{
    SourceTriBit value = SourceTriBit::Zero;
    std::uint8_t src = 0;
};

struct SourceByte
{
    std::array<SourceBit, 8> bit;
};

struct SourceState
{
    std::array<SourceByte, STATE_BYTES> byte;
};

struct Map1SourceClass
{
    std::uint32_t representative_offset = 0;
    std::uint32_t multiplicity = 0;
};

struct MaskByte
{
    std::uint8_t value = 0;
    std::uint8_t dep = 0;
};

struct MaskState
{
    std::array<MaskByte, STATE_BYTES> byte;
};

inline SourceTriBit mask_bit(const MaskByte& b, int bit)
{
    if (((b.dep >> bit) & 1u) != 0) return SourceTriBit::Dep;
    return ((b.value >> bit) & 1u) != 0 ? SourceTriBit::One : SourceTriBit::Zero;
}

inline void store_mask_bit(MaskByte& b, int bit, SourceTriBit x)
{
    const std::uint8_t m = static_cast<std::uint8_t>(1u << bit);
    if (x == SourceTriBit::Dep)
    {
        b.dep |= m;
        b.value &= static_cast<std::uint8_t>(~m);
    }
    else
    {
        b.dep &= static_cast<std::uint8_t>(~m);
        if (x == SourceTriBit::One) b.value |= m;
        else b.value &= static_cast<std::uint8_t>(~m);
    }
}

inline SourceTriBit tri_xor(SourceTriBit a, SourceTriBit b)
{
    if (a == SourceTriBit::Dep || b == SourceTriBit::Dep) return SourceTriBit::Dep;
    return a != b ? SourceTriBit::One : SourceTriBit::Zero;
}

inline SourceTriBit tri_and(SourceTriBit a, SourceTriBit b)
{
    if (a == SourceTriBit::Zero || b == SourceTriBit::Zero) return SourceTriBit::Zero;
    if (a == SourceTriBit::One && b == SourceTriBit::One) return SourceTriBit::One;
    return SourceTriBit::Dep;
}

inline SourceTriBit tri_or(SourceTriBit a, SourceTriBit b)
{
    if (a == SourceTriBit::One || b == SourceTriBit::One) return SourceTriBit::One;
    if (a == SourceTriBit::Zero && b == SourceTriBit::Zero) return SourceTriBit::Zero;
    return SourceTriBit::Dep;
}

inline void add_mask_constant_byte(MaskByte& b, std::uint8_t addend)
{
    if (b.dep == 0)
    {
        b.value = static_cast<std::uint8_t>(b.value + addend);
        return;
    }
    if (b.dep == 0xFFu)
    {
        b.value = 0;
        return;
    }

    SourceTriBit carry = SourceTriBit::Zero;
    MaskByte out;
    for (int bit = 0; bit < 8; bit++)
    {
        const SourceTriBit x = mask_bit(b, bit);
        const SourceTriBit c = ((addend >> bit) & 1u) != 0 ? SourceTriBit::One : SourceTriBit::Zero;
        store_mask_bit(out, bit, tri_xor(tri_xor(x, c), carry));
        carry = c == SourceTriBit::One ? tri_or(x, carry) : tri_and(x, carry);
    }
    b = out;
}

inline SourceBit source_const(bool one)
{
    return SourceBit{one ? SourceTriBit::One : SourceTriBit::Zero, 0};
}

inline SourceBit source_unknown(std::uint8_t src)
{
    return SourceBit{SourceTriBit::Dep, src};
}

inline SourceBit source_not(SourceBit a)
{
    if (a.src == 0)
        a.value = a.value == SourceTriBit::One ? SourceTriBit::Zero : SourceTriBit::One;
    return a;
}

inline SourceBit source_xor(SourceBit a, SourceBit b)
{
    if (a.src == 0 && b.src == 0)
        return source_const(a.value != b.value);
    if (a.src == 0) return a.value == SourceTriBit::One ? source_not(b) : b;
    if (b.src == 0) return b.value == SourceTriBit::One ? source_not(a) : a;
    return source_unknown(static_cast<std::uint8_t>(a.src | b.src));
}

inline SourceBit source_and(SourceBit a, SourceBit b)
{
    if ((a.src == 0 && a.value == SourceTriBit::Zero) ||
        (b.src == 0 && b.value == SourceTriBit::Zero))
        return source_const(false);
    if (a.src == 0 && a.value == SourceTriBit::One) return b;
    if (b.src == 0 && b.value == SourceTriBit::One) return a;
    if (a.src == 0 && b.src == 0) return source_const(a.value == SourceTriBit::One && b.value == SourceTriBit::One);
    return source_unknown(static_cast<std::uint8_t>(a.src | b.src));
}

inline SourceBit source_or(SourceBit a, SourceBit b)
{
    if ((a.src == 0 && a.value == SourceTriBit::One) ||
        (b.src == 0 && b.value == SourceTriBit::One))
        return source_const(true);
    if (a.src == 0 && a.value == SourceTriBit::Zero) return b;
    if (b.src == 0 && b.value == SourceTriBit::Zero) return a;
    if (a.src == 0 && b.src == 0) return source_const(a.value == SourceTriBit::One || b.value == SourceTriBit::One);
    return source_unknown(static_cast<std::uint8_t>(a.src | b.src));
}

inline void set_source_constant_byte(SourceState& state, std::size_t idx, std::uint8_t value)
{
    for (int bit = 0; bit < 8; bit++)
        state.byte[idx].bit[bit] = source_const(((value >> bit) & 1u) != 0);
}

inline void set_source_variable_plus_const_byte(SourceState& state, std::size_t idx, std::uint8_t addend)
{
    SourceBit carry = source_const(false);
    for (int bit = 0; bit < 8; bit++)
    {
        const SourceBit x = source_unknown(static_cast<std::uint8_t>(1u << bit));
        const SourceBit c = source_const(((addend >> bit) & 1u) != 0);
        state.byte[idx].bit[bit] = source_xor(source_xor(x, c), carry);
        carry = c.value == SourceTriBit::One ? source_or(x, carry) : source_and(x, carry);
    }
}

inline void add_source_constant_byte(SourceByte& byte, std::uint8_t addend)
{
    SourceBit carry = source_const(false);
    for (int bit = 0; bit < 8; bit++)
    {
        const SourceBit x = byte.bit[bit];
        const SourceBit c = source_const(((addend >> bit) & 1u) != 0);
        byte.bit[bit] = source_xor(source_xor(x, c), carry);
        carry = c.value == SourceTriBit::One ? source_or(x, carry) : source_and(x, carry);
    }
}

inline void ensure_map1_source_tables()
{
    static std::once_flag once;
    std::call_once(once, []()
    {
        static RNG rng;
        rng.generate_expansion_values_8();
        rng.generate_seed_forward_1();
        rng.generate_seed_forward_128();
        rng.generate_regular_rng_values_8();
        rng.generate_alg0_values_8();
        rng.generate_alg2_values_8_8();
        rng.generate_alg5_values_8_8();
        rng.generate_alg6_values_8();
    });
}

inline void init_source_expanded_state(SourceState& state, std::uint32_t key, std::uint32_t data_start)
{
    const std::uint8_t key_bytes[4] = {
        static_cast<std::uint8_t>((key >> 24) & 0xFFu),
        static_cast<std::uint8_t>((key >> 16) & 0xFFu),
        static_cast<std::uint8_t>((key >> 8) & 0xFFu),
        static_cast<std::uint8_t>(key & 0xFFu),
    };
    const std::uint8_t data_bytes[4] = {
        static_cast<std::uint8_t>((data_start >> 24) & 0xFFu),
        static_cast<std::uint8_t>((data_start >> 16) & 0xFFu),
        static_cast<std::uint8_t>((data_start >> 8) & 0xFFu),
        static_cast<std::uint8_t>(data_start & 0xFFu),
    };
    const uint16 seed = static_cast<uint16>((key >> 16) & 0xFFFFu);
    const std::uint8_t* row = RNG::expansion_values_8 + seed * STATE_BYTES;

    for (std::size_t i = 0; i < STATE_BYTES; i += 8)
    {
        set_source_constant_byte(state, i + 0, static_cast<std::uint8_t>(key_bytes[0] + row[i + 0]));
        set_source_constant_byte(state, i + 1, static_cast<std::uint8_t>(key_bytes[1] + row[i + 1]));
        set_source_constant_byte(state, i + 2, static_cast<std::uint8_t>(key_bytes[2] + row[i + 2]));
        set_source_constant_byte(state, i + 3, static_cast<std::uint8_t>(key_bytes[3] + row[i + 3]));
        set_source_constant_byte(state, i + 4, static_cast<std::uint8_t>(data_bytes[0] + row[i + 4]));
        set_source_constant_byte(state, i + 5, static_cast<std::uint8_t>(data_bytes[1] + row[i + 5]));
        set_source_constant_byte(state, i + 6, static_cast<std::uint8_t>(data_bytes[2] + row[i + 6]));
        set_source_variable_plus_const_byte(state, i + 7, static_cast<std::uint8_t>(data_bytes[3] + row[i + 7]));
    }
}

inline void init_mask_expanded_state(MaskState& state, std::uint32_t key, std::uint32_t data_start)
{
    const std::uint8_t key_bytes[4] = {
        static_cast<std::uint8_t>((key >> 24) & 0xFFu),
        static_cast<std::uint8_t>((key >> 16) & 0xFFu),
        static_cast<std::uint8_t>((key >> 8) & 0xFFu),
        static_cast<std::uint8_t>(key & 0xFFu),
    };
    const std::uint8_t data_bytes[4] = {
        static_cast<std::uint8_t>((data_start >> 24) & 0xFFu),
        static_cast<std::uint8_t>((data_start >> 16) & 0xFFu),
        static_cast<std::uint8_t>((data_start >> 8) & 0xFFu),
        static_cast<std::uint8_t>(data_start & 0xFFu),
    };
    const uint16 seed = static_cast<uint16>((key >> 16) & 0xFFFFu);
    const std::uint8_t* row = RNG::expansion_values_8 + seed * STATE_BYTES;

    for (std::size_t i = 0; i < STATE_BYTES; i += 8)
    {
        state.byte[i + 0] = MaskByte{static_cast<std::uint8_t>(key_bytes[0] + row[i + 0]), 0};
        state.byte[i + 1] = MaskByte{static_cast<std::uint8_t>(key_bytes[1] + row[i + 1]), 0};
        state.byte[i + 2] = MaskByte{static_cast<std::uint8_t>(key_bytes[2] + row[i + 2]), 0};
        state.byte[i + 3] = MaskByte{static_cast<std::uint8_t>(key_bytes[3] + row[i + 3]), 0};
        state.byte[i + 4] = MaskByte{static_cast<std::uint8_t>(data_bytes[0] + row[i + 4]), 0};
        state.byte[i + 5] = MaskByte{static_cast<std::uint8_t>(data_bytes[1] + row[i + 5]), 0};
        state.byte[i + 6] = MaskByte{static_cast<std::uint8_t>(data_bytes[2] + row[i + 6]), 0};
        state.byte[i + 7] = MaskByte{0, 0xFFu};
    }
}

inline void source_alg0(SourceState& state, uint16 seed)
{
    const std::uint8_t* row = RNG::alg0_values_8 + seed * STATE_BYTES;
    for (std::size_t i = 0; i < STATE_BYTES; i++)
    {
        for (int bit = 7; bit >= 1; bit--) state.byte[i].bit[bit] = state.byte[i].bit[bit - 1];
        state.byte[i].bit[0] = source_const((row[i] & 1u) != 0);
    }
}

inline void source_alg1(SourceState& state, uint16 seed)
{
    const std::uint8_t* row = RNG::regular_rng_values_8 + seed * STATE_BYTES;
    for (std::size_t i = 0; i < STATE_BYTES; i++) add_source_constant_byte(state.byte[i], row[i]);
}

inline void source_alg2(SourceState& state, uint16 seed)
{
    const SourceState old = state;
    SourceBit carry = source_const((RNG::alg2_values_8_8[seed] & 1u) != 0);
    for (int i = static_cast<int>(STATE_BYTES) - 1; i >= 1; i -= 2)
    {
        const SourceBit next_carry = old.byte[i - 1].bit[0];
        for (int bit = 0; bit < 7; bit++) state.byte[i - 1].bit[bit] = old.byte[i - 1].bit[bit + 1];
        state.byte[i - 1].bit[7] = old.byte[i].bit[7];
        for (int bit = 7; bit >= 1; bit--) state.byte[i].bit[bit] = old.byte[i].bit[bit - 1];
        state.byte[i].bit[0] = carry;
        carry = next_carry;
    }
}

inline void source_alg3(SourceState& state, uint16 seed)
{
    const std::uint8_t* row = RNG::regular_rng_values_8 + seed * STATE_BYTES;
    for (std::size_t i = 0; i < STATE_BYTES; i++)
        for (int bit = 0; bit < 8; bit++)
            if (((row[i] >> bit) & 1u) != 0) state.byte[i].bit[bit] = source_not(state.byte[i].bit[bit]);
}

inline void source_alg4(SourceState& state, uint16 seed)
{
    const std::uint8_t* row = RNG::regular_rng_values_8 + seed * STATE_BYTES;
    for (std::size_t i = 0; i < STATE_BYTES; i++)
        add_source_constant_byte(state.byte[i], static_cast<std::uint8_t>(0u - row[i]));
}

inline void source_alg5(SourceState& state, uint16 seed)
{
    const SourceState old = state;
    SourceBit carry = source_const((RNG::alg5_values_8_8[seed] & 0x80u) != 0);
    for (int i = static_cast<int>(STATE_BYTES) - 1; i >= 1; i -= 2)
    {
        const SourceBit next_carry = old.byte[i - 1].bit[7];
        for (int bit = 7; bit >= 1; bit--) state.byte[i - 1].bit[bit] = old.byte[i - 1].bit[bit - 1];
        state.byte[i - 1].bit[0] = old.byte[i].bit[0];
        for (int bit = 0; bit < 7; bit++) state.byte[i].bit[bit] = old.byte[i].bit[bit + 1];
        state.byte[i].bit[7] = carry;
        carry = next_carry;
    }
}

inline void source_alg6(SourceState& state, uint16 seed)
{
    const std::uint8_t* row = RNG::alg6_values_8 + seed * STATE_BYTES;
    for (std::size_t i = 0; i < STATE_BYTES; i++)
    {
        for (int bit = 0; bit < 7; bit++) state.byte[i].bit[bit] = state.byte[i].bit[bit + 1];
        state.byte[i].bit[7] = source_const((row[i] & 0x80u) != 0);
    }
}

inline void source_alg7(SourceState& state)
{
    for (std::size_t i = 0; i < STATE_BYTES; i++)
        for (int bit = 0; bit < 8; bit++)
            state.byte[i].bit[bit] = source_not(state.byte[i].bit[bit]);
}

inline void run_source_algorithm(SourceState& state, unsigned alg_id, uint16& seed)
{
    switch (alg_id)
    {
    case 0: source_alg0(state, seed); seed = RNG::seed_forward_128[seed]; break;
    case 1: source_alg1(state, seed); seed = RNG::seed_forward_128[seed]; break;
    case 2: source_alg2(state, seed); seed = RNG::seed_forward_1[seed]; break;
    case 3: source_alg3(state, seed); seed = RNG::seed_forward_128[seed]; break;
    case 4: source_alg4(state, seed); seed = RNG::seed_forward_128[seed]; break;
    case 5: source_alg5(state, seed); seed = RNG::seed_forward_1[seed]; break;
    case 6: source_alg6(state, seed); seed = RNG::seed_forward_128[seed]; break;
    case 7: source_alg7(state); break;
    default: break;
    }
}

inline void mask_alg0(MaskState& state, uint16 seed)
{
    const std::uint8_t* row = RNG::alg0_values_8 + seed * STATE_BYTES;
    for (std::size_t i = 0; i < STATE_BYTES; i++)
    {
        state.byte[i].value = static_cast<std::uint8_t>((state.byte[i].value << 1) | (row[i] & 1u));
        state.byte[i].dep = static_cast<std::uint8_t>(state.byte[i].dep << 1);
    }
}

inline void mask_alg1(MaskState& state, uint16 seed)
{
    const std::uint8_t* row = RNG::regular_rng_values_8 + seed * STATE_BYTES;
    for (std::size_t i = 0; i < STATE_BYTES; i++) add_mask_constant_byte(state.byte[i], row[i]);
}

inline void mask_alg2(MaskState& state, uint16 seed)
{
    const MaskState old = state;
    MaskByte carry{static_cast<std::uint8_t>(RNG::alg2_values_8_8[seed] & 1u), 0};
    for (int i = static_cast<int>(STATE_BYTES) - 1; i >= 1; i -= 2)
    {
        const MaskByte next_carry{
            static_cast<std::uint8_t>(old.byte[i - 1].value & 1u),
            static_cast<std::uint8_t>(old.byte[i - 1].dep & 1u)};
        state.byte[i - 1].value = static_cast<std::uint8_t>((old.byte[i - 1].value >> 1) | (old.byte[i].value & 0x80u));
        state.byte[i - 1].dep = static_cast<std::uint8_t>((old.byte[i - 1].dep >> 1) | (old.byte[i].dep & 0x80u));
        state.byte[i].value = static_cast<std::uint8_t>((old.byte[i].value << 1) | carry.value);
        state.byte[i].dep = static_cast<std::uint8_t>((old.byte[i].dep << 1) | carry.dep);
        carry = next_carry;
    }
}

inline void mask_alg3(MaskState& state, uint16 seed)
{
    const std::uint8_t* row = RNG::regular_rng_values_8 + seed * STATE_BYTES;
    for (std::size_t i = 0; i < STATE_BYTES; i++) state.byte[i].value ^= row[i];
}

inline void mask_alg4(MaskState& state, uint16 seed)
{
    const std::uint8_t* row = RNG::regular_rng_values_8 + seed * STATE_BYTES;
    for (std::size_t i = 0; i < STATE_BYTES; i++)
        add_mask_constant_byte(state.byte[i], static_cast<std::uint8_t>(0u - row[i]));
}

inline void mask_alg5(MaskState& state, uint16 seed)
{
    const MaskState old = state;
    MaskByte carry{static_cast<std::uint8_t>(RNG::alg5_values_8_8[seed] & 0x80u), 0};
    for (int i = static_cast<int>(STATE_BYTES) - 1; i >= 1; i -= 2)
    {
        const MaskByte next_carry{
            static_cast<std::uint8_t>(old.byte[i - 1].value & 0x80u),
            static_cast<std::uint8_t>(old.byte[i - 1].dep & 0x80u)};
        state.byte[i - 1].value = static_cast<std::uint8_t>((old.byte[i - 1].value << 1) | (old.byte[i].value & 1u));
        state.byte[i - 1].dep = static_cast<std::uint8_t>((old.byte[i - 1].dep << 1) | (old.byte[i].dep & 1u));
        state.byte[i].value = static_cast<std::uint8_t>((old.byte[i].value >> 1) | carry.value);
        state.byte[i].dep = static_cast<std::uint8_t>((old.byte[i].dep >> 1) | carry.dep);
        carry = next_carry;
    }
}

inline void mask_alg6(MaskState& state, uint16 seed)
{
    const std::uint8_t* row = RNG::alg6_values_8 + seed * STATE_BYTES;
    for (std::size_t i = 0; i < STATE_BYTES; i++)
    {
        state.byte[i].value = static_cast<std::uint8_t>((state.byte[i].value >> 1) | (row[i] & 0x80u));
        state.byte[i].dep = static_cast<std::uint8_t>(state.byte[i].dep >> 1);
    }
}

inline void mask_alg7(MaskState& state)
{
    for (MaskByte& b : state.byte) b.value = static_cast<std::uint8_t>(~b.value);
}

inline void run_mask_algorithm(MaskState& state, unsigned alg_id, uint16& seed)
{
    switch (alg_id)
    {
    case 0: mask_alg0(state, seed); seed = RNG::seed_forward_128[seed]; break;
    case 1: mask_alg1(state, seed); seed = RNG::seed_forward_128[seed]; break;
    case 2: mask_alg2(state, seed); seed = RNG::seed_forward_1[seed]; break;
    case 3: mask_alg3(state, seed); seed = RNG::seed_forward_128[seed]; break;
    case 4: mask_alg4(state, seed); seed = RNG::seed_forward_128[seed]; break;
    case 5: mask_alg5(state, seed); seed = RNG::seed_forward_1[seed]; break;
    case 6: mask_alg6(state, seed); seed = RNG::seed_forward_128[seed]; break;
    case 7: mask_alg7(state); break;
    default: break;
    }
}

inline bool map1_full_collapse_class(
    std::uint32_t key,
    std::uint32_t data_start,
    const key_schedule& schedule,
    std::vector<Map1SourceClass>& classes)
{
    if (schedule.entries.empty()) return false;
    if ((data_start & 0xFFu) != 0) return false;
    ensure_map1_source_tables();

    MaskState state;
    init_mask_expanded_state(state, key, data_start);

    const key_schedule::key_schedule_entry& entry = schedule.entries[0];
    uint16 seed = static_cast<uint16>((entry.rng1 << 8) | entry.rng2);
    uint16 nibble_selector = entry.nibble_selector;

    for (int step = 0; step < 16; step++)
    {
        const bool high_nibble = ((nibble_selector >> 15) & 1u) != 0;
        nibble_selector = static_cast<uint16>(nibble_selector << 1);
        const int first_bit = high_nibble ? 5 : 1;
        const std::uint8_t selector_mask = static_cast<std::uint8_t>(0x07u << first_bit);
        if ((state.byte[step].dep & selector_mask) != 0) return false;
        const unsigned alg_id = (state.byte[step].value >> first_bit) & 0x07u;
        run_mask_algorithm(state, alg_id, seed);
    }

    for (const MaskByte& b : state.byte)
        if (b.dep != 0) return false;

    classes.clear();
    classes.push_back(Map1SourceClass{0, 256});
    return true;
}

inline bool map1_source_classes(
    std::uint32_t key,
    std::uint32_t data_start,
    const key_schedule& schedule,
    std::vector<Map1SourceClass>& classes,
    std::uint32_t max_classes)
{
    if (schedule.entries.empty()) return false;
    if ((data_start & 0xFFu) != 0) return false;
    if (max_classes == 1) return map1_full_collapse_class(key, data_start, schedule, classes);
    ensure_map1_source_tables();

    SourceState state;
    init_source_expanded_state(state, key, data_start);

    const key_schedule::key_schedule_entry& entry = schedule.entries[0];
    uint16 seed = static_cast<uint16>((entry.rng1 << 8) | entry.rng2);
    uint16 nibble_selector = entry.nibble_selector;

    for (int step = 0; step < 16; step++)
    {
        const bool high_nibble = ((nibble_selector >> 15) & 1u) != 0;
        nibble_selector = static_cast<uint16>(nibble_selector << 1);
        const int first_bit = high_nibble ? 5 : 1;

        unsigned alg_id = 0;
        for (int bit = 0; bit < 3; bit++)
        {
            const SourceBit& selector_bit = state.byte[step].bit[first_bit + bit];
            if (selector_bit.src != 0) return false;
            if (selector_bit.value == SourceTriBit::One) alg_id |= 1u << bit;
        }
        run_source_algorithm(state, alg_id, seed);
    }

    std::uint8_t source_mask = 0;
    for (const SourceByte& byte : state.byte)
        for (const SourceBit& bit : byte.bit)
            source_mask = static_cast<std::uint8_t>(source_mask | bit.src);
    if (source_mask == 0xFFu) return false;

    std::array<int, 256> class_by_key;
    class_by_key.fill(-1);
    classes.clear();
    classes.reserve(1u << static_cast<unsigned>(std::min<std::uint32_t>(8, max_classes)));
    for (std::uint32_t offset = 0; offset < 256; offset++)
    {
        const std::uint32_t class_key = offset & source_mask;
        int class_idx = class_by_key[class_key];
        if (class_idx < 0)
        {
            if (classes.size() >= max_classes) return false;
            class_idx = static_cast<int>(classes.size());
            class_by_key[class_key] = class_idx;
            classes.push_back(Map1SourceClass{offset, 0});
        }
        classes[static_cast<std::size_t>(class_idx)].multiplicity++;
    }
    return !classes.empty() && classes.size() < 256;
}

inline std::size_t next_group_end(
    std::size_t entry_idx,
    std::size_t entry_count,
    std::uint32_t dedup_every_maps,
    std::uint32_t first_dedup_maps,
    const std::vector<std::uint32_t>* checkpoint_entries)
{
    std::size_t group_end = entry_count;
    if (checkpoint_entries != nullptr && !checkpoint_entries->empty())
    {
        const auto it = std::upper_bound(
            checkpoint_entries->begin(), checkpoint_entries->end(),
            static_cast<std::uint32_t>(entry_idx));
        if (it != checkpoint_entries->end())
            group_end = std::min<std::size_t>(entry_count, *it);
    }
    else
    {
        const std::uint32_t group_maps = entry_idx == 0
            ? first_dedup_maps
            : dedup_every_maps;
        group_end = std::min<std::size_t>(entry_count, entry_idx + group_maps);
    }
    if (group_end <= entry_idx) group_end = entry_idx + 1;
    return group_end;
}

// ===========================================================================
// Fingerprint-only MAP1 (stage-1) collapse — legacy characterization prototype.
// New memcmp-free MAP1 producer work should use w4b::strong64/strong128 from
// src/bruteforce/w4b_dedup_probe/strong_hash.h; this path is retained only for
// reproducing the interrupted map1_fpset_survey experiments.
//
// In the stage-1 / stage-2 partition (findings §32-§33) the stage-1 output is
// only the COLLAPSE DECISION (which data values are post-MAP1 representatives),
// not the states themselves — downstream re-derives states from the reps via
// expand+MAP1 (O(F)). So stage-1 never needs to store the 128-byte state; it
// only needs to answer "have I seen this MAP1-output state before?" That is a
// fingerprint SET, not a state pool. Dropping the 128 B state pool cuts memory
// from ~148 B/state (full-state dynamic stage-1 table) to ~32 B/state (128-bit
// fp at load 0.5) — a ~4-6x reduction that makes the full 2^32 (W4B) data axis
// surveyable even for non-closing keys.
//
// Correctness: fp-only dedup merges on hash equality, so a collision is a false
// merge = a dropped representative. A 128-bit fingerprint makes that probability
// ~10^-19 even at the worst case N=2^32, so the unique counts are exact for
// characterization (validated by parity vs the state-based MAP1 frontier_out).
// A narrower fp (96/64-bit) is a deployment-time memory tradeoff, not for
// characterization. See docs/cpu_dedup_forward_interleave_findings_20260601.md.

struct Fp128 { std::uint64_t lo, hi; };

// TWO INDEPENDENT 64-bit hashes over the raw 128-byte state. Independence is
// load-bearing: the post-schedule states are structurally sparse (differ in few
// bytes), and a single FNV-1a chain has weak avalanche on such inputs — measured
// ~2^32 effective entropy (115 collisions at N=1.1M), far below nominal 64-bit.
// FlatTable tolerates that because it memcmp-confirms every fp match; the fp-only
// set has no backstop, so the fingerprint itself must be collision-free. Two
// fully independent passes (different seeds/primes, multiplicative not XOR fold,
// per-pass strong finalizer) give a ~2^(32+32)=2^64-effective JOINT space → ~0
// collisions at N≤2^32 (verified bit-exact vs memcmp). Cost is ~2x the FNV work,
// negligible vs the per-candidate map work. (An earlier single-pass variant that
// derived both halves from the same streams only reached ~2^34 — correlated
// halves don't help; the passes must be independent.)
inline std::uint64_t fp_finalize(std::uint64_t h, std::uint64_t c0, std::uint64_t c1)
{
    h ^= h >> 33; h *= c0;
    h ^= h >> 29; h *= c1;
    h ^= h >> 32;
    return h;
}

inline Fp128 fingerprint128(const std::uint8_t* state)
{
    const std::uint64_t* w = reinterpret_cast<const std::uint64_t*>(state);
    // Pass A (FNV-64 constants), 4 streams.
    constexpr std::uint64_t sa = 0xcbf29ce484222325ull, pa = 0x100000001b3ull;
    // Pass B (independent: different seed + a different odd prime).
    constexpr std::uint64_t sb = 0x9e3779b97f4a7c15ull, pb = 0xff51afd7ed558ccdull;
    std::uint64_t a0 = sa, a1 = sa, a2 = sa, a3 = sa;
    std::uint64_t b0 = sb, b1 = sb, b2 = sb, b3 = sb;
    for (int i = 0; i < 16; i += 4)
    {
        a0 ^= w[i + 0]; a0 *= pa;  a1 ^= w[i + 1]; a1 *= pa;
        a2 ^= w[i + 2]; a2 *= pa;  a3 ^= w[i + 3]; a3 *= pa;
        // Pass B rotates the stream/word assignment so a cancellation in A's
        // XOR structure does not reproduce in B.
        b0 ^= w[i + 1]; b0 *= pb;  b1 ^= w[i + 2]; b1 *= pb;
        b2 ^= w[i + 3]; b2 *= pb;  b3 ^= w[i + 0]; b3 *= pb;
    }
    // Multiplicative/rotational fold (not plain XOR — XOR lets paired diffs cancel).
    std::uint64_t la = (a0 * 0x9e3779b97f4a7c15ull) + (a1 ^ (a2 << 23 | a2 >> 41)) + (a3 * 0xc2b2ae3d27d4eb4full);
    std::uint64_t lb = (b0 * 0xbf58476d1ce4e5b9ull) + (b1 ^ (b2 << 29 | b2 >> 35)) + (b3 * 0x94d049bb133111ebull);
    return { fp_finalize(la, 0xff51afd7ed558ccdull, 0xc4ceb9fe1a85ec53ull),
             fp_finalize(lb, 0x9e3779b97f4a7c15ull, 0xbf58476d1ce4e5b9ull) };
}

// Open-addressed 128-bit fingerprint SET (no state pool). Empty slot sentinel =
// (0,0); fingerprint128 finalizers make an all-zero 128-bit fp astronomically
// unlikely, and we map it to (1,0) defensively. Linear probe at load factor 0.5.
// Dynamic growth (slots track occupancy) so a closing key stays tiny regardless
// of window; reset() shrinks back to the floor so a huge key doesn't pin RAM for
// the next one. Allocation-light: one zero-fill per key (single window/pass).
struct FpSet
{
    struct Slot { std::uint64_t lo, hi; };   // 16 B; (0,0) == empty
    std::vector<Slot> slots;
    std::uint32_t mask = 0;
    std::size_t count = 0;
    static constexpr std::uint32_t kFloor = 1u << 14;  // 16K slots initial

    inline void reset()
    {
        slots.assign(kFloor, Slot{0, 0});
        mask = kFloor - 1u;
        count = 0;
    }

    inline void grow_and_rehash()
    {
        std::vector<Slot> old = std::move(slots);
        const std::uint32_t newsize = (mask + 1u) << 1u;
        slots.assign(newsize, Slot{0, 0});
        mask = newsize - 1u;
        for (const Slot& s : old)
        {
            if (s.lo == 0 && s.hi == 0) continue;
            std::uint32_t idx = static_cast<std::uint32_t>(s.lo) & mask;
            while (!(slots[idx].lo == 0 && slots[idx].hi == 0)) idx = (idx + 1u) & mask;
            slots[idx] = s;
        }
    }

    inline void insert(std::uint64_t lo, std::uint64_t hi)
    {
        if (lo == 0 && hi == 0) lo = 1;   // avoid the empty sentinel
        if (count >= ((static_cast<std::size_t>(mask) + 1u) >> 1)) grow_and_rehash();
        std::uint32_t idx = static_cast<std::uint32_t>(lo) & mask;
        while (true)
        {
            Slot& s = slots[idx];
            if (s.lo == 0 && s.hi == 0) { s.lo = lo; s.hi = hi; ++count; return; }
            if (s.lo == lo && s.hi == hi) return;   // duplicate
            idx = (idx + 1u) & mask;
        }
    }
};

// Run only the first map group (MAP1 / stage-1) over the window and count the
// unique post-MAP1 states via the fp-only set. Returns the post-MAP1 frontier
// size (== the state-based MAP1 frontier_out, validated by parity). Memory is
// ~32 B/unique-state, no 128-byte state pool — this is what makes W4B tractable
// for non-closing keys.
template <typename TM>
inline std::size_t map1_fpset_unique(
    TM& tm,
    std::uint32_t key,
    std::uint32_t data_start,
    std::uint32_t window,
    const key_schedule& schedule,
    FpSet& set,
    std::uint32_t dedup_every_maps,
    std::uint32_t first_dedup_maps,
    const std::vector<std::uint32_t>* checkpoint_entries,
    std::uint32_t data_stride = 1)
{
    if (first_dedup_maps == 0) first_dedup_maps = dedup_every_maps;
    const std::size_t group_end = next_group_end(
        0, schedule.entries.size(), dedup_every_maps, first_dedup_maps,
        checkpoint_entries);
    // Mirror forward_block_with_dedup: bind the schedule so map-mode kernels can
    // resolve map_idx by entry-pointer arithmetic in run_one_map (no-op for the
    // universal kernel). Without this the map kernel miscomputes a small fraction
    // of states (parity drift).
#ifndef DEDUP_NO_BIND_SCHEDULE
    if constexpr (requires(TM& t, const key_schedule& s) { t.bind_dedup_schedule(s); })
        tm.bind_dedup_schedule(schedule);
    else
        tm.bind_schedule(schedule);
#endif
    prepare_map_group(tm, schedule, 0, group_end);   // no-op unless map-table impl
    set.reset();
    for_each_window_index(window, [&](std::uint32_t i)
    {
        tm.expand(key, data_start + i * data_stride);
        run_map_group(tm, schedule, 0, group_end);
        const Fp128 fp = fingerprint128(tm.state_raw());
        set.insert(fp.lo, fp.hi);
    });
    return set.count;
}

// ===========================================================================
// Inverse-Bloom MAP1 SCREEN CAP (production form of the validated memory lever).
//
// The exact MAP1 producer (map1_fpset_unique / FlatTable) is linear in W: its
// kept set grows to the full post-MAP1 frontier, ~250 GB at W4B for a MAP1-robust
// key — infeasible on a modest host. The inverse-Bloom cap replaces the
// exact-growing set with a FIXED-CAPACITY, overwrite-on-collision cache whose RSS
// is independent of W (a flat, tunable hard budget). It is a screen-and-discard
// primitive: each post-MAP1 state is either streamed downstream (kept) or dropped
// as a known duplicate; it stores only a 64-bit fingerprint per slot, no state.
//
// SAFETY (false-negative-only): a slot matches only if THIS fingerprint was the
// last value written there, so the cap NEVER falsely reports "seen" => it NEVER
// drops a genuine state. It only FORGETS (eviction) => a forgotten duplicate is
// re-streamed (bounded over-keep, recovered/merged downstream), never a coverage
// loss. The only loss channel is a true strong64 birthday collision on the KEPT
// set (E ~ N^2/2^65; negligible once routed/capped — see Round 6). So `kept` is
// always >= the exact distinct count, and the streamed set is a superset of the
// exact frontier.
//
// THREADING: one SharedCap instance is shared by all worker threads screening a
// single key's window (the intra-key-shard W4B regime; production also supports
// across-key parallelism, one cap per key, resource-permitting). It is LOCK-FREE
// by benign race: 64-bit slots are naturally aligned so relaxed atomic load/store
// never tears, and there is no cross-slot invariant to order. A lost/torn write or
// a raced victim-selector only changes WHICH stale fp is forgotten => more
// over-keep, never a false drop. No atomics-for-correctness, no CAS, no locks.
//
// Validated: docs/w4b_g1_g2_squeeze_bighost_20260610.md (G1 flat RSS to 2^32; G2
// shared cap removes cross-shard over-keep; squeeze window policy lowers it
// further). Probe origin: src/bruteforce/w4b_dedup_probe/map1par.cpp.
// ===========================================================================

struct ScreenCapConfig
{
    bool enabled = false;
    std::uint32_t cap_bits = 22u;   // hot-tier total slots = 2^cap_bits (RAM = 2^cap_bits * 8 B, flat in W)
    std::uint32_t cap_ways = 4u;    // set-associativity; 4-way cuts over-keep 2-5x for ~free
    // Optional G6 "in-screen drain" second tier: a LARGER, COLD inverse-Bloom probed only on a hot-tier
    // miss, catching the NON-LOCAL dups the small hot tier forgot (the source of cap over-keep). 0 =
    // disabled. RAM adds 2^drain_bits * 8 B; still flat in W. A drain hit refreshes the hot tier.
    std::uint32_t drain_bits = 0u;  // 0 => no drain
    std::uint32_t drain_ways = 4u;
};

struct SharedCap
{
    // One inverse-Bloom tier: a flat slot array + a per-bucket round-robin evict selector.
    struct Tier
    {
        std::unique_ptr<std::atomic<std::uint64_t>[]> slots;
        std::vector<std::uint8_t> victim;    // per-bucket victim index (races benignly)
        std::uint32_t bucket_mask = 0u;
        std::uint32_t ways = 1u;
        std::size_t slot_count = 0u;

        void init(std::uint32_t bits, std::uint32_t w)
        {
            ways = w < 1u ? 1u : w;
            slot_count = static_cast<std::size_t>(1) << bits;
            slots.reset(new std::atomic<std::uint64_t>[slot_count]);
            for (std::size_t i = 0; i < slot_count; i++) slots[i].store(0, std::memory_order_relaxed);
            const std::size_t buckets = slot_count / ways;
            bucket_mask = static_cast<std::uint32_t>(buckets - 1u);
            victim.assign(buckets, 0u);
        }
        inline std::uint32_t bucket(std::uint64_t fp) const { return static_cast<std::uint32_t>(fp) & bucket_mask; }
        inline bool has(std::uint64_t fp, std::uint32_t bkt) const
        {
            const std::size_t base = static_cast<std::size_t>(bkt) * ways;
            for (std::uint32_t w = 0; w < ways; w++)
                if (slots[base + w].load(std::memory_order_relaxed) == fp) return true;
            return false;
        }
        inline void put(std::uint64_t fp, std::uint32_t bkt)
        {
            const std::size_t base = static_cast<std::size_t>(bkt) * ways;
            const std::uint8_t v = victim[bkt];                   // racy read (benign)
            slots[base + v].store(fp, std::memory_order_relaxed);  // evict victim (forget); benign race
            victim[bkt] = static_cast<std::uint8_t>((v + 1u) % ways);
        }
    };

    Tier hot;
    Tier drain;            // optional G6 second tier (drain.slot_count==0 => disabled)
    bool have_drain = false;

    void init(const ScreenCapConfig& cfg)
    {
        hot.init(cfg.cap_bits, cfg.cap_ways);
        have_drain = cfg.drain_bits != 0u;
        if (have_drain) drain.init(cfg.drain_bits, cfg.drain_ways);
    }

    // Screen one post-MAP1 fingerprint. Returns true if KEPT (streamed downstream): a never-seen fp,
    // or one forgotten by ALL tiers since last seen. Returns false only on an exact match in a tier (a
    // known duplicate -> drop). FN-only either way: a tier matches only on exactly THIS fp, so a genuine
    // state is never dropped; only forgetting (eviction) happens, bounded -> over-keep.
    inline bool screen(std::uint64_t fp)
    {
        if (fp == 0) fp = 1;                                   // 0 is the empty-slot sentinel
        const std::uint32_t hb = hot.bucket(fp);
        if (hot.has(fp, hb)) return false;                     // local dup -> drop
        if (have_drain)
        {
            const std::uint32_t db = drain.bucket(fp);
            if (drain.has(fp, db))                             // non-local dup the hot tier forgot
            {
                hot.put(fp, hb);                               // refresh hot so the next local recurrence is caught
                return false;                                  // -> drop (over-keep recovered)
            }
            hot.put(fp, hb);
            drain.put(fp, db);
            return true;                                       // genuinely new -> keep + remember in both
        }
        hot.put(fp, hb);
        return true;
    }

    double table_bytes() const
    {
        return static_cast<double>(hot.slot_count + (have_drain ? drain.slot_count : 0u)) * 8.0;
    }

    // Occupancy diagnostic (O(slots); call once at end). Counts live (non-zero) hot-tier slots; the
    // benign-race cap evicts rather than grows, so occupied/slot_count is its steady-state load factor
    // (saturates toward 1.0 once the working set exceeds capacity — the "cap is too small" signal).
    std::size_t hot_occupied() const
    {
        std::size_t n = 0;
        for (std::size_t i = 0; i < hot.slot_count; i++)
            if (hot.slots[i].load(std::memory_order_relaxed) != 0) n++;
        return n;
    }
    std::size_t hot_slots() const { return hot.slot_count; }
};

// Compute the data value for window index `idx` under an optional window-bit-
// selection mask (squeeze/backfill via window_policy.h). mask==0 => linear
// (data = data_start + idx*stride). Non-zero mask => data = data_start |
// deposit_bits32(idx*stride, mask), matching data_value_for_index semantics so the
// cap composes with the production squeeze enumeration with no separate code path.
inline std::uint32_t capped_data_value(
    std::uint32_t data_start, std::uint32_t off, std::uint32_t win_mask)
{
    return win_mask != 0u
        ? (data_start | tm_window_policy::deposit_bits32(off, win_mask))
        : (data_start + off);
}

// Screen a window INDEX range [idx_lo, idx_hi) of one key's MAP1 output through a
// caller-provided SHARED cap, returning the number of states KEPT (streamed) in the
// range. The caller shards [0, window) across worker threads, each with its own TM
// and the SAME `cap`, then sums the returned counts (intra-key parallelism). Pass
// win_mask = make_bit_mask(Squeeze, log2(window)) to use the production squeeze
// policy; 0 for linear. Mirrors map1_fpset_unique's first-map-group selection.
template <typename TM>
inline std::size_t map1_screen_capped_range(
    TM& tm,
    std::uint32_t key,
    std::uint32_t data_start,
    SharedCap& cap,
    std::uint64_t idx_lo,
    std::uint64_t idx_hi,
    const key_schedule& schedule,
    std::uint32_t dedup_every_maps = 1u,
    std::uint32_t first_dedup_maps = 0u,
    const std::vector<std::uint32_t>* checkpoint_entries = nullptr,
    std::uint32_t win_mask = 0u,
    std::uint32_t data_stride = 1u)
{
    if (first_dedup_maps == 0u) first_dedup_maps = dedup_every_maps;
    const std::size_t group_end = next_group_end(
        0, schedule.entries.size(), dedup_every_maps, first_dedup_maps, checkpoint_entries);
#ifndef DEDUP_NO_BIND_SCHEDULE
    if constexpr (requires(TM& t, const key_schedule& s) { t.bind_dedup_schedule(s); })
        tm.bind_dedup_schedule(schedule);
    else
        tm.bind_schedule(schedule);
#endif
    prepare_map_group(tm, schedule, 0, group_end);

    std::size_t kept = 0;
    for (std::uint64_t idx = idx_lo; idx < idx_hi; idx++)
    {
        const std::uint32_t off = static_cast<std::uint32_t>(idx) * data_stride;
        tm.expand(key, capped_data_value(data_start, off, win_mask));
        run_map_group(tm, schedule, 0, group_end);
        if (cap.screen(w4b::strong64(tm.state_raw()))) ++kept;
    }
    return kept;
}

// Ergonomic one-call capped MAP1 screen: shards a key's whole window across `threads` workers into ONE
// shared cap (intra-key parallelism, the W4B regime) and returns the total KEPT (streamed) count = the
// over-kept live-set size. Allocates the cap from `cfg`; pass win_mask = make_bit_mask(Squeeze,
// log2(window)) for the production squeeze policy (0 = linear). Requires TM constructible as TM(RNG*)
// (every CPU forward kernel here satisfies this). The single-thread warmup runs the kernel's lazy
// RNG-table init before the workers spawn (the static-init race guarded in map1par). full-2^32 window
// is the `window==0` sentinel.
template <typename TM>
inline std::size_t map1_screen_capped(
    std::uint32_t key,
    std::uint32_t window,
    const key_schedule& schedule,
    SharedCap& cap,
    unsigned threads,
    std::uint32_t win_mask = 0u,
    std::uint32_t dedup_every_maps = 1u,
    std::uint32_t first_dedup_maps = 0u,
    const std::vector<std::uint32_t>* checkpoint_entries = nullptr,
    std::uint32_t data_start = 0u,
    std::uint32_t data_stride = 1u)
{
    const std::uint64_t eff = effective_window_count(window);
    if (threads < 1u) threads = 1u;
    // Warm one TM single-threaded (empty range => bind/prepare only) so the kernel's lazy static RNG
    // tables are built before any worker thread touches them (otherwise a cold concurrent init can read
    // half-filled tables -> nondeterministic results: a cold concurrent kernel-init race).
    {
        RNG rng; TM warm(&rng);
        map1_screen_capped_range(warm, key, data_start, cap, 0, 0, schedule,
                                 dedup_every_maps, first_dedup_maps, checkpoint_entries, win_mask, data_stride);
    }
    std::vector<std::size_t> kept(threads, 0);
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (unsigned t = 0; t < threads; t++)
    {
        const std::uint64_t lo = eff * t / threads;
        const std::uint64_t hi = eff * (t + 1u) / threads;
        pool.emplace_back([&, t, lo, hi]() {
            RNG rng; TM tm(&rng);
            kept[t] = map1_screen_capped_range(tm, key, data_start, cap, lo, hi, schedule,
                                               dedup_every_maps, first_dedup_maps, checkpoint_entries,
                                               win_mask, data_stride);
        });
    }
    for (auto& th : pool) th.join();
    std::size_t total = 0;
    for (std::size_t k : kept) total += k;
    return total;
}

// Convenience overload: allocate the shared cap from `cfg` internally and return the kept count.
template <typename TM>
inline std::size_t map1_screen_capped(
    std::uint32_t key,
    std::uint32_t window,
    const key_schedule& schedule,
    const ScreenCapConfig& cfg,
    unsigned threads,
    std::uint32_t win_mask = 0u,
    std::uint32_t dedup_every_maps = 1u,
    std::uint32_t first_dedup_maps = 0u,
    const std::vector<std::uint32_t>* checkpoint_entries = nullptr,
    std::uint32_t data_start = 0u,
    std::uint32_t data_stride = 1u)
{
    SharedCap cap; cap.init(cfg);
    return map1_screen_capped<TM>(key, window, schedule, cap, threads, win_mask,
                                  dedup_every_maps, first_dedup_maps, checkpoint_entries,
                                  data_start, data_stride);
}

// Run the forward schedule over [data_start, data_start+window) with
// per-boundary state-merge dedup. On return, out_table.pool contains the
// unique final states with their multiplicity counts.
//
// out_table is caller-owned so multiple calls in a tight loop can reuse the
// underlying allocations. A second scratch table is used internally for
// double-buffering at each boundary.
template <typename TM>
void forward_block_with_dedup(
    TM& tm,
    std::uint32_t key,
    std::uint32_t data_start,
    std::uint32_t window,
    const key_schedule& schedule,
    FlatTable& out_table,
    FlatTable& scratch_table,
    std::uint32_t dedup_every_maps = 1,
    std::uint32_t first_dedup_maps = 0,
    const std::vector<std::uint32_t>* checkpoint_entries = nullptr,
    bool dedup_expanded_states = false,
    BoundaryStats* stats = nullptr,
    bool skip_final_hash = false,
    const Map1SourcePrefilterConfig* map1_source_prefilter = nullptr,
    bool batch_inserts = true,
    std::uint32_t data_stride = 1,
    const DataEnumerationConfig* data_enumeration = nullptr,
    const RoutingConfig* routing = nullptr)
{
    using clock = std::chrono::steady_clock;
    auto ns_since = [](clock::time_point t0) -> std::uint64_t {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count());
    };

    if (dedup_every_maps == 0) dedup_every_maps = 1;
    if (first_dedup_maps == 0) first_dedup_maps = dedup_every_maps;
    const bool full_window = is_full_u32_window(window);
    const std::uint64_t effective_window = effective_window_count(window);
    const std::uint32_t source_multiplicity = source_multiplicity_for_enum(data_enumeration);
    const std::uint64_t represented_window =
        effective_window * static_cast<std::uint64_t>(source_multiplicity);
    const bool old_out_dynamic = out_table.dyn;
    const bool old_scratch_dynamic = scratch_table.dyn;
    if (full_window)
    {
        // Static table sizing would require a 2^33-slot table. The exact W4B
        // sentinel is therefore always handled with frontier-sized dynamic tables.
        out_table.set_dynamic(true);
        scratch_table.set_dynamic(true);
    }

    // Optional hook for impls that precompute per-schedule data (map-mode
    // kernel). No-op for universal-table impls. Passes the parent schedule
    // so the impl can compute map_idx from entry pointer arithmetic in the
    // run_one_map loop below.
#ifndef DEDUP_NO_BIND_SCHEDULE
    if constexpr (requires(TM& t, const key_schedule& s) { t.bind_dedup_schedule(s); })
    {
        tm.bind_dedup_schedule(schedule);
    }
    else
    {
        tm.bind_schedule(schedule);
    }
#endif

    std::size_t entry_idx = 0;

    // Build initial frontier. Expanded states are injective in `data` for any
    // fixed `key`, so we skip the post-expand merge by default and run the
    // first map group before hashing. In dynamic mode reset() ignores `window`
    // and grows the table to the actual post-MAP1 frontier empirically (the
    // symbolic predictor is blind to the cross-page collapse — findings §30).
    out_table.reset(window);
    if (dedup_expanded_states || schedule.entries.empty())
    {
        std::uint64_t hash_ns = 0;
        const auto map_t0 = clock::now();
        for_each_window_index(window, [&](std::uint32_t i)
        {
            tm.expand(key, data_value_for_index(data_enumeration, data_start, data_stride, i));
            clock::time_point h0;
            if (stats != nullptr) h0 = clock::now();
            out_table.insert(tm.state_raw(), source_multiplicity);
            if (stats != nullptr) hash_ns += ns_since(h0);
        });
        if (stats != nullptr)
        {
            BoundaryRecord rec;
            rec.entry_begin = UINT32_MAX;
            rec.entry_end = 0;
            rec.frontier_in = represented_window;
            rec.frontier_out = out_table.pool.size();
            const std::uint64_t total_ns = ns_since(map_t0);
            rec.hash_ns = hash_ns;
            rec.map_ns = total_ns > hash_ns ? total_ns - hash_ns : 0;
            stats->records.push_back(rec);
        }
    }
    else
    {
        const std::size_t group_end = next_group_end(
            0, schedule.entries.size(), dedup_every_maps, first_dedup_maps,
            checkpoint_entries);
        prepare_map_group(tm, schedule, 0, group_end);
        std::uint64_t hash_ns = 0;
        const auto map_t0 = clock::now();
        // The MAP1 source-prefilter collapses the 256 low-byte page offsets into
        // their post-MAP1 equivalence classes (the analysis in map1_source_classes
        // walks only schedule entry 0 = MAP1). Equal-after-MAP1 implies
        // equal-after-(MAP1 + any further deterministic maps), so the per-class
        // representative + multiplicity stays exact even when the first group runs
        // several maps before the first hash merge. group_end == 1 is therefore the
        // classic "f1" first-stage hash; group_end > 1 (e.g. first_dedup_maps=4,
        // "pfk4") *defers* the first hash past maps 2..k and lets the prefilter's
        // intra-page winnowing stand in for it. Cross-page collapse still happens at
        // the (deferred) hash. Always valid for any first-group size >= 1.
        const bool use_map1_source_prefilter =
            map1_source_prefilter != nullptr &&
            map1_source_prefilter->enabled &&
            !(data_enumeration != nullptr && data_enumeration->enabled) &&
            !full_window &&
            (data_start & 0xFFu) == 0 &&
            (window % 256u) == 0;

        std::uint16_t learned_nibbles = map1_source_prefilter != nullptr
            ? map1_source_prefilter->sample_nibble_mask
            : 0xFFFFu;
        std::vector<Map1SourceClass> source_classes;

        if (use_map1_source_prefilter && map1_source_prefilter->global_nibble_gate)
        {
            learned_nibbles = 0;
            std::uint16_t tried_nibbles = 0;
            for (std::uint32_t base = 0; base < window && tried_nibbles != map1_source_prefilter->sample_nibble_mask; base += 256u)
            {
                const std::uint32_t page_start = data_start + base;
                const unsigned nibble = (page_start >> 8) & 0x0Fu;
                const std::uint16_t bit = static_cast<std::uint16_t>(1u << nibble);
                if ((map1_source_prefilter->sample_nibble_mask & bit) == 0) continue;
                if ((tried_nibbles & bit) != 0) continue;
                tried_nibbles = static_cast<std::uint16_t>(tried_nibbles | bit);
                if (map1_source_classes(
                        key, page_start, schedule, source_classes,
                        map1_source_prefilter->sample_max_classes))
                {
                    learned_nibbles = static_cast<std::uint16_t>(learned_nibbles | bit);
                }
            }
        }

        const bool route_initial_group =
            routing_single_map_enabled(routing, 0, group_end, true,
                                        group_end == schedule.entries.size()) &&
            !full_window &&
            !use_map1_source_prefilter;
        const float route_tau = route_initial_group
            ? routing_tau_for_entry(*routing, 0)
            : 0.0f;
        std::uint64_t routed_hashed = 0;
        std::uint64_t routed_passed = 0;

        if (use_map1_source_prefilter)
        {
            for (std::uint32_t base = 0; base < window; base += 256u)
            {
                const std::uint32_t page_start = data_start + base;
                const unsigned nibble = (page_start >> 8) & 0x0Fu;
                const bool try_page = (learned_nibbles & (1u << nibble)) != 0;
                if (try_page &&
                    map1_source_classes(
                        key, page_start, schedule, source_classes,
                        map1_source_prefilter->sample_max_classes))
                {
                    for (const Map1SourceClass& cls : source_classes)
                    {
                        tm.expand(key, page_start + cls.representative_offset);
                        run_map_group(tm, schedule, 0, group_end);
                        clock::time_point h0;
                        if (stats != nullptr) h0 = clock::now();
                        out_table.insert(tm.state_raw(), cls.multiplicity);
                        if (stats != nullptr) hash_ns += ns_since(h0);
                    }
                }
                else
                {
                    for (std::uint32_t local = 0; local < 256u; local++)
                    {
                        tm.expand(key, page_start + local);
                        run_map_group(tm, schedule, 0, group_end);
                        clock::time_point h0;
                        if (stats != nullptr) h0 = clock::now();
                        out_table.insert(tm.state_raw(), 1u);
                        if (stats != nullptr) hash_ns += ns_since(h0);
                    }
                }
            }
        }
        else if (route_initial_group && batch_inserts)
        {
            float current_score = 0.0f;
            auto run_one = [&](std::uint32_t j) {
                tm.expand(key, data_value_for_index(data_enumeration, data_start, data_stride, j));
                current_score = run_one_map_scored(tm, schedule.entries[0], routing->route_initial_proxy);
            };
            auto mult_of = [&](std::uint32_t) -> std::uint32_t { return source_multiplicity; };
            auto score_of = [&](std::uint32_t) -> float { return current_score; };
            std::vector<RoutedState> pass_buf;
            pass_buf.reserve(std::min<std::uint32_t>(window, 1024u));
            const auto routed = batched_run_insert_routed(
                tm, out_table, pass_buf, window, run_one, mult_of, score_of,
                route_tau, stats != nullptr);
            hash_ns += routed.first;
            routed_passed = routed.second;
            routed_hashed = effective_window - routed_passed;
            for (const RoutedState& carry : pass_buf)
                append_pool_state(out_table, carry.state.data(), carry.multiplicity);
        }
        else if (batch_inserts)
        {
            auto run_one = [&](std::uint32_t j) {
                tm.expand(key, data_value_for_index(data_enumeration, data_start, data_stride, j));
                run_map_group(tm, schedule, 0, group_end);
            };
            auto mult_of = [&](std::uint32_t) -> std::uint32_t { return source_multiplicity; };
            hash_ns += full_window
                ? batched_run_insert_full_u32(tm, out_table, run_one, mult_of, stats != nullptr)
                : batched_run_insert(tm, out_table, window, run_one, mult_of, stats != nullptr);
        }
        else if (route_initial_group)
        {
            std::vector<RoutedState> pass_buf;
            pass_buf.reserve(std::min<std::uint32_t>(window, 1024u));
            for_each_window_index(window, [&](std::uint32_t i)
            {
                tm.expand(key, data_value_for_index(data_enumeration, data_start, data_stride, i));
                const float score = run_one_map_scored(tm, schedule.entries[0], routing->route_initial_proxy);
                if (score >= route_tau)
                {
                    clock::time_point h0;
                    if (stats != nullptr) h0 = clock::now();
                    out_table.insert(tm.state_raw(), source_multiplicity);
                    if (stats != nullptr) hash_ns += ns_since(h0);
                    ++routed_hashed;
                }
                else
                {
                    RoutedState carry;
                    std::memcpy(carry.state.data(), tm.state_raw(), STATE_BYTES);
                    carry.multiplicity = source_multiplicity;
                    pass_buf.push_back(carry);
                    ++routed_passed;
                }
            });
            for (const RoutedState& carry : pass_buf)
                append_pool_state(out_table, carry.state.data(), carry.multiplicity);
        }
        else
        {
            for_each_window_index(window, [&](std::uint32_t i)
            {
                tm.expand(key, data_value_for_index(data_enumeration, data_start, data_stride, i));
                run_map_group(tm, schedule, 0, group_end);
                clock::time_point h0;
                if (stats != nullptr) h0 = clock::now();
                out_table.insert(tm.state_raw(), source_multiplicity);
                if (stats != nullptr) hash_ns += ns_since(h0);
            });
        }
        if (stats != nullptr)
        {
            BoundaryRecord rec;
            rec.entry_begin = 0;
            rec.entry_end = static_cast<std::uint32_t>(group_end);
            rec.frontier_in = represented_window;
            rec.frontier_out = out_table.pool.size();
            const std::uint64_t total_ns = ns_since(map_t0);
            rec.hash_ns = hash_ns;
            rec.map_ns = total_ns > hash_ns ? total_ns - hash_ns : 0;
            rec.routed_hashed = routed_hashed;
            rec.routed_passed = routed_passed;
            rec.routing_tau = route_tau;
            stats->records.push_back(rec);
        }
        entry_idx = group_end;
    }

    // Advance through schedule boundaries, double-buffering between out_table
    // (current frontier) and scratch_table (next frontier). The default
    // dedup_every_maps=1 dedups after every map. Larger values run multiple
    // maps before the next merge, which trades less hash-table work for less
    // intermediate collision collapse. first_dedup_maps can split out the
    // first group, e.g. first_dedup_maps=1 and dedup_every_maps=4 means
    // MAP1 -> merge, then four-map groups afterward.
    for (; entry_idx < schedule.entries.size(); )
    {
        const std::size_t group_end = next_group_end(
            entry_idx, schedule.entries.size(), dedup_every_maps,
            first_dedup_maps, checkpoint_entries);
        prepare_map_group(tm, schedule, entry_idx, group_end);
        const std::size_t frontier_in_size = out_table.pool.size();
        const std::uint32_t frontier_in = static_cast<std::uint32_t>(frontier_in_size);
        const bool is_last_group = (group_end == schedule.entries.size());
        const bool do_hash_this_group = !(skip_final_hash && is_last_group);
        const bool route_group =
            do_hash_this_group &&
            routing_single_map_enabled(routing, entry_idx, group_end, false, is_last_group);
        const float route_tau = route_group
            ? routing_tau_for_entry(*routing, static_cast<std::uint32_t>(entry_idx))
            : 0.0f;
        std::uint64_t routed_hashed = 0;
        std::uint64_t routed_passed = 0;

        // Final-hash skip: when the caller declares no downstream dedup
        // requirement, the last merge is pure overhead (no maps follow that
        // would benefit from frontier compaction). We still run the maps over
        // the in-frontier, but emit each post-map state directly into
        // scratch_table without hashing — preserving (a) the multiset
        // XOR-parity over states (since each input state's own multiplicity
        // carries forward) and (b) the per-data-value count. The downstream
        // consumer sees frontier_in states instead of frontier_in - last_drop
        // dedup'd states. The skip path also bypasses the slot-table zero-
        // fill, which would otherwise be a ~512KB write at typical frontiers.
        if (do_hash_this_group)  scratch_table.reset(frontier_in);
        else                     scratch_table.reset_pool_only(frontier_in);
        std::uint64_t hash_ns = 0;
        const auto map_t0 = clock::now();
        if (route_group && batch_inserts && !(routing != nullptr && routing->score_during_map))
        {
            std::vector<RoutedState> pass_buf;
            pass_buf.reserve(std::min<std::size_t>(frontier_in_size, 1024u));
            if constexpr (requires { &TM::run_maps_range_x12; })
            {
                alignas(64) std::uint8_t batchN[12 * STATE_BYTES];
                std::uint32_t mN[12];
                float scoreN[12];
                std::uint64_t fN[12];
                const std::size_t n = out_table.pool.size();
                std::size_t pi = 0;
                for (; pi + 12 <= n; pi += 12)
                {
                    for (int k = 0; k < 12; k++)
                    {
                        scoreN[k] = static_cast<float>(raw_alg0_count_for_entry(
                            tm, out_table.pool[pi + k].state.data(), schedule.entries[entry_idx]));
                        mN[k] = out_table.pool[pi + k].multiplicity;
                    }
                    tm.run_maps_range_x12(
                        schedule, entry_idx, group_end,
                        out_table.pool[pi+0].state.data(), out_table.pool[pi+1].state.data(), out_table.pool[pi+2].state.data(), out_table.pool[pi+3].state.data(), out_table.pool[pi+4].state.data(), out_table.pool[pi+5].state.data(),
                        out_table.pool[pi+6].state.data(), out_table.pool[pi+7].state.data(), out_table.pool[pi+8].state.data(), out_table.pool[pi+9].state.data(), out_table.pool[pi+10].state.data(), out_table.pool[pi+11].state.data(),
                        batchN+0*STATE_BYTES, batchN+1*STATE_BYTES, batchN+2*STATE_BYTES, batchN+3*STATE_BYTES, batchN+4*STATE_BYTES, batchN+5*STATE_BYTES,
                        batchN+6*STATE_BYTES, batchN+7*STATE_BYTES, batchN+8*STATE_BYTES, batchN+9*STATE_BYTES, batchN+10*STATE_BYTES, batchN+11*STATE_BYTES);
                    clock::time_point h0;
                    if (stats != nullptr) h0 = clock::now();
                    for (int k = 0; k < 12; k++)
                    {
                        if (scoreN[k] < route_tau) continue;
                        fN[k] = FlatTable::fingerprint(batchN + k * STATE_BYTES);
#if defined(__SSE__) || defined(_MSC_VER)
                        _mm_prefetch(reinterpret_cast<const char*>(
                            &scratch_table.slots[static_cast<std::uint32_t>(fN[k]) & scratch_table.mask]), _MM_HINT_T0);
#endif
                    }
                    for (int k = 0; k < 12; k++)
                    {
                        if (scoreN[k] >= route_tau)
                        {
                            scratch_table.insert_fp(batchN + k * STATE_BYTES, fN[k], mN[k]);
                            ++routed_hashed;
                        }
                        else
                        {
                            RoutedState carry;
                            std::memcpy(carry.state.data(), batchN + k * STATE_BYTES, STATE_BYTES);
                            carry.multiplicity = mN[k];
                            pass_buf.push_back(carry);
                            ++routed_passed;
                        }
                    }
                    if (stats != nullptr) hash_ns += ns_since(h0);
                }
                for (; pi < n; pi++)
                {
                    const FlatTable::Entry& entry = out_table.pool[pi];
                    const float score = static_cast<float>(raw_alg0_count_for_entry(
                        tm, entry.state.data(), schedule.entries[entry_idx]));
                    tm.load_state_raw(entry.state.data());
                    run_map_group(tm, schedule, entry_idx, group_end);
                    if (score >= route_tau)
                    {
                        clock::time_point h0;
                        if (stats != nullptr) h0 = clock::now();
                        scratch_table.insert(tm.state_raw(), entry.multiplicity);
                        if (stats != nullptr) hash_ns += ns_since(h0);
                        ++routed_hashed;
                    }
                    else
                    {
                        RoutedState carry;
                        std::memcpy(carry.state.data(), tm.state_raw(), STATE_BYTES);
                        carry.multiplicity = entry.multiplicity;
                        pass_buf.push_back(carry);
                        ++routed_passed;
                    }
                }
            }
            else if constexpr (requires { &TM::run_maps_range_x4; })
            {
                alignas(64) std::uint8_t batch4[4 * STATE_BYTES];
                std::uint32_t m4[4];
                float score4[4];
                std::uint64_t f4[4];
                const std::size_t n = out_table.pool.size();
                std::size_t pi = 0;
                for (; pi + 4 <= n; pi += 4)
                {
                    for (int k = 0; k < 4; k++)
                    {
                        score4[k] = static_cast<float>(raw_alg0_count_for_entry(
                            tm, out_table.pool[pi + k].state.data(), schedule.entries[entry_idx]));
                        m4[k] = out_table.pool[pi + k].multiplicity;
                    }
                    tm.run_maps_range_x4(
                        schedule, entry_idx, group_end,
                        out_table.pool[pi + 0].state.data(), out_table.pool[pi + 1].state.data(),
                        out_table.pool[pi + 2].state.data(), out_table.pool[pi + 3].state.data(),
                        batch4 + 0 * STATE_BYTES, batch4 + 1 * STATE_BYTES,
                        batch4 + 2 * STATE_BYTES, batch4 + 3 * STATE_BYTES);
                    clock::time_point h0;
                    if (stats != nullptr) h0 = clock::now();
                    for (int k = 0; k < 4; k++)
                    {
                        if (score4[k] < route_tau) continue;
                        f4[k] = FlatTable::fingerprint(batch4 + k * STATE_BYTES);
#if defined(__SSE__) || defined(_MSC_VER)
                        _mm_prefetch(reinterpret_cast<const char*>(
                            &scratch_table.slots[static_cast<std::uint32_t>(f4[k]) & scratch_table.mask]), _MM_HINT_T0);
#endif
                    }
                    for (int k = 0; k < 4; k++)
                    {
                        if (score4[k] >= route_tau)
                        {
                            scratch_table.insert_fp(batch4 + k * STATE_BYTES, f4[k], m4[k]);
                            ++routed_hashed;
                        }
                        else
                        {
                            RoutedState carry;
                            std::memcpy(carry.state.data(), batch4 + k * STATE_BYTES, STATE_BYTES);
                            carry.multiplicity = m4[k];
                            pass_buf.push_back(carry);
                            ++routed_passed;
                        }
                    }
                    if (stats != nullptr) hash_ns += ns_since(h0);
                }
                for (; pi < n; pi++)
                {
                    const FlatTable::Entry& entry = out_table.pool[pi];
                    const float score = static_cast<float>(raw_alg0_count_for_entry(
                        tm, entry.state.data(), schedule.entries[entry_idx]));
                    tm.load_state_raw(entry.state.data());
                    run_map_group(tm, schedule, entry_idx, group_end);
                    if (score >= route_tau)
                    {
                        clock::time_point h0;
                        if (stats != nullptr) h0 = clock::now();
                        scratch_table.insert(tm.state_raw(), entry.multiplicity);
                        if (stats != nullptr) hash_ns += ns_since(h0);
                        ++routed_hashed;
                    }
                    else
                    {
                        RoutedState carry;
                        std::memcpy(carry.state.data(), tm.state_raw(), STATE_BYTES);
                        carry.multiplicity = entry.multiplicity;
                        pass_buf.push_back(carry);
                        ++routed_passed;
                    }
                }
            }
            else
            {
                float current_score = 0.0f;
                const auto routed = batched_run_insert_routed(
                    tm, scratch_table, pass_buf, static_cast<std::uint32_t>(out_table.pool.size()),
                    [&](std::uint32_t j) {
                        tm.load_state_raw(out_table.pool[j].state.data());
                        current_score = static_cast<float>(raw_alg0_count_for_entry(
                            tm, tm.state_raw(), schedule.entries[entry_idx]));
                        run_map_group(tm, schedule, entry_idx, group_end);
                    },
                    [&](std::uint32_t j) -> std::uint32_t { return out_table.pool[j].multiplicity; },
                    [&](std::uint32_t) -> float { return current_score; },
                    route_tau, stats != nullptr);
                hash_ns += routed.first;
                routed_passed = routed.second;
                routed_hashed = frontier_in_size - routed_passed;
            }
            for (const RoutedState& carry : pass_buf)
                append_pool_state(scratch_table, carry.state.data(), carry.multiplicity);
        }
        else if (route_group)
        {
            std::vector<RoutedState> pass_buf;
            pass_buf.reserve(std::min<std::size_t>(frontier_in_size, 1024u));
            for (std::size_t pi = 0; pi < out_table.pool.size(); pi++)
            {
                const FlatTable::Entry& entry = out_table.pool[pi];
                tm.load_state_raw(entry.state.data());
                const float score = run_one_map_scored(tm, schedule.entries[entry_idx], false);
                if (score >= route_tau)
                {
                    clock::time_point h0;
                    if (stats != nullptr) h0 = clock::now();
                    scratch_table.insert(tm.state_raw(), entry.multiplicity);
                    if (stats != nullptr) hash_ns += ns_since(h0);
                    ++routed_hashed;
                }
                else
                {
                    RoutedState carry;
                    std::memcpy(carry.state.data(), tm.state_raw(), STATE_BYTES);
                    carry.multiplicity = entry.multiplicity;
                    pass_buf.push_back(carry);
                    ++routed_passed;
                }
            }
            for (const RoutedState& carry : pass_buf)
                append_pool_state(scratch_table, carry.state.data(), carry.multiplicity);
        }
        else if (do_hash_this_group && batch_inserts)
        {
            // Widest-available interleaved boundary replay is auto-selected: the
            // universal AVX-512 kernel exposes run_maps_range_x12 (production
            // width — the interleave-width study's spill-free ceiling, ~+12% over
            // x4 at 16t), so it takes the x12 path; kernels with only
            // run_maps_range_x4 (e.g. the map kernels) take the 4-way path; AVX2
            // (no interleave kernel) falls to the 1-way batched path. N frontier
            // states run through the interleaved kernel in lockstep so the OOO
            // engine hides each state's per-step serial-chain (dispatch-mispredict)
            // latency with the others' independent work, then the N results are
            // batch-inserted (prefetched probes). The output multiset is identical
            // to the 1-way path (insert order within a group does not change the
            // merged multiset), so every width is parity-preserving (verified
            // 761/544/1023/465). See docs/forward_interleave_prototype_20260531.md.
            if constexpr (requires { &TM::run_maps_range_x12; })
            {
                // 12-way interleaved boundary replay — the production width. The
                // interleave-width study (docs/forward_interleave_prototype_20260531.md)
                // found x12 the spill-free ceiling (GCC rematerializes the masks,
                // so 24 state ZMM fit the 32-ZMM file) and ~+12% over x4 at 16t.
                // Same parity-preserving batch-insert structure as the 4-way
                // fallback below; the <12 remainder runs 1-way.
                alignas(64) std::uint8_t batchN[12 * STATE_BYTES];
                std::uint32_t mN[12];
                std::uint64_t fN[12];
                const std::size_t n = out_table.pool.size();
                std::size_t pi = 0;
                for (; pi + 12 <= n; pi += 12)
                {
                    tm.run_maps_range_x12(
                        schedule, entry_idx, group_end,
                        out_table.pool[pi+0].state.data(), out_table.pool[pi+1].state.data(), out_table.pool[pi+2].state.data(), out_table.pool[pi+3].state.data(), out_table.pool[pi+4].state.data(), out_table.pool[pi+5].state.data(),
                        out_table.pool[pi+6].state.data(), out_table.pool[pi+7].state.data(), out_table.pool[pi+8].state.data(), out_table.pool[pi+9].state.data(), out_table.pool[pi+10].state.data(), out_table.pool[pi+11].state.data(),
                        batchN+0*STATE_BYTES, batchN+1*STATE_BYTES, batchN+2*STATE_BYTES, batchN+3*STATE_BYTES, batchN+4*STATE_BYTES, batchN+5*STATE_BYTES,
                        batchN+6*STATE_BYTES, batchN+7*STATE_BYTES, batchN+8*STATE_BYTES, batchN+9*STATE_BYTES, batchN+10*STATE_BYTES, batchN+11*STATE_BYTES);
                    for (int k = 0; k < 12; k++) mN[k] = out_table.pool[pi+k].multiplicity;
                    clock::time_point h0;
                    if (stats != nullptr) h0 = clock::now();
                    for (int k = 0; k < 12; k++)
                    {
                        fN[k] = FlatTable::fingerprint(batchN + k * STATE_BYTES);
#if defined(__SSE__) || defined(_MSC_VER)
                        _mm_prefetch(reinterpret_cast<const char*>(
                            &scratch_table.slots[static_cast<std::uint32_t>(fN[k]) & scratch_table.mask]), _MM_HINT_T0);
#endif
                    }
                    for (int k = 0; k < 12; k++)
                        scratch_table.insert_fp(batchN + k * STATE_BYTES, fN[k], mN[k]);
                    if (stats != nullptr) hash_ns += ns_since(h0);
                }
                // Remainder (<12 states): 1-way through the member working state.
                for (; pi < n; pi++)
                {
                    const FlatTable::Entry& entry = out_table.pool[pi];
                    tm.load_state_raw(entry.state.data());
                    run_map_group(tm, schedule, entry_idx, group_end);
                    clock::time_point h0;
                    if (stats != nullptr) h0 = clock::now();
                    scratch_table.insert(tm.state_raw(), entry.multiplicity);
                    if (stats != nullptr) hash_ns += ns_since(h0);
                }
            }
            else if constexpr (requires { &TM::run_maps_range_x4; })
            {
                alignas(64) std::uint8_t batch4[4 * STATE_BYTES];
                std::uint32_t m4[4];
                std::uint64_t f4[4];
                const std::size_t n = out_table.pool.size();
                std::size_t pi = 0;
                for (; pi + 4 <= n; pi += 4)
                {
                    tm.run_maps_range_x4(
                        schedule, entry_idx, group_end,
                        out_table.pool[pi + 0].state.data(), out_table.pool[pi + 1].state.data(),
                        out_table.pool[pi + 2].state.data(), out_table.pool[pi + 3].state.data(),
                        batch4 + 0 * STATE_BYTES, batch4 + 1 * STATE_BYTES,
                        batch4 + 2 * STATE_BYTES, batch4 + 3 * STATE_BYTES);
                    m4[0] = out_table.pool[pi + 0].multiplicity;
                    m4[1] = out_table.pool[pi + 1].multiplicity;
                    m4[2] = out_table.pool[pi + 2].multiplicity;
                    m4[3] = out_table.pool[pi + 3].multiplicity;
                    // Timing is pure BoundaryRecord instrumentation; skip the
                    // clock::now() pair entirely when no stats are requested
                    // (profiling showed it cost ~2% in __vdso_clock_gettime here).
                    clock::time_point h0;
                    if (stats != nullptr) h0 = clock::now();
                    for (int k = 0; k < 4; k++)
                    {
                        f4[k] = FlatTable::fingerprint(batch4 + k * STATE_BYTES);
#if defined(__SSE__) || defined(_MSC_VER)
                        _mm_prefetch(reinterpret_cast<const char*>(
                            &scratch_table.slots[static_cast<std::uint32_t>(f4[k]) & scratch_table.mask]), _MM_HINT_T0);
#endif
                    }
                    for (int k = 0; k < 4; k++)
                        scratch_table.insert_fp(batch4 + k * STATE_BYTES, f4[k], m4[k]);
                    if (stats != nullptr) hash_ns += ns_since(h0);
                }
                // Remainder (<4 states): 1-way through the member working state.
                for (; pi < n; pi++)
                {
                    const FlatTable::Entry& entry = out_table.pool[pi];
                    tm.load_state_raw(entry.state.data());
                    run_map_group(tm, schedule, entry_idx, group_end);
                    clock::time_point h0;
                    if (stats != nullptr) h0 = clock::now();
                    scratch_table.insert(tm.state_raw(), entry.multiplicity);
                    if (stats != nullptr) hash_ns += ns_since(h0);
                }
            }
            else
            {
                hash_ns += batched_run_insert(
                    tm, scratch_table, static_cast<std::uint32_t>(out_table.pool.size()),
                    [&](std::uint32_t j) { tm.load_state_raw(out_table.pool[j].state.data()); run_map_group(tm, schedule, entry_idx, group_end); },
                    [&](std::uint32_t j) -> std::uint32_t { return out_table.pool[j].multiplicity; },
                    stats != nullptr);
            }
        }
        else if (false && route_group)
        {
            std::vector<RoutedState> pass_buf;
            pass_buf.reserve(std::min<std::size_t>(frontier_in_size, 1024u));
            for (std::size_t pi = 0; pi < out_table.pool.size(); pi++)
            {
                const FlatTable::Entry& entry = out_table.pool[pi];
                tm.load_state_raw(entry.state.data());
                const float score = run_one_map_scored(tm, schedule.entries[entry_idx], false);
                if (score >= route_tau)
                {
                    clock::time_point h0;
                    if (stats != nullptr) h0 = clock::now();
                    scratch_table.insert(tm.state_raw(), entry.multiplicity);
                    if (stats != nullptr) hash_ns += ns_since(h0);
                    ++routed_hashed;
                }
                else
                {
                    RoutedState carry;
                    std::memcpy(carry.state.data(), tm.state_raw(), STATE_BYTES);
                    carry.multiplicity = entry.multiplicity;
                    pass_buf.push_back(carry);
                    ++routed_passed;
                }
            }
            for (const RoutedState& carry : pass_buf)
                append_pool_state(scratch_table, carry.state.data(), carry.multiplicity);
        }
        else if (do_hash_this_group)
        {
            for (std::size_t pi = 0; pi < out_table.pool.size(); pi++)
            {
                const FlatTable::Entry& entry = out_table.pool[pi];
                tm.load_state_raw(entry.state.data());
                run_map_group(tm, schedule, entry_idx, group_end);
                clock::time_point h0;
                if (stats != nullptr) h0 = clock::now();
                scratch_table.insert(tm.state_raw(), entry.multiplicity);
                if (stats != nullptr) hash_ns += ns_since(h0);
            }
        }
        else
        {
            // Skip-final-hash path: append post-map state directly to the
            // pool without hash-table probing. No slot-table maintenance —
            // mask stays at its reset() value but slot lookups are never
            // issued downstream when consumers iterate pool directly.
            scratch_table.pool.reserve(out_table.pool.size());
            for (std::size_t pi = 0; pi < out_table.pool.size(); pi++)
            {
                const FlatTable::Entry& entry = out_table.pool[pi];
                tm.load_state_raw(entry.state.data());
                run_map_group(tm, schedule, entry_idx, group_end);
                scratch_table.pool.emplace_back();
                std::memcpy(scratch_table.pool.back().state.data(),
                            tm.state_raw(), STATE_BYTES);
                scratch_table.pool.back().multiplicity = entry.multiplicity;
            }
        }
        if (stats != nullptr)
        {
            BoundaryRecord rec;
            rec.entry_begin = static_cast<std::uint32_t>(entry_idx);
            rec.entry_end = static_cast<std::uint32_t>(group_end);
            rec.frontier_in = frontier_in_size;
            rec.frontier_out = scratch_table.pool.size();
            const std::uint64_t total_ns = ns_since(map_t0);
            rec.hash_ns = hash_ns;
            rec.map_ns = total_ns > hash_ns ? total_ns - hash_ns : 0;
            rec.routed_hashed = routed_hashed;
            rec.routed_passed = routed_passed;
            rec.routing_tau = route_tau;
            stats->records.push_back(rec);
        }
        std::swap(out_table, scratch_table);
        entry_idx = group_end;
    }

    if (full_window)
    {
        out_table.dyn = old_out_dynamic;
        scratch_table.dyn = old_scratch_dynamic;
    }
}

// Convenience overload: allocates the scratch table internally. Use the
// two-table form in hot loops to amortize allocation.
template <typename TM>
inline void forward_block_with_dedup(
    TM& tm,
    std::uint32_t key,
    std::uint32_t data_start,
    std::uint32_t window,
    const key_schedule& schedule,
    FlatTable& out_table,
    std::uint32_t dedup_every_maps = 1,
    std::uint32_t first_dedup_maps = 0,
    const std::vector<std::uint32_t>* checkpoint_entries = nullptr,
    bool dedup_expanded_states = false,
    BoundaryStats* stats = nullptr,
    bool skip_final_hash = false,
    const Map1SourcePrefilterConfig* map1_source_prefilter = nullptr,
    bool batch_inserts = true,
    const DataEnumerationConfig* data_enumeration = nullptr,
    const RoutingConfig* routing = nullptr)
{
    FlatTable scratch;
    forward_block_with_dedup(tm, key, data_start, window, schedule,
                             out_table, scratch, dedup_every_maps, first_dedup_maps,
                             checkpoint_entries, dedup_expanded_states, stats,
                             skip_final_hash, map1_source_prefilter, batch_inserts,
                             /*data_stride=*/1, data_enumeration, routing);
}

} // namespace state_dedup

#endif // STATE_DEDUP_H
