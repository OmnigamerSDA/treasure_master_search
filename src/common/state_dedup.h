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
#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

#include "data_sizes.h"
#include "key_schedule.h"

namespace state_dedup
{

constexpr std::size_t STATE_BYTES = 128;

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
    struct Slot
    {
        std::uint64_t fingerprint;   // 0 == empty
        std::uint32_t pool_index;
    };

    struct Entry
    {
        std::array<std::uint8_t, STATE_BYTES> state;
        std::uint32_t multiplicity;
    };

    std::vector<Slot> slots;
    std::vector<Entry> pool;
    std::uint32_t mask = 0;

    // Resize the slot table to next-pow-2 of (expected_count * 2) and
    // clear the entry pool. Reuses underlying allocations where possible.
    inline void reset(std::uint32_t expected_count)
    {
        std::uint32_t size = 16;
        while (size < expected_count * 2u) size <<= 1u;
        slots.assign(size, Slot{0, 0});
        mask = size - 1u;
        pool.clear();
        pool.reserve(expected_count);
    }

    // Reset only the entry pool, skipping the slot-table zero-fill. Used by
    // forward_block_with_dedup's skip-final-hash path where the slot table is
    // never touched. Avoids the (typically) 512KB slot-array clear per call.
    inline void reset_pool_only(std::uint32_t expected_count)
    {
        pool.clear();
        pool.reserve(expected_count);
        mask = 0;
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
        const std::uint64_t fp = fingerprint(state);
        std::uint32_t idx = static_cast<std::uint32_t>(fp) & mask;
        while (true)
        {
            Slot& slot = slots[idx];
            if (slot.fingerprint == 0)
            {
                slot.fingerprint = fp;
                slot.pool_index = static_cast<std::uint32_t>(pool.size());
                pool.emplace_back();
                std::memcpy(pool.back().state.data(), state, STATE_BYTES);
                pool.back().multiplicity = mult;
                return;
            }
            if (slot.fingerprint == fp)
            {
                Entry& entry = pool[slot.pool_index];
                if (std::memcmp(entry.state.data(), state, STATE_BYTES) == 0)
                {
                    entry.multiplicity += mult;
                    return;
                }
            }
            idx = (idx + 1u) & mask;
        }
    }
};

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
    std::uint32_t frontier_in;     // unique states entering the group
    std::uint32_t frontier_out;    // unique states emitted after merge
    std::uint64_t map_ns;          // time spent running maps over the in-frontier
    std::uint64_t hash_ns;         // time spent inserting/merging into out-frontier
};

struct BoundaryStats
{
    std::vector<BoundaryRecord> records;
    inline void clear() { records.clear(); }
};

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
    bool skip_final_hash = false)
{
    using clock = std::chrono::steady_clock;
    auto ns_since = [](clock::time_point t0) -> std::uint64_t {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count());
    };

    if (dedup_every_maps == 0) dedup_every_maps = 1;
    if (first_dedup_maps == 0) first_dedup_maps = dedup_every_maps;

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
    // first map group before hashing.
    out_table.reset(window);
    if (dedup_expanded_states || schedule.entries.empty())
    {
        std::uint64_t hash_ns = 0;
        const auto map_t0 = clock::now();
        for (std::uint32_t i = 0; i < window; i++)
        {
            tm.expand(key, data_start + i);
            const auto h0 = clock::now();
            out_table.insert(tm.state_raw(), 1u);
            hash_ns += ns_since(h0);
        }
        if (stats != nullptr)
        {
            BoundaryRecord rec;
            rec.entry_begin = UINT32_MAX;
            rec.entry_end = 0;
            rec.frontier_in = window;
            rec.frontier_out = static_cast<std::uint32_t>(out_table.pool.size());
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
        for (std::uint32_t i = 0; i < window; i++)
        {
            tm.expand(key, data_start + i);
            run_map_group(tm, schedule, 0, group_end);
            const auto h0 = clock::now();
            out_table.insert(tm.state_raw(), 1u);
            hash_ns += ns_since(h0);
        }
        if (stats != nullptr)
        {
            BoundaryRecord rec;
            rec.entry_begin = 0;
            rec.entry_end = static_cast<std::uint32_t>(group_end);
            rec.frontier_in = window;
            rec.frontier_out = static_cast<std::uint32_t>(out_table.pool.size());
            const std::uint64_t total_ns = ns_since(map_t0);
            rec.hash_ns = hash_ns;
            rec.map_ns = total_ns > hash_ns ? total_ns - hash_ns : 0;
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
        const std::uint32_t frontier_in = static_cast<std::uint32_t>(out_table.pool.size());
        const bool is_last_group = (group_end == schedule.entries.size());
        const bool do_hash_this_group = !(skip_final_hash && is_last_group);

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
        if (do_hash_this_group)
        {
            for (std::size_t pi = 0; pi < out_table.pool.size(); pi++)
            {
                const FlatTable::Entry& entry = out_table.pool[pi];
                tm.load_state_raw(entry.state.data());
                run_map_group(tm, schedule, entry_idx, group_end);
                const auto h0 = clock::now();
                scratch_table.insert(tm.state_raw(), entry.multiplicity);
                hash_ns += ns_since(h0);
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
            rec.frontier_in = frontier_in;
            rec.frontier_out = static_cast<std::uint32_t>(scratch_table.pool.size());
            const std::uint64_t total_ns = ns_since(map_t0);
            rec.hash_ns = hash_ns;
            rec.map_ns = total_ns > hash_ns ? total_ns - hash_ns : 0;
            stats->records.push_back(rec);
        }
        std::swap(out_table, scratch_table);
        entry_idx = group_end;
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
    bool skip_final_hash = false)
{
    FlatTable scratch;
    forward_block_with_dedup(tm, key, data_start, window, schedule,
                             out_table, scratch, dedup_every_maps, first_dedup_maps,
                             checkpoint_entries, dedup_expanded_states, stats,
                             skip_final_hash);
}

} // namespace state_dedup

#endif // STATE_DEDUP_H
