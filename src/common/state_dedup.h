#ifndef STATE_DEDUP_H
#define STATE_DEDUP_H

// Production state-dedup primitive: open-addressed flat hash + templated
// forward-block driver for processing a contiguous range of data values
// against a key's forward schedule with per-boundary state-merge.
//
// Speedup characterization (Zen 5 9900X, 477 SAT-queued keys, base
// 0x12345600, single-thread):
//
//                    W=16   W=256  W=1024  W=4096
//   on tm_8:         1.12x  1.85x  2.18x   2.35x   (vs no-dedup baseline)
//   on tm_avx2_r256s_8 (combined SIMD + dedup, vs scalar baseline):
//                    ~2.9x  ~5.6x  ~5.3x   ~6.1x  (median)
//
// All cells parity-verified: dedup'd multiset of final states equals the
// baseline multiset.
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

    // FNV-1a over 16x uint64_t + murmur3 finalizer.
    // Result is never 0 (remapped to 1) so 0 stays usable as the empty marker.
    static inline std::uint64_t fingerprint(const std::uint8_t* state)
    {
        std::uint64_t h = 0xcbf29ce484222325ull;
        const std::uint64_t* w = reinterpret_cast<const std::uint64_t*>(state);
        for (int i = 0; i < 16; i++)
        {
            h ^= w[i];
            h *= 0x100000001b3ull;
        }
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
    FlatTable& scratch_table)
{
    // Build initial frontier from window consecutive expand() calls.
    out_table.reset(window);
    for (std::uint32_t i = 0; i < window; i++)
    {
        tm.expand(key, data_start + i);
        out_table.insert(tm.state_raw(), 1u);
    }

    // Advance through each schedule boundary, double-buffering between
    // out_table (current frontier) and scratch_table (next frontier).
    for (auto entry_it = schedule.entries.begin();
         entry_it != schedule.entries.end();
         ++entry_it)
    {
        scratch_table.reset(static_cast<std::uint32_t>(out_table.pool.size()));
        for (std::size_t pi = 0; pi < out_table.pool.size(); pi++)
        {
            const FlatTable::Entry& entry = out_table.pool[pi];
            tm.load_state_raw(entry.state.data());
            tm.run_one_map(*entry_it);
            scratch_table.insert(tm.state_raw(), entry.multiplicity);
        }
        std::swap(out_table, scratch_table);
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
    FlatTable& out_table)
{
    FlatTable scratch;
    forward_block_with_dedup(tm, key, data_start, window, schedule,
                             out_table, scratch);
}

} // namespace state_dedup

#endif // STATE_DEDUP_H
