#ifndef STATE_DEDUP_ORIGINS_H
#define STATE_DEDUP_ORIGINS_H

// Origin-tracking companion to state_dedup.h.
//
// This keeps the same flat open-addressed state table, but each unique state
// carries a linked list of data offsets that produced it. The final table can
// therefore screen one state once and emit every original data value that maps
// to that state.

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "data_sizes.h"
#include "key_schedule.h"

namespace state_dedup_origins
{

constexpr std::size_t STATE_BYTES = 128;
constexpr std::uint32_t NO_ORIGIN = std::numeric_limits<std::uint32_t>::max();

struct OriginTable
{
    struct Slot
    {
        std::uint64_t fingerprint;
        std::uint32_t pool_index;
    };

    struct OriginNode
    {
        std::uint32_t origin;
        std::uint32_t next;
    };

    struct Entry
    {
        std::array<std::uint8_t, STATE_BYTES> state;
        std::uint32_t multiplicity;
        std::uint32_t origin_head;
    };

    std::vector<Slot> slots;
    std::vector<Entry> pool;
    std::vector<OriginNode> origins;
    std::uint32_t mask = 0;

    inline void reset(std::uint32_t expected_states, std::uint32_t expected_origins)
    {
        std::uint32_t size = 16;
        while (size < expected_states * 2u) size <<= 1u;
        slots.assign(size, Slot{0, 0});
        mask = size - 1u;
        pool.clear();
        origins.clear();
        pool.reserve(expected_states);
        origins.reserve(expected_origins);
    }

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

    inline std::uint32_t copy_origins_from(
        const std::vector<OriginNode>& src,
        std::uint32_t src_head,
        std::uint32_t dst_head)
    {
        for (std::uint32_t n = src_head; n != NO_ORIGIN; n = src[n].next)
        {
            origins.push_back(OriginNode{src[n].origin, dst_head});
            dst_head = static_cast<std::uint32_t>(origins.size() - 1u);
        }
        return dst_head;
    }

    inline std::uint32_t push_origin(std::uint32_t origin, std::uint32_t dst_head)
    {
        origins.push_back(OriginNode{origin, dst_head});
        return static_cast<std::uint32_t>(origins.size() - 1u);
    }

    inline Entry& find_or_create(const std::uint8_t* state, std::uint64_t fp)
    {
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
                pool.back().multiplicity = 0;
                pool.back().origin_head = NO_ORIGIN;
                return pool.back();
            }
            if (slot.fingerprint == fp)
            {
                Entry& entry = pool[slot.pool_index];
                if (std::memcmp(entry.state.data(), state, STATE_BYTES) == 0)
                    return entry;
            }
            idx = (idx + 1u) & mask;
        }
    }

    inline void insert_single_origin(const std::uint8_t* state, std::uint32_t origin)
    {
        Entry& entry = find_or_create(state, fingerprint(state));
        entry.multiplicity += 1u;
        entry.origin_head = push_origin(origin, entry.origin_head);
    }

    inline void insert_origin_list(
        const std::uint8_t* state,
        std::uint32_t multiplicity,
        const std::vector<OriginNode>& src,
        std::uint32_t src_head)
    {
        Entry& entry = find_or_create(state, fingerprint(state));
        entry.multiplicity += multiplicity;
        entry.origin_head = copy_origins_from(src, src_head, entry.origin_head);
    }
};

template <typename TM>
void forward_block_with_origin_dedup(
    TM& tm,
    std::uint32_t key,
    std::uint32_t data_start,
    std::uint32_t window,
    const key_schedule& schedule,
    OriginTable& out_table,
    OriginTable& scratch_table)
{
    out_table.reset(window, window);
    for (std::uint32_t i = 0; i < window; i++)
    {
        tm.expand(key, data_start + i);
        out_table.insert_single_origin(tm.state_raw(), i);
    }

    for (auto entry_it = schedule.entries.begin();
         entry_it != schedule.entries.end();
         ++entry_it)
    {
        scratch_table.reset(
            static_cast<std::uint32_t>(out_table.pool.size()),
            static_cast<std::uint32_t>(out_table.origins.size()));

        for (std::size_t pi = 0; pi < out_table.pool.size(); pi++)
        {
            const OriginTable::Entry& entry = out_table.pool[pi];
            tm.load_state_raw(entry.state.data());
            tm.run_one_map(*entry_it);
            scratch_table.insert_origin_list(
                tm.state_raw(), entry.multiplicity,
                out_table.origins, entry.origin_head);
        }
        std::swap(out_table, scratch_table);
    }
}

template <typename TM>
inline void forward_block_with_origin_dedup(
    TM& tm,
    std::uint32_t key,
    std::uint32_t data_start,
    std::uint32_t window,
    const key_schedule& schedule,
    OriginTable& out_table)
{
    OriginTable scratch;
    forward_block_with_origin_dedup(
        tm, key, data_start, window, schedule, out_table, scratch);
}

} // namespace state_dedup_origins

#endif // STATE_DEDUP_ORIGINS_H
