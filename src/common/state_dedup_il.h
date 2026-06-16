#ifndef STATE_DEDUP_IL_H
#define STATE_DEDUP_IL_H

// PROTOTYPE: interleaved (2-way) variant of forward_block_with_dedup.
//
// Goal: measure whether processing the frontier pool two states at a time
// through the per-boundary map group (tm.run_maps_range_x2) recovers any of
// the dependent-SIMD / branch-mispredict / hash-probe latency that bounds the
// production dedup path (which is NOT DRAM-bound — per-boundary RNG footprint
// is L1/L2-resident, ~3.7% L1 miss). See the forward-interleaving analysis.
//
// Scope: this slim driver supports only the production-default f1k4 shape that
// state_dedup_screen_bench uses for measurement:
//   - dedup_expanded_states = false  (skip post-expand merge; expand is
//     injective in data for fixed key)
//   - no MAP1 source prefilter
//   - no skip_final_hash
// It is intentionally NOT a drop-in for every option of the production driver;
// it exists to A/B the interleave against the production path on the default
// config. The initial frontier build (expand + first map group) stays 1-way;
// only the boundary-advance replay loop is interleaved, which is where the
// bulk of per-state map work lives.
//
// Requirements on TM (compile-time): expand, run_maps_range (1-way fallback),
// run_maps_range_x2, state_raw, load_state_raw, bind_schedule (optional).

#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__SSE2__) || defined(_MSC_VER)
#include <emmintrin.h>  // _mm_prefetch
#endif

#include "data_sizes.h"
#include "key_schedule.h"
#include "state_dedup.h"

namespace state_dedup
{

// ── (A) decoupled insert-batching ────────────────────────────────────────────
// Keeps the map kernel 1-way (no register/port regression) but software-
// pipelines the FlatTable inserts: compute B fingerprints, prefetch the B
// initial slots, then resolve the B inserts. This overlaps the random
// slot-probe L2/L3 latency that dominates hash time at the heavy MAP1 boundary
// (43% hash share). Insert order within a batch does not change the output
// multiset, so parity with the production driver is preserved.

// Delegates to the authoritative FlatTable::insert_fp so the interleave paths
// inherit its dynamic grow+rehash safety net, tier-1 front cache, and trace hook
// — without these, --dynamic-table (floor-sized tables) would overflow the
// boundary-loop scratch table here. The fc/grow/trace branches are predictably
// not-taken when those features are off (default), so the production x8 path is
// unaffected. Output multiset is identical (same probe logic), parity preserved.
inline void il_insert_fp(FlatTable& t, const std::uint8_t* state, std::uint64_t fp, std::uint32_t mult)
{
    t.insert_fp(state, fp, mult);
}

// Insert a batch of [count] (state, mult) results into t, prefetching all
// initial slots before resolving any probe. states is count*STATE_BYTES bytes.
inline void il_insert_batch(FlatTable& t, const std::uint8_t* states, const std::uint32_t* mults,
                            std::uint64_t* fps, std::size_t count)
{
    for (std::size_t b = 0; b < count; b++)
    {
        fps[b] = FlatTable::fingerprint(states + b * STATE_BYTES);
#if defined(__SSE2__) || defined(_MSC_VER)
        _mm_prefetch(reinterpret_cast<const char*>(&t.slots[static_cast<std::uint32_t>(fps[b]) & t.mask]), _MM_HINT_T0);
#endif
    }
    for (std::size_t b = 0; b < count; b++)
        il_insert_fp(t, states + b * STATE_BYTES, fps[b], mults[b]);
}


// 2-way interleaved frontier replay. Same output multiset as
// forward_block_with_dedup for the default f1k4 config (parity-checked).
template <typename TM>
void forward_block_with_dedup_il2(
    TM& tm,
    std::uint32_t key,
    std::uint32_t data_start,
    std::uint32_t window,
    const key_schedule& schedule,
    FlatTable& out_table,
    FlatTable& scratch_table,
    std::uint32_t dedup_every_maps = 4,
    std::uint32_t first_dedup_maps = 1,
    BoundaryStats* stats = nullptr)
{
    using clock = std::chrono::steady_clock;
    auto ns_since = [](clock::time_point t0) -> std::uint64_t {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count());
    };

    if (dedup_every_maps == 0) dedup_every_maps = 1;
    if (first_dedup_maps == 0) first_dedup_maps = dedup_every_maps;

#ifndef DEDUP_NO_BIND_SCHEDULE
    if constexpr (requires(TM& t, const key_schedule& s) { t.bind_dedup_schedule(s); })
        tm.bind_dedup_schedule(schedule);
    else if constexpr (requires(TM& t, const key_schedule& s) { t.bind_schedule(s); })
        tm.bind_schedule(schedule);
#endif

    std::size_t entry_idx = 0;

    // Initial frontier build (1-way): expand each data value, run the first map
    // group, hash. Identical to the production default (no expand-dedup, no
    // prefilter).
    out_table.reset(window);
    {
        const std::size_t group_end = next_group_end(
            0, schedule.entries.size(), dedup_every_maps, first_dedup_maps, nullptr);
        prepare_map_group(tm, schedule, 0, group_end);
        const auto map_t0 = clock::now();
        const std::uint64_t hash_ns = batched_run_insert(
            tm, out_table, window,
            [&](std::uint32_t j) { tm.expand(key, data_start + j); run_map_group(tm, schedule, 0, group_end); },
            [](std::uint32_t) -> std::uint32_t { return 1u; },
            stats != nullptr);
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

    // Contiguous 2-state scratch so the interleaved pair can be batch-inserted
    // (prefetched probes), matching the production batched-insert path.
    alignas(32) std::uint8_t batch[2 * STATE_BYTES];
    std::uint32_t mults[2];
    std::uint64_t fps[2];

    // Boundary advance, interleaved 2-wide over the frontier pool.
    for (; entry_idx < schedule.entries.size(); )
    {
        const std::size_t group_end = next_group_end(
            entry_idx, schedule.entries.size(), dedup_every_maps, first_dedup_maps, nullptr);
        prepare_map_group(tm, schedule, entry_idx, group_end);

        const std::uint32_t frontier_in = static_cast<std::uint32_t>(out_table.pool.size());
        scratch_table.reset(frontier_in);

        std::uint64_t hash_ns = 0;
        const auto map_t0 = clock::now();

        std::size_t pi = 0;
        const std::size_t n = out_table.pool.size();
        for (; pi + 1 < n; pi += 2)
        {
            tm.run_maps_range_x2(schedule, entry_idx, group_end,
                                 out_table.pool[pi].state.data(), out_table.pool[pi + 1].state.data(),
                                 batch, batch + STATE_BYTES);
            mults[0] = out_table.pool[pi].multiplicity;
            mults[1] = out_table.pool[pi + 1].multiplicity;
            clock::time_point h0;
            if (stats != nullptr) h0 = clock::now();
            il_insert_batch(scratch_table, batch, mults, fps, 2);
            if (stats != nullptr) hash_ns += ns_since(h0);
        }
        // Odd remainder: 1-way through the member working state.
        for (; pi < n; pi++)
        {
            const FlatTable::Entry& e = out_table.pool[pi];
            tm.load_state_raw(e.state.data());
            run_map_group(tm, schedule, entry_idx, group_end);
            clock::time_point h0;
            if (stats != nullptr) h0 = clock::now();
            scratch_table.insert(tm.state_raw(), e.multiplicity);
            if (stats != nullptr) hash_ns += ns_since(h0);
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

// (A) batch-hash driver: 1-way map kernel, software-pipelined batched inserts.
// Same f1k4-default scope as forward_block_with_dedup_il2. Parity-equivalent
// to the production driver's output multiset.
template <typename TM>
void forward_block_with_dedup_bh(
    TM& tm,
    std::uint32_t key,
    std::uint32_t data_start,
    std::uint32_t window,
    const key_schedule& schedule,
    FlatTable& out_table,
    FlatTable& scratch_table,
    std::uint32_t dedup_every_maps = 4,
    std::uint32_t first_dedup_maps = 1,
    BoundaryStats* stats = nullptr)
{
    using clock = std::chrono::steady_clock;
    auto ns_since = [](clock::time_point t0) -> std::uint64_t {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count());
    };

    if (dedup_every_maps == 0) dedup_every_maps = 1;
    if (first_dedup_maps == 0) first_dedup_maps = dedup_every_maps;

#ifndef DEDUP_NO_BIND_SCHEDULE
    if constexpr (requires(TM& t, const key_schedule& s) { t.bind_dedup_schedule(s); })
        tm.bind_dedup_schedule(schedule);
    else if constexpr (requires(TM& t, const key_schedule& s) { t.bind_schedule(s); })
        tm.bind_schedule(schedule);
#endif

    constexpr std::size_t BH = 8;  // insert batch / prefetch depth
    alignas(32) std::uint8_t batch_states[BH * STATE_BYTES];
    std::uint32_t batch_mults[BH];
    std::uint64_t batch_fps[BH];

    std::size_t entry_idx = 0;

    // Initial frontier build: 1-way expand+map, batched inserts. This is the
    // boundary with the highest hash share (~43% at MAP1), so it benefits most.
    out_table.reset(window);
    {
        const std::size_t group_end = next_group_end(
            0, schedule.entries.size(), dedup_every_maps, first_dedup_maps, nullptr);
        prepare_map_group(tm, schedule, 0, group_end);
        std::uint64_t hash_ns = 0;
        const auto map_t0 = clock::now();
        std::uint32_t i = 0;
        while (i < window)
        {
            const std::size_t b = static_cast<std::size_t>(std::min<std::uint32_t>(BH, window - i));
            for (std::size_t k = 0; k < b; k++)
            {
                tm.expand(key, data_start + i + static_cast<std::uint32_t>(k));
                run_map_group(tm, schedule, 0, group_end);
                std::memcpy(batch_states + k * STATE_BYTES, tm.state_raw(), STATE_BYTES);
                batch_mults[k] = 1u;
            }
            clock::time_point h0;
            if (stats != nullptr) h0 = clock::now();
            il_insert_batch(out_table, batch_states, batch_mults, batch_fps, b);
            if (stats != nullptr) hash_ns += ns_since(h0);
            i += static_cast<std::uint32_t>(b);
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

    // Boundary advance: 1-way map replay, batched inserts.
    for (; entry_idx < schedule.entries.size(); )
    {
        const std::size_t group_end = next_group_end(
            entry_idx, schedule.entries.size(), dedup_every_maps, first_dedup_maps, nullptr);
        prepare_map_group(tm, schedule, entry_idx, group_end);

        const std::uint32_t frontier_in = static_cast<std::uint32_t>(out_table.pool.size());
        scratch_table.reset(frontier_in);

        std::uint64_t hash_ns = 0;
        const auto map_t0 = clock::now();

        const std::size_t n = out_table.pool.size();
        std::size_t pi = 0;
        while (pi < n)
        {
            const std::size_t b = std::min<std::size_t>(BH, n - pi);
            for (std::size_t k = 0; k < b; k++)
            {
                const FlatTable::Entry& e = out_table.pool[pi + k];
                tm.load_state_raw(e.state.data());
                run_map_group(tm, schedule, entry_idx, group_end);
                std::memcpy(batch_states + k * STATE_BYTES, tm.state_raw(), STATE_BYTES);
                batch_mults[k] = e.multiplicity;
            }
            clock::time_point h0;
            if (stats != nullptr) h0 = clock::now();
            il_insert_batch(scratch_table, batch_states, batch_mults, batch_fps, b);
            if (stats != nullptr) hash_ns += ns_since(h0);
            pi += b;
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

// PROTOTYPE: 4-way interleaved frontier replay (for AVX-512, which exposes
// run_maps_range_x4). Initial build uses the production batched-insert path;
// the boundary-advance loop replays the frontier four states at a time through
// the interleaved kernel and batch-inserts the four results. Only instantiated
// for TMs that have run_maps_range_x4 (guarded at the call site).
template <typename TM>
void forward_block_with_dedup_il4(
    TM& tm,
    std::uint32_t key,
    std::uint32_t data_start,
    std::uint32_t window,
    const key_schedule& schedule,
    FlatTable& out_table,
    FlatTable& scratch_table,
    std::uint32_t dedup_every_maps = 4,
    std::uint32_t first_dedup_maps = 1,
    BoundaryStats* stats = nullptr)
{
    using clock = std::chrono::steady_clock;
    auto ns_since = [](clock::time_point t0) -> std::uint64_t {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count());
    };

    if (dedup_every_maps == 0) dedup_every_maps = 1;
    if (first_dedup_maps == 0) first_dedup_maps = dedup_every_maps;

#ifndef DEDUP_NO_BIND_SCHEDULE
    if constexpr (requires(TM& t, const key_schedule& s) { t.bind_dedup_schedule(s); })
        tm.bind_dedup_schedule(schedule);
    else if constexpr (requires(TM& t, const key_schedule& s) { t.bind_schedule(s); })
        tm.bind_schedule(schedule);
#endif

    std::size_t entry_idx = 0;

    // Initial frontier build: 1-way expand+map, batched inserts (production default).
    out_table.reset(window);
    {
        const std::size_t group_end = next_group_end(
            0, schedule.entries.size(), dedup_every_maps, first_dedup_maps, nullptr);
        prepare_map_group(tm, schedule, 0, group_end);
        const auto map_t0 = clock::now();
        const std::uint64_t hash_ns = batched_run_insert(
            tm, out_table, window,
            [&](std::uint32_t j) { tm.expand(key, data_start + j); run_map_group(tm, schedule, 0, group_end); },
            [](std::uint32_t) -> std::uint32_t { return 1u; },
            stats != nullptr);
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

    alignas(64) std::uint8_t batch[4 * STATE_BYTES];
    std::uint32_t mults[4];
    std::uint64_t fps[4];

    for (; entry_idx < schedule.entries.size(); )
    {
        const std::size_t group_end = next_group_end(
            entry_idx, schedule.entries.size(), dedup_every_maps, first_dedup_maps, nullptr);
        prepare_map_group(tm, schedule, entry_idx, group_end);

        const std::uint32_t frontier_in = static_cast<std::uint32_t>(out_table.pool.size());
        scratch_table.reset(frontier_in);

        std::uint64_t hash_ns = 0;
        const auto map_t0 = clock::now();

        const std::size_t n = out_table.pool.size();
        std::size_t pi = 0;
        for (; pi + 4 <= n; pi += 4)
        {
            tm.run_maps_range_x4(
                schedule, entry_idx, group_end,
                out_table.pool[pi + 0].state.data(), out_table.pool[pi + 1].state.data(),
                out_table.pool[pi + 2].state.data(), out_table.pool[pi + 3].state.data(),
                batch + 0 * STATE_BYTES, batch + 1 * STATE_BYTES,
                batch + 2 * STATE_BYTES, batch + 3 * STATE_BYTES);
            mults[0] = out_table.pool[pi + 0].multiplicity;
            mults[1] = out_table.pool[pi + 1].multiplicity;
            mults[2] = out_table.pool[pi + 2].multiplicity;
            mults[3] = out_table.pool[pi + 3].multiplicity;
            clock::time_point h0;
            if (stats != nullptr) h0 = clock::now();
            il_insert_batch(scratch_table, batch, mults, fps, 4);
            if (stats != nullptr) hash_ns += ns_since(h0);
        }
        for (; pi < n; pi++)
        {
            const FlatTable::Entry& e = out_table.pool[pi];
            tm.load_state_raw(e.state.data());
            run_map_group(tm, schedule, entry_idx, group_end);
            clock::time_point h0;
            if (stats != nullptr) h0 = clock::now();
            scratch_table.insert(tm.state_raw(), e.multiplicity);
            if (stats != nullptr) hash_ns += ns_since(h0);
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

// PROTOTYPE: 6-way interleaved frontier replay. Mirrors il4 but batches the
// boundary-advance frontier six states at a time through run_maps_range_x6.
// Only instantiated for TMs that expose run_maps_range_x6 (guarded at the call
// site). Built to test whether wider interleave hides more of the dedup
// forward's mispredict latency (profiling: 51-54% frontend-stalled, IPC 0.81 on
// low-R keys) than 4-way; expected gain concentrated on low-R/high-frontier keys.
template <typename TM>
void forward_block_with_dedup_il6(
    TM& tm,
    std::uint32_t key,
    std::uint32_t data_start,
    std::uint32_t window,
    const key_schedule& schedule,
    FlatTable& out_table,
    FlatTable& scratch_table,
    std::uint32_t dedup_every_maps = 4,
    std::uint32_t first_dedup_maps = 1,
    BoundaryStats* stats = nullptr)
{
    using clock = std::chrono::steady_clock;
    auto ns_since = [](clock::time_point t0) -> std::uint64_t {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count());
    };

    if (dedup_every_maps == 0) dedup_every_maps = 1;
    if (first_dedup_maps == 0) first_dedup_maps = dedup_every_maps;

#ifndef DEDUP_NO_BIND_SCHEDULE
    if constexpr (requires(TM& t, const key_schedule& s) { t.bind_dedup_schedule(s); })
        tm.bind_dedup_schedule(schedule);
    else if constexpr (requires(TM& t, const key_schedule& s) { t.bind_schedule(s); })
        tm.bind_schedule(schedule);
#endif

    std::size_t entry_idx = 0;

    // Initial frontier build: 1-way expand+map, batched inserts (production default).
    out_table.reset(window);
    {
        const std::size_t group_end = next_group_end(
            0, schedule.entries.size(), dedup_every_maps, first_dedup_maps, nullptr);
        prepare_map_group(tm, schedule, 0, group_end);
        const auto map_t0 = clock::now();
        const std::uint64_t hash_ns = batched_run_insert(
            tm, out_table, window,
            [&](std::uint32_t j) { tm.expand(key, data_start + j); run_map_group(tm, schedule, 0, group_end); },
            [](std::uint32_t) -> std::uint32_t { return 1u; },
            stats != nullptr);
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

    alignas(64) std::uint8_t batch[6 * STATE_BYTES];
    std::uint32_t mults[6];
    std::uint64_t fps[6];

    for (; entry_idx < schedule.entries.size(); )
    {
        const std::size_t group_end = next_group_end(
            entry_idx, schedule.entries.size(), dedup_every_maps, first_dedup_maps, nullptr);
        prepare_map_group(tm, schedule, entry_idx, group_end);

        const std::uint32_t frontier_in = static_cast<std::uint32_t>(out_table.pool.size());
        scratch_table.reset(frontier_in);

        std::uint64_t hash_ns = 0;
        const auto map_t0 = clock::now();

        const std::size_t n = out_table.pool.size();
        std::size_t pi = 0;
        for (; pi + 6 <= n; pi += 6)
        {
            tm.run_maps_range_x6(
                schedule, entry_idx, group_end,
                out_table.pool[pi + 0].state.data(), out_table.pool[pi + 1].state.data(),
                out_table.pool[pi + 2].state.data(), out_table.pool[pi + 3].state.data(),
                out_table.pool[pi + 4].state.data(), out_table.pool[pi + 5].state.data(),
                batch + 0 * STATE_BYTES, batch + 1 * STATE_BYTES,
                batch + 2 * STATE_BYTES, batch + 3 * STATE_BYTES,
                batch + 4 * STATE_BYTES, batch + 5 * STATE_BYTES);
            for (int k = 0; k < 6; k++) mults[k] = out_table.pool[pi + k].multiplicity;
            clock::time_point h0;
            if (stats != nullptr) h0 = clock::now();
            il_insert_batch(scratch_table, batch, mults, fps, 6);
            if (stats != nullptr) hash_ns += ns_since(h0);
        }
        for (; pi < n; pi++)
        {
            const FlatTable::Entry& e = out_table.pool[pi];
            tm.load_state_raw(e.state.data());
            run_map_group(tm, schedule, entry_idx, group_end);
            clock::time_point h0;
            if (stats != nullptr) h0 = clock::now();
            scratch_table.insert(tm.state_raw(), e.multiplicity);
            if (stats != nullptr) hash_ns += ns_since(h0);
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

// PROTOTYPE: 8-way / 10-way interleaved frontier replay (interleave-width study).
// Mirror il6; batch the boundary frontier 8 / 10 states at a time.
template <typename TM>
void forward_block_with_dedup_il8(
    TM& tm, std::uint32_t key, std::uint32_t data_start, std::uint32_t window,
    const key_schedule& schedule, FlatTable& out_table, FlatTable& scratch_table,
    std::uint32_t dedup_every_maps = 4, std::uint32_t first_dedup_maps = 1, BoundaryStats* stats = nullptr)
{
    using clock = std::chrono::steady_clock;
    auto ns_since = [](clock::time_point t0) -> std::uint64_t {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count()); };
    if (dedup_every_maps == 0) dedup_every_maps = 1;
    if (first_dedup_maps == 0) first_dedup_maps = dedup_every_maps;
#ifndef DEDUP_NO_BIND_SCHEDULE
    if constexpr (requires(TM& t, const key_schedule& s) { t.bind_dedup_schedule(s); }) tm.bind_dedup_schedule(schedule);
    else if constexpr (requires(TM& t, const key_schedule& s) { t.bind_schedule(s); }) tm.bind_schedule(schedule);
#endif
    std::size_t entry_idx = 0;
    out_table.reset(window);
    {
        const std::size_t group_end = next_group_end(0, schedule.entries.size(), dedup_every_maps, first_dedup_maps, nullptr);
        prepare_map_group(tm, schedule, 0, group_end);
        const auto map_t0 = clock::now();
        const std::uint64_t hash_ns = batched_run_insert(tm, out_table, window,
            [&](std::uint32_t j) { tm.expand(key, data_start + j); run_map_group(tm, schedule, 0, group_end); },
            [](std::uint32_t) -> std::uint32_t { return 1u; }, stats != nullptr);
        if (stats != nullptr) { BoundaryRecord rec; rec.entry_begin = 0; rec.entry_end = (std::uint32_t)group_end;
            rec.frontier_in = window; rec.frontier_out = (std::uint32_t)out_table.pool.size();
            const std::uint64_t t = ns_since(map_t0); rec.hash_ns = hash_ns; rec.map_ns = t > hash_ns ? t - hash_ns : 0; stats->records.push_back(rec); }
        entry_idx = group_end;
    }
    alignas(64) std::uint8_t batch[8 * STATE_BYTES];
    std::uint32_t mults[8]; std::uint64_t fps[8];
    for (; entry_idx < schedule.entries.size(); )
    {
        const std::size_t group_end = next_group_end(entry_idx, schedule.entries.size(), dedup_every_maps, first_dedup_maps, nullptr);
        prepare_map_group(tm, schedule, entry_idx, group_end);
        const std::uint32_t frontier_in = (std::uint32_t)out_table.pool.size();
        scratch_table.reset(frontier_in);
        std::uint64_t hash_ns = 0; const auto map_t0 = clock::now();
        const std::size_t n = out_table.pool.size(); std::size_t pi = 0;
        for (; pi + 8 <= n; pi += 8)
        {
            tm.run_maps_range_x8(schedule, entry_idx, group_end,
                out_table.pool[pi+0].state.data(), out_table.pool[pi+1].state.data(), out_table.pool[pi+2].state.data(), out_table.pool[pi+3].state.data(),
                out_table.pool[pi+4].state.data(), out_table.pool[pi+5].state.data(), out_table.pool[pi+6].state.data(), out_table.pool[pi+7].state.data(),
                batch+0*STATE_BYTES, batch+1*STATE_BYTES, batch+2*STATE_BYTES, batch+3*STATE_BYTES,
                batch+4*STATE_BYTES, batch+5*STATE_BYTES, batch+6*STATE_BYTES, batch+7*STATE_BYTES);
            for (int k = 0; k < 8; k++) mults[k] = out_table.pool[pi+k].multiplicity;
            clock::time_point h0; if (stats != nullptr) h0 = clock::now();
            il_insert_batch(scratch_table, batch, mults, fps, 8);
            if (stats != nullptr) hash_ns += ns_since(h0);
        }
        for (; pi < n; pi++) {
            const FlatTable::Entry& e = out_table.pool[pi];
            tm.load_state_raw(e.state.data()); run_map_group(tm, schedule, entry_idx, group_end);
            clock::time_point h0; if (stats != nullptr) h0 = clock::now();
            scratch_table.insert(tm.state_raw(), e.multiplicity);
            if (stats != nullptr) hash_ns += ns_since(h0);
        }
        if (stats != nullptr) { BoundaryRecord rec; rec.entry_begin = (std::uint32_t)entry_idx; rec.entry_end = (std::uint32_t)group_end;
            rec.frontier_in = frontier_in; rec.frontier_out = (std::uint32_t)scratch_table.pool.size();
            const std::uint64_t t = ns_since(map_t0); rec.hash_ns = hash_ns; rec.map_ns = t > hash_ns ? t - hash_ns : 0; stats->records.push_back(rec); }
        std::swap(out_table, scratch_table); entry_idx = group_end;
    }
}

template <typename TM>
void forward_block_with_dedup_il10(
    TM& tm, std::uint32_t key, std::uint32_t data_start, std::uint32_t window,
    const key_schedule& schedule, FlatTable& out_table, FlatTable& scratch_table,
    std::uint32_t dedup_every_maps = 4, std::uint32_t first_dedup_maps = 1, BoundaryStats* stats = nullptr)
{
    using clock = std::chrono::steady_clock;
    auto ns_since = [](clock::time_point t0) -> std::uint64_t {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count()); };
    if (dedup_every_maps == 0) dedup_every_maps = 1;
    if (first_dedup_maps == 0) first_dedup_maps = dedup_every_maps;
#ifndef DEDUP_NO_BIND_SCHEDULE
    if constexpr (requires(TM& t, const key_schedule& s) { t.bind_dedup_schedule(s); }) tm.bind_dedup_schedule(schedule);
    else if constexpr (requires(TM& t, const key_schedule& s) { t.bind_schedule(s); }) tm.bind_schedule(schedule);
#endif
    std::size_t entry_idx = 0;
    out_table.reset(window);
    {
        const std::size_t group_end = next_group_end(0, schedule.entries.size(), dedup_every_maps, first_dedup_maps, nullptr);
        prepare_map_group(tm, schedule, 0, group_end);
        const auto map_t0 = clock::now();
        const std::uint64_t hash_ns = batched_run_insert(tm, out_table, window,
            [&](std::uint32_t j) { tm.expand(key, data_start + j); run_map_group(tm, schedule, 0, group_end); },
            [](std::uint32_t) -> std::uint32_t { return 1u; }, stats != nullptr);
        if (stats != nullptr) { BoundaryRecord rec; rec.entry_begin = 0; rec.entry_end = (std::uint32_t)group_end;
            rec.frontier_in = window; rec.frontier_out = (std::uint32_t)out_table.pool.size();
            const std::uint64_t t = ns_since(map_t0); rec.hash_ns = hash_ns; rec.map_ns = t > hash_ns ? t - hash_ns : 0; stats->records.push_back(rec); }
        entry_idx = group_end;
    }
    alignas(64) std::uint8_t batch[10 * STATE_BYTES];
    std::uint32_t mults[10]; std::uint64_t fps[10];
    for (; entry_idx < schedule.entries.size(); )
    {
        const std::size_t group_end = next_group_end(entry_idx, schedule.entries.size(), dedup_every_maps, first_dedup_maps, nullptr);
        prepare_map_group(tm, schedule, entry_idx, group_end);
        const std::uint32_t frontier_in = (std::uint32_t)out_table.pool.size();
        scratch_table.reset(frontier_in);
        std::uint64_t hash_ns = 0; const auto map_t0 = clock::now();
        const std::size_t n = out_table.pool.size(); std::size_t pi = 0;
        for (; pi + 10 <= n; pi += 10)
        {
            tm.run_maps_range_x10(schedule, entry_idx, group_end,
                out_table.pool[pi+0].state.data(), out_table.pool[pi+1].state.data(), out_table.pool[pi+2].state.data(), out_table.pool[pi+3].state.data(), out_table.pool[pi+4].state.data(),
                out_table.pool[pi+5].state.data(), out_table.pool[pi+6].state.data(), out_table.pool[pi+7].state.data(), out_table.pool[pi+8].state.data(), out_table.pool[pi+9].state.data(),
                batch+0*STATE_BYTES, batch+1*STATE_BYTES, batch+2*STATE_BYTES, batch+3*STATE_BYTES, batch+4*STATE_BYTES,
                batch+5*STATE_BYTES, batch+6*STATE_BYTES, batch+7*STATE_BYTES, batch+8*STATE_BYTES, batch+9*STATE_BYTES);
            for (int k = 0; k < 10; k++) mults[k] = out_table.pool[pi+k].multiplicity;
            clock::time_point h0; if (stats != nullptr) h0 = clock::now();
            il_insert_batch(scratch_table, batch, mults, fps, 10);
            if (stats != nullptr) hash_ns += ns_since(h0);
        }
        for (; pi < n; pi++) {
            const FlatTable::Entry& e = out_table.pool[pi];
            tm.load_state_raw(e.state.data()); run_map_group(tm, schedule, entry_idx, group_end);
            clock::time_point h0; if (stats != nullptr) h0 = clock::now();
            scratch_table.insert(tm.state_raw(), e.multiplicity);
            if (stats != nullptr) hash_ns += ns_since(h0);
        }
        if (stats != nullptr) { BoundaryRecord rec; rec.entry_begin = (std::uint32_t)entry_idx; rec.entry_end = (std::uint32_t)group_end;
            rec.frontier_in = frontier_in; rec.frontier_out = (std::uint32_t)scratch_table.pool.size();
            const std::uint64_t t = ns_since(map_t0); rec.hash_ns = hash_ns; rec.map_ns = t > hash_ns ? t - hash_ns : 0; stats->records.push_back(rec); }
        std::swap(out_table, scratch_table); entry_idx = group_end;
    }
}

// PROTOTYPE: 12-way (width-study ceiling probe; the x12 kernel likely spills).
template <typename TM>
void forward_block_with_dedup_il12(
    TM& tm, std::uint32_t key, std::uint32_t data_start, std::uint32_t window,
    const key_schedule& schedule, FlatTable& out_table, FlatTable& scratch_table,
    std::uint32_t dedup_every_maps = 4, std::uint32_t first_dedup_maps = 1, BoundaryStats* stats = nullptr)
{
    using clock = std::chrono::steady_clock;
    auto ns_since = [](clock::time_point t0) -> std::uint64_t {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count()); };
    if (dedup_every_maps == 0) dedup_every_maps = 1;
    if (first_dedup_maps == 0) first_dedup_maps = dedup_every_maps;
#ifndef DEDUP_NO_BIND_SCHEDULE
    if constexpr (requires(TM& t, const key_schedule& s) { t.bind_dedup_schedule(s); }) tm.bind_dedup_schedule(schedule);
    else if constexpr (requires(TM& t, const key_schedule& s) { t.bind_schedule(s); }) tm.bind_schedule(schedule);
#endif
    std::size_t entry_idx = 0;
    out_table.reset(window);
    {
        const std::size_t group_end = next_group_end(0, schedule.entries.size(), dedup_every_maps, first_dedup_maps, nullptr);
        prepare_map_group(tm, schedule, 0, group_end);
        const auto map_t0 = clock::now();
        const std::uint64_t hash_ns = batched_run_insert(tm, out_table, window,
            [&](std::uint32_t j) { tm.expand(key, data_start + j); run_map_group(tm, schedule, 0, group_end); },
            [](std::uint32_t) -> std::uint32_t { return 1u; }, stats != nullptr);
        if (stats != nullptr) { BoundaryRecord rec; rec.entry_begin = 0; rec.entry_end = (std::uint32_t)group_end;
            rec.frontier_in = window; rec.frontier_out = (std::uint32_t)out_table.pool.size();
            const std::uint64_t t = ns_since(map_t0); rec.hash_ns = hash_ns; rec.map_ns = t > hash_ns ? t - hash_ns : 0; stats->records.push_back(rec); }
        entry_idx = group_end;
    }
    alignas(64) std::uint8_t batch[12 * STATE_BYTES];
    std::uint32_t mults[12]; std::uint64_t fps[12];
    for (; entry_idx < schedule.entries.size(); )
    {
        const std::size_t group_end = next_group_end(entry_idx, schedule.entries.size(), dedup_every_maps, first_dedup_maps, nullptr);
        prepare_map_group(tm, schedule, entry_idx, group_end);
        const std::uint32_t frontier_in = (std::uint32_t)out_table.pool.size();
        scratch_table.reset(frontier_in);
        std::uint64_t hash_ns = 0; const auto map_t0 = clock::now();
        const std::size_t n = out_table.pool.size(); std::size_t pi = 0;
        for (; pi + 12 <= n; pi += 12)
        {
            tm.run_maps_range_x12(schedule, entry_idx, group_end,
                out_table.pool[pi+0].state.data(), out_table.pool[pi+1].state.data(), out_table.pool[pi+2].state.data(), out_table.pool[pi+3].state.data(), out_table.pool[pi+4].state.data(), out_table.pool[pi+5].state.data(),
                out_table.pool[pi+6].state.data(), out_table.pool[pi+7].state.data(), out_table.pool[pi+8].state.data(), out_table.pool[pi+9].state.data(), out_table.pool[pi+10].state.data(), out_table.pool[pi+11].state.data(),
                batch+0*STATE_BYTES, batch+1*STATE_BYTES, batch+2*STATE_BYTES, batch+3*STATE_BYTES, batch+4*STATE_BYTES, batch+5*STATE_BYTES,
                batch+6*STATE_BYTES, batch+7*STATE_BYTES, batch+8*STATE_BYTES, batch+9*STATE_BYTES, batch+10*STATE_BYTES, batch+11*STATE_BYTES);
            for (int k = 0; k < 12; k++) mults[k] = out_table.pool[pi+k].multiplicity;
            clock::time_point h0; if (stats != nullptr) h0 = clock::now();
            il_insert_batch(scratch_table, batch, mults, fps, 12);
            if (stats != nullptr) hash_ns += ns_since(h0);
        }
        for (; pi < n; pi++) {
            const FlatTable::Entry& e = out_table.pool[pi];
            tm.load_state_raw(e.state.data()); run_map_group(tm, schedule, entry_idx, group_end);
            clock::time_point h0; if (stats != nullptr) h0 = clock::now();
            scratch_table.insert(tm.state_raw(), e.multiplicity);
            if (stats != nullptr) hash_ns += ns_since(h0);
        }
        if (stats != nullptr) { BoundaryRecord rec; rec.entry_begin = (std::uint32_t)entry_idx; rec.entry_end = (std::uint32_t)group_end;
            rec.frontier_in = frontier_in; rec.frontier_out = (std::uint32_t)scratch_table.pool.size();
            const std::uint64_t t = ns_since(map_t0); rec.hash_ns = hash_ns; rec.map_ns = t > hash_ns ? t - hash_ns : 0; stats->records.push_back(rec); }
        std::swap(out_table, scratch_table); entry_idx = group_end;
    }
}

// PROTOTYPE: 14-way (register-wall confirmation; the x14 kernel spills).
template <typename TM>
void forward_block_with_dedup_il14(
    TM& tm, std::uint32_t key, std::uint32_t data_start, std::uint32_t window,
    const key_schedule& schedule, FlatTable& out_table, FlatTable& scratch_table,
    std::uint32_t dedup_every_maps = 4, std::uint32_t first_dedup_maps = 1, BoundaryStats* stats = nullptr)
{
    using clock = std::chrono::steady_clock;
    auto ns_since = [](clock::time_point t0) -> std::uint64_t {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count()); };
    if (dedup_every_maps == 0) dedup_every_maps = 1;
    if (first_dedup_maps == 0) first_dedup_maps = dedup_every_maps;
#ifndef DEDUP_NO_BIND_SCHEDULE
    if constexpr (requires(TM& t, const key_schedule& s) { t.bind_dedup_schedule(s); }) tm.bind_dedup_schedule(schedule);
    else if constexpr (requires(TM& t, const key_schedule& s) { t.bind_schedule(s); }) tm.bind_schedule(schedule);
#endif
    std::size_t entry_idx = 0;
    out_table.reset(window);
    {
        const std::size_t group_end = next_group_end(0, schedule.entries.size(), dedup_every_maps, first_dedup_maps, nullptr);
        prepare_map_group(tm, schedule, 0, group_end);
        const auto map_t0 = clock::now();
        const std::uint64_t hash_ns = batched_run_insert(tm, out_table, window,
            [&](std::uint32_t j) { tm.expand(key, data_start + j); run_map_group(tm, schedule, 0, group_end); },
            [](std::uint32_t) -> std::uint32_t { return 1u; }, stats != nullptr);
        if (stats != nullptr) { BoundaryRecord rec; rec.entry_begin = 0; rec.entry_end = (std::uint32_t)group_end;
            rec.frontier_in = window; rec.frontier_out = (std::uint32_t)out_table.pool.size();
            const std::uint64_t t = ns_since(map_t0); rec.hash_ns = hash_ns; rec.map_ns = t > hash_ns ? t - hash_ns : 0; stats->records.push_back(rec); }
        entry_idx = group_end;
    }
    alignas(64) std::uint8_t batch[14 * STATE_BYTES];
    std::uint32_t mults[14]; std::uint64_t fps[14];
    for (; entry_idx < schedule.entries.size(); )
    {
        const std::size_t group_end = next_group_end(entry_idx, schedule.entries.size(), dedup_every_maps, first_dedup_maps, nullptr);
        prepare_map_group(tm, schedule, entry_idx, group_end);
        const std::uint32_t frontier_in = (std::uint32_t)out_table.pool.size();
        scratch_table.reset(frontier_in);
        std::uint64_t hash_ns = 0; const auto map_t0 = clock::now();
        const std::size_t n = out_table.pool.size(); std::size_t pi = 0;
        for (; pi + 14 <= n; pi += 14)
        {
            tm.run_maps_range_x14(schedule, entry_idx, group_end,
                out_table.pool[pi+0].state.data(), out_table.pool[pi+1].state.data(), out_table.pool[pi+2].state.data(), out_table.pool[pi+3].state.data(), out_table.pool[pi+4].state.data(), out_table.pool[pi+5].state.data(), out_table.pool[pi+6].state.data(),
                out_table.pool[pi+7].state.data(), out_table.pool[pi+8].state.data(), out_table.pool[pi+9].state.data(), out_table.pool[pi+10].state.data(), out_table.pool[pi+11].state.data(), out_table.pool[pi+12].state.data(), out_table.pool[pi+13].state.data(),
                batch+0*STATE_BYTES, batch+1*STATE_BYTES, batch+2*STATE_BYTES, batch+3*STATE_BYTES, batch+4*STATE_BYTES, batch+5*STATE_BYTES, batch+6*STATE_BYTES,
                batch+7*STATE_BYTES, batch+8*STATE_BYTES, batch+9*STATE_BYTES, batch+10*STATE_BYTES, batch+11*STATE_BYTES, batch+12*STATE_BYTES, batch+13*STATE_BYTES);
            for (int k = 0; k < 14; k++) mults[k] = out_table.pool[pi+k].multiplicity;
            clock::time_point h0; if (stats != nullptr) h0 = clock::now();
            il_insert_batch(scratch_table, batch, mults, fps, 14);
            if (stats != nullptr) hash_ns += ns_since(h0);
        }
        for (; pi < n; pi++) {
            const FlatTable::Entry& e = out_table.pool[pi];
            tm.load_state_raw(e.state.data()); run_map_group(tm, schedule, entry_idx, group_end);
            clock::time_point h0; if (stats != nullptr) h0 = clock::now();
            scratch_table.insert(tm.state_raw(), e.multiplicity);
            if (stats != nullptr) hash_ns += ns_since(h0);
        }
        if (stats != nullptr) { BoundaryRecord rec; rec.entry_begin = (std::uint32_t)entry_idx; rec.entry_end = (std::uint32_t)group_end;
            rec.frontier_in = frontier_in; rec.frontier_out = (std::uint32_t)scratch_table.pool.size();
            const std::uint64_t t = ns_since(map_t0); rec.hash_ns = hash_ns; rec.map_ns = t > hash_ns ? t - hash_ns : 0; stats->records.push_back(rec); }
        std::swap(out_table, scratch_table); entry_idx = group_end;
    }
}

} // namespace state_dedup

#endif // STATE_DEDUP_IL_H
