#ifndef MAP_TABLES_SHARED_H
#define MAP_TABLES_SHARED_H

// Shared, immutable per-schedule RNG tables for the map-mode forward kernel.
// All threads using the same schedule point at the same 162 KB of read-only
// tables, avoiding the N×duplication that hurts at high thread count.
//
// Layout (entry_count × 2048 bytes per stream):
//   reg_table  — regular_rng outputs, reversed (consumed by alg 1/3/4 in 128-
//                byte chunks; read byte-by-byte for the carry bit in alg 2/5)
//   alg0_table — (run_rng() >> 7) & 0x01, reversed
//   alg6_table — run_rng() & 0x80, forward
//
// The map kernel additionally caches the per-entry nibble_selector and the
// pointer to the source schedule.entries[] array (for fast pointer-arithmetic
// map_idx computation in run_one_map under state_dedup).
//
// Usage:
//   - Call SharedMapTables::bind(rng, schedule) once before any kernel work.
//     The first thread to call it builds the tables; later threads see
//     "already built" and return immediately.
//   - The kernel reads from SharedMapTables::reg_table() etc.
//   - The schedule must be stable for the lifetime of the bench / thread
//     pool; subsequent bind() with a different schedule replaces the tables
//     (caller's responsibility to ensure no concurrent work).

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <vector>
#if defined(_WIN32)
#include <malloc.h>
#endif

#include "data_sizes.h"
#include "rng.h"
#include "rng_obj.h"
#include "key_schedule.h"

namespace map_tables_shared
{

inline void free_aligned(void* p)
{
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

inline uint8* allocate_aligned_32(std::size_t bytes)
{
#if defined(_WIN32)
    return static_cast<uint8*>(_aligned_malloc(bytes, 32));
#else
    void* p = nullptr;
    if (posix_memalign(&p, 32, bytes) != 0) return nullptr;
    return static_cast<uint8*>(p);
#endif
}

struct Tables
{
    // Allocated once; pointers do not change across bind() calls on the same
    // (entry_count, seeds) signature.
    uint8* reg_table  = nullptr;
    uint8* alg0_table = nullptr;
    uint8* alg6_table = nullptr;

    std::vector<uint16> seeds;
    std::vector<uint16> nibble_selectors;
    const key_schedule::key_schedule_entry* entries_data = nullptr;
    int entry_count = 0;
    size_t capacity_entries = 0;

    static constexpr int ALG06_BYTES_PER_ENTRY = 2048;

    // Build once. Repeated calls with the same (count, seeds[]) signature are
    // O(count). With a different signature: free and rebuild. Caller must
    // ensure no concurrent kernel work crosses a rebuild boundary.
    void bind(RNG* rng, const key_schedule& schedule_entries)
    {
        const int n = static_cast<int>(schedule_entries.entries.size());

        // Signature check — if seeds/count match, just refresh nibble_selectors
        // and entries_data (which may have moved if caller passed a different
        // key_schedule object with identical seeds).
        bool changed = (n != entry_count) || (static_cast<int>(seeds.size()) != n);
        if (!changed) {
            for (int i = 0; i < n; i++) {
                const auto& e = schedule_entries.entries[i];
                uint16 s = static_cast<uint16>((e.rng1 << 8) | e.rng2);
                if (seeds[i] != s) { changed = true; break; }
            }
        }

        if (changed) {
            if (static_cast<size_t>(n) > capacity_entries) {
                free_aligned(reg_table);  free_aligned(alg0_table);  free_aligned(alg6_table);
                reg_table = allocate_aligned_32(static_cast<std::size_t>(n) * 2048);
                alg0_table = allocate_aligned_32(static_cast<std::size_t>(n) * ALG06_BYTES_PER_ENTRY);
                alg6_table = allocate_aligned_32(static_cast<std::size_t>(n) * ALG06_BYTES_PER_ENTRY);
                capacity_entries = static_cast<size_t>(n);
            }
            seeds.resize(n);
            for (int i = 0; i < n; i++) {
                const auto& e = schedule_entries.entries[i];
                uint16 s = static_cast<uint16>((e.rng1 << 8) | e.rng2);
                seeds[i] = s;
                generate_regular_rng_values_for_seed_8(reg_table  + i * 2048, s, rng->rng_table);
                generate_alg0_values_for_seed_8(alg0_table + i * ALG06_BYTES_PER_ENTRY, s, rng->rng_table);
                generate_alg6_values_for_seed_8(alg6_table + i * ALG06_BYTES_PER_ENTRY, s, rng->rng_table);
            }
            entry_count = n;
        }

        // Always refresh the entry-pointer-cache + nibble_selectors (cheap).
        nibble_selectors.resize(n);
        for (int i = 0; i < n; i++)
            nibble_selectors[i] = schedule_entries.entries[i].nibble_selector;
        entries_data = schedule_entries.entries.data();
    }

    void bind_range(RNG* rng, const key_schedule& schedule_entries, std::size_t begin, std::size_t end)
    {
        if (begin > schedule_entries.entries.size()) begin = schedule_entries.entries.size();
        if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
        if (end < begin) end = begin;
        const int n = static_cast<int>(end - begin);

        bool changed = (n != entry_count) || (static_cast<int>(seeds.size()) != n);
        if (!changed) {
            for (int i = 0; i < n; i++) {
                const auto& e = schedule_entries.entries[begin + static_cast<std::size_t>(i)];
                uint16 s = static_cast<uint16>((e.rng1 << 8) | e.rng2);
                if (seeds[i] != s) { changed = true; break; }
            }
        }

        if (changed) {
            if (static_cast<size_t>(n) > capacity_entries) {
                free_aligned(reg_table);  free_aligned(alg0_table);  free_aligned(alg6_table);
                reg_table = allocate_aligned_32(static_cast<std::size_t>(n) * 2048);
                alg0_table = allocate_aligned_32(static_cast<std::size_t>(n) * ALG06_BYTES_PER_ENTRY);
                alg6_table = allocate_aligned_32(static_cast<std::size_t>(n) * ALG06_BYTES_PER_ENTRY);
                capacity_entries = static_cast<size_t>(n);
            }
            seeds.resize(n);
            for (int i = 0; i < n; i++) {
                const auto& e = schedule_entries.entries[begin + static_cast<std::size_t>(i)];
                uint16 s = static_cast<uint16>((e.rng1 << 8) | e.rng2);
                seeds[i] = s;
                generate_regular_rng_values_for_seed_8(reg_table  + i * 2048, s, rng->rng_table);
                generate_alg0_values_for_seed_8(alg0_table + i * ALG06_BYTES_PER_ENTRY, s, rng->rng_table);
                generate_alg6_values_for_seed_8(alg6_table + i * ALG06_BYTES_PER_ENTRY, s, rng->rng_table);
            }
            entry_count = n;
        }

        nibble_selectors.resize(n);
        for (int i = 0; i < n; i++)
            nibble_selectors[i] = schedule_entries.entries[begin + static_cast<std::size_t>(i)].nibble_selector;
        entries_data = schedule_entries.entries.data() + begin;
    }
};

} // namespace map_tables_shared

#endif // MAP_TABLES_SHARED_H
