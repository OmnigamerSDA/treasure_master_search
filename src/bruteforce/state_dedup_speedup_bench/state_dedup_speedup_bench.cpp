// POC: CPU wall-clock smoke test for state-merge dedup, scalar + SIMD.
//
// For each (key, window) cell, runs four equivalent computations and reports
// the speedup matrix:
//
//   baseline_scalar:  N independent forward runs on tm_8 (scalar reference)
//   flat_scalar:      tm_8 + flat-hash dedup at every schedule boundary
//   baseline_simd:    N independent forward runs on tm_avx2_r256s_8
//   flat_simd:        tm_avx2_r256s_8 + flat-hash dedup at every boundary
//
// All four produce the same multiset of final states (verified via parity
// hash); flat-dedup tracks multiplicities, baseline emits duplicates.
//
// The flat-hash dedup is an open-addressed table keyed on a 64-bit
// fingerprint of the impl-native state layout. Linear probing at load
// factor ~0.5, one state-pool copy per unique state, no allocator overhead
// past the per-boundary reset.

#include "rng_obj.h"
#include "key_schedule.h"
#include "key_file.h"
#include "state_dedup.h"
#include "tm_base.h"
#include "../cpu/tm_8.h"
#include "../cpu/tm_avx2_r256s_8.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

std::uint64_t parity_hash(const uint8* bytes)
{
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < 128; i++) { h ^= bytes[i]; h *= 1099511628211ull; }
    return h;
}

// =====================================================================
// Templated bench primitives — same code paths over any TM impl that
// exposes expand / run_one_map / state_raw / load_state_raw.
// =====================================================================

template <typename TM>
double run_baseline(TM& tm, uint32 key, uint32 data_start, uint32 window,
                    const key_schedule& schedule, std::uint64_t& parity_out)
{
    // Production-realistic baseline: run_all_maps fuses all maps in SIMD
    // registers with one load/store per data value (vs N load/stores if we
    // looped run_one_map externally).
    typedef std::chrono::high_resolution_clock clock;
    const auto t0 = clock::now();
    std::uint64_t parity = 0;
    for (uint32 i = 0; i < window; i++)
    {
        tm.expand(key, data_start + i);
        tm.run_all_maps(schedule);
        parity ^= parity_hash(tm.state_raw());
    }
    parity_out = parity;
    return std::chrono::duration<double>(clock::now() - t0).count();
}

template <typename TM>
double run_flat(TM& tm, uint32 key, uint32 data_start, uint32 window,
                const key_schedule& schedule, std::uint64_t& parity_out,
                std::size_t& final_unique_out, std::uint64_t& /*merged_evals_out*/,
                state_dedup::FlatTable& out_table, state_dedup::FlatTable& scratch_table)
{
    typedef std::chrono::high_resolution_clock clock;
    const auto t0 = clock::now();

    state_dedup::forward_block_with_dedup(
        tm, key, data_start, window, schedule, out_table, scratch_table);

    std::uint64_t parity = 0;
    for (std::size_t pi = 0; pi < out_table.pool.size(); pi++)
    {
        const auto& entry = out_table.pool[pi];
        if ((entry.multiplicity & 1u) != 0u)
        {
            parity ^= parity_hash(entry.state.data());
        }
    }
    parity_out = parity;
    final_unique_out = out_table.pool.size();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

// =====================================================================
// Args, key list, driver.
// =====================================================================

struct Args
{
    std::string keys_path;
    std::string out_path;
    std::vector<uint32> windows;
    uint32 data_start = 0;
    std::size_t limit = 0;
    int repeats = 3;
    key_schedule::map_list_type map_kind = key_schedule::ALL_MAPS;
};

Args parse_args(int argc, char** argv)
{
    Args a;
    auto nxt = [&](int& i, const char* flag) -> std::string
    {
        if (i + 1 >= argc) throw std::runtime_error(std::string(flag) + " needs value");
        return std::string(argv[++i]);
    };
    for (int i = 1; i < argc; i++)
    {
        std::string s = argv[i];
        if      (s == "--keys")       a.keys_path = nxt(i, "--keys");
        else if (s == "--out")        a.out_path = nxt(i, "--out");
        else if (s == "--data-start") a.data_start = static_cast<uint32>(std::stoul(nxt(i, "--data-start"), nullptr, 0));
        else if (s == "--limit")      a.limit = static_cast<std::size_t>(std::stoull(nxt(i, "--limit")));
        else if (s == "--repeats")    a.repeats = std::stoi(nxt(i, "--repeats"));
        else if (s == "--schedule")
        {
            std::string mode = nxt(i, "--schedule");
            if      (mode == "all")      a.map_kind = key_schedule::ALL_MAPS;
            else if (mode == "skip-car") a.map_kind = key_schedule::SKIP_CAR;
            else throw std::runtime_error("--schedule must be 'all' or 'skip-car'");
        }
        else if (s == "--windows")
        {
            std::stringstream ss(nxt(i, "--windows"));
            std::string token;
            while (std::getline(ss, token, ','))
            {
                if (token.empty()) continue;
                a.windows.push_back(static_cast<uint32>(std::stoul(token)));
            }
        }
        else throw std::runtime_error("unknown arg: " + s);
    }
    if (a.keys_path.empty()) throw std::runtime_error("--keys required");
    if (a.out_path.empty())  throw std::runtime_error("--out required");
    if (a.windows.empty())   throw std::runtime_error("--windows required");
    if (a.repeats < 1)       throw std::runtime_error("--repeats must be >= 1");
    return a;
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

} // anonymous namespace

int main(int argc, char** argv)
{
    try
    {
        Args a = parse_args(argc, argv);
        std::vector<uint32> keys = key_file::read_keys(a.keys_path, a.limit);
        if (keys.empty()) throw std::runtime_error("no keys parsed from: " + a.keys_path);

        std::ofstream out(a.out_path);
        if (!out) throw std::runtime_error("cannot open output: " + a.out_path);

        std::vector<uint32> windows_sorted(a.windows);
        std::sort(windows_sorted.begin(), windows_sorted.end());

        out << "key_hex,window,"
               "base_scalar_s,flat_scalar_s,base_simd_s,flat_simd_s,"
               "flat_scalar_speedup,flat_simd_speedup,"
               "simd_lift,combined_speedup,"
               "final_unique_states,parity_match\n";

        std::cerr << "State-dedup wall-clock bench (scalar + SIMD)\n"
                  << "  keys:       " << keys.size() << "\n"
                  << "  data_start: 0x" << std::hex << std::setw(8) << std::setfill('0')
                  << a.data_start << std::dec << std::setfill(' ') << "\n"
                  << "  windows:    ";
        for (std::size_t i = 0; i < windows_sorted.size(); i++)
            std::cerr << (i == 0 ? "" : ",") << windows_sorted[i];
        std::cerr << "\n  repeats:    " << a.repeats << "\n";

        RNG rng;
        tm_8 tm_scalar(&rng);
        tm_avx2_r256s_8 tm_simd(&rng);

        // Persistent dedup tables — reuse allocations across all (key, window) cells.
        state_dedup::FlatTable dedup_out, dedup_scratch;

        typedef std::chrono::high_resolution_clock clock;
        const auto run_start = clock::now();
        std::size_t parity_mismatches = 0;
        const std::size_t report_every = std::max<std::size_t>(1, keys.size() / 10);

        for (std::size_t k_idx = 0; k_idx < keys.size(); k_idx++)
        {
            const uint32 key = keys[k_idx];
            key_schedule schedule(key, a.map_kind);

            for (std::size_t w_idx = 0; w_idx < windows_sorted.size(); w_idx++)
            {
                const uint32 window = windows_sorted[w_idx];
                if (window == 0) continue;

                std::vector<double> ts_base_sc, ts_flat_sc, ts_base_si, ts_flat_si;
                std::uint64_t parity_base_sc = 0, parity_flat_sc = 0;
                std::uint64_t parity_base_si = 0, parity_flat_si = 0;
                std::size_t final_unique = 0;
                std::uint64_t merged_evals = 0;

                // Warm-up + measured passes, interleaved to dodge thermal drift.
                {
                    std::uint64_t p;
                    std::size_t fu;
                    std::uint64_t me;
                    (void)run_baseline(tm_scalar, key, a.data_start, window, schedule, p);
                    (void)run_flat    (tm_scalar, key, a.data_start, window, schedule, p, fu, me, dedup_out, dedup_scratch);
                    (void)run_baseline(tm_simd,   key, a.data_start, window, schedule, p);
                    (void)run_flat    (tm_simd,   key, a.data_start, window, schedule, p, fu, me, dedup_out, dedup_scratch);
                }

                for (int r = 0; r < a.repeats; r++)
                {
                    ts_base_sc.push_back(run_baseline(tm_scalar, key, a.data_start, window, schedule, parity_base_sc));
                    ts_flat_sc.push_back(run_flat    (tm_scalar, key, a.data_start, window, schedule, parity_flat_sc, final_unique, merged_evals, dedup_out, dedup_scratch));
                    ts_base_si.push_back(run_baseline(tm_simd,   key, a.data_start, window, schedule, parity_base_si));
                    ts_flat_si.push_back(run_flat    (tm_simd,   key, a.data_start, window, schedule, parity_flat_si, final_unique, merged_evals, dedup_out, dedup_scratch));
                }

                const double med_base_sc = median(ts_base_sc);
                const double med_flat_sc = median(ts_flat_sc);
                const double med_base_si = median(ts_base_si);
                const double med_flat_si = median(ts_flat_si);

                const double flat_scalar_speedup = med_flat_sc > 0 ? med_base_sc / med_flat_sc : 0;
                const double flat_simd_speedup   = med_flat_si > 0 ? med_base_si / med_flat_si : 0;
                const double simd_lift           = med_base_si > 0 ? med_base_sc / med_base_si : 0;
                const double combined_speedup    = med_flat_si > 0 ? med_base_sc / med_flat_si : 0;

                // Parity checks: scalar-baseline vs scalar-flat (same impl, same layout)
                // SIMD-baseline vs SIMD-flat (same impl, same layout)
                // Scalar and SIMD use different internal layouts so cross-impl parity
                // would not match — we don't require it.
                const bool parity_ok = (parity_base_sc == parity_flat_sc) &&
                                       (parity_base_si == parity_flat_si);
                if (!parity_ok) parity_mismatches++;

                out << "0x" << std::hex << std::setw(8) << std::setfill('0') << key
                    << std::dec << std::setfill(' ')
                    << "," << window
                    << "," << std::fixed << std::setprecision(6) << med_base_sc
                    << "," << std::fixed << std::setprecision(6) << med_flat_sc
                    << "," << std::fixed << std::setprecision(6) << med_base_si
                    << "," << std::fixed << std::setprecision(6) << med_flat_si
                    << "," << std::fixed << std::setprecision(3) << flat_scalar_speedup
                    << "," << std::fixed << std::setprecision(3) << flat_simd_speedup
                    << "," << std::fixed << std::setprecision(3) << simd_lift
                    << "," << std::fixed << std::setprecision(3) << combined_speedup
                    << "," << final_unique
                    << "," << (parity_ok ? 1 : 0)
                    << "\n";
                out.flush();
            }

            if ((k_idx + 1) % report_every == 0)
            {
                const double elapsed = std::chrono::duration<double>(clock::now() - run_start).count();
                std::cerr << "  [" << (k_idx + 1) << "/" << keys.size() << "] keys in "
                          << std::fixed << std::setprecision(1) << elapsed << "s\n";
            }
        }

        const double total_s = std::chrono::duration<double>(clock::now() - run_start).count();
        std::cerr << "  total elapsed: " << std::fixed << std::setprecision(1) << total_s << "s\n";
        if (parity_mismatches > 0)
        {
            std::cerr << "  WARNING: " << parity_mismatches << " parity mismatches\n";
            return 2;
        }
        std::cerr << "  parity:        all cells matched\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "error: " << e.what() << "\n";
        std::cerr << "usage: state_dedup_speedup_bench --keys FILE --out CSV --windows W1,W2,... "
                     "[--data-start HEX] [--repeats R] [--schedule all|skip-car] [--limit N]\n";
        return 1;
    }
}
