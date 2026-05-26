// State-dedup matrix bench across all CPU forward variants.
//
// For each (key, window, impl) cell, measures wall-clock for two equivalent
// computations and reports candidates/second throughput:
//
//   baseline: N independent forward runs, no dedup
//   flat:     state_dedup::forward_block_with_dedup, dedup-at-every-boundary
//
// Both produce the same multiset of final states (parity-verified).
//
// candidates/sec = window / wall_time. This is the production-relevant
// number: how many data values per second the lane processes.
//
// Impls available (selectable via --impl, comma-separated):
//   tm_8                  scalar reference
//   tm_avx_r128s_8        128-bit AVX, narrow (X3D-friendly)
//   tm_avx_r256s_8        256-bit AVX without AVX2
//   tm_avx2_r256s_8       256-bit AVX2 universal-table kernel
//   tm_avx2_r256_map_8    256-bit AVX2 map-mode kernel
//   tm_avx512_r512s_8     512-bit AVX-512 (Sapphire/Granite Rapids target)

#include "rng_obj.h"
#include "key_schedule.h"
#include "key_file.h"
#include "state_dedup.h"

#include "../cpu/tm_8.h"
#include "../cpu/tm_avx_r128s_8.h"
#include "../cpu/tm_avx_r256s_8.h"
#include "../cpu/tm_avx2_r256s_8.h"
#include "../cpu/tm_avx2_r256_map_8.h"
#include "../cpu/tm_avx512_r512s_8.h"

#include <algorithm>
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

std::uint64_t parity_hash(const std::uint8_t* bytes)
{
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < 128; i++) { h ^= bytes[i]; h *= 1099511628211ull; }
    return h;
}

template <typename TM>
double bench_baseline(TM& tm, std::uint32_t key, std::uint32_t data_start,
                      std::uint32_t window, const key_schedule& schedule,
                      std::uint64_t& parity_out)
{
    // Production-realistic baseline: run_all_maps fuses all 27 maps in SIMD
    // registers with a single load_from_mem at the start and a single
    // store_to_mem at the end. Looping over run_one_map externally would
    // pay an extra load+store per map (26x more memory traffic on SIMD),
    // which would artificially inflate the apparent dedup speedup.
    typedef std::chrono::high_resolution_clock clock;
    const auto t0 = clock::now();
    std::uint64_t parity = 0;
    for (std::uint32_t i = 0; i < window; i++)
    {
        tm.expand(key, data_start + i);
        tm.run_all_maps(schedule);
        parity ^= parity_hash(tm.state_raw());
    }
    parity_out = parity;
    return std::chrono::duration<double>(clock::now() - t0).count();
}

template <typename TM>
double bench_flat(TM& tm, std::uint32_t key, std::uint32_t data_start,
                  std::uint32_t window, const key_schedule& schedule,
                  state_dedup::FlatTable& out, state_dedup::FlatTable& scratch,
                  std::uint64_t& parity_out, std::size_t& unique_out)
{
    typedef std::chrono::high_resolution_clock clock;
    const auto t0 = clock::now();

    state_dedup::forward_block_with_dedup(tm, key, data_start, window, schedule, out, scratch);

    std::uint64_t parity = 0;
    for (std::size_t pi = 0; pi < out.pool.size(); pi++)
    {
        const auto& entry = out.pool[pi];
        if ((entry.multiplicity & 1u) != 0u)
        {
            parity ^= parity_hash(entry.state.data());
        }
    }
    parity_out = parity;
    unique_out = out.pool.size();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Args
{
    std::string keys_path;
    std::string out_path;
    std::vector<std::uint32_t> windows;
    std::vector<std::string> impls;
    std::uint32_t data_start = 0;
    std::size_t limit = 0;
    int repeats = 3;
    key_schedule::map_list_type map_kind = key_schedule::ALL_MAPS;
};

Args parse_args(int argc, char** argv)
{
    Args a;
    auto nxt = [&](int& i, const char* flag) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string(flag) + " needs value");
        return std::string(argv[++i]);
    };
    for (int i = 1; i < argc; i++)
    {
        std::string s = argv[i];
        if      (s == "--keys")       a.keys_path = nxt(i, "--keys");
        else if (s == "--out")        a.out_path = nxt(i, "--out");
        else if (s == "--data-start") a.data_start = static_cast<std::uint32_t>(std::stoul(nxt(i, "--data-start"), nullptr, 0));
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
            std::string tok;
            while (std::getline(ss, tok, ',')) if (!tok.empty()) a.windows.push_back(static_cast<std::uint32_t>(std::stoul(tok)));
        }
        else if (s == "--impl")
        {
            std::stringstream ss(nxt(i, "--impl"));
            std::string tok;
            while (std::getline(ss, tok, ',')) if (!tok.empty()) a.impls.push_back(tok);
        }
        else throw std::runtime_error("unknown arg: " + s);
    }
    if (a.keys_path.empty()) throw std::runtime_error("--keys required");
    if (a.out_path.empty())  throw std::runtime_error("--out required");
    if (a.windows.empty())   throw std::runtime_error("--windows required");
    if (a.impls.empty())     a.impls = {"tm_8", "tm_avx_r128s_8", "tm_avx_r256s_8", "tm_avx2_r256s_8", "tm_avx2_r256_map_8", "tm_avx512_r512s_8"};
    return a;
}

// One-impl bench helper: for each key, each window, run baseline + flat
// for `repeats`, write median timings + candidates/sec to the CSV.
template <typename TM>
void bench_impl(const std::string& impl_name, TM& tm,
                const std::vector<std::uint32_t>& keys,
                const std::vector<std::uint32_t>& windows,
                std::uint32_t data_start, int repeats,
                key_schedule::map_list_type map_kind,
                std::ofstream& out,
                std::size_t& parity_mismatches_out)
{
    state_dedup::FlatTable dedup_out, dedup_scratch;

    typedef std::chrono::high_resolution_clock clock;
    const auto t_start = clock::now();
    const std::size_t report_every = std::max<std::size_t>(1, keys.size() / 10);

    for (std::size_t k_idx = 0; k_idx < keys.size(); k_idx++)
    {
        const std::uint32_t key = keys[k_idx];
        key_schedule schedule(key, map_kind);

        for (std::size_t w_idx = 0; w_idx < windows.size(); w_idx++)
        {
            const std::uint32_t window = windows[w_idx];
            if (window == 0) continue;

            std::vector<double> ts_base, ts_flat;
            std::uint64_t parity_base = 0, parity_flat = 0;
            std::size_t unique = 0;

            // Warm-up to prime caches.
            {
                std::uint64_t p; std::size_t u;
                (void)bench_baseline(tm, key, data_start, window, schedule, p);
                (void)bench_flat    (tm, key, data_start, window, schedule, dedup_out, dedup_scratch, p, u);
            }

            for (int r = 0; r < repeats; r++)
            {
                ts_base.push_back(bench_baseline(tm, key, data_start, window, schedule, parity_base));
                ts_flat.push_back(bench_flat    (tm, key, data_start, window, schedule, dedup_out, dedup_scratch, parity_flat, unique));
            }

            const double m_base = median(ts_base);
            const double m_flat = median(ts_flat);
            const double cps_base = m_base > 0 ? static_cast<double>(window) / m_base : 0;
            const double cps_flat = m_flat > 0 ? static_cast<double>(window) / m_flat : 0;
            const double dedup_speedup = m_flat > 0 ? m_base / m_flat : 0;
            const bool parity_ok = (parity_base == parity_flat);
            if (!parity_ok) parity_mismatches_out++;

            out << "0x" << std::hex << std::setw(8) << std::setfill('0') << key
                << std::dec << std::setfill(' ')
                << "," << window
                << "," << impl_name
                << "," << std::fixed << std::setprecision(6) << m_base
                << "," << std::fixed << std::setprecision(6) << m_flat
                << "," << std::fixed << std::setprecision(0) << cps_base
                << "," << std::fixed << std::setprecision(0) << cps_flat
                << "," << std::fixed << std::setprecision(3) << dedup_speedup
                << "," << unique
                << "," << (parity_ok ? 1 : 0)
                << "\n";
            out.flush();
        }

        if ((k_idx + 1) % report_every == 0)
        {
            const double el = std::chrono::duration<double>(clock::now() - t_start).count();
            std::cerr << "  [" << impl_name << "] [" << (k_idx + 1) << "/" << keys.size()
                      << "] keys in " << std::fixed << std::setprecision(1) << el << "s\n";
        }
    }
}

} // anonymous namespace

int main(int argc, char** argv)
{
    try
    {
        Args a = parse_args(argc, argv);
        std::vector<std::uint32_t> keys = key_file::read_keys(a.keys_path, a.limit);
        if (keys.empty()) throw std::runtime_error("no keys parsed from: " + a.keys_path);

        std::ofstream out(a.out_path);
        if (!out) throw std::runtime_error("cannot open output: " + a.out_path);

        std::vector<std::uint32_t> windows_sorted(a.windows);
        std::sort(windows_sorted.begin(), windows_sorted.end());

        out << "key_hex,window,impl,baseline_s,flat_s,baseline_cps,flat_cps,"
               "dedup_speedup,final_unique_states,parity_match\n";

        std::cerr << "State-dedup matrix bench\n"
                  << "  keys:       " << keys.size() << "\n"
                  << "  data_start: 0x" << std::hex << std::setw(8) << std::setfill('0')
                  << a.data_start << std::dec << std::setfill(' ') << "\n"
                  << "  windows:    ";
        for (std::size_t i = 0; i < windows_sorted.size(); i++)
            std::cerr << (i == 0 ? "" : ",") << windows_sorted[i];
        std::cerr << "\n  impls:      ";
        for (std::size_t i = 0; i < a.impls.size(); i++)
            std::cerr << (i == 0 ? "" : ",") << a.impls[i];
        std::cerr << "\n  repeats:    " << a.repeats << "\n";

        RNG rng;
        std::size_t parity_mismatches = 0;

        for (const auto& impl_name : a.impls)
        {
            if (impl_name == "tm_8")
            {
                tm_8 tm(&rng);
                bench_impl(impl_name, tm, keys, windows_sorted, a.data_start, a.repeats, a.map_kind, out, parity_mismatches);
            }
            else if (impl_name == "tm_avx_r128s_8")
            {
                tm_avx_r128s_8 tm(&rng);
                bench_impl(impl_name, tm, keys, windows_sorted, a.data_start, a.repeats, a.map_kind, out, parity_mismatches);
            }
            else if (impl_name == "tm_avx_r256s_8")
            {
                tm_avx_r256s_8 tm(&rng);
                bench_impl(impl_name, tm, keys, windows_sorted, a.data_start, a.repeats, a.map_kind, out, parity_mismatches);
            }
            else if (impl_name == "tm_avx2_r256s_8")
            {
                tm_avx2_r256s_8 tm(&rng);
                bench_impl(impl_name, tm, keys, windows_sorted, a.data_start, a.repeats, a.map_kind, out, parity_mismatches);
            }
            else if (impl_name == "tm_avx2_r256_map_8")
            {
                tm_avx2_r256_map_8 tm(&rng);
                bench_impl(impl_name, tm, keys, windows_sorted, a.data_start, a.repeats, a.map_kind, out, parity_mismatches);
            }
            else if (impl_name == "tm_avx512_r512s_8")
            {
                tm_avx512_r512s_8 tm(&rng);
                bench_impl(impl_name, tm, keys, windows_sorted, a.data_start, a.repeats, a.map_kind, out, parity_mismatches);
            }
            else
            {
                std::cerr << "unknown impl: " << impl_name << " (skipped)\n";
            }
        }

        if (parity_mismatches > 0)
        {
            std::cerr << "WARNING: " << parity_mismatches << " parity mismatches\n";
            return 2;
        }
        std::cerr << "parity: all cells matched within their impl\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "error: " << e.what() << "\n";
        std::cerr << "usage: state_dedup_matrix_bench --keys FILE --out CSV --windows W1,W2,... "
                     "[--impl tm_8,tm_avx2_r256s_8,...] [--data-start HEX] [--repeats R] "
                     "[--schedule all|skip-car] [--limit N]\n";
        return 1;
    }
}
