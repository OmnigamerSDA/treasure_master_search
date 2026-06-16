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
#include "state_dedup_il.h"

#include "../cpu/tm_8.h"
#include "../cpu/tm_avx_r128s_8.h"
#include "../cpu/tm_avx_r256s_8.h"
#include "../cpu/tm_avx2_r256s_8.h"
#include "../cpu/tm_avx2_r256_map_8.h"
#include "../cpu/tm_avx512_r512s_8.h"
#include "../cpu/tm_avx512_r512_map_8.h"
#include "../cpu/tm_avx512_r512s_map_8.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <thread>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

// Tier-1 front-cache size in bits (2^N entries); 0 = disabled. Set from
// --front-cache-bits and read at FlatTable creation sites (enable_front_cache).
unsigned g_front_cache_bits = 0;

// Dynamic table sizing (slots+pool track the frontier, not the window). Set from
// --dynamic-table, applied at FlatTable creation sites (set_dynamic).
bool g_dynamic_table = false;

// Skip the cache-priming warm-up pass (--no-warmup). final_unique is
// timing-invariant, so for characterization surveys (esp. very large windows
// where the warm-up doubles wall time and can trip a memory guard) the warm-up
// is pure waste. Default off → production/timing runs keep the warm-up.
bool g_no_warmup = false;

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
    state_dedup::for_each_window_index(window, [&](std::uint32_t i)
    {
        tm.expand(key, data_start + i);
        tm.run_all_maps(schedule);
        parity ^= parity_hash(tm.state_raw());
    });
    parity_out = parity;
    return std::chrono::duration<double>(clock::now() - t0).count();
}

struct BoundaryAccum
{
    std::uint64_t calls = 0;
    std::uint64_t sum_frontier_in = 0;
    std::uint64_t sum_frontier_out = 0;
    std::uint64_t sum_map_ns = 0;
    std::uint64_t sum_hash_ns = 0;
};

using BoundaryKey = std::pair<std::uint32_t, std::uint32_t>;
using BoundaryAggregate = std::map<BoundaryKey, BoundaryAccum>;

void fold_boundary_stats(BoundaryAggregate& dst, const state_dedup::BoundaryStats& src)
{
    for (const auto& rec : src.records)
    {
        BoundaryAccum& a = dst[{rec.entry_begin, rec.entry_end}];
        a.calls += 1;
        a.sum_frontier_in += rec.frontier_in;
        a.sum_frontier_out += rec.frontier_out;
        a.sum_map_ns += rec.map_ns;
        a.sum_hash_ns += rec.hash_ns;
    }
}

// Experiment knob (1-way path only): stride between consecutive window data
// values. stride=1 sweeps the low data bytes; stride=2^8/2^16 redirects the
// window onto higher byte positions. Default 1 = production behaviour.
std::uint32_t g_data_stride = 1;
std::string g_win_policy;   // empty/"linear" => linear scan; else squeeze/backfill/... (production window bit-selection)

template <typename TM>
double bench_flat(TM& tm, std::uint32_t key, std::uint32_t data_start,
                  std::uint32_t window, const key_schedule& schedule,
                  state_dedup::FlatTable& out, state_dedup::FlatTable& scratch,
                  std::uint64_t& parity_out, std::size_t& unique_out,
                  std::uint32_t dedup_every_maps,
                  std::uint32_t first_dedup_maps,
                  const std::vector<std::uint32_t>* checkpoint_entries,
                  bool dedup_expanded_states,
                  state_dedup::BoundaryStats* stats = nullptr,
                  bool skip_final_hash = false,
                  bool use_il4 = false,
                  bool use_il6 = false,
                  bool use_il8 = false,
                  bool use_il10 = false,
                  bool use_il12 = false,
                  bool use_il14 = false,
                  bool use_il2 = false)
{
    typedef std::chrono::high_resolution_clock clock;
    const auto t0 = clock::now();
    const bool full_window = state_dedup::is_full_u32_window(window);

    if (stats != nullptr) stats->clear();
    if (!full_window && use_il14)
    {
        if constexpr (requires { &TM::run_maps_range_x14; })
            state_dedup::forward_block_with_dedup_il14(
                tm, key, data_start, window, schedule, out, scratch,
                dedup_every_maps, first_dedup_maps, stats);
        else
            state_dedup::forward_block_with_dedup(
                tm, key, data_start, window, schedule, out, scratch,
                dedup_every_maps, first_dedup_maps, checkpoint_entries,
                dedup_expanded_states, stats, skip_final_hash);
    }
    else if (!full_window && use_il12)
    {
        if constexpr (requires { &TM::run_maps_range_x12; })
            state_dedup::forward_block_with_dedup_il12(
                tm, key, data_start, window, schedule, out, scratch,
                dedup_every_maps, first_dedup_maps, stats);
        else
            state_dedup::forward_block_with_dedup(
                tm, key, data_start, window, schedule, out, scratch,
                dedup_every_maps, first_dedup_maps, checkpoint_entries,
                dedup_expanded_states, stats, skip_final_hash);
    }
    else if (!full_window && use_il10)
    {
        if constexpr (requires { &TM::run_maps_range_x10; })
            state_dedup::forward_block_with_dedup_il10(
                tm, key, data_start, window, schedule, out, scratch,
                dedup_every_maps, first_dedup_maps, stats);
        else
            state_dedup::forward_block_with_dedup(
                tm, key, data_start, window, schedule, out, scratch,
                dedup_every_maps, first_dedup_maps, checkpoint_entries,
                dedup_expanded_states, stats, skip_final_hash);
    }
    else if (!full_window && use_il8)
    {
        if constexpr (requires { &TM::run_maps_range_x8; })
            state_dedup::forward_block_with_dedup_il8(
                tm, key, data_start, window, schedule, out, scratch,
                dedup_every_maps, first_dedup_maps, stats);
        else
            state_dedup::forward_block_with_dedup(
                tm, key, data_start, window, schedule, out, scratch,
                dedup_every_maps, first_dedup_maps, checkpoint_entries,
                dedup_expanded_states, stats, skip_final_hash);
    }
    else if (!full_window && use_il6)
    {
        // PROTOTYPE: 6-way interleaved frontier replay (f1k4 default shape).
        // Only impls exposing run_maps_range_x6 (AVX-512) take this path; others
        // fall back to the standard driver.
        if constexpr (requires { &TM::run_maps_range_x6; })
            state_dedup::forward_block_with_dedup_il6(
                tm, key, data_start, window, schedule, out, scratch,
                dedup_every_maps, first_dedup_maps, stats);
        else
            state_dedup::forward_block_with_dedup(
                tm, key, data_start, window, schedule, out, scratch,
                dedup_every_maps, first_dedup_maps, checkpoint_entries,
                dedup_expanded_states, stats, skip_final_hash);
    }
    else if (!full_window && use_il4)
    {
        // PROTOTYPE: 4-way interleaved frontier replay (f1k4 default shape).
        // Only impls exposing run_maps_range_x4 (AVX-512) take this path; others
        // fall back to the standard driver.
        if constexpr (requires { &TM::run_maps_range_x4; })
            state_dedup::forward_block_with_dedup_il4(
                tm, key, data_start, window, schedule, out, scratch,
                dedup_every_maps, first_dedup_maps, stats);
        else
            state_dedup::forward_block_with_dedup(
                tm, key, data_start, window, schedule, out, scratch,
                dedup_every_maps, first_dedup_maps, checkpoint_entries,
                dedup_expanded_states, stats, skip_final_hash);
    }
    else if (!full_window && use_il2)
    {
        if constexpr (requires { &TM::run_maps_range_x2; })
            state_dedup::forward_block_with_dedup_il2(
                tm, key, data_start, window, schedule, out, scratch,
                dedup_every_maps, first_dedup_maps, stats);
        else
            state_dedup::forward_block_with_dedup(
                tm, key, data_start, window, schedule, out, scratch,
                dedup_every_maps, first_dedup_maps, checkpoint_entries,
                dedup_expanded_states, stats, skip_final_hash);
    }
    else
    {
        extern std::uint32_t g_data_stride;
        extern std::string g_win_policy;
        // Production window bit-selection (squeeze/backfill/...): scan data =
        // deposit_bits32(idx, mask) over a power-of-2 sub-2^32 window. Default (empty/linear) is the
        // contiguous low-bit scan. mask depends on log2(window), so it is built here per window.
        state_dedup::DataEnumerationConfig data_enum;
        if (!g_win_policy.empty() && g_win_policy != "linear" && !full_window &&
            (window & (window - 1u)) == 0u)
        {
            std::uint32_t bits = 0; while ((1u << bits) < window) bits++;
            data_enum.enabled = true;
            data_enum.variable_mask = tm_window_policy::make_bit_mask(
                tm_window_policy::parse_policy(g_win_policy), bits);
            data_enum.fixed_value = 0u;
            data_enum.source_multiplicity = 1u;
        }
        state_dedup::forward_block_with_dedup(
            tm, key, data_start, window, schedule, out, scratch,
            dedup_every_maps, first_dedup_maps, checkpoint_entries,
            dedup_expanded_states, stats, skip_final_hash,
            /*map1_source_prefilter=*/nullptr, /*batch_inserts=*/true,
            /*data_stride=*/g_data_stride,
            /*data_enumeration=*/data_enum.enabled ? &data_enum : nullptr);
    }

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

std::vector<std::uint32_t> parse_checkpoints(const std::string& text)
{
    std::vector<std::uint32_t> checkpoints;
    std::stringstream ss(text);
    std::string tok;
    while (std::getline(ss, tok, ','))
    {
        if (!tok.empty()) checkpoints.push_back(static_cast<std::uint32_t>(std::stoul(tok, nullptr, 0)));
    }
    std::sort(checkpoints.begin(), checkpoints.end());
    checkpoints.erase(std::unique(checkpoints.begin(), checkpoints.end()), checkpoints.end());
    return checkpoints;
}

struct Args
{
    std::string keys_path;
    std::string out_path;
    std::vector<std::uint32_t> windows;
    std::vector<std::string> impls;
    std::uint32_t data_start = 0;
    std::uint32_t dedup_every_maps = 1;
    std::uint32_t first_dedup_maps = 0;
    std::vector<std::uint32_t> checkpoint_entries;
    bool dedup_expanded_states = false;
    std::size_t limit = 0;
    int repeats = 3;
    key_schedule::map_list_type map_kind = key_schedule::ALL_MAPS;
    std::string boundary_stats_path;
    bool flat_only = false;
    int threads = 1;
    std::string label;        // optional label written to scaling CSV
    bool scaling = false;     // emit aggregate-per-impl scaling row instead of per-cell rows
    bool skip_final_hash = false;
    bool interleave2 = false;  // 2-way interleaved frontier replay (AVX2 map kernel; run_maps_range_x2)
    bool interleave4 = false;  // PROTOTYPE: 4-way interleaved frontier replay (AVX-512 only)
    bool interleave6 = false;  // PROTOTYPE: 6-way interleaved frontier replay (AVX-512 only)
    bool interleave8 = false;  // PROTOTYPE: 8-way interleaved frontier replay (AVX-512 only)
    bool interleave10 = false; // PROTOTYPE: 10-way interleaved frontier replay (AVX-512 only)
    bool interleave12 = false; // PROTOTYPE: 12-way interleaved frontier replay (AVX-512 only; width ceiling probe)
    bool interleave14 = false; // PROTOTYPE: 14-way interleaved frontier replay (AVX-512 only; spill confirmation)
    unsigned front_cache_bits = 0; // tier-1 front cache: 2^N entries (0 = disabled)
    bool dynamic_table = false;    // size dedup table to frontier, not window
    bool no_warmup = false;        // skip cache-priming warm-up pass (survey speed)
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
        else if (s == "--data-stride") g_data_stride = static_cast<std::uint32_t>(std::stoul(nxt(i, "--data-stride"), nullptr, 0));
        else if (s == "--win-policy") g_win_policy = nxt(i, "--win-policy");   // squeeze/backfill/lowbits (production window bit-selection)
        else if (s == "--dedup-every-maps") a.dedup_every_maps = static_cast<std::uint32_t>(std::stoul(nxt(i, "--dedup-every-maps"), nullptr, 0));
        else if (s == "--first-dedup-maps") a.first_dedup_maps = static_cast<std::uint32_t>(std::stoul(nxt(i, "--first-dedup-maps"), nullptr, 0));
        else if (s == "--dedup-checkpoints") a.checkpoint_entries = parse_checkpoints(nxt(i, "--dedup-checkpoints"));
        else if (s == "--skip-expand-dedup") a.dedup_expanded_states = false;  // legacy / no-op (default)
        else if (s == "--dedup-expand-states") a.dedup_expanded_states = true;  // opt back into legacy behaviour
        else if (s == "--limit")      a.limit = static_cast<std::size_t>(std::stoull(nxt(i, "--limit")));
        else if (s == "--repeats")    a.repeats = std::stoi(nxt(i, "--repeats"));
        else if (s == "--boundary-stats") a.boundary_stats_path = nxt(i, "--boundary-stats");
        else if (s == "--flat-only")  a.flat_only = true;
        else if (s == "--threads")    a.threads = std::stoi(nxt(i, "--threads"));
        else if (s == "--label")      a.label = nxt(i, "--label");
        else if (s == "--scaling")    a.scaling = true;
        else if (s == "--skip-final-hash") a.skip_final_hash = true;
        else if (s == "--interleave2") a.interleave2 = true;
        else if (s == "--interleave4") a.interleave4 = true;
        else if (s == "--interleave6") a.interleave6 = true;
        else if (s == "--interleave8") a.interleave8 = true;
        else if (s == "--interleave10") a.interleave10 = true;
        else if (s == "--interleave12") a.interleave12 = true;
        else if (s == "--interleave14") a.interleave14 = true;
        else if (s == "--front-cache-bits") a.front_cache_bits = static_cast<unsigned>(std::stoul(nxt(i, "--front-cache-bits")));
        else if (s == "--dynamic-table") a.dynamic_table = true;
        else if (s == "--no-warmup") a.no_warmup = true;
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
    if (a.dedup_every_maps == 0) throw std::runtime_error("--dedup-every-maps must be > 0");
    return a;
}

// One-impl bench helper: for each key, each window, run baseline + flat
// for `repeats`, write median timings + candidates/sec to the CSV.
template <typename TM>
void bench_impl(const std::string& impl_name, TM& tm,
                const std::vector<std::uint32_t>& keys,
                const std::vector<std::uint32_t>& windows,
                std::uint32_t data_start, int repeats,
                std::uint32_t dedup_every_maps,
                std::uint32_t first_dedup_maps,
                const std::vector<std::uint32_t>& checkpoint_entries,
                bool dedup_expanded_states,
                key_schedule::map_list_type map_kind,
                std::ofstream& out,
                std::size_t& parity_mismatches_out,
                BoundaryAggregate* boundary_agg,
                bool flat_only,
                bool skip_final_hash,
                bool use_il4,
                bool use_il6 = false,
                bool use_il8 = false,
                bool use_il10 = false,
                bool use_il12 = false,
                bool use_il14 = false,
                bool use_il2 = false)
{
    state_dedup::FlatTable dedup_out, dedup_scratch;
    if (g_front_cache_bits)
    {
        dedup_out.enable_front_cache(g_front_cache_bits);
        dedup_scratch.enable_front_cache(g_front_cache_bits);
    }
    if (g_dynamic_table)
    {
        dedup_out.set_dynamic(true);
        dedup_scratch.set_dynamic(true);
    }
    state_dedup::BoundaryStats boundary_buf;

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
            const std::uint64_t effective_window = state_dedup::effective_window_count(window);

            std::vector<double> ts_base, ts_flat;
            std::uint64_t parity_base = 0, parity_flat = 0;
            std::size_t unique = 0;

            // Warm-up to prime caches (skipped under --no-warmup; final_unique is
            // timing-invariant, so surveys avoid the doubled work / guard trip).
            if (!g_no_warmup)
            {
                std::uint64_t p; std::size_t u;
                if (!flat_only) (void)bench_baseline(tm, key, data_start, window, schedule, p);
                (void)bench_flat    (tm, key, data_start, window, schedule, dedup_out, dedup_scratch, p, u,
                                     dedup_every_maps, first_dedup_maps,
                                     checkpoint_entries.empty() ? nullptr : &checkpoint_entries,
                                     dedup_expanded_states, nullptr, skip_final_hash, use_il4, use_il6, use_il8, use_il10, use_il12, use_il14, use_il2);
            }

            for (int r = 0; r < repeats; r++)
            {
                if (!flat_only)
                    ts_base.push_back(bench_baseline(tm, key, data_start, window, schedule, parity_base));
                ts_flat.push_back(bench_flat    (tm, key, data_start, window, schedule, dedup_out, dedup_scratch, parity_flat, unique,
                                                dedup_every_maps, first_dedup_maps,
                                                checkpoint_entries.empty() ? nullptr : &checkpoint_entries,
                                                dedup_expanded_states,
                                                boundary_agg ? &boundary_buf : nullptr,
                                                skip_final_hash, use_il4, use_il6, use_il8, use_il10, use_il12, use_il14, use_il2));
                if (boundary_agg) fold_boundary_stats(*boundary_agg, boundary_buf);
            }

            const double m_base = flat_only ? 0.0 : median(ts_base);
            const double m_flat = median(ts_flat);
            const double cps_base = m_base > 0 ? static_cast<double>(effective_window) / m_base : 0;
            const double cps_flat = m_flat > 0 ? static_cast<double>(effective_window) / m_flat : 0;
            const double dedup_speedup = m_flat > 0 && m_base > 0 ? m_base / m_flat : 0;
            const bool parity_ok = flat_only ? true : (parity_base == parity_flat);
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

// Concurrency-scaling variant. Runs `threads` workers in parallel against a
// single (impl, policy) cell. Each worker pulls (key,window) tasks via an
// atomic counter and runs flat-only dedup with `repeats` iterations per task.
// Reports aggregate throughput as states/sec — total data values processed
// over total wall-clock from the moment all workers begin.
template <typename TM_FACTORY>
void bench_impl_threaded(const std::string& impl_name,
                         const std::string& label,
                         const TM_FACTORY& make_tm,
                         const std::vector<std::uint32_t>& keys,
                         const std::vector<std::uint32_t>& windows,
                         std::uint32_t data_start, int repeats, int threads,
                         std::uint32_t dedup_every_maps,
                         std::uint32_t first_dedup_maps,
                         const std::vector<std::uint32_t>& checkpoint_entries,
                         bool dedup_expanded_states,
                         key_schedule::map_list_type map_kind,
                         std::ofstream& scaling_out,
                         bool skip_final_hash,
                         bool use_il4 = false,
                         bool use_il6 = false,
                         bool use_il8 = false,
                         bool use_il10 = false,
                         bool use_il12 = false,
                         bool use_il14 = false,
                         bool use_il2 = false)
{
    const std::size_t total_tasks = keys.size() * windows.size();
    if (total_tasks == 0) return;

    std::atomic<std::uint64_t> next{0};
    std::vector<std::uint64_t> per_thread_states(threads, 0);

    auto run_one = [&](int tid)
    {
        auto tm_holder = make_tm();
        auto& tm = *tm_holder;
        state_dedup::FlatTable out, scratch;
        if (g_front_cache_bits)
        {
            out.enable_front_cache(g_front_cache_bits);
            scratch.enable_front_cache(g_front_cache_bits);
        }
        if (g_dynamic_table)
        {
            out.set_dynamic(true);
            scratch.set_dynamic(true);
        }

        while (true)
        {
            const std::uint64_t task = next.fetch_add(1);
            if (task >= total_tasks) break;
            const std::size_t k_idx = static_cast<std::size_t>(task / windows.size());
            const std::size_t w_idx = static_cast<std::size_t>(task % windows.size());
            const std::uint32_t key = keys[k_idx];
            const std::uint32_t window = windows[w_idx];
            const std::uint64_t effective_window = state_dedup::effective_window_count(window);
            key_schedule schedule(key, map_kind);

            std::uint64_t total_states_this_task = 0;
            for (int r = 0; r < repeats; r++)
            {
                std::uint64_t p; std::size_t u;
                (void)bench_flat(tm, key, data_start, window, schedule, out, scratch,
                                 p, u, dedup_every_maps, first_dedup_maps,
                                 checkpoint_entries.empty() ? nullptr : &checkpoint_entries,
                                 dedup_expanded_states, nullptr, skip_final_hash, use_il4, use_il6, use_il8, use_il10, use_il12, use_il14, use_il2);
                total_states_this_task += effective_window;
            }
            per_thread_states[static_cast<std::size_t>(tid)] += total_states_this_task;
        }
    };

    typedef std::chrono::high_resolution_clock clock;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    const auto t0 = clock::now();
    for (int t = 0; t < threads; t++) workers.emplace_back(run_one, t);
    for (auto& w : workers) w.join();
    const double wall = std::chrono::duration<double>(clock::now() - t0).count();

    std::uint64_t total_states = 0;
    for (auto v : per_thread_states) total_states += v;
    const double mps = wall > 0 ? static_cast<double>(total_states) / wall / 1e6 : 0.0;
    const double per_thread_mps = mps / static_cast<double>(threads);

    scaling_out << label << ',' << impl_name << ',' << threads << ','
                << total_states << ',' << std::fixed << std::setprecision(6) << wall << ','
                << std::setprecision(3) << mps << ',' << per_thread_mps << '\n';
    scaling_out.flush();

    std::cerr << "  [" << impl_name << "/" << label << "/t=" << threads << "] "
              << std::fixed << std::setprecision(2) << mps << " M/s ("
              << std::setprecision(2) << per_thread_mps << " M/s/thread)\n";
}

} // anonymous namespace

int main(int argc, char** argv)
{
    try
    {
        Args a = parse_args(argc, argv);
        g_front_cache_bits = a.front_cache_bits;
        g_dynamic_table = a.dynamic_table;
        g_no_warmup = a.no_warmup;
        std::vector<std::uint32_t> keys = key_file::read_keys(a.keys_path, a.limit);
        if (keys.empty()) throw std::runtime_error("no keys parsed from: " + a.keys_path);

        std::vector<std::uint32_t> windows_sorted(a.windows);
        std::sort(windows_sorted.begin(), windows_sorted.end(),
                  [](std::uint32_t lhs, std::uint32_t rhs) {
                      return state_dedup::effective_window_count(lhs) < state_dedup::effective_window_count(rhs);
                  });

        std::ofstream out;
        if (!a.scaling)
        {
            out.open(a.out_path);
            if (!out) throw std::runtime_error("cannot open output: " + a.out_path);
            out << "key_hex,window,impl,baseline_s,flat_s,baseline_cps,flat_cps,"
                   "dedup_speedup,final_unique_states,parity_match\n";
        }

        std::cerr << "State-dedup matrix bench\n"
                  << "  keys:       " << keys.size() << "\n"
                  << "  data_start: 0x" << std::hex << std::setw(8) << std::setfill('0')
                  << a.data_start << std::dec << std::setfill(' ') << "\n"
                  << "  dedup_every_maps: " << a.dedup_every_maps << "\n"
                  << "  first_dedup_maps: " << a.first_dedup_maps << "\n"
                  << "  dedup_expanded_states: " << (a.dedup_expanded_states ? "yes" : "no") << "\n"
                  << "  windows:    ";
        for (std::size_t i = 0; i < windows_sorted.size(); i++)
        {
            std::cerr << (i == 0 ? "" : ",");
            if (state_dedup::is_full_u32_window(windows_sorted[i]))
                std::cerr << "0(full-2^32)";
            else
                std::cerr << windows_sorted[i];
        }
        if (!a.checkpoint_entries.empty())
        {
            std::cerr << "\n  dedup_checkpoints: ";
            for (std::size_t i = 0; i < a.checkpoint_entries.size(); i++)
                std::cerr << (i == 0 ? "" : ",") << a.checkpoint_entries[i];
        }
        std::cerr << "\n  impls:      ";
        for (std::size_t i = 0; i < a.impls.size(); i++)
            std::cerr << (i == 0 ? "" : ",") << a.impls[i];
        std::cerr << "\n  repeats:    " << a.repeats << "\n";

        RNG rng;
        std::size_t parity_mismatches = 0;
        const bool collect_boundary = !a.boundary_stats_path.empty();
        std::map<std::string, BoundaryAggregate> per_impl_boundary;

        auto get_agg = [&](const std::string& name) -> BoundaryAggregate* {
            if (!collect_boundary) return nullptr;
            return &per_impl_boundary[name];
        };

        // Concurrency-scaling path: --scaling emits one aggregate row per
        // (impl, threads). The per-impl bench's per-cell CSV is bypassed
        // since the parallel run can't preserve serial timings.
        if (a.scaling)
        {
            if (!a.flat_only)
                throw std::runtime_error("--scaling requires --flat-only");
            if (collect_boundary)
                throw std::runtime_error("--scaling is incompatible with --boundary-stats");

            std::ofstream scaling(a.out_path);
            if (!scaling) throw std::runtime_error("cannot open scaling output: " + a.out_path);
            scaling << "label,impl,threads,total_states,wall_s,throughput_Mps,per_thread_Mps\n";

            const std::string label = a.label.empty() ? std::string("policy") : a.label;
            for (const auto& impl_name : a.impls)
            {
                if (impl_name == "tm_avx2_r256s_8")
                {
                    auto make_tm = [&](){
                        static thread_local RNG trng;
                        return std::make_unique<tm_avx2_r256s_8>(&trng);
                    };
                    bench_impl_threaded(impl_name, label, make_tm, keys, windows_sorted,
                        a.data_start, a.repeats, a.threads, a.dedup_every_maps,
                        a.first_dedup_maps, a.checkpoint_entries,
                        a.dedup_expanded_states, a.map_kind, scaling, a.skip_final_hash);
                }
                else if (impl_name == "tm_avx2_r256_map_8")
                {
                    auto make_tm = [&](){
                        static thread_local RNG trng;
                        return std::make_unique<tm_avx2_r256_map_8>(&trng);
                    };
                    bench_impl_threaded(impl_name, label, make_tm, keys, windows_sorted,
                        a.data_start, a.repeats, a.threads, a.dedup_every_maps,
                        a.first_dedup_maps, a.checkpoint_entries,
                        a.dedup_expanded_states, a.map_kind, scaling, a.skip_final_hash,
                        false, false, false, false, false, false, a.interleave2);  // AVX2 map: --interleave2 -> run_maps_range_x2
                }
                else if (impl_name == "tm_avx512_r512s_8")
                {
                    auto make_tm = [&](){
                        static thread_local RNG trng;
                        return std::make_unique<tm_avx512_r512s_8>(&trng);
                    };
                    bench_impl_threaded(impl_name, label, make_tm, keys, windows_sorted,
                        a.data_start, a.repeats, a.threads, a.dedup_every_maps,
                        a.first_dedup_maps, a.checkpoint_entries,
                        a.dedup_expanded_states, a.map_kind, scaling, a.skip_final_hash,
                        a.interleave4, a.interleave6, a.interleave8, a.interleave10, a.interleave12, a.interleave14);  // --interleaveN routes AVX-512 through the N-way kernel
                }
                else if (impl_name == "tm_avx512_r512_map_8")
                {
                    auto make_tm = [&](){
                        static thread_local RNG trng;
                        return std::make_unique<tm_avx512_r512_map_8>(&trng);
                    };
                    bench_impl_threaded(impl_name, label, make_tm, keys, windows_sorted,
                        a.data_start, a.repeats, a.threads, a.dedup_every_maps,
                        a.first_dedup_maps, a.checkpoint_entries,
                        a.dedup_expanded_states, a.map_kind, scaling, a.skip_final_hash,
                        a.interleave4, a.interleave6, a.interleave8, a.interleave10, a.interleave12, a.interleave14);  // natural map now has x8/x12/x14
                }
                else if (impl_name == "tm_avx512_r512s_map_8")
                {
                    auto make_tm = [&](){
                        static thread_local RNG trng;
                        return std::make_unique<tm_avx512_r512s_map_8>(&trng);
                    };
                    bench_impl_threaded(impl_name, label, make_tm, keys, windows_sorted,
                        a.data_start, a.repeats, a.threads, a.dedup_every_maps,
                        a.first_dedup_maps, a.checkpoint_entries,
                        a.dedup_expanded_states, a.map_kind, scaling, a.skip_final_hash,
                        a.interleave4);  // --interleave4 routes AVX-512 through the 4-way kernel
                }
                else
                {
                    std::cerr << "scaling mode skips impl: " << impl_name << "\n";
                }
            }
            std::cerr << "scaling: wrote " << a.out_path << "\n";
            return 0;
        }

        for (const auto& impl_name : a.impls)
        {
            BoundaryAggregate* agg = get_agg(impl_name);
            if (impl_name == "tm_8")
            {
                tm_8 tm(&rng);
                bench_impl(impl_name, tm, keys, windows_sorted, a.data_start, a.repeats, a.dedup_every_maps, a.first_dedup_maps, a.checkpoint_entries, a.dedup_expanded_states, a.map_kind, out, parity_mismatches, agg, a.flat_only, a.skip_final_hash, a.interleave4);
            }
            else if (impl_name == "tm_avx_r128s_8")
            {
                tm_avx_r128s_8 tm(&rng);
                bench_impl(impl_name, tm, keys, windows_sorted, a.data_start, a.repeats, a.dedup_every_maps, a.first_dedup_maps, a.checkpoint_entries, a.dedup_expanded_states, a.map_kind, out, parity_mismatches, agg, a.flat_only, a.skip_final_hash, a.interleave4);
            }
            else if (impl_name == "tm_avx_r256s_8")
            {
                tm_avx_r256s_8 tm(&rng);
                bench_impl(impl_name, tm, keys, windows_sorted, a.data_start, a.repeats, a.dedup_every_maps, a.first_dedup_maps, a.checkpoint_entries, a.dedup_expanded_states, a.map_kind, out, parity_mismatches, agg, a.flat_only, a.skip_final_hash, a.interleave4);
            }
            else if (impl_name == "tm_avx2_r256s_8")
            {
                tm_avx2_r256s_8 tm(&rng);
                bench_impl(impl_name, tm, keys, windows_sorted, a.data_start, a.repeats, a.dedup_every_maps, a.first_dedup_maps, a.checkpoint_entries, a.dedup_expanded_states, a.map_kind, out, parity_mismatches, agg, a.flat_only, a.skip_final_hash, a.interleave4);
            }
            else if (impl_name == "tm_avx2_r256_map_8")
            {
                tm_avx2_r256_map_8 tm(&rng);
                bench_impl(impl_name, tm, keys, windows_sorted, a.data_start, a.repeats, a.dedup_every_maps, a.first_dedup_maps, a.checkpoint_entries, a.dedup_expanded_states, a.map_kind, out, parity_mismatches, agg, a.flat_only, a.skip_final_hash, a.interleave4, false, false, false, false, false, a.interleave2);
            }
            else if (impl_name == "tm_avx512_r512s_8")
            {
                tm_avx512_r512s_8 tm(&rng);
                bench_impl(impl_name, tm, keys, windows_sorted, a.data_start, a.repeats, a.dedup_every_maps, a.first_dedup_maps, a.checkpoint_entries, a.dedup_expanded_states, a.map_kind, out, parity_mismatches, agg, a.flat_only, a.skip_final_hash, a.interleave4, a.interleave6, a.interleave8, a.interleave10, a.interleave12, a.interleave14);
            }
            else if (impl_name == "tm_avx512_r512_map_8")
            {
                tm_avx512_r512_map_8 tm(&rng);
                bench_impl(impl_name, tm, keys, windows_sorted, a.data_start, a.repeats, a.dedup_every_maps, a.first_dedup_maps, a.checkpoint_entries, a.dedup_expanded_states, a.map_kind, out, parity_mismatches, agg, a.flat_only, a.skip_final_hash, a.interleave4, a.interleave6, a.interleave8, a.interleave10, a.interleave12, a.interleave14);
            }
            else if (impl_name == "tm_avx512_r512s_map_8")
            {
                tm_avx512_r512s_map_8 tm(&rng);
                bench_impl(impl_name, tm, keys, windows_sorted, a.data_start, a.repeats, a.dedup_every_maps, a.first_dedup_maps, a.checkpoint_entries, a.dedup_expanded_states, a.map_kind, out, parity_mismatches, agg, a.flat_only, a.skip_final_hash, a.interleave4);
            }
            else
            {
                std::cerr << "unknown impl: " << impl_name << " (skipped)\n";
            }
        }

        if (collect_boundary)
        {
            std::ofstream bs(a.boundary_stats_path);
            if (!bs) throw std::runtime_error("cannot open boundary stats output: " + a.boundary_stats_path);
            bs << "impl,entry_begin,entry_end,calls,avg_frontier_in,avg_frontier_out,collapse_ratio,total_map_ns,total_hash_ns,map_per_state_ns,hash_per_state_ns,hash_share\n";
            for (const auto& impl_kv : per_impl_boundary)
            {
                for (const auto& kv : impl_kv.second)
                {
                    const auto& acc = kv.second;
                    const double avg_in = acc.calls ? static_cast<double>(acc.sum_frontier_in) / acc.calls : 0.0;
                    const double avg_out = acc.calls ? static_cast<double>(acc.sum_frontier_out) / acc.calls : 0.0;
                    const double collapse = avg_in > 0.0 ? avg_out / avg_in : 0.0;
                    const double map_per_state = acc.sum_frontier_in > 0
                        ? static_cast<double>(acc.sum_map_ns) / static_cast<double>(acc.sum_frontier_in) : 0.0;
                    const double hash_per_state = acc.sum_frontier_in > 0
                        ? static_cast<double>(acc.sum_hash_ns) / static_cast<double>(acc.sum_frontier_in) : 0.0;
                    const double total = static_cast<double>(acc.sum_map_ns + acc.sum_hash_ns);
                    const double hash_share = total > 0.0 ? static_cast<double>(acc.sum_hash_ns) / total : 0.0;
                    bs << impl_kv.first << ',' << kv.first.first << ',' << kv.first.second << ',' << acc.calls << ','
                       << std::fixed << std::setprecision(2) << avg_in << ',' << avg_out << ','
                       << std::setprecision(4) << collapse << ','
                       << acc.sum_map_ns << ',' << acc.sum_hash_ns << ','
                       << std::setprecision(2) << map_per_state << ',' << hash_per_state << ','
                       << std::setprecision(4) << hash_share << '\n';
                }
            }
            std::cerr << "boundary stats: " << a.boundary_stats_path << "\n";
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
                     "(use W=0 for exact 2^32/W4B) "
                     "[--impl tm_8,tm_avx2_r256s_8,...] [--data-start HEX] [--repeats R] "
                     "[--dedup-every-maps N] [--first-dedup-maps N] [--dedup-checkpoints LIST] "
                     "[--skip-expand-dedup] [--dedup-expand-states] [--boundary-stats CSV] "
                     "[--flat-only] [--scaling --threads N --label NAME] "
                     "[--schedule all|skip-car] [--limit N]\n";
        return 1;
    }
}
