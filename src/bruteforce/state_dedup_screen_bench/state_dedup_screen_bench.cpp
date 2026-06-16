#include "rng_obj.h"
#include "key_schedule.h"
#include "key_file.h"
#include "state_dedup.h"
#include "state_dedup_il.h"
#include "../cpu/tm_avx2_r256s_8.h"
#include "../cpu/tm_avx512_r512s_8.h"

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
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

constexpr uint8 OTHER_WORLD_FLAG = 0x01;

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

void merge_boundary_stats(BoundaryAggregate& dst, const BoundaryAggregate& src)
{
    for (const auto& kv : src)
    {
        BoundaryAccum& a = dst[kv.first];
        a.calls += kv.second.calls;
        a.sum_frontier_in += kv.second.sum_frontier_in;
        a.sum_frontier_out += kv.second.sum_frontier_out;
        a.sum_map_ns += kv.second.sum_map_ns;
        a.sum_hash_ns += kv.second.sum_hash_ns;
    }
}

struct Args
{
    std::string keys_path;
    std::string out_path;
    uint32 key = 0x2CA5B42Du;
    uint32 data_start = 0;
    uint32 window = 4096;
    uint32 windows_per_key = 1;
    // Default policy: first_dedup_maps=1, dedup_every_maps=4 ("first=1,K=4").
    // This dedups after MAP1 (captures the ~52% entry-0 collapse), then runs
    // K=4 merges over the rest of the schedule (7 merges total).
    //
    // Wins ~+10-14% over the historical K=1 default at the documented
    // production shape (window=4096, threads=12-24). Wins both
    // tm_avx2_r256s_8 and tm_avx2_r256_map_8 at small and large windows.
    //
    // See docs/hybrid_dedup_architecture_notes_20260527.md for the full
    // policy / per-window / scaling analysis. The architecture notes also
    // describe the i-cache pressure fix in `tm_avx2_r256_map_8::_run_maps_fixed`
    // that was load-bearing for this default — the multi-map unrolled cases
    // (count=3..9) caused billions of L1i misses on map mode; stripping them
    // and letting count > 1 fall through to the loop dropped L1i misses 100×
    // and re-balanced the policy bathtub onto f1k4 as the universal winner.
    uint32 dedup_every_maps = 4;
    uint32 first_dedup_maps = 1;
    std::vector<uint32> checkpoint_entries;
    bool dedup_expanded_states = false;
    bool interleave2 = false;  // PROTOTYPE: 2-way interleaved frontier replay
    bool batch_hash = false;   // PROTOTYPE: 1-way map + software-pipelined batched inserts (slim driver)
    bool batch_inserts = true; // PRODUCTION default: batched/prefetched inserts in forward_block_with_dedup
    bool map1_source_prefilter = false;
    bool map1_source_global_nibble_gate = true;
    uint32 map1_source_max_classes = 1;
    uint16 map1_source_sample_nibbles = 0xFFFFu;
    bool map1_window_sample_gate = false;
    uint32 map1_window_sample_pages = 4;
    double map1_window_sample_threshold = 4.0;
    std::string boundary_stats_path;
    uint8 strict_invalid_mask = USES_UNOFFICIAL_NOPS | USES_ILLEGAL_OPCODES | USES_JAM;
    int threads = 1;
    std::size_t limit = 0;
    key_schedule::map_list_type map_kind = key_schedule::ALL_MAPS;
    // Forward kernel: "avx2" (tm_avx2_r256s_8, 1-way) or "avx512"
    // (tm_avx512_r512s_8, auto-4-way in the dedup driver — the normalized
    // production forward).
    std::string impl = "avx2";
};

template <typename T>
T parse_u(const std::string& s)
{
    return static_cast<T>(std::stoull(s, nullptr, 0));
}

std::vector<uint32> parse_checkpoints(const std::string& text)
{
    std::vector<uint32> checkpoints;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        if (token.empty()) continue;
        checkpoints.push_back(parse_u<uint32>(token));
    }
    std::sort(checkpoints.begin(), checkpoints.end());
    checkpoints.erase(std::unique(checkpoints.begin(), checkpoints.end()), checkpoints.end());
    return checkpoints;
}

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
        if      (s == "--keys")            a.keys_path = nxt(i, "--keys");
        else if (s == "--key")             a.key = parse_u<uint32>(nxt(i, "--key"));
        else if (s == "--out")             a.out_path = nxt(i, "--out");
        else if (s == "--data-start")      a.data_start = parse_u<uint32>(nxt(i, "--data-start"));
        else if (s == "--window")          a.window = parse_u<uint32>(nxt(i, "--window"));
        else if (s == "--windows-per-key") a.windows_per_key = parse_u<uint32>(nxt(i, "--windows-per-key"));
        else if (s == "--dedup-every-maps") a.dedup_every_maps = parse_u<uint32>(nxt(i, "--dedup-every-maps"));
        else if (s == "--first-dedup-maps") a.first_dedup_maps = parse_u<uint32>(nxt(i, "--first-dedup-maps"));
        else if (s == "--dedup-checkpoints") a.checkpoint_entries = parse_checkpoints(nxt(i, "--dedup-checkpoints"));
        else if (s == "--skip-expand-dedup") a.dedup_expanded_states = false;  // legacy / no-op (now default)
        else if (s == "--dedup-expand-states") a.dedup_expanded_states = true;  // opt back into legacy behaviour
        else if (s == "--interleave2")     a.interleave2 = true;  // PROTOTYPE 2-way interleaved frontier replay
        else if (s == "--batch-hash")      a.batch_hash = true;   // PROTOTYPE 1-way map + batched inserts (slim driver)
        else if (s == "--serial-hash")     a.batch_inserts = false; // disable production batched inserts (A/B the promotion)
        else if (s == "--map1-source-prefilter") a.map1_source_prefilter = true;
        else if (s == "--map1-source-max-classes")
        {
            a.map1_source_prefilter = true;
            a.map1_source_max_classes = parse_u<uint32>(nxt(i, "--map1-source-max-classes"));
        }
        else if (s == "--map1-window-sample-gate") a.map1_window_sample_gate = true;
        else if (s == "--map1-window-sample-pages")
        {
            a.map1_window_sample_gate = true;
            a.map1_window_sample_pages = parse_u<uint32>(nxt(i, "--map1-window-sample-pages"));
        }
        else if (s == "--map1-window-sample-threshold")
        {
            a.map1_window_sample_gate = true;
            a.map1_window_sample_threshold = std::stod(nxt(i, "--map1-window-sample-threshold"));
        }
        else if (s == "--boundary-stats")  a.boundary_stats_path = nxt(i, "--boundary-stats");
        else if (s == "--strict-invalid-mask") a.strict_invalid_mask = parse_u<uint8>(nxt(i, "--strict-invalid-mask"));
        else if (s == "--threads")         a.threads = std::stoi(nxt(i, "--threads"));
        else if (s == "--impl")            a.impl = nxt(i, "--impl");
        else if (s == "--limit")           a.limit = static_cast<std::size_t>(std::stoull(nxt(i, "--limit")));
        else if (s == "--schedule")
        {
            std::string mode = nxt(i, "--schedule");
            if      (mode == "all")      a.map_kind = key_schedule::ALL_MAPS;
            else if (mode == "skip-car") a.map_kind = key_schedule::SKIP_CAR;
            else throw std::runtime_error("--schedule must be 'all' or 'skip-car'");
        }
        else if (s == "--help" || s == "-h")
        {
            std::cout
                << "state_dedup_screen_bench\n"
                << "  --keys FILE              CSV/key list; uses key_hex when present\n"
                << "  --key UINT32             single-key fallback\n"
                << "  --limit N                maximum keys from --keys\n"
                << "  --window N               data values per dedup block\n"
                << "  --windows-per-key N      consecutive windows sampled per key\n"
                << "  --dedup-every-maps N     merge frontier every N entries (default 4; universal bathtub-bottom)\n"
                << "                           Lower values increase merge density; higher values reduce L3\n"
                << "                           pressure but add map work between merges.\n"
                << "  --first-dedup-maps N     first merge group size; 0 uses --dedup-every-maps. Default 1\n"
                << "                           (captures the ~52% post-MAP1 collapse on its own merge step).\n"
                << "  --dedup-checkpoints LIST comma-separated map-entry cut points; overrides every/first\n"
                << "  --skip-expand-dedup      (default) run first map group before the first dedup merge\n"
                << "  --dedup-expand-states    opt back into the legacy expand-then-dedup behaviour\n"
                << "  --impl <avx2|avx512>     forward kernel: avx2 (tm_avx2_r256s_8, 1-way; default) or\n"
                << "                           avx512 (tm_avx512_r512s_8, auto-4-way dedup; production)\n"
                << "  --interleave2            PROTOTYPE: 2-way interleaved frontier replay (f1k4 default shape only; AVX2 only)\n"
                << "  --batch-hash             PROTOTYPE: slim 1-way map + batched-insert driver (f1k4 default shape only)\n"
                << "  --serial-hash            disable production batched inserts (default on) to A/B the promotion\n"
                << "  --map1-source-prefilter  static D=1 source-bit page compressor before MAP1 hash\n"
                << "  --map1-source-max-classes N  accept static page only if representative count <= N\n"
                << "  --map1-window-sample-gate sample MAP1 pages before enabling source prefilter\n"
                << "  --map1-window-sample-pages N  pages sampled per 64K window (default 4)\n"
                << "  --map1-window-sample-threshold R  enable source prefilter when sampled R >= R (default 4)\n"
                << "  --boundary-stats FILE    append per-boundary frontier sizes + map/hash wall to CSV\n"
                << "  --strict-invalid-mask N   flags rejected for strict all-entry count\n"
                << "  --threads N\n"
                << "  --data-start UINT32\n"
                << "  --schedule all|skip-car\n"
                << "  --out FILE               optional per-window CSV\n";
            std::exit(0);
        }
        else throw std::runtime_error("unknown arg: " + s);
    }
    if (a.window == 0) throw std::runtime_error("--window must be > 0");
    if (a.windows_per_key == 0) throw std::runtime_error("--windows-per-key must be > 0");
    if (a.dedup_every_maps == 0) throw std::runtime_error("--dedup-every-maps must be > 0");
    if (a.map1_window_sample_gate && a.window != 65536)
        throw std::runtime_error("--map1-window-sample-gate currently requires --window 65536");
    if (a.map1_window_sample_gate && (a.map1_window_sample_pages == 0 || (256u % a.map1_window_sample_pages) != 0))
        throw std::runtime_error("--map1-window-sample-pages must be > 0 and divide 256");
    if (a.map1_window_sample_gate && a.map1_window_sample_threshold <= 0.0)
        throw std::runtime_error("--map1-window-sample-threshold must be > 0");
    if (a.threads < 1) throw std::runtime_error("--threads must be >= 1");
    if (a.interleave2 && a.batch_hash)
        throw std::runtime_error("--interleave2 and --batch-hash are mutually exclusive");
    if (a.interleave2 || a.batch_hash)
    {
        const char* w = a.interleave2 ? "--interleave2" : "--batch-hash";
        if (a.map1_source_prefilter || a.map1_window_sample_gate)
            throw std::runtime_error(std::string(w) + " prototype does not support MAP1 prefilter/sampler");
        if (a.dedup_expanded_states)
            throw std::runtime_error(std::string(w) + " prototype does not support --dedup-expand-states");
        if (!a.checkpoint_entries.empty())
            throw std::runtime_error(std::string(w) + " prototype does not support --dedup-checkpoints");
    }
    return a;
}

struct Counts
{
    std::uint64_t windows = 0;
    std::uint64_t candidates = 0;
    std::uint64_t unique = 0;
    std::uint64_t checksum_unique = 0;
    std::uint64_t checksum_origins = 0;
    std::uint64_t carnival_unique = 0;
    std::uint64_t other_unique = 0;
    std::uint64_t dual_unique = 0;
    std::uint64_t first_entry_unique = 0;
    std::uint64_t first_entry_origins = 0;
    std::uint64_t all_entries_unique = 0;
    std::uint64_t all_entries_origins = 0;
    std::uint64_t strict_all_entries_unique = 0;
    std::uint64_t strict_all_entries_origins = 0;
    std::uint64_t nop_unique = 0;
    std::uint64_t nop2_unique = 0;
    std::uint64_t illegal_unique = 0;
    std::uint64_t jam_unique = 0;
    std::uint64_t checksum_windows = 0;
    std::uint64_t all_entries_windows = 0;
    std::uint64_t strict_all_entries_windows = 0;
    std::uint64_t map1_sampled_windows = 0;
    std::uint64_t map1_sample_pass_windows = 0;
    std::uint64_t map1_sample_pages = 0;
    std::uint64_t map1_sample_ns = 0;
    std::uint64_t forward_ns = 0;   // time in forward_block_with_dedup
    std::uint64_t screen_ns = 0;    // time in screen_table (final decrypt+checksum+machine-code)

    void add(const Counts& o)
    {
        windows += o.windows;
        candidates += o.candidates;
        unique += o.unique;
        checksum_unique += o.checksum_unique;
        checksum_origins += o.checksum_origins;
        carnival_unique += o.carnival_unique;
        other_unique += o.other_unique;
        dual_unique += o.dual_unique;
        first_entry_unique += o.first_entry_unique;
        first_entry_origins += o.first_entry_origins;
        all_entries_unique += o.all_entries_unique;
        all_entries_origins += o.all_entries_origins;
        strict_all_entries_unique += o.strict_all_entries_unique;
        strict_all_entries_origins += o.strict_all_entries_origins;
        nop_unique += o.nop_unique;
        nop2_unique += o.nop2_unique;
        illegal_unique += o.illegal_unique;
        jam_unique += o.jam_unique;
        checksum_windows += o.checksum_windows;
        all_entries_windows += o.all_entries_windows;
        strict_all_entries_windows += o.strict_all_entries_windows;
        map1_sampled_windows += o.map1_sampled_windows;
        map1_sample_pass_windows += o.map1_sample_pass_windows;
        map1_sample_pages += o.map1_sample_pages;
        map1_sample_ns += o.map1_sample_ns;
        forward_ns += o.forward_ns;
        screen_ns += o.screen_ns;
    }
};

struct WindowRow
{
    uint32 key = 0;
    uint32 window_index = 0;
    uint32 data_start = 0;
    Counts counts;
    double elapsed_s = 0.0;
    bool map1_sampled = false;
    bool map1_sample_pass = false;
    double map1_sample_R = 0.0;
    std::uint64_t map1_sample_ns = 0;
    bool map1_source_prefilter_used = false;
};

struct ScreenResult
{
    bool carnival = false;
    bool other = false;
    uint8 flags = 0;
};

template <typename TM>
ScreenResult screen_entry(
    TM& tm,
    const state_dedup::FlatTable::Entry& entry)
{
    ScreenResult result;
    uint8 unshuffled[128];

    tm.load_state_raw(entry.state.data());
    tm.decrypt_carnival_world();
    if (tm.calculate_carnival_world_checksum() == tm.fetch_carnival_world_checksum_value())
    {
        result.carnival = true;
        tm.fetch_data(unshuffled);
        result.flags |= tm.check_machine_code(unshuffled, CARNIVAL_WORLD);
    }

    tm.load_state_raw(entry.state.data());
    tm.decrypt_other_world();
    if (tm.calculate_other_world_checksum() == tm.fetch_other_world_checksum_value())
    {
        result.other = true;
        tm.fetch_data(unshuffled);
        result.flags |= static_cast<uint8>(
            tm.check_machine_code(unshuffled, OTHER_WORLD) | OTHER_WORLD_FLAG);
    }

    return result;
}

template <typename TM>
Counts screen_table(TM& tm, const state_dedup::FlatTable& table, uint8 strict_invalid_mask)
{
    Counts c;
    c.unique = table.pool.size();

    bool any_checksum = false;
    bool any_all_entries = false;
    bool any_strict_all_entries = false;
    for (const auto& entry : table.pool)
    {
        const ScreenResult r = screen_entry(tm, entry);
        if (!r.carnival && !r.other) continue;

        any_checksum = true;
        c.checksum_unique++;
        c.checksum_origins += entry.multiplicity;
        if (r.carnival) c.carnival_unique++;
        if (r.other) c.other_unique++;
        if (r.carnival && r.other) c.dual_unique++;

        if ((r.flags & FIRST_ENTRY_VALID) != 0)
        {
            c.first_entry_unique++;
            c.first_entry_origins += entry.multiplicity;
        }
        if ((r.flags & ALL_ENTRIES_VALID) != 0)
        {
            any_all_entries = true;
            c.all_entries_unique++;
            c.all_entries_origins += entry.multiplicity;

            if ((r.flags & strict_invalid_mask) == 0)
            {
                any_strict_all_entries = true;
                c.strict_all_entries_unique++;
                c.strict_all_entries_origins += entry.multiplicity;
            }
        }
        if ((r.flags & USES_NOP) != 0) c.nop_unique++;
        if ((r.flags & USES_UNOFFICIAL_NOPS) != 0) c.nop2_unique++;
        if ((r.flags & USES_ILLEGAL_OPCODES) != 0) c.illegal_unique++;
        if ((r.flags & USES_JAM) != 0) c.jam_unique++;
    }

    if (any_checksum) c.checksum_windows = 1;
    if (any_all_entries) c.all_entries_windows = 1;
    if (any_strict_all_entries) c.strict_all_entries_windows = 1;
    return c;
}

template <typename TM>
double sample_map1_window_R(
    TM& tm,
    uint32 key,
    uint32 data_start,
    const key_schedule& schedule,
    uint32 sample_pages,
    state_dedup::FlatTable& table)
{
    state_dedup::prepare_map_group(tm, schedule, 0, 1);
    const uint32 step = 256u / sample_pages;
    std::uint64_t class_sum = 0;
    for (uint32 page = 0; page < 256u; page += step)
    {
        table.reset(256);
        const uint32 page_start = data_start + page * 256u;
        for (uint32 i = 0; i < 256u; i++)
        {
            tm.expand(key, page_start + i);
            state_dedup::run_map_group(tm, schedule, 0, 1);
            table.insert(tm.state_raw(), 1u);
        }
        class_sum += table.pool.size();
    }
    return class_sum > 0
        ? 256.0 / (static_cast<double>(class_sum) / static_cast<double>(sample_pages))
        : 0.0;
}

std::string hex_key(uint32 key)
{
    std::ostringstream ss;
    ss << "0x" << std::hex << std::setw(8) << std::setfill('0') << key;
    return ss.str();
}

void write_rows(const std::string& path, const std::vector<WindowRow>& rows)
{
    if (path.empty()) return;

    std::ofstream out(path);
    if (!out) throw std::runtime_error("could not open output CSV: " + path);
    out << "key_hex,window_index,data_start,window,candidates,unique,"
        << "checksum_unique,checksum_origins,carnival_unique,other_unique,dual_unique,"
        << "first_entry_unique,first_entry_origins,all_entries_unique,all_entries_origins,"
        << "strict_all_entries_unique,strict_all_entries_origins,"
        << "nop_unique,nop2_unique,illegal_unique,jam_unique,"
        << "map1_sampled,map1_sample_pass,map1_sample_R,map1_sample_ns,"
        << "map1_source_prefilter_used,elapsed_s,rate_cps\n";
    for (const WindowRow& r : rows)
    {
        const Counts& c = r.counts;
        const double rate = r.elapsed_s > 0.0
            ? static_cast<double>(c.candidates) / r.elapsed_s
            : 0.0;
        out << hex_key(r.key) << ','
            << r.window_index << ','
            << r.data_start << ','
            << (c.windows == 0 ? 0 : c.candidates / c.windows) << ','
            << c.candidates << ','
            << c.unique << ','
            << c.checksum_unique << ','
            << c.checksum_origins << ','
            << c.carnival_unique << ','
            << c.other_unique << ','
            << c.dual_unique << ','
            << c.first_entry_unique << ','
            << c.first_entry_origins << ','
            << c.all_entries_unique << ','
            << c.all_entries_origins << ','
            << c.strict_all_entries_unique << ','
            << c.strict_all_entries_origins << ','
            << c.nop_unique << ','
            << c.nop2_unique << ','
            << c.illegal_unique << ','
            << c.jam_unique << ','
            << (r.map1_sampled ? 1 : 0) << ','
            << (r.map1_sample_pass ? 1 : 0) << ','
            << std::fixed << std::setprecision(6) << r.map1_sample_R << ','
            << r.map1_sample_ns << ','
            << (r.map1_source_prefilter_used ? 1 : 0) << ','
            << std::fixed << std::setprecision(6) << r.elapsed_s << ','
            << std::fixed << std::setprecision(0) << rate << '\n';
    }
}

template <typename TM>
Counts run_sweep(const Args& a, const std::vector<uint32>& keys, std::vector<WindowRow>& rows, double& elapsed_s)
{
    RNG warm_rng;
    TM warm_tm(&warm_rng);
    if (a.map1_source_prefilter)
        state_dedup::ensure_map1_source_tables();

    const std::uint64_t total_tasks =
        static_cast<std::uint64_t>(keys.size()) * static_cast<std::uint64_t>(a.windows_per_key);
    std::atomic<std::uint64_t> next{0};
    std::vector<Counts> thread_counts(static_cast<std::size_t>(a.threads));
    std::vector<BoundaryAggregate> thread_boundary(static_cast<std::size_t>(a.threads));
    const bool collect_boundary_stats = !a.boundary_stats_path.empty();
    std::vector<WindowRow> local_rows;
    std::mutex rows_mu;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(a.threads));

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int tid = 0; tid < a.threads; tid++)
    {
        workers.emplace_back([&, tid]()
        {
            RNG rng;
            TM tm(&rng);
            state_dedup::FlatTable out, scratch;
            state_dedup::FlatTable sample_table;
            state_dedup::Map1SourcePrefilterConfig map1_prefilter;
            map1_prefilter.enabled = a.map1_source_prefilter;
            map1_prefilter.global_nibble_gate = a.map1_source_global_nibble_gate;
            map1_prefilter.sample_nibble_mask = a.map1_source_sample_nibbles;
            map1_prefilter.sample_max_classes = a.map1_source_max_classes;
            Counts local_total;
            std::vector<WindowRow> mine;
            BoundaryAggregate boundary_local;
            state_dedup::BoundaryStats boundary_buf;

            while (true)
            {
                const std::uint64_t task = next.fetch_add(1);
                if (task >= total_tasks) break;

                const std::size_t key_idx = static_cast<std::size_t>(task / a.windows_per_key);
                const uint32 window_idx = static_cast<uint32>(task % a.windows_per_key);
                const uint32 key = keys[key_idx];
                const uint32 start = a.data_start + window_idx * a.window;
                key_schedule schedule(key, a.map_kind);

                const auto w0 = std::chrono::high_resolution_clock::now();
                bool sample_pass = true;
                double sample_R = 0.0;
                std::uint64_t sample_ns = 0;
                if (a.map1_window_sample_gate)
                {
                    const auto sample_t0 = std::chrono::steady_clock::now();
                    sample_R = sample_map1_window_R(
                        tm, key, start, schedule, a.map1_window_sample_pages, sample_table);
                    sample_ns = static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - sample_t0).count());
                    sample_pass = sample_R >= a.map1_window_sample_threshold;
                }
                const bool use_map1_source_prefilter =
                    a.map1_source_prefilter &&
                    (!a.map1_window_sample_gate || sample_pass);
                if (collect_boundary_stats) boundary_buf.clear();
                const auto fwd_t0 = std::chrono::steady_clock::now();
                if (a.interleave2 && [&]{ if constexpr (requires { &TM::run_maps_range_x2; }) return true; else return false; }())
                {
                    // PROTOTYPE path: slim 2-way interleaved driver (AVX2 only —
                    // needs run_maps_range_x2). Supports only the production-default
                    // f1k4 shape (no prefilter, no expand-dedup, no checkpoints,
                    // no skip-final-hash).
                    if constexpr (requires { &TM::run_maps_range_x2; })
                        state_dedup::forward_block_with_dedup_il2(
                            tm, key, start, a.window, schedule, out, scratch,
                            a.dedup_every_maps, a.first_dedup_maps,
                            collect_boundary_stats ? &boundary_buf : nullptr);
                }
                else if (a.batch_hash)
                {
                    // PROTOTYPE path: 1-way map + software-pipelined batched inserts.
                    state_dedup::forward_block_with_dedup_bh(
                        tm, key, start, a.window, schedule, out, scratch,
                        a.dedup_every_maps, a.first_dedup_maps,
                        collect_boundary_stats ? &boundary_buf : nullptr);
                }
                else
                {
                    state_dedup::forward_block_with_dedup(
                        tm, key, start, a.window, schedule, out, scratch,
                        a.dedup_every_maps, a.first_dedup_maps,
                        a.checkpoint_entries.empty() ? nullptr : &a.checkpoint_entries,
                        a.dedup_expanded_states,
                        collect_boundary_stats ? &boundary_buf : nullptr,
                        false,
                        use_map1_source_prefilter ? &map1_prefilter : nullptr,
                        a.batch_inserts);
                }
                const std::uint64_t fwd_ns = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - fwd_t0).count());
                if (collect_boundary_stats) fold_boundary_stats(boundary_local, boundary_buf);
                const auto scr_t0 = std::chrono::steady_clock::now();
                Counts wc = screen_table(tm, out, a.strict_invalid_mask);
                const std::uint64_t scr_ns = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - scr_t0).count());
                wc.forward_ns = fwd_ns;
                wc.screen_ns = scr_ns;
                wc.windows = 1;
                wc.candidates = a.window;
                if (a.map1_window_sample_gate)
                {
                    wc.map1_sampled_windows = 1;
                    wc.map1_sample_pass_windows = sample_pass ? 1 : 0;
                    wc.map1_sample_pages = a.map1_window_sample_pages;
                    wc.map1_sample_ns = sample_ns;
                }
                const double ws = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - w0).count();

                local_total.add(wc);
                if (!a.out_path.empty())
                {
                    WindowRow row;
                    row.key = key;
                    row.window_index = window_idx;
                    row.data_start = start;
                    row.counts = wc;
                    row.elapsed_s = ws;
                    row.map1_sampled = a.map1_window_sample_gate;
                    row.map1_sample_pass = a.map1_window_sample_gate && sample_pass;
                    row.map1_sample_R = sample_R;
                    row.map1_sample_ns = sample_ns;
                    row.map1_source_prefilter_used = use_map1_source_prefilter;
                    mine.push_back(row);
                }
            }

            thread_counts[static_cast<std::size_t>(tid)] = local_total;
            if (collect_boundary_stats)
                thread_boundary[static_cast<std::size_t>(tid)] = std::move(boundary_local);
            if (!mine.empty())
            {
                std::lock_guard<std::mutex> lock(rows_mu);
                rows.insert(rows.end(), mine.begin(), mine.end());
            }
        });
    }

    for (auto& worker : workers) worker.join();

    if (collect_boundary_stats)
    {
        BoundaryAggregate merged;
        for (const auto& tb : thread_boundary) merge_boundary_stats(merged, tb);

        std::ofstream bs(a.boundary_stats_path);
        if (!bs) throw std::runtime_error("could not open boundary stats file: " + a.boundary_stats_path);
        bs << "entry_begin,entry_end,calls,avg_frontier_in,avg_frontier_out,collapse_ratio,total_map_ns,total_hash_ns,map_per_state_ns,hash_per_state_ns,hash_share\n";
        for (const auto& kv : merged)
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
            bs << kv.first.first << ',' << kv.first.second << ',' << acc.calls << ','
               << std::fixed << std::setprecision(2) << avg_in << ',' << avg_out << ','
               << std::setprecision(4) << collapse << ','
               << acc.sum_map_ns << ',' << acc.sum_hash_ns << ','
               << std::setprecision(2) << map_per_state << ',' << hash_per_state << ','
               << std::setprecision(4) << hash_share << '\n';
        }
        std::cout << "  boundary stats:       " << a.boundary_stats_path << "\n";
    }
    elapsed_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    Counts total;
    for (const Counts& c : thread_counts) total.add(c);
    std::sort(rows.begin(), rows.end(), [](const WindowRow& a, const WindowRow& b)
    {
        if (a.key != b.key) return a.key < b.key;
        return a.window_index < b.window_index;
    });
    return total;
}

void print_summary(const Args& a, const std::vector<uint32>& keys, const Counts& c, double elapsed_s)
{
    const double total_rate = elapsed_s > 0.0 ? static_cast<double>(c.candidates) / elapsed_s / 1000000.0 : 0.0;
    const double per_thread = total_rate / static_cast<double>(a.threads);
    const double unique_rate = c.unique > 0 ? static_cast<double>(c.checksum_unique) / static_cast<double>(c.unique) : 0.0;
    const double all_rate = c.unique > 0 ? static_cast<double>(c.all_entries_unique) / static_cast<double>(c.unique) : 0.0;
    const double origin_replay_rate = c.windows > 0 ? static_cast<double>(c.all_entries_windows) / static_cast<double>(c.windows) : 0.0;
    const double strict_replay_rate = c.windows > 0 ? static_cast<double>(c.strict_all_entries_windows) / static_cast<double>(c.windows) : 0.0;

    std::cout << "Flat dedup screen bench: "
              << ((a.impl == "avx512" || a.impl == "tm_avx512_r512s_8") ? "tm_avx512_r512s_8 (auto-4-way)" : "tm_avx2_r256s_8")
              << "\n";
    std::cout << "  keys:                 " << keys.size() << "\n";
    std::cout << "  window:               " << a.window << "\n";
    std::cout << "  windows/key:          " << a.windows_per_key << "\n";
    std::cout << "  dedup every maps:     " << a.dedup_every_maps << "\n";
    std::cout << "  first dedup maps:     " << a.first_dedup_maps << "\n";
    std::cout << "  dedup expanded states:" << (a.dedup_expanded_states ? " yes" : " no") << "\n";
    std::cout << "  MAP1 source prefilter:" << (a.map1_source_prefilter ? " on" : " off") << "\n";
    if (a.map1_source_prefilter)
        std::cout << "  MAP1 source max cls:  " << a.map1_source_max_classes << "\n";
    std::cout << "  MAP1 window sampler: " << (a.map1_window_sample_gate ? " on" : " off") << "\n";
    if (a.map1_window_sample_gate)
    {
        std::cout << "  MAP1 sample pages:   " << a.map1_window_sample_pages << "\n";
        std::cout << "  MAP1 sample thresh:  " << std::fixed << std::setprecision(3)
                  << a.map1_window_sample_threshold << "\n";
    }
    if (!a.checkpoint_entries.empty())
    {
        std::cout << "  dedup checkpoints:    ";
        for (std::size_t i = 0; i < a.checkpoint_entries.size(); i++)
            std::cout << (i == 0 ? "" : ",") << a.checkpoint_entries[i];
        std::cout << "\n";
    }
    std::cout << "  threads:              " << a.threads << "\n";
    std::cout << "  windows:              " << c.windows << "\n";
    if (a.map1_window_sample_gate)
    {
        std::cout << "  sampled windows:      " << c.map1_sampled_windows << "\n";
        std::cout << "  sample-pass windows:  " << c.map1_sample_pass_windows << "\n";
        std::cout << "  sampled MAP1 pages:   " << c.map1_sample_pages << "\n";
        std::cout << "  sample time:          " << std::fixed << std::setprecision(3)
                  << c.map1_sample_ns / 1.0e6 << " ms\n";
    }
    std::cout << "  candidates:           " << c.candidates << "\n";
    std::cout << "  unique states:        " << c.unique << "\n";
    std::cout << "  elapsed_s:            " << std::fixed << std::setprecision(3) << elapsed_s << "\n";
    std::cout << "  total_rate:           " << std::fixed << std::setprecision(3) << total_rate << " M/s\n";
    std::cout << "  rate/thread:          " << std::fixed << std::setprecision(3) << per_thread << " M/s/thread\n";
    {
        const double fwd_ms = c.forward_ns / 1.0e6;
        const double scr_ms = c.screen_ns / 1.0e6;
        const double sum = static_cast<double>(c.forward_ns + c.screen_ns);
        const double scr_share = sum > 0.0 ? static_cast<double>(c.screen_ns) / sum : 0.0;
        std::cout << "  forward time (sum):   " << std::fixed << std::setprecision(1) << fwd_ms << " ms\n";
        std::cout << "  screen time (sum):    " << std::fixed << std::setprecision(1) << scr_ms << " ms ("
                  << std::setprecision(2) << (scr_share * 100.0) << "% of fwd+screen)\n";
    }
    std::cout << "  checksum unique:      " << c.checksum_unique << " (" << std::setprecision(8) << unique_rate << "/unique)\n";
    std::cout << "  checksum origins:     " << c.checksum_origins << "\n";
    std::cout << "  checksum windows:     " << c.checksum_windows << "\n";
    std::cout << "  carnival unique:      " << c.carnival_unique << "\n";
    std::cout << "  other unique:         " << c.other_unique << "\n";
    std::cout << "  dual unique:          " << c.dual_unique << "\n";
    std::cout << "  first-entry unique:   " << c.first_entry_unique << "\n";
    std::cout << "  first-entry origins:  " << c.first_entry_origins << "\n";
    std::cout << "  all-entries unique:   " << c.all_entries_unique << " (" << std::setprecision(8) << all_rate << "/unique)\n";
    std::cout << "  all-entries origins:  " << c.all_entries_origins << "\n";
    std::cout << "  all-entries windows:  " << c.all_entries_windows << " (" << std::setprecision(8) << origin_replay_rate << "/window)\n";
    std::cout << "  strict invalid mask:  0x" << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<unsigned int>(a.strict_invalid_mask) << std::dec << std::setfill(' ') << "\n";
    std::cout << "  strict-all unique:    " << c.strict_all_entries_unique << "\n";
    std::cout << "  strict-all origins:   " << c.strict_all_entries_origins << "\n";
    std::cout << "  strict-all windows:   " << c.strict_all_entries_windows << " (" << std::setprecision(8) << strict_replay_rate << "/window)\n";
    std::cout << "  nop unique:           " << c.nop_unique << "\n";
    std::cout << "  unofficial-nop unique:" << c.nop2_unique << "\n";
    std::cout << "  illegal unique:       " << c.illegal_unique << "\n";
    std::cout << "  jam unique:           " << c.jam_unique << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Args a = parse_args(argc, argv);
        std::vector<uint32> keys;
        if (!a.keys_path.empty())
            keys = key_file::read_keys(a.keys_path, a.limit);
        else
            keys.push_back(a.key);
        if (keys.empty()) throw std::runtime_error("no keys to process");

        std::vector<WindowRow> rows;
        double elapsed_s = 0.0;
        Counts total;
        if (a.impl == "avx512" || a.impl == "tm_avx512_r512s_8")
            total = run_sweep<tm_avx512_r512s_8>(a, keys, rows, elapsed_s);
        else if (a.impl == "avx2" || a.impl == "tm_avx2_r256s_8")
            total = run_sweep<tm_avx2_r256s_8>(a, keys, rows, elapsed_s);
        else
            throw std::runtime_error("unknown --impl: " + a.impl + " (use avx2 or avx512)");
        write_rows(a.out_path, rows);
        print_summary(a, keys, total, elapsed_s);
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
