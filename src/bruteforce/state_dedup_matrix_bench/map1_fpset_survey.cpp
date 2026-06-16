// Fingerprint-only MAP1 (stage-1) collapse survey.
//
// Measures the post-MAP1 frontier (unique states after the first map group) over
// a data window, using a 128-bit fingerprint SET instead of a 128-byte state
// pool. Memory ~32 B/unique-state (no state storage), so the full 2^32 (W4B)
// data axis is surveyable even for non-closing keys. This is the stage-1 metric
// of the staged W4B architecture (findings §32-§33): the frontier/artifact size,
// and the front-loaded bulk of the collapse.
//
// Output CSV: key_hex,window,map1_unique,R_map1,wall_s,peak_kb_unused
//   R_map1 = window / map1_unique (the MAP1 collapse ratio at this window).
//
// Correctness is via 128-bit fp (collision ~1e-19 at N=2^32). Parity is checked
// against the state-based MAP1 frontier_out (matrix bench --boundary-stats).

#include "rng_obj.h"
#include "key_schedule.h"
#include "key_file.h"
#include "state_dedup.h"

#include "../cpu/tm_avx512_r512s_8.h"
#include "../cpu/tm_avx512_r512_map_8.h"
#include "../cpu/tm_avx2_r256s_8.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::uint32_t> parse_u32_csv(const std::string& s)
{
    std::vector<std::uint32_t> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) v.push_back(static_cast<std::uint32_t>(std::stoul(tok, nullptr, 0)));
    return v;
}

// Per-window MAP1 screen config surface (mirrors RoutingConfig's first-class-struct style). When
// cap.enabled, the survey runs the capped+threaded inverse-Bloom screen (flat RSS, fits W4B) via
// state_dedup::map1_screen_capped instead of the exact FpSet (linear RSS); win_policy selects the
// production window bit-selection (squeeze/backfill/...). map1_unique then reports the cap's KEPT
// (over-kept live-set) count rather than the exact distinct count.
struct ScreenDriverConfig
{
    state_dedup::ScreenCapConfig cap;   // cap.enabled gates the capped path
    unsigned threads = 1;
    std::string win_policy;             // empty/"linear" => linear; else squeeze/backfill/...
};

// Resolve the window-bit-selection mask for a window under the chosen policy. Returns 0 (linear,
// data==idx) for the full-2^32 window or an empty/linear policy; otherwise requires a power-of-2
// window (selected_bit_count = log2(window)), matching map1par / the production screen.
inline std::uint32_t resolve_win_mask(const std::string& policy, std::uint32_t window)
{
    if (policy.empty() || policy == "linear" || state_dedup::is_full_u32_window(window)) return 0u;
    if ((window & (window - 1u)) != 0u)
        throw std::runtime_error("--win-policy requires a power-of-2 window (or W=0/full-2^32)");
    std::uint32_t bits = 0; while ((1u << bits) < window) bits++;
    return tm_window_policy::make_bit_mask(tm_window_policy::parse_policy(policy), bits);
}

template <typename TM>
void run_impl(TM& tm, const std::vector<std::uint32_t>& keys,
              const std::vector<std::uint32_t>& windows,
              std::uint32_t data_start, std::uint32_t first_dedup_maps,
              std::uint32_t dedup_every_maps, std::ofstream& out,
              const ScreenDriverConfig& cfg)
{
    using clock = std::chrono::high_resolution_clock;
    state_dedup::FpSet set;
    const double cap_mb = cfg.cap.enabled
        ? static_cast<double>(((std::uint64_t)1 << cfg.cap.cap_bits)
            + (cfg.cap.drain_bits ? ((std::uint64_t)1 << cfg.cap.drain_bits) : 0)) * 8.0 / 1048576.0
        : 0.0;
    for (std::size_t ki = 0; ki < keys.size(); ki++)
    {
        const std::uint32_t key = keys[ki];
        key_schedule schedule(key, key_schedule::ALL_MAPS);
        for (std::uint32_t window : windows)
        {
            const std::uint64_t effective_window = state_dedup::effective_window_count(window);
            const std::uint32_t win_mask = resolve_win_mask(cfg.win_policy, window);
            const char* mode = cfg.cap.enabled ? "cap" : "exact";
            const auto t0 = clock::now();
            const std::size_t uniq = cfg.cap.enabled
                ? state_dedup::map1_screen_capped<TM>(
                      key, window, schedule, cfg.cap, cfg.threads, win_mask,
                      dedup_every_maps, first_dedup_maps)
                : state_dedup::map1_fpset_unique(
                      tm, key, data_start, window, schedule, set,
                      dedup_every_maps, first_dedup_maps, nullptr, 1u);
            const double wall = std::chrono::duration<double>(clock::now() - t0).count();
            const double R = uniq ? static_cast<double>(effective_window) / static_cast<double>(uniq) : 0.0;
            out << "0x" << std::hex << std::setw(8) << std::setfill('0') << key
                << std::dec << std::setfill(' ')
                << ',' << window << ',' << uniq
                << ',' << std::fixed << std::setprecision(3) << R
                << ',' << std::fixed << std::setprecision(3) << wall
                << ',' << mode << ',' << std::setprecision(1) << cap_mb << "\n";
            out.flush();
            std::cerr << "  [" << (ki + 1) << "/" << keys.size() << "] 0x"
                      << std::hex << key << std::dec << " W=";
            if (state_dedup::is_full_u32_window(window))
                std::cerr << "0(full-2^32)";
            else
                std::cerr << window;
            std::cerr
                      << " " << mode << (cfg.cap.enabled ? "" : "") << "_kept=" << uniq << " R=" << std::fixed
                      << std::setprecision(2) << R << " (" << std::setprecision(1)
                      << wall << "s" << (cfg.cap.enabled ? (", cap " + std::to_string((long)cap_mb) + " MB") : std::string()) << ")\n";
        }
    }
}

// Diagnostic: run the same state stream through BOTH the fp128 FpSet and a
// memcmp FlatTable; if counts differ, fp128 is colliding (or there is an FpSet
// bug). Isolates fingerprint correctness from kernel/state correctness.
template <typename TM>
void verify_one(TM& tm, std::uint32_t key, std::uint32_t window,
                std::uint32_t first_dedup_maps, std::uint32_t dedup_every_maps)
{
    key_schedule schedule(key, key_schedule::ALL_MAPS);
    const std::size_t group_end = state_dedup::next_group_end(
        0, schedule.entries.size(), dedup_every_maps,
        first_dedup_maps ? first_dedup_maps : dedup_every_maps, nullptr);
    if constexpr (requires(TM& t, const key_schedule& s) { t.bind_dedup_schedule(s); })
        tm.bind_dedup_schedule(schedule);
    else
        tm.bind_schedule(schedule);
    state_dedup::prepare_map_group(tm, schedule, 0, group_end);

    state_dedup::FpSet set; set.reset();
    state_dedup::FlatTable ft; ft.reset(window);
    std::size_t fp_lo_distinct = 0;  // also track lo-only distinctness for entropy check
    state_dedup::FpSet lo_set; lo_set.reset();
    for (std::uint32_t i = 0; i < window; i++)
    {
        tm.expand(key, i);
        state_dedup::run_map_group(tm, schedule, 0, group_end);
        const std::uint8_t* st = tm.state_raw();
        const state_dedup::Fp128 fp = state_dedup::fingerprint128(st);
        set.insert(fp.lo, fp.hi);
        lo_set.insert(fp.lo, 0);
        ft.insert(st, 1u);
    }
    fp_lo_distinct = lo_set.count;
    std::cerr << "VERIFY key=0x" << std::hex << key << std::dec
              << " W=" << window << " group_end=" << group_end << "\n"
              << "  memcmp (FlatTable) unique = " << ft.pool.size() << "\n"
              << "  fp128  (FpSet)     unique = " << set.count << "\n"
              << "  fp64-lo only       unique = " << fp_lo_distinct << "\n"
              << "  fp128 deficit vs memcmp   = "
              << (static_cast<long long>(set.count) - static_cast<long long>(ft.pool.size())) << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    std::string keys_path, out_path, impl = "tm_avx512_r512_map_8";
    std::vector<std::uint32_t> windows;
    std::uint32_t data_start = 0, first_dedup_maps = 1, dedup_every_maps = 3;
    std::size_t limit = 0;
    bool verify = false;
    ScreenDriverConfig screen;          // cap/threads/win-policy config surface (default = exact path)

    auto nxt = [&](int& i) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing arg value");
        return argv[++i];
    };
    for (int i = 1; i < argc; i++)
    {
        std::string s = argv[i];
        if      (s == "--keys")             keys_path = nxt(i);
        else if (s == "--out")              out_path = nxt(i);
        else if (s == "--windows")          windows = parse_u32_csv(nxt(i));
        else if (s == "--impl")             impl = nxt(i);
        else if (s == "--data-start")       data_start = static_cast<std::uint32_t>(std::stoul(nxt(i), nullptr, 0));
        else if (s == "--first-dedup-maps") first_dedup_maps = static_cast<std::uint32_t>(std::stoul(nxt(i)));
        else if (s == "--dedup-every-maps") dedup_every_maps = static_cast<std::uint32_t>(std::stoul(nxt(i)));
        else if (s == "--limit")            limit = static_cast<std::size_t>(std::stoull(nxt(i)));
        else if (s == "--verify")           verify = true;
        // MAP1 screen cap config surface (gap G3 productionization). --cap-bits enables the capped path.
        else if (s == "--cap-bits")         { screen.cap.cap_bits = static_cast<std::uint32_t>(std::stoul(nxt(i))); screen.cap.enabled = true; }
        else if (s == "--cap-ways")         screen.cap.cap_ways = static_cast<std::uint32_t>(std::stoul(nxt(i)));
        else if (s == "--drain-bits")       screen.cap.drain_bits = static_cast<std::uint32_t>(std::stoul(nxt(i)));
        else if (s == "--drain-ways")       screen.cap.drain_ways = static_cast<std::uint32_t>(std::stoul(nxt(i)));
        else if (s == "--threads")          screen.threads = static_cast<unsigned>(std::stoul(nxt(i)));
        else if (s == "--win-policy")       screen.win_policy = nxt(i);
        else throw std::runtime_error("unknown arg: " + s);
    }
    if (screen.threads < 1u) screen.threads = 1u;
    if (keys_path.empty() || windows.empty())
        throw std::runtime_error("--keys, --windows required");

    std::vector<std::uint32_t> keys = key_file::read_keys(keys_path, limit);

    if (verify)
    {
        RNG rng;
        const std::uint32_t w = windows.front();
        if (state_dedup::is_full_u32_window(w))
            throw std::runtime_error("--verify does not support W=0/full-2^32; use a finite diagnostic window");
        for (std::uint32_t key : keys)
        {
            if (impl == "tm_avx512_r512_map_8") { tm_avx512_r512_map_8 tm(&rng); verify_one(tm, key, w, first_dedup_maps, dedup_every_maps); }
            else if (impl == "tm_avx512_r512s_8") { tm_avx512_r512s_8 tm(&rng); verify_one(tm, key, w, first_dedup_maps, dedup_every_maps); }
            else if (impl == "tm_avx2_r256s_8") { tm_avx2_r256s_8 tm(&rng); verify_one(tm, key, w, first_dedup_maps, dedup_every_maps); }
            else throw std::runtime_error("unknown impl: " + impl);
        }
        return 0;
    }

    if (out_path.empty()) throw std::runtime_error("--out required");
    std::ofstream out(out_path);
    if (!out) throw std::runtime_error("cannot open out: " + out_path);
    out << "key_hex,window,map1_unique,R_map1,wall_s,mode,cap_mb\n";

    std::cerr << "map1 fp-set survey: " << keys.size() << " keys, impl=" << impl
              << ", f" << first_dedup_maps << "k" << dedup_every_maps;
    if (screen.cap.enabled)
        std::cerr << " | SCREEN CAP 2^" << screen.cap.cap_bits << "x" << screen.cap.cap_ways << "way"
                  << (screen.cap.drain_bits ? (" +drain 2^" + std::to_string(screen.cap.drain_bits)) : std::string())
                  << " threads=" << screen.threads
                  << " win-policy=" << (screen.win_policy.empty() ? "linear" : screen.win_policy);
    std::cerr << "\n";

    RNG rng;
    if (impl == "tm_avx512_r512_map_8") { tm_avx512_r512_map_8 tm(&rng); run_impl(tm, keys, windows, data_start, first_dedup_maps, dedup_every_maps, out, screen); }
    else if (impl == "tm_avx512_r512s_8") { tm_avx512_r512s_8 tm(&rng); run_impl(tm, keys, windows, data_start, first_dedup_maps, dedup_every_maps, out, screen); }
    else if (impl == "tm_avx2_r256s_8") { tm_avx2_r256s_8 tm(&rng); run_impl(tm, keys, windows, data_start, first_dedup_maps, dedup_every_maps, out, screen); }
    else throw std::runtime_error("unknown impl: " + impl);
    return 0;
}
