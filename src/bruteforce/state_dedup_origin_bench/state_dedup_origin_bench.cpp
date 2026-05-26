#include "rng_obj.h"
#include "key_schedule.h"
#include "key_file.h"
#include "state_dedup_origins.h"
#include "../cpu/tm_avx2_r256s_8.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

struct Args
{
    uint32 key = 0x2CA5B42Du;
    std::string keys_path;
    uint32 data_start = 0;
    uint32 window = 4096;
    int repeats = 3;
    int threads = 1;
    std::size_t limit = 0;
    bool verify = true;
    key_schedule::map_list_type map_kind = key_schedule::ALL_MAPS;
};

template <typename T>
T parse_u(const std::string& s)
{
    return static_cast<T>(std::stoul(s, nullptr, 0));
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
        if      (s == "--key")        a.key = parse_u<uint32>(nxt(i, "--key"));
        else if (s == "--keys")       a.keys_path = nxt(i, "--keys");
        else if (s == "--data-start") a.data_start = parse_u<uint32>(nxt(i, "--data-start"));
        else if (s == "--window")     a.window = parse_u<uint32>(nxt(i, "--window"));
        else if (s == "--repeats")    a.repeats = std::stoi(nxt(i, "--repeats"));
        else if (s == "--threads")    a.threads = std::stoi(nxt(i, "--threads"));
        else if (s == "--limit")      a.limit = static_cast<std::size_t>(std::stoull(nxt(i, "--limit")));
        else if (s == "--no-verify")  a.verify = false;
        else if (s == "--schedule")
        {
            std::string mode = nxt(i, "--schedule");
            if      (mode == "all")      a.map_kind = key_schedule::ALL_MAPS;
            else if (mode == "skip-car") a.map_kind = key_schedule::SKIP_CAR;
            else throw std::runtime_error("--schedule must be 'all' or 'skip-car'");
        }
        else throw std::runtime_error("unknown arg: " + s);
    }
    if (a.window == 0) throw std::runtime_error("--window must be > 0");
    if (a.repeats < 1) throw std::runtime_error("--repeats must be >= 1");
    if (a.threads < 1) throw std::runtime_error("--threads must be >= 1");
    return a;
}

std::size_t count_origins(
    const state_dedup_origins::OriginTable& table,
    const state_dedup_origins::OriginTable::Entry& entry)
{
    std::size_t n = 0;
    for (uint32 node = entry.origin_head;
         node != state_dedup_origins::NO_ORIGIN;
         node = table.origins[node].next)
    {
        n++;
    }
    return n;
}

void verify_origins(
    tm_avx2_r256s_8& tm,
    uint32 key,
    uint32 data_start,
    uint32 window,
    const key_schedule& schedule,
    const state_dedup_origins::OriginTable& table)
{
    std::vector<uint8> seen(window, 0);
    std::size_t origin_total = 0;

    for (const auto& entry : table.pool)
    {
        const std::size_t origin_count = count_origins(table, entry);
        if (origin_count != entry.multiplicity)
            throw std::runtime_error("origin count does not match multiplicity");

        for (uint32 node = entry.origin_head;
             node != state_dedup_origins::NO_ORIGIN;
             node = table.origins[node].next)
        {
            const uint32 origin = table.origins[node].origin;
            if (origin >= window) throw std::runtime_error("origin out of range");
            if (seen[origin] != 0) throw std::runtime_error("duplicate origin");
            seen[origin] = 1;
            origin_total++;

            tm.expand(key, data_start + origin);
            tm.run_all_maps(schedule);
            if (std::memcmp(tm.state_raw(), entry.state.data(), 128) != 0)
                throw std::runtime_error("origin final state mismatch");
        }
    }

    if (origin_total != window)
        throw std::runtime_error("origin total does not match window");
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct ThreadStats
{
    std::uint64_t windows = 0;
    std::uint64_t candidates = 0;
    std::uint64_t final_unique = 0;
    std::uint64_t origins = 0;
    std::uint32_t max_origins = 0;
};

void run_threaded(
    const Args& a,
    const std::vector<uint32>& keys)
{
    // Force one-time static RNG/table initialization before worker threads.
    RNG warm_rng;
    tm_avx2_r256s_8 warm_tm(&warm_rng);

    std::atomic<std::size_t> next{0};
    std::vector<ThreadStats> stats(static_cast<std::size_t>(a.threads));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(a.threads));

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int tid = 0; tid < a.threads; tid++)
    {
        workers.emplace_back([&, tid]()
        {
            RNG rng;
            tm_avx2_r256s_8 tm(&rng);
            state_dedup_origins::OriginTable out, scratch;
            ThreadStats local;

            while (true)
            {
                const std::size_t idx = next.fetch_add(1);
                if (idx >= keys.size()) break;

                const uint32 key = keys[idx];
                key_schedule schedule(key, a.map_kind);
                for (int r = 0; r < a.repeats; r++)
                {
                    state_dedup_origins::forward_block_with_origin_dedup(
                        tm, key, a.data_start + static_cast<uint32>(r) * a.window,
                        a.window, schedule, out, scratch);
                    local.windows++;
                    local.candidates += a.window;
                    local.final_unique += out.pool.size();
                    local.origins += out.origins.size();
                    for (const auto& entry : out.pool)
                    {
                        local.max_origins = std::max<std::uint32_t>(
                            local.max_origins,
                            static_cast<std::uint32_t>(count_origins(out, entry)));
                    }
                }
            }
            stats[static_cast<std::size_t>(tid)] = local;
        });
    }

    for (auto& worker : workers) worker.join();
    const double elapsed_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    ThreadStats total;
    for (const ThreadStats& s : stats)
    {
        total.windows += s.windows;
        total.candidates += s.candidates;
        total.final_unique += s.final_unique;
        total.origins += s.origins;
        total.max_origins = std::max(total.max_origins, s.max_origins);
    }

    const double total_rate = static_cast<double>(total.candidates) / elapsed_s / 1000000.0;
    const double per_thread = total_rate / static_cast<double>(a.threads);
    const double avg_unique = total.windows > 0
        ? static_cast<double>(total.final_unique) / static_cast<double>(total.windows)
        : 0.0;

    std::cout << "Origin dedup threaded bench: tm_avx2_r256s_8\n";
    std::cout << "  keys:          " << keys.size() << "\n";
    std::cout << "  window:        " << a.window << "\n";
    std::cout << "  repeats/key:   " << a.repeats << "\n";
    std::cout << "  threads:       " << a.threads << "\n";
    std::cout << "  windows:       " << total.windows << "\n";
    std::cout << "  candidates:    " << total.candidates << "\n";
    std::cout << "  elapsed_s:     " << std::fixed << std::setprecision(6) << elapsed_s << "\n";
    std::cout << "  total_rate:    " << std::fixed << std::setprecision(3) << total_rate << " M/s\n";
    std::cout << "  rate/thread:   " << std::fixed << std::setprecision(3) << per_thread << " M/s/thread\n";
    std::cout << "  avg_unique:    " << std::fixed << std::setprecision(1) << avg_unique << "\n";
    std::cout << "  max_origins:   " << total.max_origins << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        Args a = parse_args(argc, argv);
        if (!a.keys_path.empty())
        {
            std::vector<uint32> keys = key_file::read_keys(a.keys_path, a.limit);
            if (keys.empty()) throw std::runtime_error("no keys parsed from: " + a.keys_path);
            run_threaded(a, keys);
            return 0;
        }

        RNG rng;
        tm_avx2_r256s_8 tm(&rng);
        key_schedule schedule(a.key, a.map_kind);
        state_dedup_origins::OriginTable out, scratch;

        std::vector<double> times;
        times.reserve(a.repeats);
        for (int r = 0; r < a.repeats; r++)
        {
            const auto t0 = std::chrono::high_resolution_clock::now();
            state_dedup_origins::forward_block_with_origin_dedup(
                tm, a.key, a.data_start, a.window, schedule, out, scratch);
            times.push_back(std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0).count());
        }

        if (a.verify)
            verify_origins(tm, a.key, a.data_start, a.window, schedule, out);

        const double med_s = median(times);
        const double rate = static_cast<double>(a.window) / med_s / 1000000.0;

        std::size_t max_origins = 0;
        for (const auto& entry : out.pool)
            max_origins = std::max(max_origins, count_origins(out, entry));

        std::cout << "Origin dedup bench: tm_avx2_r256s_8\n";
        std::cout << "  key:           0x" << std::hex << std::setw(8)
                  << std::setfill('0') << a.key << std::dec << std::setfill(' ') << "\n";
        std::cout << "  data_start:    " << a.data_start << "\n";
        std::cout << "  window:        " << a.window << "\n";
        std::cout << "  repeats:       " << a.repeats << "\n";
        std::cout << "  median_s:      " << std::fixed << std::setprecision(6) << med_s << "\n";
        std::cout << "  rate:          " << std::fixed << std::setprecision(3) << rate << " M/s/thread\n";
        std::cout << "  final_unique:  " << out.pool.size() << "\n";
        std::cout << "  origins:       " << out.origins.size() << "\n";
        std::cout << "  max_origins:   " << max_origins << "\n";
        std::cout << "  parity:        " << (a.verify ? "origins verified" : "not run") << "\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "error: " << e.what() << "\n";
        std::cerr << "usage: state_dedup_origin_bench [--key HEX | --keys FILE] [--data-start N] "
                     "[--window N] [--repeats N] [--threads N] [--limit N] "
                     "[--no-verify] [--schedule all|skip-car]\n";
        return 1;
    }
}
