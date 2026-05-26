// Multi-threaded CPU forward benchmark across all linked TM CPU variants.
// Each thread runs its own implementation instance over a disjoint data slice.
// Reports screen_rate and wall_rate matching test_cuda / tm_opencl_32_8_test conventions.
//
// Select the implementation via --impl <name>:
//   scalar           tm_8 (scalar; gcc -march=native auto-vectorizes to AVX-512)
//   nway             tm_8 N-way interleaved (NWAY_BATCH candidates in lockstep)
//   avx_r128s_8      tm_avx_r128s_8 (AVX1, 128-bit; historically fastest on Intel)
//   avx_r256s_8      tm_avx_r256s_8 (AVX1, 256-bit)
//   avx2_r256s_8     tm_avx2_r256s_8 (AVX2, 256-bit; fastest on Zen 5)
//   avx512_r512s_8   tm_avx512_r512s_8 (AVX-512, 512-bit; viable on Sapphire Rapids+)
//
// Legacy flags --avx2 and --nway are kept as aliases for "--impl avx2_r256s_8"
// and "--impl nway".

#include <atomic>
#include <barrier>
#include <functional>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "data_sizes.h"
#include "rng_obj.h"
#include "key_schedule.h"
#include "tm_base.h"
#include "../cpu/tm_8.h"
#include "../cpu/tm_avx_r128s_8.h"
#include "../cpu/tm_avx_r256s_8.h"
#include "../cpu/tm_avx2_r256s_8.h"
#include "../cpu/tm_avx512_r512s_8.h"

namespace
{
    static const uint8 kMapList[26] = {
        0x00, 0x02, 0x05, 0x04, 0x03, 0x1D, 0x1C, 0x1E, 0x1B,
        0x07, 0x08, 0x06, 0x09, 0x0C, 0x20, 0x21, 0x22, 0x23,
        0x24, 0x25, 0x26, 0x0E, 0x0F, 0x10, 0x12, 0x11
    };

    enum class Impl {
        Scalar,
        ScalarNway,
        AvxR128s,
        AvxR256s,
        Avx2R256s,
        Avx512R512s,
    };

    const char* impl_name(Impl i)
    {
        switch (i) {
            case Impl::Scalar:      return "tm_8 (scalar)";
            case Impl::ScalarNway:  return "tm_8 nway";
            case Impl::AvxR128s:    return "tm_avx_r128s_8";
            case Impl::AvxR256s:    return "tm_avx_r256s_8";
            case Impl::Avx2R256s:   return "tm_avx2_r256s_8";
            case Impl::Avx512R512s: return "tm_avx512_r512s_8";
        }
        return "?";
    }

    Impl impl_from_string(const std::string& s)
    {
        if (s == "scalar")          return Impl::Scalar;
        if (s == "nway")            return Impl::ScalarNway;
        if (s == "avx_r128s_8")     return Impl::AvxR128s;
        if (s == "avx_r256s_8")     return Impl::AvxR256s;
        if (s == "avx2_r256s_8")    return Impl::Avx2R256s;
        if (s == "avx512_r512s_8")  return Impl::Avx512R512s;
        throw std::runtime_error("Unknown --impl value: " + s);
    }

    struct Args
    {
        uint32 key_id        = 0x2CA5B42Du;
        uint32 range_start   = 0u;
        uint32 workunit_size = 1u << 20;
        uint32 warmup_count  = 1u << 18;
        uint32 threads       = 1u;
        Impl   impl          = Impl::Scalar;
    };

    template <typename T>
    T numeric_arg(const char* value, const char* name)
    {
        char* end = nullptr;
        unsigned long parsed = std::strtoul(value, &end, 0);
        if (end == value || *end != '\0')
        {
            std::ostringstream msg;
            msg << "Invalid value for " << name << ": " << value;
            throw std::runtime_error(msg.str());
        }
        return static_cast<T>(parsed);
    }

    Args parse_args(int argc, char** argv)
    {
        Args args;
        for (int i = 1; i < argc; i++)
        {
            std::string a(argv[i]);
            if (a == "--key_id" && i + 1 < argc)
                args.key_id = numeric_arg<uint32>(argv[++i], "--key_id");
            else if (a == "--range_start" && i + 1 < argc)
                args.range_start = numeric_arg<uint32>(argv[++i], "--range_start");
            else if (a == "--workunit_size" && i + 1 < argc)
                args.workunit_size = numeric_arg<uint32>(argv[++i], "--workunit_size");
            else if (a == "--warmup" && i + 1 < argc)
                args.warmup_count = numeric_arg<uint32>(argv[++i], "--warmup");
            else if (a == "--threads" && i + 1 < argc)
                args.threads = numeric_arg<uint32>(argv[++i], "--threads");
            else if (a == "--impl" && i + 1 < argc)
                args.impl = impl_from_string(argv[++i]);
            else if (a == "--avx2")        // legacy alias
                args.impl = Impl::Avx2R256s;
            else if (a == "--nway")        // legacy alias
                args.impl = Impl::ScalarNway;
            else if (a == "--help" || a == "-h")
            {
                std::cout
                    << "bench_cpu\n"
                    << "  --key_id <uint32>\n"
                    << "  --range_start <uint32>\n"
                    << "  --workunit_size <uint32>\n"
                    << "  --warmup <uint32>     (default 1<<18; split across threads)\n"
                    << "  --threads <uint32>    (default 1)\n"
                    << "  --impl <name>         one of:\n"
                    << "                          scalar (default)\n"
                    << "                          nway            (tm_8 N-way BATCH=" << tm_8::NWAY_BATCH << ")\n"
                    << "                          avx_r128s_8\n"
                    << "                          avx_r256s_8\n"
                    << "                          avx2_r256s_8\n"
                    << "                          avx512_r512s_8\n"
                    << "  --avx2                legacy alias for --impl avx2_r256s_8\n"
                    << "  --nway                legacy alias for --impl nway\n";
                std::exit(0);
            }
            else
            {
                throw std::runtime_error(std::string("Unknown argument: ") + a);
            }
        }
        if (args.threads == 0)
            throw std::runtime_error("--threads must be >= 1");
        return args;
    }

    // Inline checksum screening used by the AVX2 path (and optionally by scalar).
    // Mirrors the logic in tm_8::run_bruteforce_data but calls through TM_base virtuals.
    uint64 screen_loop(TM_base& tm, uint32 key_id, uint32 data_start, uint32 count,
                       const key_schedule& schedule)
    {
        uint64 hits = 0;
        uint8 state[128];
        for (uint32 i = 0; i < count; i++)
        {
            tm.expand(key_id, data_start + i);
            tm.run_all_maps(schedule);
            tm.fetch_data(state);

            // XOR-decrypt carnival world in place; check masked checksum
            for (int b = 0; b < 128; b++) state[b] ^= TM_base::carnival_world_data[b];
            uint16 sum = 0;
            for (int b = 0; b < 128; b++) sum += state[b] & TM_base::carnival_world_checksum_mask[b];
            const uint16 carnival_stored = static_cast<uint16>(
                (static_cast<uint16>(state[127 - (CARNIVAL_WORLD_CODE_LENGTH - 1)]) << 8) |
                 static_cast<uint16>(state[127 - (CARNIVAL_WORLD_CODE_LENGTH - 2)]));
            if (sum == carnival_stored) { hits++; continue; }

            // Re-fetch, XOR-decrypt other world, check
            tm.fetch_data(state);
            for (int b = 0; b < 128; b++) state[b] ^= TM_base::other_world_data[b];
            sum = 0;
            for (int b = 0; b < 128; b++) sum += state[b] & TM_base::other_world_checksum_mask[b];
            const uint16 other_stored = static_cast<uint16>(
                (static_cast<uint16>(state[127 - (OTHER_WORLD_CODE_LENGTH - 1)]) << 8) |
                 static_cast<uint16>(state[127 - (OTHER_WORLD_CODE_LENGTH - 2)]));
            if (sum == other_stored) hits++;
        }
        return hits;
    }

    // Per-thread work: owns its own TM instance over a disjoint data slice.
    // Uses a std::barrier to align the start of the timed section across all threads.
    struct ThreadResult
    {
        uint64 hits    = 0;
        double wall_s  = 0.0;
    };

    template <typename TM>
    void simd_worker(TM& tm, uint32 key_id, uint32 warmup_start, uint32 warmup_count,
                     uint32 timed_start, uint32 timed_count,
                     const key_schedule& schedule, std::barrier<>* bar, ThreadResult* out)
    {
        if (warmup_count > 0)
            screen_loop(tm, key_id, warmup_start, warmup_count, schedule);
        bar->arrive_and_wait();
        const auto t0 = std::chrono::high_resolution_clock::now();
        out->hits = screen_loop(tm, key_id, timed_start, timed_count, schedule);
        out->wall_s = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }

    template <typename TM>
    void native_screen_worker(TM& tm, uint32 key_id, uint32 warmup_start, uint32 warmup_count,
                              uint32 timed_start, uint32 timed_count,
                              const key_schedule& schedule, std::barrier<>* bar, ThreadResult* out)
    {
        const uint32 buf_size = std::max(timed_count / 16u, 4096u) * 5u;
        std::vector<uint8> buf(buf_size, 0);
        auto noop = [](double) {};
        if (warmup_count > 0) {
            uint32 sz = 0;
            tm.run_bruteforce_data(key_id, warmup_start, schedule, warmup_count,
                                   noop, buf.data(), buf_size, &sz);
        }
        bar->arrive_and_wait();
        const auto t0 = std::chrono::high_resolution_clock::now();
        uint32 sz = 0;
        tm.run_bruteforce_data(key_id, timed_start, schedule, timed_count,
                               noop, buf.data(), buf_size, &sz);
        out->hits = sz / 5;
        out->wall_s = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }

    void thread_worker(
        uint32 key_id,
        uint32 warmup_start,
        uint32 warmup_count,
        uint32 timed_start,
        uint32 timed_count,
        const key_schedule& schedule,
        RNG* rng,
        Impl impl,
        std::barrier<>* bar,
        ThreadResult* out)
    {
        switch (impl) {
            case Impl::Scalar: {
                tm_8 tm(rng);
                // Scalar path: use run_bruteforce_data which has its own internal screening
                const uint32 buf_size = std::max(timed_count / 16u, 4096u) * 5u;
                std::vector<uint8> buf(buf_size, 0);
                auto noop = [](double) {};
                if (warmup_count > 0) {
                    uint32 sz = 0;
                    tm.run_bruteforce_data(key_id, warmup_start, schedule, warmup_count,
                                           noop, buf.data(), buf_size, &sz);
                }
                bar->arrive_and_wait();
                const auto t0 = std::chrono::high_resolution_clock::now();
                uint32 sz = 0;
                tm.run_bruteforce_data(key_id, timed_start, schedule, timed_count,
                                       noop, buf.data(), buf_size, &sz);
                out->hits   = sz / 5;
                out->wall_s = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t0).count();
                break;
            }
            case Impl::ScalarNway: {
                tm_8 tm(rng);
                const uint32 buf_size = std::max(timed_count / 16u, 4096u) * 5u;
                std::vector<uint8> buf(buf_size, 0);
                auto noop = [](double) {};
                if (warmup_count > 0) {
                    uint32 sz = 0;
                    tm.run_bruteforce_data_nway(key_id, warmup_start, schedule, warmup_count,
                                                noop, buf.data(), buf_size, &sz);
                }
                bar->arrive_and_wait();
                const auto t0 = std::chrono::high_resolution_clock::now();
                uint32 sz = 0;
                tm.run_bruteforce_data_nway(key_id, timed_start, schedule, timed_count,
                                            noop, buf.data(), buf_size, &sz);
                out->hits   = sz / 5;
                out->wall_s = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t0).count();
                break;
            }
            case Impl::AvxR128s: {
                tm_avx_r128s_8 tm(rng);
                native_screen_worker(tm, key_id, warmup_start, warmup_count, timed_start, timed_count, schedule, bar, out);
                break;
            }
            case Impl::AvxR256s: {
                tm_avx_r256s_8 tm(rng);
                native_screen_worker(tm, key_id, warmup_start, warmup_count, timed_start, timed_count, schedule, bar, out);
                break;
            }
            case Impl::Avx2R256s: {
                tm_avx2_r256s_8 tm(rng);
                native_screen_worker(tm, key_id, warmup_start, warmup_count, timed_start, timed_count, schedule, bar, out);
                break;
            }
            case Impl::Avx512R512s: {
                tm_avx512_r512s_8 tm(rng);
                simd_worker(tm, key_id, warmup_start, warmup_count, timed_start, timed_count, schedule, bar, out);
                break;
            }
        }
    }
}

int main(int argc, char** argv)
{
    try
    {
        const Args args = parse_args(argc, argv);

        // Build key schedule (read-only after construction, safe to share)
        std::vector<uint8> map_list(kMapList, kMapList + sizeof(kMapList));
        key_schedule schedule(args.key_id, map_list);

        // Initialise RNG static tables once; tm_8 instances share them
        RNG rng;
        rng.generate_expansion_values_8();
        rng.generate_seed_forward_1();
        rng.generate_seed_forward_128();
        rng.generate_regular_rng_values_8();
        rng.generate_alg0_values_8();
        rng.generate_alg2_values_8_8();
        rng.generate_alg4_values_8();
        rng.generate_alg5_values_8_8();
        rng.generate_alg6_values_8();

        const uint32 T = args.threads;

        // Divide workunit evenly; last thread picks up any remainder
        const uint32 slice      = args.workunit_size / T;
        const uint32 warm_slice = args.warmup_count  / T;

        std::barrier<> bar(static_cast<ptrdiff_t>(T));
        std::vector<ThreadResult> results(T);
        std::vector<std::thread> threads;
        threads.reserve(T);

        for (uint32 t = 0; t < T; t++)
        {
            const uint32 timed_start  = args.range_start + t * slice;
            const uint32 timed_count  = (t + 1 == T)
                                        ? (args.workunit_size - t * slice)
                                        : slice;
            const uint32 warm_start   = args.range_start + t * warm_slice;
            const uint32 warm_count   = (t + 1 == T)
                                        ? (args.warmup_count - t * warm_slice)
                                        : warm_slice;

            threads.emplace_back(thread_worker,
                args.key_id,
                warm_start,  warm_count,
                timed_start, timed_count,
                std::cref(schedule),
                &rng,
                args.impl,
                &bar,
                &results[t]);
        }

        for (auto& th : threads)
            th.join();

        // Wall time = max across threads (all started at the same barrier)
        double wall_s = 0.0;
        uint64 total_hits = 0;
        for (const auto& r : results)
        {
            if (r.wall_s > wall_s) wall_s = r.wall_s;
            total_hits += r.hits;
        }

        const double rate = wall_s > 0.0
            ? static_cast<double>(args.workunit_size) / wall_s
            : 0.0;

        std::cout << "CPU forward benchmark: " << impl_name(args.impl) << "\n";
        std::cout << "  key_id:        0x" << std::hex << std::setw(8)
                  << std::setfill('0') << args.key_id << std::dec << "\n";
        std::cout << "  range_start:   " << args.range_start << "\n";
        std::cout << "  workunit_size: " << args.workunit_size << "\n";
        std::cout << "  threads:       " << T << "\n";
        std::cout << "  wall_s:        " << std::fixed << std::setprecision(3) << wall_s << "\n";
        std::cout << "  screen_rate:   " << std::fixed << std::setprecision(0)
                  << rate << " candidates/s\n";
        std::cout << "  wall_rate:     " << std::fixed << std::setprecision(0)
                  << rate << " candidates/s\n";
        std::cout << "  checksum survivors: " << total_hits << "\n";

        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << "\n";
        return 1;
    }
}
