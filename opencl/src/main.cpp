// Treasure Master — OpenCL forward-search host (32-byte working state × 8 bits/lane)
//
// Compiles and dispatches the kernels in `tm.cl` against an OpenCL device.
// Builds checksum-screen, expansion, stats, and full-process kernels and
// reports survivors plus throughput.  Cross-vendor portable (OpenCL 1.2+).
//
// PRODUCTION ENGINE (2026-06-16): the bounded-wave raceway (cross-vendor; best across
// BOTH throughput and memory). This OpenCL port reaches ~264 M represented/s on an
// RTX 5090 default-precert HM (2026-06-18) and is the recommended path for non-NVIDIA devices. The flat checksum-screen
// and on-GPU compaction kernels are RESEARCH / A-B paths (the screen is also the parity
// reference). For NVIDIA devices prefer the CUDA build (../test_cuda/).
// Watchdog-safe: per-launch candidate counts are scaled by CU count (small iGPUs run the
// same NDRange for seconds and would trip a GPU recovery), see wd_chunk_cands below.
//
// Build:    make                    (see Makefile)
// Invoke:   ./tm_opencl_forward --device <id> --key_id 0x...
//
// Key flags:
//   --map-list all|skip-car   choose ALL_MAPS or SKIP_CAR schedule variant
//   --raceway-wave-cap-mark   PRODUCTION bounded-wave raceway (+ --raceway-cap-bits/-ways,
//                             --raceway-cap-ilp, --raceway-cap-boundaries)
//   --precert / --no-precert  enable/disable default MAP1 certified-shed pre-exclusion
//   --calibrate               (research) measure screen-vs-compaction + write tm_compaction.conf
//   --parity <N>              cross-check N candidates between screen variants
//   --compaction-bench        on-GPU VRAM compaction architecture (research / A-B)
//
// Companion variant: ../tm_opencl_32_16_test (alternate 16-bit lane width).
//
// Sections (within the anonymous namespace below):
//   - Kernel screen-flag bit definitions
//   - Opcode tables (6502 instruction set for machine-code validation)
//   - Data structures (Args, KernelAssets, SurvivorRecord, ValidationSummary)
//   - Argument parsing
//   - OpenCL utilities (error checking, kernel source loading, asset allocation)
//   - Machine-code validation (check_machine_code, hash_state, checksums)
//   - Output formatting (print_summary)

#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "data_sizes.h"
#include "key_schedule.h"
#include "map1_certifier.h"
#include "rng.h"

namespace
{
	static const uint32_t kDefaultBatchSize = 1u << 20;
	static const uint32_t kRacewayDefaultCapIlp = 4u;
	static const uint32_t kRacewayDefaultPersistentGroups = 65536u;
	static const uint8_t kMapList[26] = {
		0x00, 0x02, 0x05, 0x04, 0x03, 0x1D, 0x1C, 0x1E, 0x1B,
		0x07, 0x08, 0x06, 0x09, 0x0C, 0x20, 0x21, 0x22, 0x23,
		0x24, 0x25, 0x26, 0x0E, 0x0F, 0x10, 0x12, 0x11
	};

	#include "opencl_opcodes.h"

	// ── Data structures ──────────────────────────────────────────────────────────
	struct Args
	{
		uint32_t key_id = 0x2CA5B42Du;
		uint32_t range_start = 0u;
		uint32_t workunit_size = 1u << 20;
		uint32_t batch_size = kDefaultBatchSize;
		uint32_t warmup_batches = 1u;
		uint32_t platform_index = 0u;
		uint32_t device_index = 0u;
		std::string output_csv_path;
		bool precert = true;
		bool precert_explicit = false;
		bool use_ilp6 = false;       // Use offset-stream + ILP6 + preids screen kernel
		uint32_t parity_count = 0u;  // Run parity vs baseline screen on N candidates and exit
		uint32_t compaction_count = 0u;  // Run on-GPU compaction pipeline on N candidates vs ilp6 screen
		uint32_t compaction_ilp = 8u;    // span dedup window W = SPAN_ILP (single-warp); build-time -D. W=8 optimal.
		bool compaction_auto_tile = false;  // size the tile from device VRAM / max-alloc
		bool calibrate = false;             // measure host-optimal engine/ilp and write config
		std::string config_path = "tm_compaction.conf";  // calibration config file
		bool raceway_cap_mark = false;      // Run direct offset-stream boundary-cap mark pass and exit
		bool raceway_wave_cap_mark = false; // Run state-saving cap spans with compaction between boundaries
		uint32_t raceway_cap_bits = 0u;     // 0 disables caps for all-alive launch/parity smoke
		uint32_t raceway_cap_ways = 4u;
		uint32_t raceway_cap_count = 1u;
		uint32_t raceway_first_cap_map = 1u;
		std::string raceway_cap_boundaries;
		bool raceway_cap_force_fp32 = false;
		bool raceway_cap_force_fp64 = false;
		uint32_t raceway_bench_repeats = 1u;
		std::string raceway_profile_csv_path;
		bool raceway_persistent = true;
		uint32_t raceway_persistent_groups = 0u;
		uint32_t raceway_cap_ilp = kRacewayDefaultCapIlp;
		bool raceway_subgroup = false;   // register-resident + sub_group_broadcast inner loop (AMD LDS-relief prototype)
		bool raceway_cap_plain_store = false;  // lever #2: benign-race non-atomic cap insert (FN-safe over-keep)
		bool raceway_lds_rng = false;    // LDS-resident RNG staging (x1 collapse; stage regular, derive alg0/alg6)
	};

	struct KernelAssets
	{
		cl_mem regular_rng_values = nullptr;
		cl_mem alg0_values = nullptr;
		cl_mem alg6_values = nullptr;
		cl_mem rng_seed_forward_1 = nullptr;
		cl_mem rng_seed_forward_128 = nullptr;
		cl_mem alg2_values = nullptr;
		cl_mem alg5_values = nullptr;
		cl_mem expansion_values = nullptr;
		cl_mem schedule_data = nullptr;
		cl_mem carnival_data = nullptr;
		// Offset-stream buffers (per-key, ~21.6 MB total). Allocated lazily
		// when --ilp6 path is requested.
		cl_mem offset_regular = nullptr;
		cl_mem offset_alg0 = nullptr;
		cl_mem offset_alg6 = nullptr;
		cl_mem offset_alg2 = nullptr;
		cl_mem offset_alg5 = nullptr;
	};

	struct SurvivorRecord
	{
		uint32_t data = 0;
		uint8_t screen_flags = 0;
		uint8_t machine_flags = 0;
	};

	struct ValidationSummary
	{
		uint64_t total = 0;
		uint64_t carnival = 0;
		uint64_t other = 0;
		uint64_t first_entry_valid = 0;
		uint64_t all_entries_valid = 0;
		uint64_t uses_nop = 0;
		uint64_t uses_unofficial_nops = 0;
		uint64_t uses_illegal = 0;
		uint64_t uses_jam = 0;
	};

	const char* kKernelPaths[] = {
		"./tm.cl",
		"src/tm.cl",                                  // public package layout (binary beside src/)
		"src/bruteforce/tm_opencl_32_8_test/tm.cl",   // dev tree (run from repo root)
		"tm_opencl_32_8_test/tm.cl"
	};

	template <typename T>
	T numeric_arg(const char* value, const char* name)
	{
		char* end = nullptr;
		unsigned long parsed = std::strtoul(value, &end, 0);
		if (end == value || *end != '\0')
		{
			std::ostringstream message;
			message << "Invalid numeric value for " << name << ": " << value;
			throw std::runtime_error(message.str());
		}
		return static_cast<T>(parsed);
	}

	std::vector<uint32_t> parse_u32_list(const std::string& text, const char* name)
	{
		std::vector<uint32_t> out;
		std::stringstream ss(text);
		std::string token;
		while (std::getline(ss, token, ','))
		{
			if (token.empty()) continue;
			char* end = nullptr;
			const unsigned long parsed = std::strtoul(token.c_str(), &end, 0);
			if (end == token.c_str() || *end != '\0')
			{
				std::ostringstream message;
				message << "Invalid numeric value in " << name << ": " << token;
				throw std::runtime_error(message.str());
			}
			out.push_back(static_cast<uint32_t>(parsed));
		}
		if (out.empty())
		{
			throw std::runtime_error(std::string(name) + " must contain at least one value");
		}
		return out;
	}

	std::vector<uint32_t> build_raceway_cap_maps(const Args& args)
	{
		std::vector<uint32_t> maps;
		if (!args.raceway_cap_boundaries.empty())
		{
			maps = parse_u32_list(args.raceway_cap_boundaries, "--raceway-cap-boundaries");
			for (uint32_t& completed_map : maps)
			{
				if (completed_map == 0u || completed_map >= 27u)
				{
					throw std::runtime_error("--raceway-cap-boundaries entries must be completed map numbers in [1,26]");
				}
				completed_map -= 1u;
			}
			std::sort(maps.begin(), maps.end());
			maps.erase(std::unique(maps.begin(), maps.end()), maps.end());
		}
		else
		{
			for (uint32_t i = 0u; i < args.raceway_cap_count; i++)
			{
				maps.push_back(args.raceway_first_cap_map + i);
			}
		}
		if (args.raceway_cap_bits == 0u)
		{
			maps.clear();
		}
		return maps;
	}

	// ── Argument parsing ─────────────────────────────────────────────────────────
	Args parse_args(int argc, char** argv)
	{
		Args args;
		for (int i = 1; i < argc; i++)
		{
			const std::string arg(argv[i]);
			if (arg == "--key_id" && i + 1 < argc)
			{
				args.key_id = numeric_arg<uint32_t>(argv[++i], "--key_id");
			}
			else if (arg == "--range_start" && i + 1 < argc)
			{
				args.range_start = numeric_arg<uint32_t>(argv[++i], "--range_start");
			}
			else if (arg == "--workunit_size" && i + 1 < argc)
			{
				args.workunit_size = numeric_arg<uint32_t>(argv[++i], "--workunit_size");
			}
			else if (arg == "--batch_size" && i + 1 < argc)
			{
				args.batch_size = numeric_arg<uint32_t>(argv[++i], "--batch_size");
			}
			else if (arg == "--warmup_batches" && i + 1 < argc)
			{
				args.warmup_batches = numeric_arg<uint32_t>(argv[++i], "--warmup_batches");
			}
			else if (arg == "--platform" && i + 1 < argc)
			{
				args.platform_index = numeric_arg<uint32_t>(argv[++i], "--platform");
			}
			else if (arg == "--device" && i + 1 < argc)
			{
				args.device_index = numeric_arg<uint32_t>(argv[++i], "--device");
			}
			else if (arg == "--output_csv" && i + 1 < argc)
			{
				args.output_csv_path = argv[++i];
			}
			else if (arg == "--precert")
			{
				args.precert = true;
				args.precert_explicit = true;
			}
			else if (arg == "--no-precert")
			{
				args.precert = false;
				args.precert_explicit = true;
			}
			else if (arg == "--ilp6")
			{
				args.use_ilp6 = true;
			}
			else if (arg == "--parity" && i + 1 < argc)
			{
				args.parity_count = numeric_arg<uint32_t>(argv[++i], "--parity");
			}
			else if (arg == "--compaction" && i + 1 < argc)
			{
				args.compaction_count = numeric_arg<uint32_t>(argv[++i], "--compaction");
			}
			else if (arg == "--compaction-ilp" && i + 1 < argc)
			{
				args.compaction_ilp = numeric_arg<uint32_t>(argv[++i], "--compaction-ilp");
			}
			else if (arg == "--compaction-auto-tile")
			{
				args.compaction_auto_tile = true;
				if (args.compaction_count == 0u) args.compaction_count = 1u;  // placeholder; set from VRAM
			}
			else if (arg == "--calibrate")
			{
				args.calibrate = true;
				if (i + 1 < argc && argv[i + 1][0] != '-') args.config_path = argv[++i];
			}
			else if (arg == "--config" && i + 1 < argc)
			{
				args.config_path = argv[++i];
			}
			else if (arg == "--raceway-cap-mark")
			{
				args.raceway_cap_mark = true;
			}
			else if (arg == "--raceway-wave-cap-mark")
			{
				args.raceway_wave_cap_mark = true;
			}
			else if (arg == "--raceway-cap-bits" && i + 1 < argc)
			{
				args.raceway_cap_bits = numeric_arg<uint32_t>(argv[++i], "--raceway-cap-bits");
			}
			else if (arg == "--raceway-cap-ways" && i + 1 < argc)
			{
				args.raceway_cap_ways = numeric_arg<uint32_t>(argv[++i], "--raceway-cap-ways");
			}
			else if (arg == "--raceway-cap-count" && i + 1 < argc)
			{
				args.raceway_cap_count = numeric_arg<uint32_t>(argv[++i], "--raceway-cap-count");
			}
			else if (arg == "--raceway-first-cap-map" && i + 1 < argc)
			{
				args.raceway_first_cap_map = numeric_arg<uint32_t>(argv[++i], "--raceway-first-cap-map");
			}
			else if (arg == "--raceway-cap-boundaries" && i + 1 < argc)
			{
				args.raceway_cap_boundaries = argv[++i];
			}
			else if (arg == "--raceway-cap-fp32")
			{
				args.raceway_cap_force_fp32 = true;
			}
			else if (arg == "--raceway-cap-fp64")
			{
				args.raceway_cap_force_fp64 = true;
			}
			else if (arg == "--raceway-subgroup")
			{
				args.raceway_subgroup = true;
			}
			else if (arg == "--raceway-cap-plain-store")
			{
				args.raceway_cap_plain_store = true;
			}
			else if (arg == "--raceway-lds-rng")
			{
				args.raceway_lds_rng = true;
			}
			else if (arg == "--raceway-bench-repeats" && i + 1 < argc)
			{
				args.raceway_bench_repeats = numeric_arg<uint32_t>(argv[++i], "--raceway-bench-repeats");
			}
			else if (arg == "--raceway-profile-csv" && i + 1 < argc)
			{
				args.raceway_profile_csv_path = argv[++i];
			}
			else if (arg == "--raceway-static-groups")
			{
				args.raceway_persistent = false;
			}
			else if (arg == "--raceway-persistent-groups" && i + 1 < argc)
			{
				args.raceway_persistent_groups = numeric_arg<uint32_t>(argv[++i], "--raceway-persistent-groups");
			}
			else if (arg == "--raceway-cap-ilp" && i + 1 < argc)
			{
				args.raceway_cap_ilp = numeric_arg<uint32_t>(argv[++i], "--raceway-cap-ilp");
			}
			else if (arg == "--help" || arg == "-h")
			{
				std::cout
					<< "tm_opencl_forward — OpenCL forward-search for Treasure Master Bonus World 2\n"
					<< "\n"
					<< "Sweep options:\n"
					<< "  --key_id <uint32>           Key (4-byte) to fix while sweeping data (default 0x2CA5B42D)\n"
					<< "  --range_start <uint32>      First data index to scan (default 0)\n"
					<< "  --workunit_size <uint32>    Number of data indices to scan (default 2^20)\n"
					<< "  --batch_size <uint32>       Candidates per kernel launch (default 2^20)\n"
					<< "  --warmup_batches <uint32>   Warm-up launches before timing (default 1)\n"
					<< "  --platform <index>          OpenCL platform index (default 0; see `clinfo -l`)\n"
					<< "  --device <index>            Device index within that platform (default 0)\n"
					<< "  --precert / --no-precert    Enable/disable default MAP1 certified-shed pre-exclusion for supported raceway launches\n"
					<< "\n"
					<< "PRODUCTION engine = the bounded-wave raceway (best throughput AND memory, FN-safe;\n"
					<< "~264 M represented/s on RTX 5090 default-precert HM (2026-06-18); recommended for non-NVIDIA GPUs):\n"
					<< "  --raceway-wave-cap-mark     Run the production raceway (state-saving cap spans + wave compaction).\n"
					<< "                              Tune with --raceway-cap-bits/-ways, --raceway-cap-ilp,\n"
					<< "                              --raceway-cap-boundaries.\n"
					<< "\n"
					<< "Screen-kernel selection (RESEARCH / baseline + parity reference, not production):\n"
					<< "  --ilp6                      Offset-stream + ILP6 screen kernel (fastest screen baseline).\n"
					<< "                              Costs ~22 MB extra device memory per key.\n"
					<< "\n"
					<< "Diagnostics:\n"
					<< "  --parity <count>            Compare baseline screen vs --ilp6 screen over <count>\n"
					<< "                              candidates flag-by-flag, then exit. Reports speedup + PASS/FAIL.\n"
					<< "  --raceway-cap-mark          Run OpenCL direct offset-stream boundary-cap mark pass, then exit.\n"
					<< "  --raceway-wave-cap-mark     Run state-saving cap spans with compacted survivors between boundaries.\n"
					<< "  --raceway-cap-bits <bits>   Cap bucket bits; 0 disables caps for all-alive smoke (default 0).\n"
					<< "                              Uses fp64 caps when cl_khr_int64_base_atomics is available.\n"
					<< "  --raceway-cap-ways <ways>   Cap ways per bucket (default 4).\n"
					<< "  --raceway-cap-count <count> Consecutive completed-map boundaries to probe (default 1).\n"
					<< "  --raceway-first-cap-map <m> First completed map index to probe; MAP2 is 1 (default 1).\n"
					<< "  --raceway-cap-boundaries <L> CUDA-style completed-map list, e.g. 2,4,8,14,20.\n"
					<< "  --raceway-cap-fp32          Force portable fp32 cap even if fp64 atomics are available.\n"
					<< "  --raceway-cap-fp64          Require fp64 cap; errors if the device lacks 64-bit atomics.\n"
					<< "  --raceway-subgroup          Register-resident inner loop via sub_group_broadcast (AMD LDS-relief prototype).\n"
					<< "  --raceway-cap-plain-store   Benign-race non-atomic cap insert (FN-safe over-keep; drops the global atomic).\n"
					<< "  --raceway-lds-rng           LDS-resident RNG staging: stage regular rows in LDS, derive alg0/alg6 (use with --raceway-cap-ilp 16..64).\n"
					<< "  --raceway-bench-repeats <n> Timed cap-mark repeats with fresh cap table each run (default 1).\n"
					<< "  --raceway-profile-csv <p>   Append raceway cap-mark timing/counter row to CSV path.\n"
					<< "  --raceway-static-groups     Disable persistent queue; launch one work-group per ILP span.\n"
					<< "  --raceway-persistent-groups <n> Persistent queue work-groups (default min(spans, 65536)).\n"
					<< "  --raceway-cap-ilp <n>       Candidates per 32-lane work-group for cap mark (default 4).\n"
					<< "  --output_csv <path>         Optional CSV output of validated survivors\n";
				std::exit(0);
			}
			else
			{
				std::ostringstream message;
				message << "Unknown argument: " << arg;
				throw std::runtime_error(message.str());
			}
		}

		if (args.batch_size == 0u)
		{
			throw std::runtime_error("--batch_size must be greater than zero");
		}
		if (args.workunit_size == 0u)
		{
			throw std::runtime_error("--workunit_size must be greater than zero");
		}
		if (args.raceway_cap_ways == 0u)
		{
			throw std::runtime_error("--raceway-cap-ways must be greater than zero");
		}
		if (args.raceway_cap_bits > 32u)
		{
			throw std::runtime_error("--raceway-cap-bits must be <= 32");
		}
		if (args.raceway_first_cap_map >= 27u)
		{
			throw std::runtime_error("--raceway-first-cap-map must be < 27");
		}
		if (args.raceway_cap_boundaries.empty() && args.raceway_cap_count > 27u - args.raceway_first_cap_map)
		{
			throw std::runtime_error("--raceway-cap-count extends past the 27-map schedule");
		}
		if (args.raceway_cap_force_fp32 && args.raceway_cap_force_fp64)
		{
			throw std::runtime_error("--raceway-cap-fp32 and --raceway-cap-fp64 are mutually exclusive");
		}
		if (args.raceway_bench_repeats == 0u)
		{
			throw std::runtime_error("--raceway-bench-repeats must be greater than zero");
		}
		// The LDS-staging path amortizes the per-map RNG stage over the candidates
		// resident in a work-group, so it needs ILP well above the global-read path's
		// MLP-saturated [1,8] (break-even ~9; sweep 16/32/64). LDS budget is checked at
		// build time by the kernel's __local arrays vs the 64 KB limit.
		const uint32_t max_cap_ilp = args.raceway_lds_rng ? 64u : 8u;
		if (args.raceway_cap_ilp == 0u || args.raceway_cap_ilp > max_cap_ilp)
		{
			throw std::runtime_error(args.raceway_lds_rng
				? "--raceway-cap-ilp must be in [1,64] with --raceway-lds-rng"
				: "--raceway-cap-ilp must be in [1,8]");
		}

		return args;
	}

	// ── OpenCL utilities ─────────────────────────────────────────────────────────
	void throw_if_error(cl_int status, const char* what)
	{
		if (status == CL_SUCCESS)
		{
			return;
		}

		std::ostringstream message;
		message << what << " failed with OpenCL status " << status;
		throw std::runtime_error(message.str());
	}

	std::string get_platform_string(cl_platform_id platform, cl_platform_info param)
	{
		size_t size = 0;
		throw_if_error(clGetPlatformInfo(platform, param, 0, nullptr, &size), "clGetPlatformInfo(size)");
		std::string value(size, '\0');
		throw_if_error(clGetPlatformInfo(platform, param, size, &value[0], nullptr), "clGetPlatformInfo(data)");
		if (!value.empty() && value[value.size() - 1] == '\0')
		{
			value.resize(value.size() - 1);
		}
		return value;
	}

	std::string get_device_string(cl_device_id device, cl_device_info param)
	{
		size_t size = 0;
		throw_if_error(clGetDeviceInfo(device, param, 0, nullptr, &size), "clGetDeviceInfo(size)");
		std::string value(size, '\0');
		throw_if_error(clGetDeviceInfo(device, param, size, &value[0], nullptr), "clGetDeviceInfo(data)");
		if (!value.empty() && value[value.size() - 1] == '\0')
		{
			value.resize(value.size() - 1);
		}
		return value;
	}

	bool has_opencl_extension(const std::string& extensions, const std::string& needle)
	{
		std::size_t pos = extensions.find(needle);
		while (pos != std::string::npos)
		{
			const bool left_ok = (pos == 0) || extensions[pos - 1] == ' ';
			const std::size_t end = pos + needle.size();
			const bool right_ok = (end == extensions.size()) || extensions[end] == ' ';
			if (left_ok && right_ok) return true;
			pos = extensions.find(needle, pos + 1);
		}
		return false;
	}

	std::string load_kernel_source()
	{
		for (std::size_t i = 0; i < sizeof(kKernelPaths) / sizeof(kKernelPaths[0]); i++)
		{
			std::ifstream input(kKernelPaths[i], std::ios::binary);
			if (!input)
			{
				continue;
			}

			std::ostringstream buffer;
			buffer << input.rdbuf();
			return buffer.str();
		}

		throw std::runtime_error("Could not locate tm.cl for tm_opencl_32_8_test");
	}

	std::vector<uint8_t> build_schedule_blob(uint32_t key)
	{
		key_schedule_data schedule_data = {};
		schedule_data.as_uint8[0] = static_cast<uint8_t>((key >> 24) & 0xFF);
		schedule_data.as_uint8[1] = static_cast<uint8_t>((key >> 16) & 0xFF);
		schedule_data.as_uint8[2] = static_cast<uint8_t>((key >> 8) & 0xFF);
		schedule_data.as_uint8[3] = static_cast<uint8_t>(key & 0xFF);

		std::vector<uint8_t> blob(27 * 4, 0);
		int schedule_count = 0;
		for (std::size_t i = 0; i < sizeof(kMapList) / sizeof(kMapList[0]); i++)
		{
			key_schedule_entry entry = generate_schedule_entry(kMapList[i], &schedule_data);
			blob[schedule_count * 4 + 0] = entry.rng1;
			blob[schedule_count * 4 + 1] = entry.rng2;
			blob[schedule_count * 4 + 2] = static_cast<uint8_t>((entry.nibble_selector >> 8) & 0xFF);
			blob[schedule_count * 4 + 3] = static_cast<uint8_t>(entry.nibble_selector & 0xFF);
			schedule_count++;

			if (kMapList[i] == 0x22)
			{
				entry = generate_schedule_entry(kMapList[i], &schedule_data, 4);
				blob[schedule_count * 4 + 0] = entry.rng1;
				blob[schedule_count * 4 + 1] = entry.rng2;
				blob[schedule_count * 4 + 2] = static_cast<uint8_t>((entry.nibble_selector >> 8) & 0xFF);
				blob[schedule_count * 4 + 3] = static_cast<uint8_t>(entry.nibble_selector & 0xFF);
				schedule_count++;
			}
		}

		return blob;
	}

	uint32_t popcount32(uint32_t value)
	{
		uint32_t count = 0u;
		while (value != 0u)
		{
			value &= value - 1u;
			++count;
		}
		return count;
	}

	struct PrecertPlan
	{
		uint32_t shed_mask = 0u;
		uint32_t support_mask = 0xFFFFFFFFu;
		uint32_t fixed_value = 0u;
		uint32_t bits = 0u;
		uint64_t source_mult = 1ull;
		uint32_t logical_window = 0u;
		bool active = false;
	};

	PrecertPlan make_precert_plan(const Args& args, const char* path_name)
	{
		PrecertPlan plan;
		plan.logical_window = args.workunit_size;
		if (!args.precert)
		{
			return plan;
		}

		plan.shed_mask = tm_map1_certifier::certified_shed_mask_from_schedule_blob(
			args.key_id, build_schedule_blob(args.key_id));
		plan.support_mask = ~plan.shed_mask;
		plan.bits = popcount32(plan.shed_mask);
		plan.source_mult = 1ull << plan.bits;
		if (plan.bits == 0u)
		{
			return plan;
		}

		const bool compatible =
			args.range_start == 0u &&
			(static_cast<uint64_t>(args.workunit_size) % plan.source_mult) == 0ull;
		if (!compatible)
		{
			if (args.precert_explicit)
			{
				if (args.range_start != 0u)
					throw std::runtime_error(std::string("--precert ") + path_name + " currently requires --range_start 0");
				throw std::runtime_error(std::string("--precert ") + path_name + " requires --workunit_size divisible by 2^certified_bits");
			}
			return plan;
		}

		const uint64_t logical = static_cast<uint64_t>(args.workunit_size) / plan.source_mult;
		if (logical == 0ull || logical > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
		{
			throw std::runtime_error(std::string("--precert ") + path_name + " logical support-axis window is out of range");
		}
		plan.logical_window = static_cast<uint32_t>(logical);
		plan.active = true;
		return plan;
	}

	void print_precert_plan(const PrecertPlan& plan, const Args& args)
	{
		if (!args.precert || (!plan.active && !args.precert_explicit))
		{
			return;
		}
		std::cout << "  precert: shed_bits=" << plan.bits
		          << " shed_mask=0x" << std::hex << std::setw(8) << std::setfill('0') << plan.shed_mask
		          << " support_mask=0x" << std::setw(8) << plan.support_mask
		          << std::dec << std::setfill(' ') << "\n";
		if (plan.active)
		{
			std::cout << "  precert-active: represented_window=" << args.workunit_size
			          << " logical_window=" << plan.logical_window
			          << " source_mult=" << plan.source_mult << "\n";
		}
	}

	cl_mem create_read_only_buffer(cl_context context, cl_command_queue queue, const void* data, std::size_t size)
	{
		cl_int status = CL_SUCCESS;
		cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, size, nullptr, &status);
		throw_if_error(status, "clCreateBuffer(read_only)");
		throw_if_error(clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, size, data, 0, nullptr, nullptr), "clEnqueueWriteBuffer");
		return buffer;
	}

	// Builds the per-key offset-stream buffers used by tm_checksum_screen_offset_ilp6.
	// Layout matches the CUDA port (src/bruteforce/test_cuda/main.cpp::build_offset_stream_blob):
	//   per schedule step m (0..entry_count-1):
	//     for pos = 0..2047:
	//       regular_stream[m*2048+pos][0..127]  = RNG::regular_rng_values_8[seed*128 + 0..127]
	//       alg0_stream   [m*2048+pos][0..127]  = RNG::alg0_values_8       [seed*128 + 0..127]
	//       alg6_stream   [m*2048+pos][0..127]  = RNG::alg6_values_8       [seed*128 + 0..127]
	//       alg2_stream   [m*2048+pos]          = RNG::alg2_values_32_8    [seed]
	//       alg5_stream   [m*2048+pos]          = RNG::alg5_values_32_8    [seed]
	//       seed = rng_table[seed]
	// Total per key (27 entries): 3 * 27 * 2048 * 128 + 2 * 27 * 2048 * 4 = 21,676,032 bytes.
	void build_offset_assets(cl_context context, cl_command_queue queue, uint32_t key, KernelAssets& assets,
		const std::vector<uint16_t>& rng_table,
		const std::vector<uint8_t>& regular_rng_values_8,
		const std::vector<uint8_t>& alg0_values_8,
		const std::vector<uint8_t>& alg6_values_8,
		const std::vector<uint32_t>& alg2_values_32_8,
		const std::vector<uint32_t>& alg5_values_32_8,
		const std::vector<uint8_t>& schedule_blob)
	{
		const std::size_t entry_count = schedule_blob.size() / 4;
		const std::size_t stream_bytes = entry_count * 2048ULL * 128ULL;
		const std::size_t carry_count  = entry_count * 2048ULL;
		std::vector<uint8_t>  regular_stream(stream_bytes);
		std::vector<uint8_t>  alg0_stream(stream_bytes);
		std::vector<uint8_t>  alg6_stream(stream_bytes);
		std::vector<uint32_t> alg2_stream(carry_count);
		std::vector<uint32_t> alg5_stream(carry_count);

		for (std::size_t m = 0; m < entry_count; ++m)
		{
			uint16_t seed = (static_cast<uint16_t>(schedule_blob[m * 4 + 0]) << 8)
			              |  static_cast<uint16_t>(schedule_blob[m * 4 + 1]);
			for (std::size_t pos = 0; pos < 2048ULL; ++pos)
			{
				const std::size_t stream_offset = (m * 2048ULL + pos) * 128ULL;
				std::memcpy(regular_stream.data() + stream_offset, regular_rng_values_8.data() + static_cast<std::size_t>(seed) * 128ULL, 128ULL);
				std::memcpy(alg0_stream.data()    + stream_offset, alg0_values_8.data()        + static_cast<std::size_t>(seed) * 128ULL, 128ULL);
				std::memcpy(alg6_stream.data()    + stream_offset, alg6_values_8.data()        + static_cast<std::size_t>(seed) * 128ULL, 128ULL);
				const std::size_t carry_offset = m * 2048ULL + pos;
				alg2_stream[carry_offset] = alg2_values_32_8[seed];
				alg5_stream[carry_offset] = alg5_values_32_8[seed];
				seed = rng_table[seed];
			}
		}

		assets.offset_regular = create_read_only_buffer(context, queue, regular_stream.data(), regular_stream.size());
		assets.offset_alg0    = create_read_only_buffer(context, queue, alg0_stream.data(),    alg0_stream.size());
		assets.offset_alg6    = create_read_only_buffer(context, queue, alg6_stream.data(),    alg6_stream.size());
		assets.offset_alg2    = create_read_only_buffer(context, queue, alg2_stream.data(),    alg2_stream.size() * sizeof(uint32_t));
		assets.offset_alg5    = create_read_only_buffer(context, queue, alg5_stream.data(),    alg5_stream.size() * sizeof(uint32_t));

		(void)key;  // key is implied through schedule_blob
	}

	KernelAssets build_kernel_assets(cl_context context, cl_command_queue queue, uint32_t key, bool build_offset_streams)
	{
		KernelAssets assets;

		std::vector<uint16_t> rng_table(256 * 256);
		generate_rng_table(rng_table.data());

		std::vector<uint8_t> regular_rng_values(0x10000 * 128);
		generate_regular_rng_values_8(regular_rng_values.data(), rng_table.data());
		assets.regular_rng_values = create_read_only_buffer(context, queue, regular_rng_values.data(), regular_rng_values.size());

		std::vector<uint8_t> alg0_values(0x10000 * 128);
		generate_alg0_values_8(alg0_values.data(), rng_table.data());
		assets.alg0_values = create_read_only_buffer(context, queue, alg0_values.data(), alg0_values.size());

		std::vector<uint8_t> alg6_values(0x10000 * 128);
		generate_alg6_values_8(alg6_values.data(), rng_table.data());
		assets.alg6_values = create_read_only_buffer(context, queue, alg6_values.data(), alg6_values.size());

		std::vector<uint16_t> rng_seed_forward_1(256 * 256);
		generate_seed_forward_1(rng_seed_forward_1.data(), rng_table.data());
		assets.rng_seed_forward_1 = create_read_only_buffer(context, queue, rng_seed_forward_1.data(), rng_seed_forward_1.size() * sizeof(uint16_t));

		std::vector<uint16_t> rng_seed_forward_128(256 * 256);
		generate_seed_forward_128(rng_seed_forward_128.data(), rng_table.data());
		assets.rng_seed_forward_128 = create_read_only_buffer(context, queue, rng_seed_forward_128.data(), rng_seed_forward_128.size() * sizeof(uint16_t));

		std::vector<uint32_t> alg2_values(0x10000);
		generate_alg2_values_32_8(alg2_values.data(), rng_table.data());
		assets.alg2_values = create_read_only_buffer(context, queue, alg2_values.data(), alg2_values.size() * sizeof(uint32_t));

		std::vector<uint32_t> alg5_values(0x10000);
		generate_alg5_values_32_8(alg5_values.data(), rng_table.data());
		assets.alg5_values = create_read_only_buffer(context, queue, alg5_values.data(), alg5_values.size() * sizeof(uint32_t));

		std::vector<uint8_t> expansion_values(0x10000 * 128);
		generate_expansion_values_8(expansion_values.data(), rng_table.data());
		assets.expansion_values = create_read_only_buffer(context, queue, expansion_values.data(), expansion_values.size());

		std::vector<uint8_t> schedule_blob = build_schedule_blob(key);
		assets.schedule_data = create_read_only_buffer(context, queue, schedule_blob.data(), schedule_blob.size());

		if (build_offset_streams)
		{
			build_offset_assets(context, queue, key, assets,
				rng_table, regular_rng_values, alg0_values, alg6_values,
				alg2_values, alg5_values, schedule_blob);
		}

		static const unsigned char carnival_data[128] = {
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x3D, 0x5E, 0xA1, 0xA6, 0xC8, 0x23,
			0xD7, 0x6E, 0x3F, 0x7C, 0xD2, 0x46, 0x1B, 0x9F, 0xAB, 0xD2,
			0x5C, 0x9B, 0x32, 0x43, 0x67, 0x30, 0xA0, 0xA4, 0x23, 0xF3,
			0x27, 0xBF, 0xEA, 0x21, 0x0F, 0x13, 0x31, 0x1A, 0x15, 0xA1,
			0x39, 0x34, 0xE4, 0xD2, 0x52, 0x6E, 0xA6, 0xF7, 0xF6, 0x43,
			0xD1, 0x28, 0x41, 0xD8, 0xDC, 0x55, 0xE1, 0xC5, 0x49, 0xF5,
			0xD4, 0x84, 0x52, 0x1F, 0x90, 0xAB, 0x26, 0xE4, 0x2A, 0xC3,
			0xC2, 0x59, 0xAC, 0x81, 0x58, 0x35, 0x7A, 0xC3, 0x51, 0x9A,
			0x01, 0x04, 0xF5, 0xE2, 0xFB, 0xA7, 0xAE, 0x8B, 0x46, 0x9A,
			0x27, 0x41, 0xFA, 0xDD, 0x63, 0x72, 0x23, 0x7E, 0x1B, 0x44,
			0x5A, 0x0B, 0x2A, 0x3C, 0x09, 0xFA, 0xA3, 0x59, 0x3C, 0xA1,
			0xF0, 0x90, 0x4F, 0x46, 0x9E, 0xD1, 0xD7, 0xF4
		};
		assets.carnival_data = create_read_only_buffer(context, queue, carnival_data, sizeof(carnival_data));

		return assets;
	}

	void release_assets(KernelAssets& assets)
	{
		clReleaseMemObject(assets.regular_rng_values);
		clReleaseMemObject(assets.alg0_values);
		clReleaseMemObject(assets.alg6_values);
		clReleaseMemObject(assets.rng_seed_forward_1);
		clReleaseMemObject(assets.rng_seed_forward_128);
		clReleaseMemObject(assets.alg2_values);
		clReleaseMemObject(assets.alg5_values);
		clReleaseMemObject(assets.expansion_values);
		clReleaseMemObject(assets.schedule_data);
		if (assets.offset_regular) clReleaseMemObject(assets.offset_regular);
		if (assets.offset_alg0)    clReleaseMemObject(assets.offset_alg0);
		if (assets.offset_alg6)    clReleaseMemObject(assets.offset_alg6);
		if (assets.offset_alg2)    clReleaseMemObject(assets.offset_alg2);
		if (assets.offset_alg5)    clReleaseMemObject(assets.offset_alg5);
		clReleaseMemObject(assets.carnival_data);
	}

	#include "opencl_validate.h"

	// ── Output formatting ────────────────────────────────────────────────────────
	void print_summary(const Args& args,
		const std::string& platform_name,
		const std::string& device_name,
		double setup_seconds,
		double warmup_seconds,
		double screen_kernel_seconds,
		double materialize_kernel_seconds,
		double validation_seconds,
		double wall_seconds,
		uint64_t carnival_hits,
		uint64_t other_hits,
		const ValidationSummary& validation_summary)
	{
		const uint64_t total_hits = carnival_hits + other_hits;
		const double candidate_count = static_cast<double>(args.workunit_size);
		const double screen_rate = screen_kernel_seconds == 0.0 ? 0.0 : candidate_count / screen_kernel_seconds;
		const double materialize_rate = materialize_kernel_seconds == 0.0 ? 0.0 : static_cast<double>(total_hits) / materialize_kernel_seconds;
		const double wall_rate = wall_seconds == 0.0 ? 0.0 : candidate_count / wall_seconds;

		std::cout << "GPU checksum-screen benchmark\n";
		std::cout << "  platform: " << platform_name << "\n";
		std::cout << "  device: " << device_name << "\n";
		std::cout << "  key_id: " << args.key_id << "\n";
		std::cout << "  range_start: " << args.range_start << "\n";
		std::cout << "  workunit_size: " << args.workunit_size << "\n";
		std::cout << "  batch_size: " << args.batch_size << "\n";
		std::cout << "  setup_s: " << std::fixed << std::setprecision(3) << setup_seconds << "\n";
		std::cout << "  warmup_s: " << std::fixed << std::setprecision(3) << warmup_seconds << "\n";
		std::cout << "  screen_kernel_s: " << std::fixed << std::setprecision(3) << screen_kernel_seconds << "\n";
		std::cout << "  materialize_kernel_s: " << std::fixed << std::setprecision(3) << materialize_kernel_seconds << "\n";
		std::cout << "  cpu_validation_s: " << std::fixed << std::setprecision(3) << validation_seconds << "\n";
		std::cout << "  wall_s: " << std::fixed << std::setprecision(3) << wall_seconds << "\n";
		std::cout << "  screen_rate: " << std::fixed << std::setprecision(0) << screen_rate << " candidates/s\n";
		if (total_hits > 0)
		{
			std::cout << "  materialize_rate: " << std::fixed << std::setprecision(0) << materialize_rate << " survivors/s\n";
		}
		std::cout << "  wall_rate: " << std::fixed << std::setprecision(0) << wall_rate << " candidates/s\n";
		std::cout << "  checksum survivors: " << total_hits << "\n";
		std::cout << "  checksum survivors carnival: " << carnival_hits << "\n";
		std::cout << "  checksum survivors other: " << other_hits << "\n";
		std::cout << "  validated survivors: " << validation_summary.total << "\n";
		std::cout << "  validated first_entry_valid: " << validation_summary.first_entry_valid << "\n";
		std::cout << "  validated all_entries_valid: " << validation_summary.all_entries_valid << "\n";
		std::cout << "  validated uses_nop: " << validation_summary.uses_nop << "\n";
		std::cout << "  validated uses_unofficial_nops: " << validation_summary.uses_unofficial_nops << "\n";
		std::cout << "  validated uses_illegal: " << validation_summary.uses_illegal << "\n";
		std::cout << "  validated uses_jam: " << validation_summary.uses_jam << "\n";
		if (!args.output_csv_path.empty())
		{
			std::cout << "  output_csv: " << args.output_csv_path << "\n";
		}
	}
}

// ── Entry point ──────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
	try
	{
		const Args args = parse_args(argc, argv);

		const auto setup_start = std::chrono::high_resolution_clock::now();

		cl_uint platform_count = 0;
		throw_if_error(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
		if (platform_count == 0)
		{
			throw std::runtime_error("No OpenCL platforms found");
		}

		std::vector<cl_platform_id> platforms(platform_count);
		throw_if_error(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs(data)");
		if (args.platform_index >= platform_count)
		{
			throw std::runtime_error("Requested --platform is out of range");
		}

		const cl_platform_id platform = platforms[args.platform_index];
		const std::string platform_name = get_platform_string(platform, CL_PLATFORM_NAME);

		cl_uint device_count = 0;
		throw_if_error(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count), "clGetDeviceIDs(count)");
		if (device_count == 0)
		{
			throw std::runtime_error("No GPU devices found on the selected platform");
		}

		std::vector<cl_device_id> devices(device_count);
		throw_if_error(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr), "clGetDeviceIDs(data)");
		if (args.device_index >= device_count)
		{
			throw std::runtime_error("Requested --device is out of range");
		}

		const cl_device_id device = devices[args.device_index];
		const std::string device_name = get_device_string(device, CL_DEVICE_NAME);
		const std::string device_extensions = get_device_string(device, CL_DEVICE_EXTENSIONS);
		const bool device_has_int64_atomics = has_opencl_extension(device_extensions, "cl_khr_int64_base_atomics");
		const bool raceway_mode = args.raceway_cap_mark || args.raceway_wave_cap_mark;
		if (raceway_mode && args.raceway_cap_force_fp64 && !device_has_int64_atomics)
		{
			throw std::runtime_error("--raceway-cap-fp64 requires cl_khr_int64_base_atomics");
		}
		const bool raceway_cap_fp64 = args.raceway_cap_force_fp64
			|| (!args.raceway_cap_force_fp32 && device_has_int64_atomics);
		if (raceway_mode && !raceway_cap_fp64 && args.raceway_cap_bits > 30u)
		{
			throw std::runtime_error("--raceway-cap-bits > 30 requires cl_khr_int64_base_atomics for the fp64 cap path");
		}

		// Device characteristics that govern the compaction-vs-screen tradeoff.
		// The key one is L2 size: the per-key offset streams are ~21.6 MB, so a
		// device whose L2 (CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) is below that thrashes,
		// pushing both kernels DRAM-bound and eroding compaction's extra-gather margin.
		cl_ulong dev_global_mem = 0, dev_l2 = 0, dev_max_alloc = 0;
		cl_uint  dev_cus = 0;
		clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,       sizeof(dev_global_mem), &dev_global_mem, nullptr);
		clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(dev_l2),         &dev_l2,         nullptr);
		clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,    sizeof(dev_max_alloc),  &dev_max_alloc,  nullptr);
		clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,     sizeof(dev_cus),        &dev_cus,        nullptr);
		// Watchdog-safe sub-launch budget. A big dGPU swallows multi-million-candidate
		// NDRanges in well under the GPU compute watchdog, but a 2-CU iGPU runs the same
		// launch for multiple seconds and trips a hard GPU recovery. Scale the per-launch
		// candidate count by CU count (~64K cand/CU), clamped to [256K, 8M]; both the
		// chunked screen and the calibration span size their sub-launches from this.
		size_t wd_chunk_cands = (size_t)dev_cus * (64u * 1024u);
		if (wd_chunk_cands < (256u * 1024u)) wd_chunk_cands = 256u * 1024u;
		if (wd_chunk_cands > (8u * 1024u * 1024u)) wd_chunk_cands = 8u * 1024u * 1024u;
		const double OFFSET_STREAM_MB = 21.6;  // per-key offset-stream working set
		const bool l2_fits_streams = (double)dev_l2 / (1024.0 * 1024.0) >= OFFSET_STREAM_MB;
		std::cout << "Device: " << device_name << "  CUs=" << dev_cus
		          << "  VRAM=" << (dev_global_mem >> 20) << " MB"
		          << "  L2=" << (dev_l2 >> 20) << " MB"
		          << "  maxAlloc=" << (dev_max_alloc >> 20) << " MB\n";
		// NOTE: CL_DEVICE_GLOBAL_MEM_CACHE_SIZE is unreliable on some drivers (NVIDIA
		// reports ~5 MB even for large-L2 Blackwell), so this is a rough hint only —
		// the authoritative engine choice is an empirical calibration run.
		std::cout << "  engine hint (heuristic, verify by calibration): "
		          << (l2_fits_streams ? "compaction" : "flat screen — but reported L2 may be under-stated; measure")
		          << "\n";
		if (raceway_mode)
		{
			std::cout << "  raceway cap mode: "
			          << (raceway_cap_fp64 ? "fp64 (cl_khr_int64_base_atomics)" : "fp32 fallback")
			          << (args.raceway_subgroup ? "  [inner loop: sub_group_broadcast, register-resident]" : "")
			          << (args.raceway_cap_plain_store ? "  [cap insert: benign-race plain store]" : "")
			          << "\n";
		}
		// Look up a calibrated config for this exact device (fingerprint-keyed).
		{
			std::ostringstream fpk;
			fpk << "opencl|" << device_name << "|" << (dev_global_mem >> 20) << "MB|" << dev_cus << "CU";
			std::ifstream cfg(args.config_path);
			std::string l; bool found = false;
			while (std::getline(cfg, l))
			{
				if (l.empty() || l[0] == '#') continue;
				if (l.substr(0, l.find('\t')) == fpk.str())
				{
					std::cout << "  calibrated config: " << l.substr(l.find('\t') + 1)
					          << "  (from " << args.config_path << ")\n";
					found = true; break;
				}
			}
			if (!found)
				std::cout << "  no calibrated config for this device — run --calibrate (defaults to flat screen until then)\n";
		}

		cl_int status = CL_SUCCESS;
		cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
		throw_if_error(status, "clCreateContext");

		cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
		throw_if_error(status, "clCreateCommandQueue");

		const std::string kernel_source = load_kernel_source();
		const char* source_ptr = kernel_source.c_str();
		const size_t source_size = kernel_source.size();
		cl_program program = clCreateProgramWithSource(context, 1, &source_ptr, &source_size, &status);
		throw_if_error(status, "clCreateProgramWithSource");

		// -cl-std=CL1.2 pins the OpenCL C language version for cross-vendor determinism
		// (matches CL_TARGET_OPENCL_VERSION=120). No FP flags — this is an integer/bitwise
		// kernel, so -cl-fast-relaxed-math / -cl-mad-enable are no-ops; optimization is on
		// by default (no -cl-opt-disable).
		// The subgroup variant needs the OpenCL C 2.0 subgroup builtins
		// (sub_group_broadcast); the cl_khr_subgroups pragma alone does not expose
		// them under CL1.2 on the AMD comgr front-end. The rest of the source is
		// CL1.2-compatible and compiles unchanged under CL2.0.
		const std::string cl_std = args.raceway_subgroup ? "-cl-std=CL2.0" : "-cl-std=CL1.2";
		const std::string build_opts = cl_std + " -DSPAN_ILP=" + std::to_string(args.compaction_ilp)
			+ " -DRACEWAY_CAP_ILP=" + std::to_string(args.raceway_cap_ilp)
			+ (raceway_cap_fp64 ? " -DRACEWAY_CAP_FP64=1" : "")
			+ (args.raceway_subgroup ? " -DRACEWAY_SUBGROUP=1" : "")
			+ (args.raceway_cap_plain_store ? " -DRACEWAY_CAP_NONATOMIC=1" : "")
			+ (args.raceway_lds_rng ? " -DRACEWAY_LDS_RNG=1 -DSTAGE_ROWS=160" : "");
		status = clBuildProgram(program, 1, &device, build_opts.c_str(), nullptr, nullptr);
		if (status != CL_SUCCESS)
		{
			size_t log_size = 0;
			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
			std::string log(log_size, '\0');
			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
			throw std::runtime_error("clBuildProgram failed:\n" + log);
		}

		cl_kernel screen_kernel = clCreateKernel(program, "tm_checksum_screen", &status);
		throw_if_error(status, "clCreateKernel(tm_checksum_screen)");
		cl_kernel materialize_kernel = clCreateKernel(program, "tm_materialize_survivors", &status);
		throw_if_error(status, "clCreateKernel(tm_materialize_survivors)");
		cl_kernel screen_ilp6_kernel = clCreateKernel(program, "tm_checksum_screen_offset_ilp6", &status);
		throw_if_error(status, "clCreateKernel(tm_checksum_screen_offset_ilp6)");
		cl_kernel span_kernel = nullptr, compact_kernel = nullptr, resolve_kernel = nullptr;
		cl_kernel raceway_cap_mark_kernel = nullptr;
		cl_kernel raceway_wave_first_kernel = nullptr;
		cl_kernel raceway_wave_span_kernel = nullptr;
		if (args.compaction_count > 0u || args.calibrate || args.raceway_wave_cap_mark)
		{
			span_kernel    = clCreateKernel(program, "run_span_dedup", &status);            throw_if_error(status, "clCreateKernel(run_span_dedup)");
			compact_kernel = clCreateKernel(program, "compact_survivors_ordered", &status); throw_if_error(status, "clCreateKernel(compact_survivors_ordered)");
			resolve_kernel = clCreateKernel(program, "resolve_flags", &status);             throw_if_error(status, "clCreateKernel(resolve_flags)");
		}
		if (args.raceway_cap_mark)
		{
			raceway_cap_mark_kernel = clCreateKernel(program, "raceway_boundary_cap_mark_offset", &status);
			throw_if_error(status, "clCreateKernel(raceway_boundary_cap_mark_offset)");
		}
		if (args.raceway_wave_cap_mark)
		{
			raceway_wave_first_kernel = clCreateKernel(program, "raceway_boundary_cap_state_offset", &status);
			throw_if_error(status, "clCreateKernel(raceway_boundary_cap_state_offset)");
			raceway_wave_span_kernel = clCreateKernel(program, "raceway_span_state_liveidx_cap_offset", &status);
			throw_if_error(status, "clCreateKernel(raceway_span_state_liveidx_cap_offset)");
		}

		const bool need_offset_streams = args.use_ilp6 || args.parity_count > 0u || args.compaction_count > 0u || args.calibrate || args.raceway_cap_mark || args.raceway_wave_cap_mark;
		KernelAssets assets = build_kernel_assets(context, queue, args.key_id, need_offset_streams);

		// The ilp6 screen kernel rounds its work-group count up to cover (batch_size + 5)/6
		// groups of 6 candidates, so it writes roundup6(batch_size) result bytes — up to 5
		// past batch_size when batch_size isn't a multiple of 6. Size the device buffer to
		// match; an under-sized buffer is an out-of-bounds device write that NVIDIA tolerates
		// but AMD faults (zeroing the kernel's END profiling timestamp and the result data).
		const uint32_t result_buffer_size = args.use_ilp6 ? ((args.batch_size + 5u) / 6u) * 6u : args.batch_size;
		cl_mem result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, result_buffer_size, nullptr, &status);
		throw_if_error(status, "clCreateBuffer(result_buffer)");

		throw_if_error(clSetKernelArg(screen_kernel, 0, sizeof(result_buffer), &result_buffer), "clSetKernelArg(screen.result)");
		throw_if_error(clSetKernelArg(screen_kernel, 1, sizeof(assets.regular_rng_values), &assets.regular_rng_values), "clSetKernelArg(screen.regular_rng_values)");
		throw_if_error(clSetKernelArg(screen_kernel, 2, sizeof(assets.alg0_values), &assets.alg0_values), "clSetKernelArg(screen.alg0_values)");
		throw_if_error(clSetKernelArg(screen_kernel, 3, sizeof(assets.alg6_values), &assets.alg6_values), "clSetKernelArg(screen.alg6_values)");
		throw_if_error(clSetKernelArg(screen_kernel, 4, sizeof(assets.rng_seed_forward_1), &assets.rng_seed_forward_1), "clSetKernelArg(screen.rng_seed_forward_1)");
		throw_if_error(clSetKernelArg(screen_kernel, 5, sizeof(assets.rng_seed_forward_128), &assets.rng_seed_forward_128), "clSetKernelArg(screen.rng_seed_forward_128)");
		throw_if_error(clSetKernelArg(screen_kernel, 6, sizeof(assets.alg2_values), &assets.alg2_values), "clSetKernelArg(screen.alg2_values)");
		throw_if_error(clSetKernelArg(screen_kernel, 7, sizeof(assets.alg5_values), &assets.alg5_values), "clSetKernelArg(screen.alg5_values)");
		throw_if_error(clSetKernelArg(screen_kernel, 8, sizeof(assets.expansion_values), &assets.expansion_values), "clSetKernelArg(screen.expansion_values)");
		throw_if_error(clSetKernelArg(screen_kernel, 9, sizeof(assets.schedule_data), &assets.schedule_data), "clSetKernelArg(screen.schedule_data)");
		throw_if_error(clSetKernelArg(screen_kernel, 10, sizeof(assets.carnival_data), &assets.carnival_data), "clSetKernelArg(screen.carnival_data)");
		throw_if_error(clSetKernelArg(screen_kernel, 11, sizeof(args.key_id), &args.key_id), "clSetKernelArg(screen.key_id)");

		cl_mem survivor_data_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, args.batch_size * sizeof(uint32_t), nullptr, &status);
		throw_if_error(status, "clCreateBuffer(survivor_data_buffer)");
		cl_mem survivor_flag_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, args.batch_size, nullptr, &status);
		throw_if_error(status, "clCreateBuffer(survivor_flag_buffer)");
		cl_mem materialized_state_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, args.batch_size * 128, nullptr, &status);
		throw_if_error(status, "clCreateBuffer(materialized_state_buffer)");

		throw_if_error(clSetKernelArg(materialize_kernel, 0, sizeof(materialized_state_buffer), &materialized_state_buffer), "clSetKernelArg(materialize.output)");
		throw_if_error(clSetKernelArg(materialize_kernel, 1, sizeof(survivor_data_buffer), &survivor_data_buffer), "clSetKernelArg(materialize.survivor_data)");
		throw_if_error(clSetKernelArg(materialize_kernel, 2, sizeof(survivor_flag_buffer), &survivor_flag_buffer), "clSetKernelArg(materialize.survivor_flags)");
		throw_if_error(clSetKernelArg(materialize_kernel, 3, sizeof(assets.regular_rng_values), &assets.regular_rng_values), "clSetKernelArg(materialize.regular_rng_values)");
		throw_if_error(clSetKernelArg(materialize_kernel, 4, sizeof(assets.alg0_values), &assets.alg0_values), "clSetKernelArg(materialize.alg0_values)");
		throw_if_error(clSetKernelArg(materialize_kernel, 5, sizeof(assets.alg6_values), &assets.alg6_values), "clSetKernelArg(materialize.alg6_values)");
		throw_if_error(clSetKernelArg(materialize_kernel, 6, sizeof(assets.rng_seed_forward_1), &assets.rng_seed_forward_1), "clSetKernelArg(materialize.rng_seed_forward_1)");
		throw_if_error(clSetKernelArg(materialize_kernel, 7, sizeof(assets.rng_seed_forward_128), &assets.rng_seed_forward_128), "clSetKernelArg(materialize.rng_seed_forward_128)");
		throw_if_error(clSetKernelArg(materialize_kernel, 8, sizeof(assets.alg2_values), &assets.alg2_values), "clSetKernelArg(materialize.alg2_values)");
		throw_if_error(clSetKernelArg(materialize_kernel, 9, sizeof(assets.alg5_values), &assets.alg5_values), "clSetKernelArg(materialize.alg5_values)");
		throw_if_error(clSetKernelArg(materialize_kernel, 10, sizeof(assets.expansion_values), &assets.expansion_values), "clSetKernelArg(materialize.expansion_values)");
		throw_if_error(clSetKernelArg(materialize_kernel, 11, sizeof(assets.schedule_data), &assets.schedule_data), "clSetKernelArg(materialize.schedule_data)");
		throw_if_error(clSetKernelArg(materialize_kernel, 12, sizeof(assets.carnival_data), &assets.carnival_data), "clSetKernelArg(materialize.carnival_data)");
		throw_if_error(clSetKernelArg(materialize_kernel, 13, sizeof(args.key_id), &args.key_id), "clSetKernelArg(materialize.key_id)");

		// Kernel args for tm_checksum_screen_offset_ilp6 (offset-stream-based).
		// arg0=result, arg1-3=offset byte streams, arg4-5=offset carry uint32 streams,
		// arg6=expansion, arg7=schedule, arg8=carnival, arg9=key, arg10=data_start.
		auto set_ilp6_static_args = [&](cl_mem result_buf) {
			throw_if_error(clSetKernelArg(screen_ilp6_kernel,  0, sizeof(result_buf), &result_buf), "clSetKernelArg(ilp6.result)");
			throw_if_error(clSetKernelArg(screen_ilp6_kernel,  1, sizeof(assets.offset_regular), &assets.offset_regular), "clSetKernelArg(ilp6.offset_regular)");
			throw_if_error(clSetKernelArg(screen_ilp6_kernel,  2, sizeof(assets.offset_alg0),    &assets.offset_alg0),    "clSetKernelArg(ilp6.offset_alg0)");
			throw_if_error(clSetKernelArg(screen_ilp6_kernel,  3, sizeof(assets.offset_alg6),    &assets.offset_alg6),    "clSetKernelArg(ilp6.offset_alg6)");
			throw_if_error(clSetKernelArg(screen_ilp6_kernel,  4, sizeof(assets.offset_alg2),    &assets.offset_alg2),    "clSetKernelArg(ilp6.offset_alg2)");
			throw_if_error(clSetKernelArg(screen_ilp6_kernel,  5, sizeof(assets.offset_alg5),    &assets.offset_alg5),    "clSetKernelArg(ilp6.offset_alg5)");
			throw_if_error(clSetKernelArg(screen_ilp6_kernel,  6, sizeof(assets.expansion_values),&assets.expansion_values),"clSetKernelArg(ilp6.expansion)");
			throw_if_error(clSetKernelArg(screen_ilp6_kernel,  7, sizeof(assets.schedule_data),  &assets.schedule_data),  "clSetKernelArg(ilp6.schedule)");
			throw_if_error(clSetKernelArg(screen_ilp6_kernel,  8, sizeof(assets.carnival_data),  &assets.carnival_data),  "clSetKernelArg(ilp6.carnival)");
			throw_if_error(clSetKernelArg(screen_ilp6_kernel,  9, sizeof(args.key_id), &args.key_id), "clSetKernelArg(ilp6.key)");
		};

		// TDR-safe launch of the ilp6 screen over [data_start, data_start+count). A single
		// NDRange over a full 64M tile runs ~3 s, which can trip the Windows GPU watchdog
		// (TDR, default ~2 s) on the display adapter -> driver reset / blackout / bugcheck.
		// Splitting into sub-launches keeps every dispatch well under the limit. A global
		// work-group offset makes each sub-launch write to its absolute result slot
		// (result_data[get_global_id(1)*6 + j]) and derive the correct data index, so the
		// output and total GPU work are byte-identical to one launch — per-launch overhead
		// is microseconds and the screen rate is flat across launch sizes. Requires args 1-8
		// to have been set first (via set_ilp6_static_args); this sets 0, 9, 10.
		auto enqueue_ilp6_screen = [&](cl_mem result_buf, uint32_t key_id, uint32_t ds, uint32_t count) {
			throw_if_error(clSetKernelArg(screen_ilp6_kernel, 0, sizeof(result_buf), &result_buf), "ilp6.chunk.res");
			throw_if_error(clSetKernelArg(screen_ilp6_kernel, 9, sizeof(key_id), &key_id), "ilp6.chunk.key");
			throw_if_error(clSetKernelArg(screen_ilp6_kernel, 10, sizeof(ds), &ds), "ilp6.chunk.ds");
			const size_t total_wg = (count + 5u) / 6u;
			const size_t CHUNK_WG = wd_chunk_cands / 6u;  // watchdog-safe cand/sub-launch (CU-scaled; 6 cand/wg)
			size_t lsz[3] = { 32, 1, 1 };
			for (size_t wg0 = 0; wg0 < total_wg; wg0 += CHUNK_WG)
			{
				size_t this_wg = (total_wg - wg0 < CHUNK_WG) ? (total_wg - wg0) : CHUNK_WG;
				size_t off[3] = { 0, wg0, 0 };
				size_t gsz[3] = { 32, this_wg, 1 };
				throw_if_error(clEnqueueNDRangeKernel(queue, screen_ilp6_kernel, 3, off, gsz, lsz, 0, nullptr, nullptr), "ilp6.chunk.enq");
			}
		};

		// ---- Calibration: find host-optimal engine/ILP, write a portable config. ----
		// Single binary, no shell/script dependency (cross-platform). Rebuilds only the
		// span kernel per ILP (compact/resolve/screen are ILP-independent), sweeps a
		// representative random-key/window sample, picks engine by measured aggregate.
		if (args.calibrate)
		{
			// Calibrate AT the production tile size, not a tiny fixed window. The compaction
			// win comes from duplicate decrypted states collapsing within a tile; at 1M
			// candidates too few collide, so span-dedup overhead dominates and compaction is
			// mis-measured as a loss (the false "screen wins" verdict). The tile auto-sizes
			// from VRAM using the same 144 B/candidate formula the --compaction path uses, so
			// CAL_N = tile is guaranteed to fit. (Measured RX 7800 XT: 1M -> 0.95x but the real
			// 64M tile is 1.15x aggregate across 2^32, up to 1.85x in dense data windows.)
			const cl_ulong cal_reserve = 256ull * 1024 * 1024;
			const cl_ulong cal_budget  = (dev_global_mem > cal_reserve) ? (cl_ulong)((dev_global_mem - cal_reserve) * 0.90) : 0;
			cl_ulong cal_nmax = cal_budget / 144ull;
			const cl_ulong cal_nmax_alloc = dev_max_alloc / 128ull;  // state buffer = N*128
			if (cal_nmax_alloc < cal_nmax) cal_nmax = cal_nmax_alloc;
			uint32_t tile = 4u * 1024u * 1024u;
			while (((cl_ulong)tile << 1) <= cal_nmax && (tile << 1) <= 134217728u) tile <<= 1;
			const uint32_t CAL_N = tile;              // sample at the real production tile size
			const uint32_t cuts[9] = { 0u,1u,5u,9u,13u,17u,21u,25u,27u };
			const uint32_t NSPANS = 8u;
			const size_t span_ls = 32u, loc1 = 256u;
			cl_int e = CL_SUCCESS;

			cl_mem buf_ref  = clCreateBuffer(context, CL_MEM_READ_WRITE, ((CAL_N + 5u) / 6u) * 6u, nullptr, &e); throw_if_error(e, "cal.ref");
			set_ilp6_static_args(buf_ref);  // sets screen offset-stream args 1-8 (+0,9); run_screen overrides 0/9/10
			cl_mem state    = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)CAL_N * 32 * sizeof(uint32_t), nullptr, &e); throw_if_error(e, "cal.state");
			cl_mem rep_glob = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)CAL_N * sizeof(uint32_t), nullptr, &e); throw_if_error(e, "cal.rep");
			cl_mem alive    = clCreateBuffer(context, CL_MEM_READ_WRITE, CAL_N, nullptr, &e); throw_if_error(e, "cal.alive");
			cl_mem live_a   = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)CAL_N * sizeof(uint32_t), nullptr, &e); throw_if_error(e, "cal.la");
			cl_mem live_b   = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)CAL_N * sizeof(uint32_t), nullptr, &e); throw_if_error(e, "cal.lb");
			cl_mem counter  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr, &e); throw_if_error(e, "cal.cnt");
			cl_mem flag     = clCreateBuffer(context, CL_MEM_READ_WRITE, CAL_N, nullptr, &e); throw_if_error(e, "cal.flag");
			cl_mem buf_comp = clCreateBuffer(context, CL_MEM_READ_WRITE, CAL_N, nullptr, &e); throw_if_error(e, "cal.comp");

			throw_if_error(clSetKernelArg(compact_kernel, 4, sizeof(counter), &counter), "cal.cmp.cnt");
			throw_if_error(clSetKernelArg(resolve_kernel, 0, sizeof(rep_glob), &rep_glob), "cal.res.rep");
			throw_if_error(clSetKernelArg(resolve_kernel, 1, sizeof(flag), &flag), "cal.res.flag");
			throw_if_error(clSetKernelArg(resolve_kernel, 2, sizeof(buf_comp), &buf_comp), "cal.res.res");
			throw_if_error(clSetKernelArg(resolve_kernel, 3, sizeof(CAL_N), &CAL_N), "cal.res.N");

			// Representative workload: pseudo-random keys (splitmix64) × spread windows.
			auto splitmix = [](uint64_t x) -> uint32_t {
				x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
				x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return (uint32_t)(x ^ (x >> 31));
			};
			std::vector<std::pair<uint32_t,uint32_t>> samples;
			const uint32_t windows[3] = { 0u, 0x55555555u, 0xAAAAAAAAu };
			// Each sample is now a full-tile pipeline (~100x the old per-sample work), so use
			// 2 keys x 3 windows = 6 samples. That spans the compression range adequately: the
			// data-window position dominates the dedup ratio, while the key axis barely moves it.
			for (uint32_t ki = 0; ki < 2u; ki++)
				for (uint32_t wi = 0; wi < 3u; wi++)
					samples.push_back({ splitmix(0xC0FFEEu + ki), windows[wi] });

			// Run the full compaction pipeline (8 spans, fused final, resolve) for the
			// current span_kernel into buf_comp. screen_ilp6 -> buf_ref.
			auto run_screen = [&](uint32_t key_id, uint32_t ds) {
				enqueue_ilp6_screen(buf_ref, key_id, ds, CAL_N);  // TDR-safe chunked launch
			};
			auto run_pipeline = [&](cl_kernel span_k, uint32_t key_id, uint32_t ds, uint32_t W) {
				const uint32_t fill = 0xFFFFFFFFu;
				throw_if_error(clEnqueueFillBuffer(queue, rep_glob, &fill, sizeof(fill), 0, (size_t)CAL_N * sizeof(uint32_t), 0, nullptr, nullptr), "cal.fillrep");
				uint32_t M = CAL_N; cl_mem in_live = live_a, out_live = live_b; int first = 1;
				for (uint32_t sp = 0; sp < NSPANS; sp++) {
					uint32_t m0 = cuts[sp], m1 = cuts[sp + 1]; int mode = (sp == NSPANS - 1u) ? 2 : 1;
					clSetKernelArg(span_k, 0, sizeof(in_live), &in_live);
					clSetKernelArg(span_k, 1, sizeof(M), &M);
					clSetKernelArg(span_k, 5, sizeof(m0), &m0);
					clSetKernelArg(span_k, 6, sizeof(m1), &m1);
					clSetKernelArg(span_k, 7, sizeof(first), &first);
					clSetKernelArg(span_k, 17, sizeof(key_id), &key_id);
					clSetKernelArg(span_k, 18, sizeof(ds), &ds);
					clSetKernelArg(span_k, 19, sizeof(mode), &mode);
					// Chunk the span over dim1 work-groups (run_span_dedup keys off
					// get_global_id(1), so a global offset shifts the candidate base just like
					// the screen path). A single full-tile launch runs multiple seconds at the
					// representative tile on a small iGPU and trips the GPU compute watchdog
					// (hard recovery); ~1M candidates/sub-launch is proven watchdog-safe.
					const size_t total_wg = (M + W - 1u) / W;
					const size_t CHUNK_WG = (W > 0u) ? (wd_chunk_cands / W) : (size_t)1u;
					size_t sl[3] = { span_ls, 1, 1 };
					for (size_t wg0 = 0; wg0 < total_wg; wg0 += CHUNK_WG) {
						size_t this_wg = (total_wg - wg0 < CHUNK_WG) ? (total_wg - wg0) : CHUNK_WG;
						size_t off[3] = { 0, wg0, 0 };
						size_t sg[3] = { span_ls, this_wg, 1 };
						throw_if_error(clEnqueueNDRangeKernel(queue, span_k, 3, off, sg, sl, 0, nullptr, nullptr), "cal.span");
					}
					if (mode == 1) {
						const uint32_t zero = 0u;
						clEnqueueFillBuffer(queue, counter, &zero, sizeof(zero), 0, sizeof(zero), 0, nullptr, nullptr);
						clSetKernelArg(compact_kernel, 0, sizeof(alive), &alive);
						clSetKernelArg(compact_kernel, 1, sizeof(in_live), &in_live);
						clSetKernelArg(compact_kernel, 2, sizeof(M), &M);
						clSetKernelArg(compact_kernel, 3, sizeof(out_live), &out_live);
						clSetKernelArg(compact_kernel, 5, sizeof(first), &first);
						size_t cg = ((M + 255u) / 256u) * 256u;
						throw_if_error(clEnqueueNDRangeKernel(queue, compact_kernel, 1, nullptr, &cg, &loc1, 0, nullptr, nullptr), "cal.cmp");
						clFinish(queue);
						clEnqueueReadBuffer(queue, counter, CL_TRUE, 0, sizeof(M), &M, 0, nullptr, nullptr);
						cl_mem t = in_live; in_live = out_live; out_live = t; first = 0;
					}
				}
				size_t rg = ((CAL_N + 255u) / 256u) * 256u;
				throw_if_error(clEnqueueNDRangeKernel(queue, resolve_kernel, 1, nullptr, &rg, &loc1, 0, nullptr, nullptr), "cal.res");
				clFinish(queue);
			};

			const uint32_t ilp_candidates[3] = { 4u, 8u, 16u };
			uint32_t best_ilp = 8u; double best_agg = 0.0; std::size_t total_mism = 0;
			std::cout << "Calibrating " << device_name << " — " << samples.size() << " samples x "
			          << "ilp{4,8,16}...\n";
			std::vector<uint8_t> hr(((CAL_N + 5u) / 6u) * 6u), hc(CAL_N);
			for (uint32_t ic = 0; ic < 3u; ic++)
			{
				const uint32_t ilp = ilp_candidates[ic];
				const std::string opts = "-cl-std=CL1.2 -DSPAN_ILP=" + std::to_string(ilp);
				cl_program prog_i = clCreateProgramWithSource(context, 1, &source_ptr, &source_size, &e); throw_if_error(e, "cal.prog");
				if (clBuildProgram(prog_i, 1, &device, opts.c_str(), nullptr, nullptr) != CL_SUCCESS)
				{
					size_t ls = 0; clGetProgramBuildInfo(prog_i, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ls);
					std::string log(ls, '\0'); clGetProgramBuildInfo(prog_i, device, CL_PROGRAM_BUILD_LOG, ls, &log[0], nullptr);
					throw std::runtime_error("calibrate build failed (ilp=" + std::to_string(ilp) + "):\n" + log);
				}
				cl_kernel span_k = clCreateKernel(prog_i, "run_span_dedup", &e); throw_if_error(e, "cal.span.k");
				clSetKernelArg(span_k, 2, sizeof(state), &state);
				clSetKernelArg(span_k, 3, sizeof(alive), &alive);
				clSetKernelArg(span_k, 4, sizeof(rep_glob), &rep_glob);
				clSetKernelArg(span_k, 8, sizeof(assets.offset_regular), &assets.offset_regular);
				clSetKernelArg(span_k, 9, sizeof(assets.offset_alg0), &assets.offset_alg0);
				clSetKernelArg(span_k, 10, sizeof(assets.offset_alg6), &assets.offset_alg6);
				clSetKernelArg(span_k, 11, sizeof(assets.offset_alg2), &assets.offset_alg2);
				clSetKernelArg(span_k, 12, sizeof(assets.offset_alg5), &assets.offset_alg5);
				clSetKernelArg(span_k, 13, sizeof(assets.expansion_values), &assets.expansion_values);
				clSetKernelArg(span_k, 14, sizeof(assets.schedule_data), &assets.schedule_data);
				clSetKernelArg(span_k, 15, sizeof(assets.carnival_data), &assets.carnival_data);
				clSetKernelArg(span_k, 16, sizeof(flag), &flag);

				double inv_sum = 0.0;
				for (auto& smp : samples)
				{
					clFinish(queue);
					auto a0 = std::chrono::steady_clock::now();
					run_screen(smp.first, smp.second); clFinish(queue);
					auto a1 = std::chrono::steady_clock::now();
					run_pipeline(span_k, smp.first, smp.second, ilp);
					auto a2 = std::chrono::steady_clock::now();
					const double s_ms = std::chrono::duration<double, std::milli>(a1 - a0).count();
					const double c_ms = std::chrono::duration<double, std::milli>(a2 - a1).count();
					clEnqueueReadBuffer(queue, buf_ref, CL_TRUE, 0, hr.size(), hr.data(), 0, nullptr, nullptr);
					clEnqueueReadBuffer(queue, buf_comp, CL_TRUE, 0, CAL_N, hc.data(), 0, nullptr, nullptr);
					for (uint32_t i = 0; i < CAL_N; i++) if (hr[i] != hc[i]) total_mism++;
					inv_sum += c_ms / s_ms;  // 1 / (s/c) = c/s
				}
				const double agg = (double)samples.size() / inv_sum;
				std::cout << "  ilp=" << ilp << "  aggregate speedup vs screen: " << std::fixed << std::setprecision(3) << agg << "x\n";
				if (agg > best_agg) { best_agg = agg; best_ilp = ilp; }
				clReleaseKernel(span_k); clReleaseProgram(prog_i);
			}

			// Engine from measured aggregate; tile already computed above (production size,
			// and equal to CAL_N — calibration measured at exactly this size).
			const char* engine = (best_agg >= 1.0) ? "compaction" : "screen";

			std::ostringstream fpkey;
			fpkey << "opencl|" << device_name << "|" << (dev_global_mem >> 20) << "MB|" << dev_cus << "CU";
			std::ostringstream line;
			line << fpkey.str() << "\tengine=" << engine << " ilp=" << best_ilp
			     << " tile=" << (tile >> 20) << "M aggregate=" << std::fixed << std::setprecision(3) << best_agg
			     << " runtime=opencl";

			// Read-modify-write the config (portable text; replace any line for this device).
			std::vector<std::string> lines;
			{
				std::ifstream in(args.config_path);
				std::string l;
				while (std::getline(in, l)) {
					if (l.empty() || l[0] == '#') { lines.push_back(l); continue; }
					const std::string key = l.substr(0, l.find('\t'));
					if (key != fpkey.str()) lines.push_back(l);
				}
			}
			lines.push_back(line.str());
			{
				std::ofstream out(args.config_path, std::ios::trunc);
				out << "# tm_compaction.conf  (device-fingerprint -> engine/ilp/tile)\n";
				for (auto& l : lines) if (!(l.size() && l[0] == '#')) out << l << "\n";
			}

			std::cout << "Calibration result: engine=" << engine << "  ilp=" << best_ilp
			          << "  tile=" << (tile >> 20) << "M  aggregate=" << std::fixed << std::setprecision(3) << best_agg << "x\n";
			std::cout << (total_mism == 0 ? "  parity: PASS (all samples match the screen)\n"
			             : ("  parity: FAIL (" + std::to_string(total_mism) + " mismatches)\n"));
			std::cout << "  wrote " << args.config_path << "  [" << fpkey.str() << "]\n";
			if (best_agg < 1.0)
				std::cout << "  note: compaction is below break-even here (likely small-L2) -> flat screen chosen.\n";
			// Per-device: remind to calibrate the host's other GPUs (config accumulates).
			if (devices.size() > 1u)
			{
				std::cout << "  host has " << devices.size() << " GPUs; calibrate the others (config keyed per device):\n";
				for (uint32_t d = 0; d < devices.size(); d++)
				{
					if (d == args.device_index) continue;
					std::cout << "    --calibrate " << args.config_path << " --device " << d
					          << "   (" << get_device_string(devices[d], CL_DEVICE_NAME) << ")\n";
				}
			}

			clReleaseMemObject(buf_ref); clReleaseMemObject(state); clReleaseMemObject(rep_glob);
			clReleaseMemObject(alive); clReleaseMemObject(live_a); clReleaseMemObject(live_b);
			clReleaseMemObject(counter); clReleaseMemObject(flag); clReleaseMemObject(buf_comp);
			release_assets(assets);
			if (raceway_cap_mark_kernel) clReleaseKernel(raceway_cap_mark_kernel);
			clReleaseKernel(span_kernel); clReleaseKernel(compact_kernel); clReleaseKernel(resolve_kernel);
			clReleaseKernel(screen_ilp6_kernel); clReleaseKernel(materialize_kernel); clReleaseKernel(screen_kernel);
			clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
			return 0;
		}

		if (args.raceway_wave_cap_mark)
		{
			const PrecertPlan precert_plan = make_precert_plan(args, "OpenCL wave raceway");
			const uint32_t N = precert_plan.active ? precert_plan.logical_window : args.workunit_size;
			const uint32_t data_start = precert_plan.active ? 0u : args.range_start;
			const uint32_t use_precert = precert_plan.active ? 1u : 0u;
			const std::vector<uint32_t> cap_maps_host = build_raceway_cap_maps(args);
			if (cap_maps_host.empty())
			{
				throw std::runtime_error("--raceway-wave-cap-mark requires nonzero caps and at least one boundary");
			}
			const uint32_t effective_cap_count = static_cast<uint32_t>(cap_maps_host.size());
			const uint64_t cap_buckets = (args.raceway_cap_bits == 0u) ? 1ull : (1ull << args.raceway_cap_bits);
			const uint64_t cap_slots_per_table_u64 = cap_buckets * static_cast<uint64_t>(args.raceway_cap_ways);
			const uint64_t cap_slots_u64 = cap_slots_per_table_u64 * static_cast<uint64_t>(effective_cap_count);
			const std::size_t cap_slots = static_cast<std::size_t>(cap_slots_u64);
			const std::size_t cap_slot_bytes = raceway_cap_fp64 ? sizeof(uint64_t) : sizeof(uint32_t);
			const std::size_t cap_bytes = cap_slots * cap_slot_bytes;
			const std::size_t cap_table_bytes = static_cast<std::size_t>(cap_slots_per_table_u64) * cap_slot_bytes;
			const uint32_t kMaxWaveTile = 64u * 1024u * 1024u;
			const uint64_t max_state_candidates = dev_max_alloc / (32ull * sizeof(uint32_t));
			uint32_t wave_tile = std::min<uint32_t>(N, kMaxWaveTile);
			if (max_state_candidates < wave_tile)
			{
				wave_tile = static_cast<uint32_t>(max_state_candidates);
			}
			if (wave_tile == 0u)
			{
				throw std::runtime_error("OpenCL wave raceway cannot fit even one saved-state candidate in CL_DEVICE_MAX_MEM_ALLOC_SIZE");
			}

			cl_int s_r = CL_SUCCESS;
			cl_mem alive = clCreateBuffer(context, CL_MEM_READ_WRITE, wave_tile, nullptr, &s_r);
			throw_if_error(s_r, "clCreateBuffer(raceway.wave.alive)");
			cl_mem drop_map = clCreateBuffer(context, CL_MEM_READ_WRITE, wave_tile, nullptr, &s_r);
			throw_if_error(s_r, "clCreateBuffer(raceway.wave.drop)");
			cl_mem state = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)wave_tile * 32u * sizeof(uint32_t), nullptr, &s_r);
			throw_if_error(s_r, "clCreateBuffer(raceway.wave.state)");
			cl_mem live_a = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)wave_tile * sizeof(uint32_t), nullptr, &s_r);
			throw_if_error(s_r, "clCreateBuffer(raceway.wave.live_a)");
			cl_mem live_b = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)wave_tile * sizeof(uint32_t), nullptr, &s_r);
			throw_if_error(s_r, "clCreateBuffer(raceway.wave.live_b)");
			cl_mem compact_counter = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr, &s_r);
			throw_if_error(s_r, "clCreateBuffer(raceway.wave.counter)");
			cl_mem cap_table = clCreateBuffer(context, CL_MEM_READ_WRITE, cap_bytes, nullptr, &s_r);
			throw_if_error(s_r, "clCreateBuffer(raceway.wave.cap_table)");
			cl_mem continue_flags = clCreateBuffer(context, CL_MEM_READ_WRITE, wave_tile, nullptr, &s_r);
			throw_if_error(s_r, "clCreateBuffer(raceway.wave.continue_flags)");

			const uint8_t zero_u8 = 0u;
			const uint8_t ff_u8 = 0xFFu;
			const uint32_t zero_u32 = 0u;
			const uint64_t zero_u64 = 0u;
			if (raceway_cap_fp64)
				throw_if_error(clEnqueueFillBuffer(queue, cap_table, &zero_u64, sizeof(zero_u64), 0, cap_bytes, 0, nullptr, nullptr), "raceway.wave.fill(cap64)");
			else
				throw_if_error(clEnqueueFillBuffer(queue, cap_table, &zero_u32, sizeof(zero_u32), 0, cap_bytes, 0, nullptr, nullptr), "raceway.wave.fill(cap32)");

			auto timed_finish = [&](const char* label) -> double {
				(void)label;
				const auto t0 = std::chrono::high_resolution_clock::now();
				throw_if_error(clFinish(queue), "raceway.wave.finish");
				const auto t1 = std::chrono::high_resolution_clock::now();
				return std::chrono::duration<double, std::milli>(t1 - t0).count();
			};
			auto compact_alive = [&](cl_mem live_in, uint32_t count, cl_mem live_out, int first_span) -> std::pair<uint32_t, double> {
				throw_if_error(clEnqueueFillBuffer(queue, compact_counter, &zero_u32, sizeof(zero_u32), 0, sizeof(zero_u32), 0, nullptr, nullptr), "raceway.wave.fill(counter)");
				throw_if_error(clSetKernelArg(compact_kernel, 0, sizeof(alive), &alive), "raceway.wave.comp.alive");
				throw_if_error(clSetKernelArg(compact_kernel, 1, sizeof(live_in), &live_in), "raceway.wave.comp.in");
				throw_if_error(clSetKernelArg(compact_kernel, 2, sizeof(count), &count), "raceway.wave.comp.count");
				throw_if_error(clSetKernelArg(compact_kernel, 3, sizeof(live_out), &live_out), "raceway.wave.comp.out");
				throw_if_error(clSetKernelArg(compact_kernel, 4, sizeof(compact_counter), &compact_counter), "raceway.wave.comp.counter");
				throw_if_error(clSetKernelArg(compact_kernel, 5, sizeof(first_span), &first_span), "raceway.wave.comp.first");
				const size_t local = 256u;
				const size_t global = ((static_cast<size_t>(count) + local - 1u) / local) * local;
				const auto c0 = std::chrono::high_resolution_clock::now();
				throw_if_error(clEnqueueNDRangeKernel(queue, compact_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr), "raceway.wave.comp.enqueue");
				throw_if_error(clFinish(queue), "raceway.wave.comp.finish");
				const auto c1 = std::chrono::high_resolution_clock::now();
				uint32_t compacted = 0u;
				throw_if_error(clEnqueueReadBuffer(queue, compact_counter, CL_TRUE, 0, sizeof(compacted), &compacted, 0, nullptr, nullptr), "raceway.wave.read(counter)");
				return { compacted, std::chrono::duration<double, std::milli>(c1 - c0).count() };
			};

			const size_t local3[3] = { 32u, 1u, 1u };
			double span_ms_total = 0.0;
			double compact_ms_total = 0.0;
			double continue_ms_total = 0.0;
			uint64_t estimated_map_evals = 0;
			uint64_t total_survivors = 0u;
			uint32_t batches = 0u;
			std::vector<uint64_t> boundary_inputs(effective_cap_count, 0u);
			std::vector<uint64_t> boundary_survivors(effective_cap_count, 0u);

			auto run_wave_chunk = [&](uint32_t chunk_data_start, uint32_t chunk_n) {
			const uint32_t data_start = chunk_data_start;
			const uint32_t N = chunk_n;
			throw_if_error(clEnqueueFillBuffer(queue, alive, &zero_u8, sizeof(zero_u8), 0, N, 0, nullptr, nullptr), "raceway.wave.fill(alive)");
			throw_if_error(clEnqueueFillBuffer(queue, drop_map, &ff_u8, sizeof(ff_u8), 0, N, 0, nullptr, nullptr), "raceway.wave.fill(drop)");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 0, sizeof(alive), &alive), "wave.first.alive");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 1, sizeof(drop_map), &drop_map), "wave.first.drop");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 2, sizeof(state), &state), "wave.first.state");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 3, sizeof(cap_table), &cap_table), "wave.first.cap");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 4, sizeof(args.raceway_cap_bits), &args.raceway_cap_bits), "wave.first.bits");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 5, sizeof(args.raceway_cap_ways), &args.raceway_cap_ways), "wave.first.ways");
			uint32_t cap_index = 0u;
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 6, sizeof(cap_index), &cap_index), "wave.first.cap_index");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 7, sizeof(assets.offset_regular), &assets.offset_regular), "wave.first.reg");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 8, sizeof(assets.offset_alg0), &assets.offset_alg0), "wave.first.a0");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 9, sizeof(assets.offset_alg6), &assets.offset_alg6), "wave.first.a6");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 10, sizeof(assets.offset_alg2), &assets.offset_alg2), "wave.first.a2");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 11, sizeof(assets.offset_alg5), &assets.offset_alg5), "wave.first.a5");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 12, sizeof(assets.expansion_values), &assets.expansion_values), "wave.first.exp");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 13, sizeof(assets.schedule_data), &assets.schedule_data), "wave.first.sched");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 14, sizeof(args.key_id), &args.key_id), "wave.first.key");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 15, sizeof(data_start), &data_start), "wave.first.ds");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 16, sizeof(N), &N), "wave.first.N");
			uint32_t end_map = cap_maps_host[0];
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 17, sizeof(end_map), &end_map), "wave.first.end");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 18, sizeof(precert_plan.fixed_value), &precert_plan.fixed_value), "wave.first.precert.fixed");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 19, sizeof(precert_plan.support_mask), &precert_plan.support_mask), "wave.first.precert.support");
			throw_if_error(clSetKernelArg(raceway_wave_first_kernel, 20, sizeof(use_precert), &use_precert), "wave.first.precert.use");
			const size_t first_global[3] = { 32u, (static_cast<size_t>(N) + args.raceway_cap_ilp - 1u) / args.raceway_cap_ilp, 1u };
			const auto f0 = std::chrono::high_resolution_clock::now();
			throw_if_error(clEnqueueNDRangeKernel(queue, raceway_wave_first_kernel, 3, nullptr, first_global, local3, 0, nullptr, nullptr), "wave.first.enqueue");
			throw_if_error(clFinish(queue), "wave.first.finish");
			const auto f1 = std::chrono::high_resolution_clock::now();
			span_ms_total += std::chrono::duration<double, std::milli>(f1 - f0).count();
			estimated_map_evals += static_cast<uint64_t>(N) * static_cast<uint64_t>(end_map + 1u);
			boundary_inputs[0] += N;
			auto compact0 = compact_alive(nullptr, N, live_a, 1);
			uint32_t cur_count = compact0.first;
			compact_ms_total += compact0.second;
			boundary_survivors[0] += cur_count;
			cl_mem cur_live = live_a;
			cl_mem next_live = live_b;
			uint32_t prev_boundary = end_map;

			for (uint32_t bi = 1u; bi < effective_cap_count && cur_count != 0u; bi++)
			{
				uint32_t start_map = prev_boundary + 1u;
				end_map = cap_maps_host[bi];
				boundary_inputs[bi] += cur_count;
				estimated_map_evals += static_cast<uint64_t>(cur_count) * static_cast<uint64_t>(end_map - start_map + 1u);
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 0, sizeof(cur_live), &cur_live), "wave.span.live");
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 1, sizeof(cur_count), &cur_count), "wave.span.count");
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 2, sizeof(state), &state), "wave.span.state");
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 3, sizeof(alive), &alive), "wave.span.alive");
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 4, sizeof(drop_map), &drop_map), "wave.span.drop");
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 5, sizeof(cap_table), &cap_table), "wave.span.cap");
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 6, sizeof(args.raceway_cap_bits), &args.raceway_cap_bits), "wave.span.bits");
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 7, sizeof(args.raceway_cap_ways), &args.raceway_cap_ways), "wave.span.ways");
				cap_index = bi;
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 8, sizeof(cap_index), &cap_index), "wave.span.cap_index");
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 9, sizeof(assets.offset_regular), &assets.offset_regular), "wave.span.reg");
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 10, sizeof(assets.offset_alg0), &assets.offset_alg0), "wave.span.a0");
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 11, sizeof(assets.offset_alg6), &assets.offset_alg6), "wave.span.a6");
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 12, sizeof(assets.offset_alg2), &assets.offset_alg2), "wave.span.a2");
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 13, sizeof(assets.offset_alg5), &assets.offset_alg5), "wave.span.a5");
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 14, sizeof(assets.schedule_data), &assets.schedule_data), "wave.span.sched");
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 15, sizeof(start_map), &start_map), "wave.span.start");
				throw_if_error(clSetKernelArg(raceway_wave_span_kernel, 16, sizeof(end_map), &end_map), "wave.span.end");
				const size_t span_global[3] = { 32u, (static_cast<size_t>(cur_count) + args.raceway_cap_ilp - 1u) / args.raceway_cap_ilp, 1u };
				const auto s0 = std::chrono::high_resolution_clock::now();
				throw_if_error(clEnqueueNDRangeKernel(queue, raceway_wave_span_kernel, 3, nullptr, span_global, local3, 0, nullptr, nullptr), "wave.span.enqueue");
				throw_if_error(clFinish(queue), "wave.span.finish");
				const auto s1 = std::chrono::high_resolution_clock::now();
				span_ms_total += std::chrono::duration<double, std::milli>(s1 - s0).count();
				auto compacted = compact_alive(cur_live, cur_count, next_live, 0);
				cur_count = compacted.first;
				compact_ms_total += compacted.second;
				boundary_survivors[bi] += cur_count;
				std::swap(cur_live, next_live);
				prev_boundary = end_map;
			}

			const uint32_t continue_start_map = std::min<uint32_t>(27u, prev_boundary + 1u);
			if (cur_count != 0u && continue_start_map < 27u)
			{
				throw_if_error(clEnqueueFillBuffer(queue, continue_flags, &zero_u8, sizeof(zero_u8), 0, N, 0, nullptr, nullptr), "raceway.wave.fill(continue_flags)");
				uint32_t m0 = continue_start_map;
				uint32_t m1 = 27u;
				int first_span = 0;
				int mode = 2;
				throw_if_error(clSetKernelArg(span_kernel, 0, sizeof(cur_live), &cur_live), "wave.cont.live");
				throw_if_error(clSetKernelArg(span_kernel, 1, sizeof(cur_count), &cur_count), "wave.cont.count");
				throw_if_error(clSetKernelArg(span_kernel, 2, sizeof(state), &state), "wave.cont.state");
				throw_if_error(clSetKernelArg(span_kernel, 3, sizeof(alive), &alive), "wave.cont.alive");
				throw_if_error(clSetKernelArg(span_kernel, 4, sizeof(next_live), &next_live), "wave.cont.rep_dummy");
				throw_if_error(clSetKernelArg(span_kernel, 5, sizeof(m0), &m0), "wave.cont.m0");
				throw_if_error(clSetKernelArg(span_kernel, 6, sizeof(m1), &m1), "wave.cont.m1");
				throw_if_error(clSetKernelArg(span_kernel, 7, sizeof(first_span), &first_span), "wave.cont.first");
				throw_if_error(clSetKernelArg(span_kernel, 8, sizeof(assets.offset_regular), &assets.offset_regular), "wave.cont.reg");
				throw_if_error(clSetKernelArg(span_kernel, 9, sizeof(assets.offset_alg0), &assets.offset_alg0), "wave.cont.a0");
				throw_if_error(clSetKernelArg(span_kernel, 10, sizeof(assets.offset_alg6), &assets.offset_alg6), "wave.cont.a6");
				throw_if_error(clSetKernelArg(span_kernel, 11, sizeof(assets.offset_alg2), &assets.offset_alg2), "wave.cont.a2");
				throw_if_error(clSetKernelArg(span_kernel, 12, sizeof(assets.offset_alg5), &assets.offset_alg5), "wave.cont.a5");
				throw_if_error(clSetKernelArg(span_kernel, 13, sizeof(assets.expansion_values), &assets.expansion_values), "wave.cont.exp");
				throw_if_error(clSetKernelArg(span_kernel, 14, sizeof(assets.schedule_data), &assets.schedule_data), "wave.cont.sched");
				throw_if_error(clSetKernelArg(span_kernel, 15, sizeof(assets.carnival_data), &assets.carnival_data), "wave.cont.carnival");
				throw_if_error(clSetKernelArg(span_kernel, 16, sizeof(continue_flags), &continue_flags), "wave.cont.flags");
				throw_if_error(clSetKernelArg(span_kernel, 17, sizeof(args.key_id), &args.key_id), "wave.cont.key");
				throw_if_error(clSetKernelArg(span_kernel, 18, sizeof(data_start), &data_start), "wave.cont.ds");
				throw_if_error(clSetKernelArg(span_kernel, 19, sizeof(mode), &mode), "wave.cont.mode");
				const size_t cont_global[3] = { 32u, (static_cast<size_t>(cur_count) + args.compaction_ilp - 1u) / args.compaction_ilp, 1u };
				const auto c0 = std::chrono::high_resolution_clock::now();
				throw_if_error(clEnqueueNDRangeKernel(queue, span_kernel, 3, nullptr, cont_global, local3, 0, nullptr, nullptr), "wave.cont.enqueue");
				throw_if_error(clFinish(queue), "wave.cont.finish");
				const auto c1 = std::chrono::high_resolution_clock::now();
				continue_ms_total += std::chrono::duration<double, std::milli>(c1 - c0).count();
			}
			total_survivors += cur_count;
			++batches;
			};

			for (uint32_t done = 0u; done < N; )
			{
				const uint32_t chunk_n = std::min<uint32_t>(wave_tile, N - done);
				run_wave_chunk(data_start + done, chunk_n);
				done += chunk_n;
			}

			const double pipeline_ms = span_ms_total + compact_ms_total + continue_ms_total;
			std::ostringstream boundary_list;
			for (std::size_t i = 0; i < cap_maps_host.size(); i++)
			{
				if (i) boundary_list << ",";
				boundary_list << (cap_maps_host[i] + 1u);
			}
			std::cout << "OpenCL raceway wave-cap mark: " << N << " candidates, key=0x"
			          << std::hex << std::setw(8) << std::setfill('0') << args.key_id
			          << std::dec << std::setfill(' ') << "\n";
			print_precert_plan(precert_plan, args);
			std::cout << "  boundaries    : " << boundary_list.str()
			          << "  tile=" << wave_tile
			          << " batches=" << batches
			          << "  cap: 2^" << args.raceway_cap_bits << " x " << args.raceway_cap_ways
			          << " x " << effective_cap_count
			          << " (" << std::fixed << std::setprecision(1)
			          << (static_cast<double>(cap_bytes) / (1024.0 * 1024.0)) << " MB total, "
			          << (raceway_cap_fp64 ? "fp64" : "fp32 fallback") << ")\n";
			std::cout << "  wave kernel   : span=" << std::fixed << std::setprecision(3) << span_ms_total
			          << " ms compact=" << compact_ms_total
			          << " ms continue=" << continue_ms_total
			          << " ms pipeline=" << pipeline_ms << " ms  ("
			          << std::setprecision(1) << (static_cast<double>(N) / 1.0e6 / (pipeline_ms / 1000.0))
			          << " M logical input/s";
			if (precert_plan.active)
			{
				std::cout << ", " << std::setprecision(1)
				          << (static_cast<double>(args.workunit_size) / 1.0e6 / (pipeline_ms / 1000.0))
				          << " M represented input/s";
			}
			std::cout << ")\n";
			std::cout << "  cap-span only : " << std::fixed << std::setprecision(3) << span_ms_total << " ms   "
			          << std::fixed << std::setprecision(0) << (static_cast<double>(N) / (span_ms_total / 1000.0)) << " c/s\n";
			std::cout << "  survivors     : " << total_survivors
			          << " dropped=" << (static_cast<uint64_t>(N) - total_survivors)
			          << " map_evals=" << estimated_map_evals << "\n";
			std::cout << "  boundary      :";
			for (std::size_t i = 0; i < cap_maps_host.size(); i++)
			{
				std::cout << " m" << (cap_maps_host[i] + 1u)
				          << " input=" << boundary_inputs[i]
				          << " survivors=" << boundary_survivors[i];
			}
			std::cout << "\n";

			clReleaseMemObject(alive);
			clReleaseMemObject(drop_map);
			clReleaseMemObject(state);
			clReleaseMemObject(live_a);
			clReleaseMemObject(live_b);
			clReleaseMemObject(compact_counter);
			clReleaseMemObject(cap_table);
			clReleaseMemObject(continue_flags);
			clReleaseMemObject(result_buffer);
			clReleaseMemObject(survivor_data_buffer);
			clReleaseMemObject(survivor_flag_buffer);
			clReleaseMemObject(materialized_state_buffer);
			release_assets(assets);
			if (span_kernel) clReleaseKernel(span_kernel);
			if (compact_kernel) clReleaseKernel(compact_kernel);
			if (resolve_kernel) clReleaseKernel(resolve_kernel);
			if (raceway_wave_first_kernel) clReleaseKernel(raceway_wave_first_kernel);
			if (raceway_wave_span_kernel) clReleaseKernel(raceway_wave_span_kernel);
			clReleaseKernel(screen_ilp6_kernel);
			clReleaseKernel(materialize_kernel);
			clReleaseKernel(screen_kernel);
			clReleaseProgram(program);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 0;
		}

		// OpenCL raceway foundation: direct offset-stream boundary-cap mark pass.
		// This is intentionally a diagnostic exit path. It exercises the CUDA
		// raceway cap/mark shape on non-CUDA devices without changing the normal
		// screen+materialize workflow.
		if (args.raceway_cap_mark)
		{
			const PrecertPlan precert_plan = make_precert_plan(args, "OpenCL raceway cap-mark");
			const uint32_t N = precert_plan.active ? precert_plan.logical_window : args.workunit_size;
			const uint32_t data_start = precert_plan.active ? 0u : args.range_start;
			const uint32_t use_precert = precert_plan.active ? 1u : 0u;
			const std::vector<uint32_t> cap_maps_host = build_raceway_cap_maps(args);
			const uint32_t effective_cap_count = static_cast<uint32_t>(cap_maps_host.size());
			const uint32_t cap_table_count = effective_cap_count == 0u ? 1u : effective_cap_count;
			const uint64_t cap_buckets = (args.raceway_cap_bits == 0u) ? 1ull : (1ull << args.raceway_cap_bits);
			const uint64_t cap_slots_per_table_u64 = cap_buckets * static_cast<uint64_t>(args.raceway_cap_ways);
			if (cap_slots_per_table_u64 > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(cap_table_count))
			{
				throw std::runtime_error("OpenCL raceway cap table slot count overflows uint64_t");
			}
			const uint64_t cap_slots_u64 = cap_slots_per_table_u64 * static_cast<uint64_t>(cap_table_count);
			if (cap_slots_u64 > static_cast<uint64_t>(static_cast<std::size_t>(-1) / sizeof(uint32_t)))
			{
				throw std::runtime_error("OpenCL raceway cap table is too large for host size_t");
			}
			const std::size_t cap_slots = static_cast<std::size_t>(cap_slots_u64);
			const std::size_t cap_slot_bytes = raceway_cap_fp64 ? sizeof(uint64_t) : sizeof(uint32_t);
			if (cap_slots > static_cast<std::size_t>(-1) / cap_slot_bytes)
			{
				throw std::runtime_error("OpenCL raceway cap table byte size overflows size_t");
			}
			const std::size_t cap_bytes = cap_slots * cap_slot_bytes;

			cl_int s_r = CL_SUCCESS;
			cl_mem alive = clCreateBuffer(context, CL_MEM_READ_WRITE, N, nullptr, &s_r);
			throw_if_error(s_r, "clCreateBuffer(raceway.alive)");
			cl_mem drop_map = clCreateBuffer(context, CL_MEM_READ_WRITE, N, nullptr, &s_r);
			throw_if_error(s_r, "clCreateBuffer(raceway.drop_map)");
			cl_mem cap_table = clCreateBuffer(context, CL_MEM_READ_WRITE, cap_bytes, nullptr, &s_r);
			throw_if_error(s_r, "clCreateBuffer(raceway.cap_table)");
			cl_mem work_counter = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr, &s_r);
			throw_if_error(s_r, "clCreateBuffer(raceway.work_counter)");
			const std::vector<uint32_t> cap_maps_upload = cap_maps_host.empty() ? std::vector<uint32_t>{0u} : cap_maps_host;
			cl_mem cap_maps = create_read_only_buffer(context, queue, cap_maps_upload.data(), cap_maps_upload.size() * sizeof(uint32_t));

			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 0, sizeof(alive), &alive), "raceway.alive");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 1, sizeof(drop_map), &drop_map), "raceway.drop");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 2, sizeof(cap_table), &cap_table), "raceway.cap");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 3, sizeof(args.raceway_cap_bits), &args.raceway_cap_bits), "raceway.cap_bits");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 4, sizeof(args.raceway_cap_ways), &args.raceway_cap_ways), "raceway.cap_ways");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 5, sizeof(effective_cap_count), &effective_cap_count), "raceway.cap_count");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 6, sizeof(assets.offset_regular), &assets.offset_regular), "raceway.offset_regular");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 7, sizeof(assets.offset_alg0), &assets.offset_alg0), "raceway.offset_alg0");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 8, sizeof(assets.offset_alg6), &assets.offset_alg6), "raceway.offset_alg6");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 9, sizeof(assets.offset_alg2), &assets.offset_alg2), "raceway.offset_alg2");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 10, sizeof(assets.offset_alg5), &assets.offset_alg5), "raceway.offset_alg5");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 11, sizeof(assets.expansion_values), &assets.expansion_values), "raceway.expansion");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 12, sizeof(assets.schedule_data), &assets.schedule_data), "raceway.schedule");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 13, sizeof(args.key_id), &args.key_id), "raceway.key");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 14, sizeof(data_start), &data_start), "raceway.data_start");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 15, sizeof(N), &N), "raceway.N");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 16, sizeof(args.raceway_first_cap_map), &args.raceway_first_cap_map), "raceway.first_cap_map");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 17, sizeof(cap_maps), &cap_maps), "raceway.cap_maps");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 18, sizeof(work_counter), &work_counter), "raceway.work_counter");
			const uint32_t persistent_flag = args.raceway_persistent ? 1u : 0u;
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 19, sizeof(persistent_flag), &persistent_flag), "raceway.persistent");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 20, sizeof(precert_plan.fixed_value), &precert_plan.fixed_value), "raceway.precert.fixed");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 21, sizeof(precert_plan.support_mask), &precert_plan.support_mask), "raceway.precert.support");
			throw_if_error(clSetKernelArg(raceway_cap_mark_kernel, 22, sizeof(use_precert), &use_precert), "raceway.precert.use");

			const size_t span_count = (static_cast<size_t>(N) + args.raceway_cap_ilp - 1u) / args.raceway_cap_ilp;
			size_t wg_count = span_count;
			if (args.raceway_persistent)
			{
				const uint32_t requested_groups = args.raceway_persistent_groups == 0u ? kRacewayDefaultPersistentGroups : args.raceway_persistent_groups;
				wg_count = std::min<std::size_t>(span_count, requested_groups);
				if (wg_count == 0u) wg_count = 1u;
			}
			size_t global_item_size[3] = { 32, wg_count, 1 };
			size_t local_item_size[3] = { 32, 1, 1 };

			const uint8_t zero_u8 = 0u;
			const uint8_t ff_u8 = 0xFFu;
			const uint32_t zero_u32 = 0u;
			const uint64_t zero_u64 = 0u;
			auto reset_raceway_buffers = [&]() {
				throw_if_error(clEnqueueFillBuffer(queue, alive, &zero_u8, sizeof(zero_u8), 0, N, 0, nullptr, nullptr), "raceway.fill(alive)");
				throw_if_error(clEnqueueFillBuffer(queue, drop_map, &ff_u8, sizeof(ff_u8), 0, N, 0, nullptr, nullptr), "raceway.fill(drop_map)");
				throw_if_error(clEnqueueFillBuffer(queue, work_counter, &zero_u32, sizeof(zero_u32), 0, sizeof(zero_u32), 0, nullptr, nullptr), "raceway.fill(work_counter)");
				if (raceway_cap_fp64)
				{
					throw_if_error(clEnqueueFillBuffer(queue, cap_table, &zero_u64, sizeof(zero_u64), 0, cap_bytes, 0, nullptr, nullptr), "raceway.fill(cap_table64)");
				}
				else
				{
					throw_if_error(clEnqueueFillBuffer(queue, cap_table, &zero_u32, sizeof(zero_u32), 0, cap_bytes, 0, nullptr, nullptr), "raceway.fill(cap_table32)");
				}
			};
			auto run_raceway_once = [&]() -> double {
				reset_raceway_buffers();
				cl_event raceway_event = nullptr;
				throw_if_error(clEnqueueNDRangeKernel(queue, raceway_cap_mark_kernel, 3, nullptr, global_item_size, local_item_size, 0, nullptr, &raceway_event), "raceway.enqueue");
				throw_if_error(clWaitForEvents(1, &raceway_event), "raceway.wait");
				cl_ulong raceway_t0 = 0, raceway_t1 = 0;
				throw_if_error(clGetEventProfilingInfo(raceway_event, CL_PROFILING_COMMAND_START, sizeof(raceway_t0), &raceway_t0, nullptr), "raceway.profile.start");
				throw_if_error(clGetEventProfilingInfo(raceway_event, CL_PROFILING_COMMAND_END, sizeof(raceway_t1), &raceway_t1, nullptr), "raceway.profile.end");
				clReleaseEvent(raceway_event);
				return static_cast<double>(raceway_t1 - raceway_t0) / 1.0e6;
			};

			for (uint32_t i = 0; i < args.warmup_batches; i++)
			{
				(void)run_raceway_once();
			}

			std::vector<double> raceway_ms_runs(args.raceway_bench_repeats, 0.0);
			for (uint32_t i = 0; i < args.raceway_bench_repeats; i++)
			{
				raceway_ms_runs[i] = run_raceway_once();
			}

			std::vector<uint8_t> host_alive(N);
			std::vector<uint8_t> host_drop(N);
			throw_if_error(clEnqueueReadBuffer(queue, alive, CL_TRUE, 0, N, host_alive.data(), 0, nullptr, nullptr), "raceway.read(alive)");
			throw_if_error(clEnqueueReadBuffer(queue, drop_map, CL_TRUE, 0, N, host_drop.data(), 0, nullptr, nullptr), "raceway.read(drop)");

			uint64_t alive_count = 0;
			uint64_t dropped_count = 0;
			uint64_t bad_drop_count = 0;
			std::vector<uint64_t> drop_hist(27, 0);
			for (uint32_t i = 0; i < N; i++)
			{
				if (host_alive[i] != 0u)
				{
					alive_count++;
				}
				else
				{
					dropped_count++;
					if (host_drop[i] < drop_hist.size()) drop_hist[host_drop[i]]++;
					else bad_drop_count++;
				}
			}

			double raceway_min_ms = raceway_ms_runs[0];
			double raceway_max_ms = raceway_ms_runs[0];
			double raceway_sum_ms = 0.0;
			for (std::size_t i = 0; i < raceway_ms_runs.size(); i++)
			{
				if (raceway_ms_runs[i] < raceway_min_ms) raceway_min_ms = raceway_ms_runs[i];
				if (raceway_ms_runs[i] > raceway_max_ms) raceway_max_ms = raceway_ms_runs[i];
				raceway_sum_ms += raceway_ms_runs[i];
			}
			const double raceway_mean_ms = raceway_sum_ms / static_cast<double>(raceway_ms_runs.size());
			const double raceway_cps = static_cast<double>(N) / (raceway_min_ms / 1000.0);
			const double cap_mb = static_cast<double>(cap_bytes) / (1024.0 * 1024.0);
			const uint32_t last_map = cap_maps_host.empty() ? 0u : cap_maps_host.back() + 1u;
			uint64_t estimated_map_evals = alive_count * static_cast<uint64_t>(last_map);
			uint64_t estimated_cap_probes = alive_count * static_cast<uint64_t>(effective_cap_count);
			for (std::size_t m = 0; m < drop_hist.size(); m++)
			{
				const uint64_t n_drop = drop_hist[m];
				if (n_drop == 0u) continue;
				estimated_map_evals += n_drop * static_cast<uint64_t>(m + 1u);
				estimated_cap_probes += n_drop * static_cast<uint64_t>(
					std::upper_bound(cap_maps_host.begin(), cap_maps_host.end(), static_cast<uint32_t>(m)) - cap_maps_host.begin());
			}
			std::ostringstream boundary_list;
			for (std::size_t i = 0; i < cap_maps_host.size(); i++)
			{
				if (i != 0u) boundary_list << ",";
				boundary_list << (cap_maps_host[i] + 1u);
			}
			const double map_rate = raceway_min_ms == 0.0 ? 0.0 : static_cast<double>(estimated_map_evals) / (raceway_min_ms / 1000.0);
			const double probe_rate = raceway_min_ms == 0.0 ? 0.0 : static_cast<double>(estimated_cap_probes) / (raceway_min_ms / 1000.0);
			std::cout << "OpenCL raceway boundary-cap mark: " << N << " candidates, key=0x"
			          << std::hex << std::setw(8) << std::setfill('0') << args.key_id
			          << std::dec << std::setfill(' ') << "\n";
			print_precert_plan(precert_plan, args);
			std::cout << "  cuda_equiv    : ./test_cuda --device <n> --key_id 0x"
			          << std::hex << std::setw(8) << std::setfill('0') << args.key_id
			          << std::dec << std::setfill(' ')
			          << " --workunit_size " << N
			          << " --raceway-cap-mark"
			          << " --raceway-cap-ilp " << args.raceway_cap_ilp
			          << " --raceway-cap-bits " << args.raceway_cap_bits
			          << " --raceway-cap-ways " << args.raceway_cap_ways
			          << " --raceway-first-cap-map " << args.raceway_first_cap_map
			          << " --raceway-cap-count " << effective_cap_count << "\n";
			std::cout << "  first_cap_map : " << args.raceway_first_cap_map
			          << "  cap_count: " << effective_cap_count
			          << "  boundaries: " << (cap_maps_host.empty() ? "none" : boundary_list.str())
			          << "  cap: 2^" << args.raceway_cap_bits << " x " << args.raceway_cap_ways
			          << " x " << cap_table_count
			          << " (" << std::fixed << std::setprecision(1) << cap_mb << " MB total, "
			          << (raceway_cap_fp64 ? "fp64" : "fp32 fallback") << ")\n";
			std::cout << "  repeats       : warmup=" << args.warmup_batches
			          << " timed=" << args.raceway_bench_repeats
			          << " ilp=" << args.raceway_cap_ilp
			          << " groups=" << wg_count
			          << (args.raceway_persistent ? " persistent" : " static")
			          << " min/mean/max=" << std::fixed << std::setprecision(3)
			          << raceway_min_ms << "/" << raceway_mean_ms << "/" << raceway_max_ms << " ms\n";
			std::cout << "  mark kernel   : " << std::fixed << std::setprecision(3) << raceway_min_ms << " ms   "
			          << std::fixed << std::setprecision(0) << raceway_cps << " logical c/s";
			if (precert_plan.active)
			{
				std::cout << "  represented="
				          << std::fixed << std::setprecision(0)
				          << (static_cast<double>(args.workunit_size) / (raceway_min_ms / 1000.0)) << " c/s";
			}
			std::cout << "\n";
			std::cout << "  alive/dropped : " << alive_count << " / " << dropped_count << "\n";
			std::cout << "  est work      : map_evals=" << estimated_map_evals
			          << " (" << std::fixed << std::setprecision(1) << (map_rate / 1.0e6) << " M/s)"
			          << " cap_probes=" << estimated_cap_probes
			          << " (" << std::fixed << std::setprecision(1) << (probe_rate / 1.0e6) << " M/s)\n";
			if (bad_drop_count != 0u)
			{
				std::cout << "  invalid drop-map bytes: " << bad_drop_count << "\n";
			}
			std::cout << "  drop_hist     :";
			bool any_drop = false;
			for (std::size_t m = 0; m < drop_hist.size(); m++)
			{
				if (drop_hist[m] == 0u) continue;
				any_drop = true;
				std::cout << " m" << m << "=" << drop_hist[m];
			}
			if (!any_drop) std::cout << " none";
			std::cout << "\n";
			if (args.raceway_cap_bits == 0u && dropped_count == 0u)
			{
				std::cout << "  no-cap smoke  : PASS (all candidates alive)\n";
			}
			else if (args.raceway_cap_bits == 0u)
			{
				std::cout << "  no-cap smoke  : FAIL (" << dropped_count << " candidates dropped)\n";
			}

			if (!args.raceway_profile_csv_path.empty())
			{
				const bool new_file = !static_cast<bool>(std::ifstream(args.raceway_profile_csv_path.c_str()));
				std::ofstream csv(args.raceway_profile_csv_path.c_str(), std::ios::out | std::ios::app);
				if (!csv)
				{
					throw std::runtime_error("Could not open raceway profile CSV path: " + args.raceway_profile_csv_path);
				}
				if (new_file)
				{
					csv << "platform,device,key,range_start,workunit_size,cap_bits,cap_ways,cap_count,first_cap_map,cap_mode,cap_mb,warmup,repeats,ilp,groups,persistent,min_ms,mean_ms,max_ms,candidates_per_s,alive,dropped,map_evals,map_evals_per_s,cap_probes,cap_probes_per_s\n";
				}
				csv << platform_name << ","
				    << device_name << ","
				    << "0x" << std::hex << std::setw(8) << std::setfill('0') << args.key_id << std::dec << std::setfill(' ') << ","
				    << data_start << ","
				    << N << ","
				    << args.raceway_cap_bits << ","
				    << args.raceway_cap_ways << ","
				    << effective_cap_count << ","
				    << args.raceway_first_cap_map << ","
				    << (raceway_cap_fp64 ? "fp64" : "fp32") << ","
				    << std::fixed << std::setprecision(3) << cap_mb << ","
				    << args.warmup_batches << ","
				    << args.raceway_bench_repeats << ","
				    << args.raceway_cap_ilp << ","
				    << wg_count << ","
				    << (args.raceway_persistent ? 1 : 0) << ","
				    << std::fixed << std::setprecision(6) << raceway_min_ms << ","
				    << raceway_mean_ms << ","
				    << raceway_max_ms << ","
				    << std::fixed << std::setprecision(3) << raceway_cps << ","
				    << alive_count << ","
				    << dropped_count << ","
				    << estimated_map_evals << ","
				    << std::fixed << std::setprecision(3) << map_rate << ","
				    << estimated_cap_probes << ","
				    << std::fixed << std::setprecision(3) << probe_rate << "\n";
				std::cout << "  profile_csv   : " << args.raceway_profile_csv_path << "\n";
			}

			clReleaseMemObject(alive);
			clReleaseMemObject(drop_map);
			clReleaseMemObject(cap_table);
			clReleaseMemObject(cap_maps);
			clReleaseMemObject(work_counter);
			clReleaseMemObject(result_buffer);
			clReleaseMemObject(survivor_data_buffer);
			clReleaseMemObject(survivor_flag_buffer);
			clReleaseMemObject(materialized_state_buffer);
			release_assets(assets);
			if (span_kernel) clReleaseKernel(span_kernel);
			if (compact_kernel) clReleaseKernel(compact_kernel);
			if (resolve_kernel) clReleaseKernel(resolve_kernel);
			clReleaseKernel(raceway_cap_mark_kernel);
			clReleaseKernel(screen_ilp6_kernel);
			clReleaseKernel(materialize_kernel);
			clReleaseKernel(screen_kernel);
			clReleaseProgram(program);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 0;
		}

		// On-GPU survivor-compaction pipeline vs the ilp6 screen (parity + speedup).
		if (args.compaction_count > 0u)
		{
			uint32_t N = args.compaction_count;
			if (args.compaction_auto_tile)
			{
				// Per-candidate device bytes: state(128)+rep(4)+alive(1)+live_a(4)+live_b(4)
				// +flag(1)+comp(1)+ref(~1) = 144. Bound by VRAM budget AND single-buffer
				// max-alloc (state = N*128). Cap 128M (plateau + index safety + divides 2^32).
				const cl_ulong reserve = 256ull * 1024 * 1024;
				const cl_ulong budget = (dev_global_mem > reserve) ? (cl_ulong)((dev_global_mem - reserve) * 0.90) : 0;
				cl_ulong nmax = budget / 144ull;
				const cl_ulong nmax_alloc = dev_max_alloc / 128ull;  // state buffer limit
				if (nmax_alloc < nmax) nmax = nmax_alloc;
				uint32_t tile = 4u * 1024u * 1024u;
				while (((cl_ulong)tile << 1) <= nmax && (tile << 1) <= 134217728u) tile <<= 1;
				N = tile;
				std::cout << "  auto-tile: VRAM " << (dev_global_mem >> 20) << " MB, maxAlloc "
				          << (dev_max_alloc >> 20) << " MB -> tile " << (N >> 20) << "M candidates ("
				          << (((cl_ulong)N * 128) >> 20) << " MB state)\n";
			}
			const uint32_t data_start = args.range_start;
			const uint32_t SPAN_W = args.compaction_ilp;  // single-warp dedup window
			const size_t span_ls = 32u;                    // work-group size (single warp)
			cl_int e = CL_SUCCESS;

			// Reference: ilp6 screen into buf_ref (TDR-safe chunked launch over the tile).
			cl_mem buf_ref = clCreateBuffer(context, CL_MEM_READ_WRITE, ((N + 5u) / 6u) * 6u, nullptr, &e); throw_if_error(e, "alloc(ref)");
			set_ilp6_static_args(buf_ref);
			clFinish(queue);
			auto r0 = std::chrono::steady_clock::now();
			enqueue_ilp6_screen(buf_ref, args.key_id, data_start, N);
			clFinish(queue);
			const double ref_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r0).count();

			// Compaction buffers.
			cl_mem state      = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)N * 32 * sizeof(uint32_t), nullptr, &e); throw_if_error(e, "alloc(state)");
			cl_mem rep_global = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)N * sizeof(uint32_t), nullptr, &e); throw_if_error(e, "alloc(rep)");
			cl_mem alive      = clCreateBuffer(context, CL_MEM_READ_WRITE, N, nullptr, &e); throw_if_error(e, "alloc(alive)");
			cl_mem live_a     = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)N * sizeof(uint32_t), nullptr, &e); throw_if_error(e, "alloc(live_a)");
			cl_mem live_b     = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)N * sizeof(uint32_t), nullptr, &e); throw_if_error(e, "alloc(live_b)");
			cl_mem counter    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr, &e); throw_if_error(e, "alloc(counter)");
			cl_mem flag       = clCreateBuffer(context, CL_MEM_READ_WRITE, N, nullptr, &e); throw_if_error(e, "alloc(flag)");
			cl_mem buf_comp   = clCreateBuffer(context, CL_MEM_READ_WRITE, N, nullptr, &e); throw_if_error(e, "alloc(comp)");

			const uint32_t cuts[9] = { 0u,1u,5u,9u,13u,17u,21u,25u,27u };
			const uint32_t NSPANS = 8u;

			// Static span args.
			throw_if_error(clSetKernelArg(span_kernel, 2, sizeof(state), &state), "span.state");
			throw_if_error(clSetKernelArg(span_kernel, 3, sizeof(alive), &alive), "span.alive");
			throw_if_error(clSetKernelArg(span_kernel, 4, sizeof(rep_global), &rep_global), "span.rep");
			throw_if_error(clSetKernelArg(span_kernel, 8, sizeof(assets.offset_regular), &assets.offset_regular), "span.or");
			throw_if_error(clSetKernelArg(span_kernel, 9, sizeof(assets.offset_alg0), &assets.offset_alg0), "span.a0");
			throw_if_error(clSetKernelArg(span_kernel, 10, sizeof(assets.offset_alg6), &assets.offset_alg6), "span.a6");
			throw_if_error(clSetKernelArg(span_kernel, 11, sizeof(assets.offset_alg2), &assets.offset_alg2), "span.a2");
			throw_if_error(clSetKernelArg(span_kernel, 12, sizeof(assets.offset_alg5), &assets.offset_alg5), "span.a5");
			throw_if_error(clSetKernelArg(span_kernel, 13, sizeof(assets.expansion_values), &assets.expansion_values), "span.exp");
			throw_if_error(clSetKernelArg(span_kernel, 14, sizeof(assets.schedule_data), &assets.schedule_data), "span.sched");
			throw_if_error(clSetKernelArg(span_kernel, 15, sizeof(assets.carnival_data), &assets.carnival_data), "span.carn");
			throw_if_error(clSetKernelArg(span_kernel, 16, sizeof(flag), &flag), "span.flag");
			throw_if_error(clSetKernelArg(span_kernel, 17, sizeof(args.key_id), &args.key_id), "span.key");
			throw_if_error(clSetKernelArg(span_kernel, 18, sizeof(data_start), &data_start), "span.ds");
			throw_if_error(clSetKernelArg(compact_kernel, 4, sizeof(counter), &counter), "comp.counter");
			throw_if_error(clSetKernelArg(resolve_kernel, 0, sizeof(rep_global), &rep_global), "res.rep");
			throw_if_error(clSetKernelArg(resolve_kernel, 1, sizeof(flag), &flag), "res.flag");
			throw_if_error(clSetKernelArg(resolve_kernel, 2, sizeof(buf_comp), &buf_comp), "res.res");
			throw_if_error(clSetKernelArg(resolve_kernel, 3, sizeof(N), &N), "res.N");

			auto run_pipeline = [&]() {
				const uint32_t fill = 0xFFFFFFFFu;
				throw_if_error(clEnqueueFillBuffer(queue, rep_global, &fill, sizeof(fill), 0, (size_t)N * sizeof(uint32_t), 0, nullptr, nullptr), "fill(rep)");
				uint32_t M = N;
				cl_mem in_live = live_a, out_live = live_b;
				int first = 1;
				for (uint32_t sp = 0; sp < NSPANS; sp++)
				{
					uint32_t m0 = cuts[sp], m1 = cuts[sp + 1];
					int mode = (sp == NSPANS - 1u) ? 2 : 1;  // fused screen on last span
					throw_if_error(clSetKernelArg(span_kernel, 0, sizeof(in_live), &in_live), "span.live");
					throw_if_error(clSetKernelArg(span_kernel, 1, sizeof(M), &M), "span.M");
					throw_if_error(clSetKernelArg(span_kernel, 5, sizeof(m0), &m0), "span.m0");
					throw_if_error(clSetKernelArg(span_kernel, 6, sizeof(m1), &m1), "span.m1");
					throw_if_error(clSetKernelArg(span_kernel, 7, sizeof(first), &first), "span.first");
					throw_if_error(clSetKernelArg(span_kernel, 19, sizeof(mode), &mode), "span.mode");
					size_t wg = (M + SPAN_W - 1u) / SPAN_W;
					size_t sg[3] = { span_ls, wg, 1 };
					size_t sl[3] = { span_ls, 1, 1 };
					throw_if_error(clEnqueueNDRangeKernel(queue, span_kernel, 3, nullptr, sg, sl, 0, nullptr, nullptr), "enq(span)");
					if (mode == 1)
					{
						const uint32_t zero = 0u;
						throw_if_error(clEnqueueFillBuffer(queue, counter, &zero, sizeof(zero), 0, sizeof(zero), 0, nullptr, nullptr), "fill(counter)");
						throw_if_error(clSetKernelArg(compact_kernel, 0, sizeof(alive), &alive), "comp.alive");
						throw_if_error(clSetKernelArg(compact_kernel, 1, sizeof(in_live), &in_live), "comp.in");
						throw_if_error(clSetKernelArg(compact_kernel, 2, sizeof(M), &M), "comp.M");
						throw_if_error(clSetKernelArg(compact_kernel, 3, sizeof(out_live), &out_live), "comp.out");
						throw_if_error(clSetKernelArg(compact_kernel, 5, sizeof(first), &first), "comp.first");
						size_t cg = ((M + 255u) / 256u) * 256u, clsz = 256;
						throw_if_error(clEnqueueNDRangeKernel(queue, compact_kernel, 1, nullptr, &cg, &clsz, 0, nullptr, nullptr), "enq(compact)");
						clFinish(queue);
						throw_if_error(clEnqueueReadBuffer(queue, counter, CL_TRUE, 0, sizeof(M), &M, 0, nullptr, nullptr), "read(counter)");
						cl_mem tmp = in_live; in_live = out_live; out_live = tmp;
						first = 0;
					}
				}
				size_t rg = ((N + 255u) / 256u) * 256u, rl = 256;
				throw_if_error(clEnqueueNDRangeKernel(queue, resolve_kernel, 1, nullptr, &rg, &rl, 0, nullptr, nullptr), "enq(resolve)");
				clFinish(queue);
			};

			run_pipeline();  // warm-up
			auto c0 = std::chrono::steady_clock::now();
			run_pipeline();
			const double comp_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();

			std::vector<uint8_t> hr(((N + 5u) / 6u) * 6u), hc(N);
			throw_if_error(clEnqueueReadBuffer(queue, buf_ref, CL_TRUE, 0, hr.size(), hr.data(), 0, nullptr, nullptr), "read(ref)");
			throw_if_error(clEnqueueReadBuffer(queue, buf_comp, CL_TRUE, 0, N, hc.data(), 0, nullptr, nullptr), "read(comp)");
			std::size_t mism = 0, fm = 0;
			for (std::size_t i = 0; i < N; i++) if (hr[i] != hc[i]) { if (!mism) fm = i; mism++; }

			std::cout << "OpenCL compaction vs ilp6 screen: " << N << " candidates, key=0x"
			          << std::hex << std::setw(8) << std::setfill('0') << args.key_id << std::dec << std::setfill(' ') << "\n";
			std::cout << "  ilp6 screen : " << std::fixed << std::setprecision(3) << ref_ms << " ms   " << std::setprecision(0) << (double)N / (ref_ms / 1000.0) << " c/s\n";
			std::cout << "  compaction  : " << std::fixed << std::setprecision(3) << comp_ms << " ms   " << std::setprecision(0) << (double)N / (comp_ms / 1000.0) << " c/s\n";
			std::cout << "  speedup     : " << std::fixed << std::setprecision(3) << (ref_ms / comp_ms) << "x\n";
			if (mism == 0) std::cout << "  PASS - all " << N << " flags match the ilp6 screen\n";
			else std::cout << "  FAIL - " << mism << " mismatches; first at " << fm << " ref=0x" << std::hex << (unsigned)hr[fm] << " comp=0x" << (unsigned)hc[fm] << std::dec << "\n";

			clReleaseMemObject(buf_ref); clReleaseMemObject(state); clReleaseMemObject(rep_global);
			clReleaseMemObject(alive); clReleaseMemObject(live_a); clReleaseMemObject(live_b);
			clReleaseMemObject(counter); clReleaseMemObject(flag); clReleaseMemObject(buf_comp);
			release_assets(assets);
			if (raceway_cap_mark_kernel) clReleaseKernel(raceway_cap_mark_kernel);
			clReleaseKernel(span_kernel); clReleaseKernel(compact_kernel); clReleaseKernel(resolve_kernel);
			clReleaseKernel(screen_ilp6_kernel); clReleaseKernel(materialize_kernel); clReleaseKernel(screen_kernel);
			clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
			return 0;
		}

		// Parity + speedup check: run both screen kernels over N candidates and compare flag bytes.
		// Then exit (no full workunit processing).
		if (args.parity_count > 0u)
		{
			const uint32_t N = args.parity_count;
			cl_int s_p = CL_SUCCESS;
			cl_mem buf_ref   = clCreateBuffer(context, CL_MEM_READ_WRITE, N, nullptr, &s_p); throw_if_error(s_p, "clCreateBuffer(parity.ref)");
			cl_mem buf_ilp6  = clCreateBuffer(context, CL_MEM_READ_WRITE, ((N + 5u) / 6u) * 6u, nullptr, &s_p); throw_if_error(s_p, "clCreateBuffer(parity.ilp6)");

			// Baseline screen kernel
			throw_if_error(clSetKernelArg(screen_kernel, 0, sizeof(buf_ref), &buf_ref), "clSetKernelArg(parity.screen.result)");
			throw_if_error(clSetKernelArg(screen_kernel, 12, sizeof(args.range_start), &args.range_start), "clSetKernelArg(parity.screen.range_start)");

			size_t base_global[3] = { 32, N, 1 };
			size_t base_local[3]  = { 32, 1, 1 };
			cl_event base_event = nullptr;
			throw_if_error(clEnqueueNDRangeKernel(queue, screen_kernel, 3, nullptr, base_global, base_local, 0, nullptr, &base_event), "parity.enqueue(screen)");
			throw_if_error(clWaitForEvents(1, &base_event), "parity.wait(screen)");
			cl_ulong base_t0 = 0, base_t1 = 0;
			clGetEventProfilingInfo(base_event, CL_PROFILING_COMMAND_START, sizeof(base_t0), &base_t0, nullptr);
			clGetEventProfilingInfo(base_event, CL_PROFILING_COMMAND_END,   sizeof(base_t1), &base_t1, nullptr);
			clReleaseEvent(base_event);

			// ILP6 screen kernel
			set_ilp6_static_args(buf_ilp6);
			throw_if_error(clSetKernelArg(screen_ilp6_kernel, 10, sizeof(args.range_start), &args.range_start), "clSetKernelArg(parity.ilp6.range_start)");

			const size_t wg_count = (N + 5u) / 6u;
			size_t ilp6_global[3] = { 32, wg_count, 1 };
			size_t ilp6_local[3]  = { 32, 1, 1 };
			cl_event ilp6_event = nullptr;
			throw_if_error(clEnqueueNDRangeKernel(queue, screen_ilp6_kernel, 3, nullptr, ilp6_global, ilp6_local, 0, nullptr, &ilp6_event), "parity.enqueue(ilp6)");
			throw_if_error(clWaitForEvents(1, &ilp6_event), "parity.wait(ilp6)");
			cl_ulong ilp6_t0 = 0, ilp6_t1 = 0;
			clGetEventProfilingInfo(ilp6_event, CL_PROFILING_COMMAND_START, sizeof(ilp6_t0), &ilp6_t0, nullptr);
			clGetEventProfilingInfo(ilp6_event, CL_PROFILING_COMMAND_END,   sizeof(ilp6_t1), &ilp6_t1, nullptr);
			clReleaseEvent(ilp6_event);

			std::vector<uint8_t> host_ref(N), host_ilp6(((N + 5u) / 6u) * 6u);
			throw_if_error(clEnqueueReadBuffer(queue, buf_ref,  CL_TRUE, 0, N,                host_ref.data(),  0, nullptr, nullptr), "parity.read(ref)");
			throw_if_error(clEnqueueReadBuffer(queue, buf_ilp6, CL_TRUE, 0, host_ilp6.size(), host_ilp6.data(), 0, nullptr, nullptr), "parity.read(ilp6)");

			std::size_t mismatches = 0;
			std::size_t first_mismatch = 0;
			for (std::size_t i = 0; i < N; i++)
			{
				if (host_ref[i] != host_ilp6[i])
				{
					if (mismatches == 0) first_mismatch = i;
					mismatches++;
				}
			}

			const double base_ms = (base_t1 - base_t0) / 1.0e6;
			const double ilp6_ms = (ilp6_t1 - ilp6_t0) / 1.0e6;
			const double base_cps = (double)N / (base_ms / 1000.0);
			const double ilp6_cps = (double)N / (ilp6_ms / 1000.0);

			std::cout << "OpenCL screen parity vs ilp6: " << N << " candidates, key=0x"
			          << std::hex << std::setw(8) << std::setfill('0') << args.key_id
			          << std::dec << std::setfill(' ') << "\n";
			std::cout << "  baseline_screen : " << std::fixed << std::setprecision(3) << base_ms << " ms   "
			          << std::fixed << std::setprecision(0) << base_cps << " c/s\n";
			std::cout << "  offset_ilp6     : " << std::fixed << std::setprecision(3) << ilp6_ms << " ms   "
			          << std::fixed << std::setprecision(0) << ilp6_cps << " c/s\n";
			std::cout << "  speedup         : " << std::fixed << std::setprecision(3) << (base_ms / ilp6_ms) << "x\n";
			if (mismatches == 0)
			{
				std::cout << "  PASS - all " << N << " flag bytes match\n";
			}
			else
			{
				std::cout << "  FAIL - " << mismatches << "/" << N << " mismatches; first at candidate "
				          << first_mismatch << " ref=0x" << std::hex << (unsigned)host_ref[first_mismatch]
				          << " ilp6=0x" << (unsigned)host_ilp6[first_mismatch] << std::dec << "\n";
			}

			clReleaseMemObject(buf_ref);
			clReleaseMemObject(buf_ilp6);

			release_assets(assets);
			if (raceway_cap_mark_kernel) clReleaseKernel(raceway_cap_mark_kernel);
			clReleaseKernel(screen_ilp6_kernel);
			clReleaseKernel(materialize_kernel);
			clReleaseKernel(screen_kernel);
			clReleaseProgram(program);
			clReleaseCommandQueue(queue);
			clReleaseContext(context);
			return 0;
		}

		std::vector<uint8_t> host_results(args.batch_size, 0);
		std::vector<SurvivorRecord> survivors;
		const auto setup_end = std::chrono::high_resolution_clock::now();

		// If --ilp6 was requested, the active screen kernel is the ILP6 variant;
		// rebind result buffer (arg 0 already set for baseline) and use a
		// helper to dispatch with the right shape.
		cl_kernel active_screen_kernel = args.use_ilp6 ? screen_ilp6_kernel : screen_kernel;
		const uint32_t active_data_start_arg_index = args.use_ilp6 ? 10u : 12u;
		if (args.use_ilp6)
		{
			set_ilp6_static_args(result_buffer);
		}

		double warmup_seconds = 0.0;
		for (uint32_t i = 0; i < args.warmup_batches; i++)
		{
			const uint32_t warmup_start = args.range_start + (i * args.batch_size);
			throw_if_error(clSetKernelArg(active_screen_kernel, active_data_start_arg_index, sizeof(warmup_start), &warmup_start), "clSetKernelArg(warmup_range_start)");

			const std::size_t wg_y = args.use_ilp6 ? ((args.batch_size + 5u) / 6u) : args.batch_size;
			size_t global_item_size[3] = { 32, wg_y, 1 };
			size_t local_item_size[3] = { 32, 1, 1 };

			cl_event event = nullptr;
			const auto warmup_begin = std::chrono::high_resolution_clock::now();
			throw_if_error(clEnqueueNDRangeKernel(queue, active_screen_kernel, 3, nullptr, global_item_size, local_item_size, 0, nullptr, &event), "clEnqueueNDRangeKernel(warmup)");
			throw_if_error(clWaitForEvents(1, &event), "clWaitForEvents(warmup)");
			throw_if_error(clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0, args.batch_size, host_results.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer(warmup)");
			const auto warmup_end = std::chrono::high_resolution_clock::now();
			warmup_seconds += std::chrono::duration<double>(warmup_end - warmup_begin).count();
			clReleaseEvent(event);
		}

		uint64_t screen_kernel_nanoseconds = 0;
		uint64_t materialize_kernel_nanoseconds = 0;
		uint64_t carnival_hits = 0;
		uint64_t other_hits = 0;

		const auto run_begin = std::chrono::high_resolution_clock::now();
		for (uint32_t offset = 0; offset < args.workunit_size; offset += args.batch_size)
		{
			uint32_t chunk = args.workunit_size - offset;
			if (chunk > args.batch_size)
			{
				chunk = args.batch_size;
			}

			const uint32_t data_start = args.range_start + offset;
			throw_if_error(clSetKernelArg(active_screen_kernel, active_data_start_arg_index, sizeof(data_start), &data_start), "clSetKernelArg(data_start)");

			const std::size_t wg_y = args.use_ilp6 ? ((chunk + 5u) / 6u) : chunk;
			size_t global_item_size[3] = { 32, wg_y, 1 };
			size_t local_item_size[3] = { 32, 1, 1 };

			cl_event event = nullptr;
			throw_if_error(clEnqueueNDRangeKernel(queue, active_screen_kernel, 3, nullptr, global_item_size, local_item_size, 0, nullptr, &event), "clEnqueueNDRangeKernel");
			throw_if_error(clWaitForEvents(1, &event), "clWaitForEvents");
			throw_if_error(clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0, chunk, host_results.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");

			cl_ulong event_start = 0;
			cl_ulong event_end = 0;
			throw_if_error(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(event_start), &event_start, nullptr), "clGetEventProfilingInfo(start)");
			throw_if_error(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(event_end), &event_end, nullptr), "clGetEventProfilingInfo(end)");
			screen_kernel_nanoseconds += static_cast<uint64_t>(event_end - event_start);
			clReleaseEvent(event);

			for (uint32_t i = 0; i < chunk; i++)
			{
				const uint8_t screen_flags = host_results[i];
				if ((screen_flags & CHECKSUM_SENTINEL) == 0)
				{
					continue;
				}

				if ((screen_flags & OTHER_WORLD) != 0)
				{
					other_hits++;
				}
				else
				{
					carnival_hits++;
				}

				SurvivorRecord survivor;
				survivor.data = data_start + i;
				survivor.screen_flags = screen_flags;
				survivors.push_back(survivor);
			}
		}

		ValidationSummary validation_summary;
		double cpu_validation_seconds = 0.0;
		std::vector<uint8_t> debug_decrypted_state;
		uint32_t debug_data = 0;
		uint8_t debug_screen_flags = 0;
		uint8_t debug_machine_flags = 0;
		bool debug_other_world = false;
		bool debug_has_state = false;
		if (!survivors.empty())
		{
			std::vector<uint32_t> survivor_data_host(args.batch_size, 0);
			std::vector<uint8_t> survivor_flags_host(args.batch_size, 0);
			std::vector<uint8_t> decrypted_state_host(args.batch_size * 128, 0);

			for (std::size_t offset = 0; offset < survivors.size(); offset += args.batch_size)
			{
				std::size_t chunk = survivors.size() - offset;
				if (chunk > args.batch_size)
				{
					chunk = args.batch_size;
				}

				for (std::size_t i = 0; i < chunk; i++)
				{
					survivor_data_host[i] = survivors[offset + i].data;
					survivor_flags_host[i] = survivors[offset + i].screen_flags;
				}

				throw_if_error(clEnqueueWriteBuffer(queue, survivor_data_buffer, CL_TRUE, 0, chunk * sizeof(uint32_t), survivor_data_host.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer(survivor_data)");
				throw_if_error(clEnqueueWriteBuffer(queue, survivor_flag_buffer, CL_TRUE, 0, chunk, survivor_flags_host.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer(survivor_flags)");

				size_t global_item_size[3] = { 32, chunk, 1 };
				size_t local_item_size[3] = { 32, 1, 1 };
				cl_event event = nullptr;
				throw_if_error(clEnqueueNDRangeKernel(queue, materialize_kernel, 3, nullptr, global_item_size, local_item_size, 0, nullptr, &event), "clEnqueueNDRangeKernel(materialize)");
				throw_if_error(clWaitForEvents(1, &event), "clWaitForEvents(materialize)");
				throw_if_error(clEnqueueReadBuffer(queue, materialized_state_buffer, CL_TRUE, 0, chunk * 128, decrypted_state_host.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer(materialized_state)");

				cl_ulong event_start = 0;
				cl_ulong event_end = 0;
				throw_if_error(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(event_start), &event_start, nullptr), "clGetEventProfilingInfo(materialize.start)");
				throw_if_error(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(event_end), &event_end, nullptr), "clGetEventProfilingInfo(materialize.end)");
				materialize_kernel_nanoseconds += static_cast<uint64_t>(event_end - event_start);
				clReleaseEvent(event);

				const auto cpu_validation_begin = std::chrono::high_resolution_clock::now();
				for (std::size_t i = 0; i < chunk; i++)
				{
					const bool other_world = (survivor_flags_host[i] & OTHER_WORLD) != 0;
					const uint8_t machine_flags = check_machine_code(&decrypted_state_host[i * 128], other_world);
					survivors[offset + i].machine_flags = static_cast<uint8_t>(machine_flags | (other_world ? OTHER_WORLD : 0));
					accumulate_validation_summary(validation_summary, machine_flags, other_world);
					if (args.workunit_size == 1u && !debug_has_state)
					{
						debug_data = survivors[offset + i].data;
						debug_screen_flags = survivors[offset + i].screen_flags;
						debug_machine_flags = survivors[offset + i].machine_flags;
						debug_other_world = other_world;
						debug_decrypted_state.assign(&decrypted_state_host[i * 128], &decrypted_state_host[(i + 1) * 128]);
						debug_has_state = true;
					}
				}
				const auto cpu_validation_end = std::chrono::high_resolution_clock::now();
				cpu_validation_seconds += std::chrono::duration<double>(cpu_validation_end - cpu_validation_begin).count();
			}
		}
		const auto run_end = std::chrono::high_resolution_clock::now();

		const double setup_seconds = std::chrono::duration<double>(setup_end - setup_start).count();
		const double screen_kernel_seconds = static_cast<double>(screen_kernel_nanoseconds) / 1000000000.0;
		const double materialize_kernel_seconds = static_cast<double>(materialize_kernel_nanoseconds) / 1000000000.0;
		const double wall_seconds = std::chrono::duration<double>(run_end - run_begin).count();

		if (!args.output_csv_path.empty())
		{
			std::ofstream output_csv(args.output_csv_path.c_str(), std::ios::out | std::ios::trunc);
			if (!output_csv)
			{
				throw std::runtime_error("Could not open output CSV path: " + args.output_csv_path);
			}

			output_csv << "key,data,flags\n";
			for (std::size_t i = 0; i < survivors.size(); i++)
			{
				output_csv << args.key_id << ","
					<< survivors[i].data << ","
					<< static_cast<unsigned int>(survivors[i].machine_flags) << "\n";
			}
		}

		print_summary(args, platform_name, device_name, setup_seconds, warmup_seconds, screen_kernel_seconds, materialize_kernel_seconds, cpu_validation_seconds, wall_seconds, carnival_hits, other_hits, validation_summary);
		if (args.workunit_size == 1u)
		{
			if (debug_has_state)
			{
				const uint16_t checksum = calculate_checksum_from_decrypted(debug_decrypted_state.data(), debug_other_world);
				const uint16_t expected = fetch_checksum_value_from_decrypted(debug_decrypted_state.data(), debug_other_world);
				std::cout << "  debug_candidate_data: " << debug_data << "\n";
				std::cout << "  debug_candidate_world: " << (debug_other_world ? "other" : "carnival") << "\n";
				std::cout << "  debug_screen_flags: " << static_cast<unsigned int>(debug_screen_flags) << "\n";
				std::cout << "  debug_machine_flags: " << static_cast<unsigned int>(debug_machine_flags) << "\n";
				std::cout << "  debug_decrypted_hash: 0x" << std::hex << std::uppercase << hash_state(debug_decrypted_state.data()) << std::dec << std::nouppercase << "\n";
				std::cout << "  debug_checksum: " << checksum << "\n";
				std::cout << "  debug_expected: " << expected << "\n";
			}
			else
			{
				std::cout << "  debug_candidate: no checksum survivor\n";
			}
		}

		clReleaseMemObject(result_buffer);
		clReleaseMemObject(survivor_data_buffer);
		clReleaseMemObject(survivor_flag_buffer);
		clReleaseMemObject(materialized_state_buffer);
		release_assets(assets);
		if (raceway_cap_mark_kernel) clReleaseKernel(raceway_cap_mark_kernel);
		clReleaseKernel(screen_kernel);
		clReleaseKernel(materialize_kernel);
		clReleaseKernel(screen_ilp6_kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);

		return 0;
	}
	catch (const std::exception& ex)
	{
		std::cerr << ex.what() << "\n";
		return 1;
	}
}
