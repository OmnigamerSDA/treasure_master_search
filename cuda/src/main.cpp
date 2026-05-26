// Treasure Master — CUDA forward-search host
//
// Drives the CUDA checksum-screen kernel in `tm_cuda.cu`. For a fixed key,
// sweeps the 2^32 data axis in batches and reports per-key throughput,
// checksum survivors, and CPU validation flags.
//
// Build:    `make all`   (see Makefile in this directory)
// Invoke:   `./tm_cuda --device <id> --key_id 0x... --workunit_size <N>`
//
// See README.md and ../README.md for benchmark methodology and measured rates.

#include <cuda.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <unistd.h>
#include <climits>
#include <cmath>

#include "rng_obj.h"
#include "key_schedule.h"

// Build-time tuning knobs.
//
// These were hand-tuned for sm_120 (RTX 5090, RTX PRO 6000 Blackwell). On
// smaller / older GPUs the sweet spot is often different — lower occupancy
// headroom and smaller register files per SM tend to favor fewer warps per
// block and a lower ILP factor. See README "Tuning for other GPUs".
//
// Override via -D on the compile command line, e.g.:
//   make CXXFLAGS='... -DTM_WARPS_PER_BLOCK=2 -DTM_CANDIDATES_PER_WARP=2 ...'
#ifndef TM_WARPS_PER_BLOCK
#define TM_WARPS_PER_BLOCK 4u
#endif
#ifndef TM_CANDIDATES_PER_WARP
#define TM_CANDIDATES_PER_WARP 4u
#endif

namespace
{
	static const uint32_t kDefaultBatchSize = 1u << 20;
	static const uint32_t kCudaWarpSize = 32u;
	static const uint32_t kCudaWarpsPerBlock = TM_WARPS_PER_BLOCK;
	static const uint32_t kCudaThreadsPerBlock = kCudaWarpSize * kCudaWarpsPerBlock;
	static const uint32_t kCudaScreenCandidatesPerWarp = TM_CANDIDATES_PER_WARP;
	// ALL_MAPS: 26 base entries (map 0x22 gets a duplicate via build_schedule_blob → 27 total)
	static const uint8_t kMapList[26] = {
		0x00, 0x02, 0x05, 0x04, 0x03, 0x1D, 0x1C, 0x1E, 0x1B,
		0x07, 0x08, 0x06, 0x09, 0x0C, 0x20, 0x21, 0x22, 0x23,
		0x24, 0x25, 0x26, 0x0E, 0x0F, 0x10, 0x12, 0x11
	};
	// SKIP_CAR: 25 base entries (skips map 0x06; map 0x22 gets a duplicate → 26 total)
	static const uint8_t kMapListSkipCar[25] = {
		0x00, 0x02, 0x05, 0x04, 0x03, 0x1D, 0x1C, 0x1E, 0x1B,
		0x07, 0x08, 0x09, 0x0C, 0x20, 0x21, 0x22, 0x23,
		0x24, 0x25, 0x26, 0x0E, 0x0F, 0x10, 0x12, 0x11
	};

	static const uint8_t CHECKSUM_SENTINEL = 0x08;
	static const uint8_t OTHER_WORLD       = 0x01;
	static const uint8_t DUAL_PASS         = 0x02;  // set by CUDA when BOTH checksums pass
	static const uint8_t FIRST_ENTRY_VALID = 0x02;  // set by CPU validation (different byte: machine_flags)
	static const uint8_t ALL_ENTRIES_VALID = 0x04;
	static const uint8_t USES_NOP = 0x10;
	static const uint8_t USES_UNOFFICIAL_NOPS = 0x20;
	static const uint8_t USES_ILLEGAL_OPCODES = 0x40;
	static const uint8_t USES_JAM = 0x80;

	static const uint8_t OP_JAM = 0x01;
	static const uint8_t OP_ILLEGAL = 0x02;
	static const uint8_t OP_NOP2 = 0x04;
	static const uint8_t OP_NOP = 0x08;
	static const uint8_t OP_JUMP = 0x10;

	enum OtherPredicateIndex
	{
		OPRED_FINAL_RTS_50 = 0,
		OPRED_ENTRY00_LOAD,
		OPRED_ENTRY05_LOAD,
		OPRED_ENTRY0A_LOAD,
		OPRED_ENTRY28_LOAD,
		OPRED_ENTRY40_LOAD,
		OPRED_INIT_CALL_OR_JUMP_02,
		OPRED_INIT_JMP_02,
		OPRED_CALL_OR_JUMP_07,
		OPRED_JSR_07,
		OPRED_JUMP_25,
		OPRED_BRANCH_2D,
		OPRED_JSR_3A,
		OPRED_JUMP_3D,
		OPRED_BRANCH_45,
		OPRED_JUMP_4D,
		OPRED_INIT_JMP_8595,
		OPRED_ENTRY05_JSR_80B4,
		OPRED_JUMP_25_TARGET_8464,
		OPRED_JSR_3A_TARGET_8952,
		OPRED_JUMP_3D_TARGET_81EE,
		OPRED_JUMP_4D_TARGET_80B4,
		OTHER_PREDICATE_COUNT
	};

	static const char* kOtherPredicateNames[OTHER_PREDICATE_COUNT] = {
		"final_rts_50",
		"entry00_load",
		"entry05_load",
		"entry0a_load",
		"entry28_load",
		"entry40_load",
		"init_call_or_jump_02",
		"init_jmp_02",
		"call_or_jump_07",
		"jsr_07",
		"jump_25",
		"branch_2d",
		"jsr_3a",
		"jump_3d",
		"branch_45",
		"jump_4d",
		"init_jmp_8595",
		"entry05_jsr_80b4",
		"jump_25_target_8464",
		"jsr_3a_target_8952",
		"jump_3d_target_81ee",
		"jump_4d_target_80b4",
	};

	static const uint8_t kOpcodeBytesUsed[0x100] = {
		1,2,0,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0,
		3,2,0,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0,
		1,2,0,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0,
		1,2,0,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0,
		2,2,2,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,0,3,0,0,
		2,2,2,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,3,3,3,0,
		2,2,2,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0,
		2,2,2,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0
	};

	static const uint8_t kOpcodeType[0x100] = {
		0, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, OP_JUMP, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, OP_JUMP, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		OP_NOP2, 0, OP_NOP2, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, OP_NOP2, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, OP_ILLEGAL, 0, OP_ILLEGAL, OP_ILLEGAL,
		0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		0, 0, OP_NOP2, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		0, 0, OP_NOP2, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL
	};

	struct Args
	{
		uint32_t key_id = 0x2CA5B42Du;
		uint64_t range_start = 0u;
		uint64_t workunit_size = 1u << 20;
		uint32_t batch_size = kDefaultBatchSize;
		uint32_t warmup_batches = 1u;
		uint32_t device_index = 0u;
		uint32_t parity_count = 0u;   // 0 = no parity test
		std::string output_csv_path;
		std::string map_list = "all";  // "all" or "skip-car"
		// HLL cardinality estimation is off by default — it's a profiling/diagnostic
		// feature, costs ~10% throughput, and the production goal is maximum sweep
		// rate. Opt in with --hll.
		bool hll = false;
		// Use the offset-stream + ILP screen kernel (~1.5× faster than the
		// universal-table baseline on sm_120). Opt in with --screen-offsets.
		bool screen_offsets = false;
		// ILP factor for the offset-store screen kernel: 4, 6 (recommended), or 8.
		// ILP6 is the empirical sweet spot on Blackwell; ILP4 is slightly slower
		// but lighter-register; ILP8 regresses slightly from i-cache pressure.
		uint32_t ilp = 6u;
	};

	struct KernelAssets
	{
		CUdeviceptr regular_rng_values = 0;
		CUdeviceptr alg0_values = 0;
		CUdeviceptr alg6_values = 0;
		CUdeviceptr rng_seed_forward_1 = 0;
		CUdeviceptr rng_seed_forward_128 = 0;
		CUdeviceptr alg2_values = 0;
		CUdeviceptr alg5_values = 0;
		CUdeviceptr expansion_values = 0;
		CUdeviceptr schedule_data = 0;
		CUdeviceptr carnival_data = 0;
		// Offset-stream buffers (per-key, ~21.6 MB total). Allocated lazily
		// when --screen-offsets path is requested.
		CUdeviceptr offset_regular = 0;
		CUdeviceptr offset_alg0 = 0;
		CUdeviceptr offset_alg6 = 0;
		CUdeviceptr offset_alg2 = 0;
		CUdeviceptr offset_alg5 = 0;
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
		uint64_t dual_pass = 0;             // passed both carnival and other-world checksums
		uint64_t first_entry_valid = 0;
		uint64_t all_entries_valid = 0;
		uint64_t all_entries_valid_carnival = 0;
		uint64_t all_entries_valid_other    = 0;
		uint64_t uses_nop = 0;
		uint64_t uses_unofficial_nops = 0;
		uint64_t uses_illegal = 0;
		uint64_t uses_jam = 0;
		// Linear-stream scan: walk entire code region as instruction stream;
		// count survivors where stream completes with no JAM/ILLEGAL encountered,
		// and track the maximum score (opcode positions walked before first bad byte
		// or until end of stream if clean) as a gradient proximity signal.
		uint64_t linear_scan_clean_carnival    = 0;
		uint64_t linear_scan_clean_other       = 0;
		uint64_t linear_scan_score_max_carnival = 0;   // max opcode steps walked (carnival path)
		uint64_t linear_scan_score_max_other    = 0;   // max opcode steps walked (other-world path)
		uint64_t other_final_rts = 0;
		uint64_t other_entry_opcodes = 0;
		uint64_t other_control_flow = 0;
		uint64_t other_structural = 0;
		uint64_t all_entries_other_final_rts = 0;
		uint64_t all_entries_other_entry_opcodes = 0;
		uint64_t all_entries_other_control_flow = 0;
		uint64_t all_entries_other_structural = 0;
		uint64_t other_predicate[OTHER_PREDICATE_COUNT] = {};
		uint64_t final_rts_other_predicate[OTHER_PREDICATE_COUNT] = {};
		uint64_t all_entries_other_predicate[OTHER_PREDICATE_COUNT] = {};
		uint64_t all_entries_final_rts_other_predicate[OTHER_PREDICATE_COUNT] = {};
		// Hash-based collision analysis: distinct 128-byte decrypted states
		// (checksum-survivor population only — biased sample of full output space).
		uint64_t unique_states_carnival = 0;
		uint64_t unique_states_other    = 0;
		// HyperLogLog estimate of distinct output states across ALL 2^32 data
		// values — the true output-space cardinality, unaffected by checksum bias.
		uint64_t hll_distinct_states    = 0;
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
					args.range_start = numeric_arg<uint64_t>(argv[++i], "--range_start");
				}
				else if (arg == "--workunit_size" && i + 1 < argc)
				{
					args.workunit_size = numeric_arg<uint64_t>(argv[++i], "--workunit_size");
				}
			else if (arg == "--batch_size" && i + 1 < argc)
			{
				args.batch_size = numeric_arg<uint32_t>(argv[++i], "--batch_size");
			}
			else if (arg == "--warmup_batches" && i + 1 < argc)
			{
				args.warmup_batches = numeric_arg<uint32_t>(argv[++i], "--warmup_batches");
			}
			else if (arg == "--device" && i + 1 < argc)
			{
				args.device_index = numeric_arg<uint32_t>(argv[++i], "--device");
			}
			else if (arg == "--output_csv" && i + 1 < argc)
			{
				args.output_csv_path = argv[++i];
			}
			else if (arg == "--parity" && i + 1 < argc)
			{
				args.parity_count = numeric_arg<uint32_t>(argv[++i], "--parity");
			}
			else if (arg == "--map-list" && i + 1 < argc)
			{
				args.map_list = argv[++i];
				if (args.map_list != "all" && args.map_list != "skip-car")
				{
					throw std::runtime_error("--map-list must be 'all' or 'skip-car'");
				}
			}
			else if (arg == "--hll")
			{
				args.hll = true;
			}
			else if (arg == "--no-hll")
			{
				// Back-compat no-op (HLL is off by default now). Accepted silently
				// so older scripts/wrappers don't break.
				args.hll = false;
			}
			else if (arg == "--screen-offsets")
			{
				args.screen_offsets = true;
			}
			else if (arg == "--ilp" && i + 1 < argc)
			{
				args.ilp = numeric_arg<uint32_t>(argv[++i], "--ilp");
				if (args.ilp != 4u && args.ilp != 6u && args.ilp != 8u)
				{
					throw std::runtime_error("--ilp must be 4, 6, or 8");
				}
			}
			else if (arg == "--help" || arg == "-h")
			{
				std::cout
					<< "tm_cuda — CUDA forward-search for Treasure Master Bonus World 2\n"
					<< "\n"
					<< "Sweep options:\n"
					<< "  --key_id <uint32>           Key (4-byte) to fix while sweeping data (default 0x2CA5B42D)\n"
					<< "  --range_start <uint64>      First data index to scan (default 0)\n"
					<< "  --workunit_size <uint64>    Number of data indices to scan (default 2^20)\n"
					<< "  --batch_size <uint32>       Candidates per kernel launch (default 2^20)\n"
					<< "  --warmup_batches <uint32>   Warm-up launches before timing (default 1)\n"
					<< "  --device <index>            CUDA device index (default 0; see banner for resolved name)\n"
					<< "\n"
					<< "Screen-kernel selection:\n"
					<< "  --map-list all|skip-car     Key schedule variant (default: all)\n"
					<< "  --screen-offsets            Use the offset-stream + ILP screen kernel (~1.4x faster).\n"
					<< "                              Costs ~22 MB extra device memory per key.\n"
					<< "  --ilp 4|6|8                 ILP factor when --screen-offsets is set (default: 6)\n"
					<< "  --hll                       Enable HLL cardinality estimation (~10% slower; off by default)\n"
					<< "  --no-hll                    Disable HLL (back-compat no-op; HLL is off by default)\n"
					<< "\n"
					<< "Diagnostics:\n"
					<< "  --parity <count>            Run CPU-vs-GPU byte-level parity check on <count> candidates,\n"
					<< "                              then exit. Tests the BASELINE kernel; when combined with\n"
					<< "                              --screen-offsets, also flag-compares baseline vs offset-store.\n"
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
			if (args.range_start > static_cast<uint64_t>(UINT32_MAX))
			{
				throw std::runtime_error("--range_start must fit in uint32 for the CUDA kernel interface");
			}
			if (args.workunit_size > (1ull << 32))
			{
				throw std::runtime_error("--workunit_size must be at most 2^32 candidates");
			}
			if (args.range_start + args.workunit_size > (1ull << 32))
			{
				throw std::runtime_error("--range_start + --workunit_size exceeds the 32-bit candidate space");
			}

			return args;
		}

	void check_cuda(CUresult status, const char* what)
	{
		if (status == CUDA_SUCCESS)
		{
			return;
		}

		const char* error_name = nullptr;
		const char* error_string = nullptr;
		cuGetErrorName(status, &error_name);
		cuGetErrorString(status, &error_string);

		std::ostringstream message;
		message << what << " failed with CUDA status " << static_cast<int>(status);
		if (error_name != nullptr)
		{
			message << " (" << error_name << ")";
		}
		if (error_string != nullptr)
		{
			message << ": " << error_string;
		}
		throw std::runtime_error(message.str());
	}

	std::vector<uint8_t> read_binary_file(const std::string& path)
	{
		std::ifstream input(path, std::ios::binary);
		if (!input)
		{
			return {};
		}

		std::ostringstream buffer;
		buffer << input.rdbuf();
		const std::string data = buffer.str();
		return std::vector<uint8_t>(data.begin(), data.end());
	}

	std::string executable_dir()
	{
		char path[PATH_MAX] = {};
		const ssize_t length = readlink("/proc/self/exe", path, sizeof(path) - 1);
		if (length <= 0)
		{
			return {};
		}
		path[length] = '\0';

		std::string value(path, path + length);
		const std::size_t slash = value.find_last_of("/");
		if (slash == std::string::npos)
		{
			return {};
		}
		return value.substr(0, slash);
	}

	std::vector<uint8_t> load_module_blob()
	{
		std::vector<std::string> paths = {
			"tm_cuda.fatbin",
		};

		const std::string exe_dir = executable_dir();
		if (!exe_dir.empty())
		{
			paths.push_back(exe_dir + "/tm_cuda.fatbin");
		}

		for (const std::string& path : paths)
		{
			std::vector<uint8_t> blob = read_binary_file(path);
			if (!blob.empty())
			{
				return blob;
			}
		}

		throw std::runtime_error("Could not locate tm_cuda.fatbin "
			"(searched current directory and executable directory)");
	}

	std::vector<uint8_t> build_schedule_blob(uint32_t key, const std::string& map_list = "all")
	{
		const uint8_t* map_ptr = nullptr;
		std::size_t map_count = 0;
		if (map_list == "skip-car")
		{
			map_ptr   = kMapListSkipCar;
			map_count = sizeof(kMapListSkipCar) / sizeof(kMapListSkipCar[0]);
		}
		else
		{
			map_ptr   = kMapList;
			map_count = sizeof(kMapList) / sizeof(kMapList[0]);
		}

		key_schedule_data schedule_data = {};
		schedule_data.as_uint8[0] = static_cast<uint8_t>((key >> 24) & 0xFF);
		schedule_data.as_uint8[1] = static_cast<uint8_t>((key >> 16) & 0xFF);
		schedule_data.as_uint8[2] = static_cast<uint8_t>((key >> 8) & 0xFF);
		schedule_data.as_uint8[3] = static_cast<uint8_t>(key & 0xFF);

		// Maximum blob size: map_count + 1 (for 0x22 duplicate) entries × 4 bytes each
		std::vector<uint8_t> blob((map_count + 1) * 4, 0);
		int schedule_count = 0;
		for (std::size_t i = 0; i < map_count; i++)
		{
			key_schedule_entry entry = generate_schedule_entry(map_ptr[i], &schedule_data);
			blob[schedule_count * 4 + 0] = entry.rng1;
			blob[schedule_count * 4 + 1] = entry.rng2;
			blob[schedule_count * 4 + 2] = static_cast<uint8_t>((entry.nibble_selector >> 8) & 0xFF);
			blob[schedule_count * 4 + 3] = static_cast<uint8_t>(entry.nibble_selector & 0xFF);
			schedule_count++;

			if (map_ptr[i] == 0x22)
			{
				entry = generate_schedule_entry(map_ptr[i], &schedule_data, 4);
				blob[schedule_count * 4 + 0] = entry.rng1;
				blob[schedule_count * 4 + 1] = entry.rng2;
				blob[schedule_count * 4 + 2] = static_cast<uint8_t>((entry.nibble_selector >> 8) & 0xFF);
				blob[schedule_count * 4 + 3] = static_cast<uint8_t>(entry.nibble_selector & 0xFF);
				schedule_count++;
			}
		}

		blob.resize(static_cast<std::size_t>(schedule_count) * 4);
		return blob;
	}

	// Run one algorithm step using host-side RNG tables (mirrors the CUDA kernel logic exactly)
	void cpu_run_alg(uint8_t alg_id, uint16_t* rng_seed, uint8_t* state)
	{
		if (alg_id == 0)
		{
			const uint8_t* tbl = RNG::alg0_values_8 + static_cast<uint32_t>(*rng_seed) * 128u;
			for (int i = 0; i < 128; i++)
				state[i] = static_cast<uint8_t>((state[i] << 1) | tbl[i]);
			*rng_seed = RNG::seed_forward_128[*rng_seed];
		}
		else if (alg_id == 1)
		{
			const uint8_t* tbl = RNG::regular_rng_values_8 + static_cast<uint32_t>(*rng_seed) * 128u;
			for (int i = 0; i < 128; i++)
				state[i] = static_cast<uint8_t>(state[i] + tbl[i]);
			*rng_seed = RNG::seed_forward_128[*rng_seed];
		}
		else if (alg_id == 2)
		{
			// CUDA uses alg2_values_32_8; CPU equivalent: initial carry = (alg2_values_32_8[seed] >> 24) & 1
			uint8_t carry = static_cast<uint8_t>((RNG::alg2_values_32_8[*rng_seed] >> 24) & 0x01u);
			for (int i = 127; i >= 0; i -= 2)
			{
				uint8_t next_carry = state[i - 1] & 0x01u;
				state[i - 1] = static_cast<uint8_t>((state[i - 1] >> 1) | (state[i] & 0x80u));
				state[i]     = static_cast<uint8_t>((state[i] << 1) | (carry & 0x01u));
				carry = next_carry;
			}
			*rng_seed = RNG::seed_forward_1[*rng_seed];
		}
		else if (alg_id == 3)
		{
			const uint8_t* tbl = RNG::regular_rng_values_8 + static_cast<uint32_t>(*rng_seed) * 128u;
			for (int i = 0; i < 128; i++)
				state[i] ^= tbl[i];
			*rng_seed = RNG::seed_forward_128[*rng_seed];
		}
		else if (alg_id == 4)
		{
			// CUDA: vsub4(value, regular_rng_values[seed][lane])
			const uint8_t* tbl = RNG::regular_rng_values_8 + static_cast<uint32_t>(*rng_seed) * 128u;
			for (int i = 0; i < 128; i++)
				state[i] = static_cast<uint8_t>(state[i] - tbl[i]);
			*rng_seed = RNG::seed_forward_128[*rng_seed];
		}
		else if (alg_id == 5)
		{
			// CUDA uses alg5_values_32_8; initial carry = (alg5_values_32_8[seed] >> 24) & 0x80
			uint8_t carry = static_cast<uint8_t>((RNG::alg5_values_32_8[*rng_seed] >> 24) & 0x80u);
			for (int i = 127; i >= 0; i -= 2)
			{
				uint8_t next_carry = state[i - 1] & 0x80u;
				state[i - 1] = static_cast<uint8_t>((state[i - 1] << 1) | (state[i] & 0x01u));
				state[i]     = static_cast<uint8_t>((state[i] >> 1) | carry);
				carry = next_carry;
			}
			*rng_seed = RNG::seed_forward_1[*rng_seed];
		}
		else if (alg_id == 6)
		{
			const uint8_t* tbl = RNG::alg6_values_8 + static_cast<uint32_t>(*rng_seed) * 128u;
			for (int i = 0; i < 128; i++)
				state[i] = static_cast<uint8_t>((state[i] >> 1) | tbl[i]);
			*rng_seed = RNG::seed_forward_128[*rng_seed];
		}
		else  // alg 7
		{
			for (int i = 0; i < 128; i++)
				state[i] ^= 0xFFu;
			// no rng_seed advance for alg 7
		}
	}

	// CPU forward: expand + run schedule, writes 128-byte post-schedule state to out[]
	void cpu_forward(uint32_t key, uint32_t data,
	                 const std::vector<uint8_t>& schedule_blob,
	                 uint8_t* out)
	{
		// Expand: interleave key/data bytes x16
		const uint16_t expansion_seed = static_cast<uint16_t>(key >> 16);
		for (int i = 0; i < 128; i += 8)
		{
			out[i+0] = static_cast<uint8_t>((key >> 24) & 0xFF);
			out[i+1] = static_cast<uint8_t>((key >> 16) & 0xFF);
			out[i+2] = static_cast<uint8_t>((key >>  8) & 0xFF);
			out[i+3] = static_cast<uint8_t>( key        & 0xFF);
			out[i+4] = static_cast<uint8_t>((data >> 24) & 0xFF);
			out[i+5] = static_cast<uint8_t>((data >> 16) & 0xFF);
			out[i+6] = static_cast<uint8_t>((data >>  8) & 0xFF);
			out[i+7] = static_cast<uint8_t>( data        & 0xFF);
		}
		const uint8_t* exp_tbl = RNG::expansion_values_8 + static_cast<uint32_t>(expansion_seed) * 128u;
		for (int i = 0; i < 128; i++)
			out[i] = static_cast<uint8_t>(out[i] + exp_tbl[i]);

		// Run schedule entries (count derived from blob size)
		const int schedule_count_cpu = static_cast<int>(schedule_blob.size() / 4);
		for (int s = 0; s < schedule_count_cpu; s++)
		{
			uint16_t rng_seed = static_cast<uint16_t>(
				(static_cast<uint16_t>(schedule_blob[s * 4 + 0]) << 8) |
				 static_cast<uint16_t>(schedule_blob[s * 4 + 1]));
			uint16_t nibble_selector = static_cast<uint16_t>(
				(static_cast<uint16_t>(schedule_blob[s * 4 + 2]) << 8) |
				 static_cast<uint16_t>(schedule_blob[s * 4 + 3]));

			for (int i = 0; i < 16; i++)
			{
				const uint8_t nibble = static_cast<uint8_t>((nibble_selector >> 15) & 0x01u);
				nibble_selector = static_cast<uint16_t>(nibble_selector << 1);

				uint8_t cur_byte = out[i];
				if (nibble != 0u)
					cur_byte = static_cast<uint8_t>(cur_byte >> 4);
				const uint8_t alg_id = static_cast<uint8_t>((cur_byte >> 1) & 0x07u);

				cpu_run_alg(alg_id, &rng_seed, out);
			}
		}
	}

	CUdeviceptr upload_buffer(const void* data, std::size_t size)
	{
		CUdeviceptr buffer = 0;
		check_cuda(cuMemAlloc(&buffer, size), "cuMemAlloc");
		check_cuda(cuMemcpyHtoD(buffer, data, size), "cuMemcpyHtoD");
		return buffer;
	}

	// Builds per-key offset-stream buffers consumed by the
	// tm_checksum_screen_offset_store_ilp{4,6,8}_cuda kernels. Layout
	// per schedule step m, position pos:
	//   regular_stream[m*2048+pos][0..127] = RNG::regular_rng_values_8[seed*128 + 0..127]
	//   alg0_stream   [m*2048+pos][0..127] = RNG::alg0_values_8[seed*128 + 0..127]
	//   alg6_stream   [m*2048+pos][0..127] = RNG::alg6_values_8[seed*128 + 0..127]
	//   alg2_stream   [m*2048+pos]          = RNG::alg2_values_32_8[seed]
	//   alg5_stream   [m*2048+pos]          = RNG::alg5_values_32_8[seed]
	//   seed = rng_table[seed]   (walks once per pos)
	// Total per key (27 entries): ~21.6 MB.
	void build_offset_assets(uint32_t /*key*/, const std::vector<uint8_t>& schedule_blob, KernelAssets& assets)
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
				std::memcpy(regular_stream.data() + stream_offset, RNG::regular_rng_values_8 + static_cast<std::size_t>(seed) * 128ULL, 128ULL);
				std::memcpy(alg0_stream.data()    + stream_offset, RNG::alg0_values_8        + static_cast<std::size_t>(seed) * 128ULL, 128ULL);
				std::memcpy(alg6_stream.data()    + stream_offset, RNG::alg6_values_8        + static_cast<std::size_t>(seed) * 128ULL, 128ULL);
				const std::size_t carry_offset = m * 2048ULL + pos;
				alg2_stream[carry_offset] = RNG::alg2_values_32_8[seed];
				alg5_stream[carry_offset] = RNG::alg5_values_32_8[seed];
				seed = RNG::rng_table[seed];
			}
		}

		assets.offset_regular = upload_buffer(regular_stream.data(), regular_stream.size());
		assets.offset_alg0    = upload_buffer(alg0_stream.data(),    alg0_stream.size());
		assets.offset_alg6    = upload_buffer(alg6_stream.data(),    alg6_stream.size());
		assets.offset_alg2    = upload_buffer(alg2_stream.data(),    alg2_stream.size() * sizeof(uint32_t));
		assets.offset_alg5    = upload_buffer(alg5_stream.data(),    alg5_stream.size() * sizeof(uint32_t));
	}

	KernelAssets build_kernel_assets(uint32_t key, const std::string& map_list, bool with_offset_streams = false)
	{
		KernelAssets assets;
		RNG rng;
		rng.generate_rng_table();
		rng.generate_regular_rng_values_8();
		rng.generate_alg0_values_8();
		rng.generate_alg6_values_8();
		rng.generate_seed_forward_1();
		rng.generate_seed_forward_128();
		rng.generate_alg2_values_32_8();
		rng.generate_alg5_values_32_8();
		rng.generate_expansion_values_8();

		assets.regular_rng_values = upload_buffer(RNG::regular_rng_values_8, 0x10000ull * 128ull);
		assets.alg0_values = upload_buffer(RNG::alg0_values_8, 0x10000ull * 128ull);
		assets.alg6_values = upload_buffer(RNG::alg6_values_8, 0x10000ull * 128ull);
		assets.rng_seed_forward_1 = upload_buffer(RNG::seed_forward_1, 0x10000ull * sizeof(uint16_t));
		assets.rng_seed_forward_128 = upload_buffer(RNG::seed_forward_128, 0x10000ull * sizeof(uint16_t));
		assets.alg2_values = upload_buffer(RNG::alg2_values_32_8, 0x10000ull * sizeof(uint32_t));
		assets.alg5_values = upload_buffer(RNG::alg5_values_32_8, 0x10000ull * sizeof(uint32_t));
		assets.expansion_values = upload_buffer(RNG::expansion_values_8, 0x10000ull * 128ull);

		const std::vector<uint8_t> schedule_blob = build_schedule_blob(key, map_list);
		assets.schedule_data = upload_buffer(schedule_blob.data(), schedule_blob.size());

		if (with_offset_streams)
		{
			build_offset_assets(key, schedule_blob, assets);
		}

		static const uint8_t carnival_data[128] = {
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
		assets.carnival_data = upload_buffer(carnival_data, sizeof(carnival_data));

		return assets;
	}

	void release_assets(KernelAssets& assets)
	{
		if (assets.regular_rng_values != 0) cuMemFree(assets.regular_rng_values);
		if (assets.alg0_values != 0) cuMemFree(assets.alg0_values);
		if (assets.alg6_values != 0) cuMemFree(assets.alg6_values);
		if (assets.rng_seed_forward_1 != 0) cuMemFree(assets.rng_seed_forward_1);
		if (assets.rng_seed_forward_128 != 0) cuMemFree(assets.rng_seed_forward_128);
		if (assets.alg2_values != 0) cuMemFree(assets.alg2_values);
		if (assets.alg5_values != 0) cuMemFree(assets.alg5_values);
		if (assets.expansion_values != 0) cuMemFree(assets.expansion_values);
		if (assets.schedule_data != 0) cuMemFree(assets.schedule_data);
		if (assets.carnival_data != 0) cuMemFree(assets.carnival_data);
		if (assets.offset_regular != 0) cuMemFree(assets.offset_regular);
		if (assets.offset_alg0 != 0)    cuMemFree(assets.offset_alg0);
		if (assets.offset_alg6 != 0)    cuMemFree(assets.offset_alg6);
		if (assets.offset_alg2 != 0)    cuMemFree(assets.offset_alg2);
		if (assets.offset_alg5 != 0)    cuMemFree(assets.offset_alg5);
	}

	int reverse_offset(int offset)
	{
		return 127 - offset;
	}

	uint8_t code_byte_at(const uint8_t* data, int offset)
	{
		return data[reverse_offset(offset)];
	}

	bool is_load_family(uint8_t value)
	{
		switch (value)
		{
			case 0xA0: case 0xA1: case 0xA2: case 0xA4: case 0xA5: case 0xA6:
			case 0xA9: case 0xAC: case 0xAD: case 0xAE: case 0xB1: case 0xB4:
			case 0xB5: case 0xB6: case 0xB9: case 0xBC: case 0xBD: case 0xBE:
				return true;
			default:
				return false;
		}
	}

	bool is_branch(uint8_t value)
	{
		switch (value)
		{
			case 0x10: case 0x30: case 0x50: case 0x70:
			case 0x90: case 0xB0: case 0xD0: case 0xF0:
				return true;
			default:
				return false;
		}
	}

	bool is_call_or_jump(uint8_t value)
	{
		return value == 0x20 || value == 0x4C;
	}

	bool bytes_at(const uint8_t* data, int offset, uint8_t a, uint8_t b, uint8_t c)
	{
		return code_byte_at(data, offset) == a &&
			code_byte_at(data, offset + 1) == b &&
			code_byte_at(data, offset + 2) == c;
	}

	struct OtherPredicateResults
	{
		bool values[OTHER_PREDICATE_COUNT] = {};
	};

	OtherPredicateResults evaluate_other_predicates(const uint8_t* data)
	{
		OtherPredicateResults result;
		result.values[OPRED_FINAL_RTS_50] = code_byte_at(data, 0x50) == 0x60;
		result.values[OPRED_ENTRY00_LOAD] = is_load_family(code_byte_at(data, 0x00));
		result.values[OPRED_ENTRY05_LOAD] = is_load_family(code_byte_at(data, 0x05));
		result.values[OPRED_ENTRY0A_LOAD] = is_load_family(code_byte_at(data, 0x0A));
		result.values[OPRED_ENTRY28_LOAD] = is_load_family(code_byte_at(data, 0x28));
		result.values[OPRED_ENTRY40_LOAD] = is_load_family(code_byte_at(data, 0x40));
		result.values[OPRED_INIT_CALL_OR_JUMP_02] = is_call_or_jump(code_byte_at(data, 0x02));
		result.values[OPRED_INIT_JMP_02] = code_byte_at(data, 0x02) == 0x4C;
		result.values[OPRED_CALL_OR_JUMP_07] = is_call_or_jump(code_byte_at(data, 0x07));
		result.values[OPRED_JSR_07] = code_byte_at(data, 0x07) == 0x20;
		result.values[OPRED_JUMP_25] = code_byte_at(data, 0x25) == 0x4C;
		result.values[OPRED_BRANCH_2D] = is_branch(code_byte_at(data, 0x2D));
		result.values[OPRED_JSR_3A] = code_byte_at(data, 0x3A) == 0x20;
		result.values[OPRED_JUMP_3D] = code_byte_at(data, 0x3D) == 0x4C;
		result.values[OPRED_BRANCH_45] = is_branch(code_byte_at(data, 0x45));
		result.values[OPRED_JUMP_4D] = code_byte_at(data, 0x4D) == 0x4C;
		result.values[OPRED_INIT_JMP_8595] = bytes_at(data, 0x02, 0x4C, 0x95, 0x85);
		result.values[OPRED_ENTRY05_JSR_80B4] = bytes_at(data, 0x07, 0x20, 0xB4, 0x80);
		result.values[OPRED_JUMP_25_TARGET_8464] = bytes_at(data, 0x25, 0x4C, 0x64, 0x84);
		result.values[OPRED_JSR_3A_TARGET_8952] = bytes_at(data, 0x3A, 0x20, 0x52, 0x89);
		result.values[OPRED_JUMP_3D_TARGET_81EE] = bytes_at(data, 0x3D, 0x4C, 0xEE, 0x81);
		result.values[OPRED_JUMP_4D_TARGET_80B4] = bytes_at(data, 0x4D, 0x4C, 0xB4, 0x80);
		return result;
	}

	bool other_final_rts_shape(const uint8_t* data)
	{
		return code_byte_at(data, 0x50) == 0x60;
	}

	bool other_entry_opcode_shape(const uint8_t* data)
	{
		return other_final_rts_shape(data)
			&& is_load_family(code_byte_at(data, 0x00))
			&& is_load_family(code_byte_at(data, 0x05))
			&& is_load_family(code_byte_at(data, 0x0A))
			&& is_load_family(code_byte_at(data, 0x28))
			&& is_load_family(code_byte_at(data, 0x40));
	}

	bool other_control_flow_shape(const uint8_t* data)
	{
		return other_final_rts_shape(data)
			&& is_call_or_jump(code_byte_at(data, 0x02))
			&& is_call_or_jump(code_byte_at(data, 0x07))
			&& code_byte_at(data, 0x25) == 0x4C
			&& is_branch(code_byte_at(data, 0x2D))
			&& code_byte_at(data, 0x3A) == 0x20
			&& code_byte_at(data, 0x3D) == 0x4C
			&& is_branch(code_byte_at(data, 0x45))
			&& code_byte_at(data, 0x4D) == 0x4C;
	}

	bool other_structural_shape(const uint8_t* data)
	{
		return other_entry_opcode_shape(data) && other_control_flow_shape(data);
	}

	uint8_t check_machine_code(const uint8_t* data, bool other_world)
	{
		const uint8_t code_length = other_world ? 0x53 : 0x72;
		uint8_t entry_addrs[7];
		int entry_count = 0;
		if (!other_world)
		{
			entry_count = 4;
			entry_addrs[0] = 0x00;
			entry_addrs[1] = 0x2B;
			entry_addrs[2] = 0x33;
			entry_addrs[3] = 0x3E;
			entry_addrs[4] = 0xFF;
			entry_addrs[5] = 0xFF;
			entry_addrs[6] = 0xFF;
		}
		else
		{
			entry_count = 6;
			entry_addrs[0] = 0x00;
			entry_addrs[1] = 0x05;
			entry_addrs[2] = 0x0A;
			entry_addrs[3] = 0x28;
			entry_addrs[4] = 0x40;
			entry_addrs[5] = 0x50;
			entry_addrs[6] = 0xFF;
		}

		uint8_t active_entries[7] = { 0, 0, 0, 0, 0, 0, 0 };
		uint8_t hit_entries[7] = { 0, 0, 0, 0, 0, 0, 0 };
		uint8_t valid_entries[7] = { 0, 0, 0, 0, 0, 0, 0 };
		int last_entry = -1;
		uint8_t result = 0;
		uint8_t next_entry_addr = entry_addrs[0];

		for (int i = 0; i < code_length - 2; i++)
		{
			if (i == next_entry_addr)
			{
				last_entry++;
				hit_entries[last_entry] = 1;
				active_entries[last_entry] = 1;
				next_entry_addr = entry_addrs[last_entry + 1];
			}
			else if (i > next_entry_addr)
			{
				last_entry++;
				next_entry_addr = entry_addrs[last_entry + 1];
			}

			const uint8_t opcode = data[reverse_offset(i)];
			if (kOpcodeType[opcode] & OP_JAM)
			{
				result |= USES_JAM;
				break;
			}
			else if (kOpcodeType[opcode] & OP_ILLEGAL)
			{
				result |= USES_ILLEGAL_OPCODES;
				break;
			}
			else if (kOpcodeType[opcode] & OP_NOP2)
			{
				result |= USES_UNOFFICIAL_NOPS;
			}
			else if (kOpcodeType[opcode] & OP_NOP)
			{
				result |= USES_NOP;
			}
			else if (kOpcodeType[opcode] & OP_JUMP)
			{
				for (int j = 0; j < entry_count; j++)
				{
					if (active_entries[j] == 1)
					{
						active_entries[j] = 0;
						valid_entries[j] = 1;
					}
				}
			}

			i += kOpcodeBytesUsed[opcode] - 1;
		}

		bool all_entries_valid = true;
		for (int i = 0; i < entry_count; i++)
		{
			if (hit_entries[i] == 1)
			{
				if (valid_entries[i] != 1)
				{
					all_entries_valid = false;
					break;
				}
			}
			else if (entry_addrs[i] != 0xFF)
			{
				for (int j = entry_addrs[i]; j < code_length - 2; j++)
				{
					const uint8_t opcode = data[reverse_offset(j)];
					if ((kOpcodeType[opcode] & OP_JAM) || (kOpcodeType[opcode] & OP_ILLEGAL))
					{
						all_entries_valid = false;
						break;
					}
					else if (kOpcodeType[opcode] & OP_JUMP)
					{
						break;
					}

					j += kOpcodeBytesUsed[opcode] - 1;
				}
			}
		}

		if (all_entries_valid)
		{
			result |= ALL_ENTRIES_VALID;
		}
		if (valid_entries[0] == 1)
		{
			result |= FIRST_ENTRY_VALID;
		}

		return result;
	}

	// Standard HyperLogLog cardinality estimator for m = 4096 registers.
	// Input: 4096 uint32_t registers, each holding the max rho seen for that bucket.
	// Output: estimated number of distinct elements.
	uint64_t hll_estimate(const std::vector<uint32_t>& regs)
	{
		static const int m = 4096;
		static const double alpha_m = 0.7213 / (1.0 + 1.079 / m);

		double sum = 0.0;
		for (int i = 0; i < m; i++)
		{
			sum += std::pow(2.0, -(double)regs[i]);
		}
		double estimate = alpha_m * (double)m * (double)m / sum;

		// Small-range correction (LinearCounting) — use when estimate ≤ 2.5 m
		if (estimate <= 2.5 * m)
		{
			int zeros = 0;
			for (int i = 0; i < m; i++) if (regs[i] == 0) zeros++;
			if (zeros > 0)
			{
				estimate = (double)m * std::log((double)m / (double)zeros);
			}
		}
		// Large-range correction — use when estimate > 2^32 / 30
		const double two32 = 4294967296.0;
		if (estimate > two32 / 30.0)
		{
			estimate = -two32 * std::log(1.0 - estimate / two32);
		}
		return static_cast<uint64_t>(estimate + 0.5);
	}

	// Walk the entire code region as a linear instruction stream (multi-byte aware).
	// Returns true if the stream completes without encountering any JAM or ILLEGAL opcode.
	// Operand bytes are skipped according to instruction width, so only actual opcode
	// positions are tested — this is stricter than a raw per-byte scan and avoids
	// false positives from data bytes that happen to be invalid opcode values.
	bool linear_scan_clean(const uint8_t* data, bool other_world)
	{
		const int code_length = other_world ? 0x53 : 0x72;
		for (int i = 0; i < code_length - 2; )  // last 2 bytes are the embedded checksum
		{
			const uint8_t opcode = data[reverse_offset(i)];
			if ((kOpcodeType[opcode] & OP_JAM) || (kOpcodeType[opcode] & OP_ILLEGAL))
			{
				return false;
			}
			const int width = static_cast<int>(kOpcodeBytesUsed[opcode]);
			i += (width > 0) ? width : 1;  // safety: advance at least 1 even for 0-width entries
		}
		return true;
	}

	// Count opcode positions walked before hitting JAM/ILLEGAL, or total if clean.
	// Provides a gradient: a score near the stream length means almost-valid code;
	// a score of 0 means the very first byte is already a bad opcode.
	int linear_scan_score(const uint8_t* data, bool other_world)
	{
		const int code_length = other_world ? 0x53 : 0x72;
		int count = 0;
		for (int i = 0; i < code_length - 2; )
		{
			const uint8_t opcode = data[reverse_offset(i)];
			if ((kOpcodeType[opcode] & OP_JAM) || (kOpcodeType[opcode] & OP_ILLEGAL))
			{
				return count;
			}
			count++;
			const int width = static_cast<int>(kOpcodeBytesUsed[opcode]);
			i += (width > 0) ? width : 1;
		}
		return count;  // fully clean
	}

	// FNV-1a 64-bit hash over the full 128-byte decrypted state.
	uint64_t hash_state(const uint8_t* data)
	{
		uint64_t h = 14695981039346656037ULL;
		for (int i = 0; i < 128; i++)
		{
			h ^= data[i];
			h *= 1099511628211ULL;
		}
		return h;
	}

	void accumulate_validation_summary(ValidationSummary& summary, uint8_t machine_flags, bool other_world)
	{
		summary.total++;
		if (other_world)
		{
			summary.other++;
		}
		else
		{
			summary.carnival++;
		}
		if ((machine_flags & FIRST_ENTRY_VALID) != 0)
		{
			summary.first_entry_valid++;
		}
		if ((machine_flags & ALL_ENTRIES_VALID) != 0)
		{
			summary.all_entries_valid++;
			if (other_world) summary.all_entries_valid_other++;
			else             summary.all_entries_valid_carnival++;
		}
		if ((machine_flags & USES_NOP) != 0)
		{
			summary.uses_nop++;
		}
		if ((machine_flags & USES_UNOFFICIAL_NOPS) != 0)
		{
			summary.uses_unofficial_nops++;
		}
		if ((machine_flags & USES_ILLEGAL_OPCODES) != 0)
		{
			summary.uses_illegal++;
		}
		if ((machine_flags & USES_JAM) != 0)
		{
			summary.uses_jam++;
		}
	}

	void print_summary(
		const Args& args,
		const std::string& device_name,
		double setup_seconds,
		double warmup_seconds,
		double screen_kernel_seconds,
		double materialize_kernel_seconds,
		double validation_seconds,
		double wall_seconds,
		uint64_t carnival_hits,
		uint64_t other_hits,
		uint64_t dual_hits,
		const ValidationSummary& validation_summary)
	{
		const uint64_t total_hits = carnival_hits + other_hits;
		const double candidate_count = static_cast<double>(args.workunit_size);
		const double kernel_rate = screen_kernel_seconds == 0.0 ? 0.0 : candidate_count / screen_kernel_seconds;
		const double materialize_rate = materialize_kernel_seconds == 0.0 ? 0.0 : static_cast<double>(total_hits) / materialize_kernel_seconds;
		const double wall_rate = wall_seconds == 0.0 ? 0.0 : candidate_count / wall_seconds;

		std::cout << "CUDA checksum-screen benchmark\n";
		std::cout << "  device: " << device_name << "\n";
		std::cout << "  map_list: " << args.map_list << "\n";
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
		std::cout << "  screen_rate: " << std::fixed << std::setprecision(0) << kernel_rate << " candidates/s\n";
		if (total_hits > 0)
		{
			std::cout << "  materialize_rate: " << std::fixed << std::setprecision(0) << materialize_rate << " survivors/s\n";
		}
		std::cout << "  wall_rate: " << std::fixed << std::setprecision(0) << wall_rate << " candidates/s\n";
		std::cout << "  checksum survivors: " << total_hits << "\n";
		std::cout << "  checksum survivors carnival: " << carnival_hits << "\n";
		std::cout << "  checksum survivors other: " << other_hits << "\n";
		std::cout << "  checksum survivors dual: " << dual_hits << "\n";
		std::cout << "  validated survivors: " << validation_summary.total << "\n";
		std::cout << "  validated carnival: " << validation_summary.carnival << "\n";
		std::cout << "  validated other: " << validation_summary.other << "\n";
		std::cout << "  validated dual_pass: " << validation_summary.dual_pass << "\n";
		std::cout << "  validated first_entry_valid: " << validation_summary.first_entry_valid << "\n";
		std::cout << "  validated all_entries_valid: " << validation_summary.all_entries_valid << "\n";
		std::cout << "  validated all_entries_valid_carnival: " << validation_summary.all_entries_valid_carnival << "\n";
		std::cout << "  validated all_entries_valid_other: " << validation_summary.all_entries_valid_other << "\n";
		std::cout << "  validated linear_scan_clean_carnival: " << validation_summary.linear_scan_clean_carnival << "\n";
		std::cout << "  validated linear_scan_clean_other: " << validation_summary.linear_scan_clean_other << "\n";
		std::cout << "  validated linear_scan_score_max_carnival: " << validation_summary.linear_scan_score_max_carnival << "\n";
		std::cout << "  validated linear_scan_score_max_other: " << validation_summary.linear_scan_score_max_other << "\n";
		std::cout << "  validated other_final_rts: " << validation_summary.other_final_rts << "\n";
		std::cout << "  validated other_entry_opcodes: " << validation_summary.other_entry_opcodes << "\n";
		std::cout << "  validated other_control_flow: " << validation_summary.other_control_flow << "\n";
		std::cout << "  validated other_structural: " << validation_summary.other_structural << "\n";
		std::cout << "  validated all_entries_other_final_rts: " << validation_summary.all_entries_other_final_rts << "\n";
		std::cout << "  validated all_entries_other_entry_opcodes: " << validation_summary.all_entries_other_entry_opcodes << "\n";
		std::cout << "  validated all_entries_other_control_flow: " << validation_summary.all_entries_other_control_flow << "\n";
		std::cout << "  validated all_entries_other_structural: " << validation_summary.all_entries_other_structural << "\n";
		for (int p = 0; p < OTHER_PREDICATE_COUNT; p++)
		{
			std::cout << "  validated other_pred_" << kOtherPredicateNames[p] << ": "
			          << validation_summary.other_predicate[p] << "\n";
			std::cout << "  validated final_rts_other_pred_" << kOtherPredicateNames[p] << ": "
			          << validation_summary.final_rts_other_predicate[p] << "\n";
			std::cout << "  validated all_entries_other_pred_" << kOtherPredicateNames[p] << ": "
			          << validation_summary.all_entries_other_predicate[p] << "\n";
			std::cout << "  validated all_entries_final_rts_other_pred_" << kOtherPredicateNames[p] << ": "
			          << validation_summary.all_entries_final_rts_other_predicate[p] << "\n";
		}
		std::cout << "  validated unique_states_carnival: " << validation_summary.unique_states_carnival << "\n";
		std::cout << "  validated unique_states_other: " << validation_summary.unique_states_other << "\n";
		std::cout << "  validated uses_nop: " << validation_summary.uses_nop << "\n";
		std::cout << "  validated uses_unofficial_nops: " << validation_summary.uses_unofficial_nops << "\n";
		std::cout << "  validated uses_illegal: " << validation_summary.uses_illegal << "\n";
		std::cout << "  validated uses_jam: " << validation_summary.uses_jam << "\n";
		if (args.hll)
		{
			std::cout << "  hll_distinct_states: " << validation_summary.hll_distinct_states << "\n";
			if (validation_summary.hll_distinct_states > 0)
			{
				const double collision_factor = static_cast<double>(args.workunit_size)
				                              / static_cast<double>(validation_summary.hll_distinct_states);
				std::cout << "  hll_collision_factor: " << std::fixed << std::setprecision(2) << collision_factor << "\n";
			}
		}
		if (!args.output_csv_path.empty())
		{
			std::cout << "  output_csv: " << args.output_csv_path << "\n";
		}
	}

	uint32_t state_kernel_grid_x(uint32_t candidate_count)
	{
		return (candidate_count + kCudaWarpsPerBlock - 1u) / kCudaWarpsPerBlock;
	}

	uint32_t screen_kernel_grid_x(uint32_t candidate_count)
	{
		const uint32_t candidates_per_block = kCudaWarpsPerBlock * kCudaScreenCandidatesPerWarp;
		return (candidate_count + candidates_per_block - 1u) / candidates_per_block;
	}

	uint32_t screen_kernel_grid_x_ilp(uint32_t candidate_count, uint32_t ilp)
	{
		const uint32_t candidates_per_block = kCudaWarpsPerBlock * ilp;
		return (candidate_count + candidates_per_block - 1u) / candidates_per_block;
	}
}

int main(int argc, char** argv)
{
	try
	{
		const Args args = parse_args(argc, argv);
		const auto setup_begin = std::chrono::high_resolution_clock::now();

		check_cuda(cuInit(0), "cuInit");

		int device_count = 0;
		check_cuda(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
		if (device_count == 0)
		{
			throw std::runtime_error("No CUDA devices found");
		}
		if (args.device_index >= static_cast<uint32_t>(device_count))
		{
			throw std::runtime_error("Requested --device is out of range");
		}

		CUdevice device = 0;
		check_cuda(cuDeviceGet(&device, static_cast<int>(args.device_index)), "cuDeviceGet");

		char device_name_buffer[256] = {};
		check_cuda(cuDeviceGetName(device_name_buffer, static_cast<int>(sizeof(device_name_buffer)), device), "cuDeviceGetName");
		const std::string device_name(device_name_buffer);

		CUcontext context = nullptr;
		check_cuda(cuCtxCreate(&context, nullptr, 0, device), "cuCtxCreate");

		const std::vector<uint8_t> module_blob = load_module_blob();
		CUmodule module = nullptr;
		check_cuda(cuModuleLoadData(&module, module_blob.data()), "cuModuleLoadData");

		const bool use_skipcar = (args.map_list == "skip-car");
		const std::string ksuffix = use_skipcar ? "_skipcar" : "";

		CUfunction screen_kernel = nullptr;
		check_cuda(cuModuleGetFunction(&screen_kernel, module, ("tm_checksum_screen_cuda" + ksuffix).c_str()), "cuModuleGetFunction(tm_checksum_screen_cuda)");
		CUfunction screen_hll_kernel = nullptr;
		check_cuda(cuModuleGetFunction(&screen_hll_kernel, module, ("tm_screen_and_hll_cuda" + ksuffix).c_str()), "cuModuleGetFunction(tm_screen_and_hll_cuda)");
		CUfunction materialize_kernel = nullptr;
		check_cuda(cuModuleGetFunction(&materialize_kernel, module, ("tm_materialize_survivors_cuda" + ksuffix).c_str()), "cuModuleGetFunction(tm_materialize_survivors_cuda)");
		CUfunction dump_kernel = nullptr;
		check_cuda(cuModuleGetFunction(&dump_kernel, module, ("tm_dump_state_cuda" + ksuffix).c_str()), "cuModuleGetFunction(tm_dump_state_cuda)");
		CUfunction screen_offset_ilp4_kernel = nullptr;
		CUfunction screen_offset_ilp6_kernel = nullptr;
		CUfunction screen_offset_ilp8_kernel = nullptr;
		check_cuda(cuModuleGetFunction(&screen_offset_ilp4_kernel, module, ("tm_checksum_screen_offset_store_ilp4_cuda" + ksuffix).c_str()), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp4_cuda)");
		check_cuda(cuModuleGetFunction(&screen_offset_ilp6_kernel, module, ("tm_checksum_screen_offset_store_ilp6_cuda" + ksuffix).c_str()), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp6_cuda)");
		check_cuda(cuModuleGetFunction(&screen_offset_ilp8_kernel, module, ("tm_checksum_screen_offset_store_ilp8_cuda" + ksuffix).c_str()), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp8_cuda)");

		KernelAssets assets = build_kernel_assets(args.key_id, args.map_list, args.screen_offsets);
		CUdeviceptr result_buffer = 0;
		check_cuda(cuMemAlloc(&result_buffer, args.batch_size), "cuMemAlloc(result_buffer)");
		CUdeviceptr survivor_data_buffer = 0;
		check_cuda(cuMemAlloc(&survivor_data_buffer, args.batch_size * sizeof(uint32_t)), "cuMemAlloc(survivor_data_buffer)");
		CUdeviceptr survivor_flag_buffer = 0;
		check_cuda(cuMemAlloc(&survivor_flag_buffer, args.batch_size), "cuMemAlloc(survivor_flag_buffer)");
		CUdeviceptr materialized_state_buffer = 0;
		check_cuda(cuMemAlloc(&materialized_state_buffer, args.batch_size * 128ull), "cuMemAlloc(materialized_state_buffer)");
		// HLL: 4096 uint32_t registers = 16 KB, zeroed before the sweep
		static const uint32_t kHllM = 4096u;
		CUdeviceptr hll_buffer = 0;
		check_cuda(cuMemAlloc(&hll_buffer, kHllM * sizeof(uint32_t)), "cuMemAlloc(hll_buffer)");
		check_cuda(cuMemsetD32(hll_buffer, 0u, kHllM), "cuMemsetD32(hll_buffer)");

		CUstream stream = nullptr;
		check_cuda(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate");

		CUevent kernel_start = nullptr;
		CUevent kernel_end = nullptr;
		check_cuda(cuEventCreate(&kernel_start, CU_EVENT_DEFAULT), "cuEventCreate(start)");
		check_cuda(cuEventCreate(&kernel_end, CU_EVENT_DEFAULT), "cuEventCreate(end)");

		std::vector<uint8_t> host_results(args.batch_size, 0);
		std::vector<SurvivorRecord> survivors;
		const auto setup_end = std::chrono::high_resolution_clock::now();

		// ---- Parity test ----
		if (args.parity_count > 0)
		{
			const uint32_t N = args.parity_count;
			const std::vector<uint8_t> schedule_blob = build_schedule_blob(args.key_id, args.map_list);

			CUdeviceptr gpu_dump = 0;
			check_cuda(cuMemAlloc(&gpu_dump, static_cast<std::size_t>(N) * 128u), "cuMemAlloc(parity_dump)");

			uint32_t parity_start = static_cast<uint32_t>(args.range_start);
			uint32_t parity_n = N;
			void* dump_args[] = {
				&gpu_dump,
				&assets.regular_rng_values,
				&assets.alg0_values,
				&assets.alg6_values,
				&assets.rng_seed_forward_1,
				&assets.rng_seed_forward_128,
				&assets.alg2_values,
				&assets.alg5_values,
				&assets.expansion_values,
				&assets.schedule_data,
				const_cast<uint32_t*>(&args.key_id),
				&parity_start,
				&parity_n
			};
			check_cuda(cuLaunchKernel(dump_kernel, state_kernel_grid_x(parity_n), 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, dump_args, nullptr), "cuLaunchKernel(dump)");
			check_cuda(cuStreamSynchronize(stream), "cuStreamSynchronize(dump)");

			std::vector<uint8_t> gpu_state(static_cast<std::size_t>(parity_n) * 128u);
			check_cuda(cuMemcpyDtoH(gpu_state.data(), gpu_dump, gpu_state.size()), "cuMemcpyDtoH(parity_dump)");
			cuMemFree(gpu_dump);

			uint32_t mismatch_count = 0;
			uint32_t first_mismatch_candidate = 0;
			int      first_mismatch_byte = -1;
			uint8_t  first_cpu_byte = 0;
			uint8_t  first_gpu_byte = 0;

			std::vector<uint8_t> cpu_state(128);
			for (uint32_t c = 0; c < parity_n; c++)
			{
					cpu_forward(args.key_id, static_cast<uint32_t>(args.range_start + c), schedule_blob, cpu_state.data());

				const uint8_t* gpu_c = gpu_state.data() + static_cast<std::size_t>(c) * 128u;
				for (int b = 0; b < 128; b++)
				{
					if (cpu_state[b] != gpu_c[b])
					{
						if (mismatch_count == 0)
						{
							first_mismatch_candidate = c;
							first_mismatch_byte      = b;
							first_cpu_byte           = cpu_state[b];
							first_gpu_byte           = gpu_c[b];
						}
						mismatch_count++;
						break;   // count one mismatch per candidate
					}
				}
			}

			std::cout << "Parity test: " << parity_n << " candidates  key=0x"
			          << std::hex << std::setw(8) << std::setfill('0') << args.key_id
			          << "  data_start=0x" << std::setw(8) << static_cast<uint32_t>(args.range_start) << std::dec << "\n";
			if (mismatch_count == 0)
			{
				std::cout << "  PASS — all " << parity_n << " candidates match CPU (baseline kernel)\n";
			}
			else
			{
				std::cout << "  FAIL — " << mismatch_count << "/" << N << " candidates have mismatches (baseline kernel)\n";
				std::cout << "  First mismatch: candidate " << first_mismatch_candidate
				          << "  byte[" << first_mismatch_byte << "]"
				          << "  cpu=0x" << std::hex << static_cast<unsigned>(first_cpu_byte)
				          << "  gpu=0x" << static_cast<unsigned>(first_gpu_byte) << std::dec << "\n";
			}

			// Additional cross-check when --screen-offsets is set: compare baseline
			// screen vs offset-store screen flag-by-flag. Catches divergence in the
			// offset path that the CPU vs baseline test above can't see.
			if (args.screen_offsets)
			{
				const uint32_t ilp = args.ilp;
				CUfunction offset_kernel_check = (ilp == 4u) ? screen_offset_ilp4_kernel
				                              : (ilp == 6u) ? screen_offset_ilp6_kernel
				                              :               screen_offset_ilp8_kernel;
				CUdeviceptr buf_base = 0, buf_offset = 0;
				const std::size_t buf_base_bytes   = static_cast<std::size_t>(N);
				const std::size_t buf_offset_bytes = ((static_cast<std::size_t>(N) + ilp - 1u) / ilp) * ilp;
				check_cuda(cuMemAlloc(&buf_base,   buf_base_bytes),   "cuMemAlloc(parity_base)");
				check_cuda(cuMemAlloc(&buf_offset, buf_offset_bytes), "cuMemAlloc(parity_offset)");

				uint32_t parity_n_local = N;
				void* baseline_args[] = {
					&buf_base,
					&assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
					&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
					&assets.alg2_values, &assets.alg5_values,
					&assets.expansion_values, &assets.schedule_data, &assets.carnival_data,
					const_cast<uint32_t*>(&args.key_id), &parity_start, &parity_n_local
				};
				void* offset_args[] = {
					&buf_offset,
					&assets.offset_regular, &assets.offset_alg0, &assets.offset_alg6,
					&assets.offset_alg2, &assets.offset_alg5,
					&assets.expansion_values, &assets.schedule_data, &assets.carnival_data,
					const_cast<uint32_t*>(&args.key_id), &parity_start, &parity_n_local
				};
				check_cuda(cuLaunchKernel(screen_kernel,       screen_kernel_grid_x(parity_n_local),       1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, baseline_args, nullptr), "cuLaunchKernel(parity_base)");
				check_cuda(cuLaunchKernel(offset_kernel_check, screen_kernel_grid_x_ilp(parity_n_local, ilp), 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, offset_args,   nullptr), "cuLaunchKernel(parity_offset)");
				check_cuda(cuStreamSynchronize(stream), "cuStreamSynchronize(parity_flags)");

				std::vector<uint8_t> base_flags(buf_base_bytes), offset_flags(buf_offset_bytes);
				check_cuda(cuMemcpyDtoH(base_flags.data(),   buf_base,   buf_base_bytes),   "cuMemcpyDtoH(parity_base)");
				check_cuda(cuMemcpyDtoH(offset_flags.data(), buf_offset, buf_offset_bytes), "cuMemcpyDtoH(parity_offset)");
				cuMemFree(buf_base); cuMemFree(buf_offset);

				std::size_t flag_mismatch = 0;
				std::size_t first_flag_mismatch = 0;
				for (std::size_t i = 0; i < N; i++)
				{
					if (base_flags[i] != offset_flags[i])
					{
						if (flag_mismatch == 0) first_flag_mismatch = i;
						flag_mismatch++;
					}
				}
				std::cout << "Cross-check: baseline screen vs offset-store ilp" << ilp << "\n";
				if (flag_mismatch == 0)
				{
					std::cout << "  PASS — all " << N << " flag bytes match\n";
				}
				else
				{
					std::cout << "  FAIL — " << flag_mismatch << "/" << N << " mismatches; first at candidate "
					          << first_flag_mismatch << "  base=0x" << std::hex << static_cast<unsigned>(base_flags[first_flag_mismatch])
					          << "  offset=0x" << static_cast<unsigned>(offset_flags[first_flag_mismatch]) << std::dec << "\n";
				}
			}

			return 0;
		}
		// ---- End parity test ----

		// Pick the active screen kernel up-front. The offset-store kernels
		// take 12 args (no rng_forward_1/128, no carnival_data passed by
		// pointer here — they read from the offset-stream buffers).
		CUfunction offset_kernel = nullptr;
		if (args.screen_offsets)
		{
			if      (args.ilp == 4u) offset_kernel = screen_offset_ilp4_kernel;
			else if (args.ilp == 6u) offset_kernel = screen_offset_ilp6_kernel;
			else /* 8 */             offset_kernel = screen_offset_ilp8_kernel;
		}

		double warmup_seconds = 0.0;
		for (uint32_t i = 0; i < args.warmup_batches; i++)
		{
			uint32_t warmup_start = static_cast<uint32_t>(args.range_start + (static_cast<uint64_t>(i) * args.batch_size));
			uint32_t warmup_count = args.batch_size;
			const auto warmup_begin = std::chrono::high_resolution_clock::now();

			void* kernel_args[] = {
				&result_buffer,
				&assets.regular_rng_values,
				&assets.alg0_values,
				&assets.alg6_values,
				&assets.rng_seed_forward_1,
				&assets.rng_seed_forward_128,
				&assets.alg2_values,
				&assets.alg5_values,
				&assets.expansion_values,
				&assets.schedule_data,
				&assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id),
				&warmup_start,
				&warmup_count
			};
			void* offset_args[] = {
				&result_buffer,
				&assets.offset_regular,
				&assets.offset_alg0,
				&assets.offset_alg6,
				&assets.offset_alg2,
				&assets.offset_alg5,
				&assets.expansion_values,
				&assets.schedule_data,
				&assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id),
				&warmup_start,
				&warmup_count
			};

			if (args.screen_offsets)
			{
				check_cuda(cuLaunchKernel(offset_kernel, screen_kernel_grid_x_ilp(warmup_count, args.ilp), 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, offset_args, nullptr), "cuLaunchKernel(warmup_offset)");
			}
			else
			{
				check_cuda(cuLaunchKernel(screen_kernel, screen_kernel_grid_x(warmup_count), 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, kernel_args, nullptr), "cuLaunchKernel(warmup)");
			}
			check_cuda(cuStreamSynchronize(stream), "cuStreamSynchronize(warmup)");
			check_cuda(cuMemcpyDtoH(host_results.data(), result_buffer, args.batch_size), "cuMemcpyDtoH(warmup)");

			const auto warmup_end = std::chrono::high_resolution_clock::now();
			warmup_seconds += std::chrono::duration<double>(warmup_end - warmup_begin).count();
		}

		uint64_t carnival_hits = 0;
		uint64_t other_hits = 0;
		uint64_t dual_hits = 0;
		double screen_kernel_seconds = 0.0;
		double materialize_kernel_seconds = 0.0;
		const auto wall_begin = std::chrono::high_resolution_clock::now();

		for (uint64_t offset = 0; offset < args.workunit_size; offset += args.batch_size)
		{
			uint64_t chunk = args.workunit_size - offset;
			if (chunk > args.batch_size)
			{
				chunk = args.batch_size;
			}

			const uint32_t data_start = static_cast<uint32_t>(args.range_start + offset);
			uint32_t chunk_u32 = static_cast<uint32_t>(chunk);

			void* kernel_args_nohll[] = {
				&result_buffer,
				&assets.regular_rng_values,
				&assets.alg0_values,
				&assets.alg6_values,
				&assets.rng_seed_forward_1,
				&assets.rng_seed_forward_128,
				&assets.alg2_values,
				&assets.alg5_values,
				&assets.expansion_values,
				&assets.schedule_data,
				&assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id),
				const_cast<uint32_t*>(&data_start),
				&chunk_u32
			};
			void* kernel_args_hll[] = {
				&result_buffer,
				&hll_buffer,
				&assets.regular_rng_values,
				&assets.alg0_values,
				&assets.alg6_values,
				&assets.rng_seed_forward_1,
				&assets.rng_seed_forward_128,
				&assets.alg2_values,
				&assets.alg5_values,
				&assets.expansion_values,
				&assets.schedule_data,
				&assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id),
				const_cast<uint32_t*>(&data_start),
				&chunk_u32
			};
			void* kernel_args_offset[] = {
				&result_buffer,
				&assets.offset_regular,
				&assets.offset_alg0,
				&assets.offset_alg6,
				&assets.offset_alg2,
				&assets.offset_alg5,
				&assets.expansion_values,
				&assets.schedule_data,
				&assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id),
				const_cast<uint32_t*>(&data_start),
				&chunk_u32
			};
			void** kernel_args = args.screen_offsets ? kernel_args_offset
			                                         : (args.hll ? kernel_args_hll : kernel_args_nohll);
			CUfunction active_screen = args.screen_offsets ? offset_kernel
			                                               : (args.hll ? screen_hll_kernel : screen_kernel);
			const uint32_t active_grid = args.screen_offsets ? screen_kernel_grid_x_ilp(chunk_u32, args.ilp)
			                                                  : screen_kernel_grid_x(chunk_u32);

			check_cuda(cuEventRecord(kernel_start, stream), "cuEventRecord(start)");
			check_cuda(cuLaunchKernel(active_screen, active_grid, 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, kernel_args, nullptr), "cuLaunchKernel");
			check_cuda(cuEventRecord(kernel_end, stream), "cuEventRecord(end)");
			check_cuda(cuEventSynchronize(kernel_end), "cuEventSynchronize(end)");

			float elapsed_ms = 0.0f;
			check_cuda(cuEventElapsedTime(&elapsed_ms, kernel_start, kernel_end), "cuEventElapsedTime");
			screen_kernel_seconds += static_cast<double>(elapsed_ms) / 1000.0;

			check_cuda(cuMemcpyDtoH(host_results.data(), result_buffer, chunk_u32), "cuMemcpyDtoH");

			for (uint32_t i = 0; i < chunk_u32; i++)
			{
				const uint8_t flags = host_results[i];
				if ((flags & CHECKSUM_SENTINEL) == 0)
				{
					continue;
				}

				if ((flags & OTHER_WORLD) != 0)
				{
					other_hits++;
					if ((flags & DUAL_PASS) != 0)
					{
						dual_hits++;
					}
				}
				else
				{
					carnival_hits++;
				}

				SurvivorRecord survivor = {};
				survivor.data = data_start + i;
				survivor.screen_flags = flags;
				survivors.push_back(survivor);
			}
		}

		ValidationSummary validation_summary = {};
		double cpu_validation_seconds = 0.0;
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

				check_cuda(cuMemcpyHtoD(survivor_data_buffer, survivor_data_host.data(), chunk * sizeof(uint32_t)), "cuMemcpyHtoD(survivor_data)");
				check_cuda(cuMemcpyHtoD(survivor_flag_buffer, survivor_flags_host.data(), chunk), "cuMemcpyHtoD(survivor_flags)");

				uint32_t chunk_u32 = static_cast<uint32_t>(chunk);
				void* materialize_args[] = {
					&materialized_state_buffer,
					&survivor_data_buffer,
					&survivor_flag_buffer,
					&assets.regular_rng_values,
					&assets.alg0_values,
					&assets.alg6_values,
					&assets.rng_seed_forward_1,
					&assets.rng_seed_forward_128,
					&assets.alg2_values,
					&assets.alg5_values,
					&assets.expansion_values,
					&assets.schedule_data,
					&assets.carnival_data,
					const_cast<uint32_t*>(&args.key_id),
					&chunk_u32
				};

				check_cuda(cuEventRecord(kernel_start, stream), "cuEventRecord(materialize.start)");
				check_cuda(cuLaunchKernel(materialize_kernel, state_kernel_grid_x(chunk_u32), 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, materialize_args, nullptr), "cuLaunchKernel(materialize)");
				check_cuda(cuEventRecord(kernel_end, stream), "cuEventRecord(materialize.end)");
				check_cuda(cuEventSynchronize(kernel_end), "cuEventSynchronize(materialize.end)");

				float elapsed_ms = 0.0f;
				check_cuda(cuEventElapsedTime(&elapsed_ms, kernel_start, kernel_end), "cuEventElapsedTime(materialize)");
				materialize_kernel_seconds += static_cast<double>(elapsed_ms) / 1000.0;

				check_cuda(cuMemcpyDtoH(decrypted_state_host.data(), materialized_state_buffer, chunk * 128ull), "cuMemcpyDtoH(materialized_state)");

				const auto cpu_validation_begin = std::chrono::high_resolution_clock::now();
				for (std::size_t i = 0; i < chunk; i++)
				{
					const bool other_world = (survivor_flags_host[i] & OTHER_WORLD) != 0;
					const bool dual         = (survivor_flags_host[i] & DUAL_PASS)  != 0;
					const uint8_t* state    = &decrypted_state_host[i * 128];

					const uint8_t machine_flags = check_machine_code(state, other_world);
					survivors[offset + i].machine_flags = static_cast<uint8_t>(machine_flags | (other_world ? OTHER_WORLD : 0));
					accumulate_validation_summary(validation_summary, machine_flags, other_world);

					if (dual) validation_summary.dual_pass++;

					if (other_world)
					{
						const OtherPredicateResults predicates = evaluate_other_predicates(state);
						const bool final_rts = other_final_rts_shape(state);
						const bool entry_opcodes = other_entry_opcode_shape(state);
						const bool control_flow = other_control_flow_shape(state);
						const bool structural = entry_opcodes && control_flow;
						const bool all_entries = (machine_flags & ALL_ENTRIES_VALID) != 0;

						if (final_rts) validation_summary.other_final_rts++;
						if (entry_opcodes) validation_summary.other_entry_opcodes++;
						if (control_flow) validation_summary.other_control_flow++;
						if (structural) validation_summary.other_structural++;

						for (int p = 0; p < OTHER_PREDICATE_COUNT; p++)
						{
							if (!predicates.values[p])
							{
								continue;
							}
							validation_summary.other_predicate[p]++;
							if (final_rts)
							{
								validation_summary.final_rts_other_predicate[p]++;
							}
							if (all_entries)
							{
								validation_summary.all_entries_other_predicate[p]++;
								if (final_rts)
								{
									validation_summary.all_entries_final_rts_other_predicate[p]++;
								}
							}
						}

						if (all_entries)
						{
							if (final_rts) validation_summary.all_entries_other_final_rts++;
							if (entry_opcodes) validation_summary.all_entries_other_entry_opcodes++;
							if (control_flow) validation_summary.all_entries_other_control_flow++;
							if (structural) validation_summary.all_entries_other_structural++;
						}
					}

					// Linear stream scan: clean flag + gradient score
					const int lscore = linear_scan_score(state, other_world);
					const bool lclean = linear_scan_clean(state, other_world);
					if (lclean)
					{
						if (other_world) validation_summary.linear_scan_clean_other++;
						else             validation_summary.linear_scan_clean_carnival++;
					}
					if (other_world)
					{
						if (static_cast<uint64_t>(lscore) > validation_summary.linear_scan_score_max_other)
							validation_summary.linear_scan_score_max_other = static_cast<uint64_t>(lscore);
					}
					else
					{
						if (static_cast<uint64_t>(lscore) > validation_summary.linear_scan_score_max_carnival)
							validation_summary.linear_scan_score_max_carnival = static_cast<uint64_t>(lscore);
					}

					// Hash-based unique state counting
					const uint64_t h = hash_state(state);
					if (other_world) validation_summary.unique_states_other    += 1;  // placeholder, set below
					else             validation_summary.unique_states_carnival += 1;  // placeholder, set below
					(void)h;  // suppress unused warning; set is built separately below
				}
				const auto cpu_validation_end = std::chrono::high_resolution_clock::now();
				cpu_validation_seconds += std::chrono::duration<double>(cpu_validation_end - cpu_validation_begin).count();
			}

			// Count distinct decrypted states using hash sets (after all batches are done above)
			{
				std::unordered_set<uint64_t> hashes_carnival;
				std::unordered_set<uint64_t> hashes_other;
				hashes_carnival.reserve(static_cast<std::size_t>(other_hits + carnival_hits));
				hashes_other.reserve(static_cast<std::size_t>(other_hits));

				// Re-materialize in chunks to hash; reuse existing GPU buffers
				for (std::size_t offset = 0; offset < survivors.size(); offset += args.batch_size)
				{
					std::size_t chunk = survivors.size() - offset;
					if (chunk > args.batch_size) chunk = args.batch_size;

					for (std::size_t i = 0; i < chunk; i++)
					{
						survivor_data_host[i]  = survivors[offset + i].data;
						survivor_flags_host[i] = survivors[offset + i].screen_flags;
					}

					check_cuda(cuMemcpyHtoD(survivor_data_buffer,  survivor_data_host.data(),  chunk * sizeof(uint32_t)), "cuMemcpyHtoD(hash_data)");
					check_cuda(cuMemcpyHtoD(survivor_flag_buffer,  survivor_flags_host.data(), chunk),                    "cuMemcpyHtoD(hash_flags)");

					uint32_t chunk_u32 = static_cast<uint32_t>(chunk);
					void* mat_args[] = {
						&materialized_state_buffer,
						&survivor_data_buffer,
						&survivor_flag_buffer,
						&assets.regular_rng_values,
						&assets.alg0_values,
						&assets.alg6_values,
						&assets.rng_seed_forward_1,
						&assets.rng_seed_forward_128,
						&assets.alg2_values,
						&assets.alg5_values,
						&assets.expansion_values,
						&assets.schedule_data,
						&assets.carnival_data,
						const_cast<uint32_t*>(&args.key_id),
						&chunk_u32
					};
					check_cuda(cuLaunchKernel(materialize_kernel, state_kernel_grid_x(chunk_u32), 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, mat_args, nullptr), "cuLaunchKernel(hash_mat)");
					check_cuda(cuStreamSynchronize(stream), "cuStreamSynchronize(hash_mat)");
					check_cuda(cuMemcpyDtoH(decrypted_state_host.data(), materialized_state_buffer, chunk * 128ull), "cuMemcpyDtoH(hash_state)");

					for (std::size_t i = 0; i < chunk; i++)
					{
						const bool other_world = (survivor_flags_host[i] & OTHER_WORLD) != 0;
						const uint64_t h = hash_state(&decrypted_state_host[i * 128]);
						if (other_world) hashes_other.insert(h);
						else             hashes_carnival.insert(h);
					}
				}
				// Overwrite the placeholder counts with real unique counts
				validation_summary.unique_states_carnival = hashes_carnival.size();
				validation_summary.unique_states_other    = hashes_other.size();
			}
		}

		const auto wall_end = std::chrono::high_resolution_clock::now();
		const double setup_seconds = std::chrono::duration<double>(setup_end - setup_begin).count();
		const double wall_seconds = std::chrono::duration<double>(wall_end - wall_begin).count();

		if (args.hll)
		{
			std::vector<uint32_t> hll_host(kHllM, 0);
			check_cuda(cuMemcpyDtoH(hll_host.data(), hll_buffer, kHllM * sizeof(uint32_t)), "cuMemcpyDtoH(hll)");
			validation_summary.hll_distinct_states = hll_estimate(hll_host);
		}

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

		print_summary(args, device_name, setup_seconds, warmup_seconds, screen_kernel_seconds, materialize_kernel_seconds, cpu_validation_seconds, wall_seconds, carnival_hits, other_hits, dual_hits, validation_summary);

		cuEventDestroy(kernel_end);
		cuEventDestroy(kernel_start);
		cuStreamDestroy(stream);
		cuMemFree(hll_buffer);
		cuMemFree(materialized_state_buffer);
		cuMemFree(survivor_flag_buffer);
		cuMemFree(survivor_data_buffer);
		cuMemFree(result_buffer);
		release_assets(assets);
		cuFuncSetCacheConfig(materialize_kernel, CU_FUNC_CACHE_PREFER_NONE);
		cuModuleUnload(module);
		cuCtxDestroy(context);
		return 0;
	}
	catch (const std::exception& ex)
	{
		std::cerr << ex.what() << "\n";
		return 1;
	}
}
