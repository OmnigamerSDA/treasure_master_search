// Treasure Master — OpenCL forward-search host (32-byte working state × 8 bits/lane)
//
// Compiles and dispatches the kernels in `tm.cl` against an OpenCL device.
// Builds checksum-screen, expansion, stats, and full-process kernels and
// reports survivors plus throughput. Used as the legacy GPU path before
// the CUDA implementation in `../test_cuda/`.
//
// Build:  `make`                (uses GNU make; see Makefile)
// Invoke: `./tm_opencl_32_8_test --device <id> --key_id 0x...`
//
// Companion variant: ../tm_opencl_32_16_test (alternate 16-bit lane width).

#include <CL/cl.h>

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "data_sizes.h"
#include "key_schedule.h"
#include "rng.h"

namespace
{
	static const uint32_t kDefaultBatchSize = 1u << 20;
	static const uint8_t kMapList[26] = {
		0x00, 0x02, 0x05, 0x04, 0x03, 0x1D, 0x1C, 0x1E, 0x1B,
		0x07, 0x08, 0x06, 0x09, 0x0C, 0x20, 0x21, 0x22, 0x23,
		0x24, 0x25, 0x26, 0x0E, 0x0F, 0x10, 0x12, 0x11
	};

	static const uint8_t CHECKSUM_SENTINEL = 0x08;
	static const uint8_t OTHER_WORLD = 0x01;
	static const uint8_t FIRST_ENTRY_VALID = 0x02;
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
		uint32_t range_start = 0u;
		uint32_t workunit_size = 1u << 20;
		uint32_t batch_size = kDefaultBatchSize;
		uint32_t warmup_batches = 1u;
		uint32_t platform_index = 0u;
		uint32_t device_index = 0u;
		std::string output_csv_path;
		bool use_ilp6 = false;       // Use offset-stream + ILP6 + preids screen kernel
		uint32_t parity_count = 0u;  // Run parity vs baseline screen on N candidates and exit
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
		"src/bruteforce/tm_opencl_32_8_test/tm.cl",
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
			else if (arg == "--ilp6")
			{
				args.use_ilp6 = true;
			}
			else if (arg == "--parity" && i + 1 < argc)
			{
				args.parity_count = numeric_arg<uint32_t>(argv[++i], "--parity");
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
					<< "\n"
					<< "Screen-kernel selection:\n"
					<< "  --ilp6                      Use the offset-stream + ILP6 screen kernel (~2.3x faster).\n"
					<< "                              Costs ~22 MB extra device memory per key.\n"
					<< "\n"
					<< "Diagnostics:\n"
					<< "  --parity <count>            Compare baseline screen vs --ilp6 screen over <count>\n"
					<< "                              candidates flag-by-flag, then exit. Reports speedup + PASS/FAIL.\n"
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

		return args;
	}

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

	int reverse_offset(int offset)
	{
		return 127 - offset;
	}

	uint64_t hash_state(const uint8_t* data)
	{
		uint64_t hash = 1469598103934665603ull;
		for (int i = 0; i < 128; i++)
		{
			hash ^= static_cast<uint64_t>(data[i]);
			hash *= 1099511628211ull;
		}
		return hash;
	}

	uint16_t calculate_checksum_from_decrypted(const uint8_t* data, bool other_world)
	{
		const int code_length = other_world ? 0x53 : 0x72;
		const int checksum_start = 130 - code_length;
		uint16_t sum = 0;
		for (int i = checksum_start; i < 128; i++)
		{
			sum = static_cast<uint16_t>(sum + data[i]);
		}
		return sum;
	}

	uint16_t fetch_checksum_value_from_decrypted(const uint8_t* data, bool other_world)
	{
		const int code_length = other_world ? 0x53 : 0x72;
		return static_cast<uint16_t>((static_cast<uint16_t>(data[reverse_offset(code_length - 1)]) << 8)
			| static_cast<uint16_t>(data[reverse_offset(code_length - 2)]));
	}

	uint8_t check_machine_code(const uint8_t* data, bool other_world)
	{
		uint8_t code_length = other_world ? 0x53 : 0x72;
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

		status = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
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

		const bool need_offset_streams = args.use_ilp6 || args.parity_count > 0u;
		KernelAssets assets = build_kernel_assets(context, queue, args.key_id, need_offset_streams);

		cl_mem result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, args.batch_size, nullptr, &status);
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

