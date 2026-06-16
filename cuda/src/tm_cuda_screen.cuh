// Treasure Master — CUDA checksum-screen, HLL, and materialize kernel implementations.
//
// RESEARCH / A-B path (not the production default — that is the raceway, see
// tm_cuda_raceway.cuh). The flat screen remains the exactness/parity reference that the
// raceway and compaction engines are validated against, and the HLL/materialize helpers
// are shared diagnostics. Contains the __device__ _impl templates and their extern "C"
// __global__ entry points for the screening path and its diagnostic variants:
//
//   tm_checksum_screen_impl          — baseline ILP4 screen (byte-store)
//   tm_checksum_screen_packed_store_impl — same, uint32-packed store (production base)
//   tm_checksum_screen_offset_store_impl — offset-stream, no ILP promotion
//   tm_checksum_screen_offset_store_ilp_impl — ILP6 offset-stream (production kernel)
//   Experimental offset+ILP variants  — prefetch, carry-select, interleaved layouts
//   tm_checksum_screen_maprng_impl    — map-RNG 54KB per-launch POC
//   tm_checksum_screen_maprng_preext_impl — pre-extracted 3-stream maprng
//   tm_screen_and_hll_impl            — screen + HyperLogLog update
//   tm_dump_state_impl                — write raw 128-byte states for debugging
//   tm_materialize_survivors_impl     — CPU-side replay: full expand+schedule+validate
//   tm_checksum_screen_maprng_coalesced_impl — Phase 3 coalesced maprng SCREEN
//
// All _impl functions are __device__ __forceinline__ templates; the __global__
// entry points immediately follow each impl in the same section.
//
// Include this header only from tm_cuda.cu, after tm_cuda_primitives.cuh.

#pragma once
#include <cstdint>
template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_checksum_screen_impl(
	uint8_t* result_data,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * 4u;

	if (candidate_base >= candidate_count)
	{
		return;
	}

	uint32_t working_value0 = initialize_working_word(key, data_start + candidate_base + 0u, lane, expansion_values);
	uint32_t working_value1 = initialize_working_word(key, data_start + candidate_base + 1u, lane, expansion_values);
	uint32_t working_value2 = initialize_working_word(key, data_start + candidate_base + 2u, lane, expansion_values);
	uint32_t working_value3 = initialize_working_word(key, data_start + candidate_base + 3u, lane, expansion_values);

	run_schedule_quad_t<SCHEDULE_COUNT>(&working_value0, &working_value1, &working_value2, &working_value3, lane, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values, schedule_data);

	const uint8_t screen_flags0 = screen_candidate(working_value0, lane, carnival_data);
	const uint8_t screen_flags1 = screen_candidate(working_value1, lane, carnival_data);
	const uint8_t screen_flags2 = screen_candidate(working_value2, lane, carnival_data);
	const uint8_t screen_flags3 = screen_candidate(working_value3, lane, carnival_data);

	if (lane == 0u)
	{
		result_data[candidate_base + 0u] = screen_flags0;
		if ((candidate_base + 1u) < candidate_count)
		{
			result_data[candidate_base + 1u] = screen_flags1;
		}
		if ((candidate_base + 2u) < candidate_count)
		{
			result_data[candidate_base + 2u] = screen_flags2;
		}
		if ((candidate_base + 3u) < candidate_count)
		{
			result_data[candidate_base + 3u] = screen_flags3;
		}
	}
}

extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_cuda(
	uint8_t* result_data,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	tm_checksum_screen_impl<27u>(result_data, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
}

// Packed-store variant of tm_checksum_screen_cuda. Identical compute path,
// but lane 0 packs the 4 ILP flag bytes into a single uint32 and writes once,
// dropping 4 STG.B → 1 STG.E to relieve LSU-issue pressure (per ncu profile:
// stores were at 1.0/32 byte sector utilization, all from lane 0).
// Falls back to per-byte stores when the tail isn't a full quad.
template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_checksum_screen_packed_store_impl(
	uint8_t* result_data,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * 4u;

	if (candidate_base >= candidate_count)
	{
		return;
	}

	uint32_t working_value0 = initialize_working_word(key, data_start + candidate_base + 0u, lane, expansion_values);
	uint32_t working_value1 = initialize_working_word(key, data_start + candidate_base + 1u, lane, expansion_values);
	uint32_t working_value2 = initialize_working_word(key, data_start + candidate_base + 2u, lane, expansion_values);
	uint32_t working_value3 = initialize_working_word(key, data_start + candidate_base + 3u, lane, expansion_values);

	run_schedule_quad_t<SCHEDULE_COUNT>(&working_value0, &working_value1, &working_value2, &working_value3, lane, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values, schedule_data);

	const uint8_t screen_flags0 = screen_candidate(working_value0, lane, carnival_data);
	const uint8_t screen_flags1 = screen_candidate(working_value1, lane, carnival_data);
	const uint8_t screen_flags2 = screen_candidate(working_value2, lane, carnival_data);
	const uint8_t screen_flags3 = screen_candidate(working_value3, lane, carnival_data);

	if (lane == 0u)
	{
		// candidate_base is always a multiple of 4 (warp_index * 4), so
		// result_data + candidate_base is 4-byte aligned.
		if ((candidate_base + 4u) <= candidate_count)
		{
			const uint32_t packed =
				static_cast<uint32_t>(screen_flags0)
				| (static_cast<uint32_t>(screen_flags1) << 8)
				| (static_cast<uint32_t>(screen_flags2) << 16)
				| (static_cast<uint32_t>(screen_flags3) << 24);
			*reinterpret_cast<uint32_t*>(result_data + candidate_base) = packed;
		}
		else
		{
			result_data[candidate_base + 0u] = screen_flags0;
			if ((candidate_base + 1u) < candidate_count)
			{
				result_data[candidate_base + 1u] = screen_flags1;
			}
			if ((candidate_base + 2u) < candidate_count)
			{
				result_data[candidate_base + 2u] = screen_flags2;
			}
		}
	}
}

extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_packed_store_cuda(
	uint8_t* result_data,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	tm_checksum_screen_packed_store_impl<27u>(result_data, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
}

	extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_packed_store_cuda_skipcar(
		uint8_t* result_data,
		const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
	{
		tm_checksum_screen_packed_store_impl<26u>(result_data, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
	}

	template<uint32_t SCHEDULE_COUNT>
	__device__ __forceinline__ void tm_checksum_screen_offset_store_impl(
		uint8_t* result_data,
		const uint8_t* regular_rng_values,
		const uint8_t* alg0_values,
		const uint8_t* alg6_values,
		const uint32_t* alg2_values,
		const uint32_t* alg5_values,
		const uint8_t* expansion_values,
		const uint8_t* schedule_data,
		const uint8_t* carnival_data,
		uint32_t key,
		uint32_t data_start,
		uint32_t candidate_count)
	{
		const uint32_t lane = threadIdx.x & 31u;
		const uint32_t warp_index = threadIdx.x >> 5;
		const uint32_t warps_per_block = blockDim.x >> 5;
		const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * 4u;

		if (candidate_base >= candidate_count)
		{
			return;
		}

		uint32_t working_value0 = initialize_working_word(key, data_start + candidate_base + 0u, lane, expansion_values);
		uint32_t working_value1 = initialize_working_word(key, data_start + candidate_base + 1u, lane, expansion_values);
		uint32_t working_value2 = initialize_working_word(key, data_start + candidate_base + 2u, lane, expansion_values);
		uint32_t working_value3 = initialize_working_word(key, data_start + candidate_base + 3u, lane, expansion_values);

		run_schedule_quad_offset_t<SCHEDULE_COUNT>(&working_value0, &working_value1, &working_value2, &working_value3, lane, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, schedule_data);

		const uint8_t screen_flags0 = screen_candidate(working_value0, lane, carnival_data);
		const uint8_t screen_flags1 = screen_candidate(working_value1, lane, carnival_data);
		const uint8_t screen_flags2 = screen_candidate(working_value2, lane, carnival_data);
		const uint8_t screen_flags3 = screen_candidate(working_value3, lane, carnival_data);

		if (lane == 0u)
		{
			if ((candidate_base + 4u) <= candidate_count)
			{
				const uint32_t packed =
					static_cast<uint32_t>(screen_flags0)
					| (static_cast<uint32_t>(screen_flags1) << 8)
					| (static_cast<uint32_t>(screen_flags2) << 16)
					| (static_cast<uint32_t>(screen_flags3) << 24);
				*reinterpret_cast<uint32_t*>(result_data + candidate_base) = packed;
			}
			else
			{
				result_data[candidate_base + 0u] = screen_flags0;
				if ((candidate_base + 1u) < candidate_count) result_data[candidate_base + 1u] = screen_flags1;
				if ((candidate_base + 2u) < candidate_count) result_data[candidate_base + 2u] = screen_flags2;
			}
		}
	}

		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
		const uint8_t* alg0_values,
		const uint8_t* alg6_values,
		const uint32_t* alg2_values,
		const uint32_t* alg5_values,
		const uint8_t* expansion_values,
		const uint8_t* schedule_data,
		const uint8_t* carnival_data,
		uint32_t key,
		uint32_t data_start,
		uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_impl<27u>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		template<uint32_t SCHEDULE_COUNT, uint32_t ILP>
		__device__ __forceinline__ void tm_checksum_screen_offset_store_ilp_impl(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			const uint32_t lane = threadIdx.x & 31u;
			const uint32_t warp_index = threadIdx.x >> 5;
			const uint32_t warps_per_block = blockDim.x >> 5;
			const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * ILP;

			if (candidate_base >= candidate_count)
			{
				return;
			}

			uint32_t working_value[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				working_value[j] = initialize_working_word(key, data_start + candidate_base + j, lane, expansion_values);
			}

			for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
			{
				uint32_t packed_schedule = 0u;
				if (lane == 0u)
				{
					packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
				}
				packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

				uint32_t rng_offset[ILP];
				#pragma unroll
				for (uint32_t j = 0u; j < ILP; j++)
				{
					rng_offset[j] = 0u;
				}
				uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
					| ((packed_schedule & 0xFF000000u) >> 24));

				for (uint32_t i = 0u; i < 16u; i++)
					{
						const uint32_t source_lane = i >> 2;
						const uint32_t source_shift = (i & 3u) * 8u;
						const uint32_t algorithm_shift = source_shift + 1u + ((static_cast<uint32_t>(nibble_selector) >> 13) & 4u);
						#pragma unroll
						for (uint32_t j = 0u; j < ILP; j++)
						{
							const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, working_value[j], source_lane);
							const uint8_t algorithm_id = static_cast<uint8_t>((source_word >> algorithm_shift) & 0x07u);
							working_value[j] = run_alg_offset(working_value[j], lane, algorithm_id, schedule_index, &rng_offset[j],
								regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
					}
					nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
				}
			}

			uint8_t screen_flags[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				screen_flags[j] = screen_candidate(working_value[j], lane, carnival_data);
			}

			if (lane == 0u)
			{
				if (ILP == 1u)
				{
					result_data[candidate_base] = screen_flags[0];
				}
				else if (ILP == 2u)
				{
					if ((candidate_base + 2u) <= candidate_count)
					{
						const uint16_t packed =
							static_cast<uint16_t>(screen_flags[0])
							| static_cast<uint16_t>(static_cast<uint16_t>(screen_flags[1]) << 8);
						*reinterpret_cast<uint16_t*>(result_data + candidate_base) = packed;
					}
					else
					{
						result_data[candidate_base] = screen_flags[0];
					}
				}
				else
				{
					#pragma unroll
					for (uint32_t j = 0u; j < ILP; j += 4u)
					{
						if ((candidate_base + j + 4u) <= candidate_count)
						{
							const uint32_t packed =
								static_cast<uint32_t>(screen_flags[j + 0u])
								| (static_cast<uint32_t>(screen_flags[j + 1u]) << 8)
								| (static_cast<uint32_t>(screen_flags[j + 2u]) << 16)
								| (static_cast<uint32_t>(screen_flags[j + 3u]) << 24);
							*reinterpret_cast<uint32_t*>(result_data + candidate_base + j) = packed;
						}
						else
						{
							#pragma unroll
							for (uint32_t k = 0u; k < 4u; k++)
							{
								if ((candidate_base + j + k) < candidate_count)
								{
									result_data[candidate_base + j + k] = screen_flags[j + k];
								}
							}
						}
					}
				}
			}
		}

		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp1_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_impl<27u, 1u>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp2_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_impl<27u, 2u>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

			extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp8_cuda(
				uint8_t* result_data,
				const uint8_t* regular_rng_values,
				const uint8_t* alg0_values,
				const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
			{
				tm_checksum_screen_offset_store_ilp_impl<27u, 8u>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
			}

			extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp12_cuda(
				uint8_t* result_data,
				const uint8_t* regular_rng_values,
				const uint8_t* alg0_values,
				const uint8_t* alg6_values,
				const uint32_t* alg2_values,
				const uint32_t* alg5_values,
				const uint8_t* expansion_values,
				const uint8_t* schedule_data,
				const uint8_t* carnival_data,
				uint32_t key,
				uint32_t data_start,
				uint32_t candidate_count)
			{
				tm_checksum_screen_offset_store_ilp_impl<27u, 12u>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
			}

			extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp16_cuda(
				uint8_t* result_data,
				const uint8_t* regular_rng_values,
				const uint8_t* alg0_values,
				const uint8_t* alg6_values,
				const uint32_t* alg2_values,
				const uint32_t* alg5_values,
				const uint8_t* expansion_values,
				const uint8_t* schedule_data,
				const uint8_t* carnival_data,
				uint32_t key,
				uint32_t data_start,
				uint32_t candidate_count)
			{
				tm_checksum_screen_offset_store_ilp_impl<27u, 16u>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
			}

		template<uint32_t SOURCE_SHIFT>
		__device__ __forceinline__ uint8_t extract_algorithm_id_fixed_shift(uint32_t source_word, bool high_nibble)
		{
			const uint32_t low_algorithm_id = (source_word >> (SOURCE_SHIFT + 1u)) & 0x07u;
			const uint32_t high_algorithm_id = (source_word >> (SOURCE_SHIFT + 5u)) & 0x07u;
			return static_cast<uint8_t>(high_nibble ? high_algorithm_id : low_algorithm_id);
		}

		template<uint32_t SOURCE_SHIFT, uint32_t ILP>
		__device__ __forceinline__ void run_offset_fixed_step(
			uint32_t source_lane,
			uint32_t lane,
			bool high_nibble,
			uint32_t schedule_index,
			uint32_t (&working_value)[ILP],
			uint32_t (&rng_offset)[ILP],
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values)
		{
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, working_value[j], source_lane);
				const uint8_t algorithm_id = extract_algorithm_id_fixed_shift<SOURCE_SHIFT>(source_word, high_nibble);
				working_value[j] = run_alg_offset(working_value[j], lane, algorithm_id, schedule_index, &rng_offset[j],
					regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			}
		}

		template<uint32_t SOURCE_SHIFT, uint32_t ILP>
		__device__ __forceinline__ void run_offset_fixed_precompute_step(
			uint32_t source_lane,
			uint32_t lane,
			bool high_nibble,
			uint32_t schedule_index,
			uint32_t (&working_value)[ILP],
			uint32_t (&rng_offset)[ILP],
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values)
		{
			uint8_t algorithm_id[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, working_value[j], source_lane);
				algorithm_id[j] = extract_algorithm_id_fixed_shift<SOURCE_SHIFT>(source_word, high_nibble);
			}
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				working_value[j] = run_alg_offset(working_value[j], lane, algorithm_id[j], schedule_index, &rng_offset[j],
					regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			}
		}

		template<uint32_t ILP>
		__device__ __forceinline__ void store_screen_flags_ilp(
			uint8_t* result_data,
			uint32_t candidate_base,
			uint32_t candidate_count,
			const uint8_t (&screen_flags)[ILP])
		{
			if constexpr ((ILP % 4u) == 0u)
			{
				#pragma unroll
				for (uint32_t j = 0u; j < ILP; j += 4u)
				{
					if ((candidate_base + j + 4u) <= candidate_count)
					{
						const uint32_t packed =
							static_cast<uint32_t>(screen_flags[j + 0u])
							| (static_cast<uint32_t>(screen_flags[j + 1u]) << 8)
							| (static_cast<uint32_t>(screen_flags[j + 2u]) << 16)
							| (static_cast<uint32_t>(screen_flags[j + 3u]) << 24);
						*reinterpret_cast<uint32_t*>(result_data + candidate_base + j) = packed;
					}
					else
					{
						#pragma unroll
						for (uint32_t k = 0u; k < 4u; k++)
						{
							if ((candidate_base + j + k) < candidate_count)
							{
								result_data[candidate_base + j + k] = screen_flags[j + k];
							}
						}
					}
				}
			}
			else
			{
				// ILP not divisible by 4 (e.g. 6) — candidate_base is also not
				// 4-aligned, so per-byte stores are the only correct option.
				#pragma unroll
				for (uint32_t j = 0u; j < ILP; j++)
				{
					if ((candidate_base + j) < candidate_count)
					{
						result_data[candidate_base + j] = screen_flags[j];
					}
				}
			}
		}

		template<uint32_t SCHEDULE_COUNT, uint32_t ILP, bool FIXED_SHIFT, bool PRECOMPUTE_IDS, bool CARRY_SELECT, bool UNROLL_SCHEDULE = true, bool USE_LDG = false>
		__device__ __forceinline__ void tm_checksum_screen_offset_store_ilp_experiment_impl(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			const uint32_t lane = threadIdx.x & 31u;
			const uint32_t warp_index = threadIdx.x >> 5;
			const uint32_t warps_per_block = blockDim.x >> 5;
			const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * ILP;

			if (candidate_base >= candidate_count)
			{
				return;
			}

			uint32_t working_value[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				working_value[j] = initialize_working_word(key, data_start + candidate_base + j, lane, expansion_values);
			}

			// Schedule loop. NVCC fully unrolls a short literal-bound loop by
			// default; for large ILP that explodes the code footprint and the
			// "no_instruction" stall dominates (24% of cycles at ILP8). The
			// UNROLL_SCHEDULE=false variant forces a runtime loop to keep the
			// kernel inside the I-cache.
			// UNROLL_SCHEDULE=true preserves the original behavior: NO explicit
			// pragma — let NVCC's loop-unroll heuristic decide. Adding an
			// explicit `#pragma unroll` here forces FULL unroll of the 27-iter
			// schedule loop, which catastrophically inflates code size and
			// thrashes the I-cache (measured 5× slowdown on ILP8 with explicit
			// full-unroll vs heuristic default).
			if constexpr (UNROLL_SCHEDULE)
			{
				for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
				{
					uint32_t packed_schedule = 0u;
					if (lane == 0u)
					{
						packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
					}
					packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

					uint32_t rng_offset[ILP];
					#pragma unroll
					for (uint32_t j = 0u; j < ILP; j++)
					{
						rng_offset[j] = 0u;
					}
					uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
						| ((packed_schedule & 0xFF000000u) >> 24));

					if (FIXED_SHIFT)
					{
						for (uint32_t source_lane = 0u; source_lane < 4u; source_lane++)
						{
							const bool high_nibble0 = (nibble_selector & 0x8000u) != 0u;
							if (PRECOMPUTE_IDS) run_offset_fixed_precompute_step<0u>(source_lane, lane, high_nibble0, schedule_index, working_value, rng_offset, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
							else run_offset_fixed_step<0u>(source_lane, lane, high_nibble0, schedule_index, working_value, rng_offset, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
							nibble_selector = static_cast<uint16_t>(nibble_selector << 1);

							const bool high_nibble8 = (nibble_selector & 0x8000u) != 0u;
							if (PRECOMPUTE_IDS) run_offset_fixed_precompute_step<8u>(source_lane, lane, high_nibble8, schedule_index, working_value, rng_offset, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
							else run_offset_fixed_step<8u>(source_lane, lane, high_nibble8, schedule_index, working_value, rng_offset, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
							nibble_selector = static_cast<uint16_t>(nibble_selector << 1);

							const bool high_nibble16 = (nibble_selector & 0x8000u) != 0u;
							if (PRECOMPUTE_IDS) run_offset_fixed_precompute_step<16u>(source_lane, lane, high_nibble16, schedule_index, working_value, rng_offset, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
							else run_offset_fixed_step<16u>(source_lane, lane, high_nibble16, schedule_index, working_value, rng_offset, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
							nibble_selector = static_cast<uint16_t>(nibble_selector << 1);

							const bool high_nibble24 = (nibble_selector & 0x8000u) != 0u;
							if (PRECOMPUTE_IDS) run_offset_fixed_precompute_step<24u>(source_lane, lane, high_nibble24, schedule_index, working_value, rng_offset, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
							else run_offset_fixed_step<24u>(source_lane, lane, high_nibble24, schedule_index, working_value, rng_offset, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
							nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
						}
					}
					else
					{
						for (uint32_t i = 0u; i < 16u; i++)
						{
							const uint32_t source_lane = i >> 2;
							const uint32_t source_shift = (i & 3u) * 8u;
							const uint32_t algorithm_shift = source_shift + 1u + ((static_cast<uint32_t>(nibble_selector) >> 13) & 4u);
							uint8_t algorithm_id[ILP];
							#pragma unroll
							for (uint32_t j = 0u; j < ILP; j++)
							{
								const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, working_value[j], source_lane);
								algorithm_id[j] = static_cast<uint8_t>((source_word >> algorithm_shift) & 0x07u);
							}
							#pragma unroll
							for (uint32_t j = 0u; j < ILP; j++)
							{
								if (CARRY_SELECT)
								{
									working_value[j] = run_alg_offset_carry_select(working_value[j], lane, algorithm_id[j], schedule_index, &rng_offset[j],
										regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
								}
								else if constexpr (USE_LDG)
								{
									working_value[j] = run_alg_offset_ldg(working_value[j], lane, algorithm_id[j], schedule_index, &rng_offset[j],
										regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
								}
								else
								{
									working_value[j] = run_alg_offset(working_value[j], lane, algorithm_id[j], schedule_index, &rng_offset[j],
										regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
								}
							}
							nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
						}
					}
				}
			}
			else
			{
				#pragma unroll 1
				for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
				{
					uint32_t packed_schedule = 0u;
					if (lane == 0u)
					{
						packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
					}
					packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

					uint32_t rng_offset[ILP];
					#pragma unroll
					for (uint32_t j = 0u; j < ILP; j++)
					{
						rng_offset[j] = 0u;
					}
					uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
						| ((packed_schedule & 0xFF000000u) >> 24));

					if (FIXED_SHIFT)
					{
						for (uint32_t source_lane = 0u; source_lane < 4u; source_lane++)
						{
							const bool high_nibble0 = (nibble_selector & 0x8000u) != 0u;
							if (PRECOMPUTE_IDS) run_offset_fixed_precompute_step<0u>(source_lane, lane, high_nibble0, schedule_index, working_value, rng_offset, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
							else run_offset_fixed_step<0u>(source_lane, lane, high_nibble0, schedule_index, working_value, rng_offset, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
							nibble_selector = static_cast<uint16_t>(nibble_selector << 1);

							const bool high_nibble8 = (nibble_selector & 0x8000u) != 0u;
							if (PRECOMPUTE_IDS) run_offset_fixed_precompute_step<8u>(source_lane, lane, high_nibble8, schedule_index, working_value, rng_offset, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
							else run_offset_fixed_step<8u>(source_lane, lane, high_nibble8, schedule_index, working_value, rng_offset, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
							nibble_selector = static_cast<uint16_t>(nibble_selector << 1);

							const bool high_nibble16 = (nibble_selector & 0x8000u) != 0u;
							if (PRECOMPUTE_IDS) run_offset_fixed_precompute_step<16u>(source_lane, lane, high_nibble16, schedule_index, working_value, rng_offset, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
							else run_offset_fixed_step<16u>(source_lane, lane, high_nibble16, schedule_index, working_value, rng_offset, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
							nibble_selector = static_cast<uint16_t>(nibble_selector << 1);

							const bool high_nibble24 = (nibble_selector & 0x8000u) != 0u;
							if (PRECOMPUTE_IDS) run_offset_fixed_precompute_step<24u>(source_lane, lane, high_nibble24, schedule_index, working_value, rng_offset, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
							else run_offset_fixed_step<24u>(source_lane, lane, high_nibble24, schedule_index, working_value, rng_offset, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
							nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
						}
					}
					else
					{
						for (uint32_t i = 0u; i < 16u; i++)
						{
							const uint32_t source_lane = i >> 2;
							const uint32_t source_shift = (i & 3u) * 8u;
							const uint32_t algorithm_shift = source_shift + 1u + ((static_cast<uint32_t>(nibble_selector) >> 13) & 4u);
							uint8_t algorithm_id[ILP];
							#pragma unroll
							for (uint32_t j = 0u; j < ILP; j++)
							{
								const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, working_value[j], source_lane);
								algorithm_id[j] = static_cast<uint8_t>((source_word >> algorithm_shift) & 0x07u);
							}
							#pragma unroll
							for (uint32_t j = 0u; j < ILP; j++)
							{
								if (CARRY_SELECT)
								{
									working_value[j] = run_alg_offset_carry_select(working_value[j], lane, algorithm_id[j], schedule_index, &rng_offset[j],
										regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
								}
								else if constexpr (USE_LDG)
								{
									working_value[j] = run_alg_offset_ldg(working_value[j], lane, algorithm_id[j], schedule_index, &rng_offset[j],
										regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
								}
								else
								{
									working_value[j] = run_alg_offset(working_value[j], lane, algorithm_id[j], schedule_index, &rng_offset[j],
										regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
								}
							}
							nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
						}
					}
				}
			}

			uint8_t screen_flags[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				screen_flags[j] = screen_candidate(working_value[j], lane, carnival_data);
			}

			if (lane == 0u)
			{
				store_screen_flags_ilp<ILP>(result_data, candidate_base, candidate_count, screen_flags);
			}
		}

		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp8_fixed_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_experiment_impl<27u, 8u, true, false, false>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp8_preids_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_experiment_impl<27u, 8u, false, true, false>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp12_preids_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_experiment_impl<27u, 12u, false, true, false>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp16_preids_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_experiment_impl<27u, 16u, false, true, false>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp8_fixed_preids_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_experiment_impl<27u, 8u, true, true, false>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp8_preids_carrysel_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_experiment_impl<27u, 8u, false, true, true>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		// ILP8 + preids + schedule loop NOT unrolled — targets the no_instruction
		// stall (24% of cycles at ILP8) by keeping the 27 schedule iterations
		// as a runtime loop, dramatically shrinking the I-cache footprint.
		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp8_preids_unroll1_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_experiment_impl<27u, 8u, false, true, false, false>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		// ILP6 + preids — sweet-spot search between ILP4 and ILP8. ILP not
		// divisible by 4 forces per-byte store fallback (see store_screen_flags_ilp).
		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp6_preids_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_experiment_impl<27u, 6u, false, true, false>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		// ILP6 + preids + schedule loop NOT unrolled.
		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp6_preids_unroll1_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_experiment_impl<27u, 6u, false, true, false, false>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		// ILP5 + preids — explore U-shape floor between ILP4 (112 Mc/s) and ILP6 (131 Mc/s).
		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp5_preids_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_experiment_impl<27u, 5u, false, true, false>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp5_preids_unroll1_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_experiment_impl<27u, 5u, false, true, false, false>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		// Software-prefetch variant. Splits each inner-loop step into three
		// phases: (1) precompute algorithm_id[j] (same as preids); (2) issue
		// all ILP LDGs back-to-back into a `prefetched[ILP]` register array,
		// with no intervening branch (Always-Load-Dummy: algs 2/5/7 load
		// from regular_rng but discard the value); (3) apply alg using the
		// prefetched register (no LDG in the apply phase). Targets long_sb
		// stall by giving NVCC's scheduler a straight-line LDG block to
		// overlap with L1 hit latency.
		template<uint32_t SCHEDULE_COUNT, uint32_t ILP>
		__device__ __forceinline__ void tm_checksum_screen_offset_store_ilp_prefetch_impl(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			const uint32_t lane = threadIdx.x & 31u;
			const uint32_t warp_index = threadIdx.x >> 5;
			const uint32_t warps_per_block = blockDim.x >> 5;
			const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * ILP;

			if (candidate_base >= candidate_count)
			{
				return;
			}

			uint32_t working_value[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				working_value[j] = initialize_working_word(key, data_start + candidate_base + j, lane, expansion_values);
			}

			for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
			{
				uint32_t packed_schedule = 0u;
				if (lane == 0u)
				{
					packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
				}
				packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

				uint32_t rng_offset[ILP];
				#pragma unroll
				for (uint32_t j = 0u; j < ILP; j++)
				{
					rng_offset[j] = 0u;
				}
				uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
					| ((packed_schedule & 0xFF000000u) >> 24));

				for (uint32_t i = 0u; i < 16u; i++)
				{
					const uint32_t source_lane = i >> 2;
					const uint32_t source_shift = (i & 3u) * 8u;
					const uint32_t algorithm_shift = source_shift + 1u + ((static_cast<uint32_t>(nibble_selector) >> 13) & 4u);

					// Phase 1: precompute algorithm_id (same as preids)
					uint8_t algorithm_id[ILP];
					#pragma unroll
					for (uint32_t j = 0u; j < ILP; j++)
					{
						const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, working_value[j], source_lane);
						algorithm_id[j] = static_cast<uint8_t>((source_word >> algorithm_shift) & 0x07u);
					}

					// Phase 2: prefetch — issue all ILP LDGs back-to-back.
					// Always-Load-Dummy: algs 2/5/7 read regular_rng_values
					// but the apply phase ignores the loaded value.
					uint32_t prefetched[ILP];
					#pragma unroll
					for (uint32_t j = 0u; j < ILP; j++)
					{
						const uint8_t* table = (algorithm_id[j] == 0u) ? alg0_values :
						                       (algorithm_id[j] == 6u) ? alg6_values :
						                       regular_rng_values;
						prefetched[j] = load_offset_u32(table, schedule_index, rng_offset[j], lane);
					}

					// Phase 3: apply — consume prefetched register, no LDG
					#pragma unroll
					for (uint32_t j = 0u; j < ILP; j++)
					{
						working_value[j] = apply_alg_with_prefetched(
							working_value[j], lane, algorithm_id[j], prefetched[j],
							schedule_index, &rng_offset[j], alg2_values, alg5_values);
					}

					nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
				}
			}

			uint8_t screen_flags[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				screen_flags[j] = screen_candidate(working_value[j], lane, carnival_data);
			}

			if (lane == 0u)
			{
				store_screen_flags_ilp<ILP>(result_data, candidate_base, candidate_count, screen_flags);
			}
		}

		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp5_preids_prefetch_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_prefetch_impl<27u, 5u>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp6_preids_prefetch_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_prefetch_impl<27u, 6u>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp8_preids_prefetch_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_prefetch_impl<27u, 8u>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		// USE_LDG variants — route alg-0/1/3/4/6 loads through __ldg (read-only
		// data cache hint). Targets long_scoreboard stall (16% of cycles in
		// the ilp6_preids profile). Expected to be a small win or null on
		// sm_120 (unified L1/RO cache); also includes the non-aliasing hint
		// which may let the compiler reorder LDG ahead of the dependent LOP3.
		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp6_preids_ldg_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_experiment_impl<27u, 6u, false, true, false, true, true>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp8_preids_ldg_cuda(
			uint8_t* result_data,
			const uint8_t* regular_rng_values,
			const uint8_t* alg0_values,
			const uint8_t* alg6_values,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_store_ilp_experiment_impl<27u, 8u, false, true, false, true, true>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}

		template<uint32_t SCHEDULE_COUNT>
		__device__ __forceinline__ void tm_checksum_screen_offset_interleaved_store_impl(
			uint8_t* result_data,
			const uint8_t* offset_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			const uint32_t lane = threadIdx.x & 31u;
			const uint32_t warp_index = threadIdx.x >> 5;
			const uint32_t warps_per_block = blockDim.x >> 5;
			const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * 4u;

			if (candidate_base >= candidate_count)
			{
				return;
			}

			uint32_t working_value0 = initialize_working_word(key, data_start + candidate_base + 0u, lane, expansion_values);
			uint32_t working_value1 = initialize_working_word(key, data_start + candidate_base + 1u, lane, expansion_values);
			uint32_t working_value2 = initialize_working_word(key, data_start + candidate_base + 2u, lane, expansion_values);
			uint32_t working_value3 = initialize_working_word(key, data_start + candidate_base + 3u, lane, expansion_values);

			run_schedule_quad_offset_interleaved_t<SCHEDULE_COUNT>(&working_value0, &working_value1, &working_value2, &working_value3, lane, offset_values, schedule_data);

			const uint8_t screen_flags0 = screen_candidate(working_value0, lane, carnival_data);
			const uint8_t screen_flags1 = screen_candidate(working_value1, lane, carnival_data);
			const uint8_t screen_flags2 = screen_candidate(working_value2, lane, carnival_data);
			const uint8_t screen_flags3 = screen_candidate(working_value3, lane, carnival_data);

			if (lane == 0u)
			{
				if ((candidate_base + 4u) <= candidate_count)
				{
					const uint32_t packed =
						static_cast<uint32_t>(screen_flags0)
						| (static_cast<uint32_t>(screen_flags1) << 8)
						| (static_cast<uint32_t>(screen_flags2) << 16)
						| (static_cast<uint32_t>(screen_flags3) << 24);
					*reinterpret_cast<uint32_t*>(result_data + candidate_base) = packed;
				}
				else
				{
					result_data[candidate_base + 0u] = screen_flags0;
					if ((candidate_base + 1u) < candidate_count) result_data[candidate_base + 1u] = screen_flags1;
					if ((candidate_base + 2u) < candidate_count) result_data[candidate_base + 2u] = screen_flags2;
				}
			}
		}

		extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_interleaved_store_cuda(
			uint8_t* result_data,
			const uint8_t* offset_values,
			const uint8_t* expansion_values,
			const uint8_t* schedule_data,
			const uint8_t* carnival_data,
			uint32_t key,
			uint32_t data_start,
			uint32_t candidate_count)
		{
			tm_checksum_screen_offset_interleaved_store_impl<27u>(result_data, offset_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
		}
	
		// map_rng POC kernel — same launch shape as the universal-table version
// (128 threads/block, 4 warps/block, 4 candidates per warp). Replaces 8 MB
// universal tables with a 54 KB per-launch map_rng buffer + expansion table.
template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_checksum_screen_maprng_impl(
	uint8_t* result_data,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	const uint8_t* map_rng,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * 4u;

	if (candidate_base >= candidate_count)
	{
		return;
	}

	uint32_t v0 = initialize_working_word(key, data_start + candidate_base + 0u, lane, expansion_values);
	uint32_t v1 = initialize_working_word(key, data_start + candidate_base + 1u, lane, expansion_values);
	uint32_t v2 = initialize_working_word(key, data_start + candidate_base + 2u, lane, expansion_values);
	uint32_t v3 = initialize_working_word(key, data_start + candidate_base + 3u, lane, expansion_values);

	run_schedule_quad_maprng_t<SCHEDULE_COUNT>(&v0, &v1, &v2, &v3, lane, schedule_data, map_rng);

	const uint8_t f0 = screen_candidate(v0, lane, carnival_data);
	const uint8_t f1 = screen_candidate(v1, lane, carnival_data);
	const uint8_t f2 = screen_candidate(v2, lane, carnival_data);
	const uint8_t f3 = screen_candidate(v3, lane, carnival_data);

	if (lane == 0u)
	{
		result_data[candidate_base + 0u] = f0;
		if ((candidate_base + 1u) < candidate_count) result_data[candidate_base + 1u] = f1;
		if ((candidate_base + 2u) < candidate_count) result_data[candidate_base + 2u] = f2;
		if ((candidate_base + 3u) < candidate_count) result_data[candidate_base + 3u] = f3;
	}
}

extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_cuda_maprng_27(
	uint8_t* result_data,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	const uint8_t* map_rng,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	tm_checksum_screen_maprng_impl<27u>(result_data, expansion_values, schedule_data, carnival_data, map_rng, key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_cuda_maprng_skipcar(
	uint8_t* result_data,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	const uint8_t* map_rng,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	tm_checksum_screen_maprng_impl<26u>(result_data, expansion_values, schedule_data, carnival_data, map_rng, key, data_start, candidate_count);
}

template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_checksum_screen_maprng_preext_impl(
	uint8_t* result_data,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	const uint8_t* map_rng,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	constexpr uint32_t STREAM_SIZE = SCHEDULE_COUNT * 2048u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * 4u;

	if (candidate_base >= candidate_count)
	{
		return;
	}

	const uint8_t* raw_base_all  = map_rng;
	const uint8_t* alg0_base_all = map_rng + STREAM_SIZE;
	const uint8_t* alg6_base_all = map_rng + STREAM_SIZE * 2u;

	uint32_t v0 = initialize_working_word(key, data_start + candidate_base + 0u, lane, expansion_values);
	uint32_t v1 = initialize_working_word(key, data_start + candidate_base + 1u, lane, expansion_values);
	uint32_t v2 = initialize_working_word(key, data_start + candidate_base + 2u, lane, expansion_values);
	uint32_t v3 = initialize_working_word(key, data_start + candidate_base + 3u, lane, expansion_values);

	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));

		const uint8_t* raw_base  = raw_base_all  + schedule_index * 2048u;
		const uint8_t* alg0_base = alg0_base_all + schedule_index * 2048u;
		const uint8_t* alg6_base = alg6_base_all + schedule_index * 2048u;
		uint32_t pos0 = 0u, pos1 = 0u, pos2 = 0u, pos3 = 0u;

		auto inline_run = [&](uint32_t value, uint32_t lane, uint8_t alg_id, uint32_t* pos) -> uint32_t
		{
			if (alg_id == 0u)
			{
				value = ((value << 1) & 0xFEFEFEFEu) | maprng_pack_raw(alg0_base, *pos, lane);
				*pos += 128u;
			}
			else if (alg_id == 1u)
			{
				value = __vadd4(value, maprng_pack_raw(raw_base, *pos, lane));
				*pos += 128u;
			}
			else if (alg_id == 3u)
			{
				value ^= maprng_pack_raw(raw_base, *pos, lane);
				*pos += 128u;
			}
			else if (alg_id == 4u)
			{
				value = __vsub4(value, maprng_pack_raw(raw_base, *pos, lane));
				*pos += 128u;
			}
			else if (alg_id == 6u)
			{
				const uint32_t b0 = static_cast<uint32_t>(alg6_base[*pos + lane * 4u + 0u]);
				const uint32_t b1 = static_cast<uint32_t>(alg6_base[*pos + lane * 4u + 1u]);
				const uint32_t b2 = static_cast<uint32_t>(alg6_base[*pos + lane * 4u + 2u]);
				const uint32_t b3 = static_cast<uint32_t>(alg6_base[*pos + lane * 4u + 3u]);
				value = ((value >> 1) & 0x7F7F7F7Fu) | (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
				*pos += 128u;
			}
			else if (alg_id == 7u)
			{
				value ^= 0xFFFFFFFFu;
			}
			else
			{
				value = run_alg_maprng(value, lane, alg_id, pos, raw_base);
			}
			return value;
		};

		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, v0, sl);
			const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, v1, sl);
			const uint32_t w2 = __shfl_sync(0xFFFFFFFFu, v2, sl);
			const uint32_t w3 = __shfl_sync(0xFFFFFFFFu, v3, sl);
			uint8_t cb0 = static_cast<uint8_t>((w0 >> ss) & 0xFFu);
			uint8_t cb1 = static_cast<uint8_t>((w1 >> ss) & 0xFFu);
			uint8_t cb2 = static_cast<uint8_t>((w2 >> ss) & 0xFFu);
			uint8_t cb3 = static_cast<uint8_t>((w3 >> ss) & 0xFFu);
			if ((nsel & 0x8000u) != 0u) { cb0 >>= 4; cb1 >>= 4; cb2 >>= 4; cb3 >>= 4; }
			const uint8_t a0 = static_cast<uint8_t>((cb0 >> 1) & 0x07u);
			const uint8_t a1 = static_cast<uint8_t>((cb1 >> 1) & 0x07u);
			const uint8_t a2 = static_cast<uint8_t>((cb2 >> 1) & 0x07u);
			const uint8_t a3 = static_cast<uint8_t>((cb3 >> 1) & 0x07u);
			v0 = inline_run(v0, lane, a0, &pos0);
			v1 = inline_run(v1, lane, a1, &pos1);
			v2 = inline_run(v2, lane, a2, &pos2);
			v3 = inline_run(v3, lane, a3, &pos3);
			nsel = static_cast<uint16_t>(nsel << 1);
		}
	}

	const uint8_t f0 = screen_candidate(v0, lane, carnival_data);
	const uint8_t f1 = screen_candidate(v1, lane, carnival_data);
	const uint8_t f2 = screen_candidate(v2, lane, carnival_data);
	const uint8_t f3 = screen_candidate(v3, lane, carnival_data);

	if (lane == 0u)
	{
		if ((candidate_base + 4u) <= candidate_count)
		{
			const uint32_t packed =
				static_cast<uint32_t>(f0)
				| (static_cast<uint32_t>(f1) << 8)
				| (static_cast<uint32_t>(f2) << 16)
				| (static_cast<uint32_t>(f3) << 24);
			*reinterpret_cast<uint32_t*>(result_data + candidate_base) = packed;
		}
		else
		{
			result_data[candidate_base + 0u] = f0;
			if ((candidate_base + 1u) < candidate_count) result_data[candidate_base + 1u] = f1;
			if ((candidate_base + 2u) < candidate_count) result_data[candidate_base + 2u] = f2;
		}
	}
}

extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_maprng_preext_cuda(
	uint8_t* result_data,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	const uint8_t* map_rng,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	tm_checksum_screen_maprng_preext_impl<27u>(result_data, expansion_values, schedule_data, carnival_data, map_rng, key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_cuda_skipcar(
	uint8_t* result_data,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	tm_checksum_screen_impl<26u>(result_data, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
}

// Combined screen + HyperLogLog kernel.
// Identical to tm_checksum_screen_cuda but additionally hashes every output
// state (not just survivors) and updates a 4096-register HLL in global memory.
// This lets us estimate the number of DISTINCT output states across all 2^32
// data values for a given key — the true output-space cardinality, independent
// of which states happen to pass the checksum.
template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_screen_and_hll_impl(
	uint8_t* result_data,
	uint32_t* hll_registers,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	const uint32_t lane           = threadIdx.x & 31u;
	const uint32_t warp_index     = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * 4u;

	if (candidate_base >= candidate_count)
	{
		return;
	}

	uint32_t working_value0 = initialize_working_word(key, data_start + candidate_base + 0u, lane, expansion_values);
	uint32_t working_value1 = initialize_working_word(key, data_start + candidate_base + 1u, lane, expansion_values);
	uint32_t working_value2 = initialize_working_word(key, data_start + candidate_base + 2u, lane, expansion_values);
	uint32_t working_value3 = initialize_working_word(key, data_start + candidate_base + 3u, lane, expansion_values);

	run_schedule_quad_t<SCHEDULE_COUNT>(&working_value0, &working_value1, &working_value2, &working_value3, lane,
		regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128,
		alg2_values, alg5_values, schedule_data);

	const uint8_t screen_flags0 = screen_candidate(working_value0, lane, carnival_data);
	const uint8_t screen_flags1 = screen_candidate(working_value1, lane, carnival_data);
	const uint8_t screen_flags2 = screen_candidate(working_value2, lane, carnival_data);
	const uint8_t screen_flags3 = screen_candidate(working_value3, lane, carnival_data);

	if (lane == 0u)
	{
		result_data[candidate_base + 0u] = screen_flags0;
		if ((candidate_base + 1u) < candidate_count) result_data[candidate_base + 1u] = screen_flags1;
		if ((candidate_base + 2u) < candidate_count) result_data[candidate_base + 2u] = screen_flags2;
		if ((candidate_base + 3u) < candidate_count) result_data[candidate_base + 3u] = screen_flags3;
	}

	uint32_t h_lo, h_hi;

	warp_hash_state(working_value0, lane, &h_lo, &h_hi);
	if (lane == 0u) hll_update(hll_registers, h_lo, h_hi);

	if ((candidate_base + 1u) < candidate_count)
	{
		warp_hash_state(working_value1, lane, &h_lo, &h_hi);
		if (lane == 0u) hll_update(hll_registers, h_lo, h_hi);
	}
	if ((candidate_base + 2u) < candidate_count)
	{
		warp_hash_state(working_value2, lane, &h_lo, &h_hi);
		if (lane == 0u) hll_update(hll_registers, h_lo, h_hi);
	}
	if ((candidate_base + 3u) < candidate_count)
	{
		warp_hash_state(working_value3, lane, &h_lo, &h_hi);
		if (lane == 0u) hll_update(hll_registers, h_lo, h_hi);
	}
}

extern "C" __global__ __launch_bounds__(128) void tm_screen_and_hll_cuda(
	uint8_t* result_data,
	uint32_t* hll_registers,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	tm_screen_and_hll_impl<27u>(result_data, hll_registers, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(128) void tm_screen_and_hll_cuda_skipcar(
	uint8_t* result_data,
	uint32_t* hll_registers,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	tm_screen_and_hll_impl<26u>(result_data, hll_registers, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
}

template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_dump_state_impl(
	uint8_t* output_data,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t code_index = blockIdx.x * warps_per_block + warp_index;

	if (code_index >= candidate_count)
	{
		return;
	}

	const uint32_t cur_data = data_start + code_index;
	uint32_t working_value = initialize_working_word(key, cur_data, lane, expansion_values);
	working_value = run_schedule_t<SCHEDULE_COUNT>(working_value, lane, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values, schedule_data);

	reinterpret_cast<uint32_t*>(output_data)[code_index * 32u + lane] = working_value;
}

extern "C" __global__ __launch_bounds__(128) void tm_dump_state_cuda(
	uint8_t* output_data,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	tm_dump_state_impl<27u>(output_data, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values, schedule_data, key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(128) void tm_dump_state_cuda_skipcar(
	uint8_t* output_data,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	tm_dump_state_impl<26u>(output_data, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values, schedule_data, key, data_start, candidate_count);
}

template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_materialize_survivors_impl(
	uint8_t* output_data,
	const uint32_t* survivor_data,
	const uint8_t* survivor_flags,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t survivor_count)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t survivor_index = blockIdx.x * warps_per_block + warp_index;

	if (survivor_index >= survivor_count)
	{
		return;
	}

	const uint32_t cur_data = survivor_data[survivor_index];
	const uint8_t screen_flags = survivor_flags[survivor_index];

	uint32_t working_value = initialize_working_word(key, cur_data, lane, expansion_values);
	working_value = run_schedule_t<SCHEDULE_COUNT>(working_value, lane, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values, schedule_data);

	uint32_t decrypted_word = 0;
	if ((screen_flags & 0x01u) == 0u)
	{
		decrypted_word = working_value ^ reinterpret_cast<const uint32_t*>(carnival_data)[lane];
	}
	else
	{
		decrypted_word = other_world_word(working_value, lane);
	}

	reinterpret_cast<uint32_t*>(output_data)[survivor_index * 32u + lane] = decrypted_word;
}

extern "C" __global__ __launch_bounds__(128) void tm_materialize_survivors_cuda(
	uint8_t* output_data,
	const uint32_t* survivor_data,
	const uint8_t* survivor_flags,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t survivor_count)
{
	tm_materialize_survivors_impl<27u>(output_data, survivor_data, survivor_flags, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, survivor_count);
}

extern "C" __global__ __launch_bounds__(128) void tm_materialize_survivors_cuda_skipcar(
	uint8_t* output_data,
	const uint32_t* survivor_data,
	const uint8_t* survivor_flags,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t survivor_count)
{
	tm_materialize_survivors_impl<26u>(output_data, survivor_data, survivor_flags, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, survivor_count);
}

