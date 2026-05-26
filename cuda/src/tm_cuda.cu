// Treasure Master — CUDA checksum-screen kernel
//
// Design (from docs/gpu_forward_benchmark_notes.md):
//   - 128 threads/block, 4 warps/block.
//   - Checksum-screen kernel processes 4 candidates per warp (ILP4).
//   - Schedule loop is warp-synchronous (no block-wide sync).
//   - alg2 and alg5 use warp shuffles instead of shared-memory handoff.
//   - Checksum accumulation is a warp reduction.
//   - Other-world targets and checksum masks live in __constant__ memory.
//
// Measured (full 2^32 sweep, key_id=0x2CA5B42D):
//   - RTX 5090:                    90.4 M cand/s wall  (~47.5s per key)
//   - RTX PRO 6000 Blackwell Max-Q: 72.8 M cand/s wall (~59.0s per key)
//
// The remaining hot site is the dependent table-load chain in `run_alg`.

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <cstdint>

namespace
{
	__device__ __constant__ uint32_t kOtherWorldWords[32] = {
		0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
		0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
		0x00000000u, 0x00000000u, 0x00000000u, 0xC168CA00u,
		0x04D24466u, 0x8681900Bu, 0xE2D2F4C7u, 0x0CE322F1u,
		0xFFFB54D9u, 0x7281CF0Au, 0x989A940Au, 0x80ABFFD3u,
		0x45B7E59Au, 0xF0D28F6Eu, 0xAEB3FF67u, 0x069CBB49u,
		0xA3494012u, 0x7B32DB9Au, 0xB95AA158u, 0x6E2D2B2Bu,
		0x1A1C9336u, 0xE4180352u, 0xBDC1B15Eu, 0x50F1FB44u
	};

	__device__ __constant__ uint32_t kCarnivalWorldChecksumMaskWords[32] = {
		0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
		0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
		0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
		0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
		0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
		0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
		0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
		0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
	};

	__device__ __constant__ uint32_t kOtherWorldChecksumMaskWords[32] = {
		0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
		0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
		0x00000000u, 0x00000000u, 0x00000000u, 0xFF000000u,
		0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
		0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
		0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
		0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
		0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
	};

	__device__ __forceinline__ uint32_t pack_be_u32(uint32_t value)
	{
		return ((value & 0x000000FFu) << 24)
			| ((value & 0x0000FF00u) << 8)
			| ((value & 0x00FF0000u) >> 8)
			| ((value & 0xFF000000u) >> 24);
	}

	__device__ __forceinline__ uint32_t load_u32(const uint8_t* table, uint16_t rng_seed, uint32_t lane)
	{
		return reinterpret_cast<const uint32_t*>(table)[static_cast<uint32_t>(rng_seed) * 32u + lane];
	}

	__device__ __forceinline__ uint32_t expand_value(uint32_t value, uint32_t lane, uint16_t rng_seed, const uint8_t* expansion_values)
	{
		return __vadd4(value, load_u32(expansion_values, rng_seed, lane));
	}

	__device__ __forceinline__ uint32_t other_world_word(uint32_t value, uint32_t lane)
	{
		return value ^ kOtherWorldWords[lane];
	}

	__device__ __forceinline__ uint16_t sum_word_bytes(uint32_t value)
	{
		return static_cast<uint16_t>((value & 0xFFu)
			+ ((value >> 8) & 0xFFu)
			+ ((value >> 16) & 0xFFu)
			+ ((value >> 24) & 0xFFu));
	}

	__device__ __forceinline__ uint16_t masked_sum_word(uint32_t value, uint32_t lane, const uint32_t* mask_words)
	{
		return sum_word_bytes(value & mask_words[lane]);
	}

	__device__ __forceinline__ uint32_t run_alg(
		uint32_t value,
		uint32_t lane,
		uint8_t algorithm_id,
		uint16_t* rng_seed,
		const uint8_t* regular_rng_values,
		const uint8_t* alg0_values,
		const uint8_t* alg6_values,
		const uint16_t* rng_forward_1,
		const uint16_t* rng_forward_128,
		const uint32_t* alg2_values,
		const uint32_t* alg5_values)
	{
		if (algorithm_id == 0u)
		{
			value = ((value << 1) & 0xFEFEFEFEu) | load_u32(alg0_values, *rng_seed, lane);
		}
		else if (algorithm_id == 1u)
		{
			value = __vadd4(value, load_u32(regular_rng_values, *rng_seed, lane));
		}
		else if (algorithm_id == 3u)
		{
			value ^= load_u32(regular_rng_values, *rng_seed, lane);
		}
		else if (algorithm_id == 4u)
		{
			value = __vsub4(value, load_u32(regular_rng_values, *rng_seed, lane));
		}
		else if (algorithm_id == 6u)
		{
			value = ((value >> 1) & 0x7F7F7F7Fu) | load_u32(alg6_values, *rng_seed, lane);
		}
		else if (algorithm_id == 7u)
		{
			value ^= 0xFFFFFFFFu;
		}

		if (algorithm_id == 2u || algorithm_id == 5u)
		{
			const uint32_t next_lane_value = __shfl_down_sync(0xFFFFFFFFu, value, 1);

			if (algorithm_id == 2u)
			{
				uint32_t temp = (value & 0x00010000u) >> 8;
				if (lane == 31u)
				{
					temp |= alg2_values[*rng_seed];
				}
				else
				{
					temp |= ((next_lane_value & 0x00000001u) << 24);
				}
				temp |= (value >> 1) & 0x007F007Fu;
				temp |= (value << 1) & 0xFE00FE00u;
				temp |= (value >> 8) & 0x00800080u;
				value = temp;
			}
			else
			{
				uint32_t temp = (value & 0x00800000u) >> 8;
				if (lane == 31u)
				{
					temp |= alg5_values[*rng_seed];
				}
				else
				{
					temp |= ((next_lane_value & 0x00000080u) << 24);
				}
				temp |= (value >> 1) & 0x7F007F00u;
				temp |= (value << 1) & 0x00FE00FEu;
				temp |= (value >> 8) & 0x00010001u;
				value = temp;
			}
		}

		if (algorithm_id == 0u || algorithm_id == 1u || algorithm_id == 3u || algorithm_id == 4u || algorithm_id == 6u)
		{
			*rng_seed = rng_forward_128[*rng_seed];
		}
		else if (algorithm_id == 2u || algorithm_id == 5u)
		{
			*rng_seed = rng_forward_1[*rng_seed];
		}

		return value;
	}

	// ── HyperLogLog support ──────────────────────────────────────────────────
	// m = 4096 (2^12) registers, 12-bit index from the top of the hash,
	// rho from the remaining 52 bits.  Accuracy ≈ 1.6% relative error.
	// Registers are uint32_t (using only 6 bits worth of range in practice)
	// so that atomicMax works without needing packed-byte CAS.

	// Produce a warp-wide 64-bit hash of the 128-byte output state.
	// Each lane mixes its own 32-bit word with a lane-specific constant,
	// then all 32 per-lane values are XOR-folded across the warp via
	// butterfly shuffles so every lane ends up with the same combined hash.
	__device__ __forceinline__ void warp_hash_state(
		uint32_t value, uint32_t lane,
		uint32_t* h_lo_out, uint32_t* h_hi_out)
	{
		// Mix lane value with independent constants for lo and hi halves
		uint32_t lo = value + lane * 0x9e3779b9u;
		uint32_t hi = value + lane * 0x6b43a9b5u;
		// Murmur3 finalizer on each per-lane contribution
		lo ^= lo >> 16; lo *= 0x85ebca6bu; lo ^= lo >> 13; lo *= 0xc2b2ae35u; lo ^= lo >> 16;
		hi ^= hi >> 16; hi *= 0x85ebca6bu; hi ^= hi >> 13; hi *= 0xc2b2ae35u; hi ^= hi >> 16;
		// XOR-butterfly: after the loop every lane holds XOR of all lanes
		for (uint32_t off = 16u; off > 0u; off >>= 1u)
		{
			lo ^= __shfl_xor_sync(0xFFFFFFFFu, lo, off);
			hi ^= __shfl_xor_sync(0xFFFFFFFFu, hi, off);
		}
		*h_lo_out = lo;
		*h_hi_out = hi;
	}

	// Update a 4096-register HLL with one 64-bit hash value (call from lane 0 only).
	__device__ __forceinline__ void hll_update(uint32_t* hll_regs, uint32_t h_lo, uint32_t h_hi)
	{
		const uint64_t h  = ((uint64_t)h_hi << 32) | h_lo;
		const uint32_t idx = (uint32_t)(h >> 52);               // top 12 bits → [0, 4095]
		const uint64_t w   = h & 0x000FFFFFFFFFFFFFull;         // remaining 52 bits
		// __clzll(0) == 64 in CUDA; rho = (leading zeros from bit 51) + 1 = clzll - 11
		const uint32_t rho = (uint32_t)(__clzll(w) - 11u);
		atomicMax(&hll_regs[idx], rho);
	}
	// ─────────────────────────────────────────────────────────────────────────

	__device__ __forceinline__ uint32_t initialize_working_word(
		uint32_t key,
		uint32_t data,
		uint32_t lane,
		const uint8_t* expansion_values)
	{
		const uint32_t base_value = ((lane & 1u) == 0u) ? pack_be_u32(key) : pack_be_u32(data);
		return expand_value(base_value, lane, static_cast<uint16_t>(key >> 16), expansion_values);
	}

	template<uint32_t SCHEDULE_COUNT>
	__device__ __forceinline__ uint32_t run_schedule_t(
		uint32_t value,
		uint32_t lane,
		const uint8_t* regular_rng_values,
		const uint8_t* alg0_values,
		const uint8_t* alg6_values,
		const uint16_t* rng_forward_1,
		const uint16_t* rng_forward_128,
		const uint32_t* alg2_values,
		const uint32_t* alg5_values,
		const uint8_t* schedule_data)
	{
		for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
		{
			uint32_t packed_schedule = 0u;
			if (lane == 0u)
			{
				packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
			}
			packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

			uint16_t rng_seed = static_cast<uint16_t>(((packed_schedule & 0x000000FFu) << 8)
				| ((packed_schedule & 0x0000FF00u) >> 8));
			uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
				| ((packed_schedule & 0xFF000000u) >> 24));

			for (uint32_t i = 0; i < 16u; i++)
			{
				const uint32_t source_lane = i >> 2;
				const uint32_t source_shift = (i & 3u) * 8u;
				const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, value, source_lane);
				uint8_t current_byte = static_cast<uint8_t>((source_word >> source_shift) & 0xFFu);
				if ((nibble_selector & 0x8000u) != 0u)
				{
					current_byte = static_cast<uint8_t>(current_byte >> 4);
				}

				const uint8_t algorithm_id = static_cast<uint8_t>((current_byte >> 1) & 0x07u);
				value = run_alg(value, lane, algorithm_id, &rng_seed, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
				nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
			}
		}

		return value;
	}

	template<uint32_t SCHEDULE_COUNT>
	__device__ __forceinline__ void run_schedule_quad_t(
		uint32_t* value0,
		uint32_t* value1,
		uint32_t* value2,
		uint32_t* value3,
		uint32_t lane,
		const uint8_t* regular_rng_values,
		const uint8_t* alg0_values,
		const uint8_t* alg6_values,
		const uint16_t* rng_forward_1,
		const uint16_t* rng_forward_128,
		const uint32_t* alg2_values,
		const uint32_t* alg5_values,
		const uint8_t* schedule_data)
	{
		for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
		{
			uint32_t packed_schedule = 0u;
			if (lane == 0u)
			{
				packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
			}
			packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

			uint16_t rng_seed0 = static_cast<uint16_t>(((packed_schedule & 0x000000FFu) << 8)
				| ((packed_schedule & 0x0000FF00u) >> 8));
			uint16_t rng_seed1 = rng_seed0;
			uint16_t rng_seed2 = rng_seed0;
			uint16_t rng_seed3 = rng_seed0;
			uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
				| ((packed_schedule & 0xFF000000u) >> 24));

			for (uint32_t i = 0; i < 16u; i++)
			{
				const uint32_t source_lane = i >> 2;
				const uint32_t source_shift = (i & 3u) * 8u;
				const uint32_t source_word0 = __shfl_sync(0xFFFFFFFFu, *value0, source_lane);
				const uint32_t source_word1 = __shfl_sync(0xFFFFFFFFu, *value1, source_lane);
				const uint32_t source_word2 = __shfl_sync(0xFFFFFFFFu, *value2, source_lane);
				const uint32_t source_word3 = __shfl_sync(0xFFFFFFFFu, *value3, source_lane);
				uint8_t current_byte0 = static_cast<uint8_t>((source_word0 >> source_shift) & 0xFFu);
				uint8_t current_byte1 = static_cast<uint8_t>((source_word1 >> source_shift) & 0xFFu);
				uint8_t current_byte2 = static_cast<uint8_t>((source_word2 >> source_shift) & 0xFFu);
				uint8_t current_byte3 = static_cast<uint8_t>((source_word3 >> source_shift) & 0xFFu);
				if ((nibble_selector & 0x8000u) != 0u)
				{
					current_byte0 = static_cast<uint8_t>(current_byte0 >> 4);
					current_byte1 = static_cast<uint8_t>(current_byte1 >> 4);
					current_byte2 = static_cast<uint8_t>(current_byte2 >> 4);
					current_byte3 = static_cast<uint8_t>(current_byte3 >> 4);
				}

				const uint8_t algorithm_id0 = static_cast<uint8_t>((current_byte0 >> 1) & 0x07u);
				const uint8_t algorithm_id1 = static_cast<uint8_t>((current_byte1 >> 1) & 0x07u);
				const uint8_t algorithm_id2 = static_cast<uint8_t>((current_byte2 >> 1) & 0x07u);
				const uint8_t algorithm_id3 = static_cast<uint8_t>((current_byte3 >> 1) & 0x07u);
				*value0 = run_alg(*value0, lane, algorithm_id0, &rng_seed0, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
				*value1 = run_alg(*value1, lane, algorithm_id1, &rng_seed1, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
				*value2 = run_alg(*value2, lane, algorithm_id2, &rng_seed2, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
				*value3 = run_alg(*value3, lane, algorithm_id3, &rng_seed3, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
				nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
			}
		}
	}

	__device__ __forceinline__ uint8_t screen_candidate(uint32_t working_value, uint32_t lane, const uint8_t* carnival_data)
	{
		const uint32_t carnival_word = working_value ^ reinterpret_cast<const uint32_t*>(carnival_data)[lane];
		const uint32_t other_word = other_world_word(working_value, lane);

		uint32_t carnival_sum = masked_sum_word(carnival_word, lane, kCarnivalWorldChecksumMaskWords);
		uint32_t other_sum = masked_sum_word(other_word, lane, kOtherWorldChecksumMaskWords);
		for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
		{
			carnival_sum += __shfl_down_sync(0xFFFFFFFFu, carnival_sum, offset);
			other_sum += __shfl_down_sync(0xFFFFFFFFu, other_sum, offset);
		}

		uint16_t carnival_checksum_value = 0u;
		if (lane == 3u)
		{
			carnival_checksum_value = static_cast<uint16_t>(((carnival_word >> 16) & 0xFFu) << 8)
				| static_cast<uint16_t>((carnival_word >> 24) & 0xFFu);
		}

		uint16_t other_checksum_value = 0u;
		if (lane == 11u)
		{
			other_checksum_value = static_cast<uint16_t>(((other_word >> 8) & 0xFFu) << 8)
				| static_cast<uint16_t>((other_word >> 16) & 0xFFu);
		}

		carnival_checksum_value = __shfl_sync(0xFFFFFFFFu, carnival_checksum_value, 3u);
		other_checksum_value = __shfl_sync(0xFFFFFFFFu, other_checksum_value, 11u);

		if (lane != 0u)
		{
			return 0u;
		}

		// Check both checksums independently — do NOT short-circuit on carnival.
		// If only carnival passes:    0x08
		// If only other-world passes: 0x08 | 0x01
		// If both pass (dual):        0x08 | 0x01 | 0x02
		// The game executes carnival first when both pass, but we track both so
		// the other-world count is not undercounted in density experiments.
		const bool carnival_ok = (static_cast<uint16_t>(carnival_sum) == carnival_checksum_value);
		const bool other_ok    = (static_cast<uint16_t>(other_sum)    == other_checksum_value);

		if (other_ok && carnival_ok) return 0x08u | 0x01u | 0x02u;
		if (other_ok)                return 0x08u | 0x01u;
		if (carnival_ok)             return 0x08u;
		return 0u;
	}
}

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

// ====================================================================
// Offset-stream + ILP variant of the checksum-screen kernel.
// ====================================================================
//
// May 2026 advancement (~1.5× over the universal-table kernel above):
//   * Per-key OFFSET-STREAM RNG: host precomputes the (regular / alg0 /
//     alg6) byte streams and (alg2 / alg5) carry uint32 streams indexed
//     by (schedule_step, rng_offset_within_step). The kernel does an
//     indexed read instead of walking rng_seed via rng_forward_1/128.
//     `rng_offset[j]` advances by 128 per alg-0/1/3/4/6 and by 1 per
//     alg-2/5 (alg-7 doesn't advance).
//   * PreIDs: precompute all ILP candidates' algorithm_id from one
//     source_lane shuffle before dispatching alg-apply — lets the
//     compiler hoist the LDGs as a block.
//   * Packed-store helper: when ILP is divisible by 4, lane 0 packs
//     the flags into a uint32 and writes once.
//
// ILP4/6/8 wrappers are exposed; ILP6 is the empirical sweet spot on
// sm_120 Blackwell (~+5–8% over ILP8, +2 % over ILP4). ILP5/12/16 were
// tested and either tied or regressed.
namespace {

__device__ __forceinline__ uint32_t load_offset_u32(
	const uint8_t* table, uint32_t schedule_index, uint32_t rng_offset, uint32_t lane)
{
	return reinterpret_cast<const uint32_t*>(table)[((schedule_index * 2048u) + rng_offset) * 32u + lane];
}

__device__ __forceinline__ uint32_t run_alg_offset(
	uint32_t value,
	uint32_t lane,
	uint8_t algorithm_id,
	uint32_t schedule_index,
	uint32_t* rng_offset,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values)
{
	if (algorithm_id == 0u)
	{
		value = ((value << 1) & 0xFEFEFEFEu) | load_offset_u32(alg0_values, schedule_index, *rng_offset, lane);
		*rng_offset += 128u;
	}
	else if (algorithm_id == 1u)
	{
		value = __vadd4(value, load_offset_u32(regular_rng_values, schedule_index, *rng_offset, lane));
		*rng_offset += 128u;
	}
	else if (algorithm_id == 3u)
	{
		value ^= load_offset_u32(regular_rng_values, schedule_index, *rng_offset, lane);
		*rng_offset += 128u;
	}
	else if (algorithm_id == 4u)
	{
		value = __vsub4(value, load_offset_u32(regular_rng_values, schedule_index, *rng_offset, lane));
		*rng_offset += 128u;
	}
	else if (algorithm_id == 6u)
	{
		value = ((value >> 1) & 0x7F7F7F7Fu) | load_offset_u32(alg6_values, schedule_index, *rng_offset, lane);
		*rng_offset += 128u;
	}
	else if (algorithm_id == 7u)
	{
		value ^= 0xFFFFFFFFu;
	}

	if (algorithm_id == 2u || algorithm_id == 5u)
	{
		const uint32_t next_lane_value = __shfl_down_sync(0xFFFFFFFFu, value, 1);
		const uint32_t carry_index = schedule_index * 2048u + *rng_offset;
		if (algorithm_id == 2u)
		{
			uint32_t temp = (value & 0x00010000u) >> 8;
			if (lane == 31u) temp |= alg2_values[carry_index];
			else             temp |= ((next_lane_value & 0x00000001u) << 24);
			temp |= (value >> 1) & 0x007F007Fu;
			temp |= (value << 1) & 0xFE00FE00u;
			temp |= (value >> 8) & 0x00800080u;
			value = temp;
		}
		else
		{
			uint32_t temp = (value & 0x00800000u) >> 8;
			if (lane == 31u) temp |= alg5_values[carry_index];
			else             temp |= ((next_lane_value & 0x00000080u) << 24);
			temp |= (value >> 1) & 0x7F007F00u;
			temp |= (value << 1) & 0x00FE00FEu;
			temp |= (value >> 8) & 0x00010001u;
			value = temp;
		}
		*rng_offset += 1u;
	}
	return value;
}

// Store ILP flag bytes from lane 0. Uses packed uint32 stores when ILP
// is divisible by 4 (candidate_base is also a multiple of 4), per-byte
// fallback otherwise (ILP=6 etc.). Guards the tail-of-workunit case.
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
						result_data[candidate_base + j + k] = screen_flags[j + k];
				}
			}
		}
	}
	else
	{
		// ILP not divisible by 4 (e.g. 6) — per-byte stores.
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			if ((candidate_base + j) < candidate_count)
				result_data[candidate_base + j] = screen_flags[j];
		}
	}
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

	if (candidate_base >= candidate_count) return;

	uint32_t working_value[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
		working_value[j] = initialize_working_word(key, data_start + candidate_base + j, lane, expansion_values);

	// Note: do NOT add an explicit `#pragma unroll` here. NVCC's default
	// heuristic correctly partial-unrolls the 27-iter schedule loop; forcing
	// full unroll inflates code size ~8× and tanks throughput from I-cache
	// thrash (measured 5× slowdown).
	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

		uint32_t rng_offset[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) rng_offset[j] = 0u;

		uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
			| ((packed_schedule & 0xFF000000u) >> 24));

		for (uint32_t i = 0u; i < 16u; i++)
		{
			const uint32_t source_lane = i >> 2;
			const uint32_t source_shift = (i & 3u) * 8u;
			const uint32_t algorithm_shift = source_shift + 1u + ((static_cast<uint32_t>(nibble_selector) >> 13) & 4u);

			// PreIDs: precompute algorithm_id for all ILP candidates first.
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
				working_value[j] = run_alg_offset(working_value[j], lane, algorithm_id[j], schedule_index, &rng_offset[j],
					regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			}
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}
	}

	uint8_t screen_flags[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
		screen_flags[j] = screen_candidate(working_value[j], lane, carnival_data);

	if (lane == 0u)
		store_screen_flags_ilp<ILP>(result_data, candidate_base, candidate_count, screen_flags);
}

}  // namespace

extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp4_cuda(
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
	tm_checksum_screen_offset_store_ilp_impl<27u, 4u>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp6_cuda(
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
	tm_checksum_screen_offset_store_ilp_impl<27u, 6u>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
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

extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp4_cuda_skipcar(
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
	tm_checksum_screen_offset_store_ilp_impl<26u, 4u>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp6_cuda_skipcar(
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
	tm_checksum_screen_offset_store_ilp_impl<26u, 6u>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_offset_store_ilp8_cuda_skipcar(
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
	tm_checksum_screen_offset_store_ilp_impl<26u, 8u>(result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, expansion_values, schedule_data, carnival_data, key, data_start, candidate_count);
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
