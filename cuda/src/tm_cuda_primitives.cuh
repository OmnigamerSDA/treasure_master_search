// Treasure Master — CUDA device-function primitives.
//
// Contains all __device__ helpers used by every kernel in this build:
//   - __constant__ world-transform and checksum-mask tables
//   - Core per-lane helpers: byte-swap, table-load, expansion, world-XOR
//   - Eight-algorithm dispatch: run_alg (universal-table) and the offset-stream
//     variants used by the production ILP6 kernel
//   - Interleaved-offset stream loaders (schedule-grouped layout)
//   - HyperLogLog warp hash + register update
//   - State initialisation (initialize_working_word)
//   - Schedule runners (run_schedule_t, run_schedule_quad_t, offset variants)
//   - Checksum candidate screener (screen_candidate)
//   - Map-RNG POC infrastructure (maprng_pack_*, run_alg_maprng,
//     run_schedule_quad_maprng_t, run_schedule_single_maprng_t)
//
// All functions are __device__ __forceinline__ — they must compile into the
// same TU as their callers. Include this header only from tm_cuda.cu.

#pragma once
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

	__device__ __forceinline__ uint32_t load_offset_u32(
		const uint8_t* table,
		uint32_t schedule_index,
		uint32_t rng_offset,
		uint32_t lane)
	{
		const uint32_t* p = reinterpret_cast<const uint32_t*>(table) + ((schedule_index * 2048u) + rng_offset) * 32u + lane;
#ifdef RACEWAY_INSTRUMENT
		// mark this (map, rng_offset) RNG row as touched (working-set probe; benign 32-lane redundant write)
		g_rng_touched[schedule_index * 2048u + rng_offset] = 1u;
#endif
#ifdef RACEWAY_NONTEMPORAL
		// RDNA-native experiment: stream the single-use RNG rows non-temporally so they
		// don't allocate/pollute L0/L1 (measured L1 reuse was ~20%). Clang AMDGPU only;
		// guarded so the NVIDIA build never sees it.
		return __builtin_nontemporal_load(p);
#else
		return p[0];
#endif
	}

	// __ldg variant: routes the load through the read-only data cache path.
	// On sm_120 the L1/RO caches are unified so SASS may be identical, but
	// the __ldg() intrinsic also tells the compiler the memory is non-aliasing,
	// which can enable reordering of the LDG ahead of the dependent LOP3.
	__device__ __forceinline__ uint32_t load_offset_u32_ldg(
		const uint8_t* table,
		uint32_t schedule_index,
		uint32_t rng_offset,
		uint32_t lane)
	{
		return __ldg(reinterpret_cast<const uint32_t*>(table) + ((schedule_index * 2048u) + rng_offset) * 32u + lane);
	}

		__device__ __forceinline__ uint32_t run_alg_offset_ldg(
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
				value = ((value << 1) & 0xFEFEFEFEu) | load_offset_u32_ldg(alg0_values, schedule_index, *rng_offset, lane);
				*rng_offset += 128u;
			}
			else if (algorithm_id == 1u)
			{
				value = __vadd4(value, load_offset_u32_ldg(regular_rng_values, schedule_index, *rng_offset, lane));
				*rng_offset += 128u;
			}
			else if (algorithm_id == 3u)
			{
				value ^= load_offset_u32_ldg(regular_rng_values, schedule_index, *rng_offset, lane);
				*rng_offset += 128u;
			}
			else if (algorithm_id == 4u)
			{
				value = __vsub4(value, load_offset_u32_ldg(regular_rng_values, schedule_index, *rng_offset, lane));
				*rng_offset += 128u;
			}
			else if (algorithm_id == 6u)
			{
				value = ((value >> 1) & 0x7F7F7F7Fu) | load_offset_u32_ldg(alg6_values, schedule_index, *rng_offset, lane);
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
					if (lane == 31u)
					{
						temp |= __ldg(alg2_values + carry_index);
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
						temp |= __ldg(alg5_values + carry_index);
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
				*rng_offset += 1u;
			}

			return value;
		}

		// Software-prefetch apply phase. Consumes a value that was already
		// loaded into a register by a prior LDG (issued in the prefetch
		// phase), so this function has NO LDG for algs 0/1/3/4/6. Algs 2/5
		// keep their inline shuffle+carry path (different access pattern,
		// lane=31 only). Alg 7 is XOR-only, no load.
		__device__ __forceinline__ uint32_t apply_alg_with_prefetched(
			uint32_t value,
			uint32_t lane,
			uint8_t algorithm_id,
			uint32_t prefetched,
			uint32_t schedule_index,
			uint32_t* rng_offset,
			const uint32_t* alg2_values,
			const uint32_t* alg5_values)
		{
			if (algorithm_id == 0u)
			{
				value = ((value << 1) & 0xFEFEFEFEu) | prefetched;
				*rng_offset += 128u;
			}
			else if (algorithm_id == 1u)
			{
				value = __vadd4(value, prefetched);
				*rng_offset += 128u;
			}
			else if (algorithm_id == 3u)
			{
				value ^= prefetched;
				*rng_offset += 128u;
			}
			else if (algorithm_id == 4u)
			{
				value = __vsub4(value, prefetched);
				*rng_offset += 128u;
			}
			else if (algorithm_id == 6u)
			{
				value = ((value >> 1) & 0x7F7F7F7Fu) | prefetched;
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
					if (lane == 31u)
					{
						temp |= alg2_values[carry_index];
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
						temp |= alg5_values[carry_index];
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
				*rng_offset += 1u;
			}

			return value;
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
				if (lane == 31u)
				{
					temp |= alg2_values[carry_index];
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
					temp |= alg5_values[carry_index];
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
			*rng_offset += 1u;
		}

				return value;
			}

	// switch-dispatch variant of run_alg_offset (close-out of the AVX2-port "switch
	// beats if/else cascade" idea). On GPU the algorithm_id is warp-uniform, so no
	// divergence either way — expected ~neutral.
	__device__ __forceinline__ uint32_t run_alg_offset_sw(
		uint32_t value, uint32_t lane, uint8_t algorithm_id,
		uint32_t schedule_index, uint32_t* rng_offset,
		const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
		const uint32_t* alg2_values, const uint32_t* alg5_values)
	{
		switch (algorithm_id)
		{
			case 0u: value = ((value << 1) & 0xFEFEFEFEu) | load_offset_u32(alg0_values, schedule_index, *rng_offset, lane); *rng_offset += 128u; break;
			case 1u: value = __vadd4(value, load_offset_u32(regular_rng_values, schedule_index, *rng_offset, lane)); *rng_offset += 128u; break;
			case 3u: value ^= load_offset_u32(regular_rng_values, schedule_index, *rng_offset, lane); *rng_offset += 128u; break;
			case 4u: value = __vsub4(value, load_offset_u32(regular_rng_values, schedule_index, *rng_offset, lane)); *rng_offset += 128u; break;
			case 6u: value = ((value >> 1) & 0x7F7F7F7Fu) | load_offset_u32(alg6_values, schedule_index, *rng_offset, lane); *rng_offset += 128u; break;
			case 7u: value ^= 0xFFFFFFFFu; break;
			case 2u:
			{
				const uint32_t next_lane_value = __shfl_down_sync(0xFFFFFFFFu, value, 1);
				const uint32_t carry_index = schedule_index * 2048u + *rng_offset;
				uint32_t temp = (value & 0x00010000u) >> 8;
				if (lane == 31u) temp |= alg2_values[carry_index];
				else temp |= ((next_lane_value & 0x00000001u) << 24);
				temp |= (value >> 1) & 0x007F007Fu;
				temp |= (value << 1) & 0xFE00FE00u;
				temp |= (value >> 8) & 0x00800080u;
				value = temp; *rng_offset += 1u; break;
			}
			case 5u:
			{
				const uint32_t next_lane_value = __shfl_down_sync(0xFFFFFFFFu, value, 1);
				const uint32_t carry_index = schedule_index * 2048u + *rng_offset;
				uint32_t temp = (value & 0x00800000u) >> 8;
				if (lane == 31u) temp |= alg5_values[carry_index];
				else temp |= ((next_lane_value & 0x00000080u) << 24);
				temp |= (value >> 1) & 0x7F007F00u;
				temp |= (value << 1) & 0x00FE00FEu;
				temp |= (value >> 8) & 0x00010001u;
				value = temp; *rng_offset += 1u; break;
			}
			default: break;
		}
		return value;
	}

		__device__ __forceinline__ uint32_t run_alg_offset_carry_select(
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
					const uint32_t lane_carry = (next_lane_value & 0x00000001u) << 24;
					const uint32_t table_carry = (lane == 31u) ? alg2_values[carry_index] : 0u;
					uint32_t temp = (value & 0x00010000u) >> 8;
					temp |= (lane == 31u) ? table_carry : lane_carry;
					temp |= (value >> 1) & 0x007F007Fu;
					temp |= (value << 1) & 0xFE00FE00u;
					temp |= (value >> 8) & 0x00800080u;
					value = temp;
				}
				else
				{
					const uint32_t lane_carry = (next_lane_value & 0x00000080u) << 24;
					const uint32_t table_carry = (lane == 31u) ? alg5_values[carry_index] : 0u;
					uint32_t temp = (value & 0x00800000u) >> 8;
					temp |= (lane == 31u) ? table_carry : lane_carry;
					temp |= (value >> 1) & 0x7F007F00u;
					temp |= (value << 1) & 0x00FE00FEu;
					temp |= (value >> 8) & 0x00010001u;
					value = temp;
				}
				*rng_offset += 1u;
			}

			return value;
		}

			static constexpr uint32_t kOffsetScheduleStreamBytes = 2048u * 128u;
		static constexpr uint32_t kOffsetScheduleCarryBytes = 2048u * 4u;
		static constexpr uint32_t kOffsetInterleavedScheduleBytes =
			kOffsetScheduleStreamBytes * 3u + kOffsetScheduleCarryBytes * 2u;

		__device__ __forceinline__ uint32_t load_offset_interleaved_u32(
			const uint8_t* offset_values,
			uint32_t schedule_index,
			uint32_t stream_index,
			uint32_t rng_offset,
			uint32_t lane)
		{
			const uint8_t* schedule_base = offset_values + schedule_index * kOffsetInterleavedScheduleBytes;
			const uint8_t* stream_base = schedule_base + stream_index * kOffsetScheduleStreamBytes;
			return reinterpret_cast<const uint32_t*>(stream_base + rng_offset * 128u)[lane];
		}

		__device__ __forceinline__ uint32_t load_offset_interleaved_carry(
			const uint8_t* offset_values,
			uint32_t schedule_index,
			uint32_t carry_index,
			uint32_t rng_offset)
		{
			const uint8_t* schedule_base = offset_values + schedule_index * kOffsetInterleavedScheduleBytes;
			const uint8_t* carry_base = schedule_base + kOffsetScheduleStreamBytes * 3u + carry_index * kOffsetScheduleCarryBytes;
			return reinterpret_cast<const uint32_t*>(carry_base)[rng_offset];
		}

		__device__ __forceinline__ uint32_t run_alg_offset_interleaved(
			uint32_t value,
			uint32_t lane,
			uint8_t algorithm_id,
			uint32_t schedule_index,
			uint32_t* rng_offset,
			const uint8_t* offset_values)
		{
			if (algorithm_id == 0u)
			{
				value = ((value << 1) & 0xFEFEFEFEu) | load_offset_interleaved_u32(offset_values, schedule_index, 1u, *rng_offset, lane);
				*rng_offset += 128u;
			}
			else if (algorithm_id == 1u)
			{
				value = __vadd4(value, load_offset_interleaved_u32(offset_values, schedule_index, 0u, *rng_offset, lane));
				*rng_offset += 128u;
			}
			else if (algorithm_id == 3u)
			{
				value ^= load_offset_interleaved_u32(offset_values, schedule_index, 0u, *rng_offset, lane);
				*rng_offset += 128u;
			}
			else if (algorithm_id == 4u)
			{
				value = __vsub4(value, load_offset_interleaved_u32(offset_values, schedule_index, 0u, *rng_offset, lane));
				*rng_offset += 128u;
			}
			else if (algorithm_id == 6u)
			{
				value = ((value >> 1) & 0x7F7F7F7Fu) | load_offset_interleaved_u32(offset_values, schedule_index, 2u, *rng_offset, lane);
				*rng_offset += 128u;
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
						temp |= load_offset_interleaved_carry(offset_values, schedule_index, 0u, *rng_offset);
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
						temp |= load_offset_interleaved_carry(offset_values, schedule_index, 1u, *rng_offset);
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
				*rng_offset += 1u;
			}

			return value;
		}

	// ── HyperLogLog support ──────────────────────────────────────────────────
	// m = 4096 (2^12) registers, 12-bit index from the top of the hash,
	// rho from the remaining 52 bits.  Accuracy ≈ 1.6% relative error.
	// Registers are uint32_t (using only 6 bits worth of range in practice)
	// so that atomicMax works without needing packed-byte CAS.
	static const uint32_t kHllM = 4096u;

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

		template<uint32_t SCHEDULE_COUNT>
		__device__ __forceinline__ void run_schedule_quad_offset_t(
			uint32_t* value0,
			uint32_t* value1,
		uint32_t* value2,
		uint32_t* value3,
		uint32_t lane,
		const uint8_t* regular_rng_values,
		const uint8_t* alg0_values,
		const uint8_t* alg6_values,
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

			uint32_t rng_offset0 = 0u;
			uint32_t rng_offset1 = 0u;
			uint32_t rng_offset2 = 0u;
			uint32_t rng_offset3 = 0u;
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
				*value0 = run_alg_offset(*value0, lane, algorithm_id0, schedule_index, &rng_offset0, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
				*value1 = run_alg_offset(*value1, lane, algorithm_id1, schedule_index, &rng_offset1, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
				*value2 = run_alg_offset(*value2, lane, algorithm_id2, schedule_index, &rng_offset2, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
				*value3 = run_alg_offset(*value3, lane, algorithm_id3, schedule_index, &rng_offset3, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
				nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
				}
			}
		}

		template<uint32_t SCHEDULE_COUNT>
		__device__ __forceinline__ void run_schedule_quad_offset_interleaved_t(
			uint32_t* value0,
			uint32_t* value1,
			uint32_t* value2,
			uint32_t* value3,
			uint32_t lane,
			const uint8_t* offset_values,
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

				uint32_t rng_offset0 = 0u;
				uint32_t rng_offset1 = 0u;
				uint32_t rng_offset2 = 0u;
				uint32_t rng_offset3 = 0u;
				uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
					| ((packed_schedule & 0xFF000000u) >> 24));

				for (uint32_t i = 0; i < 16u; i++)
				{
					const uint32_t source_lane = i >> 2;
					const uint32_t source_shift = (i & 3u) * 8u;
					const uint32_t algorithm_shift = source_shift + 1u + ((static_cast<uint32_t>(nibble_selector) >> 13) & 4u);
					const uint32_t source_word0 = __shfl_sync(0xFFFFFFFFu, *value0, source_lane);
					const uint32_t source_word1 = __shfl_sync(0xFFFFFFFFu, *value1, source_lane);
					const uint32_t source_word2 = __shfl_sync(0xFFFFFFFFu, *value2, source_lane);
					const uint32_t source_word3 = __shfl_sync(0xFFFFFFFFu, *value3, source_lane);
					const uint8_t algorithm_id0 = static_cast<uint8_t>((source_word0 >> algorithm_shift) & 0x07u);
					const uint8_t algorithm_id1 = static_cast<uint8_t>((source_word1 >> algorithm_shift) & 0x07u);
					const uint8_t algorithm_id2 = static_cast<uint8_t>((source_word2 >> algorithm_shift) & 0x07u);
					const uint8_t algorithm_id3 = static_cast<uint8_t>((source_word3 >> algorithm_shift) & 0x07u);
					*value0 = run_alg_offset_interleaved(*value0, lane, algorithm_id0, schedule_index, &rng_offset0, offset_values);
					*value1 = run_alg_offset_interleaved(*value1, lane, algorithm_id1, schedule_index, &rng_offset1, offset_values);
					*value2 = run_alg_offset_interleaved(*value2, lane, algorithm_id2, schedule_index, &rng_offset2, offset_values);
					*value3 = run_alg_offset_interleaved(*value3, lane, algorithm_id3, schedule_index, &rng_offset3, offset_values);
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

	// ── map_rng POC support ─────────────────────────────────────────────────
	// Per-schedule pre-built raw RNG-output sequence. map_rng has shape
	// [SCHEDULE_COUNT][2048] bytes. Lane L reads 4 raw bytes per alg step
	// and extracts the alg-specific transformation inline. Convention matches
	// the universal-table layout (lane L = working_value bytes 4L..4L+3, and
	// universal_table[seed*128 + b] = output of (128-b)-th call), so:
	//   lane L byte i ← mr[base + pos + (127 - 4L - i)]
	// for the raw bytes used in alg_1/3/4 and the MSB-extracted bits used by
	// alg_0/6. alg_2/5 take a single 1-bit carry from mr[base + pos].

	__device__ __forceinline__ uint32_t maprng_pack_raw(const uint8_t* mr, uint32_t pos, uint32_t lane)
	{
		const uint32_t b0 = mr[pos + (127u - lane * 4u)];
		const uint32_t b1 = mr[pos + (126u - lane * 4u)];
		const uint32_t b2 = mr[pos + (125u - lane * 4u)];
		const uint32_t b3 = mr[pos + (124u - lane * 4u)];
		return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
	}
	__device__ __forceinline__ uint32_t maprng_pack_msb_lo(const uint8_t* mr, uint32_t pos, uint32_t lane)
	{
		const uint32_t b0 = (mr[pos + (127u - lane * 4u)] >> 7) & 1u;
		const uint32_t b1 = (mr[pos + (126u - lane * 4u)] >> 7) & 1u;
		const uint32_t b2 = (mr[pos + (125u - lane * 4u)] >> 7) & 1u;
		const uint32_t b3 = (mr[pos + (124u - lane * 4u)] >> 7) & 1u;
		return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
	}
	// alg_6 uses the FORWARD-stored convention (universal alg6_values_8[i] is
	// the i-th call, not (128-i)-th). Lane L byte k ← mr[pos + 4L + k].
	__device__ __forceinline__ uint32_t maprng_pack_msb_hi_fwd(const uint8_t* mr, uint32_t pos, uint32_t lane)
	{
		const uint32_t b0 = (uint32_t)(mr[pos + lane * 4u + 0u] & 0x80u);
		const uint32_t b1 = (uint32_t)(mr[pos + lane * 4u + 1u] & 0x80u);
		const uint32_t b2 = (uint32_t)(mr[pos + lane * 4u + 2u] & 0x80u);
		const uint32_t b3 = (uint32_t)(mr[pos + lane * 4u + 3u] & 0x80u);
		return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
	}

	__device__ __forceinline__ uint32_t run_alg_maprng(
		uint32_t value, uint32_t lane, uint8_t algorithm_id,
		uint32_t* pos, const uint8_t* mr_base)
	{
		if (algorithm_id == 0u)
		{
			value = ((value << 1) & 0xFEFEFEFEu) | maprng_pack_msb_lo(mr_base, *pos, lane);
			*pos += 128u;
		}
		else if (algorithm_id == 1u)
		{
			value = __vadd4(value, maprng_pack_raw(mr_base, *pos, lane));
			*pos += 128u;
		}
		else if (algorithm_id == 3u)
		{
			value ^= maprng_pack_raw(mr_base, *pos, lane);
			*pos += 128u;
		}
		else if (algorithm_id == 4u)
		{
			value = __vsub4(value, maprng_pack_raw(mr_base, *pos, lane));
			*pos += 128u;
		}
		else if (algorithm_id == 6u)
		{
			value = ((value >> 1) & 0x7F7F7F7Fu) | maprng_pack_msb_hi_fwd(mr_base, *pos, lane);
			*pos += 128u;
		}
		else if (algorithm_id == 7u)
		{
			value ^= 0xFFFFFFFFu;
		}

		if (algorithm_id == 2u || algorithm_id == 5u)
		{
			const uint32_t next_lane_value = __shfl_down_sync(0xFFFFFFFFu, value, 1);
			const uint8_t carry_byte = mr_base[*pos];  // 1-byte carry from current pos
			if (algorithm_id == 2u)
			{
				uint32_t temp = (value & 0x00010000u) >> 8;
				if (lane == 31u)
				{
					temp |= ((static_cast<uint32_t>(carry_byte) >> 7) & 1u) << 24;
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
					temp |= (static_cast<uint32_t>(carry_byte) & 0x80u) << 24;
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
			*pos += 1u;
		}
		return value;
	}

	template<uint32_t SCHEDULE_COUNT>
	__device__ __forceinline__ void run_schedule_quad_maprng_t(
		uint32_t* value0, uint32_t* value1, uint32_t* value2, uint32_t* value3,
		uint32_t lane, const uint8_t* schedule_data, const uint8_t* map_rng)
	{
		for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
		{
			uint32_t packed_schedule = 0u;
			if (lane == 0u)
			{
				packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
			}
			packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
			uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
				| ((packed_schedule & 0xFF000000u) >> 24));

			const uint8_t* mr_base = map_rng + schedule_index * 2048u;
			// Each candidate walks its own pos through the shared mr_base.
			// Algs (and therefore pos advances) are data-dependent and may diverge.
			uint32_t pos0 = 0u, pos1 = 0u, pos2 = 0u, pos3 = 0u;

			for (uint32_t i = 0; i < 16u; i++)
			{
				const uint32_t source_lane = i >> 2;
				const uint32_t source_shift = (i & 3u) * 8u;
				const uint32_t source_word0 = __shfl_sync(0xFFFFFFFFu, *value0, source_lane);
				const uint32_t source_word1 = __shfl_sync(0xFFFFFFFFu, *value1, source_lane);
				const uint32_t source_word2 = __shfl_sync(0xFFFFFFFFu, *value2, source_lane);
				const uint32_t source_word3 = __shfl_sync(0xFFFFFFFFu, *value3, source_lane);
				uint8_t cb0 = static_cast<uint8_t>((source_word0 >> source_shift) & 0xFFu);
				uint8_t cb1 = static_cast<uint8_t>((source_word1 >> source_shift) & 0xFFu);
				uint8_t cb2 = static_cast<uint8_t>((source_word2 >> source_shift) & 0xFFu);
				uint8_t cb3 = static_cast<uint8_t>((source_word3 >> source_shift) & 0xFFu);
				if ((nibble_selector & 0x8000u) != 0u)
				{
					cb0 = static_cast<uint8_t>(cb0 >> 4);
					cb1 = static_cast<uint8_t>(cb1 >> 4);
					cb2 = static_cast<uint8_t>(cb2 >> 4);
					cb3 = static_cast<uint8_t>(cb3 >> 4);
				}
				const uint8_t alg0 = static_cast<uint8_t>((cb0 >> 1) & 0x07u);
				const uint8_t alg1 = static_cast<uint8_t>((cb1 >> 1) & 0x07u);
				const uint8_t alg2 = static_cast<uint8_t>((cb2 >> 1) & 0x07u);
				const uint8_t alg3 = static_cast<uint8_t>((cb3 >> 1) & 0x07u);
				*value0 = run_alg_maprng(*value0, lane, alg0, &pos0, mr_base);
				*value1 = run_alg_maprng(*value1, lane, alg1, &pos1, mr_base);
				*value2 = run_alg_maprng(*value2, lane, alg2, &pos2, mr_base);
				*value3 = run_alg_maprng(*value3, lane, alg3, &pos3, mr_base);
				nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
			}
		}
	}

	// Per-candidate variant — each candidate walks its own pos through
	// its own per-candidate-but-shared mr_base. Correct under alg divergence.
	template<uint32_t SCHEDULE_COUNT>
	__device__ __forceinline__ uint32_t run_schedule_single_maprng_t(
		uint32_t value, uint32_t lane, const uint8_t* schedule_data, const uint8_t* map_rng)
	{
		for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
		{
			uint32_t packed_schedule = 0u;
			if (lane == 0u)
			{
				packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
			}
			packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
			uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
				| ((packed_schedule & 0xFF000000u) >> 24));

			const uint8_t* mr_base = map_rng + schedule_index * 2048u;
			uint32_t pos = 0u;

			for (uint32_t i = 0; i < 16u; i++)
			{
				const uint32_t source_lane = i >> 2;
				const uint32_t source_shift = (i & 3u) * 8u;
				const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, value, source_lane);
				uint8_t cb = static_cast<uint8_t>((source_word >> source_shift) & 0xFFu);
				if ((nibble_selector & 0x8000u) != 0u) cb >>= 4;
				const uint8_t algorithm_id = static_cast<uint8_t>((cb >> 1) & 0x07u);
				value = run_alg_maprng(value, lane, algorithm_id, &pos, mr_base);
				nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
			}
		}
		return value;
	}
}
