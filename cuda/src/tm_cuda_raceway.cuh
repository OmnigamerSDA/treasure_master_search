// Treasure Master - CUDA raceway: the PRODUCTION forward-search engine.
//
// Bounded-wave raceway (cap-span + wave-local compaction + fixed-capacity fingerprint
// caps). As of 2026-06-16 this is the production default for any system — best across
// BOTH throughput and memory. The flat screen (tm_cuda_screen.cuh) and on-GPU
// compaction (tm_cuda_dedup.cuh) paths are retained as RESEARCH / A-B comparisons.
// Per-device span-ILP is tuned by `test_cuda --calibrate-raceway` (auto-applied).
//
// Include this header only from tm_cuda.cu, after tm_cuda_dedup.cuh. It uses the
// existing warp-native state layout, run_alg(), screen_candidate(), and
// tm_strong64_state() helpers.

#pragma once
#include <cstdint>

struct tm_raceway_stats
{
	unsigned long long reps_started;
	unsigned long long reps_completed;
	unsigned long long reps_dropped;
	unsigned long long map_evals;
};

__device__ __forceinline__ uint32_t tm_raceway_run_one_map(
	uint32_t value,
	uint32_t lane,
	uint32_t schedule_index,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* schedule_data)
{
	uint32_t packed_schedule = 0u;
	if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
	packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

	uint16_t rng_seed = static_cast<uint16_t>(((packed_schedule & 0x000000FFu) << 8)
		| ((packed_schedule & 0x0000FF00u) >> 8));
	uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
		| ((packed_schedule & 0xFF000000u) >> 24));

	for (uint32_t i = 0u; i < 16u; i++)
	{
		const uint32_t source_lane = i >> 2;
		const uint32_t source_shift = (i & 3u) * 8u;
		const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, value, source_lane);
		uint8_t current_byte = static_cast<uint8_t>((source_word >> source_shift) & 0xFFu);
		if ((nibble_selector & 0x8000u) != 0u) current_byte = static_cast<uint8_t>(current_byte >> 4);

		const uint8_t algorithm_id = static_cast<uint8_t>((current_byte >> 1) & 0x07u);
		value = run_alg(value, lane, algorithm_id, &rng_seed,
			regular_rng_values, alg0_values, alg6_values,
			rng_forward_1, rng_forward_128, alg2_values, alg5_values);
		nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
	}

	return value;
}

__device__ __forceinline__ uint32_t tm_raceway_run_one_map_offset(
	uint32_t value,
	uint32_t lane,
	uint32_t schedule_index,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* schedule_data)
{
	uint32_t packed_schedule = 0u;
	if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
	packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

	uint32_t rng_offset = 0u;
	uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
		| ((packed_schedule & 0xFF000000u) >> 24));

	for (uint32_t i = 0u; i < 16u; i++)
	{
		const uint32_t source_lane = i >> 2;
		const uint32_t source_shift = (i & 3u) * 8u;
		const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, value, source_lane);
		uint8_t current_byte = static_cast<uint8_t>((source_word >> source_shift) & 0xFFu);
		if ((nibble_selector & 0x8000u) != 0u) current_byte = static_cast<uint8_t>(current_byte >> 4);

		const uint8_t algorithm_id = static_cast<uint8_t>((current_byte >> 1) & 0x07u);
		value = run_alg_offset(value, lane, algorithm_id, schedule_index, &rng_offset,
			regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
		nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
	}

	return value;
}

// Benign-race fixed-cap probe. Return true when fp was already resident (drop),
// false when this state should continue. A race can evict/forget a fingerprint,
// which only over-keeps; it never manufactures a resident match for a different
// fingerprint except the accepted fp64 collision risk.
__device__ __forceinline__ bool tm_raceway_cap_probe_or_keep(
	unsigned long long fp,
	unsigned long long* table,
	uint32_t cap_bits,
	uint32_t cap_ways)
{
	if (table == nullptr || cap_bits == 0u || cap_bits > 32u || cap_ways == 0u) return false;
	if (fp == 0ull) fp = 1ull;

	const uint32_t bucket = static_cast<uint32_t>((fp * 0x9E3779B97F4A7C15ull) >> (64u - cap_bits));
	const unsigned long long base = static_cast<unsigned long long>(bucket) * cap_ways;

	for (uint32_t w = 0u; w < cap_ways; w++)
	{
		const unsigned long long cur = table[base + w];
		if (cur == fp) return true;
	}

	const uint32_t victim = static_cast<uint32_t>((fp >> 32) ^ (fp >> 17) ^ fp) % cap_ways;
	table[base + victim] = fp;
	return false;
}

extern "C" __global__ void tm_raceway_cap_clear_cuda(
	unsigned long long* table,
	uint64_t slots)
{
	const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (i < slots) table[i] = 0ull;
}

// Persistent data-seeded raceway. Input reps are data values from the MAP1
// pre-pass. Existing MAP1 frontier kernels usually emit absolute data values;
// set rep_values_are_absolute=0 only for a relative-offset list. The kernel
// recomputes MAP1 to materialize the post-MAP1 state, then streams through the
// remaining schedule. Set first_cap_map to the first completed map boundary that
// should be cap-probed; for MAP2+ caps, pass 1.
//
// cap_tables/cap_bits/cap_ways are arrays of length cap_count. Boundary map m
// uses index m - first_cap_map after map m has completed.
extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_stream_offsets_cuda(
	const uint32_t* rep_offsets,
	uint32_t rep_count,
	uint32_t* work_counter,
	uint32_t* survivor_offsets,
	uint32_t* survivor_count,
	uint8_t* survivor_flags,
	uint8_t* drop_map_out,
	tm_raceway_stats* stats,
	unsigned long long** cap_tables,
	const uint32_t* cap_bits,
	const uint32_t* cap_ways,
	uint32_t cap_count,
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
	uint32_t rep_values_are_absolute,
	uint32_t schedule_count,
	uint32_t first_cap_map)
{
	const uint32_t lane = threadIdx.x & 31u;
	unsigned long long local_started = 0ull;
	unsigned long long local_completed = 0ull;
	unsigned long long local_dropped = 0ull;
	unsigned long long local_maps = 0ull;

	while (true)
	{
		uint32_t task = 0u;
		if (lane == 0u) task = atomicAdd(work_counter, 1u);
		task = __shfl_sync(0xFFFFFFFFu, task, 0);
		if (task >= rep_count) break;

		local_started++;
		const uint32_t rep_value = rep_offsets[task];
		const uint32_t data = (rep_values_are_absolute != 0u) ? rep_value : (data_start + rep_value);
		uint32_t value = initialize_working_word(key, data, lane, expansion_values);
		bool alive = true;
		uint32_t drop_map = 0xFFFFFFFFu;

		for (uint32_t map = 0u; map < schedule_count; map++)
		{
			value = tm_raceway_run_one_map(value, lane, map,
				regular_rng_values, alg0_values, alg6_values,
				rng_forward_1, rng_forward_128, alg2_values, alg5_values,
				schedule_data);
			local_maps++;

			const bool has_cap = cap_tables != nullptr && cap_bits != nullptr && cap_ways != nullptr
				&& map >= first_cap_map && (map - first_cap_map) < cap_count;
			if (has_cap)
			{
				const uint32_t cap_idx = map - first_cap_map;
				const uint64_t fp = tm_strong64_state(value, lane);
				bool drop = false;
				if (lane == 0u)
				{
					drop = tm_raceway_cap_probe_or_keep(static_cast<unsigned long long>(fp),
						cap_tables[cap_idx], cap_bits[cap_idx], cap_ways[cap_idx]);
				}
				drop = __shfl_sync(0xFFFFFFFFu, drop, 0);
				if (drop)
				{
					alive = false;
					drop_map = map;
					local_dropped++;
					break;
				}
			}
		}

		if (alive)
		{
			uint8_t flag = 0u;
			if (carnival_data != nullptr) flag = screen_candidate(value, lane, carnival_data);
			if (lane == 0u)
			{
				const uint32_t out = atomicAdd(survivor_count, 1u);
				if (survivor_offsets != nullptr) survivor_offsets[out] = rep_offsets[task];
				if (survivor_flags != nullptr) survivor_flags[out] = flag;
			}
			local_completed++;
		}
		else if (drop_map_out != nullptr && lane == 0u)
		{
			drop_map_out[task] = static_cast<uint8_t>(drop_map);
		}
	}

	if (lane == 0u && stats != nullptr)
	{
		atomicAdd(&stats->reps_started, local_started);
		atomicAdd(&stats->reps_completed, local_completed);
		atomicAdd(&stats->reps_dropped, local_dropped);
		atomicAdd(&stats->map_evals, local_maps);
	}
}

// Direct full-window raceway. This is the normal forward shape: candidates enter
// at data_start, MAP1 is processed inline, and optional boundary caps can drop
// states after completed map boundaries. With caps disabled, output_flags[task]
// is a one-to-one parity surface against the flat screener.
extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_stream_window_cuda(
	uint32_t candidate_count,
	uint32_t data_start,
	uint32_t* work_counter,
	uint8_t* output_flags,
	uint8_t* drop_map_out,
	tm_raceway_stats* stats,
	unsigned long long** cap_tables,
	const uint32_t* cap_bits,
	const uint32_t* cap_ways,
	uint32_t cap_count,
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
	uint32_t schedule_count,
	uint32_t first_cap_map)
{
	const uint32_t lane = threadIdx.x & 31u;
	unsigned long long local_started = 0ull;
	unsigned long long local_completed = 0ull;
	unsigned long long local_dropped = 0ull;
	unsigned long long local_maps = 0ull;

	while (true)
	{
		uint32_t task = 0u;
		if (lane == 0u) task = atomicAdd(work_counter, 1u);
		task = __shfl_sync(0xFFFFFFFFu, task, 0);
		if (task >= candidate_count) break;

		local_started++;
		const uint32_t data = data_start + task;
		uint32_t value = initialize_working_word(key, data, lane, expansion_values);
		bool alive = true;
		uint32_t drop_map = 0xFFFFFFFFu;

		for (uint32_t map = 0u; map < schedule_count; map++)
		{
			value = tm_raceway_run_one_map(value, lane, map,
				regular_rng_values, alg0_values, alg6_values,
				rng_forward_1, rng_forward_128, alg2_values, alg5_values,
				schedule_data);
			local_maps++;

			const bool has_cap = cap_tables != nullptr && cap_bits != nullptr && cap_ways != nullptr
				&& map >= first_cap_map && (map - first_cap_map) < cap_count;
			if (has_cap)
			{
				const uint32_t cap_idx = map - first_cap_map;
				const uint64_t fp = tm_strong64_state(value, lane);
				bool drop = false;
				if (lane == 0u)
				{
					drop = tm_raceway_cap_probe_or_keep(static_cast<unsigned long long>(fp),
						cap_tables[cap_idx], cap_bits[cap_idx], cap_ways[cap_idx]);
				}
				drop = __shfl_sync(0xFFFFFFFFu, drop, 0);
				if (drop)
				{
					alive = false;
					drop_map = map;
					local_dropped++;
					break;
				}
			}
		}

		if (alive)
		{
			uint8_t flag = 0u;
			if (carnival_data != nullptr) flag = screen_candidate(value, lane, carnival_data);
			if (lane == 0u && output_flags != nullptr) output_flags[task] = flag;
			local_completed++;
		}
		else if (drop_map_out != nullptr && lane == 0u)
		{
			drop_map_out[task] = static_cast<uint8_t>(drop_map);
		}
	}

	if (lane == 0u && stats != nullptr)
	{
		atomicAdd(&stats->reps_started, local_started);
		atomicAdd(&stats->reps_completed, local_completed);
		atomicAdd(&stats->reps_dropped, local_dropped);
		atomicAdd(&stats->map_evals, local_maps);
	}
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_stream_window_offset_cuda(
	uint32_t candidate_count,
	uint32_t data_start,
	uint32_t* work_counter,
	uint8_t* output_flags,
	uint8_t* drop_map_out,
	tm_raceway_stats* stats,
	unsigned long long** cap_tables,
	const uint32_t* cap_bits,
	const uint32_t* cap_ways,
	uint32_t cap_count,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t schedule_count,
	uint32_t first_cap_map)
{
	const uint32_t lane = threadIdx.x & 31u;
	unsigned long long local_started = 0ull;
	unsigned long long local_completed = 0ull;
	unsigned long long local_dropped = 0ull;
	unsigned long long local_maps = 0ull;

	while (true)
	{
		uint32_t task = 0u;
		if (lane == 0u) task = atomicAdd(work_counter, 1u);
		task = __shfl_sync(0xFFFFFFFFu, task, 0);
		if (task >= candidate_count) break;

		local_started++;
		const uint32_t data = data_start + task;
		uint32_t value = initialize_working_word(key, data, lane, expansion_values);
		bool alive = true;
		uint32_t drop_map = 0xFFFFFFFFu;

		for (uint32_t map = 0u; map < schedule_count; map++)
		{
			value = tm_raceway_run_one_map_offset(value, lane, map,
				offset_regular_rng_values, offset_alg0_values, offset_alg6_values,
				offset_alg2_values, offset_alg5_values, schedule_data);
			local_maps++;

			const bool has_cap = cap_tables != nullptr && cap_bits != nullptr && cap_ways != nullptr
				&& map >= first_cap_map && (map - first_cap_map) < cap_count;
			if (has_cap)
			{
				const uint32_t cap_idx = map - first_cap_map;
				const uint64_t fp = tm_strong64_state(value, lane);
				bool drop = false;
				if (lane == 0u)
				{
					drop = tm_raceway_cap_probe_or_keep(static_cast<unsigned long long>(fp),
						cap_tables[cap_idx], cap_bits[cap_idx], cap_ways[cap_idx]);
				}
				drop = __shfl_sync(0xFFFFFFFFu, drop, 0);
				if (drop)
				{
					alive = false;
					drop_map = map;
					local_dropped++;
					break;
				}
			}
		}

		if (alive)
		{
			uint8_t flag = 0u;
			if (carnival_data != nullptr) flag = screen_candidate(value, lane, carnival_data);
			if (lane == 0u && output_flags != nullptr) output_flags[task] = flag;
			local_completed++;
		}
		else if (drop_map_out != nullptr && lane == 0u)
		{
			drop_map_out[task] = static_cast<uint8_t>(drop_map);
		}
	}

	if (lane == 0u && stats != nullptr)
	{
		atomicAdd(&stats->reps_started, local_started);
		atomicAdd(&stats->reps_completed, local_completed);
		atomicAdd(&stats->reps_dropped, local_dropped);
		atomicAdd(&stats->map_evals, local_maps);
	}
}

// Boundary-cap mark pass for the first raceway compaction foundation.
// The signature intentionally matches the direct offset-stream kernel so the
// host can reuse cap allocation and launch plumbing. output_flags is alive_out:
// lane 0 writes 1 for a candidate that survived the configured cap boundaries,
// or 0 for a candidate dropped by a resident cap hit. No final screening occurs.
extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_boundary_cap_mark_offset_cuda(
	uint32_t candidate_count,
	uint32_t data_start,
	uint32_t* work_counter,
	uint8_t* alive_out,
	uint8_t* drop_map_out,
	tm_raceway_stats* stats,
	unsigned long long** cap_tables,
	const uint32_t* cap_bits,
	const uint32_t* cap_ways,
	uint32_t cap_count,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t schedule_count,
	uint32_t first_cap_map)
{
	(void)carnival_data;
	const uint32_t lane = threadIdx.x & 31u;
	unsigned long long local_started = 0ull;
	unsigned long long local_completed = 0ull;
	unsigned long long local_dropped = 0ull;
	unsigned long long local_maps = 0ull;
	const uint32_t last_map = (cap_count == 0u) ? 0u : min(schedule_count, first_cap_map + cap_count);

	while (true)
	{
		uint32_t task = 0u;
		if (lane == 0u) task = atomicAdd(work_counter, 1u);
		task = __shfl_sync(0xFFFFFFFFu, task, 0);
		if (task >= candidate_count) break;

		local_started++;
		const uint32_t data = data_start + task;
		uint32_t value = initialize_working_word(key, data, lane, expansion_values);
		bool alive = true;
		uint32_t drop_map = 0xFFFFFFFFu;

		for (uint32_t map = 0u; map < last_map; map++)
		{
			value = tm_raceway_run_one_map_offset(value, lane, map,
				offset_regular_rng_values, offset_alg0_values, offset_alg6_values,
				offset_alg2_values, offset_alg5_values, schedule_data);
			local_maps++;

			const bool has_cap = cap_tables != nullptr && cap_bits != nullptr && cap_ways != nullptr
				&& map >= first_cap_map && (map - first_cap_map) < cap_count;
			if (has_cap)
			{
				const uint32_t cap_idx = map - first_cap_map;
				const uint64_t fp = tm_strong64_state(value, lane);
				bool drop = false;
				if (lane == 0u)
				{
					drop = tm_raceway_cap_probe_or_keep(static_cast<unsigned long long>(fp),
						cap_tables[cap_idx], cap_bits[cap_idx], cap_ways[cap_idx]);
				}
				drop = __shfl_sync(0xFFFFFFFFu, drop, 0);
				if (drop)
				{
					alive = false;
					drop_map = map;
					local_dropped++;
					break;
				}
			}
		}

		if (lane == 0u)
		{
			if (alive_out != nullptr) alive_out[task] = alive ? 1u : 0u;
			if (!alive && drop_map_out != nullptr) drop_map_out[task] = static_cast<uint8_t>(drop_map);
		}
		if (alive) local_completed++;
	}

	if (lane == 0u && stats != nullptr)
	{
		atomicAdd(&stats->reps_started, local_started);
		atomicAdd(&stats->reps_completed, local_completed);
		atomicAdd(&stats->reps_dropped, local_dropped);
		atomicAdd(&stats->map_evals, local_maps);
	}
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_boundary_cap_state_offset_cuda(
	uint32_t candidate_count,
	uint32_t data_start,
	uint32_t* work_counter,
	uint8_t* alive_out,
	uint8_t* drop_map_out,
	tm_raceway_stats* stats,
	uint32_t* state,
	unsigned long long** cap_tables,
	const uint32_t* cap_bits,
	const uint32_t* cap_ways,
	uint32_t cap_count,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	uint32_t key,
	uint32_t schedule_count,
	uint32_t first_cap_map)
{
	const uint32_t lane = threadIdx.x & 31u;
	unsigned long long local_started = 0ull;
	unsigned long long local_completed = 0ull;
	unsigned long long local_dropped = 0ull;
	unsigned long long local_maps = 0ull;
	const uint32_t last_map = (cap_count == 0u) ? 0u : min(schedule_count, first_cap_map + cap_count);

	while (true)
	{
		uint32_t task = 0u;
		if (lane == 0u) task = atomicAdd(work_counter, 1u);
		task = __shfl_sync(0xFFFFFFFFu, task, 0);
		if (task >= candidate_count) break;

		local_started++;
		const uint32_t data = data_start + task;
		uint32_t value = initialize_working_word(key, data, lane, expansion_values);
		bool alive = true;
		uint32_t drop_map = 0xFFFFFFFFu;

		for (uint32_t map = 0u; map < last_map; map++)
		{
			value = tm_raceway_run_one_map_offset(value, lane, map,
				offset_regular_rng_values, offset_alg0_values, offset_alg6_values,
				offset_alg2_values, offset_alg5_values, schedule_data);
			local_maps++;

			const bool has_cap = cap_tables != nullptr && cap_bits != nullptr && cap_ways != nullptr
				&& map >= first_cap_map && (map - first_cap_map) < cap_count;
			if (has_cap)
			{
				const uint32_t cap_idx = map - first_cap_map;
				const uint64_t fp = tm_strong64_state(value, lane);
				bool drop = false;
				if (lane == 0u)
				{
					drop = tm_raceway_cap_probe_or_keep(static_cast<unsigned long long>(fp),
						cap_tables[cap_idx], cap_bits[cap_idx], cap_ways[cap_idx]);
				}
				drop = __shfl_sync(0xFFFFFFFFu, drop, 0);
				if (drop)
				{
					alive = false;
					drop_map = map;
					local_dropped++;
					break;
				}
			}
		}

		if (alive)
		{
			if (state != nullptr) state[static_cast<size_t>(task) * 32u + lane] = value;
			local_completed++;
		}
		if (lane == 0u)
		{
			if (alive_out != nullptr) alive_out[task] = alive ? 1u : 0u;
			if (!alive && drop_map_out != nullptr) drop_map_out[task] = static_cast<uint8_t>(drop_map);
		}
	}

	if (lane == 0u && stats != nullptr)
	{
		atomicAdd(&stats->reps_started, local_started);
		atomicAdd(&stats->reps_completed, local_completed);
		atomicAdd(&stats->reps_dropped, local_dropped);
		atomicAdd(&stats->map_evals, local_maps);
	}
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_stream_window_offset_ilp4_cuda(
	uint32_t candidate_count,
	uint32_t data_start,
	uint32_t* work_counter,
	uint8_t* output_flags,
	uint8_t* drop_map_out,
	tm_raceway_stats* stats,
	unsigned long long** cap_tables,
	const uint32_t* cap_bits,
	const uint32_t* cap_ways,
	uint32_t cap_count,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t schedule_count,
	uint32_t first_cap_map)
{
	(void)drop_map_out;
	(void)cap_tables;
	(void)cap_bits;
	(void)cap_ways;
	(void)first_cap_map;

	const uint32_t lane = threadIdx.x & 31u;
	unsigned long long local_started = 0ull;
	unsigned long long local_completed = 0ull;
	unsigned long long local_maps = 0ull;

	while (true)
	{
		uint32_t task_base = 0u;
		if (lane == 0u) task_base = atomicAdd(work_counter, 4u);
		task_base = __shfl_sync(0xFFFFFFFFu, task_base, 0);
		if (task_base >= candidate_count) break;

		const uint32_t valid = min(4u, candidate_count - task_base);
		uint32_t value0 = initialize_working_word(key, data_start + task_base + 0u, lane, expansion_values);
		uint32_t value1 = initialize_working_word(key, data_start + task_base + 1u, lane, expansion_values);
		uint32_t value2 = initialize_working_word(key, data_start + task_base + 2u, lane, expansion_values);
		uint32_t value3 = initialize_working_word(key, data_start + task_base + 3u, lane, expansion_values);

		// Current POC ILP path is deliberately no-cap and all-map only. It isolates
		// instruction-level parallelism effects from cap/drop control flow.
		if (cap_count == 0u && schedule_count == 27u)
		{
			run_schedule_quad_offset_t<27u>(&value0, &value1, &value2, &value3, lane,
				offset_regular_rng_values, offset_alg0_values, offset_alg6_values,
				offset_alg2_values, offset_alg5_values, schedule_data);
		}

		const uint8_t flag0 = (carnival_data != nullptr) ? screen_candidate(value0, lane, carnival_data) : 0u;
		const uint8_t flag1 = (carnival_data != nullptr) ? screen_candidate(value1, lane, carnival_data) : 0u;
		const uint8_t flag2 = (carnival_data != nullptr) ? screen_candidate(value2, lane, carnival_data) : 0u;
		const uint8_t flag3 = (carnival_data != nullptr) ? screen_candidate(value3, lane, carnival_data) : 0u;

		if (lane == 0u && output_flags != nullptr)
		{
			if (valid == 4u)
			{
				const uint32_t packed = static_cast<uint32_t>(flag0)
					| (static_cast<uint32_t>(flag1) << 8)
					| (static_cast<uint32_t>(flag2) << 16)
					| (static_cast<uint32_t>(flag3) << 24);
				reinterpret_cast<uint32_t*>(output_flags)[task_base >> 2] = packed;
			}
			else
			{
				output_flags[task_base] = flag0;
				if (valid > 1u) output_flags[task_base + 1u] = flag1;
				if (valid > 2u) output_flags[task_base + 2u] = flag2;
			}
		}

		local_started += valid;
		local_completed += valid;
		local_maps += static_cast<unsigned long long>(valid) * 27ull;
	}

	if (lane == 0u && stats != nullptr)
	{
		atomicAdd(&stats->reps_started, local_started);
		atomicAdd(&stats->reps_completed, local_completed);
		atomicAdd(&stats->map_evals, local_maps);
	}
}

template<uint32_t ILP>
__device__ __forceinline__ void tm_raceway_stream_window_offset_ilp_body(
	uint32_t candidate_count,
	uint32_t data_start,
	uint32_t* work_counter,
	uint8_t* output_flags,
	tm_raceway_stats* stats,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key)
{
	const uint32_t lane = threadIdx.x & 31u;
	unsigned long long local_started = 0ull;
	unsigned long long local_completed = 0ull;
	unsigned long long local_maps = 0ull;

	while (true)
	{
		uint32_t task_base = 0u;
		if (lane == 0u) task_base = atomicAdd(work_counter, ILP);
		task_base = __shfl_sync(0xFFFFFFFFu, task_base, 0);
		if (task_base >= candidate_count) break;

		const uint32_t valid = min(ILP, candidate_count - task_base);
		uint32_t working_value[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; ++j)
		{
			working_value[j] = initialize_working_word(key, data_start + task_base + j, lane, expansion_values);
		}

		for (uint32_t schedule_index = 0u; schedule_index < 27u; ++schedule_index)
		{
			uint32_t packed_schedule = 0u;
			if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
			packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

			uint32_t rng_offset[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; ++j) rng_offset[j] = 0u;

			uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
				| ((packed_schedule & 0xFF000000u) >> 24));

			for (uint32_t i = 0u; i < 16u; ++i)
			{
				const uint32_t source_lane = i >> 2;
				const uint32_t source_shift = (i & 3u) * 8u;
				const uint32_t algorithm_shift = source_shift + 1u + ((static_cast<uint32_t>(nibble_selector) >> 13) & 4u);
				uint8_t algorithm_id[ILP];
				#pragma unroll
				for (uint32_t j = 0u; j < ILP; ++j)
				{
					const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, working_value[j], source_lane);
					algorithm_id[j] = static_cast<uint8_t>((source_word >> algorithm_shift) & 0x07u);
				}
				#pragma unroll
				for (uint32_t j = 0u; j < ILP; ++j)
				{
					working_value[j] = run_alg_offset(working_value[j], lane, algorithm_id[j], schedule_index, &rng_offset[j],
						offset_regular_rng_values, offset_alg0_values, offset_alg6_values, offset_alg2_values, offset_alg5_values);
				}
				nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
			}
		}

		uint8_t flag[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; ++j)
		{
			flag[j] = (carnival_data != nullptr) ? screen_candidate(working_value[j], lane, carnival_data) : 0u;
		}

		if (lane == 0u && output_flags != nullptr)
		{
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; ++j)
			{
				if (j < valid) output_flags[task_base + j] = flag[j];
			}
		}

		local_started += valid;
		local_completed += valid;
		local_maps += static_cast<unsigned long long>(valid) * 27ull;
	}

	if (lane == 0u && stats != nullptr)
	{
		atomicAdd(&stats->reps_started, local_started);
		atomicAdd(&stats->reps_completed, local_completed);
		atomicAdd(&stats->map_evals, local_maps);
	}
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_stream_window_offset_ilp6_cuda(
	uint32_t candidate_count,
	uint32_t data_start,
	uint32_t* work_counter,
	uint8_t* output_flags,
	uint8_t* drop_map_out,
	tm_raceway_stats* stats,
	unsigned long long** cap_tables,
	const uint32_t* cap_bits,
	const uint32_t* cap_ways,
	uint32_t cap_count,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t schedule_count,
	uint32_t first_cap_map)
{
	(void)drop_map_out;
	(void)cap_tables;
	(void)cap_bits;
	(void)cap_ways;
	(void)first_cap_map;
	if (cap_count != 0u || schedule_count != 27u) return;
	tm_raceway_stream_window_offset_ilp_body<6u>(candidate_count, data_start, work_counter,
		output_flags, stats, offset_regular_rng_values, offset_alg0_values, offset_alg6_values,
		offset_alg2_values, offset_alg5_values, expansion_values, schedule_data, carnival_data, key);
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_stream_window_offset_ilp8_cuda(
	uint32_t candidate_count,
	uint32_t data_start,
	uint32_t* work_counter,
	uint8_t* output_flags,
	uint8_t* drop_map_out,
	tm_raceway_stats* stats,
	unsigned long long** cap_tables,
	const uint32_t* cap_bits,
	const uint32_t* cap_ways,
	uint32_t cap_count,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t schedule_count,
	uint32_t first_cap_map)
{
	(void)drop_map_out;
	(void)cap_tables;
	(void)cap_bits;
	(void)cap_ways;
	(void)first_cap_map;
	if (cap_count != 0u || schedule_count != 27u) return;
	tm_raceway_stream_window_offset_ilp_body<8u>(candidate_count, data_start, work_counter,
		output_flags, stats, offset_regular_rng_values, offset_alg0_values, offset_alg6_values,
		offset_alg2_values, offset_alg5_values, expansion_values, schedule_data, carnival_data, key);
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_stream_window_offset_ilp6_static_cuda(
	uint32_t candidate_count,
	uint32_t data_start,
	uint32_t* work_counter,
	uint8_t* output_flags,
	uint8_t* drop_map_out,
	tm_raceway_stats* stats,
	unsigned long long** cap_tables,
	const uint32_t* cap_bits,
	const uint32_t* cap_ways,
	uint32_t cap_count,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key,
	uint32_t schedule_count,
	uint32_t first_cap_map)
{
	(void)work_counter;
	(void)drop_map_out;
	(void)stats;
	(void)cap_tables;
	(void)cap_bits;
	(void)cap_ways;
	(void)cap_count;
	(void)first_cap_map;
	if (schedule_count != 27u) return;

	constexpr uint32_t ILP = 6u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * ILP;
	if (candidate_base >= candidate_count) return;

	const uint32_t valid = min(ILP, candidate_count - candidate_base);
	uint32_t working_value[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; ++j)
	{
		working_value[j] = initialize_working_word(key, data_start + candidate_base + j, lane, expansion_values);
	}

	for (uint32_t schedule_index = 0u; schedule_index < 27u; ++schedule_index)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

		uint32_t rng_offset[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; ++j) rng_offset[j] = 0u;

		uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
			| ((packed_schedule & 0xFF000000u) >> 24));

		for (uint32_t i = 0u; i < 16u; ++i)
		{
			const uint32_t source_lane = i >> 2;
			const uint32_t source_shift = (i & 3u) * 8u;
			const uint32_t algorithm_shift = source_shift + 1u + ((static_cast<uint32_t>(nibble_selector) >> 13) & 4u);
			uint8_t algorithm_id[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; ++j)
			{
				const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, working_value[j], source_lane);
				algorithm_id[j] = static_cast<uint8_t>((source_word >> algorithm_shift) & 0x07u);
			}
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; ++j)
			{
				working_value[j] = run_alg_offset(working_value[j], lane, algorithm_id[j], schedule_index, &rng_offset[j],
					offset_regular_rng_values, offset_alg0_values, offset_alg6_values, offset_alg2_values, offset_alg5_values);
			}
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}
	}

	uint8_t flag[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; ++j)
	{
		flag[j] = (carnival_data != nullptr) ? screen_candidate(working_value[j], lane, carnival_data) : 0u;
	}

	if (lane == 0u && output_flags != nullptr)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; ++j)
		{
			if (j < valid) output_flags[candidate_base + j] = flag[j];
		}
	}
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_continue_liveidx_offset_ilp6_static_cuda(
	const uint32_t* live_idx,
	uint32_t live_count,
	uint32_t data_start,
	uint8_t* output_flags,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t key)
{
	constexpr uint32_t ILP = 6u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t live_base = (blockIdx.x * warps_per_block + warp_index) * ILP;
	if (live_base >= live_count) return;

	const uint32_t valid = min(ILP, live_count - live_base);
	uint32_t orig[ILP];
	uint32_t working_value[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; ++j)
	{
		orig[j] = (j < valid) ? live_idx[live_base + j] : 0u;
		working_value[j] = initialize_working_word(key, data_start + orig[j], lane, expansion_values);
	}

	for (uint32_t schedule_index = 0u; schedule_index < 27u; ++schedule_index)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

		uint32_t rng_offset[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; ++j) rng_offset[j] = 0u;

		uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
			| ((packed_schedule & 0xFF000000u) >> 24));

		for (uint32_t i = 0u; i < 16u; ++i)
		{
			const uint32_t source_lane = i >> 2;
			const uint32_t source_shift = (i & 3u) * 8u;
			const uint32_t algorithm_shift = source_shift + 1u + ((static_cast<uint32_t>(nibble_selector) >> 13) & 4u);
			uint8_t algorithm_id[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; ++j)
			{
				const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, working_value[j], source_lane);
				algorithm_id[j] = static_cast<uint8_t>((source_word >> algorithm_shift) & 0x07u);
			}
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; ++j)
			{
				working_value[j] = run_alg_offset(working_value[j], lane, algorithm_id[j], schedule_index, &rng_offset[j],
					offset_regular_rng_values, offset_alg0_values, offset_alg6_values, offset_alg2_values, offset_alg5_values);
			}
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}
	}

	uint8_t flag[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; ++j)
	{
		flag[j] = (carnival_data != nullptr) ? screen_candidate(working_value[j], lane, carnival_data) : 0u;
	}

	if (lane == 0u && output_flags != nullptr)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; ++j)
		{
			if (j < valid) output_flags[orig[j]] = flag[j];
		}
	}
}

template<uint32_t ILP>
__device__ __forceinline__ void tm_raceway_continue_state_liveidx_offset_static_body(
	const uint32_t* live_idx,
	uint32_t live_count,
	const uint32_t* state,
	uint8_t* output_flags,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t schedule_start)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t live_base = (blockIdx.x * warps_per_block + warp_index) * ILP;
	if (live_base >= live_count) return;

	const uint32_t valid = min(ILP, live_count - live_base);
	uint32_t orig[ILP];
	uint32_t working_value[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; ++j)
	{
		orig[j] = (j < valid) ? live_idx[live_base + j] : 0u;
		working_value[j] = state[static_cast<size_t>(orig[j]) * 32u + lane];
	}

	for (uint32_t schedule_index = schedule_start; schedule_index < 27u; ++schedule_index)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

		uint32_t rng_offset[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; ++j) rng_offset[j] = 0u;

		uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
			| ((packed_schedule & 0xFF000000u) >> 24));

		for (uint32_t i = 0u; i < 16u; ++i)
		{
			const uint32_t source_lane = i >> 2;
			const uint32_t source_shift = (i & 3u) * 8u;
			const uint32_t algorithm_shift = source_shift + 1u + ((static_cast<uint32_t>(nibble_selector) >> 13) & 4u);
			uint8_t algorithm_id[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; ++j)
			{
				const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, working_value[j], source_lane);
				algorithm_id[j] = static_cast<uint8_t>((source_word >> algorithm_shift) & 0x07u);
			}
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; ++j)
			{
				working_value[j] = run_alg_offset(working_value[j], lane, algorithm_id[j], schedule_index, &rng_offset[j],
					offset_regular_rng_values, offset_alg0_values, offset_alg6_values, offset_alg2_values, offset_alg5_values);
			}
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}
	}

	uint8_t flag[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; ++j)
	{
		flag[j] = (carnival_data != nullptr) ? screen_candidate(working_value[j], lane, carnival_data) : 0u;
	}

	if (lane == 0u && output_flags != nullptr)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; ++j)
		{
			if (j < valid) output_flags[orig[j]] = flag[j];
		}
	}
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_continue_state_liveidx_offset_ilp4_static_cuda(
	const uint32_t* live_idx,
	uint32_t live_count,
	const uint32_t* state,
	uint8_t* output_flags,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t schedule_start)
{
	tm_raceway_continue_state_liveidx_offset_static_body<4u>(live_idx, live_count, state, output_flags,
		offset_regular_rng_values, offset_alg0_values, offset_alg6_values, offset_alg2_values,
		offset_alg5_values, schedule_data, carnival_data, schedule_start);
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_continue_state_liveidx_offset_ilp6_static_cuda(
	const uint32_t* live_idx,
	uint32_t live_count,
	const uint32_t* state,
	uint8_t* output_flags,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t schedule_start)
{
	tm_raceway_continue_state_liveidx_offset_static_body<6u>(live_idx, live_count, state, output_flags,
		offset_regular_rng_values, offset_alg0_values, offset_alg6_values, offset_alg2_values,
		offset_alg5_values, schedule_data, carnival_data, schedule_start);
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_continue_state_liveidx_offset_ilp8_static_cuda(
	const uint32_t* live_idx,
	uint32_t live_count,
	const uint32_t* state,
	uint8_t* output_flags,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	uint32_t schedule_start)
{
	tm_raceway_continue_state_liveidx_offset_static_body<8u>(live_idx, live_count, state, output_flags,
		offset_regular_rng_values, offset_alg0_values, offset_alg6_values, offset_alg2_values,
		offset_alg5_values, schedule_data, carnival_data, schedule_start);
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_span_state_liveidx_cap_offset_cuda(
	const uint32_t* live_idx,
	uint32_t live_count,
	uint32_t* work_counter,
	uint32_t* state,
	uint8_t* alive_out,
	uint8_t* drop_map_out,
	tm_raceway_stats* stats,
	unsigned long long* cap_table,
	uint32_t cap_bits,
	uint32_t cap_ways,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* schedule_data,
	uint32_t start_map,
	uint32_t end_map)
{
	const uint32_t lane = threadIdx.x & 31u;
	unsigned long long local_started = 0ull;
	unsigned long long local_completed = 0ull;
	unsigned long long local_dropped = 0ull;
	unsigned long long local_maps = 0ull;

	while (true)
	{
		uint32_t slot = 0u;
		if (lane == 0u) slot = atomicAdd(work_counter, 1u);
		slot = __shfl_sync(0xFFFFFFFFu, slot, 0);
		if (slot >= live_count) break;

		local_started++;
		const uint32_t orig = live_idx[slot];
		uint32_t value = state[static_cast<size_t>(orig) * 32u + lane];

		for (uint32_t map = start_map; map <= end_map && map < 27u; ++map)
		{
			value = tm_raceway_run_one_map_offset(value, lane, map,
				offset_regular_rng_values, offset_alg0_values, offset_alg6_values,
				offset_alg2_values, offset_alg5_values, schedule_data);
			local_maps++;
		}

		const uint64_t fp = tm_strong64_state(value, lane);
		bool drop = false;
		if (lane == 0u)
		{
			drop = tm_raceway_cap_probe_or_keep(static_cast<unsigned long long>(fp),
				cap_table, cap_bits, cap_ways);
		}
		drop = __shfl_sync(0xFFFFFFFFu, drop, 0);

		if (lane == 0u)
		{
			if (alive_out != nullptr) alive_out[slot] = drop ? 0u : 1u;
			if (drop && drop_map_out != nullptr) drop_map_out[orig] = static_cast<uint8_t>(end_map);
		}
		if (drop)
		{
			local_dropped++;
		}
		else
		{
			state[static_cast<size_t>(orig) * 32u + lane] = value;
			local_completed++;
		}
	}

	if (lane == 0u && stats != nullptr)
	{
		atomicAdd(&stats->reps_started, local_started);
		atomicAdd(&stats->reps_completed, local_completed);
		atomicAdd(&stats->reps_dropped, local_dropped);
		atomicAdd(&stats->map_evals, local_maps);
	}
}

// Advance ILP candidates through one map with the candidates interleaved at the
// innermost byte level (NOT one full map per candidate sequentially). This is the
// same interleaving that gives the no-cap forward ILP kernels their latency hiding;
// it is bit-identical to calling tm_raceway_run_one_map_offset per candidate. The
// algorithm_shift form (source_shift + 1 + ((nibble>>13)&4)) matches the scalar
// helper's "current_byte >>= 4 when nibble high bit set, then (byte>>1)&7".
template<uint32_t ILP>
__device__ __forceinline__ void tm_raceway_run_one_map_offset_ilp(
	uint32_t (&working_value)[ILP],
	uint32_t lane,
	uint32_t schedule_index,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* schedule_data)
{
	uint32_t packed_schedule = 0u;
	if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
	packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

	uint32_t rng_offset[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; ++j) rng_offset[j] = 0u;

	uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
		| ((packed_schedule & 0xFF000000u) >> 24));

	for (uint32_t i = 0u; i < 16u; ++i)
	{
		const uint32_t source_lane = i >> 2;
		const uint32_t source_shift = (i & 3u) * 8u;
		const uint32_t algorithm_shift = source_shift + 1u + ((static_cast<uint32_t>(nibble_selector) >> 13) & 4u);
		uint8_t algorithm_id[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; ++j)
		{
			const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, working_value[j], source_lane);
			algorithm_id[j] = static_cast<uint8_t>((source_word >> algorithm_shift) & 0x07u);
		}
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; ++j)
		{
			working_value[j] = run_alg_offset(working_value[j], lane, algorithm_id[j], schedule_index, &rng_offset[j],
				offset_regular_rng_values, offset_alg0_values, offset_alg6_values, offset_alg2_values, offset_alg5_values);
		}
		nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
	}
}

// ILP cap-span variants of the two hot bounded-wave kernels. The ILP1 versions
// above carry one candidate per warp; these carry ILP candidates per warp and
// interleave them at the inner byte level (via the helper above), the same latency-
// hiding lever that took the no-cap forward kernel from 71 to ~96-104 M/s. State
// model is unchanged: candidate j is owned by the whole warp (its 1024-bit state is
// spread across the 32 lanes' working_value[j]), so alive[j]/valid are warp-uniform
// and the cap probe stays lane-0-only. FN-safety is preserved: the cap is still
// benign-race over-keep-only, probed once per surviving candidate per boundary.
template<uint32_t ILP>
__device__ __forceinline__ void tm_raceway_boundary_cap_state_offset_ilp_body(
	uint32_t candidate_count,
	uint32_t data_start,
	uint32_t* work_counter,
	uint8_t* alive_out,
	uint8_t* drop_map_out,
	tm_raceway_stats* stats,
	uint32_t* state,
	unsigned long long** cap_tables,
	const uint32_t* cap_bits,
	const uint32_t* cap_ways,
	uint32_t cap_count,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	uint32_t key,
	uint32_t schedule_count,
	uint32_t first_cap_map)
{
	const uint32_t lane = threadIdx.x & 31u;
	unsigned long long local_started = 0ull;
	unsigned long long local_completed = 0ull;
	unsigned long long local_dropped = 0ull;
	unsigned long long local_maps = 0ull;
	const uint32_t last_map = (cap_count == 0u) ? 0u : min(schedule_count, first_cap_map + cap_count);

	while (true)
	{
		uint32_t task_base = 0u;
		if (lane == 0u) task_base = atomicAdd(work_counter, ILP);
		task_base = __shfl_sync(0xFFFFFFFFu, task_base, 0);
		if (task_base >= candidate_count) break;

		const uint32_t valid = min(ILP, candidate_count - task_base);
		uint32_t working_value[ILP];
		bool alive[ILP];
		uint32_t drop_map[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; ++j)
		{
			working_value[j] = initialize_working_word(key, data_start + task_base + j, lane, expansion_values);
			alive[j] = (j < valid);
			drop_map[j] = 0xFFFFFFFFu;
		}
		local_started += valid;

		for (uint32_t map = 0u; map < last_map; ++map)
		{
			tm_raceway_run_one_map_offset_ilp<ILP>(working_value, lane, map,
				offset_regular_rng_values, offset_alg0_values, offset_alg6_values,
				offset_alg2_values, offset_alg5_values, schedule_data);
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; ++j)
				if (alive[j]) local_maps++;

			const bool has_cap = cap_tables != nullptr && cap_bits != nullptr && cap_ways != nullptr
				&& map >= first_cap_map && (map - first_cap_map) < cap_count;
			if (has_cap)
			{
				const uint32_t cap_idx = map - first_cap_map;
				#pragma unroll
				for (uint32_t j = 0u; j < ILP; ++j)
				{
					if (!alive[j]) continue; // warp-uniform: candidate j is owned by the whole warp
					const uint64_t fp = tm_strong64_state(working_value[j], lane);
					bool drop = false;
					if (lane == 0u)
					{
						drop = tm_raceway_cap_probe_or_keep(static_cast<unsigned long long>(fp),
							cap_tables[cap_idx], cap_bits[cap_idx], cap_ways[cap_idx]);
					}
					drop = __shfl_sync(0xFFFFFFFFu, drop, 0);
					if (drop)
					{
						alive[j] = false;
						drop_map[j] = map;
						local_dropped++;
					}
				}
			}
		}

		#pragma unroll
		for (uint32_t j = 0u; j < ILP; ++j)
		{
			if (j >= valid) continue;
			if (alive[j])
			{
				if (state != nullptr) state[(static_cast<size_t>(task_base) + j) * 32u + lane] = working_value[j];
				local_completed++;
			}
			if (lane == 0u)
			{
				if (alive_out != nullptr) alive_out[task_base + j] = alive[j] ? 1u : 0u;
				if (!alive[j] && drop_map_out != nullptr) drop_map_out[task_base + j] = static_cast<uint8_t>(drop_map[j]);
			}
		}
	}

	if (lane == 0u && stats != nullptr)
	{
		atomicAdd(&stats->reps_started, local_started);
		atomicAdd(&stats->reps_completed, local_completed);
		atomicAdd(&stats->reps_dropped, local_dropped);
		atomicAdd(&stats->map_evals, local_maps);
	}
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_boundary_cap_state_offset_ilp4_cuda(
	uint32_t candidate_count, uint32_t data_start, uint32_t* work_counter, uint8_t* alive_out,
	uint8_t* drop_map_out, tm_raceway_stats* stats, uint32_t* state, unsigned long long** cap_tables,
	const uint32_t* cap_bits, const uint32_t* cap_ways, uint32_t cap_count,
	const uint8_t* offset_regular_rng_values, const uint8_t* offset_alg0_values, const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values, const uint32_t* offset_alg5_values, const uint8_t* expansion_values,
	const uint8_t* schedule_data, uint32_t key, uint32_t schedule_count, uint32_t first_cap_map)
{
	tm_raceway_boundary_cap_state_offset_ilp_body<4u>(candidate_count, data_start, work_counter, alive_out,
		drop_map_out, stats, state, cap_tables, cap_bits, cap_ways, cap_count,
		offset_regular_rng_values, offset_alg0_values, offset_alg6_values, offset_alg2_values, offset_alg5_values,
		expansion_values, schedule_data, key, schedule_count, first_cap_map);
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_boundary_cap_state_offset_ilp5_cuda(
	uint32_t candidate_count, uint32_t data_start, uint32_t* work_counter, uint8_t* alive_out,
	uint8_t* drop_map_out, tm_raceway_stats* stats, uint32_t* state, unsigned long long** cap_tables,
	const uint32_t* cap_bits, const uint32_t* cap_ways, uint32_t cap_count,
	const uint8_t* offset_regular_rng_values, const uint8_t* offset_alg0_values, const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values, const uint32_t* offset_alg5_values, const uint8_t* expansion_values,
	const uint8_t* schedule_data, uint32_t key, uint32_t schedule_count, uint32_t first_cap_map)
{
	tm_raceway_boundary_cap_state_offset_ilp_body<5u>(candidate_count, data_start, work_counter, alive_out,
		drop_map_out, stats, state, cap_tables, cap_bits, cap_ways, cap_count,
		offset_regular_rng_values, offset_alg0_values, offset_alg6_values, offset_alg2_values, offset_alg5_values,
		expansion_values, schedule_data, key, schedule_count, first_cap_map);
}

// ILP3/ILP6 close-out variants (2026-06-13): added after the span-state register bottleneck was resolved,
// to confirm the cap-span ILP optimum did not move. ILP3 lb(256,6) for light occupancy; ILP6 lb(256,3) so
// the heavier carry is not forced to spill.
extern "C" __global__ __launch_bounds__(256, 6) void tm_raceway_boundary_cap_state_offset_ilp3_cuda(
	uint32_t candidate_count, uint32_t data_start, uint32_t* work_counter, uint8_t* alive_out,
	uint8_t* drop_map_out, tm_raceway_stats* stats, uint32_t* state, unsigned long long** cap_tables,
	const uint32_t* cap_bits, const uint32_t* cap_ways, uint32_t cap_count,
	const uint8_t* offset_regular_rng_values, const uint8_t* offset_alg0_values, const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values, const uint32_t* offset_alg5_values, const uint8_t* expansion_values,
	const uint8_t* schedule_data, uint32_t key, uint32_t schedule_count, uint32_t first_cap_map)
{
	tm_raceway_boundary_cap_state_offset_ilp_body<3u>(candidate_count, data_start, work_counter, alive_out,
		drop_map_out, stats, state, cap_tables, cap_bits, cap_ways, cap_count,
		offset_regular_rng_values, offset_alg0_values, offset_alg6_values, offset_alg2_values, offset_alg5_values,
		expansion_values, schedule_data, key, schedule_count, first_cap_map);
}

extern "C" __global__ __launch_bounds__(256, 3) void tm_raceway_boundary_cap_state_offset_ilp6_cuda(
	uint32_t candidate_count, uint32_t data_start, uint32_t* work_counter, uint8_t* alive_out,
	uint8_t* drop_map_out, tm_raceway_stats* stats, uint32_t* state, unsigned long long** cap_tables,
	const uint32_t* cap_bits, const uint32_t* cap_ways, uint32_t cap_count,
	const uint8_t* offset_regular_rng_values, const uint8_t* offset_alg0_values, const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values, const uint32_t* offset_alg5_values, const uint8_t* expansion_values,
	const uint8_t* schedule_data, uint32_t key, uint32_t schedule_count, uint32_t first_cap_map)
{
	tm_raceway_boundary_cap_state_offset_ilp_body<6u>(candidate_count, data_start, work_counter, alive_out,
		drop_map_out, stats, state, cap_tables, cap_bits, cap_ways, cap_count,
		offset_regular_rng_values, offset_alg0_values, offset_alg6_values, offset_alg2_values, offset_alg5_values,
		expansion_values, schedule_data, key, schedule_count, first_cap_map);
}

template<uint32_t ILP>
__device__ __forceinline__ void tm_raceway_span_state_liveidx_cap_offset_ilp_body(
	const uint32_t* live_idx,
	uint32_t live_count,
	uint32_t* work_counter,
	uint32_t* state,
	uint8_t* alive_out,
	uint8_t* drop_map_out,
	tm_raceway_stats* stats,
	unsigned long long* cap_table,
	uint32_t cap_bits,
	uint32_t cap_ways,
	const uint8_t* offset_regular_rng_values,
	const uint8_t* offset_alg0_values,
	const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values,
	const uint32_t* offset_alg5_values,
	const uint8_t* schedule_data,
	uint32_t start_map,
	uint32_t end_map)
{
	const uint32_t lane = threadIdx.x & 31u;
	unsigned long long local_started = 0ull;
	unsigned long long local_completed = 0ull;
	unsigned long long local_dropped = 0ull;
	unsigned long long local_maps = 0ull;

	while (true)
	{
		uint32_t slot_base = 0u;
		if (lane == 0u) slot_base = atomicAdd(work_counter, ILP);
		slot_base = __shfl_sync(0xFFFFFFFFu, slot_base, 0);
		if (slot_base >= live_count) break;

		const uint32_t valid = min(ILP, live_count - slot_base);
		uint32_t orig[ILP];
		uint32_t working_value[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; ++j)
		{
			orig[j] = (j < valid) ? live_idx[slot_base + j] : 0u;
			working_value[j] = state[static_cast<size_t>(orig[j]) * 32u + lane];
		}
		local_started += valid;

		for (uint32_t map = start_map; map <= end_map && map < 27u; ++map)
		{
			tm_raceway_run_one_map_offset_ilp<ILP>(working_value, lane, map,
				offset_regular_rng_values, offset_alg0_values, offset_alg6_values,
				offset_alg2_values, offset_alg5_values, schedule_data);
			local_maps += valid;
		}

		#pragma unroll
		for (uint32_t j = 0u; j < ILP; ++j)
		{
			if (j >= valid) continue; // warp-uniform
			const uint64_t fp = tm_strong64_state(working_value[j], lane);
			bool drop = false;
			if (lane == 0u)
			{
				drop = tm_raceway_cap_probe_or_keep(static_cast<unsigned long long>(fp),
					cap_table, cap_bits, cap_ways);
			}
			drop = __shfl_sync(0xFFFFFFFFu, drop, 0);
			if (lane == 0u)
			{
				if (alive_out != nullptr) alive_out[slot_base + j] = drop ? 0u : 1u;
				if (drop && drop_map_out != nullptr) drop_map_out[orig[j]] = static_cast<uint8_t>(end_map);
			}
			if (drop)
			{
				local_dropped++;
			}
			else
			{
				state[static_cast<size_t>(orig[j]) * 32u + lane] = working_value[j];
				local_completed++;
			}
		}
	}

	if (lane == 0u && stats != nullptr)
	{
		atomicAdd(&stats->reps_started, local_started);
		atomicAdd(&stats->reps_completed, local_completed);
		atomicAdd(&stats->reps_dropped, local_dropped);
		atomicAdd(&stats->map_evals, local_maps);
	}
}

extern "C" __global__ __launch_bounds__(256, 5) void tm_raceway_span_state_liveidx_cap_offset_ilp4_cuda(
	const uint32_t* live_idx, uint32_t live_count, uint32_t* work_counter, uint32_t* state, uint8_t* alive_out,
	uint8_t* drop_map_out, tm_raceway_stats* stats, unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways,
	const uint8_t* offset_regular_rng_values, const uint8_t* offset_alg0_values, const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values, const uint32_t* offset_alg5_values, const uint8_t* schedule_data,
	uint32_t start_map, uint32_t end_map)
{
	tm_raceway_span_state_liveidx_cap_offset_ilp_body<4u>(live_idx, live_count, work_counter, state, alive_out,
		drop_map_out, stats, cap_table, cap_bits, cap_ways, offset_regular_rng_values, offset_alg0_values,
		offset_alg6_values, offset_alg2_values, offset_alg5_values, schedule_data, start_map, end_map);
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_span_state_liveidx_cap_offset_ilp5_cuda(
	const uint32_t* live_idx, uint32_t live_count, uint32_t* work_counter, uint32_t* state, uint8_t* alive_out,
	uint8_t* drop_map_out, tm_raceway_stats* stats, unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways,
	const uint8_t* offset_regular_rng_values, const uint8_t* offset_alg0_values, const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values, const uint32_t* offset_alg5_values, const uint8_t* schedule_data,
	uint32_t start_map, uint32_t end_map)
{
	tm_raceway_span_state_liveidx_cap_offset_ilp_body<5u>(live_idx, live_count, work_counter, state, alive_out,
		drop_map_out, stats, cap_table, cap_bits, cap_ways, offset_regular_rng_values, offset_alg0_values,
		offset_alg6_values, offset_alg2_values, offset_alg5_values, schedule_data, start_map, end_map);
}

// ILP3/ILP6 close-out variants (2026-06-13). ILP3 lb(256,6); ILP6 lb(256,3) to avoid forced spill.
extern "C" __global__ __launch_bounds__(256, 6) void tm_raceway_span_state_liveidx_cap_offset_ilp3_cuda(
	const uint32_t* live_idx, uint32_t live_count, uint32_t* work_counter, uint32_t* state, uint8_t* alive_out,
	uint8_t* drop_map_out, tm_raceway_stats* stats, unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways,
	const uint8_t* offset_regular_rng_values, const uint8_t* offset_alg0_values, const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values, const uint32_t* offset_alg5_values, const uint8_t* schedule_data,
	uint32_t start_map, uint32_t end_map)
{
	tm_raceway_span_state_liveidx_cap_offset_ilp_body<3u>(live_idx, live_count, work_counter, state, alive_out,
		drop_map_out, stats, cap_table, cap_bits, cap_ways, offset_regular_rng_values, offset_alg0_values,
		offset_alg6_values, offset_alg2_values, offset_alg5_values, schedule_data, start_map, end_map);
}

extern "C" __global__ __launch_bounds__(256, 3) void tm_raceway_span_state_liveidx_cap_offset_ilp6_cuda(
	const uint32_t* live_idx, uint32_t live_count, uint32_t* work_counter, uint32_t* state, uint8_t* alive_out,
	uint8_t* drop_map_out, tm_raceway_stats* stats, unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways,
	const uint8_t* offset_regular_rng_values, const uint8_t* offset_alg0_values, const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values, const uint32_t* offset_alg5_values, const uint8_t* schedule_data,
	uint32_t start_map, uint32_t end_map)
{
	tm_raceway_span_state_liveidx_cap_offset_ilp_body<6u>(live_idx, live_count, work_counter, state, alive_out,
		drop_map_out, stats, cap_table, cap_bits, cap_ways, offset_regular_rng_values, offset_alg0_values,
		offset_alg6_values, offset_alg2_values, offset_alg5_values, schedule_data, start_map, end_map);
}

// ----- #5 technical-completeness ILP ports (2026-06-13) -----
// ILP variants of the two remaining ILP1 cap kernels: the no-state wave MARK kernel and the dynamic
// full-window direct-offset cap kernel. Both reuse the inner-interleaved tm_raceway_run_one_map_offset_ilp
// helper. State model + benign-race cap semantics unchanged; alive[j]/valid are warp-uniform.

// No-state mark: run to the configured cap boundaries, write alive_out only (no state writeback).
template<uint32_t ILP>
__device__ __forceinline__ void tm_raceway_boundary_cap_mark_offset_ilp_body(
	uint32_t candidate_count, uint32_t data_start, uint32_t* work_counter,
	uint8_t* alive_out, uint8_t* drop_map_out, tm_raceway_stats* stats,
	unsigned long long** cap_tables, const uint32_t* cap_bits, const uint32_t* cap_ways, uint32_t cap_count,
	const uint8_t* offset_regular_rng_values, const uint8_t* offset_alg0_values, const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values, const uint32_t* offset_alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t schedule_count, uint32_t first_cap_map)
{
	const uint32_t lane = threadIdx.x & 31u;
	unsigned long long local_started = 0ull, local_completed = 0ull, local_dropped = 0ull, local_maps = 0ull;
	const uint32_t last_map = (cap_count == 0u) ? 0u : min(schedule_count, first_cap_map + cap_count);
	while (true)
	{
		uint32_t task_base = 0u;
		if (lane == 0u) task_base = atomicAdd(work_counter, ILP);
		task_base = __shfl_sync(0xFFFFFFFFu, task_base, 0);
		if (task_base >= candidate_count) break;
		const uint32_t valid = min(ILP, candidate_count - task_base);
		uint32_t working_value[ILP]; bool alive[ILP]; uint32_t drop_map[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; ++j)
		{
			working_value[j] = initialize_working_word(key, data_start + task_base + j, lane, expansion_values);
			alive[j] = (j < valid); drop_map[j] = 0xFFFFFFFFu;
		}
		local_started += valid;
		for (uint32_t map = 0u; map < last_map; ++map)
		{
			tm_raceway_run_one_map_offset_ilp<ILP>(working_value, lane, map,
				offset_regular_rng_values, offset_alg0_values, offset_alg6_values,
				offset_alg2_values, offset_alg5_values, schedule_data);
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; ++j) if (alive[j]) local_maps++;
			const bool has_cap = cap_tables != nullptr && cap_bits != nullptr && cap_ways != nullptr
				&& map >= first_cap_map && (map - first_cap_map) < cap_count;
			if (has_cap)
			{
				const uint32_t cap_idx = map - first_cap_map;
				#pragma unroll
				for (uint32_t j = 0u; j < ILP; ++j)
				{
					if (!alive[j]) continue;
					const uint64_t fp = tm_strong64_state(working_value[j], lane);
					bool drop = false;
					if (lane == 0u)
						drop = tm_raceway_cap_probe_or_keep(static_cast<unsigned long long>(fp),
							cap_tables[cap_idx], cap_bits[cap_idx], cap_ways[cap_idx]);
					drop = __shfl_sync(0xFFFFFFFFu, drop, 0);
					if (drop) { alive[j] = false; drop_map[j] = map; local_dropped++; }
				}
			}
		}
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; ++j)
		{
			if (j >= valid) continue;
			if (lane == 0u)
			{
				if (alive_out != nullptr) alive_out[task_base + j] = alive[j] ? 1u : 0u;
				if (!alive[j] && drop_map_out != nullptr) drop_map_out[task_base + j] = static_cast<uint8_t>(drop_map[j]);
			}
			if (alive[j]) local_completed++;
		}
	}
	if (lane == 0u && stats != nullptr)
	{
		atomicAdd(&stats->reps_started, local_started);
		atomicAdd(&stats->reps_completed, local_completed);
		atomicAdd(&stats->reps_dropped, local_dropped);
		atomicAdd(&stats->map_evals, local_maps);
	}
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_boundary_cap_mark_offset_ilp4_cuda(
	uint32_t candidate_count, uint32_t data_start, uint32_t* work_counter, uint8_t* alive_out,
	uint8_t* drop_map_out, tm_raceway_stats* stats, unsigned long long** cap_tables,
	const uint32_t* cap_bits, const uint32_t* cap_ways, uint32_t cap_count,
	const uint8_t* offset_regular_rng_values, const uint8_t* offset_alg0_values, const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values, const uint32_t* offset_alg5_values, const uint8_t* expansion_values,
	const uint8_t* schedule_data, const uint8_t* carnival_data, uint32_t key, uint32_t schedule_count, uint32_t first_cap_map)
{
	(void)carnival_data;
	tm_raceway_boundary_cap_mark_offset_ilp_body<4u>(candidate_count, data_start, work_counter, alive_out,
		drop_map_out, stats, cap_tables, cap_bits, cap_ways, cap_count, offset_regular_rng_values,
		offset_alg0_values, offset_alg6_values, offset_alg2_values, offset_alg5_values, expansion_values,
		schedule_data, key, schedule_count, first_cap_map);
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_raceway_boundary_cap_mark_offset_ilp5_cuda(
	uint32_t candidate_count, uint32_t data_start, uint32_t* work_counter, uint8_t* alive_out,
	uint8_t* drop_map_out, tm_raceway_stats* stats, unsigned long long** cap_tables,
	const uint32_t* cap_bits, const uint32_t* cap_ways, uint32_t cap_count,
	const uint8_t* offset_regular_rng_values, const uint8_t* offset_alg0_values, const uint8_t* offset_alg6_values,
	const uint32_t* offset_alg2_values, const uint32_t* offset_alg5_values, const uint8_t* expansion_values,
	const uint8_t* schedule_data, const uint8_t* carnival_data, uint32_t key, uint32_t schedule_count, uint32_t first_cap_map)
{
	(void)carnival_data;
	tm_raceway_boundary_cap_mark_offset_ilp_body<5u>(candidate_count, data_start, work_counter, alive_out,
		drop_map_out, stats, cap_tables, cap_bits, cap_ways, cap_count, offset_regular_rng_values,
		offset_alg0_values, offset_alg6_values, offset_alg2_values, offset_alg5_values, expansion_values,
		schedule_data, key, schedule_count, first_cap_map);
}


extern "C" __global__ __launch_bounds__(128) void tm_raceway_repflag_table_build_cuda(
	uint64_t* table_fp,
	uint8_t* table_flag,
	uint32_t logm,
	const uint32_t* rep_values,
	const uint8_t* rep_flags,
	uint32_t rep_count,
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
	uint32_t rep_values_are_absolute,
	unsigned long long* counters)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t rep = blockIdx.x * warps_per_block + warp_index;
	if (rep >= rep_count) return;

	const uint32_t rv = rep_values[rep];
	const uint32_t data = (rep_values_are_absolute != 0u) ? rv : (data_start + rv);
	uint32_t value = initialize_working_word(key, data, lane, expansion_values);
	value = tm_raceway_run_one_map(value, lane, 0u,
		regular_rng_values, alg0_values, alg6_values,
		rng_forward_1, rng_forward_128, alg2_values, alg5_values,
		schedule_data);
	const uint64_t fp64 = tm_strong64_state(value, lane);
	if (lane != 0u) return;

	const uint64_t fp = fp64 == 0ull ? 1ull : fp64;
	const uint32_t mask = (1u << logm) - 1u;
	uint32_t slot = static_cast<uint32_t>((fp * 0x9E3779B97F4A7C15ull) >> (64u - logm));
	constexpr uint32_t PROBE_CAP = 256u;
	for (uint32_t p = 0u; p < PROBE_CAP; ++p)
	{
		const uint32_t s = (slot + p) & mask;
		const unsigned long long old = atomicCAS(reinterpret_cast<unsigned long long*>(&table_fp[s]),
			0ull, static_cast<unsigned long long>(fp));
		if (old == 0ull || old == fp)
		{
			table_flag[s] = rep_flags[rep];
			if (counters != nullptr) atomicAdd(&counters[0], 1ull);
			return;
		}
	}
	if (counters != nullptr) atomicAdd(&counters[1], 1ull);
}

extern "C" __global__ __launch_bounds__(128) void tm_raceway_flat_parity_cuda(
	const uint8_t* flat_flags,
	uint32_t candidate_count,
	const uint64_t* table_fp,
	const uint8_t* table_flag,
	uint32_t logm,
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
	unsigned long long* counters)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t cand = blockIdx.x * warps_per_block + warp_index;
	if (cand >= candidate_count) return;

	const uint32_t data = data_start + cand;
	uint32_t value = initialize_working_word(key, data, lane, expansion_values);
	value = tm_raceway_run_one_map(value, lane, 0u,
		regular_rng_values, alg0_values, alg6_values,
		rng_forward_1, rng_forward_128, alg2_values, alg5_values,
		schedule_data);
	const uint64_t fp64 = tm_strong64_state(value, lane);
	if (lane != 0u) return;

	const uint64_t fp = fp64 == 0ull ? 1ull : fp64;
	const uint32_t mask = (1u << logm) - 1u;
	uint32_t slot = static_cast<uint32_t>((fp * 0x9E3779B97F4A7C15ull) >> (64u - logm));
	constexpr uint32_t PROBE_CAP = 256u;
	for (uint32_t p = 0u; p < PROBE_CAP; ++p)
	{
		const uint32_t s = (slot + p) & mask;
		const uint64_t cur = table_fp[s];
		if (cur == fp)
		{
			atomicAdd(&counters[0], 1ull);
			if (table_flag[s] != flat_flags[cand]) atomicAdd(&counters[2], 1ull);
			return;
		}
		if (cur == 0ull) break;
	}
	atomicAdd(&counters[0], 1ull);
	atomicAdd(&counters[1], 1ull);
}
