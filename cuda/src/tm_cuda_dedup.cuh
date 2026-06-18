// Treasure Master — CUDA state-dedup and on-GPU compaction kernel implementations.
//
// RESEARCH / A-B path (not the production default — that is the raceway, see
// tm_cuda_raceway.cuh). Retained for high-R-key comparison and for the bit-exact BFS
// compaction sweep + hit-finder. Contains two experimental GPU-side deduplication
// architectures, both operating on the 1024-bit (128-byte) candidate state produced by
// the forward schedule:
//
// ── Phase 1-2: Within-block state dedup (observation / skip variants) ──────────
//   tm_dump_state_dedup_w32_impl     — W=32 POC: observe dedup, always compute
//   tm_dump_state_dedup_w128_impl    — W=128 variant (wider dedup window)
//   tm_dump_state_dedup_w128_skip_impl — W=128 + skip dead warps
//   tm_dump_state_dedup_skip_tmpl    — shared skip template for w32/w64 variants
//   tm_checksum_screen_dedup_w32_impl — dedup + screen, output 1 byte per candidate
//   tm_checksum_screen_dedup_w32_offset_impl — same on offset-stream tables
//   tm_checksum_screen_dedup_offset_ilp_sched_impl — ILP-geometry offset dedup
//   TM_OFFSET_ILP_KERNEL macro       — instantiates geometry variants at build time
//
// ── On-GPU VRAM survivor-compaction architecture (2026-05-29) ─────────────────
//   run_span_dedup_impl              — map a SPAN, dedup within block, flag alive
//   compact_survivors_ordered        — block-stable prefix-scan compaction
//   resolve_flags                    — union-find chain resolution
//   tm_checksum_screen_dedup_periodic_impl — periodic-K dedup variant
//   tm_checksum_screen_dedup_w32_packed_impl  — 64-bit packed-slot, no spin-wait
//   tm_checksum_screen_dedup_w32_fasthash_impl — cheaper warp hash (no salt)
//   tm_checksum_screen_dedup_w32_maprng_impl  — maprng-backed dedup
//   tm_checksum_screen_dedup_w32_maprng_preext_impl — pre-extracted 3-stream maprng
//   tm_checksum_screen_dedup_w32_maprng_preext_sched_impl — dedup-schedule experiments
//   tm_checksum_screen_dedup_w32_maprng_preext_1sync_impl — 1 sync/boundary variant
//
// All _impl functions are __device__ __forceinline__ templates. Each is followed
// by its extern "C" __global__ entry points in the same section.
//
// Include this header only from tm_cuda.cu, after tm_cuda_screen.cuh.

#pragma once
#include <cstdint>
// ────────────────────────────────────────────────────────────────────────────
// State-dedup POC kernel (Phase 1: observation only — no compaction yet).
//
// Goal: validate that warp-cooperative per-boundary state dedup works on GPU,
// measure dedup overhead, and produce per-candidate metadata identifying which
// warp owns each unique state at the end of the schedule.
//
// Geometry:
//   - 1024 threads/block = 32 warps × 32 lanes
//   - 1 candidate per warp → W=32 candidates per block
//   - Per-boundary cooperative dedup using 64-slot shared-mem hash
//     (load factor 0.5, linear probe, 64-bit fingerprint from warp_hash_state)
//
// Phase 1 behavior: ALL warps continue computing through the full schedule
// regardless of dedup status. rep_warp[w] records, at the FINAL boundary,
// which warp index "owned" each candidate's final state. Validates the
// primitive end-to-end before adding Phase 2 (compaction/skip) on top.
//
// Output:
//   - output_state[c * 32 + lane] = lane's uint32 of candidate c's final state
//   - output_rep_index[c] = warp index of c's final-state rep (0..31)
template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_dump_state_dedup_w32_impl(
	uint8_t* output_state,
	uint32_t* output_rep_index,
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
	const uint32_t warp_index = threadIdx.x >> 5;   // 0..31 (1024/32)
	const uint32_t my_candidate = blockIdx.x * 32u + warp_index;
	const bool active = (my_candidate < candidate_count);

	// 64-slot hash table in shared mem. Each slot: 64-bit fingerprint (0 = empty)
	// + 32-bit rep warp index. Load factor 0.5 at 32 candidates.
	__shared__ unsigned long long slot_fp[64];
	__shared__ volatile uint32_t  slot_rep[64];  // volatile for spin-wait correctness
	__shared__ uint32_t           rep_warp[32];  // populated only at final boundary

	// Initialize each candidate's working_value. Inactive warps still
	// participate in __syncthreads but their state never collides with
	// real states (we'd just be doing wasted hash work — measurable but
	// uniform across the block).
	uint32_t working_value = 0u;
	if (active)
	{
		working_value = initialize_working_word(key, data_start + my_candidate, lane, expansion_values);
	}

	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		// Clear hash slots — first 64 threads do one slot each.
		if (threadIdx.x < 64u)
		{
			slot_fp[threadIdx.x]  = 0ull;
			slot_rep[threadIdx.x] = 0xFFFFFFFFu;
		}
		__syncthreads();

		// Run one map (inlined version of run_schedule_t's per-boundary body).
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
			const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, working_value, source_lane);
			uint8_t current_byte = static_cast<uint8_t>((source_word >> source_shift) & 0xFFu);
			if ((nibble_selector & 0x8000u) != 0u) current_byte = static_cast<uint8_t>(current_byte >> 4);
			const uint8_t alg_id = static_cast<uint8_t>((current_byte >> 1) & 0x07u);
			working_value = run_alg(working_value, lane, alg_id,
				&rng_seed, regular_rng_values, alg0_values, alg6_values,
				rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}

		// Cooperative dedup pass.
		// Compute warp-wide 64-bit fingerprint (reuses warp_hash_state).
		uint32_t h_lo, h_hi;
		warp_hash_state(working_value, lane, &h_lo, &h_hi);
		const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
		const unsigned long long fp = (fp_raw == 0ull) ? 1ull : fp_raw;

		// Lane 0 of each active warp walks the linear-probe chain.
		uint32_t my_rep = 0xFFFFFFFFu;
		if (lane == 0u && active)
		{
			uint32_t idx = static_cast<uint32_t>(fp) & 63u;
			for (uint32_t probe = 0; probe < 64u; probe++)
			{
				const unsigned long long prev = atomicCAS(&slot_fp[idx], 0ull, fp);
				if (prev == 0ull)
				{
					slot_rep[idx] = warp_index;
					__threadfence_block();
					my_rep = warp_index;
					break;
				}
				if (prev == fp)
				{
					// Spin until owner's slot_rep is visible (sentinel cleared)
					do { my_rep = slot_rep[idx]; } while (my_rep == 0xFFFFFFFFu);
					break;
				}
				idx = (idx + 1u) & 63u;
			}
			rep_warp[warp_index] = my_rep;
		}
		__syncthreads();
	}

	// Dump per-candidate final state + rep_warp metadata.
	if (active)
	{
		reinterpret_cast<uint32_t*>(output_state)[my_candidate * 32u + lane] = working_value;
		if (lane == 0u)
		{
			output_rep_index[my_candidate] = rep_warp[warp_index];
		}
	}
}

extern "C" __global__ __launch_bounds__(1024) void tm_dump_state_dedup_w32_cuda(
	uint8_t* output_state,
	uint32_t* output_rep_index,
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
	tm_dump_state_dedup_w32_impl<27u>(output_state, output_rep_index,
		regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128,
		alg2_values, alg5_values, expansion_values, schedule_data,
		key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(1024) void tm_dump_state_dedup_w32_cuda_skipcar(
	uint8_t* output_state,
	uint32_t* output_rep_index,
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
	tm_dump_state_dedup_w32_impl<26u>(output_state, output_rep_index,
		regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128,
		alg2_values, alg5_values, expansion_values, schedule_data,
		key, data_start, candidate_count);
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 1.5 dedup kernel: W=128 per block via ILP4 per warp.
//
// Geometry change from Phase 1 (W=32):
//   - 1024 threads/block = 32 warps × 32 lanes
//   - 4 candidates per warp (ILP4 — same pattern as tm_checksum_screen_cuda)
//   - W=128 candidates per block (32 warps × 4 ILP)
//
// Why: Phase 1 showed dedup adds ~0.43 ns/candidate/boundary fixed overhead
// (atomic CAS, warp_hash_state butterflies, slot init, syncs). At W=32 that
// overhead was 86% of baseline compute time → no path to a net win even with
// perfect compaction. Wider W amortizes the per-block fixed overheads
// (slot init, syncthreads) over 4× more candidates.
//
// Each warp does 4 sequential dedup passes per boundary (one per ILP slot),
// but the per-warp and per-block fixed costs (slot init, syncthreads, hash
// table size) are paid once per boundary regardless of W.
//
// Hash table: 256 slots (load factor 0.5 at W=128). ~3.5 KB shared mem.
template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_dump_state_dedup_w128_impl(
	uint8_t* output_state,
	uint32_t* output_rep_index,
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
	const uint32_t warp_index = threadIdx.x >> 5;     // 0..31
	const uint32_t my_candidate_base = blockIdx.x * 128u + warp_index * 4u;
	const bool active0 = (my_candidate_base + 0u) < candidate_count;
	const bool active1 = (my_candidate_base + 1u) < candidate_count;
	const bool active2 = (my_candidate_base + 2u) < candidate_count;
	const bool active3 = (my_candidate_base + 3u) < candidate_count;

	// Hash table: 256 slots × {64-bit fp, 32-bit rep} = ~3 KB shared mem
	__shared__ unsigned long long slot_fp[256];
	__shared__ volatile uint32_t  slot_rep[256];  // volatile for spin-wait correctness
	// Per-candidate final rep (only populated by lane 0 of each warp)
	__shared__ uint32_t           rep_per_candidate[128];

	// ILP4 working values
	uint32_t value0 = active0 ? initialize_working_word(key, data_start + my_candidate_base + 0u, lane, expansion_values) : 0u;
	uint32_t value1 = active1 ? initialize_working_word(key, data_start + my_candidate_base + 1u, lane, expansion_values) : 0u;
	uint32_t value2 = active2 ? initialize_working_word(key, data_start + my_candidate_base + 2u, lane, expansion_values) : 0u;
	uint32_t value3 = active3 ? initialize_working_word(key, data_start + my_candidate_base + 3u, lane, expansion_values) : 0u;

	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		// Init slot table — first 256 threads each init one slot
		if (threadIdx.x < 256u)
		{
			slot_fp[threadIdx.x]  = 0ull;
			slot_rep[threadIdx.x] = 0xFFFFFFFFu;
		}
		__syncthreads();

		// Run one map on all 4 ILP candidates in lockstep
		// (Inlined from run_schedule_quad_t's per-boundary body.)
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
			const uint32_t source_word0 = __shfl_sync(0xFFFFFFFFu, value0, source_lane);
			const uint32_t source_word1 = __shfl_sync(0xFFFFFFFFu, value1, source_lane);
			const uint32_t source_word2 = __shfl_sync(0xFFFFFFFFu, value2, source_lane);
			const uint32_t source_word3 = __shfl_sync(0xFFFFFFFFu, value3, source_lane);
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
			const uint8_t a0 = static_cast<uint8_t>((cb0 >> 1) & 0x07u);
			const uint8_t a1 = static_cast<uint8_t>((cb1 >> 1) & 0x07u);
			const uint8_t a2 = static_cast<uint8_t>((cb2 >> 1) & 0x07u);
			const uint8_t a3 = static_cast<uint8_t>((cb3 >> 1) & 0x07u);
			value0 = run_alg(value0, lane, a0, &rng_seed0, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			value1 = run_alg(value1, lane, a1, &rng_seed1, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			value2 = run_alg(value2, lane, a2, &rng_seed2, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			value3 = run_alg(value3, lane, a3, &rng_seed3, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}

		// Cooperative dedup pass — 4 sequential per-warp inserts (one per ILP slot)
		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			uint32_t value;
			bool active;
			switch (ilp)
			{
				case 0: value = value0; active = active0; break;
				case 1: value = value1; active = active1; break;
				case 2: value = value2; active = active2; break;
				default: value = value3; active = active3; break;
			}
			uint32_t h_lo, h_hi;
			warp_hash_state(value, lane, &h_lo, &h_hi);
			const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
			const unsigned long long fp = (fp_raw == 0ull) ? 1ull : fp_raw;

			if (lane == 0u && active)
			{
				const uint32_t my_candidate_index = warp_index * 4u + ilp;
				uint32_t idx = static_cast<uint32_t>(fp) & 255u;
				for (uint32_t probe = 0u; probe < 256u; probe++)
				{
					const unsigned long long prev = atomicCAS(&slot_fp[idx], 0ull, fp);
					if (prev == 0ull)
					{
						slot_rep[idx] = my_candidate_index;
						__threadfence_block();
						rep_per_candidate[my_candidate_index] = my_candidate_index;
						break;
					}
					if (prev == fp)
					{
						uint32_t r;
						do { r = slot_rep[idx]; } while (r == 0xFFFFFFFFu);
						rep_per_candidate[my_candidate_index] = r;
						break;
					}
					idx = (idx + 1u) & 255u;
				}
			}
		}
		__syncthreads();
	}

	// Dump per-candidate final state + rep index (each candidate uses one warp's lanes)
	const uint32_t cand0_idx = my_candidate_base + 0u;
	const uint32_t cand1_idx = my_candidate_base + 1u;
	const uint32_t cand2_idx = my_candidate_base + 2u;
	const uint32_t cand3_idx = my_candidate_base + 3u;
	if (active0) reinterpret_cast<uint32_t*>(output_state)[cand0_idx * 32u + lane] = value0;
	if (active1) reinterpret_cast<uint32_t*>(output_state)[cand1_idx * 32u + lane] = value1;
	if (active2) reinterpret_cast<uint32_t*>(output_state)[cand2_idx * 32u + lane] = value2;
	if (active3) reinterpret_cast<uint32_t*>(output_state)[cand3_idx * 32u + lane] = value3;
	if (lane == 0u)
	{
		if (active0) output_rep_index[cand0_idx] = rep_per_candidate[warp_index * 4u + 0u];
		if (active1) output_rep_index[cand1_idx] = rep_per_candidate[warp_index * 4u + 1u];
		if (active2) output_rep_index[cand2_idx] = rep_per_candidate[warp_index * 4u + 2u];
		if (active3) output_rep_index[cand3_idx] = rep_per_candidate[warp_index * 4u + 3u];
	}
}

extern "C" __global__ __launch_bounds__(1024) void tm_dump_state_dedup_w128_cuda(
	uint8_t* output_state, uint32_t* output_rep_index,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_dump_state_dedup_w128_impl<27u>(output_state, output_rep_index,
		regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128,
		alg2_values, alg5_values, expansion_values, schedule_data,
		key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(1024) void tm_dump_state_dedup_w128_cuda_skipcar(
	uint8_t* output_state, uint32_t* output_rep_index,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_dump_state_dedup_w128_impl<26u>(output_state, output_rep_index,
		regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128,
		alg2_values, alg5_values, expansion_values, schedule_data,
		key, data_start, candidate_count);
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 2 dedup kernel: W=128 + compaction/skip.
//
// Builds on Phase 1.5 (W=128 ILP4): once a candidate is found to be a
// duplicate at some boundary, it's marked dead and skips run_alg in all
// subsequent boundaries. The expensive RNG-table loads are bypassed for dead
// ILP slots within each warp via per-ILP-slot uniform branches (32 lanes of
// a warp share the same alive flags → no divergence).
//
// Dead candidates' final state in output_state is STALE (frozen at the
// boundary where they were marked dead). Host validation follows the
// rep_per_candidate chain to retrieve the correct final state.
//
// Expected gain vs Phase 1.5: ~1/R compute reduction post-first-dedup,
// minus dedup overhead. At R=2 typical → ~1.4× over screen kernel baseline.
template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_dump_state_dedup_w128_skip_impl(
	uint8_t* output_state,
	uint32_t* output_rep_index,
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
	const uint32_t my_candidate_base = blockIdx.x * 128u + warp_index * 4u;
	const uint32_t cb = warp_index * 4u;

	__shared__ unsigned long long slot_fp[256];
	// volatile prevents the compiler from caching the spin-wait read in a
	// register, which would otherwise turn the do/while into an infinite loop
	// the first time slot_rep[idx] is read while still 0xFFFFFFFFu.
	__shared__ volatile uint32_t  slot_rep[256];
	__shared__ uint32_t           rep_per_candidate[128];
	__shared__ uint8_t            alive_per_candidate[128];

	// Initialize per-candidate metadata (rep = self, alive = active flag)
	if (lane == 0u)
	{
		const bool a0 = (my_candidate_base + 0u) < candidate_count;
		const bool a1 = (my_candidate_base + 1u) < candidate_count;
		const bool a2 = (my_candidate_base + 2u) < candidate_count;
		const bool a3 = (my_candidate_base + 3u) < candidate_count;
		alive_per_candidate[cb + 0] = a0 ? 1u : 0u;
		alive_per_candidate[cb + 1] = a1 ? 1u : 0u;
		alive_per_candidate[cb + 2] = a2 ? 1u : 0u;
		alive_per_candidate[cb + 3] = a3 ? 1u : 0u;
		rep_per_candidate[cb + 0] = cb + 0u;
		rep_per_candidate[cb + 1] = cb + 1u;
		rep_per_candidate[cb + 2] = cb + 2u;
		rep_per_candidate[cb + 3] = cb + 3u;
	}
	__syncthreads();

	const bool active0_init = alive_per_candidate[cb + 0] != 0u;
	const bool active1_init = alive_per_candidate[cb + 1] != 0u;
	const bool active2_init = alive_per_candidate[cb + 2] != 0u;
	const bool active3_init = alive_per_candidate[cb + 3] != 0u;

	uint32_t value0 = active0_init ? initialize_working_word(key, data_start + my_candidate_base + 0u, lane, expansion_values) : 0u;
	uint32_t value1 = active1_init ? initialize_working_word(key, data_start + my_candidate_base + 1u, lane, expansion_values) : 0u;
	uint32_t value2 = active2_init ? initialize_working_word(key, data_start + my_candidate_base + 2u, lane, expansion_values) : 0u;
	uint32_t value3 = active3_init ? initialize_working_word(key, data_start + my_candidate_base + 3u, lane, expansion_values) : 0u;

	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		// Init slot table
		if (threadIdx.x < 256u)
		{
			slot_fp[threadIdx.x]  = 0ull;
			slot_rep[threadIdx.x] = 0xFFFFFFFFu;
		}
		__syncthreads();

		// Read alive flags for this warp's ILP slots (uniform across lanes)
		const bool alive0 = alive_per_candidate[cb + 0] != 0u;
		const bool alive1 = alive_per_candidate[cb + 1] != 0u;
		const bool alive2 = alive_per_candidate[cb + 2] != 0u;
		const bool alive3 = alive_per_candidate[cb + 3] != 0u;

		// Run one map — skip run_alg for dead ILP slots (uniform branch per warp)
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
			const uint32_t source_word0 = __shfl_sync(0xFFFFFFFFu, value0, source_lane);
			const uint32_t source_word1 = __shfl_sync(0xFFFFFFFFu, value1, source_lane);
			const uint32_t source_word2 = __shfl_sync(0xFFFFFFFFu, value2, source_lane);
			const uint32_t source_word3 = __shfl_sync(0xFFFFFFFFu, value3, source_lane);
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
			const uint8_t a0 = static_cast<uint8_t>((cb0 >> 1) & 0x07u);
			const uint8_t a1 = static_cast<uint8_t>((cb1 >> 1) & 0x07u);
			const uint8_t a2 = static_cast<uint8_t>((cb2 >> 1) & 0x07u);
			const uint8_t a3 = static_cast<uint8_t>((cb3 >> 1) & 0x07u);
			if (alive0) value0 = run_alg(value0, lane, a0, &rng_seed0, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive1) value1 = run_alg(value1, lane, a1, &rng_seed1, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive2) value2 = run_alg(value2, lane, a2, &rng_seed2, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive3) value3 = run_alg(value3, lane, a3, &rng_seed3, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}

		// Cooperative dedup pass — alive candidates only insert.
		// Dead candidates skip hash (their state hasn't changed; their rep was set previously).
		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			uint32_t value;
			bool alive;
			switch (ilp)
			{
				case 0: value = value0; alive = alive0; break;
				case 1: value = value1; alive = alive1; break;
				case 2: value = value2; alive = alive2; break;
				default: value = value3; alive = alive3; break;
			}
			if (!alive)
			{
				// Still need to participate in __syncthreads via the outer
				// __syncthreads after the unrolled loop. The warp_hash_state
				// uses __shfl_sync over the whole warp — but since alive is
				// uniform across all 32 lanes (it's a per-warp value), all
				// lanes skip together → no divergence.
				continue;
			}
			uint32_t h_lo, h_hi;
			warp_hash_state(value, lane, &h_lo, &h_hi);
			const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
			const unsigned long long fp = (fp_raw == 0ull) ? 1ull : fp_raw;

			if (lane == 0u)
			{
				const uint32_t my_idx = cb + ilp;
				uint32_t idx = static_cast<uint32_t>(fp) & 255u;
				for (uint32_t probe = 0u; probe < 256u; probe++)
				{
					const unsigned long long prev = atomicCAS(&slot_fp[idx], 0ull, fp);
					if (prev == 0ull)
					{
						// Owner path: write rep then fence so followers see it.
						// Without the fence, a follower warp's atomicCAS can see
						// the new fp while its subsequent non-atomic load of
						// slot_rep still reads the stale 0xFFFFFFFF sentinel —
						// confirmed by ncu, which perturbs scheduling enough to
						// expose the race.
						slot_rep[idx] = my_idx;
						__threadfence_block();
						break;
					}
					if (prev == fp)
					{
						// Follower path: spin-wait until owner's slot_rep write
						// is visible. Owner is in the same block and just
						// succeeded its CAS; the wait is bounded by L1 cache
						// propagation (sub-microsecond on Blackwell).
						uint32_t my_rep;
						do { my_rep = slot_rep[idx]; } while (my_rep == 0xFFFFFFFFu);
						rep_per_candidate[my_idx] = my_rep;
						alive_per_candidate[my_idx] = 0u;
						break;
					}
					idx = (idx + 1u) & 255u;
				}
			}
		}
		__syncthreads();
	}

	// Dump per-candidate final state (stale for dead candidates) + rep index.
	// Host validation will follow the rep chain to the alive descendant.
	const uint32_t cand0_idx = my_candidate_base + 0u;
	const uint32_t cand1_idx = my_candidate_base + 1u;
	const uint32_t cand2_idx = my_candidate_base + 2u;
	const uint32_t cand3_idx = my_candidate_base + 3u;
	if (cand0_idx < candidate_count) reinterpret_cast<uint32_t*>(output_state)[cand0_idx * 32u + lane] = value0;
	if (cand1_idx < candidate_count) reinterpret_cast<uint32_t*>(output_state)[cand1_idx * 32u + lane] = value1;
	if (cand2_idx < candidate_count) reinterpret_cast<uint32_t*>(output_state)[cand2_idx * 32u + lane] = value2;
	if (cand3_idx < candidate_count) reinterpret_cast<uint32_t*>(output_state)[cand3_idx * 32u + lane] = value3;
	if (lane == 0u)
	{
		if (cand0_idx < candidate_count) output_rep_index[cand0_idx] = rep_per_candidate[cb + 0u];
		if (cand1_idx < candidate_count) output_rep_index[cand1_idx] = rep_per_candidate[cb + 1u];
		if (cand2_idx < candidate_count) output_rep_index[cand2_idx] = rep_per_candidate[cb + 2u];
		if (cand3_idx < candidate_count) output_rep_index[cand3_idx] = rep_per_candidate[cb + 3u];
	}
}

extern "C" __global__ __launch_bounds__(1024) void tm_dump_state_dedup_w128_skip_cuda(
	uint8_t* output_state, uint32_t* output_rep_index,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_dump_state_dedup_w128_skip_impl<27u>(output_state, output_rep_index,
		regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128,
		alg2_values, alg5_values, expansion_values, schedule_data,
		key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(1024) void tm_dump_state_dedup_w128_skip_cuda_skipcar(
	uint8_t* output_state, uint32_t* output_rep_index,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_dump_state_dedup_w128_skip_impl<26u>(output_state, output_rep_index,
		regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128,
		alg2_values, alg5_values, expansion_values, schedule_data,
		key, data_start, candidate_count);
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 2.5 dedup kernel variants — block-size sweep for occupancy tuning.
//
// ncu on Phase 2 (1024-thread blocks) showed 66.7% theoretical occupancy
// limited by "Block Limit Warps: 1" (32 warps/block, max 48 warps/SM). Going
// to smaller blocks should give multiple blocks/SM → higher occupancy.
//
// Geometry:
//   W=32  → 256 threads/block (8 warps × 4 ILP), expect 6 blocks/SM = 48 warps = 100% occ
//   W=64  → 512 threads/block (16 warps × 4 ILP), expect 3 blocks/SM = 48 warps = 100% occ
//
// Templated on W; same skip semantics as the W=128 Phase 2 kernel.
template<uint32_t SCHEDULE_COUNT, uint32_t W, uint32_t NSLOTS>
__device__ __forceinline__ void tm_dump_state_dedup_skip_tmpl(
	uint8_t* output_state,
	uint32_t* output_rep_index,
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
	const uint32_t my_candidate_base = blockIdx.x * W + warp_index * 4u;
	const uint32_t cb = warp_index * 4u;
	const uint32_t slot_mask = NSLOTS - 1u;

	__shared__ unsigned long long slot_fp[NSLOTS];
	__shared__ volatile uint32_t  slot_rep[NSLOTS];
	__shared__ uint32_t           rep_per_candidate[W];
	__shared__ uint8_t            alive_per_candidate[W];

	if (lane == 0u)
	{
		const bool a0 = (my_candidate_base + 0u) < candidate_count;
		const bool a1 = (my_candidate_base + 1u) < candidate_count;
		const bool a2 = (my_candidate_base + 2u) < candidate_count;
		const bool a3 = (my_candidate_base + 3u) < candidate_count;
		alive_per_candidate[cb + 0] = a0 ? 1u : 0u;
		alive_per_candidate[cb + 1] = a1 ? 1u : 0u;
		alive_per_candidate[cb + 2] = a2 ? 1u : 0u;
		alive_per_candidate[cb + 3] = a3 ? 1u : 0u;
		rep_per_candidate[cb + 0] = cb + 0u;
		rep_per_candidate[cb + 1] = cb + 1u;
		rep_per_candidate[cb + 2] = cb + 2u;
		rep_per_candidate[cb + 3] = cb + 3u;
	}
	__syncthreads();

	const bool a0i = alive_per_candidate[cb + 0] != 0u;
	const bool a1i = alive_per_candidate[cb + 1] != 0u;
	const bool a2i = alive_per_candidate[cb + 2] != 0u;
	const bool a3i = alive_per_candidate[cb + 3] != 0u;

	uint32_t value0 = a0i ? initialize_working_word(key, data_start + my_candidate_base + 0u, lane, expansion_values) : 0u;
	uint32_t value1 = a1i ? initialize_working_word(key, data_start + my_candidate_base + 1u, lane, expansion_values) : 0u;
	uint32_t value2 = a2i ? initialize_working_word(key, data_start + my_candidate_base + 2u, lane, expansion_values) : 0u;
	uint32_t value3 = a3i ? initialize_working_word(key, data_start + my_candidate_base + 3u, lane, expansion_values) : 0u;

	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		if (threadIdx.x < NSLOTS)
		{
			slot_fp[threadIdx.x]  = 0ull;
			slot_rep[threadIdx.x] = 0xFFFFFFFFu;
		}
		__syncthreads();

		const bool alive0 = alive_per_candidate[cb + 0] != 0u;
		const bool alive1 = alive_per_candidate[cb + 1] != 0u;
		const bool alive2 = alive_per_candidate[cb + 2] != 0u;
		const bool alive3 = alive_per_candidate[cb + 3] != 0u;

		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

		uint16_t rs0 = static_cast<uint16_t>(((packed_schedule & 0xFFu) << 8) | ((packed_schedule & 0xFF00u) >> 8));
		uint16_t rs1 = rs0, rs2 = rs0, rs3 = rs0;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));

		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, value0, sl);
			const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, value1, sl);
			const uint32_t w2 = __shfl_sync(0xFFFFFFFFu, value2, sl);
			const uint32_t w3 = __shfl_sync(0xFFFFFFFFu, value3, sl);
			uint8_t cb0 = static_cast<uint8_t>((w0 >> ss) & 0xFFu);
			uint8_t cb1 = static_cast<uint8_t>((w1 >> ss) & 0xFFu);
			uint8_t cb2 = static_cast<uint8_t>((w2 >> ss) & 0xFFu);
			uint8_t cb3 = static_cast<uint8_t>((w3 >> ss) & 0xFFu);
			if ((nsel & 0x8000u) != 0u) { cb0 >>= 4; cb1 >>= 4; cb2 >>= 4; cb3 >>= 4; }
			const uint8_t a0 = static_cast<uint8_t>((cb0 >> 1) & 0x07u);
			const uint8_t a1 = static_cast<uint8_t>((cb1 >> 1) & 0x07u);
			const uint8_t a2 = static_cast<uint8_t>((cb2 >> 1) & 0x07u);
			const uint8_t a3 = static_cast<uint8_t>((cb3 >> 1) & 0x07u);
			if (alive0) value0 = run_alg(value0, lane, a0, &rs0, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive1) value1 = run_alg(value1, lane, a1, &rs1, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive2) value2 = run_alg(value2, lane, a2, &rs2, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive3) value3 = run_alg(value3, lane, a3, &rs3, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nsel = static_cast<uint16_t>(nsel << 1);
		}

		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			uint32_t value;
			bool alive;
			switch (ilp)
			{
				case 0: value = value0; alive = alive0; break;
				case 1: value = value1; alive = alive1; break;
				case 2: value = value2; alive = alive2; break;
				default: value = value3; alive = alive3; break;
			}
			if (!alive) continue;
			uint32_t h_lo, h_hi;
			warp_hash_state(value, lane, &h_lo, &h_hi);
			const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
			const unsigned long long fp = (fp_raw == 0ull) ? 1ull : fp_raw;

			if (lane == 0u)
			{
				const uint32_t my_idx = cb + ilp;
				uint32_t idx = static_cast<uint32_t>(fp) & slot_mask;
				for (uint32_t probe = 0u; probe < NSLOTS; probe++)
				{
					const unsigned long long prev = atomicCAS(&slot_fp[idx], 0ull, fp);
					if (prev == 0ull)
					{
						slot_rep[idx] = my_idx;
						__threadfence_block();
						break;
					}
					if (prev == fp)
					{
						uint32_t r;
						do { r = slot_rep[idx]; } while (r == 0xFFFFFFFFu);
						rep_per_candidate[my_idx] = r;
						alive_per_candidate[my_idx] = 0u;
						break;
					}
					idx = (idx + 1u) & slot_mask;
				}
			}
		}
		__syncthreads();
	}

	const uint32_t c0 = my_candidate_base + 0u;
	const uint32_t c1 = my_candidate_base + 1u;
	const uint32_t c2 = my_candidate_base + 2u;
	const uint32_t c3 = my_candidate_base + 3u;
	if (c0 < candidate_count) reinterpret_cast<uint32_t*>(output_state)[c0 * 32u + lane] = value0;
	if (c1 < candidate_count) reinterpret_cast<uint32_t*>(output_state)[c1 * 32u + lane] = value1;
	if (c2 < candidate_count) reinterpret_cast<uint32_t*>(output_state)[c2 * 32u + lane] = value2;
	if (c3 < candidate_count) reinterpret_cast<uint32_t*>(output_state)[c3 * 32u + lane] = value3;
	if (lane == 0u)
	{
		if (c0 < candidate_count) output_rep_index[c0] = rep_per_candidate[cb + 0u];
		if (c1 < candidate_count) output_rep_index[c1] = rep_per_candidate[cb + 1u];
		if (c2 < candidate_count) output_rep_index[c2] = rep_per_candidate[cb + 2u];
		if (c3 < candidate_count) output_rep_index[c3] = rep_per_candidate[cb + 3u];
	}
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_dump_state_dedup_w32_skip_cuda(
	uint8_t* output_state, uint32_t* output_rep_index,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_dump_state_dedup_skip_tmpl<27u, 32u, 64u>(output_state, output_rep_index,
		regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128,
		alg2_values, alg5_values, expansion_values, schedule_data,
		key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(512, 2) void tm_dump_state_dedup_w64_skip_cuda(
	uint8_t* output_state, uint32_t* output_rep_index,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_dump_state_dedup_skip_tmpl<27u, 64u, 128u>(output_state, output_rep_index,
		regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128,
		alg2_values, alg5_values, expansion_values, schedule_data,
		key, data_start, candidate_count);
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 2.5 production-shape kernel: checksum-screen with state dedup.
//
// Same dedup-skip semantics as tm_dump_state_dedup_w32_skip_cuda but the
// final-boundary output is the 1-byte carnival/other-world screen flag
// per candidate (matching tm_checksum_screen_cuda's output shape) rather
// than 128 bytes of full state.
//
// This is the fair production-comparison kernel: same output as the
// existing screen kernel, adds dedup on top.
//
// Geometry: 256 threads = 8 warps × 4 ILP = W=32 candidates per block.
template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_checksum_screen_dedup_w32_impl(
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
	constexpr uint32_t W = 32u;
	constexpr uint32_t NSLOTS = 64u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t my_candidate_base = blockIdx.x * W + warp_index * 4u;
	const uint32_t cb = warp_index * 4u;
	const uint32_t slot_mask = NSLOTS - 1u;

	__shared__ unsigned long long slot_fp[NSLOTS];
	__shared__ volatile uint32_t  slot_rep[NSLOTS];
	__shared__ uint32_t           rep_per_candidate[W];
	__shared__ uint8_t            alive_per_candidate[W];
	__shared__ uint8_t            flag_per_candidate[W];

	if (lane == 0u)
	{
		const bool a0 = (my_candidate_base + 0u) < candidate_count;
		const bool a1 = (my_candidate_base + 1u) < candidate_count;
		const bool a2 = (my_candidate_base + 2u) < candidate_count;
		const bool a3 = (my_candidate_base + 3u) < candidate_count;
		alive_per_candidate[cb + 0] = a0 ? 1u : 0u;
		alive_per_candidate[cb + 1] = a1 ? 1u : 0u;
		alive_per_candidate[cb + 2] = a2 ? 1u : 0u;
		alive_per_candidate[cb + 3] = a3 ? 1u : 0u;
		rep_per_candidate[cb + 0] = cb + 0u;
		rep_per_candidate[cb + 1] = cb + 1u;
		rep_per_candidate[cb + 2] = cb + 2u;
		rep_per_candidate[cb + 3] = cb + 3u;
	}
	__syncthreads();

	const bool a0i = alive_per_candidate[cb + 0] != 0u;
	const bool a1i = alive_per_candidate[cb + 1] != 0u;
	const bool a2i = alive_per_candidate[cb + 2] != 0u;
	const bool a3i = alive_per_candidate[cb + 3] != 0u;

	uint32_t value0 = a0i ? initialize_working_word(key, data_start + my_candidate_base + 0u, lane, expansion_values) : 0u;
	uint32_t value1 = a1i ? initialize_working_word(key, data_start + my_candidate_base + 1u, lane, expansion_values) : 0u;
	uint32_t value2 = a2i ? initialize_working_word(key, data_start + my_candidate_base + 2u, lane, expansion_values) : 0u;
	uint32_t value3 = a3i ? initialize_working_word(key, data_start + my_candidate_base + 3u, lane, expansion_values) : 0u;

	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		if (threadIdx.x < NSLOTS)
		{
			slot_fp[threadIdx.x]  = 0ull;
			slot_rep[threadIdx.x] = 0xFFFFFFFFu;
		}
		__syncthreads();

		const bool alive0 = alive_per_candidate[cb + 0] != 0u;
		const bool alive1 = alive_per_candidate[cb + 1] != 0u;
		const bool alive2 = alive_per_candidate[cb + 2] != 0u;
		const bool alive3 = alive_per_candidate[cb + 3] != 0u;

		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

		uint16_t rs0 = static_cast<uint16_t>(((packed_schedule & 0xFFu) << 8) | ((packed_schedule & 0xFF00u) >> 8));
		uint16_t rs1 = rs0, rs2 = rs0, rs3 = rs0;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));

		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, value0, sl);
			const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, value1, sl);
			const uint32_t w2 = __shfl_sync(0xFFFFFFFFu, value2, sl);
			const uint32_t w3 = __shfl_sync(0xFFFFFFFFu, value3, sl);
			uint8_t cb0 = static_cast<uint8_t>((w0 >> ss) & 0xFFu);
			uint8_t cb1 = static_cast<uint8_t>((w1 >> ss) & 0xFFu);
			uint8_t cb2 = static_cast<uint8_t>((w2 >> ss) & 0xFFu);
			uint8_t cb3 = static_cast<uint8_t>((w3 >> ss) & 0xFFu);
			if ((nsel & 0x8000u) != 0u) { cb0 >>= 4; cb1 >>= 4; cb2 >>= 4; cb3 >>= 4; }
			const uint8_t a0 = static_cast<uint8_t>((cb0 >> 1) & 0x07u);
			const uint8_t a1 = static_cast<uint8_t>((cb1 >> 1) & 0x07u);
			const uint8_t a2 = static_cast<uint8_t>((cb2 >> 1) & 0x07u);
			const uint8_t a3 = static_cast<uint8_t>((cb3 >> 1) & 0x07u);
			if (alive0) value0 = run_alg(value0, lane, a0, &rs0, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive1) value1 = run_alg(value1, lane, a1, &rs1, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive2) value2 = run_alg(value2, lane, a2, &rs2, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive3) value3 = run_alg(value3, lane, a3, &rs3, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nsel = static_cast<uint16_t>(nsel << 1);
		}

		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			uint32_t value;
			bool alive;
			switch (ilp)
			{
				case 0: value = value0; alive = alive0; break;
				case 1: value = value1; alive = alive1; break;
				case 2: value = value2; alive = alive2; break;
				default: value = value3; alive = alive3; break;
			}
			if (!alive) continue;
			uint32_t h_lo, h_hi;
			warp_hash_state(value, lane, &h_lo, &h_hi);
			const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
			const unsigned long long fp = (fp_raw == 0ull) ? 1ull : fp_raw;

			if (lane == 0u)
			{
				const uint32_t my_idx = cb + ilp;
				uint32_t idx = static_cast<uint32_t>(fp) & slot_mask;
				for (uint32_t probe = 0u; probe < NSLOTS; probe++)
				{
					const unsigned long long prev = atomicCAS(&slot_fp[idx], 0ull, fp);
					if (prev == 0ull)
					{
						slot_rep[idx] = my_idx;
						__threadfence_block();
						break;
					}
					if (prev == fp)
					{
						uint32_t r;
						do { r = slot_rep[idx]; } while (r == 0xFFFFFFFFu);
						rep_per_candidate[my_idx] = r;
						alive_per_candidate[my_idx] = 0u;
						break;
					}
					idx = (idx + 1u) & slot_mask;
				}
			}
		}
		__syncthreads();
	}

	// Post-schedule: ALL warps run screen_candidate on all 4 ILPs.
	// screen_candidate uses __shfl_sync across the whole warp, so all lanes
	// must participate. Dead candidates' computed flags are STALE-state-based
	// and therefore wrong; we'll overwrite them below by following the rep chain.
	const uint8_t f0 = screen_candidate(value0, lane, carnival_data);
	const uint8_t f1 = screen_candidate(value1, lane, carnival_data);
	const uint8_t f2 = screen_candidate(value2, lane, carnival_data);
	const uint8_t f3 = screen_candidate(value3, lane, carnival_data);

	// Lane 0 stages ALIVE candidates' flags into shared mem.
	if (lane == 0u)
	{
		const bool al0 = alive_per_candidate[cb + 0] != 0u;
		const bool al1 = alive_per_candidate[cb + 1] != 0u;
		const bool al2 = alive_per_candidate[cb + 2] != 0u;
		const bool al3 = alive_per_candidate[cb + 3] != 0u;
		if (al0) flag_per_candidate[cb + 0] = f0;
		if (al1) flag_per_candidate[cb + 1] = f1;
		if (al2) flag_per_candidate[cb + 2] = f2;
		if (al3) flag_per_candidate[cb + 3] = f3;
	}
	__syncthreads();

	// Lane 0 writes per-candidate output. Alive candidates use their own flag;
	// dead candidates follow the rep chain (at most W hops) to their alive rep
	// and inherit its flag.
	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			const uint32_t c_local = cb + ilp;
			const uint32_t c_global = my_candidate_base + ilp;
			if (c_global >= candidate_count) continue;
			uint32_t cur = c_local;
			#pragma unroll 1
			for (uint32_t hop = 0u; hop < W; hop++)
			{
				const uint32_t r = rep_per_candidate[cur];
				if (r == cur || r >= W) break;
				cur = r;
			}
			result_data[c_global] = flag_per_candidate[cur];
		}
	}
}

extern "C" __global__ __launch_bounds__(256, 6) void tm_checksum_screen_dedup_w32_cuda(
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
	tm_checksum_screen_dedup_w32_impl<27u>(result_data, regular_rng_values, alg0_values, alg6_values,
		rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values,
		schedule_data, carnival_data, key, data_start, candidate_count);
}

template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_checksum_screen_dedup_w32_offset_impl(
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
	constexpr uint32_t W = 32u;
	constexpr uint32_t NSLOTS = 64u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t my_candidate_base = blockIdx.x * W + warp_index * 4u;
	const uint32_t cb = warp_index * 4u;
	const uint32_t slot_mask = NSLOTS - 1u;

	__shared__ unsigned long long slot_fp[NSLOTS];
	__shared__ volatile uint32_t  slot_rep[NSLOTS];
	__shared__ uint32_t           rep_per_candidate[W];
	__shared__ uint8_t            alive_per_candidate[W];
	__shared__ uint8_t            flag_per_candidate[W];

	if (lane == 0u)
	{
		const bool a0 = (my_candidate_base + 0u) < candidate_count;
		const bool a1 = (my_candidate_base + 1u) < candidate_count;
		const bool a2 = (my_candidate_base + 2u) < candidate_count;
		const bool a3 = (my_candidate_base + 3u) < candidate_count;
		alive_per_candidate[cb + 0] = a0 ? 1u : 0u;
		alive_per_candidate[cb + 1] = a1 ? 1u : 0u;
		alive_per_candidate[cb + 2] = a2 ? 1u : 0u;
		alive_per_candidate[cb + 3] = a3 ? 1u : 0u;
		rep_per_candidate[cb + 0] = cb + 0u;
		rep_per_candidate[cb + 1] = cb + 1u;
		rep_per_candidate[cb + 2] = cb + 2u;
		rep_per_candidate[cb + 3] = cb + 3u;
	}
	__syncthreads();

	const bool a0i = alive_per_candidate[cb + 0] != 0u;
	const bool a1i = alive_per_candidate[cb + 1] != 0u;
	const bool a2i = alive_per_candidate[cb + 2] != 0u;
	const bool a3i = alive_per_candidate[cb + 3] != 0u;

	uint32_t value0 = a0i ? initialize_working_word(key, data_start + my_candidate_base + 0u, lane, expansion_values) : 0u;
	uint32_t value1 = a1i ? initialize_working_word(key, data_start + my_candidate_base + 1u, lane, expansion_values) : 0u;
	uint32_t value2 = a2i ? initialize_working_word(key, data_start + my_candidate_base + 2u, lane, expansion_values) : 0u;
	uint32_t value3 = a3i ? initialize_working_word(key, data_start + my_candidate_base + 3u, lane, expansion_values) : 0u;

	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		if (threadIdx.x < NSLOTS)
		{
			slot_fp[threadIdx.x]  = 0ull;
			slot_rep[threadIdx.x] = 0xFFFFFFFFu;
		}
		__syncthreads();

		const bool alive0 = alive_per_candidate[cb + 0] != 0u;
		const bool alive1 = alive_per_candidate[cb + 1] != 0u;
		const bool alive2 = alive_per_candidate[cb + 2] != 0u;
		const bool alive3 = alive_per_candidate[cb + 3] != 0u;

		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

		uint32_t ro0 = 0u, ro1 = 0u, ro2 = 0u, ro3 = 0u;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));

		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, value0, sl);
			const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, value1, sl);
			const uint32_t w2 = __shfl_sync(0xFFFFFFFFu, value2, sl);
			const uint32_t w3 = __shfl_sync(0xFFFFFFFFu, value3, sl);
			uint8_t cb0 = static_cast<uint8_t>((w0 >> ss) & 0xFFu);
			uint8_t cb1 = static_cast<uint8_t>((w1 >> ss) & 0xFFu);
			uint8_t cb2 = static_cast<uint8_t>((w2 >> ss) & 0xFFu);
			uint8_t cb3 = static_cast<uint8_t>((w3 >> ss) & 0xFFu);
			if ((nsel & 0x8000u) != 0u) { cb0 >>= 4; cb1 >>= 4; cb2 >>= 4; cb3 >>= 4; }
			const uint8_t a0 = static_cast<uint8_t>((cb0 >> 1) & 0x07u);
			const uint8_t a1 = static_cast<uint8_t>((cb1 >> 1) & 0x07u);
			const uint8_t a2 = static_cast<uint8_t>((cb2 >> 1) & 0x07u);
			const uint8_t a3 = static_cast<uint8_t>((cb3 >> 1) & 0x07u);
			if (alive0) value0 = run_alg_offset(value0, lane, a0, schedule_index, &ro0, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			if (alive1) value1 = run_alg_offset(value1, lane, a1, schedule_index, &ro1, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			if (alive2) value2 = run_alg_offset(value2, lane, a2, schedule_index, &ro2, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			if (alive3) value3 = run_alg_offset(value3, lane, a3, schedule_index, &ro3, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			nsel = static_cast<uint16_t>(nsel << 1);
		}

		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			uint32_t value;
			bool alive;
			switch (ilp)
			{
				case 0: value = value0; alive = alive0; break;
				case 1: value = value1; alive = alive1; break;
				case 2: value = value2; alive = alive2; break;
				default: value = value3; alive = alive3; break;
			}
			if (!alive) continue;
			uint32_t h_lo, h_hi;
			warp_hash_state(value, lane, &h_lo, &h_hi);
			const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
			const unsigned long long fp = (fp_raw == 0ull) ? 1ull : fp_raw;

			if (lane == 0u)
			{
				const uint32_t my_idx = cb + ilp;
				uint32_t idx = static_cast<uint32_t>(fp) & slot_mask;
				for (uint32_t probe = 0u; probe < NSLOTS; probe++)
				{
					const unsigned long long prev = atomicCAS(&slot_fp[idx], 0ull, fp);
					if (prev == 0ull)
					{
						slot_rep[idx] = my_idx;
						__threadfence_block();
						break;
					}
					if (prev == fp)
					{
						uint32_t r;
						do { r = slot_rep[idx]; } while (r == 0xFFFFFFFFu);
						rep_per_candidate[my_idx] = r;
						alive_per_candidate[my_idx] = 0u;
						break;
					}
					idx = (idx + 1u) & slot_mask;
				}
			}
		}
		__syncthreads();
	}

	const uint8_t f0 = screen_candidate(value0, lane, carnival_data);
	const uint8_t f1 = screen_candidate(value1, lane, carnival_data);
	const uint8_t f2 = screen_candidate(value2, lane, carnival_data);
	const uint8_t f3 = screen_candidate(value3, lane, carnival_data);

	if (lane == 0u)
	{
		const bool al0 = alive_per_candidate[cb + 0] != 0u;
		const bool al1 = alive_per_candidate[cb + 1] != 0u;
		const bool al2 = alive_per_candidate[cb + 2] != 0u;
		const bool al3 = alive_per_candidate[cb + 3] != 0u;
		if (al0) flag_per_candidate[cb + 0] = f0;
		if (al1) flag_per_candidate[cb + 1] = f1;
		if (al2) flag_per_candidate[cb + 2] = f2;
		if (al3) flag_per_candidate[cb + 3] = f3;
	}
	__syncthreads();

	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			const uint32_t c_local = cb + ilp;
			const uint32_t c_global = my_candidate_base + ilp;
			if (c_global >= candidate_count) continue;
			uint32_t cur = c_local;
			#pragma unroll 1
			for (uint32_t hop = 0u; hop < W; hop++)
			{
				const uint32_t r = rep_per_candidate[cur];
				if (r == cur || r >= W) break;
				cur = r;
			}
			result_data[c_global] = flag_per_candidate[cur];
		}
	}
}

extern "C" __global__ __launch_bounds__(256, 6) void tm_checksum_screen_dedup_w32_offset_cuda(
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
	tm_checksum_screen_dedup_w32_offset_impl<27u>(result_data, regular_rng_values, alg0_values, alg6_values,
		alg2_values, alg5_values, expansion_values, schedule_data, carnival_data,
		key, data_start, candidate_count);
}

// ────────────────────────────────────────────────────────────────────────────
// ILP-geometry offset dedup experiment (2026-05-29).
//
// The production SCREEN runs at 128t / ILP6 (one warp per candidate, 6
// candidates/warp). All prior dedup kernels were locked to 256t / ILP4 / W=32,
// which starts ~27% behind the ILP6 screen geometry. This template lets the
// dedup match the screen's geometry: parametrized on WARPS/block, ILP, hash-slot
// count, and dedup period (k4 = first + every 4th was the schedule optimum).
//
//   W = WARPS * ILP candidates/block; NSLOTS ≈ 2*W (power of two).
//   Packed single-slot table fp(56)|rep(8), same as the preext sched kernel.
//   survivor_count (optional): block-local unique-state atomic counter.
template<uint32_t SCHEDULE_COUNT, uint32_t WARPS, uint32_t ILP, uint32_t NSLOTS, uint32_t DEDUP_PERIOD>
__device__ __forceinline__ void tm_checksum_screen_dedup_offset_ilp_sched_impl(
	uint8_t* result_data,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	unsigned long long* survivor_count,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	constexpr uint32_t W = WARPS * ILP;
	constexpr unsigned long long FP_MASK = 0xFFFFFFFFFFFFFF00ull;
	const uint32_t slot_mask = NSLOTS - 1u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t cb = warp_index * ILP;
	const uint32_t my_candidate_base = blockIdx.x * W + cb;

	__shared__ unsigned long long slot[NSLOTS];
	__shared__ uint32_t           rep_per_candidate[W];
	__shared__ uint8_t            alive_per_candidate[W];
	__shared__ uint8_t            flag_per_candidate[W];

	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			alive_per_candidate[cb + j] = ((my_candidate_base + j) < candidate_count) ? 1u : 0u;
			rep_per_candidate[cb + j]   = cb + j;
		}
	}
	__syncthreads();

	uint32_t value[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		value[j] = alive_per_candidate[cb + j]
			? initialize_working_word(key, data_start + my_candidate_base + j, lane, expansion_values)
			: 0u;
	}

	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		const bool dedup_this = ((schedule_index % DEDUP_PERIOD) == 0u);
		if (dedup_this)
		{
			for (uint32_t s = threadIdx.x; s < NSLOTS; s += blockDim.x) slot[s] = 0ull;
			__syncthreads();
		}

		bool alive[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) alive[j] = alive_per_candidate[cb + j] != 0u;

		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

		uint32_t ro[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) ro[j] = 0u;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));

		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const bool high = (nsel & 0x8000u) != 0u;
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				const uint32_t w = __shfl_sync(0xFFFFFFFFu, value[j], sl);
				uint8_t cbj = static_cast<uint8_t>((w >> ss) & 0xFFu);
				if (high) cbj >>= 4;
				const uint8_t aid = static_cast<uint8_t>((cbj >> 1) & 0x07u);
				if (alive[j]) value[j] = run_alg_offset(value[j], lane, aid, schedule_index, &ro[j], regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			}
			nsel = static_cast<uint16_t>(nsel << 1);
		}

		if (dedup_this)
		{
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				if (!alive[j]) continue;
				uint32_t h_lo, h_hi;
				warp_hash_state(value[j], lane, &h_lo, &h_hi);
				const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
				const unsigned long long fp_top56 = fp_raw & FP_MASK;
				const unsigned long long fp_safe = (fp_top56 == 0ull) ? 0x100ull : fp_top56;
				if (lane == 0u)
				{
					const uint32_t my_idx = cb + j;
					const unsigned long long my_packed = fp_safe | (unsigned long long)(my_idx & 0xFFu);
					uint32_t idx = static_cast<uint32_t>(fp_safe >> 8) & slot_mask;
					for (uint32_t probe = 0u; probe < NSLOTS; probe++)
					{
						const unsigned long long prev = atomicCAS(&slot[idx], 0ull, my_packed);
						if (prev == 0ull) break;
						if ((prev & FP_MASK) == fp_safe)
						{
							rep_per_candidate[my_idx] = static_cast<uint32_t>(prev & 0xFFull);
							alive_per_candidate[my_idx] = 0u;
							break;
						}
						idx = (idx + 1u) & slot_mask;
					}
				}
			}
			__syncthreads();
		}
	}

	uint8_t f[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++) f[j] = screen_candidate(value[j], lane, carnival_data);

	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
			if (alive_per_candidate[cb + j]) flag_per_candidate[cb + j] = f[j];
	}
	__syncthreads();

	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			const uint32_t c_local = cb + j;
			const uint32_t c_global = my_candidate_base + j;
			if (c_global >= candidate_count) continue;
			uint32_t cur = c_local;
			#pragma unroll 1
			for (uint32_t hop = 0u; hop < W; hop++)
			{
				const uint32_t r = rep_per_candidate[cur];
				if (r == cur || r >= W) break;
				cur = r;
			}
			result_data[c_global] = flag_per_candidate[cur];
		}

		if (survivor_count != nullptr)
		{
			uint32_t local = 0u;
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
				if ((my_candidate_base + j) < candidate_count && alive_per_candidate[cb + j]) local++;
			if (local) atomicAdd(survivor_count, static_cast<unsigned long long>(local));
		}
	}
}

#define TM_OFFSET_ILP_KERNEL(NAME, WARPS, ILP, NSLOTS, K)                                          \
extern "C" __global__ __launch_bounds__((WARPS)*32u, 6) void NAME(                                 \
	uint8_t* result_data, const uint8_t* regular_rng_values, const uint8_t* alg0_values,           \
	const uint8_t* alg6_values, const uint32_t* alg2_values, const uint32_t* alg5_values,          \
	const uint8_t* expansion_values, const uint8_t* schedule_data, const uint8_t* carnival_data,   \
	unsigned long long* survivor_count, uint32_t key, uint32_t data_start, uint32_t candidate_count)\
{                                                                                                  \
	tm_checksum_screen_dedup_offset_ilp_sched_impl<27u, (WARPS), (ILP), (NSLOTS), (K)>(            \
		result_data, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values,        \
		expansion_values, schedule_data, carnival_data, survivor_count, key, data_start, candidate_count); \
}

// 256t (8 warps): W = 8*ILP.  128t (4 warps): W = 4*ILP.
TM_OFFSET_ILP_KERNEL(tm_checksum_screen_dedup_offset_ilp4w8_k4_cuda, 8u, 4u,  64u, 4u)  // W=32 control
TM_OFFSET_ILP_KERNEL(tm_checksum_screen_dedup_offset_ilp6w8_k4_cuda, 8u, 6u, 128u, 4u)  // W=48
TM_OFFSET_ILP_KERNEL(tm_checksum_screen_dedup_offset_ilp6w8_k1_cuda, 8u, 6u, 128u, 1u)  // W=48 every-boundary
TM_OFFSET_ILP_KERNEL(tm_checksum_screen_dedup_offset_ilp6w4_k4_cuda, 4u, 6u,  64u, 4u)  // W=24 (matches screen 128t/ILP6)
TM_OFFSET_ILP_KERNEL(tm_checksum_screen_dedup_offset_ilp8w8_k4_cuda, 8u, 8u, 128u, 4u)  // W=64
TM_OFFSET_ILP_KERNEL(tm_checksum_screen_dedup_offset_ilp5w8_k4_cuda, 8u, 5u, 128u, 4u)  // W=40
TM_OFFSET_ILP_KERNEL(tm_checksum_screen_dedup_offset_ilp10w8_k4_cuda, 8u, 10u, 256u, 4u) // W=80
TM_OFFSET_ILP_KERNEL(tm_checksum_screen_dedup_offset_ilp12w8_k4_cuda, 8u, 12u, 256u, 4u) // W=96
TM_OFFSET_ILP_KERNEL(tm_checksum_screen_dedup_offset_ilp8w8_k2_cuda, 8u, 8u, 128u, 2u)   // W=64 K=2
TM_OFFSET_ILP_KERNEL(tm_checksum_screen_dedup_offset_ilp8w8_k3_cuda, 8u, 8u, 128u, 3u)   // W=64 K=3
TM_OFFSET_ILP_KERNEL(tm_checksum_screen_dedup_offset_ilp8w8_k6_cuda, 8u, 8u, 128u, 6u)   // W=64 K=6
TM_OFFSET_ILP_KERNEL(tm_checksum_screen_dedup_offset_ilp8w16_k4_cuda, 16u, 8u, 256u, 4u) // W=128, 512t

// ════════════════════════════════════════════════════════════════════════════
// EFFICIENT MULTI-MERGE DESIGN (2026-06-03): this compaction is ALREADY the ideal
// architecture — state-resident (continues from state[], no re-derive), on-device
// dedup (shared slot[]), offset-stream tables (full canonical-equiv collapse). Its
// ONLY limitation vs the wide-merge: the span-end dedup is WITHIN-BLOCK (slot[NSLOTS],
// W<=96), so at span 0 (map 1) where the 3x collapse is GLOBAL across the window,
// almost nothing collides within a 96-cand block → it catches ~none of the front-
// loaded collapse (this is the 1.8x-vs-11x gap). FIX = give span 0 a GLOBAL VRAM
// dedup table (like tm_wide_merge_hash_collapse), keep within-block for spans 1+
// (collapse there is marginal). Reuses the tuned state-resident downstream; avoids
// re-derive + host round-trip by construction. The naive re-derive multi-merge was
// the wrong impl; this is the right one. Implementation = host orchestration swap of
// the span-0 dedup.
//
// On-GPU VRAM survivor-compaction architecture (2026-05-29).
//
// Multi-pass: run a map SPAN on the live frontier (state in VRAM), dedup within
// each block at the span end, then host-side relaunch compaction packs survivors
// so the NEXT span's grid is smaller — recovering the occupancy that within-block
// dedup alone leaves idle. Correctness is decoupled from compaction effectiveness
// (an uncaught duplicate just runs as its own survivor; rep_global union-find +
// final resolve still produce the exact screen flag for every original candidate).
//
// state[N*32]: lane L holds word L of a candidate's 1024-bit state (128 B/cand).
// rep_global[N]: union-find parent in ORIGINAL-index space (init identity).
// live_idx[M]: original indices of the current frontier (identity for span 0).
// alive_out[M]: 1 if the processed slot survived this span (for compaction).
__device__ __forceinline__ void tm_merge_multiplicity(uint32_t* mult, uint32_t loser, uint32_t winner)
{
	if (!mult || loser == winner || winner >= 0xFE000000u) return;
	atomicAdd(&mult[winner], mult[loser]);
}

template<uint32_t WARPS, uint32_t ILP, uint32_t NSLOTS>
__device__ __forceinline__ void run_span_dedup_impl(
	const uint32_t* live_idx, uint32_t M,
	uint32_t* state, uint8_t* alive_out, uint32_t* rep_global, uint32_t* mult,
	uint32_t m0, uint32_t m1, int first_span,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start,
	const uint8_t* carnival_data, uint8_t* flag_out, int mode)  // mode: 0=writeback,1=dedup,2=screen+flag
{
	constexpr uint32_t W = WARPS * ILP;
	constexpr unsigned long long FP_MASK = 0xFFFFFFFFFFFFFF00ull;
	const uint32_t slot_mask = NSLOTS - 1u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t cb = warp_index * ILP;
	const uint32_t block_base = blockIdx.x * W;

	__shared__ unsigned long long slot[NSLOTS];
	__shared__ uint32_t rep_local[W];
	__shared__ uint8_t  alive_local[W];
	__shared__ uint32_t orig_of[W];

	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			const uint32_t s = block_base + cb + j;
			const bool active = s < M;
			orig_of[cb + j]     = active ? (first_span ? s : live_idx[s]) : 0xFFFFFFFFu;
			alive_local[cb + j] = active ? 1u : 0u;
			rep_local[cb + j]   = cb + j;
		}
	}
	__syncthreads();

	uint32_t value[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t orig = orig_of[cb + j];
		value[j] = alive_local[cb + j]
			? (first_span ? initialize_working_word(key, data_start + orig, lane, expansion_values)
			              : state[(size_t)orig * 32u + lane])
			: 0u;
	}

	for (uint32_t schedule_index = m0; schedule_index < m1; schedule_index++)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
		uint32_t ro[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) ro[j] = 0u;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));
		// preids: precompute all ILP algorithm-ids (shfl+extract) THEN run them, so
		// the scheduler hides the shfl latency behind independent run_alg work — the
		// same trick that made the production ilp6_preids screen the winner. The per-
		// step `alive` check is dropped: every candidate in a span's frontier is alive
		// throughout the map loop (dead ones were removed by the previous compaction;
		// dedup only marks dead at the span end). Invalid last-block slots compute
		// garbage that the writeback/dedup guards ignore.
		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const bool high = (nsel & 0x8000u) != 0u;
			uint8_t aid[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				const uint32_t w = __shfl_sync(0xFFFFFFFFu, value[j], sl);
				uint8_t cbj = static_cast<uint8_t>((w >> ss) & 0xFFu);
				if (high) cbj >>= 4;
				aid[j] = static_cast<uint8_t>((cbj >> 1) & 0x07u);
			}
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
				value[j] = run_alg_offset_sw(value[j], lane, aid[j], schedule_index, &ro[j], regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			nsel = static_cast<uint16_t>(nsel << 1);
		}
	}

	// within-block dedup at span end. Skipped on the final span (do_dedup==0):
	// the hash after the last map saves no map-work (none follow) — the CPU's
	// "after MAP27, results already materialized" wasteful stage. When skipped,
	// all valid slots stay alive and are screened directly (mode 2) or by final_flag.
	if (mode == 1)
	{
		for (uint32_t s = threadIdx.x; s < NSLOTS; s += blockDim.x) slot[s] = 0ull;
		__syncthreads();
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			if (!alive_local[cb + j]) continue;
			uint32_t h_lo, h_hi;
			warp_hash_state(value[j], lane, &h_lo, &h_hi);
			const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
			const unsigned long long fp_top56 = fp_raw & FP_MASK;
			const unsigned long long fp_safe = (fp_top56 == 0ull) ? 0x100ull : fp_top56;
			if (lane == 0u)
			{
				const uint32_t my_idx = cb + j;
				const unsigned long long my_packed = fp_safe | (unsigned long long)(my_idx & 0xFFu);
				uint32_t idx = static_cast<uint32_t>(fp_safe >> 8) & slot_mask;
				#pragma unroll 2
				for (uint32_t probe = 0u; probe < NSLOTS; probe++)
				{
					const unsigned long long prev = atomicCAS(&slot[idx], 0ull, my_packed);
					if (prev == 0ull) break;
					if ((prev & FP_MASK) == fp_safe) { rep_local[my_idx] = static_cast<uint32_t>(prev & 0xFFull); alive_local[my_idx] = 0u; break; }
					idx = (idx + 1u) & slot_mask;
				}
			}
		}
		__syncthreads();
	}

	if (mode == 2)
	{
		// Fused final span: screen each candidate in-register and write its flag
		// directly — no state writeback, no separate final_flag kernel / state reload.
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			const uint32_t s = block_base + cb + j;
			const uint8_t f = screen_candidate(value[j], lane, carnival_data);
			if (lane == 0u && s < M) flag_out[orig_of[cb + j]] = f;
		}
		return;
	}

	// writeback survivor state; record alive/rep for compaction + resolve
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t s = block_base + cb + j;
		if (s >= M) continue;
		const uint32_t orig = orig_of[cb + j];
		if (alive_local[cb + j])
		{
			state[(size_t)orig * 32u + lane] = value[j];
			if (lane == 0u) alive_out[s] = 1u;
		}
		else if (lane == 0u)
		{
			uint32_t cur = cb + j;
			#pragma unroll 1
			for (uint32_t hop = 0u; hop < W; hop++) { const uint32_t r = rep_local[cur]; if (r == cur || r >= W) break; cur = r; }
			const uint32_t rep = orig_of[cur];
			rep_global[orig] = rep;
			tm_merge_multiplicity(mult, orig, rep);
			alive_out[s] = 0u;
		}
	}
}

#define TM_SPAN_KERNEL(NAME, WARPS, ILP, NSLOTS)                                                   \
extern "C" __global__ __launch_bounds__((WARPS)*32u, 6) void NAME(                                 \
	const uint32_t* live_idx, uint32_t M, uint32_t* state, uint8_t* alive_out, uint32_t* rep_global,\
	uint32_t* mult,                                                                                 \
	uint32_t m0, uint32_t m1, int first_span,                                                       \
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,      \
	const uint32_t* alg2_values, const uint32_t* alg5_values,                                       \
	const uint8_t* expansion_values, const uint8_t* schedule_data,                                  \
	uint32_t key, uint32_t data_start,                                                              \
	const uint8_t* carnival_data, uint8_t* flag_out, int mode)                                      \
{                                                                                                  \
	run_span_dedup_impl<(WARPS), (ILP), (NSLOTS)>(live_idx, M, state, alive_out, rep_global, mult,  \
		m0, m1,                                                                                    \
		first_span, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values,         \
		expansion_values, schedule_data, key, data_start, carnival_data, flag_out, mode);           \
}

// 128t (4 warps) and 256t (8 warps) at a spread of ILP; W = WARPS*ILP, NSLOTS≈2W.
TM_SPAN_KERNEL(run_span_dedup_w4i4_cuda,  4u,  4u,  64u)  // W=16
TM_SPAN_KERNEL(run_span_dedup_w4i6_cuda,  4u,  6u,  64u)  // W=24
TM_SPAN_KERNEL(run_span_dedup_w4i8_cuda,  4u,  8u,  64u)  // W=32
TM_SPAN_KERNEL(run_span_dedup_w4i10_cuda, 4u, 10u, 128u)  // W=40
TM_SPAN_KERNEL(run_span_dedup_w4i12_cuda, 4u, 12u, 128u)  // W=48
TM_SPAN_KERNEL(run_span_dedup_w8i4_cuda,  8u,  4u,  64u)  // W=32
TM_SPAN_KERNEL(run_span_dedup_w8i5_cuda,  8u,  5u, 128u)  // W=40
TM_SPAN_KERNEL(run_span_dedup_w8i6_cuda,  8u,  6u, 128u)  // W=48
TM_SPAN_KERNEL(run_span_dedup_w8i8_cuda,  8u,  8u, 128u)  // W=64 (prior default)
TM_SPAN_KERNEL(run_span_dedup_w8i10_cuda, 8u, 10u, 256u)  // W=80
TM_SPAN_KERNEL(run_span_dedup_w8i12_cuda, 8u, 12u, 256u)  // W=96
TM_SPAN_KERNEL(run_span_dedup_w16i6_cuda, 16u, 6u, 256u)  // W=96, 512t
// Back-compat alias for the prior hard-coded name.
TM_SPAN_KERNEL(run_span_dedup_ilp8w8_cuda, 8u, 8u, 128u)

// ── GLOBAL-table span-0 dedup (2026-06-03) — surgical multi-merge fix ──────────
// Identical map loop + state writeback to run_span_dedup_impl, but the span-end
// dedup uses a GLOBAL VRAM open-addressing table (like tm_wide_merge_hash_collapse)
// instead of __shared__ slot[NSLOTS]. This catches the front-loaded GLOBAL collapse
// at map 1 (3x at the 3-byte window) that the within-block table (W<=96/block) misses
// — the diagnosed 1.8x-vs-11x gap. Used on SPAN 0 ONLY; spans 1+ keep within-block
// (collapse there is marginal and shared-mem is cheaper). first_span is always 1
// here (span 0 = identity indices), mode is always 1 (dedup; span 0 is never final).
//
//   table_fp[slots]  : 0 = empty, else claimed 64-bit fingerprint (host pre-zeroed)
//   table_rep[slots] : 0xFFFFFFFF until owner publishes its ORIGINAL index (spin sentinel)
//   logm             : log2(slots); slots is a power of two
// Winner of a slot stays a union-find root (rep_global[orig] left at 0xFFFFFFFF) and
// writes its state; a duplicate marks itself dead, sets rep_global[orig]=winner_orig,
// and does NOT write state (it is dropped from the next frontier by compaction).
template<uint32_t WARPS, uint32_t ILP>
__device__ __forceinline__ void run_span_dedup_global_impl(
	uint32_t M, uint32_t* state, uint8_t* alive_out, uint32_t* rep_global, uint32_t* mult,
	uint32_t m0, uint32_t m1,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start,
	uint64_t* table_fp, volatile uint32_t* table_rep, uint32_t logm)
{
	constexpr uint32_t W = WARPS * ILP;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t cb = warp_index * ILP;
	const uint32_t block_base = blockIdx.x * W;
	const uint32_t tmask = (1u << logm) - 1u;

	// orig == slot (first_span identity). Inactive slots compute garbage that the
	// guarded writeback/insert ignore.
	uint32_t orig[ILP];
	bool active[ILP];
	uint32_t value[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		orig[j]   = block_base + cb + j;
		active[j] = orig[j] < M;
		value[j]  = active[j] ? initialize_working_word(key, data_start + orig[j], lane, expansion_values) : 0u;
	}

	for (uint32_t schedule_index = m0; schedule_index < m1; schedule_index++)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
		uint32_t ro[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) ro[j] = 0u;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));
		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const bool high = (nsel & 0x8000u) != 0u;
			uint8_t aid[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				const uint32_t w = __shfl_sync(0xFFFFFFFFu, value[j], sl);
				uint8_t cbj = static_cast<uint8_t>((w >> ss) & 0xFFu);
				if (high) cbj >>= 4;
				aid[j] = static_cast<uint8_t>((cbj >> 1) & 0x07u);
			}
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
				value[j] = run_alg_offset_sw(value[j], lane, aid[j], schedule_index, &ro[j], regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			nsel = static_cast<uint16_t>(nsel << 1);
		}
	}

	// Global open-addressing dedup over the whole window's frontier.
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		uint32_t h_lo, h_hi;
		warp_hash_state(value[j], lane, &h_lo, &h_hi);          // all 32 lanes
		bool alive = active[j];
		if (active[j] && lane == 0u)
		{
			uint64_t fp = ((uint64_t)h_hi << 32) | (uint64_t)h_lo;
			if (fp == 0ull) fp = 1ull;
			uint32_t slot = (uint32_t)((fp * 0x9E3779B97F4A7C15ull) >> (64 - logm));
			// Bounded linear probe. At the default LF 0.5 a slot is found in a few
			// probes; the cap only bites when the window-cap overfills the table, where
			// a candidate that exhausts PROBE_CAP simply stays its own survivor (an
			// un-merged dup runs separately — correctness-safe, dedup degrades). Without
			// the cap an overfilled table probes O(slots) per insert and never returns.
			constexpr uint32_t PROBE_CAP = 64u;
			const uint32_t cap = (tmask < PROBE_CAP) ? (tmask + 1u) : PROBE_CAP;
			for (uint32_t probe = 0u; probe < cap; ++probe)
			{
				const uint32_t s = (slot + probe) & tmask;
				const uint64_t old = atomicCAS(reinterpret_cast<unsigned long long*>(&table_fp[s]),
					0ull, (unsigned long long)fp);
				if (old == 0ull)                       // won the slot → run representative
				{
					table_rep[s] = orig[j];
					__threadfence();
					break;
				}
				if (old == fp)                         // duplicate → adopt the owner's orig index
				{
					uint32_t r;
					do { r = table_rep[s]; } while (r == 0xFFFFFFFFu);
					rep_global[orig[j]] = r;
					tm_merge_multiplicity(mult, orig[j], r);
					alive = false;
					break;
				}
			}
		}
		alive = __shfl_sync(0xFFFFFFFFu, alive, 0);             // broadcast lane0's verdict
		if (!active[j]) continue;
		if (alive)
		{
			state[(size_t)orig[j] * 32u + lane] = value[j];
			if (lane == 0u) alive_out[orig[j]] = 1u;
		}
		else if (lane == 0u)
		{
			alive_out[orig[j]] = 0u;
		}
	}
}

#define TM_SPAN_GLOBAL_KERNEL(NAME, WARPS, ILP)                                                    \
extern "C" __global__ __launch_bounds__((WARPS)*32u, 6) void NAME(                                 \
	uint32_t M, uint32_t* state, uint8_t* alive_out, uint32_t* rep_global,                          \
	uint32_t* mult,                                                                                 \
	uint32_t m0, uint32_t m1,                                                                       \
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,      \
	const uint32_t* alg2_values, const uint32_t* alg5_values,                                       \
	const uint8_t* expansion_values, const uint8_t* schedule_data,                                  \
	uint32_t key, uint32_t data_start,                                                              \
	uint64_t* table_fp, volatile uint32_t* table_rep, uint32_t logm)                                \
{                                                                                                  \
	run_span_dedup_global_impl<(WARPS), (ILP)>(M, state, alive_out, rep_global, mult, m0, m1,       \
		regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values,                     \
		expansion_values, schedule_data, key, data_start, table_fp, table_rep, logm);               \
}

TM_SPAN_GLOBAL_KERNEL(run_span_dedup_global_w4i4_cuda,  4u,  4u)
TM_SPAN_GLOBAL_KERNEL(run_span_dedup_global_w4i6_cuda,  4u,  6u)
TM_SPAN_GLOBAL_KERNEL(run_span_dedup_global_w4i8_cuda,  4u,  8u)
TM_SPAN_GLOBAL_KERNEL(run_span_dedup_global_w4i10_cuda, 4u, 10u)
TM_SPAN_GLOBAL_KERNEL(run_span_dedup_global_w4i12_cuda, 4u, 12u)
TM_SPAN_GLOBAL_KERNEL(run_span_dedup_global_w8i4_cuda,  8u,  4u)
TM_SPAN_GLOBAL_KERNEL(run_span_dedup_global_w8i5_cuda,  8u,  5u)
TM_SPAN_GLOBAL_KERNEL(run_span_dedup_global_w8i6_cuda,  8u,  6u)
TM_SPAN_GLOBAL_KERNEL(run_span_dedup_global_w8i8_cuda,  8u,  8u)
TM_SPAN_GLOBAL_KERNEL(run_span_dedup_global_w8i10_cuda, 8u, 10u)
TM_SPAN_GLOBAL_KERNEL(run_span_dedup_global_w8i12_cuda, 8u, 12u)
TM_SPAN_GLOBAL_KERNEL(run_span_dedup_global_w16i6_cuda, 16u, 6u)

// ── Pool B for the PRODUCER deep stream: persistent fixed-capacity cap probe ─────────
// The offset-carry producer streams the window in CHUNKS (its streaming unit). The deep
// span below already does within-BLOCK dedup; a cap that PERSISTS across chunks (never
// cleared) recaptures the cross-block + cross-chunk duplicate deep-states that the
// per-chunk count otherwise double-counts. Epoch = boundary DEPTH (m1) so states at
// different depths occupy separate logical sub-tables (same scheme as the whole-tile
// cross-tile cap). 64a layout: one 64-bit CAS claims word0 = (epoch16<<48)|fp48; word1
// carries the epoch so the active-epoch occupancy counter sees a published slot.
// Returns true if fp was already resident at this depth → caller DROPS the rep; false on
// fresh insert OR bucket-full → caller KEEPS (bounded over-keep). Benign-race, FN-safe:
// only an exact fp48 match at the current epoch ever drops (fp48 collision risk 2^-48).
__device__ __forceinline__ bool cap_probe_drop_64a(
	unsigned long long fp, uint32_t epoch,
	unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways)
{
	const uint32_t bucket = (uint32_t)((fp * 0x9E3779B97F4A7C15ull) >> (64 - cap_bits));
	const unsigned long long base = (unsigned long long)bucket * cap_ways;
	const uint32_t e16 = epoch & 0xFFFFu;
	unsigned long long fp48 = fp & 0xFFFFFFFFFFFFull;
	if (fp48 == 0ull) fp48 = 1ull;
	const unsigned long long w0_mine = ((unsigned long long)e16 << 48) | fp48;
	for (uint32_t w = 0u; w < cap_ways; )
	{
		volatile unsigned long long* s0 = (volatile unsigned long long*)(cap_table + (base + w) * 2ull);
		const unsigned long long cur0 = *s0;
		const uint32_t e0 = (uint32_t)(cur0 >> 48);
		if (e0 == e16)
		{
			if (cur0 == w0_mine) return true;          // dup at this depth → drop
			w++; continue;                              // foreign fp48, same depth → next way
		}
		const unsigned long long old = atomicCAS((unsigned long long*)s0, cur0, w0_mine);
		if (old == cur0) { *(s0 + 1) = ((unsigned long long)epoch << 32); __threadfence(); return false; }  // won → keep
		// lost the race → re-read the SAME way (do not advance w)
	}
	return false;   // bucket full of foreign current-epoch fps → leave un-merged → keep
}

__device__ __forceinline__ uint32_t frontier_trajsketch_atomic_old(
	uint32_t* sketch, uint32_t sk_bits, uint32_t key24)
{
	const uint32_t shift = 32u - sk_bits;
	const uint32_t cells = 1u << sk_bits;
	const uint32_t i0 = (key24 * 0x9E3779B1u) >> shift;
	const uint32_t i1 = ((key24 * 0x85EBCA77u) ^ 0xABCD1234u) >> shift;
	const uint32_t i2 = ((key24 * 0xC2B2AE3Du) ^ 0x12345678u) >> shift;
	const uint32_t i3 = ((key24 * 0x27D4EB2Fu) ^ 0xDEADBEEFu) >> shift;
	const uint32_t a = atomicAdd(sketch + i0, 1u);
	const uint32_t b = atomicAdd(sketch + cells + i1, 1u);
	const uint32_t c = atomicAdd(sketch + cells * 2u + i2, 1u);
	const uint32_t d = atomicAdd(sketch + cells * 3u + i3, 1u);
	const uint32_t m0 = a < b ? a : b;
	const uint32_t m1 = c < d ? c : d;
	return m0 < m1 ? m0 : m1;
}

__device__ __forceinline__ bool tm_mult_route(const uint32_t* multiplicity, uint32_t idx, uint32_t mult_tau)
{
	return multiplicity != nullptr && mult_tau > 0u && multiplicity[idx] >= mult_tau;
}

// Local-frontier compaction after MAP1 frontier emission. `rep_data[local]` is the
// absolute 32-bit data value for a MAP1 representative. State/liveness arrays are
// sized to the representative frontier, not the original window. This deliberately
// uses the same within-block span dedup as production compaction so we can measure
// how much the existing deeper compaction stages buy after a global MAP1 collapse.
template<uint32_t WARPS, uint32_t ILP, uint32_t NSLOTS,
	bool WITH_SHED = false, bool WITH_TRAJ = false, bool WITH_SHED_GATE = false, bool WITH_FP_GATE = false>
__device__ __forceinline__ void run_frontier_span_local_impl(
	const uint32_t* live_idx, uint32_t M, const uint32_t* rep_data,
	uint32_t* state, uint8_t* alive_out,
	uint32_t m0, uint32_t m1, int first_span,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, const uint8_t* carnival_data, uint8_t* flag_out, int mode,
	uint32_t* multiplicity,
	unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways,
	uint32_t shed_tau = 0u,
	uint32_t shed_gate_tau = 0u,
	uint32_t fp_gate_log = 0u,
	uint32_t* traj_sketch = nullptr, uint32_t traj_bits = 0u,
	uint32_t traj_dens_tau = 0u, uint32_t traj_alg0_tau = 0u, uint32_t mult_tau = 0u,
	unsigned long long* route_stats = nullptr)
{
	constexpr uint32_t W = WARPS * ILP;
	constexpr unsigned long long FP_MASK = 0xFFFFFFFFFFFFFF00ull;
	const uint32_t slot_mask = NSLOTS - 1u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t cb = warp_index * ILP;
	const uint32_t block_base = blockIdx.x * W;

	__shared__ unsigned long long slot[NSLOTS];
	__shared__ uint32_t rep_local[W];
	__shared__ uint8_t  alive_local[W];
	__shared__ uint32_t local_of[W];

	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			const uint32_t s = block_base + cb + j;
			const bool active = s < M;
			local_of[cb + j]  = active ? (first_span ? s : (live_idx ? live_idx[s] : s)) : 0xFFFFFFFFu;
			alive_local[cb + j] = active ? 1u : 0u;
			rep_local[cb + j] = cb + j;
		}
	}
	__syncthreads();

	uint32_t value[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t local = local_of[cb + j];
		value[j] = alive_local[cb + j]
			? (first_span ? initialize_working_word(key, rep_data[local], lane, expansion_values)
			              : state[(size_t)local * 32u + lane])
			: 0u;
	}

	uint32_t shed[ILP];
	if constexpr (WITH_SHED || WITH_SHED_GATE)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) shed[j] = 0u;
	}
	uint32_t optail[ILP], maxA0[ILP];
	if constexpr (WITH_TRAJ)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) { optail[j] = 0u; maxA0[j] = 0u; }
	}

	for (uint32_t schedule_index = m0; schedule_index < m1; schedule_index++)
	{
		const bool final_map = (schedule_index + 1u == m1);
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
		uint32_t ro[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) ro[j] = 0u;
		uint32_t mapA0[ILP];
		if constexpr (WITH_TRAJ)
		{
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++) mapA0[j] = 0u;
		}
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));
		for (uint32_t i = 0u; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const bool high = (nsel & 0x8000u) != 0u;
			uint8_t aid[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				const uint32_t w = __shfl_sync(0xFFFFFFFFu, value[j], sl);
				uint8_t cbj = static_cast<uint8_t>((w >> ss) & 0xFFu);
				if (high) cbj >>= 4;
				aid[j] = static_cast<uint8_t>((cbj >> 1) & 0x07u);
			}
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				if constexpr (WITH_SHED || WITH_SHED_GATE)
					if (aid[j] == 0u || aid[j] == 6u) shed[j]++;  // shed-proxy: count info-loss ops
				if constexpr (WITH_TRAJ)
				{
					if (aid[j] == 0u) mapA0[j]++;
					if (final_map) optail[j] = (optail[j] << 3) | aid[j];
				}
				value[j] = run_alg_offset_sw(value[j], lane, aid[j], schedule_index, &ro[j],
					regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			}
			nsel = static_cast<uint16_t>(nsel << 1);
		}
		if constexpr (WITH_TRAJ)
		{
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
				if (mapA0[j] > maxA0[j]) maxA0[j] = mapA0[j];
		}
	}

	if (mode == 1)
	{
		uint32_t stat_routed = 0u, stat_passed = 0u, stat_sticky = 0u, stat_mult = 0u, stat_drops = 0u;
		for (uint32_t s = threadIdx.x; s < NSLOTS; s += blockDim.x) slot[s] = 0ull;
		__syncthreads();
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			if (!alive_local[cb + j]) continue;
			uint32_t h_lo, h_hi;
			warp_hash_state(value[j], lane, &h_lo, &h_hi);
			const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
			const unsigned long long fp_top56 = fp_raw & FP_MASK;
			const unsigned long long fp_safe = (fp_top56 == 0ull) ? 0x100ull : fp_top56;
			if (lane == 0u)
			{
				const uint32_t my_idx = cb + j;
				const unsigned long long my_packed = fp_safe | (unsigned long long)(my_idx & 0xFFu);
				uint32_t idx = static_cast<uint32_t>(fp_safe >> 8) & slot_mask;
				#pragma unroll 2
				for (uint32_t probe = 0u; probe < NSLOTS; probe++)
				{
					const unsigned long long prev = atomicCAS(&slot[idx], 0ull, my_packed);
					if (prev == 0ull) break;
					if ((prev & FP_MASK) == fp_safe)
					{
						const uint32_t rep_idx = static_cast<uint32_t>(prev & 0xFFull);
						rep_local[my_idx] = rep_idx;
						if (multiplicity != nullptr)
							tm_merge_multiplicity(multiplicity, local_of[my_idx], local_of[rep_idx]);
						alive_local[my_idx] = 0u;
						break;
					}
					idx = (idx + 1u) & slot_mask;
				}
				// Pool B: a rep still alive after block-local dedup is the block's canonical
				// at this depth. Probe the persistent cross-chunk cap (epoch = m1 = boundary
				// depth); a resident match kills it (cross-block / cross-chunk duplicate).
				// Drain cap probes are ungated unless the producer trajgate is active.
				bool route_to_cap = true;
				bool sticky_flag = false;
				bool dens_flag = false;
				bool mult_flag = false;
				if constexpr (WITH_SHED_GATE)
				{
					route_to_cap = shed[j] >= shed_gate_tau;
				}
				if constexpr (WITH_FP_GATE)
				{
					if (fp_gate_log != 0u)
					{
						const uint32_t mask = (1u << fp_gate_log) - 1u;
						route_to_cap = (h_lo & mask) == 0u;
					}
				}
				if constexpr (WITH_TRAJ)
				{
					sticky_flag = (traj_alg0_tau > 0u) && (maxA0[j] >= traj_alg0_tau);
					if (traj_sketch && traj_bits != 0u)
					{
						const uint32_t key24 = optail[j] & 0xFFFFFFu;
						const uint32_t dens = frontier_trajsketch_atomic_old(traj_sketch, traj_bits, key24);
						dens_flag = dens >= traj_dens_tau;
						route_to_cap = sticky_flag || dens_flag;
					}
					else if (traj_alg0_tau > 0u)
					{
						route_to_cap = sticky_flag;
					}
					else
					{
						route_to_cap = false;
					}
					mult_flag = tm_mult_route(multiplicity, local_of[my_idx], mult_tau);
					if (mult_flag)
						route_to_cap = true;
				}
				if (cap_table && alive_local[my_idx])
				{
					if (route_stats)
					{
						if (route_to_cap)
						{
							stat_routed++;
							if (sticky_flag && !dens_flag) stat_sticky++;
							if (mult_flag && !sticky_flag && !dens_flag) stat_mult++;
						}
						else
						{
							stat_passed++;
						}
					}
					if (route_to_cap && cap_probe_drop_64a(fp_safe, m1, cap_table, cap_bits, cap_ways))
					{
						alive_local[my_idx] = 0u;
						if (route_stats) stat_drops++;
					}
				}
			}
		}
		if (route_stats && lane == 0u)
		{
			if (stat_routed) atomicAdd(route_stats, static_cast<unsigned long long>(stat_routed));
			if (stat_passed) atomicAdd(route_stats + 1, static_cast<unsigned long long>(stat_passed));
			if (stat_sticky) atomicAdd(route_stats + 2, static_cast<unsigned long long>(stat_sticky));
			if (stat_mult) atomicAdd(route_stats + 3, static_cast<unsigned long long>(stat_mult));
			if (stat_drops) atomicAdd(route_stats + 4, static_cast<unsigned long long>(stat_drops));
		}
		__syncthreads();
	}

	if (mode == 2)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			const uint32_t s = block_base + cb + j;
			const uint8_t f = screen_candidate(value[j], lane, carnival_data);
			if (lane == 0u && s < M) flag_out[local_of[cb + j]] = f;
		}
		return;
	}

	// WITH_SHED: at MAP1 (mode==0), kill low-shed survivors before they enter the deep drain
	// chain. Shed-proxy routing is valid only for MAP1; deeper spans don't gate on shed.
	if constexpr (WITH_SHED)
	{
		if (mode == 0 && lane == 0u)
		{
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
				if (alive_local[cb + j] && shed[j] < shed_tau)
					alive_local[cb + j] = 0u;
		}
	}

	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t s = block_base + cb + j;
		if (s >= M) continue;
		const uint32_t local = local_of[cb + j];
		if (alive_local[cb + j])
		{
			state[(size_t)local * 32u + lane] = value[j];
			if (lane == 0u) alive_out[s] = 1u;
		}
		else if (lane == 0u)
		{
			alive_out[s] = 0u;
		}
	}
}

#define TM_FRONTIER_SPAN_LOCAL_KERNEL(NAME, WARPS, ILP, NSLOTS)                                    \
extern "C" __global__ __launch_bounds__((WARPS)*32u, 6) void NAME(                                 \
	const uint32_t* live_idx, uint32_t M, const uint32_t* rep_data, uint32_t* state, uint8_t* alive_out,\
	uint32_t m0, uint32_t m1, int first_span,                                                       \
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,      \
	const uint32_t* alg2_values, const uint32_t* alg5_values,                                       \
	const uint8_t* expansion_values, const uint8_t* schedule_data,                                  \
	uint32_t key, const uint8_t* carnival_data, uint8_t* flag_out, int mode, uint32_t* multiplicity)\
{                                                                                                  \
	run_frontier_span_local_impl<(WARPS), (ILP), (NSLOTS)>(live_idx, M, rep_data, state, alive_out, \
		m0, m1, first_span, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, \
		expansion_values, schedule_data, key, carnival_data, flag_out, mode, multiplicity, nullptr, 0u, 0u);      \
}

TM_FRONTIER_SPAN_LOCAL_KERNEL(run_frontier_span_local_w8i10_cuda, 8u, 10u, 256u)

// Pool B variant: same deep span but with a persistent cross-chunk cap probe at mode==1
// (trailing cap_table/cap_bits/cap_ways params). The producer launches this when
// --map1-frontier-drain-cap-bits is set; the cap is allocated + zeroed ONCE and persists
// across all chunks of the window, so it recaptures cross-chunk duplicate deep-states.
#define TM_FRONTIER_SPAN_LOCAL_CAP_KERNEL(NAME, WARPS, ILP, NSLOTS)                                \
extern "C" __global__ __launch_bounds__((WARPS)*32u, 6) void NAME(                                 \
	const uint32_t* live_idx, uint32_t M, const uint32_t* rep_data, uint32_t* state, uint8_t* alive_out,\
	uint32_t m0, uint32_t m1, int first_span,                                                       \
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,      \
	const uint32_t* alg2_values, const uint32_t* alg5_values,                                       \
	const uint8_t* expansion_values, const uint8_t* schedule_data,                                  \
	uint32_t key, const uint8_t* carnival_data, uint8_t* flag_out, int mode, uint32_t* multiplicity,\
	unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways)                            \
{                                                                                                  \
	run_frontier_span_local_impl<(WARPS), (ILP), (NSLOTS)>(live_idx, M, rep_data, state, alive_out, \
		m0, m1, first_span, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values, \
		expansion_values, schedule_data, key, carnival_data, flag_out, mode, multiplicity, cap_table, cap_bits, cap_ways);\
}

TM_FRONTIER_SPAN_LOCAL_CAP_KERNEL(run_frontier_span_local_cap_w8i10_cuda, 8u, 10u, 256u)

// Pool B + shed-routing variant: extends the cap variant with shed-proxy pre-filtering at
// MAP1. During mode==0 (MAP1 span), survivors with shed-count (alg0/alg6 ops) < shed_tau
// are discarded before entering the deep drain chain. At mode==1 drain boundaries the cap
// is probed without shed gating (shed-proxy is only valid at MAP1 length spans).
#define TM_FRONTIER_SPAN_LOCAL_SHED_KERNEL(NAME, WARPS, ILP, NSLOTS)                              \
extern "C" __global__ __launch_bounds__((WARPS)*32u, 6) void NAME(                                \
	const uint32_t* live_idx, uint32_t M, const uint32_t* rep_data, uint32_t* state, uint8_t* alive_out,\
	uint32_t m0, uint32_t m1, int first_span,                                                      \
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,   \
	const uint32_t* alg2_values, const uint32_t* alg5_values,                                     \
	const uint8_t* expansion_values, const uint8_t* schedule_data,                                \
	uint32_t key, const uint8_t* carnival_data, uint8_t* flag_out, int mode, uint32_t* multiplicity,\
	unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways,                           \
	uint32_t shed_tau)                                                                             \
{                                                                                                  \
	run_frontier_span_local_impl<(WARPS), (ILP), (NSLOTS), true>(live_idx, M, rep_data, state,    \
		alive_out, m0, m1, first_span, regular_rng_values, alg0_values, alg6_values,              \
		alg2_values, alg5_values, expansion_values, schedule_data, key, carnival_data,            \
		flag_out, mode, multiplicity, cap_table, cap_bits, cap_ways, shed_tau);                   \
}

TM_FRONTIER_SPAN_LOCAL_SHED_KERNEL(run_frontier_span_local_shed_w8i10_cuda, 8u, 10u, 256u)

// Pool B + shed-gate variant: gates only the persistent cap probe at mode==1.
// Low-shed states are kept without probing Pool B; cap matches are the only drops.
#define TM_FRONTIER_SPAN_LOCAL_SHED_GATE_KERNEL(NAME, WARPS, ILP, NSLOTS)                         \
extern "C" __global__ __launch_bounds__((WARPS)*32u, 6) void NAME(                                \
	const uint32_t* live_idx, uint32_t M, const uint32_t* rep_data, uint32_t* state, uint8_t* alive_out,\
	uint32_t m0, uint32_t m1, int first_span,                                                      \
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,   \
	const uint32_t* alg2_values, const uint32_t* alg5_values,                                     \
	const uint8_t* expansion_values, const uint8_t* schedule_data,                                \
	uint32_t key, const uint8_t* carnival_data, uint8_t* flag_out, int mode, uint32_t* multiplicity,\
	unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways,                           \
	uint32_t shed_gate_tau)                                                                        \
{                                                                                                  \
	run_frontier_span_local_impl<(WARPS), (ILP), (NSLOTS), false, false, true>(live_idx, M, rep_data, state,\
		alive_out, m0, m1, first_span, regular_rng_values, alg0_values, alg6_values,                \
		alg2_values, alg5_values, expansion_values, schedule_data, key, carnival_data,              \
		flag_out, mode, multiplicity, cap_table, cap_bits, cap_ways, 0u, shed_gate_tau);            \
}

TM_FRONTIER_SPAN_LOCAL_SHED_GATE_KERNEL(run_frontier_span_local_shed_gate_w8i10_cuda, 8u, 10u, 256u)

// Pool B + fingerprint-sampling gate: near-zero-overhead single-pass gate used to
// measure the ceiling from reducing cap probes without classifier bookkeeping.
#define TM_FRONTIER_SPAN_LOCAL_FP_GATE_KERNEL(NAME, WARPS, ILP, NSLOTS)                           \
extern "C" __global__ __launch_bounds__((WARPS)*32u, 6) void NAME(                                \
	const uint32_t* live_idx, uint32_t M, const uint32_t* rep_data, uint32_t* state, uint8_t* alive_out,\
	uint32_t m0, uint32_t m1, int first_span,                                                      \
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,   \
	const uint32_t* alg2_values, const uint32_t* alg5_values,                                     \
	const uint8_t* expansion_values, const uint8_t* schedule_data,                                \
	uint32_t key, const uint8_t* carnival_data, uint8_t* flag_out, int mode, uint32_t* multiplicity,\
	unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways,                           \
	uint32_t fp_gate_log)                                                                          \
{                                                                                                  \
	run_frontier_span_local_impl<(WARPS), (ILP), (NSLOTS), false, false, false, true>(live_idx, M, rep_data, state,\
		alive_out, m0, m1, first_span, regular_rng_values, alg0_values, alg6_values,                \
		alg2_values, alg5_values, expansion_values, schedule_data, key, carnival_data,              \
		flag_out, mode, multiplicity, cap_table, cap_bits, cap_ways, 0u, 0u, fp_gate_log);          \
}

TM_FRONTIER_SPAN_LOCAL_FP_GATE_KERNEL(run_frontier_span_local_fp_gate_w8i10_cuda, 8u, 10u, 256u)

// Producer Pool B trajgate experiment: online per-chunk/per-boundary count-min
// density plus sticky alg0 gates the persistent cap probe. Passed states are kept,
// so the failure mode is over-keep, not false drop.
#define TM_FRONTIER_SPAN_LOCAL_TRAJ_KERNEL(NAME, WARPS, ILP, NSLOTS)                              \
extern "C" __global__ __launch_bounds__((WARPS)*32u, 6) void NAME(                                \
	const uint32_t* live_idx, uint32_t M, const uint32_t* rep_data, uint32_t* state, uint8_t* alive_out,\
	uint32_t m0, uint32_t m1, int first_span,                                                      \
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,   \
	const uint32_t* alg2_values, const uint32_t* alg5_values,                                     \
	const uint8_t* expansion_values, const uint8_t* schedule_data,                                \
	uint32_t key, const uint8_t* carnival_data, uint8_t* flag_out, int mode, uint32_t* multiplicity,\
	unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways,                           \
	uint32_t* traj_sketch, uint32_t traj_bits, uint32_t dens_tau, uint32_t alg0_tau, uint32_t mult_tau,\
	unsigned long long* route_stats)                                                                \
{                                                                                                  \
	run_frontier_span_local_impl<(WARPS), (ILP), (NSLOTS), false, true>(live_idx, M, rep_data, state,\
		alive_out, m0, m1, first_span, regular_rng_values, alg0_values, alg6_values,                \
		alg2_values, alg5_values, expansion_values, schedule_data, key, carnival_data,              \
		flag_out, mode, multiplicity, cap_table, cap_bits, cap_ways, 0u, 0u, 0u, traj_sketch, traj_bits, dens_tau, alg0_tau, mult_tau, route_stats);\
}

TM_FRONTIER_SPAN_LOCAL_TRAJ_KERNEL(run_frontier_span_local_traj_w8i10_cuda, 8u, 10u, 256u)

// Atomic-append stream compaction: pack surviving original indices into out_live,
// bump *counter. first_span → orig index = slot index; else live_idx_in[slot].
extern "C" __global__ void compact_survivors_cuda(
	const uint8_t* alive_in, const uint32_t* live_idx_in, uint32_t M,
	uint32_t* out_live, uint32_t* counter, int first_span)
{
	const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= M) return;
	if (alive_in[tid] == 0u) return;
	const uint32_t orig = first_span ? tid : live_idx_in[tid];
	const uint32_t pos = atomicAdd(counter, 1u);
	out_live[pos] = orig;
}

// Block-stable ordered compaction: block-local prefix scan + one atomicAdd per
// block for its base offset, ordered writes. Preserves within-block data order
// (and near-global order at low R), so the next span's state gather reads
// contiguous orig indices → coalesced, killing the long_scoreboard that
// per-candidate atomic-append incurs from scattered live_idx.
extern "C" __global__ __launch_bounds__(256) void compact_survivors_ordered_cuda(
	const uint8_t* alive_in, const uint32_t* live_idx_in, uint32_t M,
	uint32_t* out_live, uint32_t* counter, int first_span)
{
	__shared__ uint32_t s_scan[256];
	__shared__ uint32_t s_base;
	const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t a = (tid < M && alive_in[tid] != 0u) ? 1u : 0u;
	s_scan[threadIdx.x] = a;
	__syncthreads();
	// Hillis-Steele inclusive scan over the 256-thread block.
	for (uint32_t off = 1u; off < blockDim.x; off <<= 1)
	{
		uint32_t v = (threadIdx.x >= off) ? s_scan[threadIdx.x - off] : 0u;
		__syncthreads();
		s_scan[threadIdx.x] += v;
		__syncthreads();
	}
	if (threadIdx.x == blockDim.x - 1u) s_base = atomicAdd(counter, s_scan[threadIdx.x]);
	__syncthreads();
	if (a)
	{
		const uint32_t pos = s_base + s_scan[threadIdx.x] - 1u;  // inclusive→exclusive
		out_live[pos] = first_span ? tid : live_idx_in[tid];
	}
}

// Screen the final survivors (warp-cooperative); lane0 writes flag[orig].
extern "C" __global__ __launch_bounds__(256, 6) void final_flag_survivors_cuda(
	const uint32_t* live_idx, uint32_t M, const uint32_t* state,
	uint8_t* flag, const uint8_t* carnival_data)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
	if (warp >= M) return;
	const uint32_t orig = live_idx[warp];
	const uint32_t value = state[(size_t)orig * 32u + lane];
	const uint8_t f = screen_candidate(value, lane, carnival_data);
	if (lane == 0u) flag[orig] = f;
}

// Resolve every original candidate's flag by following the union-find chain.
extern "C" __global__ void resolve_flags_cuda(
	const uint32_t* rep_global, const uint8_t* flag, uint8_t* result, uint32_t N)
{
	const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c >= N) return;
	// rep_global[x] == 0xFFFFFFFF marks a root (a candidate that never died);
	// lets the host init the whole array with a single cuMemsetD32.
	// A cross-tile drop encodes (0xFE000000 | flag) = the rep's RESOLVED flag from a prior
	// tile (its index is gone) — use it directly instead of chasing a stale index.
	uint32_t root = c;
	#pragma unroll 1
	for (uint32_t hop = 0u; hop < 64u; hop++)
	{
		const uint32_t r = rep_global[root];
		if (r == 0xFFFFFFFFu) break;
		if ((r >> 24) == 0xFEu) { result[c] = (uint8_t)(r & 0xFFu); return; }   // cross-tile resolved flag
		root = r;
	}
	result[c] = flag[root];
}


// ────────────────────────────────────────────────────────────────────────────
// Phase 2.5 dedup-frequency-reduction variant: dedup every K=3 boundaries.
//
// The per-boundary dedup tax (slot init, atomicCAS contention, syncthreads,
// warp_hash_state) was diagnosed as the dominant overhead. ncu showed the
// kernel achieves 72% occupancy and 26% predication waste, vs the screen
// kernel's 81% occupancy and 0% waste.
//
// Doing dedup less often trades convergence speed for less tax. Once a
// candidate is marked dead, it stays dead — so delayed dedup only loses the
// compute-savings for the K-1 boundaries between dedup events. Total
// compute-savings drop by ~(K-1)/total_boundaries; total tax drops by K×.
//
// At K=3 over 27 boundaries: 9 dedup passes vs 27 — ⅔ tax reduction, with
// expected ~7% compute-savings loss (avg detection delay = 1 boundary out
// of 27).
template<uint32_t SCHEDULE_COUNT, uint32_t DEDUP_PERIOD>
__device__ __forceinline__ void tm_checksum_screen_dedup_periodic_impl(
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
	constexpr uint32_t W = 32u;
	constexpr uint32_t NSLOTS = 64u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t my_candidate_base = blockIdx.x * W + warp_index * 4u;
	const uint32_t cb = warp_index * 4u;
	const uint32_t slot_mask = NSLOTS - 1u;

	__shared__ unsigned long long slot_fp[NSLOTS];
	__shared__ volatile uint32_t  slot_rep[NSLOTS];
	__shared__ uint32_t           rep_per_candidate[W];
	__shared__ uint8_t            alive_per_candidate[W];
	__shared__ uint8_t            flag_per_candidate[W];

	if (lane == 0u)
	{
		const bool a0 = (my_candidate_base + 0u) < candidate_count;
		const bool a1 = (my_candidate_base + 1u) < candidate_count;
		const bool a2 = (my_candidate_base + 2u) < candidate_count;
		const bool a3 = (my_candidate_base + 3u) < candidate_count;
		alive_per_candidate[cb + 0] = a0 ? 1u : 0u;
		alive_per_candidate[cb + 1] = a1 ? 1u : 0u;
		alive_per_candidate[cb + 2] = a2 ? 1u : 0u;
		alive_per_candidate[cb + 3] = a3 ? 1u : 0u;
		rep_per_candidate[cb + 0] = cb + 0u;
		rep_per_candidate[cb + 1] = cb + 1u;
		rep_per_candidate[cb + 2] = cb + 2u;
		rep_per_candidate[cb + 3] = cb + 3u;
	}
	__syncthreads();

	const bool a0i = alive_per_candidate[cb + 0] != 0u;
	const bool a1i = alive_per_candidate[cb + 1] != 0u;
	const bool a2i = alive_per_candidate[cb + 2] != 0u;
	const bool a3i = alive_per_candidate[cb + 3] != 0u;

	uint32_t value0 = a0i ? initialize_working_word(key, data_start + my_candidate_base + 0u, lane, expansion_values) : 0u;
	uint32_t value1 = a1i ? initialize_working_word(key, data_start + my_candidate_base + 1u, lane, expansion_values) : 0u;
	uint32_t value2 = a2i ? initialize_working_word(key, data_start + my_candidate_base + 2u, lane, expansion_values) : 0u;
	uint32_t value3 = a3i ? initialize_working_word(key, data_start + my_candidate_base + 3u, lane, expansion_values) : 0u;

	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		const bool dedup_this_boundary = ((schedule_index % DEDUP_PERIOD) == 0u);

		if (dedup_this_boundary)
		{
			if (threadIdx.x < NSLOTS)
			{
				slot_fp[threadIdx.x]  = 0ull;
				slot_rep[threadIdx.x] = 0xFFFFFFFFu;
			}
			__syncthreads();
		}

		const bool alive0 = alive_per_candidate[cb + 0] != 0u;
		const bool alive1 = alive_per_candidate[cb + 1] != 0u;
		const bool alive2 = alive_per_candidate[cb + 2] != 0u;
		const bool alive3 = alive_per_candidate[cb + 3] != 0u;

		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

		uint16_t rs0 = static_cast<uint16_t>(((packed_schedule & 0xFFu) << 8) | ((packed_schedule & 0xFF00u) >> 8));
		uint16_t rs1 = rs0, rs2 = rs0, rs3 = rs0;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));

		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, value0, sl);
			const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, value1, sl);
			const uint32_t w2 = __shfl_sync(0xFFFFFFFFu, value2, sl);
			const uint32_t w3 = __shfl_sync(0xFFFFFFFFu, value3, sl);
			uint8_t cb0 = static_cast<uint8_t>((w0 >> ss) & 0xFFu);
			uint8_t cb1 = static_cast<uint8_t>((w1 >> ss) & 0xFFu);
			uint8_t cb2 = static_cast<uint8_t>((w2 >> ss) & 0xFFu);
			uint8_t cb3 = static_cast<uint8_t>((w3 >> ss) & 0xFFu);
			if ((nsel & 0x8000u) != 0u) { cb0 >>= 4; cb1 >>= 4; cb2 >>= 4; cb3 >>= 4; }
			const uint8_t a0 = static_cast<uint8_t>((cb0 >> 1) & 0x07u);
			const uint8_t a1 = static_cast<uint8_t>((cb1 >> 1) & 0x07u);
			const uint8_t a2 = static_cast<uint8_t>((cb2 >> 1) & 0x07u);
			const uint8_t a3 = static_cast<uint8_t>((cb3 >> 1) & 0x07u);
			if (alive0) value0 = run_alg(value0, lane, a0, &rs0, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive1) value1 = run_alg(value1, lane, a1, &rs1, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive2) value2 = run_alg(value2, lane, a2, &rs2, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive3) value3 = run_alg(value3, lane, a3, &rs3, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nsel = static_cast<uint16_t>(nsel << 1);
		}

		if (dedup_this_boundary)
		{
			#pragma unroll
			for (uint32_t ilp = 0u; ilp < 4u; ilp++)
			{
				uint32_t value;
				bool alive;
				switch (ilp)
				{
					case 0: value = value0; alive = alive0; break;
					case 1: value = value1; alive = alive1; break;
					case 2: value = value2; alive = alive2; break;
					default: value = value3; alive = alive3; break;
				}
				if (!alive) continue;
				uint32_t h_lo, h_hi;
				warp_hash_state(value, lane, &h_lo, &h_hi);
				const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
				const unsigned long long fp = (fp_raw == 0ull) ? 1ull : fp_raw;

				if (lane == 0u)
				{
					const uint32_t my_idx = cb + ilp;
					uint32_t idx = static_cast<uint32_t>(fp) & slot_mask;
					for (uint32_t probe = 0u; probe < NSLOTS; probe++)
					{
						const unsigned long long prev = atomicCAS(&slot_fp[idx], 0ull, fp);
						if (prev == 0ull)
						{
							slot_rep[idx] = my_idx;
							__threadfence_block();
							break;
						}
						if (prev == fp)
						{
							uint32_t r;
							do { r = slot_rep[idx]; } while (r == 0xFFFFFFFFu);
							rep_per_candidate[my_idx] = r;
							alive_per_candidate[my_idx] = 0u;
							break;
						}
						idx = (idx + 1u) & slot_mask;
					}
				}
			}
			__syncthreads();
		}
	}

	// Post-schedule: same as eager-dedup variant.
	const uint8_t f0 = screen_candidate(value0, lane, carnival_data);
	const uint8_t f1 = screen_candidate(value1, lane, carnival_data);
	const uint8_t f2 = screen_candidate(value2, lane, carnival_data);
	const uint8_t f3 = screen_candidate(value3, lane, carnival_data);

	if (lane == 0u)
	{
		if (alive_per_candidate[cb + 0]) flag_per_candidate[cb + 0] = f0;
		if (alive_per_candidate[cb + 1]) flag_per_candidate[cb + 1] = f1;
		if (alive_per_candidate[cb + 2]) flag_per_candidate[cb + 2] = f2;
		if (alive_per_candidate[cb + 3]) flag_per_candidate[cb + 3] = f3;
	}
	__syncthreads();

	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			const uint32_t c_local = cb + ilp;
			const uint32_t c_global = my_candidate_base + ilp;
			if (c_global >= candidate_count) continue;
			uint32_t cur = c_local;
			#pragma unroll 1
			for (uint32_t hop = 0u; hop < W; hop++)
			{
				const uint32_t r = rep_per_candidate[cur];
				if (r == cur || r >= W) break;
				cur = r;
			}
			result_data[c_global] = flag_per_candidate[cur];
		}
	}
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_checksum_screen_dedup_w32_k2_cuda(
	uint8_t* result_data, const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data, const uint8_t* carnival_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_checksum_screen_dedup_periodic_impl<27u, 2u>(result_data, regular_rng_values, alg0_values, alg6_values,
		rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values,
		schedule_data, carnival_data, key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_checksum_screen_dedup_w32_k3_cuda(
	uint8_t* result_data, const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data, const uint8_t* carnival_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_checksum_screen_dedup_periodic_impl<27u, 3u>(result_data, regular_rng_values, alg0_values, alg6_values,
		rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values,
		schedule_data, carnival_data, key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(256, 4) void tm_checksum_screen_dedup_w32_k6_cuda(
	uint8_t* result_data, const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data, const uint8_t* carnival_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_checksum_screen_dedup_periodic_impl<27u, 6u>(result_data, regular_rng_values, alg0_values, alg6_values,
		rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values,
		schedule_data, carnival_data, key, data_start, candidate_count);
}

// ────────────────────────────────────────────────────────────────────────────
// Diagnostic: screen kernel at 256 threads/block (vs production 128).
//
// Reuses tm_checksum_screen_impl. Lets us isolate the cost of 256-thread
// blocks from the cost of the dedup work itself — if this kernel matches
// the 128-thread screen kernel's throughput, the dedup tax is the
// bottleneck. If it's slower, the block size itself costs.
extern "C" __global__ __launch_bounds__(256, 4) void tm_checksum_screen_cuda_256thr(
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

// ────────────────────────────────────────────────────────────────────────────
// Phase 2.7: packed-slot variant (eliminates spin-wait + separate rep read).
//
// Old design: slot_fp[NSLOTS] (uint64) + slot_rep[NSLOTS] (uint32). Owner does
// atomicCAS on fp, then non-atomic write to rep. Followers atomic-CAS fp,
// then spin-wait on rep until owner's write is visible. Adds:
//   - 2 shared-mem arrays (cache pressure)
//   - __threadfence_block on owner
//   - volatile spin-wait on followers (per-warp serialization risk)
//
// New design: single slot[NSLOTS] (uint64) = (fp_top56 << 8) | rep_low8.
// Single atomicCAS atomically claims fp+rep. Followers read returned value
// from atomicCAS — no separate read, no spin, no fence needed.
//
// Tradeoff: fp truncated to 56 bits. False-positive rate at our scales
// (~11M inserts/kernel, ~50 inserts/block): ~1e-9 collisions/kernel. Safe.
// W=32 fits in 8-bit rep field with headroom.
template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_checksum_screen_dedup_w32_packed_impl(
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
	constexpr uint32_t W = 32u;
	constexpr uint32_t NSLOTS = 64u;
	constexpr unsigned long long FP_MASK = 0xFFFFFFFFFFFFFF00ull;  // top 56 bits
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t my_candidate_base = blockIdx.x * W + warp_index * 4u;
	const uint32_t cb = warp_index * 4u;
	const uint32_t slot_mask = NSLOTS - 1u;

	__shared__ unsigned long long slot[NSLOTS];
	__shared__ uint32_t           rep_per_candidate[W];
	__shared__ uint8_t            alive_per_candidate[W];
	__shared__ uint8_t            flag_per_candidate[W];

	if (lane == 0u)
	{
		const bool a0 = (my_candidate_base + 0u) < candidate_count;
		const bool a1 = (my_candidate_base + 1u) < candidate_count;
		const bool a2 = (my_candidate_base + 2u) < candidate_count;
		const bool a3 = (my_candidate_base + 3u) < candidate_count;
		alive_per_candidate[cb + 0] = a0 ? 1u : 0u;
		alive_per_candidate[cb + 1] = a1 ? 1u : 0u;
		alive_per_candidate[cb + 2] = a2 ? 1u : 0u;
		alive_per_candidate[cb + 3] = a3 ? 1u : 0u;
		rep_per_candidate[cb + 0] = cb + 0u;
		rep_per_candidate[cb + 1] = cb + 1u;
		rep_per_candidate[cb + 2] = cb + 2u;
		rep_per_candidate[cb + 3] = cb + 3u;
	}
	__syncthreads();

	const bool a0i = alive_per_candidate[cb + 0] != 0u;
	const bool a1i = alive_per_candidate[cb + 1] != 0u;
	const bool a2i = alive_per_candidate[cb + 2] != 0u;
	const bool a3i = alive_per_candidate[cb + 3] != 0u;

	uint32_t value0 = a0i ? initialize_working_word(key, data_start + my_candidate_base + 0u, lane, expansion_values) : 0u;
	uint32_t value1 = a1i ? initialize_working_word(key, data_start + my_candidate_base + 1u, lane, expansion_values) : 0u;
	uint32_t value2 = a2i ? initialize_working_word(key, data_start + my_candidate_base + 2u, lane, expansion_values) : 0u;
	uint32_t value3 = a3i ? initialize_working_word(key, data_start + my_candidate_base + 3u, lane, expansion_values) : 0u;

	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		if (threadIdx.x < NSLOTS)
		{
			slot[threadIdx.x] = 0ull;
		}
		__syncthreads();

		const bool alive0 = alive_per_candidate[cb + 0] != 0u;
		const bool alive1 = alive_per_candidate[cb + 1] != 0u;
		const bool alive2 = alive_per_candidate[cb + 2] != 0u;
		const bool alive3 = alive_per_candidate[cb + 3] != 0u;

		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

		uint16_t rs0 = static_cast<uint16_t>(((packed_schedule & 0xFFu) << 8) | ((packed_schedule & 0xFF00u) >> 8));
		uint16_t rs1 = rs0, rs2 = rs0, rs3 = rs0;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));

		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, value0, sl);
			const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, value1, sl);
			const uint32_t w2 = __shfl_sync(0xFFFFFFFFu, value2, sl);
			const uint32_t w3 = __shfl_sync(0xFFFFFFFFu, value3, sl);
			uint8_t cb0 = static_cast<uint8_t>((w0 >> ss) & 0xFFu);
			uint8_t cb1 = static_cast<uint8_t>((w1 >> ss) & 0xFFu);
			uint8_t cb2 = static_cast<uint8_t>((w2 >> ss) & 0xFFu);
			uint8_t cb3 = static_cast<uint8_t>((w3 >> ss) & 0xFFu);
			if ((nsel & 0x8000u) != 0u) { cb0 >>= 4; cb1 >>= 4; cb2 >>= 4; cb3 >>= 4; }
			const uint8_t a0 = static_cast<uint8_t>((cb0 >> 1) & 0x07u);
			const uint8_t a1 = static_cast<uint8_t>((cb1 >> 1) & 0x07u);
			const uint8_t a2 = static_cast<uint8_t>((cb2 >> 1) & 0x07u);
			const uint8_t a3 = static_cast<uint8_t>((cb3 >> 1) & 0x07u);
			if (alive0) value0 = run_alg(value0, lane, a0, &rs0, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive1) value1 = run_alg(value1, lane, a1, &rs1, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive2) value2 = run_alg(value2, lane, a2, &rs2, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive3) value3 = run_alg(value3, lane, a3, &rs3, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nsel = static_cast<uint16_t>(nsel << 1);
		}

		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			uint32_t value;
			bool alive;
			switch (ilp)
			{
				case 0: value = value0; alive = alive0; break;
				case 1: value = value1; alive = alive1; break;
				case 2: value = value2; alive = alive2; break;
				default: value = value3; alive = alive3; break;
			}
			if (!alive) continue;
			uint32_t h_lo, h_hi;
			warp_hash_state(value, lane, &h_lo, &h_hi);
			const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
			const unsigned long long fp_top56 = fp_raw & FP_MASK;
			// Ensure packed != 0: if top 56 bits happened to be 0, set bit 8.
			const unsigned long long fp_safe = (fp_top56 == 0ull) ? 0x100ull : fp_top56;

			if (lane == 0u)
			{
				const uint32_t my_idx = cb + ilp;
				const unsigned long long my_packed = fp_safe | (unsigned long long)(my_idx & 0xFFu);
				uint32_t idx = static_cast<uint32_t>(fp_safe >> 8) & slot_mask;
				#pragma unroll 2
				for (uint32_t probe = 0u; probe < NSLOTS; probe++)
				{
					const unsigned long long prev = atomicCAS(&slot[idx], 0ull, my_packed);
					if (prev == 0ull)
					{
						// Owner — slot now atomically holds fp+rep. Done.
						break;
					}
					if ((prev & FP_MASK) == fp_safe)
					{
						// Duplicate — read rep from the bottom 8 bits of prev.
						// No spin: prev is the value as observed by CAS, already
						// fully populated by the owner's atomicCAS.
						const uint32_t prev_rep = static_cast<uint32_t>(prev & 0xFFull);
						rep_per_candidate[my_idx] = prev_rep;
						alive_per_candidate[my_idx] = 0u;
						break;
					}
					idx = (idx + 1u) & slot_mask;
				}
			}
		}
		__syncthreads();
	}

	const uint8_t f0 = screen_candidate(value0, lane, carnival_data);
	const uint8_t f1 = screen_candidate(value1, lane, carnival_data);
	const uint8_t f2 = screen_candidate(value2, lane, carnival_data);
	const uint8_t f3 = screen_candidate(value3, lane, carnival_data);

	if (lane == 0u)
	{
		if (alive_per_candidate[cb + 0]) flag_per_candidate[cb + 0] = f0;
		if (alive_per_candidate[cb + 1]) flag_per_candidate[cb + 1] = f1;
		if (alive_per_candidate[cb + 2]) flag_per_candidate[cb + 2] = f2;
		if (alive_per_candidate[cb + 3]) flag_per_candidate[cb + 3] = f3;
	}
	__syncthreads();

	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			const uint32_t c_local = cb + ilp;
			const uint32_t c_global = my_candidate_base + ilp;
			if (c_global >= candidate_count) continue;
			uint32_t cur = c_local;
			#pragma unroll 1
			for (uint32_t hop = 0u; hop < W; hop++)
			{
				const uint32_t r = rep_per_candidate[cur];
				if (r == cur || r >= W) break;
				cur = r;
			}
			result_data[c_global] = flag_per_candidate[cur];
		}
	}
}

extern "C" __global__ __launch_bounds__(256, 6) void tm_checksum_screen_dedup_w32_packed_cuda(
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
	tm_checksum_screen_dedup_w32_packed_impl<27u>(result_data, regular_rng_values, alg0_values, alg6_values,
		rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values,
		schedule_data, carnival_data, key, data_start, candidate_count);
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 2.9: cheaper warp hash (no per-lane salt, no murmur finalizer).
//
// Original warp_hash_state: 2 lane×constant adds, 2 murmur finalizers
// (~8 mult/shift ops), 5×2 XOR-butterflies (10 shuffle+xor). ~30 cycles/warp.
//
// Cheaper version: skip salt and finalizer, just XOR-fold with two
// differently-initialized halves. ~10 cycles/warp. Still 64-bit so
// collision rate stays ~2^-64 per pair (effectively zero false positives).
__device__ __forceinline__ void warp_hash_state_fast(
	uint32_t value, uint32_t lane,
	uint32_t* h_lo_out, uint32_t* h_hi_out)
{
	// hi MUST get per-lane salt — the constant XOR cancels across 32 lanes
	// during the butterfly, collapsing hi to equal lo and reducing 64-bit
	// hash to 32-bit entropy. With per-lane add, the salt accumulates as
	// (sum of lane indices) * K through the XOR-fold, preserving 64-bit entropy.
	uint32_t lo = value;
	uint32_t hi = value + lane * 0x9e3779b9u;
	#pragma unroll
	for (uint32_t off = 16u; off > 0u; off >>= 1u)
	{
		lo ^= __shfl_xor_sync(0xFFFFFFFFu, lo, off);
		hi ^= __shfl_xor_sync(0xFFFFFFFFu, hi, off);
	}
	*h_lo_out = lo;
	*h_hi_out = hi;
}

template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_checksum_screen_dedup_w32_fasthash_impl(
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
	constexpr uint32_t W = 32u;
	constexpr uint32_t NSLOTS = 64u;
	constexpr unsigned long long FP_MASK = 0xFFFFFFFFFFFFFF00ull;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t my_candidate_base = blockIdx.x * W + warp_index * 4u;
	const uint32_t cb = warp_index * 4u;
	const uint32_t slot_mask = NSLOTS - 1u;

	__shared__ unsigned long long slot[NSLOTS];
	__shared__ uint32_t           rep_per_candidate[W];
	__shared__ uint8_t            alive_per_candidate[W];
	__shared__ uint8_t            flag_per_candidate[W];

	if (lane == 0u)
	{
		const bool a0 = (my_candidate_base + 0u) < candidate_count;
		const bool a1 = (my_candidate_base + 1u) < candidate_count;
		const bool a2 = (my_candidate_base + 2u) < candidate_count;
		const bool a3 = (my_candidate_base + 3u) < candidate_count;
		alive_per_candidate[cb + 0] = a0 ? 1u : 0u;
		alive_per_candidate[cb + 1] = a1 ? 1u : 0u;
		alive_per_candidate[cb + 2] = a2 ? 1u : 0u;
		alive_per_candidate[cb + 3] = a3 ? 1u : 0u;
		rep_per_candidate[cb + 0] = cb + 0u;
		rep_per_candidate[cb + 1] = cb + 1u;
		rep_per_candidate[cb + 2] = cb + 2u;
		rep_per_candidate[cb + 3] = cb + 3u;
	}
	__syncthreads();

	const bool a0i = alive_per_candidate[cb + 0] != 0u;
	const bool a1i = alive_per_candidate[cb + 1] != 0u;
	const bool a2i = alive_per_candidate[cb + 2] != 0u;
	const bool a3i = alive_per_candidate[cb + 3] != 0u;

	uint32_t value0 = a0i ? initialize_working_word(key, data_start + my_candidate_base + 0u, lane, expansion_values) : 0u;
	uint32_t value1 = a1i ? initialize_working_word(key, data_start + my_candidate_base + 1u, lane, expansion_values) : 0u;
	uint32_t value2 = a2i ? initialize_working_word(key, data_start + my_candidate_base + 2u, lane, expansion_values) : 0u;
	uint32_t value3 = a3i ? initialize_working_word(key, data_start + my_candidate_base + 3u, lane, expansion_values) : 0u;

	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		if (threadIdx.x < NSLOTS)
		{
			slot[threadIdx.x] = 0ull;
		}
		__syncthreads();

		const bool alive0 = alive_per_candidate[cb + 0] != 0u;
		const bool alive1 = alive_per_candidate[cb + 1] != 0u;
		const bool alive2 = alive_per_candidate[cb + 2] != 0u;
		const bool alive3 = alive_per_candidate[cb + 3] != 0u;

		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

		uint16_t rs0 = static_cast<uint16_t>(((packed_schedule & 0xFFu) << 8) | ((packed_schedule & 0xFF00u) >> 8));
		uint16_t rs1 = rs0, rs2 = rs0, rs3 = rs0;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));

		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, value0, sl);
			const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, value1, sl);
			const uint32_t w2 = __shfl_sync(0xFFFFFFFFu, value2, sl);
			const uint32_t w3 = __shfl_sync(0xFFFFFFFFu, value3, sl);
			uint8_t cb0 = static_cast<uint8_t>((w0 >> ss) & 0xFFu);
			uint8_t cb1 = static_cast<uint8_t>((w1 >> ss) & 0xFFu);
			uint8_t cb2 = static_cast<uint8_t>((w2 >> ss) & 0xFFu);
			uint8_t cb3 = static_cast<uint8_t>((w3 >> ss) & 0xFFu);
			if ((nsel & 0x8000u) != 0u) { cb0 >>= 4; cb1 >>= 4; cb2 >>= 4; cb3 >>= 4; }
			const uint8_t a0 = static_cast<uint8_t>((cb0 >> 1) & 0x07u);
			const uint8_t a1 = static_cast<uint8_t>((cb1 >> 1) & 0x07u);
			const uint8_t a2 = static_cast<uint8_t>((cb2 >> 1) & 0x07u);
			const uint8_t a3 = static_cast<uint8_t>((cb3 >> 1) & 0x07u);
			if (alive0) value0 = run_alg(value0, lane, a0, &rs0, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive1) value1 = run_alg(value1, lane, a1, &rs1, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive2) value2 = run_alg(value2, lane, a2, &rs2, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			if (alive3) value3 = run_alg(value3, lane, a3, &rs3, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nsel = static_cast<uint16_t>(nsel << 1);
		}

		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			uint32_t value;
			bool alive;
			switch (ilp)
			{
				case 0: value = value0; alive = alive0; break;
				case 1: value = value1; alive = alive1; break;
				case 2: value = value2; alive = alive2; break;
				default: value = value3; alive = alive3; break;
			}
			if (!alive) continue;
			uint32_t h_lo, h_hi;
			warp_hash_state_fast(value, lane, &h_lo, &h_hi);
			const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
			const unsigned long long fp_top56 = fp_raw & FP_MASK;
			const unsigned long long fp_safe = (fp_top56 == 0ull) ? 0x100ull : fp_top56;

			if (lane == 0u)
			{
				const uint32_t my_idx = cb + ilp;
				const unsigned long long my_packed = fp_safe | (unsigned long long)(my_idx & 0xFFu);
				uint32_t idx = static_cast<uint32_t>(fp_safe >> 8) & slot_mask;
				#pragma unroll 2
				for (uint32_t probe = 0u; probe < NSLOTS; probe++)
				{
					const unsigned long long prev = atomicCAS(&slot[idx], 0ull, my_packed);
					if (prev == 0ull)
					{
						break;
					}
					if ((prev & FP_MASK) == fp_safe)
					{
						const uint32_t prev_rep = static_cast<uint32_t>(prev & 0xFFull);
						rep_per_candidate[my_idx] = prev_rep;
						alive_per_candidate[my_idx] = 0u;
						break;
					}
					idx = (idx + 1u) & slot_mask;
				}
			}
		}
		__syncthreads();
	}

	const uint8_t f0 = screen_candidate(value0, lane, carnival_data);
	const uint8_t f1 = screen_candidate(value1, lane, carnival_data);
	const uint8_t f2 = screen_candidate(value2, lane, carnival_data);
	const uint8_t f3 = screen_candidate(value3, lane, carnival_data);

	if (lane == 0u)
	{
		if (alive_per_candidate[cb + 0]) flag_per_candidate[cb + 0] = f0;
		if (alive_per_candidate[cb + 1]) flag_per_candidate[cb + 1] = f1;
		if (alive_per_candidate[cb + 2]) flag_per_candidate[cb + 2] = f2;
		if (alive_per_candidate[cb + 3]) flag_per_candidate[cb + 3] = f3;
	}
	__syncthreads();

	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			const uint32_t c_local = cb + ilp;
			const uint32_t c_global = my_candidate_base + ilp;
			if (c_global >= candidate_count) continue;
			uint32_t cur = c_local;
			#pragma unroll 1
			for (uint32_t hop = 0u; hop < W; hop++)
			{
				const uint32_t r = rep_per_candidate[cur];
				if (r == cur || r >= W) break;
				cur = r;
			}
			result_data[c_global] = flag_per_candidate[cur];
		}
	}
}

extern "C" __global__ __launch_bounds__(256, 6) void tm_checksum_screen_dedup_w32_fasthash_cuda(
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
	tm_checksum_screen_dedup_w32_fasthash_impl<27u>(result_data, regular_rng_values, alg0_values, alg6_values,
		rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values,
		schedule_data, carnival_data, key, data_start, candidate_count);
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 2.10: map_rng + dedup. Tests the user's hypothesis that the dedup
// kernel's higher cache pressure (L1 hit 70.5% vs screen's 78%) might
// benefit from the 54KB per-launch map_rng buffer that fits entirely in L1,
// even though map_rng was shown to be a net loss for the screen kernel
// (which is compute-bound and not memory-bound). The dedup kernel adds
// shared-mem hash activity that competes with the 8 MB universal table for
// cache budget — map_rng may flip the trade-off.
//
// Map_rng arithmetic overhead is paid per-alg-per-warp (bit extraction at
// runtime), but the locality benefit applies to all warps.
template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_checksum_screen_dedup_w32_maprng_impl(
	uint8_t* result_data,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	const uint8_t* map_rng,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	constexpr uint32_t W = 32u;
	constexpr uint32_t NSLOTS = 64u;
	constexpr unsigned long long FP_MASK = 0xFFFFFFFFFFFFFF00ull;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t my_candidate_base = blockIdx.x * W + warp_index * 4u;
	const uint32_t cb = warp_index * 4u;
	const uint32_t slot_mask = NSLOTS - 1u;

	__shared__ unsigned long long slot[NSLOTS];
	__shared__ uint32_t           rep_per_candidate[W];
	__shared__ uint8_t            alive_per_candidate[W];
	__shared__ uint8_t            flag_per_candidate[W];

	if (lane == 0u)
	{
		const bool a0 = (my_candidate_base + 0u) < candidate_count;
		const bool a1 = (my_candidate_base + 1u) < candidate_count;
		const bool a2 = (my_candidate_base + 2u) < candidate_count;
		const bool a3 = (my_candidate_base + 3u) < candidate_count;
		alive_per_candidate[cb + 0] = a0 ? 1u : 0u;
		alive_per_candidate[cb + 1] = a1 ? 1u : 0u;
		alive_per_candidate[cb + 2] = a2 ? 1u : 0u;
		alive_per_candidate[cb + 3] = a3 ? 1u : 0u;
		rep_per_candidate[cb + 0] = cb + 0u;
		rep_per_candidate[cb + 1] = cb + 1u;
		rep_per_candidate[cb + 2] = cb + 2u;
		rep_per_candidate[cb + 3] = cb + 3u;
	}
	__syncthreads();

	const bool a0i = alive_per_candidate[cb + 0] != 0u;
	const bool a1i = alive_per_candidate[cb + 1] != 0u;
	const bool a2i = alive_per_candidate[cb + 2] != 0u;
	const bool a3i = alive_per_candidate[cb + 3] != 0u;

	uint32_t value0 = a0i ? initialize_working_word(key, data_start + my_candidate_base + 0u, lane, expansion_values) : 0u;
	uint32_t value1 = a1i ? initialize_working_word(key, data_start + my_candidate_base + 1u, lane, expansion_values) : 0u;
	uint32_t value2 = a2i ? initialize_working_word(key, data_start + my_candidate_base + 2u, lane, expansion_values) : 0u;
	uint32_t value3 = a3i ? initialize_working_word(key, data_start + my_candidate_base + 3u, lane, expansion_values) : 0u;

	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		if (threadIdx.x < NSLOTS)
		{
			slot[threadIdx.x] = 0ull;
		}
		__syncthreads();

		const bool alive0 = alive_per_candidate[cb + 0] != 0u;
		const bool alive1 = alive_per_candidate[cb + 1] != 0u;
		const bool alive2 = alive_per_candidate[cb + 2] != 0u;
		const bool alive3 = alive_per_candidate[cb + 3] != 0u;

		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));

		const uint8_t* mr_base = map_rng + schedule_index * 2048u;
		// Per-candidate position counter (independent across ILP slots since each
		// candidate may take a different alg sequence).
		uint32_t pos0 = 0u, pos1 = 0u, pos2 = 0u, pos3 = 0u;

		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, value0, sl);
			const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, value1, sl);
			const uint32_t w2 = __shfl_sync(0xFFFFFFFFu, value2, sl);
			const uint32_t w3 = __shfl_sync(0xFFFFFFFFu, value3, sl);
			uint8_t cb0 = static_cast<uint8_t>((w0 >> ss) & 0xFFu);
			uint8_t cb1 = static_cast<uint8_t>((w1 >> ss) & 0xFFu);
			uint8_t cb2 = static_cast<uint8_t>((w2 >> ss) & 0xFFu);
			uint8_t cb3 = static_cast<uint8_t>((w3 >> ss) & 0xFFu);
			if ((nsel & 0x8000u) != 0u) { cb0 >>= 4; cb1 >>= 4; cb2 >>= 4; cb3 >>= 4; }
			const uint8_t a0 = static_cast<uint8_t>((cb0 >> 1) & 0x07u);
			const uint8_t a1 = static_cast<uint8_t>((cb1 >> 1) & 0x07u);
			const uint8_t a2 = static_cast<uint8_t>((cb2 >> 1) & 0x07u);
			const uint8_t a3 = static_cast<uint8_t>((cb3 >> 1) & 0x07u);
			if (alive0) value0 = run_alg_maprng(value0, lane, a0, &pos0, mr_base);
			if (alive1) value1 = run_alg_maprng(value1, lane, a1, &pos1, mr_base);
			if (alive2) value2 = run_alg_maprng(value2, lane, a2, &pos2, mr_base);
			if (alive3) value3 = run_alg_maprng(value3, lane, a3, &pos3, mr_base);
			nsel = static_cast<uint16_t>(nsel << 1);
		}

		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			uint32_t value;
			bool alive;
			switch (ilp)
			{
				case 0: value = value0; alive = alive0; break;
				case 1: value = value1; alive = alive1; break;
				case 2: value = value2; alive = alive2; break;
				default: value = value3; alive = alive3; break;
			}
			if (!alive) continue;
			uint32_t h_lo, h_hi;
			warp_hash_state(value, lane, &h_lo, &h_hi);
			const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
			const unsigned long long fp_top56 = fp_raw & FP_MASK;
			const unsigned long long fp_safe = (fp_top56 == 0ull) ? 0x100ull : fp_top56;

			if (lane == 0u)
			{
				const uint32_t my_idx = cb + ilp;
				const unsigned long long my_packed = fp_safe | (unsigned long long)(my_idx & 0xFFu);
				uint32_t idx = static_cast<uint32_t>(fp_safe >> 8) & slot_mask;
				#pragma unroll 2
				for (uint32_t probe = 0u; probe < NSLOTS; probe++)
				{
					const unsigned long long prev = atomicCAS(&slot[idx], 0ull, my_packed);
					if (prev == 0ull) break;
					if ((prev & FP_MASK) == fp_safe)
					{
						const uint32_t prev_rep = static_cast<uint32_t>(prev & 0xFFull);
						rep_per_candidate[my_idx] = prev_rep;
						alive_per_candidate[my_idx] = 0u;
						break;
					}
					idx = (idx + 1u) & slot_mask;
				}
			}
		}
		__syncthreads();
	}

	const uint8_t f0 = screen_candidate(value0, lane, carnival_data);
	const uint8_t f1 = screen_candidate(value1, lane, carnival_data);
	const uint8_t f2 = screen_candidate(value2, lane, carnival_data);
	const uint8_t f3 = screen_candidate(value3, lane, carnival_data);

	if (lane == 0u)
	{
		if (alive_per_candidate[cb + 0]) flag_per_candidate[cb + 0] = f0;
		if (alive_per_candidate[cb + 1]) flag_per_candidate[cb + 1] = f1;
		if (alive_per_candidate[cb + 2]) flag_per_candidate[cb + 2] = f2;
		if (alive_per_candidate[cb + 3]) flag_per_candidate[cb + 3] = f3;
	}
	__syncthreads();

	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			const uint32_t c_local = cb + ilp;
			const uint32_t c_global = my_candidate_base + ilp;
			if (c_global >= candidate_count) continue;
			uint32_t cur = c_local;
			#pragma unroll 1
			for (uint32_t hop = 0u; hop < W; hop++)
			{
				const uint32_t r = rep_per_candidate[cur];
				if (r == cur || r >= W) break;
				cur = r;
			}
			result_data[c_global] = flag_per_candidate[cur];
		}
	}
}

extern "C" __global__ __launch_bounds__(256, 6) void tm_checksum_screen_dedup_w32_maprng_cuda(
	uint8_t* result_data,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	const uint8_t* map_rng,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	tm_checksum_screen_dedup_w32_maprng_impl<27u>(result_data, expansion_values,
		schedule_data, carnival_data, map_rng, key, data_start, candidate_count);
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 2.11: pre-extracted maprng + dedup (cache locality WITHOUT arithmetic).
//
// Buffer layout (passed as concatenated 162 KB blob):
//   raw_stream[0..54KB]   — raw RNG bytes (alg 1, 3, 4 use this via pack_raw)
//   alg0_stream[54..108]  — (raw_byte >> 7) & 1, same reverse layout as raw
//   alg6_stream[108..162] — raw_byte & 0x80, FORWARD layout (alg6 reads forward)
//
// Kernel uses pack_raw for alg 1/3/4 against raw_stream, and pack_raw against
// alg0_stream for alg 0 (since alg0_stream is pre-extracted bits in the same
// reverse layout). For alg 6 we use a forward-pack variant against alg6_stream.
//
// This pre-extracts the per-byte bit work that run_alg_maprng was doing at
// runtime — eliminates ~5-10% of compute overhead while keeping map_rng's
// L1 hit benefit.
template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_checksum_screen_dedup_w32_maprng_preext_impl(
	uint8_t* result_data,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	const uint8_t* map_rng,  // [raw|alg0|alg6] concatenated, each 27*2048 bytes
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	constexpr uint32_t W = 32u;
	constexpr uint32_t NSLOTS = 64u;
	constexpr unsigned long long FP_MASK = 0xFFFFFFFFFFFFFF00ull;
	constexpr uint32_t STREAM_SIZE = 27u * 2048u;  // 54 KB per stream
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t my_candidate_base = blockIdx.x * W + warp_index * 4u;
	const uint32_t cb = warp_index * 4u;
	const uint32_t slot_mask = NSLOTS - 1u;

	const uint8_t* raw_base_all  = map_rng;
	const uint8_t* alg0_base_all = map_rng + STREAM_SIZE;
	const uint8_t* alg6_base_all = map_rng + STREAM_SIZE * 2u;

	__shared__ unsigned long long slot[NSLOTS];
	__shared__ uint32_t           rep_per_candidate[W];
	__shared__ uint8_t            alive_per_candidate[W];
	__shared__ uint8_t            flag_per_candidate[W];

	if (lane == 0u)
	{
		const bool a0 = (my_candidate_base + 0u) < candidate_count;
		const bool a1 = (my_candidate_base + 1u) < candidate_count;
		const bool a2 = (my_candidate_base + 2u) < candidate_count;
		const bool a3 = (my_candidate_base + 3u) < candidate_count;
		alive_per_candidate[cb + 0] = a0 ? 1u : 0u;
		alive_per_candidate[cb + 1] = a1 ? 1u : 0u;
		alive_per_candidate[cb + 2] = a2 ? 1u : 0u;
		alive_per_candidate[cb + 3] = a3 ? 1u : 0u;
		rep_per_candidate[cb + 0] = cb + 0u;
		rep_per_candidate[cb + 1] = cb + 1u;
		rep_per_candidate[cb + 2] = cb + 2u;
		rep_per_candidate[cb + 3] = cb + 3u;
	}
	__syncthreads();

	const bool a0i = alive_per_candidate[cb + 0] != 0u;
	const bool a1i = alive_per_candidate[cb + 1] != 0u;
	const bool a2i = alive_per_candidate[cb + 2] != 0u;
	const bool a3i = alive_per_candidate[cb + 3] != 0u;

	uint32_t value0 = a0i ? initialize_working_word(key, data_start + my_candidate_base + 0u, lane, expansion_values) : 0u;
	uint32_t value1 = a1i ? initialize_working_word(key, data_start + my_candidate_base + 1u, lane, expansion_values) : 0u;
	uint32_t value2 = a2i ? initialize_working_word(key, data_start + my_candidate_base + 2u, lane, expansion_values) : 0u;
	uint32_t value3 = a3i ? initialize_working_word(key, data_start + my_candidate_base + 3u, lane, expansion_values) : 0u;

	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		if (threadIdx.x < NSLOTS)
		{
			slot[threadIdx.x] = 0ull;
		}
		__syncthreads();

		const bool alive0 = alive_per_candidate[cb + 0] != 0u;
		const bool alive1 = alive_per_candidate[cb + 1] != 0u;
		const bool alive2 = alive_per_candidate[cb + 2] != 0u;
		const bool alive3 = alive_per_candidate[cb + 3] != 0u;

		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));

		// Per-stream bases for this map. raw and alg0 share REVERSE layout
		// (pack_raw); alg6 uses FORWARD layout (pack_msb_hi_fwd's read pattern).
		const uint8_t* raw_base  = raw_base_all  + schedule_index * 2048u;
		const uint8_t* alg0_base = alg0_base_all + schedule_index * 2048u;
		const uint8_t* alg6_base = alg6_base_all + schedule_index * 2048u;
		uint32_t pos0 = 0u, pos1 = 0u, pos2 = 0u, pos3 = 0u;

		// Inline run_alg_maprng with pre-extracted streams (no bit extraction).
		// Each alg call advances its candidate's pos by 128.
		// rng-seed advance (forward_128 vs forward_1) is implicit in pos
		// because the maprng buffer is laid out byte-sequentially.
		//
		// For alg 2/5 we still need the runtime extraction (cross-byte rotation)
		// — those aren't a hot path. We fall back to original maprng_pack_raw
		// here for any alg id we haven't pre-extracted.
		auto inline_run = [&](uint32_t value, uint32_t lane, uint8_t alg_id, uint32_t* pos)
			-> uint32_t
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
				// alg6's pre-extracted stream is in forward layout (same as
				// pack_msb_hi_fwd would read); pre-stored as (b & 0x80) so
				// pack-assembly produces the alg6 bit pattern with no runtime
				// mask.
				const uint32_t b0 = (uint32_t)(alg6_base[*pos + lane * 4u + 0u]);
				const uint32_t b1 = (uint32_t)(alg6_base[*pos + lane * 4u + 1u]);
				const uint32_t b2 = (uint32_t)(alg6_base[*pos + lane * 4u + 2u]);
				const uint32_t b3 = (uint32_t)(alg6_base[*pos + lane * 4u + 3u]);
				value = ((value >> 1) & 0x7F7F7F7Fu) | (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
				*pos += 128u;
			}
			else if (alg_id == 7u)
			{
				value ^= 0xFFFFFFFFu;
			}
			else
			{
				// alg 2/5 — runtime extraction still needed (rotation, single-bit carries).
				// Fall back to existing run_alg_maprng for these.
				value = run_alg_maprng(value, lane, alg_id, pos, raw_base);
			}
			return value;
		};

		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, value0, sl);
			const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, value1, sl);
			const uint32_t w2 = __shfl_sync(0xFFFFFFFFu, value2, sl);
			const uint32_t w3 = __shfl_sync(0xFFFFFFFFu, value3, sl);
			uint8_t cb0 = static_cast<uint8_t>((w0 >> ss) & 0xFFu);
			uint8_t cb1 = static_cast<uint8_t>((w1 >> ss) & 0xFFu);
			uint8_t cb2 = static_cast<uint8_t>((w2 >> ss) & 0xFFu);
			uint8_t cb3 = static_cast<uint8_t>((w3 >> ss) & 0xFFu);
			if ((nsel & 0x8000u) != 0u) { cb0 >>= 4; cb1 >>= 4; cb2 >>= 4; cb3 >>= 4; }
			const uint8_t a0 = static_cast<uint8_t>((cb0 >> 1) & 0x07u);
			const uint8_t a1 = static_cast<uint8_t>((cb1 >> 1) & 0x07u);
			const uint8_t a2 = static_cast<uint8_t>((cb2 >> 1) & 0x07u);
			const uint8_t a3 = static_cast<uint8_t>((cb3 >> 1) & 0x07u);
			if (alive0) value0 = inline_run(value0, lane, a0, &pos0);
			if (alive1) value1 = inline_run(value1, lane, a1, &pos1);
			if (alive2) value2 = inline_run(value2, lane, a2, &pos2);
			if (alive3) value3 = inline_run(value3, lane, a3, &pos3);
			nsel = static_cast<uint16_t>(nsel << 1);
		}

		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			uint32_t value;
			bool alive;
			switch (ilp)
			{
				case 0: value = value0; alive = alive0; break;
				case 1: value = value1; alive = alive1; break;
				case 2: value = value2; alive = alive2; break;
				default: value = value3; alive = alive3; break;
			}
			if (!alive) continue;
			uint32_t h_lo, h_hi;
			warp_hash_state(value, lane, &h_lo, &h_hi);
			const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
			const unsigned long long fp_top56 = fp_raw & FP_MASK;
			const unsigned long long fp_safe = (fp_top56 == 0ull) ? 0x100ull : fp_top56;

			if (lane == 0u)
			{
				const uint32_t my_idx = cb + ilp;
				const unsigned long long my_packed = fp_safe | (unsigned long long)(my_idx & 0xFFu);
				uint32_t idx = static_cast<uint32_t>(fp_safe >> 8) & slot_mask;
				#pragma unroll 2
				for (uint32_t probe = 0u; probe < NSLOTS; probe++)
				{
					const unsigned long long prev = atomicCAS(&slot[idx], 0ull, my_packed);
					if (prev == 0ull) break;
					if ((prev & FP_MASK) == fp_safe)
					{
						const uint32_t prev_rep = static_cast<uint32_t>(prev & 0xFFull);
						rep_per_candidate[my_idx] = prev_rep;
						alive_per_candidate[my_idx] = 0u;
						break;
					}
					idx = (idx + 1u) & slot_mask;
				}
			}
		}
		__syncthreads();
	}

	const uint8_t f0 = screen_candidate(value0, lane, carnival_data);
	const uint8_t f1 = screen_candidate(value1, lane, carnival_data);
	const uint8_t f2 = screen_candidate(value2, lane, carnival_data);
	const uint8_t f3 = screen_candidate(value3, lane, carnival_data);

	if (lane == 0u)
	{
		if (alive_per_candidate[cb + 0]) flag_per_candidate[cb + 0] = f0;
		if (alive_per_candidate[cb + 1]) flag_per_candidate[cb + 1] = f1;
		if (alive_per_candidate[cb + 2]) flag_per_candidate[cb + 2] = f2;
		if (alive_per_candidate[cb + 3]) flag_per_candidate[cb + 3] = f3;
	}
	__syncthreads();

	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			const uint32_t c_local = cb + ilp;
			const uint32_t c_global = my_candidate_base + ilp;
			if (c_global >= candidate_count) continue;
			uint32_t cur = c_local;
			#pragma unroll 1
			for (uint32_t hop = 0u; hop < W; hop++)
			{
				const uint32_t r = rep_per_candidate[cur];
				if (r == cur || r >= W) break;
				cur = r;
			}
			result_data[c_global] = flag_per_candidate[cur];
		}
	}
}

extern "C" __global__ __launch_bounds__(256, 6) void tm_checksum_screen_dedup_w32_maprng_preext_cuda(
	uint8_t* result_data,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	const uint8_t* map_rng,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	tm_checksum_screen_dedup_w32_maprng_preext_impl<27u>(result_data, expansion_values,
		schedule_data, carnival_data, map_rng, key, data_start, candidate_count);
}

// ────────────────────────────────────────────────────────────────────────────
// Dedup-schedule experiment kernels on the fast pre-extracted maprng base.
//
// Same compute path as tm_checksum_screen_dedup_w32_maprng_preext_impl, but the
// dedup boundary (slot reset + hash pass + the two __syncthreads it forces) is
// gated by a policy:
//
//   FINAL_ONLY=true   → dedup only after the LAST map (output-only dedup).
//                       Runs the whole schedule at full screen throughput with
//                       no per-boundary barriers, then does ONE within-block
//                       dedup pass for unique-output study (Experiment 3).
//   FINAL_ONLY=false  → dedup at every boundary where (schedule_index %
//                       DEDUP_PERIOD == 0). DEDUP_PERIOD=1 reproduces the
//                       every-boundary kernel; DEDUP_PERIOD=4 is "first map
//                       then every 4th" (Experiment 2); DEDUP_PERIOD>=SCHEDULE
//                       degenerates to "first map only" (Experiment 1).
//
// survivor_count (optional, may be null): each block atomic-adds the number of
// its candidates still alive at the end — i.e. within-block unique states. Lets
// the host compare the block-local collapse ratio against the global R.
template<uint32_t SCHEDULE_COUNT, uint32_t DEDUP_PERIOD, bool FINAL_ONLY>
__device__ __forceinline__ void tm_checksum_screen_dedup_w32_maprng_preext_sched_impl(
	uint8_t* result_data,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	const uint8_t* map_rng,
	unsigned long long* survivor_count,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	constexpr uint32_t W = 32u;
	constexpr uint32_t NSLOTS = 64u;
	constexpr unsigned long long FP_MASK = 0xFFFFFFFFFFFFFF00ull;
	constexpr uint32_t STREAM_SIZE = 27u * 2048u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t my_candidate_base = blockIdx.x * W + warp_index * 4u;
	const uint32_t cb = warp_index * 4u;
	const uint32_t slot_mask = NSLOTS - 1u;

	const uint8_t* raw_base_all  = map_rng;
	const uint8_t* alg0_base_all = map_rng + STREAM_SIZE;
	const uint8_t* alg6_base_all = map_rng + STREAM_SIZE * 2u;

	__shared__ unsigned long long slot[NSLOTS];
	__shared__ uint32_t           rep_per_candidate[W];
	__shared__ uint8_t            alive_per_candidate[W];
	__shared__ uint8_t            flag_per_candidate[W];

	if (lane == 0u)
	{
		const bool a0 = (my_candidate_base + 0u) < candidate_count;
		const bool a1 = (my_candidate_base + 1u) < candidate_count;
		const bool a2 = (my_candidate_base + 2u) < candidate_count;
		const bool a3 = (my_candidate_base + 3u) < candidate_count;
		alive_per_candidate[cb + 0] = a0 ? 1u : 0u;
		alive_per_candidate[cb + 1] = a1 ? 1u : 0u;
		alive_per_candidate[cb + 2] = a2 ? 1u : 0u;
		alive_per_candidate[cb + 3] = a3 ? 1u : 0u;
		rep_per_candidate[cb + 0] = cb + 0u;
		rep_per_candidate[cb + 1] = cb + 1u;
		rep_per_candidate[cb + 2] = cb + 2u;
		rep_per_candidate[cb + 3] = cb + 3u;
	}
	__syncthreads();

	const bool a0i = alive_per_candidate[cb + 0] != 0u;
	const bool a1i = alive_per_candidate[cb + 1] != 0u;
	const bool a2i = alive_per_candidate[cb + 2] != 0u;
	const bool a3i = alive_per_candidate[cb + 3] != 0u;

	uint32_t value0 = a0i ? initialize_working_word(key, data_start + my_candidate_base + 0u, lane, expansion_values) : 0u;
	uint32_t value1 = a1i ? initialize_working_word(key, data_start + my_candidate_base + 1u, lane, expansion_values) : 0u;
	uint32_t value2 = a2i ? initialize_working_word(key, data_start + my_candidate_base + 2u, lane, expansion_values) : 0u;
	uint32_t value3 = a3i ? initialize_working_word(key, data_start + my_candidate_base + 3u, lane, expansion_values) : 0u;

	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		const bool dedup_this = FINAL_ONLY
			? (schedule_index == (SCHEDULE_COUNT - 1u))
			: ((schedule_index % DEDUP_PERIOD) == 0u);

		if (dedup_this)
		{
			if (threadIdx.x < NSLOTS)
			{
				slot[threadIdx.x] = 0ull;
			}
			__syncthreads();
		}

		const bool alive0 = alive_per_candidate[cb + 0] != 0u;
		const bool alive1 = alive_per_candidate[cb + 1] != 0u;
		const bool alive2 = alive_per_candidate[cb + 2] != 0u;
		const bool alive3 = alive_per_candidate[cb + 3] != 0u;

		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);

		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));

		const uint8_t* raw_base  = raw_base_all  + schedule_index * 2048u;
		const uint8_t* alg0_base = alg0_base_all + schedule_index * 2048u;
		const uint8_t* alg6_base = alg6_base_all + schedule_index * 2048u;
		uint32_t pos0 = 0u, pos1 = 0u, pos2 = 0u, pos3 = 0u;

		auto inline_run = [&](uint32_t value, uint32_t lane, uint8_t alg_id, uint32_t* pos)
			-> uint32_t
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
				const uint32_t b0 = (uint32_t)(alg6_base[*pos + lane * 4u + 0u]);
				const uint32_t b1 = (uint32_t)(alg6_base[*pos + lane * 4u + 1u]);
				const uint32_t b2 = (uint32_t)(alg6_base[*pos + lane * 4u + 2u]);
				const uint32_t b3 = (uint32_t)(alg6_base[*pos + lane * 4u + 3u]);
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
			const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, value0, sl);
			const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, value1, sl);
			const uint32_t w2 = __shfl_sync(0xFFFFFFFFu, value2, sl);
			const uint32_t w3 = __shfl_sync(0xFFFFFFFFu, value3, sl);
			uint8_t cb0 = static_cast<uint8_t>((w0 >> ss) & 0xFFu);
			uint8_t cb1 = static_cast<uint8_t>((w1 >> ss) & 0xFFu);
			uint8_t cb2 = static_cast<uint8_t>((w2 >> ss) & 0xFFu);
			uint8_t cb3 = static_cast<uint8_t>((w3 >> ss) & 0xFFu);
			if ((nsel & 0x8000u) != 0u) { cb0 >>= 4; cb1 >>= 4; cb2 >>= 4; cb3 >>= 4; }
			const uint8_t a0 = static_cast<uint8_t>((cb0 >> 1) & 0x07u);
			const uint8_t a1 = static_cast<uint8_t>((cb1 >> 1) & 0x07u);
			const uint8_t a2 = static_cast<uint8_t>((cb2 >> 1) & 0x07u);
			const uint8_t a3 = static_cast<uint8_t>((cb3 >> 1) & 0x07u);
			if (alive0) value0 = inline_run(value0, lane, a0, &pos0);
			if (alive1) value1 = inline_run(value1, lane, a1, &pos1);
			if (alive2) value2 = inline_run(value2, lane, a2, &pos2);
			if (alive3) value3 = inline_run(value3, lane, a3, &pos3);
			nsel = static_cast<uint16_t>(nsel << 1);
		}

		if (dedup_this)
		{
			#pragma unroll
			for (uint32_t ilp = 0u; ilp < 4u; ilp++)
			{
				uint32_t value;
				bool alive;
				switch (ilp)
				{
					case 0: value = value0; alive = alive0; break;
					case 1: value = value1; alive = alive1; break;
					case 2: value = value2; alive = alive2; break;
					default: value = value3; alive = alive3; break;
				}
				if (!alive) continue;
				uint32_t h_lo, h_hi;
				warp_hash_state(value, lane, &h_lo, &h_hi);
				const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
				const unsigned long long fp_top56 = fp_raw & FP_MASK;
				const unsigned long long fp_safe = (fp_top56 == 0ull) ? 0x100ull : fp_top56;

				if (lane == 0u)
				{
					const uint32_t my_idx = cb + ilp;
					const unsigned long long my_packed = fp_safe | (unsigned long long)(my_idx & 0xFFu);
					uint32_t idx = static_cast<uint32_t>(fp_safe >> 8) & slot_mask;
					for (uint32_t probe = 0u; probe < NSLOTS; probe++)
					{
						const unsigned long long prev = atomicCAS(&slot[idx], 0ull, my_packed);
						if (prev == 0ull) break;
						if ((prev & FP_MASK) == fp_safe)
						{
							const uint32_t prev_rep = static_cast<uint32_t>(prev & 0xFFull);
							rep_per_candidate[my_idx] = prev_rep;
							alive_per_candidate[my_idx] = 0u;
							break;
						}
						idx = (idx + 1u) & slot_mask;
					}
				}
			}
			__syncthreads();
		}
	}

	const uint8_t f0 = screen_candidate(value0, lane, carnival_data);
	const uint8_t f1 = screen_candidate(value1, lane, carnival_data);
	const uint8_t f2 = screen_candidate(value2, lane, carnival_data);
	const uint8_t f3 = screen_candidate(value3, lane, carnival_data);

	if (lane == 0u)
	{
		if (alive_per_candidate[cb + 0]) flag_per_candidate[cb + 0] = f0;
		if (alive_per_candidate[cb + 1]) flag_per_candidate[cb + 1] = f1;
		if (alive_per_candidate[cb + 2]) flag_per_candidate[cb + 2] = f2;
		if (alive_per_candidate[cb + 3]) flag_per_candidate[cb + 3] = f3;
	}
	__syncthreads();

	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			const uint32_t c_local = cb + ilp;
			const uint32_t c_global = my_candidate_base + ilp;
			if (c_global >= candidate_count) continue;
			uint32_t cur = c_local;
			#pragma unroll 1
			for (uint32_t hop = 0u; hop < W; hop++)
			{
				const uint32_t r = rep_per_candidate[cur];
				if (r == cur || r >= W) break;
				cur = r;
			}
			result_data[c_global] = flag_per_candidate[cur];
		}

		if (survivor_count != nullptr)
		{
			uint32_t local = 0u;
			#pragma unroll
			for (uint32_t ilp = 0u; ilp < 4u; ilp++)
			{
				const uint32_t c_global = my_candidate_base + ilp;
				if (c_global < candidate_count && alive_per_candidate[cb + ilp]) local++;
			}
			if (local) atomicAdd(survivor_count, static_cast<unsigned long long>(local));
		}
	}
}

// Experiment 1: dedup after the FIRST map only (DEDUP_PERIOD == SCHEDULE_COUNT
// makes only schedule_index 0 satisfy index % period == 0).
extern "C" __global__ __launch_bounds__(256, 6) void tm_checksum_screen_dedup_w32_maprng_preext_first_cuda(
	uint8_t* result_data, const uint8_t* expansion_values, const uint8_t* schedule_data,
	const uint8_t* carnival_data, const uint8_t* map_rng, unsigned long long* survivor_count,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_checksum_screen_dedup_w32_maprng_preext_sched_impl<27u, 27u, false>(result_data, expansion_values,
		schedule_data, carnival_data, map_rng, survivor_count, key, data_start, candidate_count);
}

// Experiment 2: dedup at the first map, then every Kth map (index % K == 0).
extern "C" __global__ __launch_bounds__(256, 6) void tm_checksum_screen_dedup_w32_maprng_preext_k2_cuda(
	uint8_t* result_data, const uint8_t* expansion_values, const uint8_t* schedule_data,
	const uint8_t* carnival_data, const uint8_t* map_rng, unsigned long long* survivor_count,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_checksum_screen_dedup_w32_maprng_preext_sched_impl<27u, 2u, false>(result_data, expansion_values,
		schedule_data, carnival_data, map_rng, survivor_count, key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(256, 6) void tm_checksum_screen_dedup_w32_maprng_preext_k3_cuda(
	uint8_t* result_data, const uint8_t* expansion_values, const uint8_t* schedule_data,
	const uint8_t* carnival_data, const uint8_t* map_rng, unsigned long long* survivor_count,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_checksum_screen_dedup_w32_maprng_preext_sched_impl<27u, 3u, false>(result_data, expansion_values,
		schedule_data, carnival_data, map_rng, survivor_count, key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(256, 6) void tm_checksum_screen_dedup_w32_maprng_preext_k4_cuda(
	uint8_t* result_data, const uint8_t* expansion_values, const uint8_t* schedule_data,
	const uint8_t* carnival_data, const uint8_t* map_rng, unsigned long long* survivor_count,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_checksum_screen_dedup_w32_maprng_preext_sched_impl<27u, 4u, false>(result_data, expansion_values,
		schedule_data, carnival_data, map_rng, survivor_count, key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(256, 6) void tm_checksum_screen_dedup_w32_maprng_preext_k5_cuda(
	uint8_t* result_data, const uint8_t* expansion_values, const uint8_t* schedule_data,
	const uint8_t* carnival_data, const uint8_t* map_rng, unsigned long long* survivor_count,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_checksum_screen_dedup_w32_maprng_preext_sched_impl<27u, 5u, false>(result_data, expansion_values,
		schedule_data, carnival_data, map_rng, survivor_count, key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(256, 6) void tm_checksum_screen_dedup_w32_maprng_preext_k6_cuda(
	uint8_t* result_data, const uint8_t* expansion_values, const uint8_t* schedule_data,
	const uint8_t* carnival_data, const uint8_t* map_rng, unsigned long long* survivor_count,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_checksum_screen_dedup_w32_maprng_preext_sched_impl<27u, 6u, false>(result_data, expansion_values,
		schedule_data, carnival_data, map_rng, survivor_count, key, data_start, candidate_count);
}

extern "C" __global__ __launch_bounds__(256, 6) void tm_checksum_screen_dedup_w32_maprng_preext_k8_cuda(
	uint8_t* result_data, const uint8_t* expansion_values, const uint8_t* schedule_data,
	const uint8_t* carnival_data, const uint8_t* map_rng, unsigned long long* survivor_count,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_checksum_screen_dedup_w32_maprng_preext_sched_impl<27u, 8u, false>(result_data, expansion_values,
		schedule_data, carnival_data, map_rng, survivor_count, key, data_start, candidate_count);
}

// Experiment 3: dedup only after the last map (output-only). Full screen
// throughput; one within-block dedup pass for unique-output study.
extern "C" __global__ __launch_bounds__(256, 6) void tm_checksum_screen_dedup_w32_maprng_preext_output_cuda(
	uint8_t* result_data, const uint8_t* expansion_values, const uint8_t* schedule_data,
	const uint8_t* carnival_data, const uint8_t* map_rng, unsigned long long* survivor_count,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	tm_checksum_screen_dedup_w32_maprng_preext_sched_impl<27u, 1u, true>(result_data, expansion_values,
		schedule_data, carnival_data, map_rng, survivor_count, key, data_start, candidate_count);
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 2.12 feasibility test: collapse 2 syncthreads/boundary → 1.
//
// Original: [init slots] sync [run_map] [dedup pass] sync — 2 syncs/boundary × 27 = 54 syncs/kernel.
// New: [run_map] [dedup pass] [reset slots for NEXT iter] sync — 1 sync/boundary.
//
// Requires initial slot zeroing at kernel start (before any boundary).
// Correctness invariant: at start of each boundary's dedup, slots are zero.
// Maintained by: initial reset + per-boundary reset-at-end with sync.
template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_checksum_screen_dedup_w32_maprng_preext_1sync_impl(
	uint8_t* result_data,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	const uint8_t* map_rng,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	constexpr uint32_t W = 32u;
	constexpr uint32_t NSLOTS = 64u;
	constexpr unsigned long long FP_MASK = 0xFFFFFFFFFFFFFF00ull;
	constexpr uint32_t STREAM_SIZE = 27u * 2048u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t my_candidate_base = blockIdx.x * W + warp_index * 4u;
	const uint32_t cb = warp_index * 4u;
	const uint32_t slot_mask = NSLOTS - 1u;

	const uint8_t* raw_base_all  = map_rng;
	const uint8_t* alg0_base_all = map_rng + STREAM_SIZE;
	const uint8_t* alg6_base_all = map_rng + STREAM_SIZE * 2u;

	__shared__ unsigned long long slot[NSLOTS];
	__shared__ uint32_t           rep_per_candidate[W];
	__shared__ uint8_t            alive_per_candidate[W];
	__shared__ uint8_t            flag_per_candidate[W];

	// Initial setup: zero slots + init per-candidate metadata.
	if (threadIdx.x < NSLOTS) slot[threadIdx.x] = 0ull;
	if (lane == 0u)
	{
		const bool a0 = (my_candidate_base + 0u) < candidate_count;
		const bool a1 = (my_candidate_base + 1u) < candidate_count;
		const bool a2 = (my_candidate_base + 2u) < candidate_count;
		const bool a3 = (my_candidate_base + 3u) < candidate_count;
		alive_per_candidate[cb + 0] = a0 ? 1u : 0u;
		alive_per_candidate[cb + 1] = a1 ? 1u : 0u;
		alive_per_candidate[cb + 2] = a2 ? 1u : 0u;
		alive_per_candidate[cb + 3] = a3 ? 1u : 0u;
		rep_per_candidate[cb + 0] = cb + 0u;
		rep_per_candidate[cb + 1] = cb + 1u;
		rep_per_candidate[cb + 2] = cb + 2u;
		rep_per_candidate[cb + 3] = cb + 3u;
	}
	__syncthreads();

	const bool a0i = alive_per_candidate[cb + 0] != 0u;
	const bool a1i = alive_per_candidate[cb + 1] != 0u;
	const bool a2i = alive_per_candidate[cb + 2] != 0u;
	const bool a3i = alive_per_candidate[cb + 3] != 0u;

	uint32_t value0 = a0i ? initialize_working_word(key, data_start + my_candidate_base + 0u, lane, expansion_values) : 0u;
	uint32_t value1 = a1i ? initialize_working_word(key, data_start + my_candidate_base + 1u, lane, expansion_values) : 0u;
	uint32_t value2 = a2i ? initialize_working_word(key, data_start + my_candidate_base + 2u, lane, expansion_values) : 0u;
	uint32_t value3 = a3i ? initialize_working_word(key, data_start + my_candidate_base + 3u, lane, expansion_values) : 0u;

	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		// Slots are already zero at this point (either from initial setup or
		// from previous boundary's end-reset).

		const bool alive0 = alive_per_candidate[cb + 0] != 0u;
		const bool alive1 = alive_per_candidate[cb + 1] != 0u;
		const bool alive2 = alive_per_candidate[cb + 2] != 0u;
		const bool alive3 = alive_per_candidate[cb + 3] != 0u;

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
				const uint32_t b0 = (uint32_t)(alg6_base[*pos + lane * 4u + 0u]);
				const uint32_t b1 = (uint32_t)(alg6_base[*pos + lane * 4u + 1u]);
				const uint32_t b2 = (uint32_t)(alg6_base[*pos + lane * 4u + 2u]);
				const uint32_t b3 = (uint32_t)(alg6_base[*pos + lane * 4u + 3u]);
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
			const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, value0, sl);
			const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, value1, sl);
			const uint32_t w2 = __shfl_sync(0xFFFFFFFFu, value2, sl);
			const uint32_t w3 = __shfl_sync(0xFFFFFFFFu, value3, sl);
			uint8_t cb0 = static_cast<uint8_t>((w0 >> ss) & 0xFFu);
			uint8_t cb1 = static_cast<uint8_t>((w1 >> ss) & 0xFFu);
			uint8_t cb2 = static_cast<uint8_t>((w2 >> ss) & 0xFFu);
			uint8_t cb3 = static_cast<uint8_t>((w3 >> ss) & 0xFFu);
			if ((nsel & 0x8000u) != 0u) { cb0 >>= 4; cb1 >>= 4; cb2 >>= 4; cb3 >>= 4; }
			const uint8_t a0 = static_cast<uint8_t>((cb0 >> 1) & 0x07u);
			const uint8_t a1 = static_cast<uint8_t>((cb1 >> 1) & 0x07u);
			const uint8_t a2 = static_cast<uint8_t>((cb2 >> 1) & 0x07u);
			const uint8_t a3 = static_cast<uint8_t>((cb3 >> 1) & 0x07u);
			if (alive0) value0 = inline_run(value0, lane, a0, &pos0);
			if (alive1) value1 = inline_run(value1, lane, a1, &pos1);
			if (alive2) value2 = inline_run(value2, lane, a2, &pos2);
			if (alive3) value3 = inline_run(value3, lane, a3, &pos3);
			nsel = static_cast<uint16_t>(nsel << 1);
		}

		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			uint32_t value;
			bool alive;
			switch (ilp)
			{
				case 0: value = value0; alive = alive0; break;
				case 1: value = value1; alive = alive1; break;
				case 2: value = value2; alive = alive2; break;
				default: value = value3; alive = alive3; break;
			}
			if (!alive) continue;
			uint32_t h_lo, h_hi;
			warp_hash_state(value, lane, &h_lo, &h_hi);
			const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
			const unsigned long long fp_top56 = fp_raw & FP_MASK;
			const unsigned long long fp_safe = (fp_top56 == 0ull) ? 0x100ull : fp_top56;

			if (lane == 0u)
			{
				const uint32_t my_idx = cb + ilp;
				const unsigned long long my_packed = fp_safe | (unsigned long long)(my_idx & 0xFFu);
				uint32_t idx = static_cast<uint32_t>(fp_safe >> 8) & slot_mask;
				#pragma unroll 2
				for (uint32_t probe = 0u; probe < NSLOTS; probe++)
				{
					const unsigned long long prev = atomicCAS(&slot[idx], 0ull, my_packed);
					if (prev == 0ull) break;
					if ((prev & FP_MASK) == fp_safe)
					{
						const uint32_t prev_rep = static_cast<uint32_t>(prev & 0xFFull);
						rep_per_candidate[my_idx] = prev_rep;
						alive_per_candidate[my_idx] = 0u;
						break;
					}
					idx = (idx + 1u) & slot_mask;
				}
			}
		}

		// COLLAPSED SYNC: reset slots for next boundary's dedup (skip on last
		// boundary since no next dedup needs it), then ONE sync covers
		// both the dedup-CAS completion AND the slot-reset completion.
		if (schedule_index < SCHEDULE_COUNT - 1u && threadIdx.x < NSLOTS)
		{
			slot[threadIdx.x] = 0ull;
		}
		__syncthreads();
	}

	const uint8_t f0 = screen_candidate(value0, lane, carnival_data);
	const uint8_t f1 = screen_candidate(value1, lane, carnival_data);
	const uint8_t f2 = screen_candidate(value2, lane, carnival_data);
	const uint8_t f3 = screen_candidate(value3, lane, carnival_data);

	if (lane == 0u)
	{
		if (alive_per_candidate[cb + 0]) flag_per_candidate[cb + 0] = f0;
		if (alive_per_candidate[cb + 1]) flag_per_candidate[cb + 1] = f1;
		if (alive_per_candidate[cb + 2]) flag_per_candidate[cb + 2] = f2;
		if (alive_per_candidate[cb + 3]) flag_per_candidate[cb + 3] = f3;
	}
	__syncthreads();

	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			const uint32_t c_local = cb + ilp;
			const uint32_t c_global = my_candidate_base + ilp;
			if (c_global >= candidate_count) continue;
			uint32_t cur = c_local;
			#pragma unroll 1
			for (uint32_t hop = 0u; hop < W; hop++)
			{
				const uint32_t r = rep_per_candidate[cur];
				if (r == cur || r >= W) break;
				cur = r;
			}
			result_data[c_global] = flag_per_candidate[cur];
		}
	}
}

extern "C" __global__ __launch_bounds__(256, 6) void tm_checksum_screen_dedup_w32_maprng_preext_1sync_cuda(
	uint8_t* result_data,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	const uint8_t* map_rng,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	tm_checksum_screen_dedup_w32_maprng_preext_1sync_impl<27u>(result_data, expansion_values,
		schedule_data, carnival_data, map_rng, key, data_start, candidate_count);
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 2.13: coalesced store + pre-extracted maprng.
//
// ncu --set full on the prior production kernel flagged 49.5% potential
// speedup on global stores: "only 1.0 of 32 bytes utilized per sector".
// Root cause: end-of-kernel result_data writes done by lane 0 only of each
// warp, 4 separate 1-byte stores per warp × 8 warps per block. Each store
// instruction has 1/32 lanes active.
//
// Fix: stage flags in shared mem (already done), then have warp 0's 32 lanes
// each do their own chain-follow and write one byte to global memory. The
// write becomes a single coalesced 32-byte store per block.
template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_checksum_screen_dedup_w32_maprng_preext_cstore_impl(
	uint8_t* result_data,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	const uint8_t* map_rng,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	constexpr uint32_t W = 32u;
	constexpr uint32_t NSLOTS = 64u;
	constexpr unsigned long long FP_MASK = 0xFFFFFFFFFFFFFF00ull;
	constexpr uint32_t STREAM_SIZE = 27u * 2048u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t my_candidate_base = blockIdx.x * W + warp_index * 4u;
	const uint32_t cb = warp_index * 4u;
	const uint32_t slot_mask = NSLOTS - 1u;

	const uint8_t* raw_base_all  = map_rng;
	const uint8_t* alg0_base_all = map_rng + STREAM_SIZE;
	const uint8_t* alg6_base_all = map_rng + STREAM_SIZE * 2u;

	__shared__ unsigned long long slot[NSLOTS];
	__shared__ uint32_t           rep_per_candidate[W];
	__shared__ uint8_t            alive_per_candidate[W];
	__shared__ uint8_t            flag_per_candidate[W];

	if (lane == 0u)
	{
		const bool a0 = (my_candidate_base + 0u) < candidate_count;
		const bool a1 = (my_candidate_base + 1u) < candidate_count;
		const bool a2 = (my_candidate_base + 2u) < candidate_count;
		const bool a3 = (my_candidate_base + 3u) < candidate_count;
		alive_per_candidate[cb + 0] = a0 ? 1u : 0u;
		alive_per_candidate[cb + 1] = a1 ? 1u : 0u;
		alive_per_candidate[cb + 2] = a2 ? 1u : 0u;
		alive_per_candidate[cb + 3] = a3 ? 1u : 0u;
		rep_per_candidate[cb + 0] = cb + 0u;
		rep_per_candidate[cb + 1] = cb + 1u;
		rep_per_candidate[cb + 2] = cb + 2u;
		rep_per_candidate[cb + 3] = cb + 3u;
	}
	__syncthreads();

	const bool a0i = alive_per_candidate[cb + 0] != 0u;
	const bool a1i = alive_per_candidate[cb + 1] != 0u;
	const bool a2i = alive_per_candidate[cb + 2] != 0u;
	const bool a3i = alive_per_candidate[cb + 3] != 0u;

	uint32_t value0 = a0i ? initialize_working_word(key, data_start + my_candidate_base + 0u, lane, expansion_values) : 0u;
	uint32_t value1 = a1i ? initialize_working_word(key, data_start + my_candidate_base + 1u, lane, expansion_values) : 0u;
	uint32_t value2 = a2i ? initialize_working_word(key, data_start + my_candidate_base + 2u, lane, expansion_values) : 0u;
	uint32_t value3 = a3i ? initialize_working_word(key, data_start + my_candidate_base + 3u, lane, expansion_values) : 0u;

	for (uint32_t schedule_index = 0; schedule_index < SCHEDULE_COUNT; schedule_index++)
	{
		if (threadIdx.x < NSLOTS) slot[threadIdx.x] = 0ull;
		__syncthreads();

		const bool alive0 = alive_per_candidate[cb + 0] != 0u;
		const bool alive1 = alive_per_candidate[cb + 1] != 0u;
		const bool alive2 = alive_per_candidate[cb + 2] != 0u;
		const bool alive3 = alive_per_candidate[cb + 3] != 0u;

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
			if      (alg_id == 0u) { value = ((value << 1) & 0xFEFEFEFEu) | maprng_pack_raw(alg0_base, *pos, lane); *pos += 128u; }
			else if (alg_id == 1u) { value = __vadd4(value, maprng_pack_raw(raw_base, *pos, lane)); *pos += 128u; }
			else if (alg_id == 3u) { value ^= maprng_pack_raw(raw_base, *pos, lane); *pos += 128u; }
			else if (alg_id == 4u) { value = __vsub4(value, maprng_pack_raw(raw_base, *pos, lane)); *pos += 128u; }
			else if (alg_id == 6u)
			{
				const uint32_t b0 = (uint32_t)(alg6_base[*pos + lane * 4u + 0u]);
				const uint32_t b1 = (uint32_t)(alg6_base[*pos + lane * 4u + 1u]);
				const uint32_t b2 = (uint32_t)(alg6_base[*pos + lane * 4u + 2u]);
				const uint32_t b3 = (uint32_t)(alg6_base[*pos + lane * 4u + 3u]);
				value = ((value >> 1) & 0x7F7F7F7Fu) | (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
				*pos += 128u;
			}
			else if (alg_id == 7u) { value ^= 0xFFFFFFFFu; }
			else { value = run_alg_maprng(value, lane, alg_id, pos, raw_base); }
			return value;
		};

		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, value0, sl);
			const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, value1, sl);
			const uint32_t w2 = __shfl_sync(0xFFFFFFFFu, value2, sl);
			const uint32_t w3 = __shfl_sync(0xFFFFFFFFu, value3, sl);
			uint8_t cb0 = static_cast<uint8_t>((w0 >> ss) & 0xFFu);
			uint8_t cb1 = static_cast<uint8_t>((w1 >> ss) & 0xFFu);
			uint8_t cb2 = static_cast<uint8_t>((w2 >> ss) & 0xFFu);
			uint8_t cb3 = static_cast<uint8_t>((w3 >> ss) & 0xFFu);
			if ((nsel & 0x8000u) != 0u) { cb0 >>= 4; cb1 >>= 4; cb2 >>= 4; cb3 >>= 4; }
			const uint8_t a0 = static_cast<uint8_t>((cb0 >> 1) & 0x07u);
			const uint8_t a1 = static_cast<uint8_t>((cb1 >> 1) & 0x07u);
			const uint8_t a2 = static_cast<uint8_t>((cb2 >> 1) & 0x07u);
			const uint8_t a3 = static_cast<uint8_t>((cb3 >> 1) & 0x07u);
			if (alive0) value0 = inline_run(value0, lane, a0, &pos0);
			if (alive1) value1 = inline_run(value1, lane, a1, &pos1);
			if (alive2) value2 = inline_run(value2, lane, a2, &pos2);
			if (alive3) value3 = inline_run(value3, lane, a3, &pos3);
			nsel = static_cast<uint16_t>(nsel << 1);
		}

		#pragma unroll
		for (uint32_t ilp = 0u; ilp < 4u; ilp++)
		{
			uint32_t value;
			bool alive;
			switch (ilp)
			{
				case 0: value = value0; alive = alive0; break;
				case 1: value = value1; alive = alive1; break;
				case 2: value = value2; alive = alive2; break;
				default: value = value3; alive = alive3; break;
			}
			if (!alive) continue;
			uint32_t h_lo, h_hi;
			warp_hash_state(value, lane, &h_lo, &h_hi);
			const unsigned long long fp_raw = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
			const unsigned long long fp_top56 = fp_raw & FP_MASK;
			const unsigned long long fp_safe = (fp_top56 == 0ull) ? 0x100ull : fp_top56;

			if (lane == 0u)
			{
				const uint32_t my_idx = cb + ilp;
				const unsigned long long my_packed = fp_safe | (unsigned long long)(my_idx & 0xFFu);
				uint32_t idx = static_cast<uint32_t>(fp_safe >> 8) & slot_mask;
				#pragma unroll 2
				for (uint32_t probe = 0u; probe < NSLOTS; probe++)
				{
					const unsigned long long prev = atomicCAS(&slot[idx], 0ull, my_packed);
					if (prev == 0ull) break;
					if ((prev & FP_MASK) == fp_safe)
					{
						const uint32_t prev_rep = static_cast<uint32_t>(prev & 0xFFull);
						rep_per_candidate[my_idx] = prev_rep;
						alive_per_candidate[my_idx] = 0u;
						break;
					}
					idx = (idx + 1u) & slot_mask;
				}
			}
		}
		__syncthreads();
	}

	const uint8_t f0 = screen_candidate(value0, lane, carnival_data);
	const uint8_t f1 = screen_candidate(value1, lane, carnival_data);
	const uint8_t f2 = screen_candidate(value2, lane, carnival_data);
	const uint8_t f3 = screen_candidate(value3, lane, carnival_data);

	if (lane == 0u)
	{
		if (alive_per_candidate[cb + 0]) flag_per_candidate[cb + 0] = f0;
		if (alive_per_candidate[cb + 1]) flag_per_candidate[cb + 1] = f1;
		if (alive_per_candidate[cb + 2]) flag_per_candidate[cb + 2] = f2;
		if (alive_per_candidate[cb + 3]) flag_per_candidate[cb + 3] = f3;
	}
	__syncthreads();

	// COALESCED STORE: warp 0's 32 lanes each handle one of the block's 32
	// candidates (chain-follow + global write). All 32 lanes write to
	// contiguous result_data bytes → single 32-byte coalesced store per block
	// instead of 8 separate 4-byte stores (each with 1/32 lane efficiency).
	if (warp_index == 0u)
	{
		const uint32_t c_local = lane;
		const uint32_t c_global = blockIdx.x * W + lane;
		if (c_global < candidate_count)
		{
			uint32_t cur = c_local;
			#pragma unroll 1
			for (uint32_t hop = 0u; hop < W; hop++)
			{
				const uint32_t r = rep_per_candidate[cur];
				if (r == cur || r >= W) break;
				cur = r;
			}
			result_data[c_global] = flag_per_candidate[cur];
		}
	}
}

extern "C" __global__ __launch_bounds__(256, 6) void tm_checksum_screen_dedup_w32_maprng_preext_cstore_cuda(
	uint8_t* result_data,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	const uint8_t* map_rng,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	tm_checksum_screen_dedup_w32_maprng_preext_cstore_impl<27u>(result_data, expansion_values,
		schedule_data, carnival_data, map_rng, key, data_start, candidate_count);
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 3: Coalesced + pre-extracted maprng SCREEN kernel (no dedup).
//
// This combines the 5/23 maprng-cache hypothesis with the 2026-05-24
// pre-extracted-streams + uint32-coalesced-loads ideas. Tests whether the
// combined optimizations finally make maprng beat universal tables for the
// production screen kernel — which is used on 92% of keys.
//
// Uses production screen kernel geometry (128 threads = 4 warps × 4 ILP =
// 16 candidates per block). Reads from the coalesced 3-stream buffer with
// single uint32 loads per lane per alg call — fully coalesced (lane L's
// uint32 at offset 4L, 32 lanes cover 128 contiguous bytes in one sector
// group).
template<uint32_t SCHEDULE_COUNT>
__device__ __forceinline__ void tm_checksum_screen_maprng_coalesced_impl(
	uint8_t* result_data,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	const uint8_t* map_rng,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	constexpr uint32_t STREAM_SIZE = 27u * 2048u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * 4u;

	if (candidate_base >= candidate_count) return;

	const uint8_t* raw_base_all  = map_rng;
	const uint8_t* alg0_base_all = map_rng + STREAM_SIZE;
	const uint8_t* alg6_base_all = map_rng + STREAM_SIZE * 2u;

	uint32_t value0 = initialize_working_word(key, data_start + candidate_base + 0u, lane, expansion_values);
	uint32_t value1 = initialize_working_word(key, data_start + candidate_base + 1u, lane, expansion_values);
	uint32_t value2 = initialize_working_word(key, data_start + candidate_base + 2u, lane, expansion_values);
	uint32_t value3 = initialize_working_word(key, data_start + candidate_base + 3u, lane, expansion_values);

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

		// Coalesced uint32 load per alg call: lane L reads (uint32*)(base+pos)[lane].
		// All 32 lanes' loads in one instruction cover 128 contiguous bytes →
		// single coalesced sector-group fetch, 100% byte utilization.
		// alg 2/5 still use original byte-granular maprng pack (cross-byte
		// rotations need single-bit access).
		auto inline_run = [&](uint32_t value, uint32_t lane, uint8_t alg_id, uint32_t* pos) -> uint32_t
		{
			// Coalesced uint32 fast path is only correct when pos is
			// SEGMENT-aligned (pos % 128 == 0), not just 4-byte aligned.
			// The coalesced reorder happens within each 128-byte segment;
			// any pos within a segment maps to scrambled coalesced offsets.
			//
			// After alg 2/5 advances pos by 1, pos is no longer segment-aligned
			// → byte fallback that computes per-byte coalesced offset for each
			// original byte position.
			const bool seg_aligned = ((*pos) % 128u) == 0u;
			auto coalesced_byte_read = [&](const uint8_t* base) -> uint32_t {
				// Per-byte: for original position K, coalesced offset =
				// (K / 128) * 128 + (127 - (K % 128)).
				auto rd = [&](uint32_t orig_off) -> uint8_t {
					const uint32_t seg_base = (orig_off / 128u) * 128u;
					const uint32_t within   = orig_off % 128u;
					return base[seg_base + (127u - within)];
				};
				const uint32_t b0 = (uint32_t)rd(*pos + lane * 4u + 0u);
				const uint32_t b1 = (uint32_t)rd(*pos + lane * 4u + 1u);
				const uint32_t b2 = (uint32_t)rd(*pos + lane * 4u + 2u);
				const uint32_t b3 = (uint32_t)rd(*pos + lane * 4u + 3u);
				return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
			};
			if (alg_id == 0u) {
				uint32_t v = seg_aligned
					? reinterpret_cast<const uint32_t*>(alg0_base + *pos)[lane]
					: coalesced_byte_read(alg0_base);
				value = ((value << 1) & 0xFEFEFEFEu) | v; *pos += 128u;
			}
			else if (alg_id == 1u) {
				uint32_t v = seg_aligned
					? reinterpret_cast<const uint32_t*>(raw_base + *pos)[lane]
					: coalesced_byte_read(raw_base);
				value = __vadd4(value, v); *pos += 128u;
			}
			else if (alg_id == 3u) {
				uint32_t v = seg_aligned
					? reinterpret_cast<const uint32_t*>(raw_base + *pos)[lane]
					: coalesced_byte_read(raw_base);
				value ^= v; *pos += 128u;
			}
			else if (alg_id == 4u) {
				uint32_t v = seg_aligned
					? reinterpret_cast<const uint32_t*>(raw_base + *pos)[lane]
					: coalesced_byte_read(raw_base);
				value = __vsub4(value, v); *pos += 128u;
			}
			else if (alg_id == 6u) {
				uint32_t v = seg_aligned
					? reinterpret_cast<const uint32_t*>(alg6_base + *pos)[lane]
					: coalesced_byte_read(alg6_base);
				value = ((value >> 1) & 0x7F7F7F7Fu) | v; *pos += 128u;
			}
			else if (alg_id == 7u) { value ^= 0xFFFFFFFFu; }
			else if (alg_id == 2u || alg_id == 5u) {
				// Inline alg 2/5 against coalesced layout. Carry byte at original
				// offset pos in coalesced layout = base[seg_base + 127 - within_seg].
				const uint32_t seg_base = (*pos / 128u) * 128u;
				const uint32_t within   = (*pos) % 128u;
				const uint32_t coal_off = seg_base + (127u - within);
				const uint8_t carry_byte = raw_base[coal_off];
				const uint32_t next_lane_value = __shfl_down_sync(0xFFFFFFFFu, value, 1);
				if (alg_id == 2u) {
					uint32_t temp = (value & 0x00010000u) >> 8;
					if (lane == 31u) temp |= ((static_cast<uint32_t>(carry_byte) >> 7) & 1u) << 24;
					else             temp |= ((next_lane_value & 0x00000001u) << 24);
					temp |= (value >> 1) & 0x007F007Fu;
					temp |= (value << 1) & 0xFE00FE00u;
					temp |= (value >> 8) & 0x00800080u;
					value = temp;
				} else {
					uint32_t temp = (value & 0x00800000u) >> 8;
					if (lane == 31u) temp |= (static_cast<uint32_t>(carry_byte) & 0x80u) << 24;
					else             temp |= ((next_lane_value & 0x00000080u) << 24);
					temp |= (value >> 1) & 0x7F007F00u;
					temp |= (value << 1) & 0x00FE00FEu;
					temp |= (value >> 8) & 0x00010001u;
					value = temp;
				}
				*pos += 1u;
			}
			return value;
		};

		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, value0, sl);
			const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, value1, sl);
			const uint32_t w2 = __shfl_sync(0xFFFFFFFFu, value2, sl);
			const uint32_t w3 = __shfl_sync(0xFFFFFFFFu, value3, sl);
			uint8_t cb0 = static_cast<uint8_t>((w0 >> ss) & 0xFFu);
			uint8_t cb1 = static_cast<uint8_t>((w1 >> ss) & 0xFFu);
			uint8_t cb2 = static_cast<uint8_t>((w2 >> ss) & 0xFFu);
			uint8_t cb3 = static_cast<uint8_t>((w3 >> ss) & 0xFFu);
			if ((nsel & 0x8000u) != 0u) { cb0 >>= 4; cb1 >>= 4; cb2 >>= 4; cb3 >>= 4; }
			const uint8_t a0 = static_cast<uint8_t>((cb0 >> 1) & 0x07u);
			const uint8_t a1 = static_cast<uint8_t>((cb1 >> 1) & 0x07u);
			const uint8_t a2 = static_cast<uint8_t>((cb2 >> 1) & 0x07u);
			const uint8_t a3 = static_cast<uint8_t>((cb3 >> 1) & 0x07u);
			value0 = inline_run(value0, lane, a0, &pos0);
			value1 = inline_run(value1, lane, a1, &pos1);
			value2 = inline_run(value2, lane, a2, &pos2);
			value3 = inline_run(value3, lane, a3, &pos3);
			nsel = static_cast<uint16_t>(nsel << 1);
		}
	}

	const uint8_t f0 = screen_candidate(value0, lane, carnival_data);
	const uint8_t f1 = screen_candidate(value1, lane, carnival_data);
	const uint8_t f2 = screen_candidate(value2, lane, carnival_data);
	const uint8_t f3 = screen_candidate(value3, lane, carnival_data);
	if (lane == 0u)
	{
		result_data[candidate_base + 0u] = f0;
		if ((candidate_base + 1u) < candidate_count) result_data[candidate_base + 1u] = f1;
		if ((candidate_base + 2u) < candidate_count) result_data[candidate_base + 2u] = f2;
		if ((candidate_base + 3u) < candidate_count) result_data[candidate_base + 3u] = f3;
	}
}

extern "C" __global__ __launch_bounds__(128) void tm_checksum_screen_maprng_coalesced_cuda(
	uint8_t* result_data,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	const uint8_t* carnival_data,
	const uint8_t* map_rng,
	uint32_t key,
	uint32_t data_start,
	uint32_t candidate_count)
{
	tm_checksum_screen_maprng_coalesced_impl<27u>(result_data, expansion_values,
		schedule_data, carnival_data, map_rng, key, data_start, candidate_count);
}


// ─────────────────────────────────────────────────────────────────────────────
// Wide-merge fp-dump (2026-06-03). Maps the first `first_maps` schedule entries
// over a whole-byte window and emits one 64-bit fingerprint per candidate — the
// shared front-end for the wide-merge collapse variants (sort / global-hash).
// Uses the CANONICAL run_alg path (rng_seed + seed_forward tables), matching the
// state-validated Phase-1 dedup kernels — NOT run_alg_offset (offset streams are
// only screen-equivalent, not full-state-faithful, and under-collapse ~2x).
// One candidate per warp; lane 0 writes the fp (fp=0 shifted to 1 = empty slot).
extern "C" __global__ __launch_bounds__(128) void tm_wide_merge_fp_dump_n_cuda(
	uint64_t* fp_out,
	const uint8_t* regular_rng_values,
	const uint8_t* alg0_values,
	const uint8_t* alg6_values,
	const uint16_t* rng_forward_1,
	const uint16_t* rng_forward_128,
	const uint32_t* alg2_values,
	const uint32_t* alg5_values,
	const uint8_t* expansion_values,
	const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count, uint32_t first_maps)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t cand = blockIdx.x * warps_per_block + warp_index;
	if (cand >= candidate_count) return;

	uint32_t value = initialize_working_word(key, data_start + cand, lane, expansion_values);
	for (uint32_t schedule_index = 0u; schedule_index < first_maps; schedule_index++)
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
				regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}
	}
	uint32_t h_lo, h_hi;
	warp_hash_state(value, lane, &h_lo, &h_hi);
	if (lane == 0u)
	{
		uint64_t fp = (static_cast<uint64_t>(h_hi) << 32) | static_cast<uint64_t>(h_lo);
		if (fp == 0ull) fp = 1ull;
		fp_out[cand] = fp;
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Global-VRAM open-addressing collapse (2026-06-03) — the MEASURED BASELINE for
// the wide-merge sort variant. Collapses the per-candidate fingerprint stream
// (tm_wide_merge_fp_dump_n_cuda output) into one slot per unique state via a
// shared VRAM hash table. Cost model predicts this DEGRADES at high R (≈95% of
// inserts collide on hot slots when R≈22) where sort stays flat — this kernel lets
// us measure that head-to-head. Operates directly on the driver-allocated fp buffer
// (pure fatbin kernel, no cub/runtime interop needed).
//   table_fp[M]  : 0 = empty slot, else the claimed fingerprint
//   table_rep[M] : 0xFFFFFFFF until the slot owner writes its index (spin sentinel)
// Per candidate: survivor_flag=1 iff it claimed a slot (is a run rep); rep_out=its
// representative's index. unique_counter is atomically incremented per claimed slot.
extern "C" __global__ void tm_wide_merge_hash_zero_cuda(
	uint64_t* table_fp, uint32_t* table_rep, uint32_t slots)
{
	const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
	if (s >= slots) return;
	table_fp[s] = 0ull;
	table_rep[s] = 0xFFFFFFFFu;
}

extern "C" __global__ void tm_map1_frontier_zero_cuda(
	uint64_t* table_fp, uint32_t slots)
{
	const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
	if (s >= slots) return;
	table_fp[s] = 0ull;
}

extern "C" __global__ void tm_map1_frontier_zero_log_cuda(
	uint64_t* table_fp, uint32_t logm)
{
	const uint64_t s = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const uint64_t slots = 1ull << logm;
	if (s >= slots) return;
	table_fp[s] = 0ull;
}

__device__ __forceinline__ uint64_t tm_rotl64(uint64_t x, int r)
{
	return (x << r) | (x >> (64 - r));
}

__device__ __forceinline__ uint64_t tm_map1_hstrong_step(uint64_t h, uint64_t w)
{
	h ^= w;
	h = tm_rotl64(h, 31);
	h *= 0x9E3779B97F4A7C15ull;
	h ^= h >> 29;
	h *= 0xBF58476D1CE4E5B9ull;
	return h;
}

__device__ __forceinline__ uint64_t tm_strong64_state(uint32_t value, uint32_t lane)
{
	uint64_t h0 = 0xcbf29ce484222325ull;
	#pragma unroll
	for (uint32_t i = 0u; i < 16u; i++)
	{
		const uint32_t lo = __shfl_sync(0xFFFFFFFFu, value, i * 2u);
		const uint32_t hi = __shfl_sync(0xFFFFFFFFu, value, i * 2u + 1u);
		const uint64_t w = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
		h0 = tm_map1_hstrong_step(h0, w);
	}
	h0 ^= h0 >> 32;
	if (lane == 0u && h0 == 0ull) h0 = 1ull;
	return h0;
}

// 128-bit atomic compare-and-swap (native atom.global.cas.b128; Blackwell sm_90+,
// validated on sm_120). Returns true iff the slot held (exp_lo,exp_hi) and was
// swapped to (des_lo,des_hi); the prior value is returned in (old_lo,old_hi). addr
// must be 16-byte aligned. One atomic claims AND publishes a 128-bit slot, so there
// is no two-word publication race (the reason the first prototype used strong64).
__device__ __forceinline__ bool tm_atomic_cas128(
	unsigned long long* addr,
	unsigned long long exp_lo, unsigned long long exp_hi,
	unsigned long long des_lo, unsigned long long des_hi,
	unsigned long long& old_lo, unsigned long long& old_hi)
{
	asm volatile(
		"{\n\t"
		".reg .b128 c;\n\t.reg .b128 s;\n\t.reg .b128 t;\n\t"
		"mov.b128 c, {%3, %4};\n\t"
		"mov.b128 s, {%5, %6};\n\t"
		"atom.global.cas.b128 t, [%2], c, s;\n\t"
		"mov.b128 {%0, %1}, t;\n\t"
		"}"
		: "=l"(old_lo), "=l"(old_hi)
		: "l"(addr), "l"(exp_lo), "l"(exp_hi), "l"(des_lo), "l"(des_hi)
		: "memory");
	return (old_lo == exp_lo && old_hi == exp_hi);
}

// 128-bit strong fingerprint matching CPU w4b::strong128 (same hstrong as strong64,
// two independent seeds). (h0,h1) is never (0,0) so callers can use (0,0) as the
// empty-slot sentinel. All lanes receive the full fingerprint (needed for the
// warp-cooperative bucketed insert below). strong64 birthday-collides past ~600M F1
// (survey saw false merges at F1~3.2B); strong128 is bulletproof to billions.
__device__ __forceinline__ void tm_strong128_state(uint32_t value, uint64_t& h0, uint64_t& h1)
{
	uint64_t a = 0xcbf29ce484222325ull;
	uint64_t b = 0x9E3779B97F4A7C15ull;
	#pragma unroll
	for (uint32_t i = 0u; i < 16u; i++)
	{
		const uint32_t lo = __shfl_sync(0xFFFFFFFFu, value, i * 2u);
		const uint32_t hi = __shfl_sync(0xFFFFFFFFu, value, i * 2u + 1u);
		const uint64_t w = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
		a = tm_map1_hstrong_step(a, w);
		b = tm_map1_hstrong_step(b, w);
	}
	a ^= a >> 32;
	b ^= b >> 32;
	if (a == 0ull && b == 0ull) a = 1ull;
	h0 = a; h1 = b;
}

// Scatter the low set-bits of `bits` onto the set positions of `mask` (PDEP). Shared
// by the window-policy remap (squeeze tiling) and the MAP-K selected-bit frontier probe.
__device__ __forceinline__ uint32_t tm_deposit_bits32(uint32_t bits, uint32_t mask)
{
	uint32_t out = 0u;
	while (mask != 0u)
	{
		const uint32_t bit = mask & (0u - mask);
		if ((bits & 1u) != 0u) out |= bit;
		bits >>= 1u;
		mask ^= bit;
	}
	return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Compression-study gather (2026-06-18): full-schedule (fp128, flag) dump over an
// EXPLICIT list of data values. Used to turn a screen's hit list (the rare flag!=0
// candidates) into a deduplicable (fingerprint, flag) stream so the host can count
// UNIQUE passing outputs — not the weighted all-data hit count that inflates when
// many inputs collapse to the same passing state. One warp per list entry; the full
// schedule is run generically (schedule_count read at runtime, so variable-length
// key schedules work). fp128 == tm_strong128_state (the canonical full-state hash,
// bulletproof to billions), and the flag is recomputed via screen_candidate so the
// (fp, flag) pair is internally consistent. data_list[i] is the absolute 32-bit data
// value (not a tile offset). fp_lo/fp_hi/flag_out are parallel output arrays of
// length count; lane 0 writes them.
extern "C" __global__ __launch_bounds__(128) void tm_study_gather_fp_flag_cuda(
	const uint32_t* data_list, uint32_t count,
	uint64_t* fp_lo_out, uint64_t* fp_hi_out, uint8_t* flag_out,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	const uint8_t* carnival_data, uint32_t key, uint32_t schedule_count)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t entry = blockIdx.x * warps_per_block + warp_index;
	if (entry >= count) return;

	const uint32_t data = data_list[entry];
	uint32_t value = initialize_working_word(key, data, lane, expansion_values);
	for (uint32_t schedule_index = 0u; schedule_index < schedule_count; schedule_index++)
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
				regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}
	}
	uint64_t h0, h1;
	tm_strong128_state(value, h0, h1);
	const uint8_t f = (carnival_data != nullptr) ? screen_candidate(value, lane, carnival_data) : 0u;
	if (lane == 0u)
	{
		fp_lo_out[entry] = h0;
		fp_hi_out[entry] = h1;
		flag_out[entry] = f;
	}
}

// Compression-study screen+HLL with WINDOW-POLICY remap (2026-06-18). Identical to
// tm_screen_and_hll_cuda except the candidate index is REMAPPED to a data value via a
// bit-deposit: data = fixed_value | deposit(index, inner_mask). With inner_mask = the
// squeeze-selected log2(tile) bits and fixed_value = deposit(tile_index, outer_mask),
// each tile holds the co-collapsing inputs (the squeeze wave-locality enumeration the
// CPU raceway uses), so the per-tile HLL distinct reflects squeeze tile compression
// (not the linear contiguous tiling, which under-collapses). The global union over all
// tiles is unchanged (squeeze is a bijection of 2^32), so true_R / flags are identical.
extern "C" __global__ __launch_bounds__(128) void tm_screen_and_hll_remap_cuda(
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
	uint32_t fixed_value,
	uint32_t inner_mask,
	uint32_t candidate_count)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * 4u;
	if (candidate_base >= candidate_count) return;

	const uint32_t d0 = fixed_value | tm_deposit_bits32(candidate_base + 0u, inner_mask);
	const uint32_t d1 = fixed_value | tm_deposit_bits32(candidate_base + 1u, inner_mask);
	const uint32_t d2 = fixed_value | tm_deposit_bits32(candidate_base + 2u, inner_mask);
	const uint32_t d3 = fixed_value | tm_deposit_bits32(candidate_base + 3u, inner_mask);

	uint32_t v0 = initialize_working_word(key, d0, lane, expansion_values);
	uint32_t v1 = initialize_working_word(key, d1, lane, expansion_values);
	uint32_t v2 = initialize_working_word(key, d2, lane, expansion_values);
	uint32_t v3 = initialize_working_word(key, d3, lane, expansion_values);

	run_schedule_quad_t<27u>(&v0, &v1, &v2, &v3, lane,
		regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128,
		alg2_values, alg5_values, schedule_data);

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

	uint32_t h_lo, h_hi;
	warp_hash_state(v0, lane, &h_lo, &h_hi);
	if (lane == 0u) hll_update(hll_registers, h_lo, h_hi);
	if ((candidate_base + 1u) < candidate_count) { warp_hash_state(v1, lane, &h_lo, &h_hi); if (lane == 0u) hll_update(hll_registers, h_lo, h_hi); }
	if ((candidate_base + 2u) < candidate_count) { warp_hash_state(v2, lane, &h_lo, &h_hi); if (lane == 0u) hll_update(hll_registers, h_lo, h_hi); }
	if ((candidate_base + 3u) < candidate_count) { warp_hash_state(v3, lane, &h_lo, &h_hi); if (lane == 0u) hll_update(hll_registers, h_lo, h_hi); }
}

extern "C" __global__ __launch_bounds__(128) void tm_screen_remap_offset_cuda(
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
	uint32_t fixed_value,
	uint32_t selected_mask,
	uint32_t candidate_start,
	uint32_t candidate_count)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * 4u;
	if (candidate_base >= candidate_count) return;

	const uint32_t logical = candidate_start + candidate_base;
	const uint32_t d0 = fixed_value | tm_deposit_bits32(logical + 0u, selected_mask);
	const uint32_t d1 = fixed_value | tm_deposit_bits32(logical + 1u, selected_mask);
	const uint32_t d2 = fixed_value | tm_deposit_bits32(logical + 2u, selected_mask);
	const uint32_t d3 = fixed_value | tm_deposit_bits32(logical + 3u, selected_mask);

	uint32_t v0 = initialize_working_word(key, d0, lane, expansion_values);
	uint32_t v1 = initialize_working_word(key, d1, lane, expansion_values);
	uint32_t v2 = initialize_working_word(key, d2, lane, expansion_values);
	uint32_t v3 = initialize_working_word(key, d3, lane, expansion_values);

	run_schedule_quad_t<27u>(&v0, &v1, &v2, &v3, lane,
		regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128,
		alg2_values, alg5_values, schedule_data);

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

// Classify every input candidate by a high-bit prefix of its MAP1 strong64
// fingerprint. The full-domain byte label array lets the host process one
// fingerprint partition at a time while computing MAP1 only twice overall:
// once here and once in the selected partition's insert pass.
extern "C" __global__ __launch_bounds__(128) void tm_map1_frontier_classify_cuda(
	uint8_t* partition_out, uint32_t log_partitions,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t cand = blockIdx.x * warps_per_block + warp_index;
	if (cand >= candidate_count) return;

	const uint32_t data = data_start + cand;
	uint32_t value = initialize_working_word(key, data, lane, expansion_values);
	uint32_t packed_schedule = 0u;
	if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[0];
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
			regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
		nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
	}

	const uint64_t fp0 = tm_strong64_state(value, lane);
	if (lane == 0u)
		partition_out[cand] = static_cast<uint8_t>(fp0 >> (64u - log_partitions));
}

// MAP1 frontier decision emitter prototype (2026-06-03): stream chunks of the
// data axis through expand+MAP1 and insert only a strong 64-bit fingerprint into
// a persistent VRAM hash set. This is intentionally frontier-sized: no per-input
// 128-byte state buffer and no per-input collapse arrays. A new slot increments
// unique_counter and optionally appends the representative data value. Strong64 is
// used here to avoid a two-word publication race in the first prototype; the CPU
// probes showed it exact at W4B-relevant single-key scales, while a later production
// emitter can add 128-bit/re-derive-confirm.
extern "C" __global__ __launch_bounds__(128) void tm_map1_frontier_insert_cuda(
	uint64_t* table_fp, uint32_t logm,
	uint32_t* unique_counter, uint32_t* overflow_counter, uint32_t* rep_out, uint32_t rep_cap,
	uint8_t* unique_out,
	const uint8_t* partition_labels, uint32_t target_partition,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t cand = blockIdx.x * warps_per_block + warp_index;
	if (cand >= candidate_count) return;
	if (partition_labels != nullptr)
	{
		uint32_t selected = 0u;
		if (lane == 0u) selected = partition_labels[cand] == target_partition ? 1u : 0u;
		selected = __shfl_sync(0xFFFFFFFFu, selected, 0);
		if (selected == 0u) return;
	}

	const uint32_t data = data_start + cand;
	uint32_t value = initialize_working_word(key, data, lane, expansion_values);
	uint32_t packed_schedule = 0u;
	if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[0];
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
			regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
		nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
	}

	const uint64_t fp0 = tm_strong64_state(value, lane);
	if (lane != 0u) return;

	const uint32_t mask = (logm == 32u) ? 0xFFFFFFFFu : ((1u << logm) - 1u);
	uint32_t slot = static_cast<uint32_t>((fp0 * 0x9E3779B97F4A7C15ull) >> (64 - logm));
	constexpr uint32_t PROBE_CAP = 128u;
	for (uint32_t probe = 0u; probe < PROBE_CAP; ++probe)
	{
		const uint32_t s = (slot + probe) & mask;
		const uint64_t old = atomicCAS(reinterpret_cast<unsigned long long*>(&table_fp[s]),
			0ull, static_cast<unsigned long long>(fp0));
		if (old == 0ull)
		{
			const uint32_t u = atomicAdd(unique_counter, 1u);
			if (rep_out != nullptr && u < rep_cap) rep_out[u] = data;
			if (unique_out != nullptr) unique_out[cand] = 1u;
			return;
		}
		if (old == fp0)
		{
			return;
		}
	}
	atomicAdd(overflow_counter, 1u);
	const uint32_t u = atomicAdd(unique_counter, 1u);
	if (rep_out != nullptr && u < rep_cap) rep_out[u] = data;
	if (unique_out != nullptr) unique_out[cand] = 1u;
}

__device__ __forceinline__ uint32_t tm_map1_frontier_emit_rep(
	uint32_t data, uint32_t cand,
	uint32_t* unique_counter, uint32_t* rep_out, uint32_t rep_cap, uint8_t* unique_out,
	uint32_t* rep_mult, uint32_t rep_mult_cap, uint32_t* rep_idx_by_cand)
{
	const uint32_t u = atomicAdd(unique_counter, 1u);
	if (rep_out != nullptr && u < rep_cap) rep_out[u] = data;
	if (unique_out != nullptr) unique_out[cand] = 1u;
	if (rep_mult != nullptr && u < rep_mult_cap) rep_mult[u] = 1u;
	if (rep_idx_by_cand != nullptr) rep_idx_by_cand[cand] = u;
	return u;
}

__device__ __forceinline__ void tm_map1_frontier_publish_owner(uint32_t* table_owner, uint32_t slot, uint32_t owner)
{
	if (table_owner != nullptr) atomicExch(table_owner + slot, owner);
}

__device__ __forceinline__ uint32_t tm_map1_frontier_wait_owner(uint32_t* table_owner, uint32_t slot)
{
	if (table_owner == nullptr) return 0xFFFFFFFFu;
	volatile uint32_t* owner_v = reinterpret_cast<volatile uint32_t*>(table_owner);
	uint32_t owner = owner_v[slot];
	while (owner == 0xFFFFFFFFu) owner = owner_v[slot];
	return owner;
}

__device__ __forceinline__ void tm_map1_frontier_merge_mult(
	uint32_t* table_owner, uint32_t slot, uint32_t* rep_mult, uint32_t rep_mult_cap)
{
	if (rep_mult == nullptr) return;
	const uint32_t owner = tm_map1_frontier_wait_owner(table_owner, slot);
	if (owner < rep_mult_cap) atomicAdd(rep_mult + owner, 1u);
}

// Warp-cooperative bucketed 128-bit MAP1 frontier insert (2026-06-04). Same
// expand+MAP1 front end as tm_map1_frontier_insert_cuda, but fixes the two limits
// the strong64 lane-0 linear-probe insert hit at W4B scale:
//   (1) correctness — strong64 birthday-collides past ~600M F1 (CPU survey saw
//       false merges at F1~3.2B = dropped unlike states); strong128 is bulletproof.
//   (2) memory-level parallelism — the old insert had ONE outstanding probe per
//       warp (lane 0 serial), so the DRAM-scatter probe latency was fully exposed
//       (ncu: 82% of stall cycles waiting on the table load at the 32 GiB tail).
//       Here the table is ulonglong2[slots] in 32-slot buckets; all 32 lanes read a
//       whole bucket in ONE coalesced 128-bit load and ballot for match/empty, and
//       a single atom.cas.b128 claims+publishes a slot (no two-word race). Empty
//       slot == (0,0); a real fingerprint is never (0,0).
extern "C" __global__ __launch_bounds__(128) void tm_map1_frontier_insert128_cuda(
	ulonglong2* table, uint32_t logm,
	uint32_t* unique_counter, uint32_t* overflow_counter, uint32_t* rep_out, uint32_t rep_cap,
	uint8_t* unique_out,
	const uint8_t* partition_labels, uint32_t target_partition,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count, uint32_t tau,
	uint32_t unused_map_depth, uint32_t unused_sample_mult,
	uint32_t* table_owner, uint32_t* rep_mult, uint32_t rep_mult_cap, uint32_t* rep_idx_by_cand)
{
	const uint32_t lane = threadIdx.x & 31u;
	(void)unused_map_depth;
	(void)unused_sample_mult;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t cand = blockIdx.x * warps_per_block + warp_index;
	if (cand >= candidate_count) return;
	if (partition_labels != nullptr)
	{
		uint32_t selected = 0u;
		if (lane == 0u) selected = partition_labels[cand] == target_partition ? 1u : 0u;
		selected = __shfl_sync(0xFFFFFFFFu, selected, 0);
		if (selected == 0u) return;
	}

	const uint32_t data = data_start + cand;
	uint32_t value = initialize_working_word(key, data, lane, expansion_values);
	uint32_t packed_schedule = 0u;
	if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[0];
	packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
	uint16_t rng_seed = static_cast<uint16_t>(((packed_schedule & 0x000000FFu) << 8)
		| ((packed_schedule & 0x0000FF00u) >> 8));
	uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
		| ((packed_schedule & 0xFF000000u) >> 24));
	uint32_t shed = 0u;
	for (uint32_t i = 0u; i < 16u; i++)
	{
		const uint32_t source_lane = i >> 2;
		const uint32_t source_shift = (i & 3u) * 8u;
		const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, value, source_lane);
		uint8_t current_byte = static_cast<uint8_t>((source_word >> source_shift) & 0xFFu);
		if ((nibble_selector & 0x8000u) != 0u) current_byte = static_cast<uint8_t>(current_byte >> 4);
		const uint8_t algorithm_id = static_cast<uint8_t>((current_byte >> 1) & 0x07u);
		if (algorithm_id == 0u || algorithm_id == 6u) shed++;  // shed proxy (warp-uniform)
		value = run_alg(value, lane, algorithm_id, &rng_seed,
			regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
		nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
	}

	uint64_t h0, h1;
	tm_strong128_state(value, h0, h1);  // all lanes hold (h0,h1); never (0,0)

	// Routing: low-shed states bypass the table and are emitted directly as their own rep.
	// High-shed states (shed >= tau) are deduplicated via the table as normal. tau==0 = no routing.
	if (tau > 0u && shed < tau)
	{
		if (lane == 0u)
			tm_map1_frontier_emit_rep(data, cand, unique_counter, rep_out, rep_cap, unique_out,
				rep_mult, rep_mult_cap, rep_idx_by_cand);
		return;
	}

	// Bucket addressing: slots = 2^logm grouped into 2^(logm-5) buckets of 32.
	// Host guarantees logm >= 10, so logb >= 5.
	const uint32_t logb = logm - 5u;
	const uint32_t bucket_mask = (logb >= 32u) ? 0xFFFFFFFFu : ((1u << logb) - 1u);
	const uint32_t home = static_cast<uint32_t>((h0 * 0x9E3779B97F4A7C15ull) >> (64u - logb));
	constexpr uint32_t PROBE_BUCKETS = 16u;

	bool done = false;
	for (uint32_t pb = 0u; pb < PROBE_BUCKETS && !done; ++pb)
	{
			const uint32_t bucket = (home + pb) & bucket_mask;
			const uint32_t slot = bucket * 32u + lane;
			for (;;)  // rescan this bucket until match / claimed / full
			{
				const ulonglong2 cur = table[slot];                 // coalesced 128-bit load
				const uint32_t match_ballot = __ballot_sync(0xFFFFFFFFu, cur.x == h0 && cur.y == h1);
				if (match_ballot)
				{
					if (lane == 0u)
					{
						const uint32_t m = static_cast<uint32_t>(__ffs(static_cast<int>(match_ballot))) - 1u;
						tm_map1_frontier_merge_mult(table_owner, bucket * 32u + m, rep_mult, rep_mult_cap);
					}
					done = true;
					break;
				}
				const uint32_t empty_ballot = __ballot_sync(0xFFFFFFFFu, cur.x == 0ull && cur.y == 0ull);
				if (empty_ballot == 0u) break;                      // bucket full -> next bucket
				const uint32_t e = static_cast<uint32_t>(__ffs(static_cast<int>(empty_ballot))) - 1u;
			unsigned long long ol = 0ull, oh = 0ull; int won = 0;
			if (lane == e)
				won = tm_atomic_cas128(reinterpret_cast<unsigned long long*>(&table[slot]),
					0ull, 0ull, h0, h1, ol, oh) ? 1 : 0;
			won = __shfl_sync(0xFFFFFFFFu, won, e);
			ol  = __shfl_sync(0xFFFFFFFFu, ol, e);
			oh  = __shfl_sync(0xFFFFFFFFu, oh, e);
			if (won)
				{
					if (lane == 0u)
					{
						const uint32_t u = tm_map1_frontier_emit_rep(data, cand, unique_counter,
							rep_out, rep_cap, unique_out, rep_mult, rep_mult_cap, rep_idx_by_cand);
						tm_map1_frontier_publish_owner(table_owner, bucket * 32u + e, u);
					}
				done = true; break;
			}
			if (ol == h0 && oh == h1)
			{
				if (lane == 0u)
					tm_map1_frontier_merge_mult(table_owner, bucket * 32u + e, rep_mult, rep_mult_cap);
				done = true;
				break;
			}   // a concurrent warp inserted our fp
			// slot e was taken by a different fp -> rescan bucket (now resident in L1)
		}
	}
	if (!done && lane == 0u)  // PROBE_BUCKETS exhausted: carry as its own representative (bounded, safe)
	{
		atomicAdd(overflow_counter, 1u);
		tm_map1_frontier_emit_rep(data, cand, unique_counter, rep_out, rep_cap, unique_out,
			rep_mult, rep_mult_cap, rep_idx_by_cand);
	}
}

// Target A (2026-06-04): ILP variant of the bucketed 128-bit MAP1 insert. The
// profile showed the W4B-tail bottleneck is LOW memory-level parallelism — one
// outstanding table probe per warp, so the cold-DRAM scatter latency is fully
// exposed (82% of stall cycles). Here each warp processes ILP candidates and
// ISSUES ALL ILP HOME-BUCKET READS TOGETHER (phase 2) so their independent DRAM
// round-trips overlap, then resolves each (phase 3). The 31-idle-lane map compute
// stays warp-cooperative (one candidate at a time); only the memory side is
// pipelined. F1 is identical to the ILP=1 kernel.
template<uint32_t ILP>
__device__ __forceinline__ void tm_map1_frontier_insert128_ilp_core(
	ulonglong2* table, uint32_t logm,
	uint32_t* unique_counter, uint32_t* overflow_counter, uint32_t* rep_out, uint32_t rep_cap,
	uint8_t* unique_out,
	const uint8_t* partition_labels, uint32_t target_partition,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count, uint32_t tau,
	uint32_t unused_map_depth, uint32_t unused_sample_mult,
	uint32_t* table_owner, uint32_t* rep_mult, uint32_t rep_mult_cap, uint32_t* rep_idx_by_cand)
{
	const uint32_t lane = threadIdx.x & 31u;
	(void)unused_map_depth;
	(void)unused_sample_mult;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t cand_base = (blockIdx.x * warps_per_block + warp_index) * ILP;
	if (cand_base >= candidate_count) return;

	const uint32_t logb = logm - 5u;
	const uint32_t bucket_mask = (logb >= 32u) ? 0xFFFFFFFFu : ((1u << logb) - 1u);

	uint32_t packed_schedule = 0u;
	if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[0];
	packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
	const uint16_t rng_seed0 = static_cast<uint16_t>(((packed_schedule & 0x000000FFu) << 8)
		| ((packed_schedule & 0x0000FF00u) >> 8));
	const uint16_t nibble0 = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
		| ((packed_schedule & 0xFF000000u) >> 24));

	uint64_t h0[ILP], h1[ILP];
	uint32_t home[ILP], data_j[ILP];
	uint32_t shed[ILP];
	bool active[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++) shed[j] = 0u;

	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)  // phase 1: map + fingerprint for each candidate
	{
		const uint32_t cand = cand_base + j;
		active[j] = (cand < candidate_count);
		if (partition_labels != nullptr)
		{
			uint32_t sel = 0u;
			if (lane == 0u && active[j]) sel = (partition_labels[cand] == target_partition) ? 1u : 0u;
			sel = __shfl_sync(0xFFFFFFFFu, sel, 0);
			active[j] = active[j] && (sel != 0u);
		}
		const uint32_t data = data_start + cand;
		data_j[j] = data;
		uint32_t value = initialize_working_word(key, data, lane, expansion_values);
		uint16_t rng_seed = rng_seed0;
		uint16_t nibble_selector = nibble0;
		for (uint32_t i = 0u; i < 16u; i++)
		{
			const uint32_t source_lane = i >> 2;
			const uint32_t source_shift = (i & 3u) * 8u;
			const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, value, source_lane);
			uint8_t current_byte = static_cast<uint8_t>((source_word >> source_shift) & 0xFFu);
			if ((nibble_selector & 0x8000u) != 0u) current_byte = static_cast<uint8_t>(current_byte >> 4);
			const uint8_t algorithm_id = static_cast<uint8_t>((current_byte >> 1) & 0x07u);
			if (algorithm_id == 0u || algorithm_id == 6u) shed[j]++;  // shed proxy (warp-uniform)
			value = run_alg(value, lane, algorithm_id, &rng_seed,
				regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}
		tm_strong128_state(value, h0[j], h1[j]);
		home[j] = static_cast<uint32_t>((h0[j] * 0x9E3779B97F4A7C15ull) >> (64u - logb));
	}

	ulonglong2 pre[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)  // phase 2: issue all ILP home-bucket reads together (MLP)
		pre[j] = table[home[j] * 32u + lane];

	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)  // phase 3: resolve each candidate's insert
	{
		if (!active[j]) continue;
		// Routing: low-shed candidates bypass the table and are emitted directly as their own rep.
		if (tau > 0u && shed[j] < tau)
		{
			if (lane == 0u)
				tm_map1_frontier_emit_rep(data_j[j], cand_base + j, unique_counter,
					rep_out, rep_cap, unique_out, rep_mult, rep_mult_cap, rep_idx_by_cand);
			continue;
		}
		constexpr uint32_t PROBE_BUCKETS = 16u;
		bool done = false;
		bool first = true;
		for (uint32_t pb = 0u; pb < PROBE_BUCKETS && !done; ++pb)
		{
			const uint32_t bucket = (home[j] + pb) & bucket_mask;
			const uint32_t slot = bucket * 32u + lane;
			for (;;)
			{
				const ulonglong2 cur = first ? pre[j] : table[slot];  // first probe uses the preloaded bucket
				first = false;
				const uint32_t match_ballot = __ballot_sync(0xFFFFFFFFu, cur.x == h0[j] && cur.y == h1[j]);
				if (match_ballot)
				{
					if (lane == 0u)
					{
						const uint32_t m = static_cast<uint32_t>(__ffs(static_cast<int>(match_ballot))) - 1u;
						tm_map1_frontier_merge_mult(table_owner, bucket * 32u + m, rep_mult, rep_mult_cap);
					}
					done = true;
					break;
				}
				const uint32_t empty_ballot = __ballot_sync(0xFFFFFFFFu, cur.x == 0ull && cur.y == 0ull);
				if (empty_ballot == 0u) break;
				const uint32_t e = static_cast<uint32_t>(__ffs(static_cast<int>(empty_ballot))) - 1u;
				unsigned long long ol = 0ull, oh = 0ull; int won = 0;
				if (lane == e)
					won = tm_atomic_cas128(reinterpret_cast<unsigned long long*>(&table[slot]),
						0ull, 0ull, h0[j], h1[j], ol, oh) ? 1 : 0;
				won = __shfl_sync(0xFFFFFFFFu, won, e);
				ol  = __shfl_sync(0xFFFFFFFFu, ol, e);
				oh  = __shfl_sync(0xFFFFFFFFu, oh, e);
				if (won)
				{
					if (lane == 0u)
					{
						const uint32_t u = tm_map1_frontier_emit_rep(data_j[j], cand_base + j,
							unique_counter, rep_out, rep_cap, unique_out, rep_mult, rep_mult_cap, rep_idx_by_cand);
						tm_map1_frontier_publish_owner(table_owner, bucket * 32u + e, u);
					}
					done = true; break;
				}
				if (ol == h0[j] && oh == h1[j])
				{
					if (lane == 0u)
						tm_map1_frontier_merge_mult(table_owner, bucket * 32u + e, rep_mult, rep_mult_cap);
					done = true;
					break;
				}
			}
		}
		if (!done && lane == 0u)
		{
			atomicAdd(overflow_counter, 1u);
			tm_map1_frontier_emit_rep(data_j[j], cand_base + j, unique_counter,
				rep_out, rep_cap, unique_out, rep_mult, rep_mult_cap, rep_idx_by_cand);
		}
	}
}

extern "C" __global__ __launch_bounds__(128) void tm_map1_frontier_insert128_ilp2_cuda(
	ulonglong2* table, uint32_t logm,
	uint32_t* unique_counter, uint32_t* overflow_counter, uint32_t* rep_out, uint32_t rep_cap,
	uint8_t* unique_out, const uint8_t* partition_labels, uint32_t target_partition,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count, uint32_t tau,
	uint32_t unused_map_depth, uint32_t unused_sample_mult,
	uint32_t* table_owner, uint32_t* rep_mult, uint32_t rep_mult_cap, uint32_t* rep_idx_by_cand)
{
	tm_map1_frontier_insert128_ilp_core<2u>(table, logm, unique_counter, overflow_counter, rep_out, rep_cap,
		unique_out, partition_labels, target_partition, regular_rng_values, alg0_values, alg6_values,
		rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values, schedule_data,
		key, data_start, candidate_count, tau, unused_map_depth, unused_sample_mult,
		table_owner, rep_mult, rep_mult_cap, rep_idx_by_cand);
}

extern "C" __global__ __launch_bounds__(128) void tm_map1_frontier_insert128_ilp4_cuda(
	ulonglong2* table, uint32_t logm,
	uint32_t* unique_counter, uint32_t* overflow_counter, uint32_t* rep_out, uint32_t rep_cap,
	uint8_t* unique_out, const uint8_t* partition_labels, uint32_t target_partition,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count, uint32_t tau,
	uint32_t unused_map_depth, uint32_t unused_sample_mult,
	uint32_t* table_owner, uint32_t* rep_mult, uint32_t rep_mult_cap, uint32_t* rep_idx_by_cand)
{
	tm_map1_frontier_insert128_ilp_core<4u>(table, logm, unique_counter, overflow_counter, rep_out, rep_cap,
		unique_out, partition_labels, target_partition, regular_rng_values, alg0_values, alg6_values,
		rng_forward_1, rng_forward_128, alg2_values, alg5_values, expansion_values, schedule_data,
		key, data_start, candidate_count, tau, unused_map_depth, unused_sample_mult,
		table_owner, rep_mult, rep_mult_cap, rep_idx_by_cand);
}

// Depth-K MAP1 frontier insert (2026-06-04): same bucketed 128-bit insert, but
// runs the first `map_depth` schedule maps before fingerprinting, so the table
// dedups at the MAP-K reachable-set frontier (R_K, F_K) instead of MAP1 (R_1, F_1).
// Naive: no inter-map dedup, so maps 2..K run on all W candidates incl. duplicates.
// Measures the collapse-vs-depth curve and the per-depth producer cost. map_depth=1
// reproduces tm_map1_frontier_insert128_cuda exactly.
extern "C" __global__ __launch_bounds__(128) void tm_map1_frontier_insertK128_cuda(
	ulonglong2* table, uint32_t logm,
	uint32_t* unique_counter, uint32_t* overflow_counter, uint32_t* rep_out, uint32_t rep_cap,
	uint8_t* unique_out,
	const uint8_t* partition_labels, uint32_t target_partition,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count, uint32_t tau, uint32_t map_depth, uint32_t sample_mult)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t cand = blockIdx.x * warps_per_block + warp_index;
	if (cand >= candidate_count) return;
	if (partition_labels != nullptr)
	{
		uint32_t selected = 0u;
		if (lane == 0u) selected = partition_labels[cand] == target_partition ? 1u : 0u;
		selected = __shfl_sync(0xFFFFFFFFu, selected, 0);
		if (selected == 0u) return;
	}

	// sample_mult=1 -> contiguous data axis (producer). An odd Weyl multiplier turns a
	// small candidate range into a low-discrepancy quasi-random sample across the full
	// 2^32 data-byte space (class spot-check): data = ((data_start+cand)*mult) mod 2^32.
	const uint32_t data = (data_start + cand) * sample_mult;
	uint32_t value = initialize_working_word(key, data, lane, expansion_values);
	for (uint32_t m = 0u; m < map_depth; m++)  // run the first map_depth schedule maps
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[m];
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
				regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}
	}

	uint64_t h0, h1;
	tm_strong128_state(value, h0, h1);

	const uint32_t logb = logm - 5u;
	const uint32_t bucket_mask = (logb >= 32u) ? 0xFFFFFFFFu : ((1u << logb) - 1u);
	const uint32_t home = static_cast<uint32_t>((h0 * 0x9E3779B97F4A7C15ull) >> (64u - logb));
	constexpr uint32_t PROBE_BUCKETS = 16u;

	bool done = false;
	for (uint32_t pb = 0u; pb < PROBE_BUCKETS && !done; ++pb)
	{
		const uint32_t bucket = (home + pb) & bucket_mask;
		const uint32_t slot = bucket * 32u + lane;
		for (;;)
		{
			const ulonglong2 cur = table[slot];
			if (__ballot_sync(0xFFFFFFFFu, cur.x == h0 && cur.y == h1)) { done = true; break; }
			const uint32_t empty_ballot = __ballot_sync(0xFFFFFFFFu, cur.x == 0ull && cur.y == 0ull);
			if (empty_ballot == 0u) break;
			const uint32_t e = static_cast<uint32_t>(__ffs(static_cast<int>(empty_ballot))) - 1u;
			unsigned long long ol = 0ull, oh = 0ull; int won = 0;
			if (lane == e)
				won = tm_atomic_cas128(reinterpret_cast<unsigned long long*>(&table[slot]),
					0ull, 0ull, h0, h1, ol, oh) ? 1 : 0;
			won = __shfl_sync(0xFFFFFFFFu, won, e);
			ol  = __shfl_sync(0xFFFFFFFFu, ol, e);
			oh  = __shfl_sync(0xFFFFFFFFu, oh, e);
			if (won)
			{
				if (lane == 0u)
				{
					const uint32_t u = atomicAdd(unique_counter, 1u);
					if (rep_out != nullptr && u < rep_cap) rep_out[u] = data;
					if (unique_out != nullptr) unique_out[cand] = 1u;
				}
				done = true; break;
			}
			if (ol == h0 && oh == h1) { done = true; break; }
		}
	}
	if (!done && lane == 0u)
	{
		atomicAdd(overflow_counter, 1u);
		const uint32_t u = atomicAdd(unique_counter, 1u);
		if (rep_out != nullptr && u < rep_cap) rep_out[u] = data;
		if (unique_out != nullptr) unique_out[cand] = 1u;
	}
}

// Arbitrary selected-bit MAP-K frontier insert. This is a research probe for
// non-contiguous data windows: `selected_mask` marks the varying data bits, and
// `fixed_value` supplies every non-selected bit for the sampled group.
extern "C" __global__ __launch_bounds__(128) void tm_map1_frontier_insert_mask128_cuda(
	ulonglong2* table, uint32_t logm,
	uint32_t* unique_counter, uint32_t* overflow_counter, uint32_t* rep_out, uint32_t rep_cap,
	uint8_t* unique_out,
	const uint8_t* partition_labels, uint32_t target_partition,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t fixed_value, uint32_t candidate_start, uint32_t candidate_count, uint32_t map_depth, uint32_t selected_mask)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t cand = blockIdx.x * warps_per_block + warp_index;
	if (cand >= candidate_count) return;
	if (partition_labels != nullptr)
	{
		uint32_t selected = 0u;
		if (lane == 0u) selected = partition_labels[cand] == target_partition ? 1u : 0u;
		selected = __shfl_sync(0xFFFFFFFFu, selected, 0);
		if (selected == 0u) return;
	}

	const uint32_t data = fixed_value | tm_deposit_bits32(candidate_start + cand, selected_mask);
	uint32_t value = initialize_working_word(key, data, lane, expansion_values);
	for (uint32_t m = 0u; m < map_depth; m++)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[m];
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
				regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}
	}

	uint64_t h0, h1;
	tm_strong128_state(value, h0, h1);

	const uint32_t logb = logm - 5u;
	const uint32_t bucket_mask = (logb >= 32u) ? 0xFFFFFFFFu : ((1u << logb) - 1u);
	const uint32_t home = static_cast<uint32_t>((h0 * 0x9E3779B97F4A7C15ull) >> (64u - logb));
	constexpr uint32_t PROBE_BUCKETS = 16u;

	bool done = false;
	for (uint32_t pb = 0u; pb < PROBE_BUCKETS && !done; ++pb)
	{
		const uint32_t bucket = (home + pb) & bucket_mask;
		const uint32_t slot = bucket * 32u + lane;
		for (;;)
		{
			const ulonglong2 cur = table[slot];
			if (__ballot_sync(0xFFFFFFFFu, cur.x == h0 && cur.y == h1)) { done = true; break; }
			const uint32_t empty_ballot = __ballot_sync(0xFFFFFFFFu, cur.x == 0ull && cur.y == 0ull);
			if (empty_ballot == 0u) break;
			const uint32_t e = static_cast<uint32_t>(__ffs(static_cast<int>(empty_ballot))) - 1u;
			unsigned long long ol = 0ull, oh = 0ull; int won = 0;
			if (lane == e)
				won = tm_atomic_cas128(reinterpret_cast<unsigned long long*>(&table[slot]),
					0ull, 0ull, h0, h1, ol, oh) ? 1 : 0;
			won = __shfl_sync(0xFFFFFFFFu, won, e);
			ol  = __shfl_sync(0xFFFFFFFFu, ol, e);
			oh  = __shfl_sync(0xFFFFFFFFu, oh, e);
			if (won)
			{
				if (lane == 0u)
				{
					const uint32_t u = atomicAdd(unique_counter, 1u);
					if (rep_out != nullptr && u < rep_cap) rep_out[u] = data;
					if (unique_out != nullptr) unique_out[cand] = 1u;
				}
				done = true; break;
			}
			if (ol == h0 && oh == h1) { done = true; break; }
		}
	}
	if (!done && lane == 0u)
	{
		atomicAdd(overflow_counter, 1u);
		const uint32_t u = atomicAdd(unique_counter, 1u);
		if (rep_out != nullptr && u < rep_cap) rep_out[u] = data;
		if (unique_out != nullptr) unique_out[cand] = 1u;
	}
}

// Compact MAP1-new candidates from one streamed input chunk into absolute data
// values. Output order is stable within each 256-candidate block, preserving the
// local data-axis neighborhoods needed by the post-MAP1 within-block dedup path.
extern "C" __global__ __launch_bounds__(256) void tm_map1_frontier_compact_data_cuda(
	const uint8_t* unique_in, uint32_t candidate_count, uint32_t data_start,
	uint32_t* rep_out, uint32_t* counter)
{
	__shared__ uint32_t s_scan[256];
	__shared__ uint32_t s_base;
	const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t a = (tid < candidate_count && unique_in[tid] != 0u) ? 1u : 0u;
	s_scan[threadIdx.x] = a;
	__syncthreads();
	for (uint32_t off = 1u; off < blockDim.x; off <<= 1)
	{
		uint32_t v = (threadIdx.x >= off) ? s_scan[threadIdx.x - off] : 0u;
		__syncthreads();
		s_scan[threadIdx.x] += v;
		__syncthreads();
	}
	if (threadIdx.x == blockDim.x - 1u) s_base = atomicAdd(counter, s_scan[threadIdx.x]);
	__syncthreads();
	if (a)
	{
		const uint32_t pos = s_base + s_scan[threadIdx.x] - 1u;
		rep_out[pos] = data_start + tid;
	}
}

extern "C" __global__ __launch_bounds__(256) void tm_map1_frontier_compact_data_mult_cuda(
	const uint8_t* unique_in, uint32_t candidate_count, uint32_t data_start,
	const uint32_t* rep_idx_by_cand, const uint32_t* rep_mult, uint32_t rep_mult_cap,
	uint32_t* rep_out, uint32_t* mult_out, uint32_t* counter)
{
	__shared__ uint32_t s_scan[256];
	__shared__ uint32_t s_base;
	const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t a = (tid < candidate_count && unique_in[tid] != 0u) ? 1u : 0u;
	s_scan[threadIdx.x] = a;
	__syncthreads();
	for (uint32_t off = 1u; off < blockDim.x; off <<= 1)
	{
		uint32_t v = (threadIdx.x >= off) ? s_scan[threadIdx.x - off] : 0u;
		__syncthreads();
		s_scan[threadIdx.x] += v;
		__syncthreads();
	}
	if (threadIdx.x == blockDim.x - 1u) s_base = atomicAdd(counter, s_scan[threadIdx.x]);
	__syncthreads();
	if (a)
	{
		const uint32_t pos = s_base + s_scan[threadIdx.x] - 1u;
		rep_out[pos] = data_start + tid;
		uint32_t m = 1u;
		if (rep_idx_by_cand != nullptr && rep_mult != nullptr)
		{
			const uint32_t rep_idx = rep_idx_by_cand[tid];
			if (rep_idx < rep_mult_cap) m = rep_mult[rep_idx];
		}
		mult_out[pos] = m;
	}
}

extern "C" __global__ __launch_bounds__(256) void tm_wide_merge_hash_collapse_cuda(
	const uint64_t* fps, uint32_t n,
	uint64_t* table_fp, volatile uint32_t* table_rep, uint32_t logm,
	uint8_t* survivor_flag, uint32_t* rep_out, uint32_t* unique_counter)
{
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const uint64_t fp = fps[i];
	const uint32_t mask = (1u << logm) - 1u;
	uint32_t slot = static_cast<uint32_t>((fp * 0x9E3779B97F4A7C15ull) >> (64 - logm));

	// Open addressing, linear probe. Bounded by the table size.
	for (uint32_t probe = 0u; probe <= mask; ++probe)
	{
		const uint32_t s = (slot + probe) & mask;
		const uint64_t old = atomicCAS(reinterpret_cast<unsigned long long*>(&table_fp[s]),
			0ull, static_cast<unsigned long long>(fp));
		if (old == 0ull)
		{
			// Won the slot: I'm the run representative.
			table_rep[s] = i;
			__threadfence();
			survivor_flag[i] = 1u;
			rep_out[i] = i;
			atomicAdd(unique_counter, 1u);
			return;
		}
		if (old == fp)
		{
			// Duplicate of an existing run: spin until the owner publishes its rep.
			uint32_t r;
			do { r = table_rep[s]; } while (r == 0xFFFFFFFFu);
			survivor_flag[i] = 0u;
			rep_out[i] = r;
			return;
		}
		// Different fingerprint in this slot — probe the next.
	}
	// Table full (should not happen with load factor < 1): mark as own survivor.
	survivor_flag[i] = 1u;
	rep_out[i] = i;
	atomicAdd(unique_counter, 1u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Downstream survivor-screen (2026-06-03). The third stage of the wide-merge
// pipeline: after collapse, run the FULL canonical schedule + screen on the U
// survivor reps only (re-running maps 0..K is negligible for small K). One
// survivor per warp. flag_out[u] = screen result for survivor_idx[u]. All
// candidates sharing that rep's map-K state share this screen result (broadcast via
// the collapse's rep map), so screening U << N reps reproduces the N-candidate
// screen — that's the dedup win.
extern "C" __global__ __launch_bounds__(128) void tm_wide_merge_survivor_screen_cuda(
	const uint32_t* survivor_idx, uint32_t num_survivors,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data, const uint8_t* carnival_data,
	uint32_t key, uint32_t data_start, uint8_t* flag_out)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
	if (warp >= num_survivors) return;
	const uint32_t data = data_start + survivor_idx[warp];

	uint32_t value = initialize_working_word(key, data, lane, expansion_values);
	for (uint32_t schedule_index = 0u; schedule_index < 27u; schedule_index++)
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
				regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}
	}
	const uint8_t f = screen_candidate(value, lane, carnival_data);
	if (lane == 0u) flag_out[warp] = f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Production-fast indexed survivor screen (2026-06-03). ILP6, 1 cand/warp×ILP,
// indexed by survivor_idx[] — the wide-merge downstream. Uses the CANONICAL
// run_alg path (rng_seed + seed_forward), MATCHING the fp-dump's grouping map so
// every candidate in a rep's group screens identically (offset path is a DIFFERENT
// forward representation and is NOT valid to mix with canonical grouping). ILP
// hides the seed_forward load-use latency that makes the 1-warp canonical slow.
// result_data[u] = screen flag for survivor_idx[u]. Trailing ILP slots clamp to
// the last survivor (their stores are guarded off).
extern "C" __global__ __launch_bounds__(128) void tm_wide_merge_survivor_screen_ilp6_cuda(
	const uint32_t* survivor_idx, uint32_t num_survivors,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data, const uint8_t* carnival_data,
	uint32_t key, uint32_t data_start, uint8_t* result_data)
{
	constexpr uint32_t ILP = 6u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * ILP;
	if (candidate_base >= num_survivors) return;

	uint32_t working_value[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t slot = candidate_base + j;
		const uint32_t si = survivor_idx[slot < num_survivors ? slot : (num_survivors - 1u)];
		working_value[j] = initialize_working_word(key, data_start + si, lane, expansion_values);
	}

	for (uint32_t schedule_index = 0; schedule_index < 27u; schedule_index++)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
		uint16_t rng_seed[ILP];
		const uint16_t seed0 = static_cast<uint16_t>(((packed_schedule & 0x000000FFu) << 8)
			| ((packed_schedule & 0x0000FF00u) >> 8));
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) rng_seed[j] = seed0;
		uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8)
			| ((packed_schedule & 0xFF000000u) >> 24));
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
				working_value[j] = run_alg(working_value[j], lane, algorithm_id[j], &rng_seed[j],
					regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			}
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}
	}

	uint8_t screen_flags[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++) screen_flags[j] = screen_candidate(working_value[j], lane, carnival_data);
	if (lane == 0u) store_screen_flags_ilp<ILP>(result_data, candidate_base, num_survivors, screen_flags);
}

// ─────────────────────────────────────────────────────────────────────────────
// Indexed OFFSET survivor screen (2026-06-03). Same as the canonical ILP6 variant
// but uses run_alg_offset + OFFSET-STREAM tables (off_regular/alg0/alg6/alg2/alg5,
// built by build_offset_stream_blob) — the production-fast path (~130 vs canonical
// ~100 M/s). Valid because offset+correct-streams is BIT-IDENTICAL to canonical
// (verified: ilp6_preids vs canonical bytes_differ=0), so it both screens correctly
// AND is consistent with the canonical fp-dump grouping. result_data[u] = flag for
// survivor_idx[u]; trailing ILP slots clamp to the last survivor.
extern "C" __global__ __launch_bounds__(128) void tm_wide_merge_survivor_screen_offset_ilp6_cuda(
	const uint32_t* survivor_idx, uint32_t num_survivors,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data, const uint8_t* carnival_data,
	uint32_t key, uint32_t data_start, uint8_t* result_data)
{
	constexpr uint32_t ILP = 6u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * ILP;
	if (candidate_base >= num_survivors) return;

	uint32_t working_value[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t slot = candidate_base + j;
		const uint32_t si = survivor_idx[slot < num_survivors ? slot : (num_survivors - 1u)];
		working_value[j] = initialize_working_word(key, data_start + si, lane, expansion_values);
	}

	for (uint32_t schedule_index = 0; schedule_index < 27u; schedule_index++)
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
	for (uint32_t j = 0u; j < ILP; j++) screen_flags[j] = screen_candidate(working_value[j], lane, carnival_data);
	if (lane == 0u) store_screen_flags_ilp<ILP>(result_data, candidate_base, num_survivors, screen_flags);
}

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostic (2026-06-03): dump the full 32-word state AND the per-step alg-id
// sequence after `first_maps` schedule entries, for variance analysis across data.
// Answers: which of the 32 state words are data-influenced after MAP1, and is the
// alg-dispatch sequence data-invariant (key-derived) or data-dependent. One
// candidate per warp; writes state_out[cand*32+lane] and alg_out[cand*16+step].
extern "C" __global__ void tm_wide_merge_state_alg_dump_cuda(
	uint32_t* state_out, uint8_t* alg_out,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count, uint32_t first_maps)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t cand = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
	if (cand >= candidate_count) return;
	uint32_t value = initialize_working_word(key, data_start + cand, lane, expansion_values);
	for (uint32_t schedule_index = 0u; schedule_index < first_maps; schedule_index++)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
		uint16_t rng_seed = static_cast<uint16_t>(((packed_schedule & 0x000000FFu) << 8) | ((packed_schedule & 0x0000FF00u) >> 8));
		uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));
		for (uint32_t i = 0u; i < 16u; i++)
		{
			const uint32_t source_lane = i >> 2;
			const uint32_t source_shift = (i & 3u) * 8u;
			const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, value, source_lane);
			uint8_t current_byte = static_cast<uint8_t>((source_word >> source_shift) & 0xFFu);
			if ((nibble_selector & 0x8000u) != 0u) current_byte = static_cast<uint8_t>(current_byte >> 4);
			const uint8_t algorithm_id = static_cast<uint8_t>((current_byte >> 1) & 0x07u);
			if (lane == 0u && schedule_index == 0u) alg_out[cand * 16u + i] = algorithm_id;
			value = run_alg(value, lane, algorithm_id, &rng_seed,
				regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}
	}
	state_out[cand * 32u + lane] = value;
}

// ─────────────────────────────────────────────────────────────────────────────
// Strong-hash fp-dump (2026-06-03): identical state walk to tm_wide_merge_fp_dump_n
// but fingerprints with a STRONG avalanched hash (rotate-multiply fold on lane 0)
// instead of warp_hash_state (murmur+XOR-butterfly). Comparing unique counts weak
// vs strong at scale tells us whether warp_hash false-merges structured post-MAP1
// states (CPU §45: their FNV+murmur was ~35 effective bits → drops states at W4B).
extern "C" __global__ __launch_bounds__(128) void tm_wide_merge_fp_dump_strong_cuda(
	uint64_t* fp_out,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count, uint32_t first_maps)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t cand = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
	if (cand >= candidate_count) return;
	uint32_t value = initialize_working_word(key, data_start + cand, lane, expansion_values);
	for (uint32_t schedule_index = 0u; schedule_index < first_maps; schedule_index++)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
		uint16_t rng_seed = static_cast<uint16_t>(((packed_schedule & 0x000000FFu) << 8) | ((packed_schedule & 0x0000FF00u) >> 8));
		uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));
		for (uint32_t i = 0u; i < 16u; i++)
		{
			const uint32_t source_lane = i >> 2;
			const uint32_t source_shift = (i & 3u) * 8u;
			const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, value, source_lane);
			uint8_t current_byte = static_cast<uint8_t>((source_word >> source_shift) & 0xFFu);
			if ((nibble_selector & 0x8000u) != 0u) current_byte = static_cast<uint8_t>(current_byte >> 4);
			const uint8_t algorithm_id = static_cast<uint8_t>((current_byte >> 1) & 0x07u);
			value = run_alg(value, lane, algorithm_id, &rng_seed,
				regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}
	}
	// Strong fold over all 32 lane-words (rotate-multiply, full avalanche).
	uint64_t hs = 0x9E3779B97F4A7C15ull;
	#pragma unroll
	for (uint32_t L = 0u; L < 32u; L++)
	{
		const uint32_t v = __shfl_sync(0xFFFFFFFFu, value, L);
		hs ^= static_cast<uint64_t>(v);
		hs = ((hs << 23) | (hs >> 41)) * 0xD1B54A32D192ED03ull;
		hs ^= hs >> 29;
	}
	if (lane == 0u) { if (hs == 0ull) hs = 1ull; fp_out[cand] = hs; }
}

// ─────────────────────────────────────────────────────────────────────────────
// Indexed fp-dump (2026-06-03): like tm_wide_merge_fp_dump_n but the candidate's
// data value is survivor_idx[cand] (a subset), for PERIODIC multi-merge re-derive.
// Canonical run_alg (matches grouping). One candidate per warp; lane 0 writes fp.
extern "C" __global__ __launch_bounds__(128) void tm_wide_merge_fp_dump_indexed_cuda(
	uint64_t* fp_out, const uint32_t* survivor_idx,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint16_t* rng_forward_1, const uint16_t* rng_forward_128,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count, uint32_t first_maps)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t cand = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
	if (cand >= candidate_count) return;
	uint32_t value = initialize_working_word(key, data_start + survivor_idx[cand], lane, expansion_values);
	for (uint32_t schedule_index = 0u; schedule_index < first_maps; schedule_index++)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
		uint16_t rng_seed = static_cast<uint16_t>(((packed_schedule & 0x000000FFu) << 8) | ((packed_schedule & 0x0000FF00u) >> 8));
		uint16_t nibble_selector = static_cast<uint16_t>(((packed_schedule & 0x00FF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));
		for (uint32_t i = 0u; i < 16u; i++)
		{
			const uint32_t source_lane = i >> 2;
			const uint32_t source_shift = (i & 3u) * 8u;
			const uint32_t source_word = __shfl_sync(0xFFFFFFFFu, value, source_lane);
			uint8_t current_byte = static_cast<uint8_t>((source_word >> source_shift) & 0xFFu);
			if ((nibble_selector & 0x8000u) != 0u) current_byte = static_cast<uint8_t>(current_byte >> 4);
			const uint8_t algorithm_id = static_cast<uint8_t>((current_byte >> 1) & 0x07u);
			value = run_alg(value, lane, algorithm_id, &rng_seed,
				regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}
	}
	uint32_t h_lo, h_hi; warp_hash_state(value, lane, &h_lo, &h_hi);
	if (lane == 0u) { uint64_t fp=((uint64_t)h_hi<<32)|h_lo; if(fp==0ull)fp=1ull; fp_out[cand]=fp; }
}

// ────────────────────────────────────────────────────────────────────────────
// WS1 drop-drain POC — a dedup CHECKPOINT at map DRAIN_AT inside the production
// ILP6 offset-stream screen. At the checkpoint each candidate's post-map-DRAIN_AT
// state is fingerprinted (tm_strong64_state) and probed against a dedup table; a
// resident-hit marks the candidate dead (alive=0), so it skips run_alg_offset for
// every remaining map — saving (SCHEDULE_COUNT-DRAIN_AT) map applications per
// dropped duplicate (the GPU's dominant, RNG-table-bandwidth cost).
//
//   MODE 1 = WITHIN-BLOCK shared table (catches only block-local dups)
//   MODE 2 = GLOBAL VRAM table        (catches cross-block dups too — the reach
//                                       that tests the GPU-bandwidth reframe)
//
// Table is EXACT open-addressing (atomicCAS): a drop fires only on a true strong64
// match, so the dropped candidate has the resident representative's exact state →
// same final state/flag → FN-safe BY CONSTRUCTION (the rep runs to completion and
// carries the screen hit). Dropped candidates emit flag 0; the survivor SET is
// preserved via the representative. `table_fp`/`logm` unused in MODE 1 (pass 0).
// `drop_count` (global) tallies drops for the drop-rate / saved-map readout.
//
// This is the exact-table BENEFIT CEILING. The probe is isolated in drain_probe()
// so an inverse-bloom cap (flat memory, FN-only) can replace it later (the memory
// axis) without touching the kernel body.
template<uint32_t MODE>
__device__ __forceinline__ bool drain_probe(uint64_t fp, uint64_t* table_fp, uint32_t logm,
                                            unsigned long long* blk_fp, uint32_t blk_bits)
{
	// lane-0-only caller. Returns true => fp already resident (this is a duplicate).
	if (fp == 0ull) fp = 1ull;
	if (MODE == 1u)
	{
		const uint32_t bmask = (1u << blk_bits) - 1u;
		uint32_t slot = (uint32_t)((fp * 0x9E3779B97F4A7C15ull) >> (64 - blk_bits));
		for (uint32_t p = 0u; p <= bmask; ++p)
		{
			const uint32_t s = (slot + p) & bmask;
			const unsigned long long old = atomicCAS(&blk_fp[s], 0ull, (unsigned long long)fp);
			if (old == 0ull) return false;                 // claimed an empty slot → first occurrence
			if (old == (unsigned long long)fp) return true; // resident → duplicate
		}
		return false;                                       // table full → treat as new (safe over-keep)
	}
	else
	{
		const uint32_t tmask = (1u << logm) - 1u;
		uint32_t slot = (uint32_t)((fp * 0x9E3779B97F4A7C15ull) >> (64 - logm));
		constexpr uint32_t PROBE_CAP = 64u;
		const uint32_t cap = (tmask < PROBE_CAP) ? (tmask + 1u) : PROBE_CAP;
		for (uint32_t p = 0u; p < cap; ++p)
		{
			const uint32_t s = (slot + p) & tmask;
			const unsigned long long old = atomicCAS(reinterpret_cast<unsigned long long*>(&table_fp[s]),
				0ull, (unsigned long long)fp);
			if (old == 0ull) return false;
			if (old == (unsigned long long)fp) return true;
		}
		return false;                                       // probe-cap exhausted → treat as new (safe)
	}
}

template<uint32_t SCHEDULE_COUNT, uint32_t ILP, uint32_t DRAIN_AT, uint32_t MODE>
__device__ __forceinline__ void tm_screen_drain_impl(
	uint8_t* result_data,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data, const uint8_t* carnival_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count,
	uint64_t* table_fp, uint32_t logm, unsigned long long* drop_count)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t warps_per_block = blockDim.x >> 5;
	const uint32_t candidate_base = (blockIdx.x * warps_per_block + warp_index) * ILP;

	// MODE 1: a per-block shared table sized comfortably above the block's
	// candidate count (warps_per_block*ILP <= 24 at 128t/ILP6) → low load factor.
	constexpr uint32_t BLK_BITS = 7u;        // 128 slots
	constexpr uint32_t BLK_SLOTS = 1u << BLK_BITS;
	__shared__ unsigned long long blk_fp[(MODE == 1u) ? BLK_SLOTS : 1u];
	if (MODE == 1u)
	{
		for (uint32_t s = threadIdx.x; s < BLK_SLOTS; s += blockDim.x) blk_fp[s] = 0ull;
		__syncthreads();
	}

	if (candidate_base >= candidate_count) return;

	uint32_t working_value[ILP];
	bool alive[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		working_value[j] = initialize_working_word(key, data_start + candidate_base + j, lane, expansion_values);
		alive[j] = true;
	}

	#pragma unroll 1
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
				if (alive[j])
					working_value[j] = run_alg_offset(working_value[j], lane, algorithm_id[j], schedule_index, &rng_offset[j],
						regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			}
			nibble_selector = static_cast<uint16_t>(nibble_selector << 1);
		}

		// Checkpoint AFTER map DRAIN_AT (schedule_index == DRAIN_AT-1 just completed).
		if (schedule_index == (DRAIN_AT - 1u))
		{
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				const uint64_t fp = tm_strong64_state(working_value[j], lane);  // warp-cooperative; lane0 = full fp
				bool drop = false;
				if (lane == 0u)
				{
					drop = drain_probe<MODE>(fp, table_fp, logm, blk_fp, BLK_BITS);
					if (drop) atomicAdd(drop_count, 1ull);
				}
				drop = __shfl_sync(0xFFFFFFFFu, drop, 0);
				if (drop) alive[j] = false;
			}
		}
	}

	uint8_t screen_flags[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
		screen_flags[j] = alive[j] ? screen_candidate(working_value[j], lane, carnival_data) : (uint8_t)0u;

	if (lane == 0u)
		store_screen_flags_ilp<ILP>(result_data, candidate_base, candidate_count, screen_flags);
}

extern "C" __global__ __launch_bounds__(128) void tm_screen_drain_block_map6_ilp6_cuda(
	uint8_t* result_data,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data, const uint8_t* carnival_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count,
	uint64_t* table_fp, uint32_t logm, unsigned long long* drop_count)
{
	tm_screen_drain_impl<27u, 6u, 6u, 1u>(result_data, regular_rng_values, alg0_values, alg6_values,
		alg2_values, alg5_values, expansion_values, schedule_data, carnival_data,
		key, data_start, candidate_count, table_fp, logm, drop_count);
}

extern "C" __global__ __launch_bounds__(128) void tm_screen_drain_global_map6_ilp6_cuda(
	uint8_t* result_data,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data, const uint8_t* carnival_data,
	uint32_t key, uint32_t data_start, uint32_t candidate_count,
	uint64_t* table_fp, uint32_t logm, unsigned long long* drop_count)
{
	tm_screen_drain_impl<27u, 6u, 6u, 2u>(result_data, regular_rng_values, alg0_values, alg6_values,
		alg2_values, alg5_values, expansion_values, schedule_data, carnival_data,
		key, data_start, candidate_count, table_fp, logm, drop_count);
}

// ────────────────────────────────────────────────────────────────────────────
// WS1 Path B — GLOBAL-table drain as a span dedup at a deep boundary, INSIDE the
// re-densifying compaction pipeline (the contextually-faithful measurement). This
// is run_span_dedup_impl's continue-kernel (reads compacted survivor state, runs
// maps [m0,m1)) but its span-end dedup is a GLOBAL VRAM table (cross-block reach)
// instead of the within-block shared table — i.e. the drain catches the post-MAP1
// residual the within-block stage (W<=96/block) misses, and compaction then packs
// the survivors so the drops convert to throughput. mode is always 1 (dedup): a
// deep drain boundary is never the final span. table_rep volatile for the publish
// spin. Hash = warp_hash_state to match the span-0 global merge (apples-to-apples);
// FN-safety of the whole pipeline is checked by final-survivor-count == drain-off.
template<uint32_t WARPS, uint32_t ILP>
__device__ __forceinline__ void run_span_dedup_drain_impl(
	const uint32_t* live_idx, uint32_t M,
	uint32_t* state, uint8_t* alive_out, uint32_t* rep_global, uint32_t* mult,
	uint32_t m0, uint32_t m1, int first_span,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start,
	uint64_t* table_fp, volatile uint32_t* table_rep, uint32_t logm)
{
	constexpr uint32_t W = WARPS * ILP;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t cb = warp_index * ILP;
	const uint32_t block_base = blockIdx.x * W;
	const uint32_t tmask = (1u << logm) - 1u;

	__shared__ uint8_t  alive_local[W];
	__shared__ uint32_t orig_of[W];
	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			const uint32_t s = block_base + cb + j;
			const bool active = s < M;
			orig_of[cb + j]     = active ? (first_span ? s : live_idx[s]) : 0xFFFFFFFFu;
			alive_local[cb + j] = active ? 1u : 0u;
		}
	}
	__syncthreads();

	uint32_t value[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t orig = orig_of[cb + j];
		value[j] = alive_local[cb + j]
			? (first_span ? initialize_working_word(key, data_start + orig, lane, expansion_values)
			              : state[(size_t)orig * 32u + lane])
			: 0u;
	}

	for (uint32_t schedule_index = m0; schedule_index < m1; schedule_index++)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
		uint32_t ro[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) ro[j] = 0u;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));
		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const bool high = (nsel & 0x8000u) != 0u;
			uint8_t aid[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				const uint32_t w = __shfl_sync(0xFFFFFFFFu, value[j], sl);
				uint8_t cbj = static_cast<uint8_t>((w >> ss) & 0xFFu);
				if (high) cbj >>= 4;
				aid[j] = static_cast<uint8_t>((cbj >> 1) & 0x07u);
			}
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
				value[j] = run_alg_offset_sw(value[j], lane, aid[j], schedule_index, &ro[j], regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			nsel = static_cast<uint16_t>(nsel << 1);
		}
	}

	// span-end GLOBAL dedup (the drain): cross-block reach via the VRAM table.
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		uint32_t h_lo, h_hi;
		warp_hash_state(value[j], lane, &h_lo, &h_hi);   // all lanes
		if (alive_local[cb + j] && lane == 0u)
		{
			uint64_t fp = ((uint64_t)h_hi << 32) | (uint64_t)h_lo;
			if (fp == 0ull) fp = 1ull;
			uint32_t slot0 = (uint32_t)((fp * 0x9E3779B97F4A7C15ull) >> (64 - logm));
			constexpr uint32_t PROBE_CAP = 64u;
			const uint32_t cap = (tmask < PROBE_CAP) ? (tmask + 1u) : PROBE_CAP;
			const uint32_t my_idx = cb + j;
			for (uint32_t probe = 0u; probe < cap; ++probe)
			{
				const uint32_t slt = (slot0 + probe) & tmask;
				const unsigned long long old = atomicCAS(reinterpret_cast<unsigned long long*>(&table_fp[slt]),
					0ull, (unsigned long long)fp);
				if (old == 0ull) { table_rep[slt] = orig_of[my_idx]; __threadfence(); break; }
				if (old == (unsigned long long)fp)
				{
					uint32_t r; do { r = table_rep[slt]; } while (r == 0xFFFFFFFFu);
					rep_global[orig_of[my_idx]] = r;
					tm_merge_multiplicity(mult, orig_of[my_idx], r);
					alive_local[my_idx] = 0u;
					break;
				}
			}
		}
	}
	__syncthreads();   // publish alive_local (lane-0 writes) to all lanes for writeback

	// writeback survivor state + alive/rep for compaction + resolve.
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t s = block_base + cb + j;
		if (s >= M) continue;
		const uint32_t orig = orig_of[cb + j];
		if (alive_local[cb + j])
		{
			state[(size_t)orig * 32u + lane] = value[j];
			if (lane == 0u) alive_out[s] = 1u;
		}
		else if (lane == 0u)
		{
			alive_out[s] = 0u;   // rep_global already set at the probe (winner's orig)
		}
	}
}

// ── inverse-bloom cap insert: the swappable CAS protocol ────────────────────────────
// Two FN-safe schemes share the same bucketed (2^cap_bits x cap_ways) epoch-tagged table.
// The MODE template selects which 64-bit/128-bit atomic publishes the (fp,rep) pair so we
// can A/B whether the 128-bit width (not the protocol) is the perf cost (owner hypothesis,
// 2026-06-11). Called by both the drain cap and the routed span-0 cap.
//
//   CAP_CAS128 — original: 16-byte slot {lo=fp, hi=epoch<<32|rep} published atomically via
//     tm_atomic_cas128. The (fp,rep) pair is always consistent (no two-word race) at the
//     cost of a heavier 128-bit atomic.
//   CAP_CAS64A — epoch-packed, 64-bit CAS, NO re-zero. word0 (the CAS/linearization word) =
//     (epoch16<<48)|fp48; word1 = (epoch32<<32)|rep. A single 64-bit atomicCAS on word0
//     claims a stale/empty slot (e0 != cur epoch) or detects a current-epoch dup (word0 ==
//     mine). The rep is published into word1 AFTER the claim and read with a spin that waits
//     for word1's epoch field to equal the current epoch — so a dup never reads a stale rep
//     (the publication race that a naive two-word swap would hit). Slot is written once per
//     epoch (monotonic stale/0 -> mine), so (fp48,rep) stay consistent. Discriminator =
//     cap_bits + 48 (>= 68 @bits>=20) >= the exact table's effective 64 -> FN-safe.
//     Requires (epoch & 0xFFFF) != 0 (host guarantees: epochs are small positive ints).
enum { CAP_CAS128 = 0, CAP_CAS64A = 1, CAP_CAS_XTILE = 2 };  // XTILE = cross-tile flag-carrying cap (see cap_probe_insert_xtile)

// Forward decl so cap_probe_insert<CAP_CAS_XTILE> can delegate to the flag-carrying xtile cap
// (defined below). Same 9-arg signature.
__device__ __forceinline__ void cap_probe_insert_xtile(
	unsigned long long fp, uint32_t myrep, uint32_t epoch,
	unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways,
	uint32_t* rep_global, uint32_t* mult, uint8_t* alive_slot);

template<int MODE>
__device__ __forceinline__ void cap_probe_insert(
	unsigned long long fp, uint32_t myrep, uint32_t epoch,
	unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways,
	uint32_t* rep_global, uint32_t* mult, uint8_t* alive_slot)   // *alive_slot set to 0 on drop
{
	// XTILE has a distinct 128-bit flag-carrying slot layout and its own probe. Callers that
	// instantiate cap_probe_insert<CAP_CAS_XTILE> (the trajgate routed deep kernels) must use the
	// xtile probe, NOT the CAS64A fallthrough below (which corrupts the slot / faults). Delegate.
	// FIX (2026-06-13): the trajroute/trajroute2 cap kernels called cap_probe_insert<MODE> without
	// dispatching to the xtile probe, so cross-tile + deep-routing hung on a GPU fault. (#6)
	if constexpr (MODE == CAP_CAS_XTILE)
	{
		cap_probe_insert_xtile(fp, myrep, epoch, cap_table, cap_bits, cap_ways, rep_global, mult, alive_slot);
		return;
	}
	const uint32_t bucket = (uint32_t)((fp * 0x9E3779B97F4A7C15ull) >> (64 - cap_bits));
	const unsigned long long base = (unsigned long long)bucket * cap_ways;
	if (MODE == CAP_CAS128)
	{
		const unsigned long long hi_mine = ((unsigned long long)epoch << 32) | (unsigned long long)myrep;
		for (uint32_t w = 0u; w < cap_ways; w++)
		{
			unsigned long long* slot = cap_table + (base + w) * 2ull;
			unsigned long long ol, oh;
			if (tm_atomic_cas128(slot, 0ull, 0ull, fp, hi_mine, ol, oh)) break;  // claimed empty → winner (alive)
			if (ol == fp && (uint32_t)(oh >> 32) == epoch)                         // dup (same fp, current epoch)
			{
				const uint32_t r = (uint32_t)(oh & 0xFFFFFFFFull);
				rep_global[myrep] = r;
				tm_merge_multiplicity(mult, myrep, r);
				*alive_slot = 0u; break;
			}
			if ((uint32_t)(oh >> 32) != epoch)                                     // stale slot → try to claim it
			{
				unsigned long long n_ol, n_oh;
				if (tm_atomic_cas128(slot, ol, oh, fp, hi_mine, n_ol, n_oh)) break; // claimed stale → winner
				if (n_ol == fp && (uint32_t)(n_oh >> 32) == epoch)                 // raced: my fp claimed → dup
				{
					const uint32_t r = (uint32_t)(n_oh & 0xFFFFFFFFull);
					rep_global[myrep] = r;
					tm_merge_multiplicity(mult, myrep, r);
					*alive_slot = 0u; break;
				}
			}
			// else: current-epoch different fp → next way; if none free, leave un-merged (alive) → over-keep
		}
	}
	else   // CAP_CAS64A
	{
		const uint32_t e16 = epoch & 0xFFFFu;
		unsigned long long fp48 = fp & 0xFFFFFFFFFFFFull;
		if (fp48 == 0ull) fp48 = 1ull;
		const unsigned long long w0_mine = ((unsigned long long)e16 << 48) | fp48;
		const unsigned long long w1_mine = ((unsigned long long)epoch << 32) | (unsigned long long)myrep;
		for (uint32_t w = 0u; w < cap_ways; )
		{
			volatile unsigned long long* s0 = (volatile unsigned long long*)(cap_table + (base + w) * 2ull);
			const unsigned long long cur0 = *s0;
			const uint32_t e0 = (uint32_t)(cur0 >> 48);
			if (e0 == e16)                                  // current-epoch occupant
			{
				if (cur0 == w0_mine)                        // same fp48 → dup
				{
					volatile unsigned long long* s1 = s0 + 1;
					unsigned long long r1;
					do { r1 = *s1; } while ((uint32_t)(r1 >> 32) != epoch);   // wait for this-epoch publish
					const uint32_t r = (uint32_t)(r1 & 0xFFFFFFFFull);
					rep_global[myrep] = r;
					tm_merge_multiplicity(mult, myrep, r);
					*alive_slot = 0u; return;
				}
				w++; continue;                              // different fp48, same epoch → next way
			}
			// stale/empty (e0 != e16) → claim word0 with a single 64-bit CAS
			const unsigned long long old = atomicCAS((unsigned long long*)s0, cur0, w0_mine);
			if (old == cur0)                                // won the slot
			{
				*(s0 + 1) = w1_mine; __threadfence(); return;   // publish rep+epoch; winner (alive)
			}
			// lost the race → re-read the SAME way (do not advance w)
		}
		// bucket full of current-epoch foreign fps → leave un-merged (alive) → bounded over-keep
	}
}

// ── CROSS-TILE flag-carrying cap (FN-safe cross-window drain) ────────────────────────
// The plain cross-tile cap stores a TILE-LOCAL rep index; across tiles resolve_flags then
// reads flag[stale_index] = garbage (lost/mis-flagged hits — the self-imposed FN bug). Fix:
// carry the rep's RESOLVED FLAG in the slot. 128-bit slot (word0=fp, word1 packed below),
// published atomically (no publish-spin race). word1 layout:
//   [63:48] epoch (boundary m1)   [47] resolved bit   [46:39] flag   [27:0] rep (orig index)
// A this-tile claim sets resolved=0 (only the rep index known mid-pipeline). At END of tile
// (after final flags exist) cap_resolve_xtile fills resolved=1 + flag=flag[rep] for every
// resolved=0 slot. NEXT tile: a dup hitting a resolved=1 slot (a PRIOR tile's rep) adopts the
// stored flag directly (rep_global = TM_XFLAG_MARKER|flag); a dup hitting a resolved=0 slot
// (this tile) uses the rep index as before. resolve_flags decodes the marker. FN-safe.
#define TM_XFLAG_MARKER 0xFE000000u
__device__ __forceinline__ void cap_probe_insert_xtile(
	unsigned long long fp, uint32_t myrep, uint32_t epoch,
	unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways,
	uint32_t* rep_global, uint32_t* mult, uint8_t* alive_slot)
{
	const unsigned long long e = (unsigned long long)(epoch & 0xFFFFu);
	const unsigned long long hi_mine = (e << 48) | (unsigned long long)(myrep & 0x0FFFFFFFu);  // resolved=0
	const uint32_t bucket = (uint32_t)((fp * 0x9E3779B97F4A7C15ull) >> (64 - cap_bits));
	const unsigned long long base = (unsigned long long)bucket * cap_ways;
	for (uint32_t w = 0u; w < cap_ways; w++)
	{
		unsigned long long* slot = cap_table + (base + w) * 2ull;
		unsigned long long ol, oh;
		if (tm_atomic_cas128(slot, 0ull, 0ull, fp, hi_mine, ol, oh)) break;   // claimed empty → winner
		const uint32_t oe = (uint32_t)(oh >> 48);
		if (ol == fp && oe == (uint32_t)e)                                    // same fp, same boundary epoch → dup
		{
			if ((oh >> 47) & 1ull)                                           // resolved (PRIOR tile) → adopt flag
				rep_global[myrep] = TM_XFLAG_MARKER | (uint32_t)((oh >> 39) & 0xFFull);
			else                                                             // unresolved (THIS tile) → rep index
			{
				const uint32_t r = (uint32_t)(oh & 0x0FFFFFFFull);
				rep_global[myrep] = r;
				tm_merge_multiplicity(mult, myrep, r);
			}
			*alive_slot = 0u; break;
		}
		if (oe != (uint32_t)e)                                               // stale (other boundary/tile) → reclaim
		{
			unsigned long long n_ol, n_oh;
			if (tm_atomic_cas128(slot, ol, oh, fp, hi_mine, n_ol, n_oh)) break;  // reclaimed → winner
			const uint32_t ne = (uint32_t)(n_oh >> 48);
			if (n_ol == fp && ne == (uint32_t)e)                            // raced: my fp claimed
			{
				if ((n_oh >> 47) & 1ull) rep_global[myrep] = TM_XFLAG_MARKER | (uint32_t)((n_oh >> 39) & 0xFFull);
				else
				{
					const uint32_t r = (uint32_t)(n_oh & 0x0FFFFFFFull);
					rep_global[myrep] = r;
					tm_merge_multiplicity(mult, myrep, r);
				}
				*alive_slot = 0u; break;
			}
		}
		// else current-epoch different fp → next way
	}
}

// End-of-tile: fill resolved=1 + flag for every THIS-TILE slot (resolved=0). After this, all
// occupied slots carry a final flag for the NEXT tile's cross-tile dups to adopt.
extern "C" __global__ void cap_resolve_xtile_cuda(
	unsigned long long* cap_table, uint64_t nslots, const uint8_t* flag)
{
	const uint64_t s = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (s >= nslots) return;
	const unsigned long long w0 = cap_table[s * 2ull];
	if (w0 == 0ull) return;                       // empty
	unsigned long long w1 = cap_table[s * 2ull + 1ull];
	if ((w1 >> 47) & 1ull) return;                // already resolved (prior tile)
	const uint32_t rep = (uint32_t)(w1 & 0x0FFFFFFFull);
	cap_table[s * 2ull + 1ull] = w1 | (1ull << 47) | ((unsigned long long)(flag[rep] & 0xFFu) << 39);
}

// ── span-HLL: walk maps [m0,m1) then HLL the state — estimates the frontier cardinality
// at depth m1 (e.g. m0=0,m1=1 → MAP1 frontier) WITHOUT a dedup table, for HLL-auto table
// sizing. No insert, no writeback: just the walk + a 4096-register HLL. Same walk as
// run_span_dedup_global_impl. (warp_hash_state is the same fp the merge dedups on.)
template<uint32_t WARPS, uint32_t ILP>
__device__ __forceinline__ void run_span_hll_impl(
	uint32_t M, uint32_t m0, uint32_t m1,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start, uint32_t* hll_registers, uint32_t tau)
{
	// tau>0 → ROUTED HLL: count shed (alg0/alg6 ops) over the span and HLL only shed>=tau
	// states ⇒ estimates the ROUTED cardinality (what a shed-routed cap would hold), for
	// sizing the routed cap to 2x(routed cardinality). tau==0 → HLL all (full frontier).
	constexpr uint32_t W = WARPS * ILP;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t cb = warp_index * ILP;
	const uint32_t block_base = blockIdx.x * W;

	uint32_t orig[ILP]; bool active[ILP]; uint32_t value[ILP]; uint32_t shed[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		orig[j]   = block_base + cb + j;
		active[j] = orig[j] < M;
		value[j]  = active[j] ? initialize_working_word(key, data_start + orig[j], lane, expansion_values) : 0u;
		shed[j]   = 0u;
	}

	for (uint32_t schedule_index = m0; schedule_index < m1; schedule_index++)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
		uint32_t ro[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) ro[j] = 0u;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));
		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const bool high = (nsel & 0x8000u) != 0u;
			uint8_t aid[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				const uint32_t w = __shfl_sync(0xFFFFFFFFu, value[j], sl);
				uint8_t cbj = static_cast<uint8_t>((w >> ss) & 0xFFu);
				if (high) cbj >>= 4;
				aid[j] = static_cast<uint8_t>((cbj >> 1) & 0x07u);
			}
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				if (aid[j] == 0u || aid[j] == 6u) shed[j]++;   // shed proxy (warp-uniform)
				value[j] = run_alg_offset_sw(value[j], lane, aid[j], schedule_index, &ro[j], regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			}
			nsel = static_cast<uint16_t>(nsel << 1);
		}
	}

	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		uint32_t h_lo, h_hi;
		warp_hash_state(value[j], lane, &h_lo, &h_hi);
		if (active[j] && lane == 0u && (tau == 0u || shed[j] >= tau)) hll_update(hll_registers, h_lo, h_hi);
	}
}

#define TM_SPAN_HLL_KERNEL(NAME, WARPS, ILP)                                                          \
	extern "C" __global__ void NAME(                                                                 \
		uint32_t M, uint32_t m0, uint32_t m1,                                                         \
		const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,    \
		const uint32_t* alg2_values, const uint32_t* alg5_values,                                     \
		const uint8_t* expansion_values, const uint8_t* schedule_data,                                \
		uint32_t key, uint32_t data_start, uint32_t* hll_registers, uint32_t tau)                     \
	{                                                                                                \
		run_span_hll_impl<(WARPS), (ILP)>(M, m0, m1, regular_rng_values, alg0_values, alg6_values,    \
			alg2_values, alg5_values, expansion_values, schedule_data, key, data_start, hll_registers, tau); \
	}
TM_SPAN_HLL_KERNEL(run_span_hll_w8i8_cuda, 8u, 8u)

// WS1 inverse-bloom drain cap: like run_span_dedup_drain_impl but the span-end dedup
// probes a FIXED-CAPACITY, CAP_WAYS-bucketed, EPOCH-tagged table instead of a 2x-tile
// open-addressing table. Wins vs the exact table on the small-VRAM target:
//   - flat footprint (2^cap_bits * cap_ways * 16 B), INDEPENDENT of tile/frontier;
//   - short bucket probe (cap_ways, e.g. 4) — no 64-deep chains, so it does NOT slow
//     down when overfull (the exact table's degradation; cap-size sweep 2026-06-10);
//   - EPOCH tag (slot.hi = epoch<<32 | rep) ⇒ NO per-boundary re-zero (bump a monotonic
//     counter; stale-epoch slots are claimable), table zeroed ONCE at alloc.
// 16-byte slots (lo=fp, hi=epoch:rep) written via atom.cas.b128 ⇒ the (fp,rep) pair is
// published atomically, so a dup reads a rep CONSISTENT with the matched fp (no two-word
// race) → FN-safe rep tracking. Bucket full of current-epoch non-matching fps ⇒ leave
// UN-MERGED (candidate stays alive = bounded over-keep, recovered by the multi-drain
// cadence). Never overwrites a current-epoch entry, so never a false drop.
template<uint32_t WARPS, uint32_t ILP, int MODE>
__device__ __forceinline__ void run_span_dedup_drain_cap_impl(
	const uint32_t* live_idx, uint32_t M,
	uint32_t* state, uint8_t* alive_out, uint32_t* rep_global, uint32_t* mult,
	uint32_t m0, uint32_t m1, int first_span,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start,
	unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways, uint32_t epoch)
{
	constexpr uint32_t W = WARPS * ILP;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t cb = warp_index * ILP;
	const uint32_t block_base = blockIdx.x * W;

	__shared__ uint8_t  alive_local[W];
	__shared__ uint32_t orig_of[W];
	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			const uint32_t s = block_base + cb + j;
			const bool active = s < M;
			orig_of[cb + j]     = active ? (first_span ? s : live_idx[s]) : 0xFFFFFFFFu;
			alive_local[cb + j] = active ? 1u : 0u;
		}
	}
	__syncthreads();

	uint32_t value[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t orig = orig_of[cb + j];
		value[j] = alive_local[cb + j]
			? (first_span ? initialize_working_word(key, data_start + orig, lane, expansion_values)
			              : state[(size_t)orig * 32u + lane])
			: 0u;
	}

	for (uint32_t schedule_index = m0; schedule_index < m1; schedule_index++)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
		uint32_t ro[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) ro[j] = 0u;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));
		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const bool high = (nsel & 0x8000u) != 0u;
			uint8_t aid[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				const uint32_t w = __shfl_sync(0xFFFFFFFFu, value[j], sl);
				uint8_t cbj = static_cast<uint8_t>((w >> ss) & 0xFFu);
				if (high) cbj >>= 4;
				aid[j] = static_cast<uint8_t>((cbj >> 1) & 0x07u);
			}
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
				value[j] = run_alg_offset_sw(value[j], lane, aid[j], schedule_index, &ro[j], regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			nsel = static_cast<uint16_t>(nsel << 1);
		}
	}

	// span-end inverse-bloom CAP dedup (the drain).
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		uint32_t h_lo, h_hi;
		warp_hash_state(value[j], lane, &h_lo, &h_hi);   // all lanes
		if (alive_local[cb + j] && lane == 0u)
		{
			unsigned long long fp = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
			if (fp == 0ull) fp = 1ull;
			if constexpr (MODE == CAP_CAS_XTILE)
				cap_probe_insert_xtile(fp, orig_of[cb + j], epoch, cap_table, cap_bits, cap_ways,
					rep_global, mult, &alive_local[cb + j]);
			else
				cap_probe_insert<MODE>(fp, orig_of[cb + j], epoch, cap_table, cap_bits, cap_ways,
					rep_global, mult, &alive_local[cb + j]);
		}
	}
	__syncthreads();

	// writeback survivor state + alive/rep for compaction + resolve.
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t s = block_base + cb + j;
		if (s >= M) continue;
		const uint32_t orig = orig_of[cb + j];
		if (alive_local[cb + j]) { state[(size_t)orig * 32u + lane] = value[j]; if (lane == 0u) alive_out[s] = 1u; }
		else if (lane == 0u) alive_out[s] = 0u;   // rep_global set at the probe
	}
}

#define TM_SPAN_DRAIN_CAP_KERNEL(NAME, WARPS, ILP, MODE)                                              \
	extern "C" __global__ void NAME(                                                                 \
	const uint32_t* live_idx, uint32_t M,                                                        \
	uint32_t* state, uint8_t* alive_out, uint32_t* rep_global,                                   \
	uint32_t* mult,                                                                                 \
	uint32_t m0, uint32_t m1, int first_span,                                                    \
		const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,   \
		const uint32_t* alg2_values, const uint32_t* alg5_values,                                    \
		const uint8_t* expansion_values, const uint8_t* schedule_data,                               \
		uint32_t key, uint32_t data_start,                                                           \
		unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways, uint32_t epoch)         \
	{                                                                                                \
		run_span_dedup_drain_cap_impl<(WARPS), (ILP), (MODE)>(live_idx, M, state, alive_out, rep_global, \
			mult, m0, m1, first_span, regular_rng_values, alg0_values, alg6_values, alg2_values,     \
			alg5_values, expansion_values, schedule_data, key, data_start,                           \
			cap_table, cap_bits, cap_ways, epoch);                                                   \
	}

TM_SPAN_DRAIN_CAP_KERNEL(run_span_dedup_drain_cap_w8i8_cuda, 8u, 8u, CAP_CAS128)
TM_SPAN_DRAIN_CAP_KERNEL(run_span_dedup_drain_cap_w8i10_cuda, 8u, 10u, CAP_CAS128)
TM_SPAN_DRAIN_CAP_KERNEL(run_span_dedup_drain_capa_w8i8_cuda, 8u, 8u, CAP_CAS64A)
TM_SPAN_DRAIN_CAP_KERNEL(run_span_dedup_drain_capa_w8i10_cuda, 8u, 10u, CAP_CAS64A)
TM_SPAN_DRAIN_CAP_KERNEL(run_span_dedup_drain_capx_w8i8_cuda, 8u, 8u, CAP_CAS_XTILE)
TM_SPAN_DRAIN_CAP_KERNEL(run_span_dedup_drain_capx_w8i10_cuda, 8u, 10u, CAP_CAS_XTILE)

// WS1 shed-proxy-ROUTED span-0 MAP1 merge: the in-place MAP1 pre-processing stage.
// Computes a per-candidate SHED score during the MAP1 walk (cheap proxy = count of
// alg0 shift-left / alg6 shift-right info-loss ops in this candidate's 16-op map) and
// only DEDUPLICATES high-shed (>= tau) states into the inverse-bloom cap; LOW-shed
// states are PASSED through un-hashed (kept as survivors, no table insert). Rationale
// (validated on the CPU shed-routing study, ρ=[2.72,6.76]): collisions
// concentrate in high-shed states, so this one-shots most duplicates with a table that
// only holds the likely-duplicate subset — never the full 2x-tile frontier. Passing a
// low-shed state is FN-safe (it's KEPT; a rare low-shed dup just runs separately =
// bounded over-keep). route_stats[0]+=hashed, [1]+=passed. Same epoch-tagged 16B cap
// (atom.cas.b128) as the drain cap → FN-safe rep + flat memory + no re-zero.
template<uint32_t WARPS, uint32_t ILP, int MODE>
__device__ __forceinline__ void run_span0_routed_cap_impl(
	const uint32_t* live_idx, uint32_t M,
	uint32_t* state, uint8_t* alive_out, uint32_t* rep_global, uint32_t* mult,
	uint32_t m0, uint32_t m1, int first_span,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start,
	unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways, uint32_t epoch,
	uint32_t tau, unsigned long long* route_stats)
{
	constexpr uint32_t W = WARPS * ILP;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t cb = warp_index * ILP;
	const uint32_t block_base = blockIdx.x * W;

	__shared__ uint8_t  alive_local[W];
	__shared__ uint32_t orig_of[W];
	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			const uint32_t s = block_base + cb + j;
			const bool active = s < M;
			orig_of[cb + j]     = active ? (first_span ? s : live_idx[s]) : 0xFFFFFFFFu;
			alive_local[cb + j] = active ? 1u : 0u;
		}
	}
	__syncthreads();

	uint32_t value[ILP];
	uint32_t shed[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t orig = orig_of[cb + j];
		value[j] = alive_local[cb + j]
			? (first_span ? initialize_working_word(key, data_start + orig, lane, expansion_values)
			              : state[(size_t)orig * 32u + lane])
			: 0u;
		shed[j] = 0u;
	}

	for (uint32_t schedule_index = m0; schedule_index < m1; schedule_index++)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
		uint32_t ro[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) ro[j] = 0u;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));
		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const bool high = (nsel & 0x8000u) != 0u;
			uint8_t aid[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				const uint32_t w = __shfl_sync(0xFFFFFFFFu, value[j], sl);
				uint8_t cbj = static_cast<uint8_t>((w >> ss) & 0xFFu);
				if (high) cbj >>= 4;
				aid[j] = static_cast<uint8_t>((cbj >> 1) & 0x07u);
			}
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				if (aid[j] == 0u || aid[j] == 6u) shed[j]++;   // shed-proxy: count info-loss ops
				value[j] = run_alg_offset_sw(value[j], lane, aid[j], schedule_index, &ro[j], regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			}
			nsel = static_cast<uint16_t>(nsel << 1);
		}
	}

	// ROUTED span-end dedup: high-shed → cap probe (dedup); low-shed → pass (keep).
	// shed[j] is WARP-UNIFORM (derived from shfl-broadcast aid[j], identical on all 32 lanes),
	// so gating the cooperative warp_hash_state on shed>=tau is divergence-free and skips the
	// expensive hash for passed states — mirrors the CPU flow (map1par worker_routed: strong64
	// is computed only inside the score>=tau branch). The hash (shfl/MIO) is the span-0 bottleneck,
	// so this is where routing must save, not just at the insert.
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		if (shed[j] >= tau)                                // warp-uniform → all lanes hash, or none
		{
			uint32_t h_lo, h_hi;
			warp_hash_state(value[j], lane, &h_lo, &h_hi);   // all lanes (high-shed only)
			if (alive_local[cb + j] && lane == 0u)
			{
				if (route_stats) atomicAdd(route_stats, 1ull);
				unsigned long long fp = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
				if (fp == 0ull) fp = 1ull;
				cap_probe_insert<MODE>(fp, orig_of[cb + j], epoch, cap_table, cap_bits, cap_ways,
					rep_global, mult, &alive_local[cb + j]);
			}
		}
		else if (route_stats && alive_local[cb + j] && lane == 0u)
			atomicAdd(route_stats + 1, 1ull);              // low-shed → passed un-hashed (kept)
	}
	__syncthreads();

	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t s = block_base + cb + j;
		if (s >= M) continue;
		const uint32_t orig = orig_of[cb + j];
		if (alive_local[cb + j]) { state[(size_t)orig * 32u + lane] = value[j]; if (lane == 0u) alive_out[s] = 1u; }
		else if (lane == 0u) alive_out[s] = 0u;
	}
}

#define TM_SPAN0_ROUTED_CAP_KERNEL(NAME, WARPS, ILP, MODE)                                            \
	extern "C" __global__ void NAME(                                                                 \
		const uint32_t* live_idx, uint32_t M,                                                        \
		uint32_t* state, uint8_t* alive_out, uint32_t* rep_global,                                   \
		uint32_t* mult,                                                                              \
		uint32_t m0, uint32_t m1, int first_span,                                                    \
		const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,   \
		const uint32_t* alg2_values, const uint32_t* alg5_values,                                    \
		const uint8_t* expansion_values, const uint8_t* schedule_data,                               \
		uint32_t key, uint32_t data_start,                                                           \
		unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways, uint32_t epoch,         \
		uint32_t tau, unsigned long long* route_stats)                                               \
	{                                                                                                \
		run_span0_routed_cap_impl<(WARPS), (ILP), (MODE)>(live_idx, M, state, alive_out, rep_global, \
			mult, m0, m1, first_span, regular_rng_values, alg0_values, alg6_values, alg2_values,     \
			alg5_values, expansion_values, schedule_data, key, data_start,                           \
			cap_table, cap_bits, cap_ways, epoch, tau, route_stats);                                 \
	}

TM_SPAN0_ROUTED_CAP_KERNEL(run_span0_routed_cap_w8i8_cuda, 8u, 8u, CAP_CAS128)
TM_SPAN0_ROUTED_CAP_KERNEL(run_span0_routed_cap_w8i10_cuda, 8u, 10u, CAP_CAS128)
TM_SPAN0_ROUTED_CAP_KERNEL(run_span0_routed_capa_w8i8_cuda, 8u, 8u, CAP_CAS64A)
TM_SPAN0_ROUTED_CAP_KERNEL(run_span0_routed_capa_w8i10_cuda, 8u, 10u, CAP_CAS64A)
TM_SPAN0_ROUTED_CAP_KERNEL(run_span0_routed_capx_w8i8_cuda, 8u, 8u, CAP_CAS_XTILE)
TM_SPAN0_ROUTED_CAP_KERNEL(run_span0_routed_capx_w8i10_cuda, 8u, 10u, CAP_CAS_XTILE)

// ── two-array exact 64-bit cap (CAP_CAS64B/64C host policies) ────────────────────────
// The exact table's proven protocol (run_span_dedup_global_impl): separate cap_fp[u64]
// (empty=0, the CAS/linearization word) + cap_rep[u32] (volatile, publish-spin sentinel
// 0xFFFFFFFF), bucketed into 2^cap_bits x cap_ways. Full 64-bit fp compare (no epoch
// packing). Because the table is re-zeroed before use (host policy 64b = one table re-zeroed
// per drain boundary; 64c = K distinct tables, one per drain site, zeroed once), fp is written
// once per use (monotonic 0->fp) so the published rep is always consistent — FN-safe, no
// two-word race, only a 64-bit atomicCAS. Bucket full of foreign fps => leave un-merged.
__device__ __forceinline__ void cap_probe_insert_b(
	unsigned long long fp, uint32_t myrep,
	unsigned long long* cap_fp, volatile uint32_t* cap_rep, uint32_t cap_bits, uint32_t cap_ways,
	uint32_t* rep_global, uint32_t* mult, uint8_t* alive_slot)
{
	const uint32_t bucket = (uint32_t)((fp * 0x9E3779B97F4A7C15ull) >> (64 - cap_bits));
	const unsigned long long base = (unsigned long long)bucket * cap_ways;
	for (uint32_t w = 0u; w < cap_ways; w++)
	{
		const unsigned long long si = base + w;
		const unsigned long long old = atomicCAS(reinterpret_cast<unsigned long long*>(&cap_fp[si]), 0ull, fp);
		if (old == 0ull) { cap_rep[si] = myrep; __threadfence(); return; }   // won the slot → winner (alive)
		if (old == fp)                                                       // duplicate → adopt owner's rep
		{
			uint32_t r; do { r = cap_rep[si]; } while (r == 0xFFFFFFFFu);
			rep_global[myrep] = r;
			tm_merge_multiplicity(mult, myrep, r);
			*alive_slot = 0u; return;
		}
		// different fp → next way; if none free, leave un-merged (alive) → bounded over-keep
	}
}

template<uint32_t WARPS, uint32_t ILP>
__device__ __forceinline__ void run_span_dedup_drain_capb_impl(
	const uint32_t* live_idx, uint32_t M,
	uint32_t* state, uint8_t* alive_out, uint32_t* rep_global, uint32_t* mult,
	uint32_t m0, uint32_t m1, int first_span,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start,
	unsigned long long* cap_fp, volatile uint32_t* cap_rep, uint32_t cap_bits, uint32_t cap_ways)
{
	constexpr uint32_t W = WARPS * ILP;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t cb = warp_index * ILP;
	const uint32_t block_base = blockIdx.x * W;

	__shared__ uint8_t  alive_local[W];
	__shared__ uint32_t orig_of[W];
	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			const uint32_t s = block_base + cb + j;
			const bool active = s < M;
			orig_of[cb + j]     = active ? (first_span ? s : live_idx[s]) : 0xFFFFFFFFu;
			alive_local[cb + j] = active ? 1u : 0u;
		}
	}
	__syncthreads();

	uint32_t value[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t orig = orig_of[cb + j];
		value[j] = alive_local[cb + j]
			? (first_span ? initialize_working_word(key, data_start + orig, lane, expansion_values)
			              : state[(size_t)orig * 32u + lane])
			: 0u;
	}

	for (uint32_t schedule_index = m0; schedule_index < m1; schedule_index++)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
		uint32_t ro[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) ro[j] = 0u;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));
		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const bool high = (nsel & 0x8000u) != 0u;
			uint8_t aid[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				const uint32_t w = __shfl_sync(0xFFFFFFFFu, value[j], sl);
				uint8_t cbj = static_cast<uint8_t>((w >> ss) & 0xFFu);
				if (high) cbj >>= 4;
				aid[j] = static_cast<uint8_t>((cbj >> 1) & 0x07u);
			}
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
				value[j] = run_alg_offset_sw(value[j], lane, aid[j], schedule_index, &ro[j], regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			nsel = static_cast<uint16_t>(nsel << 1);
		}
	}

	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		uint32_t h_lo, h_hi;
		warp_hash_state(value[j], lane, &h_lo, &h_hi);   // all lanes
		if (alive_local[cb + j] && lane == 0u)
		{
			unsigned long long fp = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
			if (fp == 0ull) fp = 1ull;
			cap_probe_insert_b(fp, orig_of[cb + j], cap_fp, cap_rep, cap_bits, cap_ways,
				rep_global, mult, &alive_local[cb + j]);
		}
	}
	__syncthreads();

	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t s = block_base + cb + j;
		if (s >= M) continue;
		const uint32_t orig = orig_of[cb + j];
		if (alive_local[cb + j]) { state[(size_t)orig * 32u + lane] = value[j]; if (lane == 0u) alive_out[s] = 1u; }
		else if (lane == 0u) alive_out[s] = 0u;
	}
}

template<uint32_t WARPS, uint32_t ILP>
__device__ __forceinline__ void run_span0_routed_capb_impl(
	const uint32_t* live_idx, uint32_t M,
	uint32_t* state, uint8_t* alive_out, uint32_t* rep_global, uint32_t* mult,
	uint32_t m0, uint32_t m1, int first_span,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start,
	unsigned long long* cap_fp, volatile uint32_t* cap_rep, uint32_t cap_bits, uint32_t cap_ways,
	uint32_t tau, unsigned long long* route_stats)
{
	constexpr uint32_t W = WARPS * ILP;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t cb = warp_index * ILP;
	const uint32_t block_base = blockIdx.x * W;

	__shared__ uint8_t  alive_local[W];
	__shared__ uint32_t orig_of[W];
	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			const uint32_t s = block_base + cb + j;
			const bool active = s < M;
			orig_of[cb + j]     = active ? (first_span ? s : live_idx[s]) : 0xFFFFFFFFu;
			alive_local[cb + j] = active ? 1u : 0u;
		}
	}
	__syncthreads();

	uint32_t value[ILP];
	uint32_t shed[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t orig = orig_of[cb + j];
		value[j] = alive_local[cb + j]
			? (first_span ? initialize_working_word(key, data_start + orig, lane, expansion_values)
			              : state[(size_t)orig * 32u + lane])
			: 0u;
		shed[j] = 0u;
	}

	for (uint32_t schedule_index = m0; schedule_index < m1; schedule_index++)
	{
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
		uint32_t ro[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) ro[j] = 0u;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));
		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const bool high = (nsel & 0x8000u) != 0u;
			uint8_t aid[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				const uint32_t w = __shfl_sync(0xFFFFFFFFu, value[j], sl);
				uint8_t cbj = static_cast<uint8_t>((w >> ss) & 0xFFu);
				if (high) cbj >>= 4;
				aid[j] = static_cast<uint8_t>((cbj >> 1) & 0x07u);
			}
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				if (aid[j] == 0u || aid[j] == 6u) shed[j]++;
				value[j] = run_alg_offset_sw(value[j], lane, aid[j], schedule_index, &ro[j], regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			}
			nsel = static_cast<uint16_t>(nsel << 1);
		}
	}

	// shed[j] warp-uniform → gate the cooperative hash on shed>=tau (skip it for passed states).
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		if (shed[j] >= tau)
		{
			uint32_t h_lo, h_hi;
			warp_hash_state(value[j], lane, &h_lo, &h_hi);   // all lanes (high-shed only)
			if (alive_local[cb + j] && lane == 0u)
			{
				if (route_stats) atomicAdd(route_stats, 1ull);
				unsigned long long fp = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
				if (fp == 0ull) fp = 1ull;
				cap_probe_insert_b(fp, orig_of[cb + j], cap_fp, cap_rep, cap_bits, cap_ways,
					rep_global, mult, &alive_local[cb + j]);
			}
		}
		else if (route_stats && alive_local[cb + j] && lane == 0u)
			atomicAdd(route_stats + 1, 1ull);
	}
	__syncthreads();

	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t s = block_base + cb + j;
		if (s >= M) continue;
		const uint32_t orig = orig_of[cb + j];
		if (alive_local[cb + j]) { state[(size_t)orig * 32u + lane] = value[j]; if (lane == 0u) alive_out[s] = 1u; }
		else if (lane == 0u) alive_out[s] = 0u;
	}
}

#define TM_SPAN_DRAIN_CAPB_KERNEL(NAME, WARPS, ILP)                                                   \
	extern "C" __global__ void NAME(                                                                 \
		const uint32_t* live_idx, uint32_t M,                                                        \
		uint32_t* state, uint8_t* alive_out, uint32_t* rep_global,                                   \
		uint32_t* mult,                                                                              \
		uint32_t m0, uint32_t m1, int first_span,                                                    \
		const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,   \
		const uint32_t* alg2_values, const uint32_t* alg5_values,                                    \
		const uint8_t* expansion_values, const uint8_t* schedule_data,                               \
		uint32_t key, uint32_t data_start,                                                           \
		unsigned long long* cap_fp, uint32_t* cap_rep, uint32_t cap_bits, uint32_t cap_ways)         \
	{                                                                                                \
		run_span_dedup_drain_capb_impl<(WARPS), (ILP)>(live_idx, M, state, alive_out, rep_global,    \
			mult, m0, m1, first_span, regular_rng_values, alg0_values, alg6_values, alg2_values,     \
			alg5_values, expansion_values, schedule_data, key, data_start,                           \
			cap_fp, cap_rep, cap_bits, cap_ways);                                                    \
	}

TM_SPAN_DRAIN_CAPB_KERNEL(run_span_dedup_drain_capb_w8i8_cuda, 8u, 8u)
TM_SPAN_DRAIN_CAPB_KERNEL(run_span_dedup_drain_capb_w8i10_cuda, 8u, 10u)

#define TM_SPAN0_ROUTED_CAPB_KERNEL(NAME, WARPS, ILP)                                                 \
	extern "C" __global__ void NAME(                                                                 \
		const uint32_t* live_idx, uint32_t M,                                                        \
		uint32_t* state, uint8_t* alive_out, uint32_t* rep_global,                                   \
		uint32_t* mult,                                                                              \
		uint32_t m0, uint32_t m1, int first_span,                                                    \
		const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,   \
		const uint32_t* alg2_values, const uint32_t* alg5_values,                                    \
		const uint8_t* expansion_values, const uint8_t* schedule_data,                               \
		uint32_t key, uint32_t data_start,                                                           \
		unsigned long long* cap_fp, uint32_t* cap_rep, uint32_t cap_bits, uint32_t cap_ways,         \
		uint32_t tau, unsigned long long* route_stats)                                               \
	{                                                                                                \
		run_span0_routed_capb_impl<(WARPS), (ILP)>(live_idx, M, state, alive_out, rep_global,        \
			mult, m0, m1, first_span, regular_rng_values, alg0_values, alg6_values, alg2_values,     \
			alg5_values, expansion_values, schedule_data, key, data_start,                           \
			cap_fp, cap_rep, cap_bits, cap_ways, tau, route_stats);                                  \
	}

TM_SPAN0_ROUTED_CAPB_KERNEL(run_span0_routed_capb_w8i8_cuda, 8u, 8u)
TM_SPAN0_ROUTED_CAPB_KERNEL(run_span0_routed_capb_w8i10_cuda, 8u, 10u)

// ── trajgate DEEP-routing classifier (op-tail count-min trajDens + sticky alg0) ──────
// The deep-stage classifier the CPU exploration converged on (docs/w4b_trajgate_*): route
// (dedup into the cap) only the likely-duplicate states; PASS the rest un-hashed so the deep
// cap holds only the high-collision subset = the MEMORY lever for massive diffuse deep
// frontiers. Two cheap, in-kernel features (no 2-pass re-read):
//   trajDens : op-tail key = last-8 alg_ids of the FINAL map (24-bit) → 4-bit-ish count-min
//              sketch (cap_min over D hashes) = trajectory multiplicity. High = common
//              trajectory = likely dup. Online (probe-before-update).
//   sticky   : did ANY map in the span have >= alg0_tau alg0 ops (max per-map alg0 count
//              over the span)? Catches FIRST-SEEN high-collision reps that the online density
//              misses (density~0 on first sight) — almost free (a compare+add already in the
//              dispatch), and crucially needs no second pass. (Owner: keep the sticky bit; the
//              CPU "drop alg0" applied to the FULL lseA0 softmax = K× capture, not this.)
// route = (trajDens >= dens_tau) || (alg0_tau>0 && maxA0 >= alg0_tau). FN-safe: passed states
// are KEPT (a passed dup just survives = bounded over-keep). D=4 count-min over 2^sk_bits u32.
__device__ __forceinline__ uint32_t trajsketch_probe(const uint32_t* sk, uint32_t sk_bits, uint32_t key24)
{
	const uint32_t s = 32u - sk_bits;
	uint32_t a = sk[(key24 * 0x9E3779B1u) >> s];
	uint32_t b = sk[((key24 * 0x85EBCA77u) ^ 0xABCD1234u) >> s];
	uint32_t c = sk[((key24 * 0xC2B2AE3Du) ^ 0x12345678u) >> s];
	uint32_t d = sk[((key24 * 0x27D4EB2Fu) ^ 0xDEADBEEFu) >> s];
	uint32_t m = a < b ? a : b; uint32_t n = c < d ? c : d; return m < n ? m : n;
}
__device__ __forceinline__ void trajsketch_update(uint32_t* sk, uint32_t sk_bits, uint32_t key24)
{
	const uint32_t s = 32u - sk_bits;
	atomicAdd(&sk[(key24 * 0x9E3779B1u) >> s], 1u);
	atomicAdd(&sk[((key24 * 0x85EBCA77u) ^ 0xABCD1234u) >> s], 1u);
	atomicAdd(&sk[((key24 * 0xC2B2AE3Du) ^ 0x12345678u) >> s], 1u);
	atomicAdd(&sk[((key24 * 0x27D4EB2Fu) ^ 0xDEADBEEFu) >> s], 1u);
}

template<uint32_t WARPS, uint32_t ILP, int MODE>
__device__ __forceinline__ void run_span_deep_trajroute_cap_impl(
	const uint32_t* live_idx, uint32_t M,
	uint32_t* state, uint8_t* alive_out, uint32_t* rep_global, uint32_t* mult,
	uint32_t m0, uint32_t m1, int first_span,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start,
	unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways, uint32_t epoch,
	uint32_t dens_tau, uint32_t alg0_tau, uint32_t* sketch, uint32_t sk_bits,
	unsigned long long* route_stats, uint32_t mult_tau)
{
	constexpr uint32_t W = WARPS * ILP;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t cb = warp_index * ILP;
	const uint32_t block_base = blockIdx.x * W;

	__shared__ uint8_t  alive_local[W];
	__shared__ uint32_t orig_of[W];
	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			const uint32_t s = block_base + cb + j;
			const bool active = s < M;
			orig_of[cb + j]     = active ? (first_span ? s : live_idx[s]) : 0xFFFFFFFFu;
			alive_local[cb + j] = active ? 1u : 0u;
		}
	}
	__syncthreads();

	uint32_t value[ILP];
	uint32_t optail[ILP];
	uint32_t maxA0[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t orig = orig_of[cb + j];
		value[j] = alive_local[cb + j]
			? (first_span ? initialize_working_word(key, data_start + orig, lane, expansion_values)
			              : state[(size_t)orig * 32u + lane])
			: 0u;
		optail[j] = 0u; maxA0[j] = 0u;
	}

	for (uint32_t schedule_index = m0; schedule_index < m1; schedule_index++)
	{
		const bool final_map = (schedule_index + 1u == m1);
		uint32_t mapA0[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) mapA0[j] = 0u;
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
		uint32_t ro[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) ro[j] = 0u;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));
		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const bool high = (nsel & 0x8000u) != 0u;
			uint8_t aid[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				const uint32_t w = __shfl_sync(0xFFFFFFFFu, value[j], sl);
				uint8_t cbj = static_cast<uint8_t>((w >> ss) & 0xFFu);
				if (high) cbj >>= 4;
				aid[j] = static_cast<uint8_t>((cbj >> 1) & 0x07u);
			}
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				if (aid[j] == 0u) mapA0[j]++;                              // per-map alg0 count (sticky)
				if (final_map) optail[j] = (optail[j] << 3) | aid[j];      // op-tail key (last 8 → low 24b)
				value[j] = run_alg_offset_sw(value[j], lane, aid[j], schedule_index, &ro[j], regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			}
			nsel = static_cast<uint16_t>(nsel << 1);
		}
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) if (mapA0[j] > maxA0[j]) maxA0[j] = mapA0[j];
	}

	// trajgate routing decision (warp-uniform: optail/maxA0 derive from shfl-broadcast aids).
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		if (!alive_local[cb + j]) continue;                               // uniform across the warp
		const uint32_t key24 = optail[j] & 0xFFFFFFu;
		const bool sticky = (alg0_tau > 0u) && (maxA0[j] >= alg0_tau);
		uint32_t dens = 0u;
		if (lane == 0u) dens = trajsketch_probe(sketch, sk_bits, key24);  // online probe
		dens = __shfl_sync(0xFFFFFFFFu, dens, 0);
		const bool mult_flag = tm_mult_route(mult, orig_of[cb + j], mult_tau);
		const bool route = sticky || (dens >= dens_tau) || mult_flag;
		if (route)
		{
			uint32_t h_lo, h_hi;
			warp_hash_state(value[j], lane, &h_lo, &h_hi);                // all lanes (routed only)
			if (lane == 0u)
			{
				if (route_stats)
				{
					atomicAdd(route_stats, 1ull);
					if (sticky && dens < dens_tau) atomicAdd(route_stats + 2, 1ull);
					if (mult_flag && !sticky && dens < dens_tau) atomicAdd(route_stats + 3, 1ull);
				}
				unsigned long long fp = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
				if (fp == 0ull) fp = 1ull;
				cap_probe_insert<MODE>(fp, orig_of[cb + j], epoch, cap_table, cap_bits, cap_ways,
					rep_global, mult, &alive_local[cb + j]);
			}
		}
		else if (lane == 0u && route_stats) atomicAdd(route_stats + 1, 1ull);   // passed un-hashed (kept)
		if (lane == 0u) trajsketch_update(sketch, sk_bits, key24);        // build density (all alive)
	}
	__syncthreads();

	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t s = block_base + cb + j;
		if (s >= M) continue;
		const uint32_t orig = orig_of[cb + j];
		if (alive_local[cb + j]) { state[(size_t)orig * 32u + lane] = value[j]; if (lane == 0u) alive_out[s] = 1u; }
		else if (lane == 0u) alive_out[s] = 0u;
	}
}

#define TM_DEEP_TRAJROUTE_KERNEL(NAME, WARPS, ILP, MODE)                                              \
	extern "C" __global__ void NAME(                                                                 \
		const uint32_t* live_idx, uint32_t M,                                                        \
		uint32_t* state, uint8_t* alive_out, uint32_t* rep_global,                                   \
		uint32_t* mult,                                                                              \
		uint32_t m0, uint32_t m1, int first_span,                                                    \
		const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,   \
		const uint32_t* alg2_values, const uint32_t* alg5_values,                                    \
		const uint8_t* expansion_values, const uint8_t* schedule_data,                               \
		uint32_t key, uint32_t data_start,                                                           \
		unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways, uint32_t epoch,         \
		uint32_t dens_tau, uint32_t alg0_tau, uint32_t* sketch, uint32_t sk_bits,                    \
		unsigned long long* route_stats, uint32_t mult_tau)                                          \
	{                                                                                                \
		run_span_deep_trajroute_cap_impl<(WARPS), (ILP), (MODE)>(live_idx, M, state, alive_out,      \
			rep_global, mult, m0, m1, first_span, regular_rng_values, alg0_values, alg6_values,      \
			alg2_values, alg5_values, expansion_values, schedule_data, key, data_start,              \
			cap_table, cap_bits, cap_ways, epoch, dens_tau, alg0_tau, sketch, sk_bits, route_stats, mult_tau); \
	}

TM_DEEP_TRAJROUTE_KERNEL(run_span_deep_trajroute_capa_w8i8_cuda, 8u, 8u, CAP_CAS64A)
TM_DEEP_TRAJROUTE_KERNEL(run_span_deep_trajroute_capa_w8i10_cuda, 8u, 10u, CAP_CAS64A)
TM_DEEP_TRAJROUTE_KERNEL(run_span_deep_trajroute_capx_w8i8_cuda, 8u, 8u, CAP_CAS_XTILE)
TM_DEEP_TRAJROUTE_KERNEL(run_span_deep_trajroute_capx_w8i10_cuda, 8u, 10u, CAP_CAS_XTILE)

// ── TWO-PASS trajgate (correct density under GPU parallelism) ────────────────────────
// The single-pass kernel above probes the count-min sketch while millions of threads
// concurrently update it ⇒ a RACY partial count (the CPU's sequential probe_update gives a
// deterministic prior-count; parallel probe-then-update gives noise). Fix: split into
//   PASS 1 (build): walk the span, write span-end state, capture op-tail key + sticky, STORE
//     them per candidate (keybuf), and UPDATE the sketch. No routing — every state kept.
//   PASS 2 (route): reload the span-end state + keybuf, PROBE the now-complete sketch (every
//     state sees the TRUE total multiplicity, better than CPU prior-count), route the
//     likely-dups into the cap. No re-walk.
// keybuf[s] = (sticky << 24) | (key24 & 0xFFFFFF).
template<uint32_t WARPS, uint32_t ILP>
__device__ __forceinline__ void run_span_deep_trajbuild_impl(
	const uint32_t* live_idx, uint32_t M,
	uint32_t* state, uint8_t* alive_out,
	uint32_t m0, uint32_t m1, int first_span,
	const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,
	const uint32_t* alg2_values, const uint32_t* alg5_values,
	const uint8_t* expansion_values, const uint8_t* schedule_data,
	uint32_t key, uint32_t data_start,
	uint32_t* keybuf, uint32_t* sketch, uint32_t sk_bits, uint32_t alg0_tau)
{
	constexpr uint32_t W = WARPS * ILP;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t cb = warp_index * ILP;
	const uint32_t block_base = blockIdx.x * W;

	__shared__ uint8_t  alive_local[W];
	__shared__ uint32_t orig_of[W];
	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			const uint32_t s = block_base + cb + j;
			const bool active = s < M;
			orig_of[cb + j]     = active ? (first_span ? s : live_idx[s]) : 0xFFFFFFFFu;
			alive_local[cb + j] = active ? 1u : 0u;
		}
	}
	__syncthreads();

	uint32_t value[ILP]; uint32_t optail[ILP]; uint32_t maxA0[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t orig = orig_of[cb + j];
		value[j] = alive_local[cb + j]
			? (first_span ? initialize_working_word(key, data_start + orig, lane, expansion_values)
			              : state[(size_t)orig * 32u + lane])
			: 0u;
		optail[j] = 0u; maxA0[j] = 0u;
	}

	for (uint32_t schedule_index = m0; schedule_index < m1; schedule_index++)
	{
		const bool final_map = (schedule_index + 1u == m1);
		uint32_t mapA0[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) mapA0[j] = 0u;
		uint32_t packed_schedule = 0u;
		if (lane == 0u) packed_schedule = reinterpret_cast<const uint32_t*>(schedule_data)[schedule_index];
		packed_schedule = __shfl_sync(0xFFFFFFFFu, packed_schedule, 0);
		uint32_t ro[ILP];
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) ro[j] = 0u;
		uint16_t nsel = static_cast<uint16_t>(((packed_schedule & 0xFF0000u) >> 8) | ((packed_schedule & 0xFF000000u) >> 24));
		for (uint32_t i = 0; i < 16u; i++)
		{
			const uint32_t sl = i >> 2;
			const uint32_t ss = (i & 3u) * 8u;
			const bool high = (nsel & 0x8000u) != 0u;
			uint8_t aid[ILP];
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				const uint32_t w = __shfl_sync(0xFFFFFFFFu, value[j], sl);
				uint8_t cbj = static_cast<uint8_t>((w >> ss) & 0xFFu);
				if (high) cbj >>= 4;
				aid[j] = static_cast<uint8_t>((cbj >> 1) & 0x07u);
			}
			#pragma unroll
			for (uint32_t j = 0u; j < ILP; j++)
			{
				if (aid[j] == 0u) mapA0[j]++;
				if (final_map) optail[j] = (optail[j] << 3) | aid[j];      // op-tail key (last 8 → low 24b)
				value[j] = run_alg_offset_sw(value[j], lane, aid[j], schedule_index, &ro[j], regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values);
			}
			nsel = static_cast<uint16_t>(nsel << 1);
		}
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++) if (mapA0[j] > maxA0[j]) maxA0[j] = mapA0[j];
	}

	// capture key+sticky + populate sketch (no routing; all states kept)
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		if (!alive_local[cb + j]) continue;
		const uint32_t key24 = optail[j] & 0xFFFFFFu;
		if (lane == 0u)
		{
			const uint32_t sticky = ((alg0_tau > 0u) && (maxA0[j] >= alg0_tau)) ? 1u : 0u;
			keybuf[block_base + cb + j] = (sticky << 24) | key24;          // bit24=sticky, bits[23:0]=key
			trajsketch_update(sketch, sk_bits, key24);
		}
	}
	__syncthreads();

	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t s = block_base + cb + j;
		if (s >= M) continue;
		const uint32_t orig = orig_of[cb + j];
		state[(size_t)orig * 32u + lane] = value[j];
		if (lane == 0u) alive_out[s] = 1u;
	}
}

template<uint32_t WARPS, uint32_t ILP, int MODE>
__device__ __forceinline__ void run_span_deep_trajroute2_impl(
	const uint32_t* live_idx, uint32_t M,
	uint32_t* state, uint8_t* alive_out, uint32_t* rep_global, uint32_t* mult, int first_span,
	unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways, uint32_t epoch,
	uint32_t dens_tau, const uint32_t* keybuf, const uint32_t* sketch, uint32_t sk_bits,
	unsigned long long* route_stats, uint32_t mult_tau)
{
	constexpr uint32_t W = WARPS * ILP;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t warp_index = threadIdx.x >> 5;
	const uint32_t cb = warp_index * ILP;
	const uint32_t block_base = blockIdx.x * W;

	__shared__ uint8_t  alive_local[W];
	__shared__ uint32_t orig_of[W];
	if (lane == 0u)
	{
		#pragma unroll
		for (uint32_t j = 0u; j < ILP; j++)
		{
			const uint32_t s = block_base + cb + j;
			const bool active = s < M;
			orig_of[cb + j]     = active ? (first_span ? s : live_idx[s]) : 0xFFFFFFFFu;
			alive_local[cb + j] = active ? 1u : 0u;
		}
	}
	__syncthreads();

	uint32_t value[ILP];
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
		value[j] = alive_local[cb + j] ? state[(size_t)orig_of[cb + j] * 32u + lane] : 0u;

	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		if (!alive_local[cb + j]) continue;
		const uint32_t kb = keybuf[block_base + cb + j];
		const bool sticky = (kb >> 24) != 0u;
		uint32_t dens = 0u;
		if (lane == 0u) dens = trajsketch_probe(sketch, sk_bits, kb & 0xFFFFFFu); // complete sketch
		dens = __shfl_sync(0xFFFFFFFFu, dens, 0);
		const bool mult_flag = tm_mult_route(mult, orig_of[cb + j], mult_tau);
		const bool route = sticky || (dens >= dens_tau) || mult_flag;
		if (route)
		{
			uint32_t h_lo, h_hi;
			warp_hash_state(value[j], lane, &h_lo, &h_hi);
			if (lane == 0u)
			{
				if (route_stats)
				{
					atomicAdd(route_stats, 1ull);
					if (sticky && dens < dens_tau) atomicAdd(route_stats + 2, 1ull);
					if (mult_flag && !sticky && dens < dens_tau) atomicAdd(route_stats + 3, 1ull);
				}
				unsigned long long fp = ((unsigned long long)h_hi << 32) | (unsigned long long)h_lo;
				if (fp == 0ull) fp = 1ull;
				cap_probe_insert<MODE>(fp, orig_of[cb + j], epoch, cap_table, cap_bits, cap_ways,
					rep_global, mult, &alive_local[cb + j]);
			}
		}
		else if (lane == 0u && route_stats) atomicAdd(route_stats + 1, 1ull);
	}
	__syncthreads();

	// state already written by pass 1; pass 2 only updates alive (merged dups → 0).
	#pragma unroll
	for (uint32_t j = 0u; j < ILP; j++)
	{
		const uint32_t s = block_base + cb + j;
		if (s >= M) continue;
		if (lane == 0u) alive_out[s] = alive_local[cb + j];
	}
}

#define TM_DEEP_TRAJBUILD_KERNEL(NAME, WARPS, ILP)                                                    \
	extern "C" __global__ void NAME(                                                                 \
		const uint32_t* live_idx, uint32_t M, uint32_t* state, uint8_t* alive_out,                   \
		uint32_t m0, uint32_t m1, int first_span,                                                    \
		const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,   \
		const uint32_t* alg2_values, const uint32_t* alg5_values,                                    \
		const uint8_t* expansion_values, const uint8_t* schedule_data,                               \
		uint32_t key, uint32_t data_start,                                                           \
		uint32_t* keybuf, uint32_t* sketch, uint32_t sk_bits, uint32_t alg0_tau)                     \
	{                                                                                                \
		run_span_deep_trajbuild_impl<(WARPS), (ILP)>(live_idx, M, state, alive_out, m0, m1,          \
			first_span, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values,      \
			expansion_values, schedule_data, key, data_start, keybuf, sketch, sk_bits, alg0_tau);    \
	}
TM_DEEP_TRAJBUILD_KERNEL(run_span_deep_trajbuild_w8i8_cuda, 8u, 8u)
TM_DEEP_TRAJBUILD_KERNEL(run_span_deep_trajbuild_w8i10_cuda, 8u, 10u)

#define TM_DEEP_TRAJROUTE2_KERNEL(NAME, WARPS, ILP, MODE)                                             \
	extern "C" __global__ void NAME(                                                                 \
		const uint32_t* live_idx, uint32_t M, uint32_t* state, uint8_t* alive_out,                   \
		uint32_t* rep_global, uint32_t* mult, int first_span,                                        \
		unsigned long long* cap_table, uint32_t cap_bits, uint32_t cap_ways, uint32_t epoch,         \
		uint32_t dens_tau, const uint32_t* keybuf, const uint32_t* sketch, uint32_t sk_bits,         \
		unsigned long long* route_stats, uint32_t mult_tau)                                          \
	{                                                                                                \
		run_span_deep_trajroute2_impl<(WARPS), (ILP), (MODE)>(live_idx, M, state, alive_out,         \
			rep_global, mult, first_span, cap_table, cap_bits, cap_ways, epoch, dens_tau, keybuf,    \
			sketch, sk_bits, route_stats, mult_tau);                                                  \
	}
TM_DEEP_TRAJROUTE2_KERNEL(run_span_deep_trajroute2_capa_w8i8_cuda, 8u, 8u, CAP_CAS64A)
TM_DEEP_TRAJROUTE2_KERNEL(run_span_deep_trajroute2_capa_w8i10_cuda, 8u, 10u, CAP_CAS64A)
TM_DEEP_TRAJROUTE2_KERNEL(run_span_deep_trajroute2_capx_w8i8_cuda, 8u, 8u, CAP_CAS_XTILE)
TM_DEEP_TRAJROUTE2_KERNEL(run_span_deep_trajroute2_capx_w8i10_cuda, 8u, 10u, CAP_CAS_XTILE)

#define TM_SPAN_DRAIN_KERNEL(NAME, WARPS, ILP)                                                        \
	extern "C" __global__ void NAME(                                                                 \
		const uint32_t* live_idx, uint32_t M,                                                        \
		uint32_t* state, uint8_t* alive_out, uint32_t* rep_global,                                   \
		uint32_t* mult,                                                                              \
		uint32_t m0, uint32_t m1, int first_span,                                                    \
		const uint8_t* regular_rng_values, const uint8_t* alg0_values, const uint8_t* alg6_values,   \
		const uint32_t* alg2_values, const uint32_t* alg5_values,                                    \
		const uint8_t* expansion_values, const uint8_t* schedule_data,                               \
		uint32_t key, uint32_t data_start,                                                           \
		uint64_t* table_fp, volatile uint32_t* table_rep, uint32_t logm)                             \
{                                                                                                \
		run_span_dedup_drain_impl<(WARPS), (ILP)>(live_idx, M, state, alive_out, rep_global, mult, \
			m0, m1,                                                                                \
			first_span, regular_rng_values, alg0_values, alg6_values, alg2_values, alg5_values,      \
			expansion_values, schedule_data, key, data_start, table_fp, table_rep, logm);            \
	}

// Counts occupied (non-zero) slots in a global dedup table → the ACTUAL utilized
// footprint at a drain boundary (= the distinct frontier there); allocated table is
// 2x tile but real use is far smaller, which is what an inverse-bloom cap would size to.
extern "C" __global__ void count_nonzero_u64_cuda(const uint64_t* a, uint32_t n, uint32_t* out)
{
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n && a[i] != 0ull) atomicAdd(out, 1u);
}

// Counts active slots in an epoch-tagged 16-byte cap table. Plain nonzero counts
// are misleading because 128/64a/xtile caps are intentionally not re-zeroed.
// mode: 0=CAP_CAS128, 1=CAP_CAS64A, 2=CAP_CAS_XTILE.
extern "C" __global__ void count_cap_slots_epoch_cuda(
	const uint64_t* table, uint32_t nslots, uint32_t mode, uint32_t epoch, uint32_t* out)
{
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nslots) return;
	const uint64_t w0 = table[(uint64_t)i * 2ull];
	const uint64_t w1 = table[(uint64_t)i * 2ull + 1ull];
	bool active = false;
	if (mode == 1u)
		active = ((uint32_t)(w0 >> 48) == (epoch & 0xFFFFu)) && ((w0 & 0x0000FFFFFFFFFFFFull) != 0ull);
	else if (mode == 2u)
		active = ((uint32_t)(w1 >> 48) == (epoch & 0xFFFFu)) && (w0 != 0ull);
	else
		active = ((uint32_t)(w1 >> 32) == epoch) && (w0 != 0ull);
	if (active) atomicAdd(out, 1u);
}

// Counts occupied 16-byte slots regardless of epoch. Used for pooled cap tables
// where several epochs/depths share one allocation, and for wide 128-bit tables.
extern "C" __global__ void count_slots16_any_cuda(
	const uint64_t* table, uint32_t nslots, uint32_t* out)
{
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nslots) return;
	const uint64_t w0 = table[(uint64_t)i * 2ull];
	const uint64_t w1 = table[(uint64_t)i * 2ull + 1ull];
	if ((w0 | w1) != 0ull) atomicAdd(out, 1u);
}

TM_SPAN_DRAIN_KERNEL(run_span_dedup_drain_w8i4_cuda,  8u,  4u)
TM_SPAN_DRAIN_KERNEL(run_span_dedup_drain_w8i5_cuda,  8u,  5u)
TM_SPAN_DRAIN_KERNEL(run_span_dedup_drain_w8i6_cuda,  8u,  6u)
TM_SPAN_DRAIN_KERNEL(run_span_dedup_drain_w8i8_cuda,  8u,  8u)
TM_SPAN_DRAIN_KERNEL(run_span_dedup_drain_w8i10_cuda, 8u, 10u)
TM_SPAN_DRAIN_KERNEL(run_span_dedup_drain_w4i8_cuda,  4u,  8u)
