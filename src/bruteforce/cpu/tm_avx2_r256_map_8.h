#ifndef TM_AVX2_R256_MAP_8_H
#define TM_AVX2_R256_MAP_8_H
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>

#include <cstdint>
#include <cstddef>
#include <vector>

#include "data_sizes.h"
#include "alignment2.h"
#include "rng_obj.h"
#include "tm_base.h"
#include "map_tables_shared.h"

// AVX2 map-mode forward kernel. Ported from micro500/treasure-master-hack
// (commit ac11bdb, 2026-05-18). Natural (non-shuffled) byte layout in 4×__m256i.
//
// Key difference from tm_avx2_r256s_8 (the universal-table impl): each
// schedule entry has 2048 bytes of precomputed RNG values built once at
// schedule-bind time. The kernel walks a moving pointer through that buffer
// instead of indexing the 8 MB universal RNG table.
//
// Working set per schedule entry: 2048 bytes. For our default 27-entry
// ALL_MAPS schedule: 54 KB per of the three streams (regular/alg0/alg6) =
// ~162 KB total, which stays in L2. Versus the universal-table path's
// 8 MB regular_rng_values_256_8_shuffled.

class tm_avx2_r256_map_8 : public TM_base
{
public:
	tm_avx2_r256_map_8(RNG* rng);
	// Shared-tables constructor. The kernel reads per-schedule data from the
	// caller-owned Tables object instead of building its own per-instance
	// copy. Useful for multi-threaded use: N threads share one 162 KB table
	// set, avoiding N×duplication in L3.
	tm_avx2_r256_map_8(RNG* rng, map_tables_shared::Tables* shared_tables);

	void load_data(uint8* new_data) override;
	void fetch_data(uint8* new_data) override;

	void expand(uint32 key, uint32 data) override;

	void run_alg(int algorithm_id, uint16* rng_seed, int iterations) override;

	void run_one_map(const key_schedule::key_schedule_entry& schedule_entry) override;
	void run_all_maps(const key_schedule& schedule_entries) override;
	void run_maps_range(const key_schedule& schedule_entries, std::size_t begin, std::size_t end);
	void bind_schedule(const key_schedule& schedule_entries) override;
	void bind_dedup_schedule(const key_schedule& schedule_entries);
	void bind_maps_range(const key_schedule& schedule_entries, std::size_t begin, std::size_t end);

	// state_dedup interface (compile-time concept; no virtual). The map kernel
	// uses natural layout, so state_raw bytes are the canonical 128-byte state.
	const uint8* state_raw() const { return working_code_data; }
	void load_state_raw(const uint8* src) { for (int i = 0; i < 128; i++) working_code_data[i] = src[i]; }

	void run_bruteforce_data(uint32 key, uint32 data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size) override;

	// 2-way interleaved map-range kernel (mirrors the 512 maps' x4 at 2-way; AVX2
	// is 4 YMM/state so 2-way uses 8 of 16 YMM). Natural layout, so in/out are
	// canonical 128-byte states.
	void run_maps_range_x2(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, uint8* out0, uint8* out1);

	// ---- Raceway interleaved kernels (AVX2 port of the AVX-512 r512_map raceway) ----
	// AVX2 has 16 YMM and 4 YMM/state, so the no-spill "half register file" sweet
	// spot is x2 (8 of 16 YMM, leaving 8 for the alg_2/5 butterfly temps) — the
	// direct analog of the AVX-512 natmap x8 (16 of 32 ZMM). x4 (16 YMM, full file)
	// is the aggressive higher-interleave point: more independent chains to hide
	// latency, but the butterfly temps spill (analog of the AVX-512 x12/x14 regime).
	//
	// These mirror the x8-signature API the cpu_raceway harness drives, but process
	// the 8 lanes as register-resident sweeps of width AVX2_RACEWAY_W (compile-time,
	// default 2). The harness's deep dispatch fixes g_ilp=8 on AVX2, so only the
	// x8-signature forms are instantiated. in/out are canonical 128-byte states.
	void run_maps_range_x4(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3);
	void run_maps_range_x8(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		uint8* out4, uint8* out5, uint8* out6, uint8* out7);
	// Branchless-dispatch variants. The AVX2 alg dispatch is already a data-driven
	// switch; a true branch-free butterfly-fold port is a future lever, so these are
	// semantic aliases of run_maps_range_x8 (kept so deep_kfn's member-pointer table
	// resolves on the AVX2 Kern). Parity-identical to run_maps_range_x8.
	void run_maps_range_x8_blarith(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		uint8* out4, uint8* out5, uint8* out6, uint8* out7);
	void run_maps_range_x8_blmerge(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		uint8* out4, uint8* out5, uint8* out6, uint8* out7);
	void run_maps_range_x8_blall(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		uint8* out4, uint8* out5, uint8* out6, uint8* out7);
	// MAP1 (entry 0 only) + per-lane shed-proxy score (the producer routing signal).
	// Mirrors the AVX-512 run_map1_range_x8_scores.
	void run_map1_range_x8_scores(const key_schedule& schedule_entries,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		uint8* out4, uint8* out5, uint8* out6, uint8* out7,
		float scores[8]);

	// Screen one post-maps state (natural layout); emit world/flags on a checksum
	// hit. Mirrors the AVX-512 screen_state_raw used by the raceway screen path.
	bool screen_state_raw(const uint8* src, uint8& flags_out);

	// Interleaved (2-way) bulk forward + screen. Batches 2 independent candidates
	// through run_maps_range_x2, then screens each; the tail (<2) falls back to
	// 1-way.
	void run_bruteforce_data_il(uint32 key, uint32 data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size);

private:
	// One schedule entry over two interleaved states (4 YMM each).
	void _run_map_entry_x2(__m256i& a0, __m256i& a1, __m256i& a2, __m256i& a3,
	                       __m256i& b0, __m256i& b1, __m256i& b2, __m256i& b3, int map_idx);
	// One schedule entry over four interleaved states (16 YMM; spills the butterfly
	// temps — the aggressive AVX2 interleave point).
	void _run_map_entry_x4(__m256i& a0, __m256i& a1, __m256i& a2, __m256i& a3,
	                       __m256i& b0, __m256i& b1, __m256i& b2, __m256i& b3,
	                       __m256i& c0, __m256i& c1, __m256i& c2, __m256i& c3,
	                       __m256i& d0, __m256i& d1, __m256i& d2, __m256i& d3, int map_idx);
	// Resolve [begin,end) against the bound owned/shared tables; outputs the local
	// (table-relative) map range. Shared bind preamble for the x2/x4/x8 sweeps.
	void _ileave_resolve(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
	                     std::size_t& local_begin, std::size_t& local_end);
	// Register-resident sweeps of the local map range [lb,le) over 2 / 4 lanes
	// (tables assumed already bound by _ileave_resolve).
	void _sweep_x2(const uint8* in0, const uint8* in1, uint8* out0, uint8* out1,
	               std::size_t lb, std::size_t le);
	void _sweep_x4(const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
	               uint8* out0, uint8* out1, uint8* out2, uint8* out3,
	               std::size_t lb, std::size_t le);

	// Branchless alg dispatch (AVX2 port of _alg_dispatch_blmerge). Folds the arith
	// algs {0,1,3,4,6,7} into 3 candidates + a blendv tree (zero branches); the
	// butterflies {2,5} stay branched. Removes the 8-way switch's ~23% mispredict.
	// Bit-identical to _run_alg. Operates on one state's 4 YMM.
	void _alg_dispatch_blmerge(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3,
	                           int alg_id, uint16& local_pos,
	                           const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base);
	void _run_map_entry_x2_blmerge(__m256i& a0, __m256i& a1, __m256i& a2, __m256i& a3,
	                               __m256i& b0, __m256i& b1, __m256i& b2, __m256i& b3, int map_idx);
	void _run_map_entry_x4_blmerge(__m256i& a0, __m256i& a1, __m256i& a2, __m256i& a3,
	                               __m256i& b0, __m256i& b1, __m256i& b2, __m256i& b3,
	                               __m256i& c0, __m256i& c1, __m256i& c2, __m256i& c3,
	                               __m256i& d0, __m256i& d1, __m256i& d2, __m256i& d3, int map_idx);
	void _sweep_x2_blmerge(const uint8* in0, const uint8* in1, uint8* out0, uint8* out1,
	                       std::size_t lb, std::size_t le);
	void _sweep_x4_blmerge(const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
	                       uint8* out0, uint8* out1, uint8* out2, uint8* out3,
	                       std::size_t lb, std::size_t le);

	// Screen one post-maps state (natural layout) and emit a 5-byte record on a
	// checksum hit. Shared by the 1-way and interleaved bulk paths.
	void _screen_emit(__m256i w0, __m256i w1, __m256i w2, __m256i w3, uint32 idx, uint8* result_data, uint32& output_pos);

	void initialize();

	void _bind_schedule(const key_schedule& schedule_entries);
	void _bind_key(uint32 new_key);

	// Active tables (either _owned or *_shared).
	map_tables_shared::Tables& _t() { return _shared ? *_shared : _owned; }
	const map_tables_shared::Tables& _t() const { return _shared ? *_shared : _owned; }

	void alg_0(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, const uint8* block_start);
	void alg_1(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, const uint8* block_start);
	void alg_2(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, uint8 carry_byte);
	void alg_3(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, const uint8* block_start);
	void alg_4(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, const uint8* block_start);
	void alg_5(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, uint8 carry_byte);
	void alg_6(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, const uint8* block_start);
	void alg_7(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3);

	void _run_alg(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3,
	              int algorithm_id, uint16* local_pos,
	              const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base);

	void _run_one_map(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, int map_idx);
	void _run_maps_fixed(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, int map_idx, int count);
	void _run_all_maps(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3);

	void _load_from_mem(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3);
	void _store_to_mem(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3);

	void _expand_code(uint32 data, __m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3);

	void _decrypt_carnival_world(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3);
	void _decrypt_other_world(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3);

	bool check_carnival_world_checksum(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3);
	bool check_other_world_checksum(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3);

	uint16 masked_checksum(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, const uint8* mask);
	void mid_sum(__m128i& sum, __m256i& wc, __m256i& sum_mask, __m128i& lo_mask);
	uint16 fetch_checksum_value(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, uint8 code_length);
	void xor_alg(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, const uint8* values);

	uint32 _key = 0;

	// Per-instance fallback tables (allocated only when _shared == nullptr).
	// When _shared != nullptr we route reads through it instead.
	map_tables_shared::Tables  _owned;
	map_tables_shared::Tables* _shared = nullptr;   // non-null = shared-mode

	uint8* _expansion_for_seed_128 = nullptr;       // pointer into rng->expansion_values_8 (no malloc)

	const __m256i mask_FF;
	const __m256i mask_FE;
	const __m256i mask_7F;
	const __m256i mask_80;
	const __m256i mask_01;

	ALIGNED(32) uint8 working_code_data[128];

	static bool initialized;
};
#endif // TM_AVX2_R256_MAP_8_H
