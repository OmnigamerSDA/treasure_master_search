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

private:
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

	void alg_2_sub(__m256i& wc, __m256i& carry);
	void alg_5_sub(__m256i& wc, __m256i& carry);

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
	const __m256i mask_top_01;
	const __m256i mask_top_80;

	ALIGNED(32) uint8 working_code_data[128];

	static bool initialized;
};
#endif // TM_AVX2_R256_MAP_8_H
