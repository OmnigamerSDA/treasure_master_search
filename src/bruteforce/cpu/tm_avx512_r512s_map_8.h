#ifndef TM_AVX512_R512S_MAP_8_H
#define TM_AVX512_R512S_MAP_8_H
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

// AVX-512 map-mode forward kernel, SHUFFLED byte layout.
//
// State is held in the SAME shuffled 2-ZMM layout as tm_avx512_r512s_8 (the
// universal AVX-512 kernel): physical reg0 byte p = canonical byte 2p, reg1
// byte p = canonical byte 2p+1 (shuffle(addr) = (addr%2)*64 + (addr/2)%64).
// The alg MATH is identical to tm_avx512_r512s_8.
//
// The RNG source is map-mode (NATURAL flat 2048-byte per-entry blocks built by
// map_tables_shared, walked by a moving local_pos — same blocks the NATURAL
// tm_avx512_r512_map_8 uses). At each row-alg load, _load_deinterleave converts
// the natural 128-byte window into the shuffled r0/r1 form so it matches the
// shuffled state registers. alg_2/alg_5 take the carry directly from the block
// byte (micro500-style), not from a precomputed shuffled alg2/alg5 table.
//
// This is the shuffled analog of tm_avx512_r512_map_8; only the RNG operand
// source and the byte layout differ from tm_avx512_r512s_8.

class tm_avx512_r512s_map_8 : public TM_base
{
public:
	static constexpr int DEDUP_STATE_SHUFFLE = 512;

	tm_avx512_r512s_map_8(RNG* rng);
	tm_avx512_r512s_map_8(RNG* rng, map_tables_shared::Tables* shared_tables);

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

	// 4-way interleaved map-range kernel (mirrors tm_avx512_r512s_8 and
	// tm_avx512_r512_map_8). Shuffled layout, so in/out are shuffled 128-byte
	// states (same as state_raw / load_state_raw).
	void run_maps_range_x4(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3);

	// state_dedup interface (compile-time concept; no virtual). Shuffled layout,
	// so state_raw bytes are the impl's shuffled 128-byte state.
	const uint8* state_raw() const { return working_code_data; }
	void load_state_raw(const uint8* src) { for (int i = 0; i < 128; i++) working_code_data[i] = src[i]; }

	// Native bulk forward + checksum screen (the production NON-dedup path).
	// Mirrors tm_avx512_r512_map_8 / tm_avx2_r256_map_8 but on the shuffled
	// 2-ZMM layout: the checksum masks and world-decrypt operands are kept in a
	// pre-shuffled copy so the per-byte AND/XOR line up with the shuffled state.
	void run_bruteforce_data(uint32 key, uint32 data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size) override;

	// Interleaved (4-way) bulk forward + screen. Batches 4 independent
	// candidates through the validated run_maps_range_x4, then screens each
	// (shuffled layout); the tail (<4) falls back to 1-way.
	void run_bruteforce_data_il(uint32 key, uint32 data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size);

private:
	// Screen one post-maps state (shuffled layout) and emit a 5-byte record on a
	// checksum hit. Shared by the 1-way and interleaved bulk paths.
	void _screen_emit(__m512i w0, __m512i w1, uint32 idx, uint8* result_data, uint32& output_pos);

	void initialize();

	void _bind_schedule(const key_schedule& schedule_entries);

	map_tables_shared::Tables& _t() { return _shared ? *_shared : _owned; }
	const map_tables_shared::Tables& _t() const { return _shared ? *_shared : _owned; }

	// Deinterleave a natural 128-byte RNG window into the shuffled r0/r1 form:
	// even canonical bytes -> r0, odd canonical bytes -> r1 (ported verbatim
	// from micro500's tm_avx512bwvl_r512s_map_8).
	void _load_deinterleave(const uint8* block_start, __m512i& r0, __m512i& r1);

	void alg_0(__m512i& wc0, __m512i& wc1, const uint8* block_start);
	void alg_1(__m512i& wc0, __m512i& wc1, const uint8* block_start);
	void alg_2(__m512i& wc0, __m512i& wc1, uint8 carry_byte);
	void alg_3(__m512i& wc0, __m512i& wc1, const uint8* block_start);
	void alg_4(__m512i& wc0, __m512i& wc1, const uint8* block_start);
	void alg_5(__m512i& wc0, __m512i& wc1, uint8 carry_byte);
	void alg_6(__m512i& wc0, __m512i& wc1, const uint8* block_start);
	void alg_7(__m512i& wc0, __m512i& wc1);

	void alg_2_sub(__m512i& working_a, __m512i& working_b, __m512i& carry);
	void alg_5_sub(__m512i& working_a, __m512i& working_b, __m512i& carry);

	void _run_alg(__m512i& wc0, __m512i& wc1,
	              int algorithm_id, uint16* local_pos,
	              const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base);

	void _run_one_map(__m512i& wc0, __m512i& wc1, int map_idx);
	void _run_maps_fixed(__m512i& wc0, __m512i& wc1, int map_idx, int count);

	// One alg step for a single state (shared by 1-way and 4-way kernels).
	void _alg_dispatch(__m512i& w0, __m512i& w1, int alg_id, uint16& local_pos,
	                   const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base);

	// One schedule entry over four interleaved states.
	void _run_map_entry_x4(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
		__m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		int map_idx);

	void _load_from_mem(__m512i& wc0, __m512i& wc1);
	void _store_to_mem(__m512i& wc0, __m512i& wc1);

	void _expand_code(uint32 data, __m512i& wc0, __m512i& wc1);

	int shuffle(int addr);

	// Screening helpers (shuffled layout). The masks/world-data operands are
	// the pre-shuffled instance copies built in the constructor.
	void _run_all_maps(__m512i& wc0, __m512i& wc1);
	void _decrypt_carnival_world(__m512i& wc0, __m512i& wc1);
	void _decrypt_other_world(__m512i& wc0, __m512i& wc1);
	bool check_carnival_world_checksum(__m512i& wc0, __m512i& wc1);
	bool check_other_world_checksum(__m512i& wc0, __m512i& wc1);
	uint16 masked_checksum(__m512i& wc0, __m512i& wc1, const uint8* mask);
	uint16 fetch_checksum_value(__m512i& wc0, __m512i& wc1, uint8 code_length);
	void xor_alg(__m512i& wc0, __m512i& wc1, const uint8* values);

	uint32 _key = 0;

	map_tables_shared::Tables  _owned;
	map_tables_shared::Tables* _shared = nullptr;   // non-null = shared-mode

	const __m512i mask_FF;
	const __m512i mask_FE;
	const __m512i mask_7F;
	const __m512i mask_80;
	const __m512i mask_01;
	const __m512i mask_top_01;
	const __m512i mask_top_80;

	ALIGNED(64) uint8 working_code_data[128];

	// Pre-shuffled (physical-layout) copies of the world-decrypt operands and
	// checksum masks, built once in the constructor via shuffle_mem(..., 512).
	ALIGNED(64) uint8 carnival_world_data_shuf[128];
	ALIGNED(64) uint8 other_world_data_shuf[128];
	ALIGNED(64) uint8 carnival_checksum_mask_shuf[128];
	ALIGNED(64) uint8 other_checksum_mask_shuf[128];

	static bool initialized;
};
#endif // TM_AVX512_R512S_MAP_8_H
