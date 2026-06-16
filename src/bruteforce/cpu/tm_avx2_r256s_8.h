#ifndef TM_AVX2_R256S_8_H
#define TM_AVX2_R256S_8_H
#include <mmintrin.h>  //MMX
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSE4.2
//#include <ammintrin.h> //SSE4A
#include <immintrin.h> //AVX
//#include <zmmintrin.h> //AVX512

#include <cstddef>

#include "data_sizes.h"
#include "alignment2.h"
#include "rng_obj.h"
#include "tm_base.h"

class tm_avx2_r256s_8 : public TM_base
{
public:
	static constexpr int DEDUP_STATE_SHUFFLE = 256;

	tm_avx2_r256s_8(RNG* rng);

	virtual void load_data(uint8* new_data);
	void fetch_data(uint8* new_data);

	virtual void expand(uint32 key, uint32 data);

	void decrypt_carnival_world();
	void decrypt_other_world();

	uint16 calculate_carnival_world_checksum();
	uint16 calculate_other_world_checksum();

	uint16 fetch_carnival_world_checksum_value();
	uint16 fetch_other_world_checksum_value();

	virtual void run_alg(int algorithm_id, uint16* rng_seed, int iterations);

	virtual void run_one_map(const key_schedule::key_schedule_entry& schedule_entry);

	virtual void run_all_maps(const key_schedule& schedule_entries);
	void run_maps_range(const key_schedule& schedule_entries, std::size_t begin, std::size_t end);

	// PROTOTYPE: 2-way interleaved map-range kernel. Runs the schedule entries
	// [begin,end) over TWO independent states in lockstep, holding both states'
	// 8 YMM registers live across the whole group so the OOO engine overlaps
	// each step's dependent-SIMD/RNG-load latency for state 0 with state 1's
	// independent work. Reads from in0/in1 (impl shuffled layout, e.g. pool
	// entries), writes results to out0/out1. Used by the interleaved dedup
	// driver in state_dedup_il.h to process the frontier pool two states at a
	// time. See docs analysis: frontier interleaving targets the dedup path's
	// compute-latency + branch-mispredict + hash-probe bound (not DRAM).
	void run_maps_range_x2(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
	                       const uint8* in0, const uint8* in1, uint8* out0, uint8* out1);

	// Raw internal-state accessors for dedup POCs. The layout is the impl's
	// shuffled representation, NOT canonical — but it's invariant across
	// expand/run_one_map calls (each ends with _store_to_mem), so hashing
	// and replay can stay in the shuffled form without round-tripping
	// through load_data/fetch_data (which would shuffle/unshuffle each call).
	const uint8* state_raw() const { return working_code_data; }
	void load_state_raw(const uint8* src) { for (int i = 0; i < 128; i++) working_code_data[i] = src[i]; }

	void run_bruteforce_data(uint32 key, uint32 data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size);

private:
	void initialize();
	void alg_0(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint16* rng_seed, __m256i& mask_FE);
	void alg_2(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint16* rng_seed, __m256i& mask_top_01, __m256i& mask_80, __m256i& mask_7F, __m256i& mask_FE, __m256i& mask_01);
	void alg_3(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint16* rng_seed);
	void alg_5(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint16* rng_seed, __m256i& mask_top_80, __m256i& mask_80, __m256i& mask_7F, __m256i& mask_FE, __m256i& mask_01);
	void alg_6(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint16* rng_seed, __m256i& mask_7F);
	void alg_7(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, __m256i& mask_FF);
	void xor_alg(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint8* values);
	void add_alg(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint16* rng_seed, uint8* rng_start);
	void sub_alg(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint16* rng_seed, uint8* rng_start);
	void alg_2_sub(__m256i& working_a, __m256i& working_b, __m256i& carry, __m256i& mask_top_01, __m256i& mask_80, __m256i& mask_7F, __m256i& mask_FE, __m256i& mask_01);
	void alg_5_sub(__m256i& working_a, __m256i& working_b, __m256i& carry, __m256i& mask_top_80, __m256i& mask_80, __m256i& mask_7F, __m256i& mask_FE, __m256i& mask_01);

	void _run_all_maps(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, const key_schedule& schedule_entries, __m256i& mask_FF, __m256i& mask_FE, __m256i& mask_7F, __m256i& mask_80, __m256i& mask_01, __m256i& mask_top_01, __m256i& mask_top_80);
	void _run_map_entry(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, const key_schedule::key_schedule_entry& schedule_entry, __m256i& mask_FF, __m256i& mask_FE, __m256i& mask_7F, __m256i& mask_80, __m256i& mask_01, __m256i& mask_top_01, __m256i& mask_top_80);

	// PROTOTYPE: one schedule entry over two interleaved states (see run_maps_range_x2).
	void _run_map_entry_x2(
		__m256i& a0, __m256i& a1, __m256i& a2, __m256i& a3,
		__m256i& b0, __m256i& b1, __m256i& b2, __m256i& b3,
		const key_schedule::key_schedule_entry& schedule_entry);

	// PROTOTYPE tier-1 alg variants: build the 1-2 needed constants locally
	// (only 0x0101/0x8080; 0xFE=~0x01 and 0x7F=~0x80 folded via andnot;
	// 0xFFFF via cmpeq) so the register allocator can rematerialize them under
	// the 8-state-register pressure of the 2-way kernel instead of pinning 7
	// masks. add_alg/sub_alg/alg_3 need no masks and are reused as-is.
	void alg_0_il(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3, uint16* rng_seed);
	void alg_2_il(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3, uint16* rng_seed);
	void alg_5_il(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3, uint16* rng_seed);
	void alg_6_il(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3, uint16* rng_seed);
	void alg_7_il(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3);
	void alg_2_sub_il(__m256i& working_a, __m256i& working_b, __m256i& carry);
	void alg_5_sub_il(__m256i& working_a, __m256i& working_b, __m256i& carry);

	void _decrypt_carnival_world(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3);
	void _decrypt_other_world(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3);

	uint16 _calculate_carnival_world_checksum(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3);
	uint16 _calculate_other_world_checksum(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3);

	uint16 _fetch_carnival_world_checksum_value(__m256i& working_code0, __m256i& working_code1);
	uint16 _fetch_other_world_checksum_value(__m256i& working_code0, __m256i& working_code1);

	bool check_carnival_world_checksum(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3);
	bool check_other_world_checksum(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3);

	uint16 masked_checksum(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint8* mask);
	void mid_sum(__m128i& sum, __m256i& working_code, __m256i& sum_mask, __m128i& lo_mask);
	uint16 fetch_checksum_value(__m256i& working_code0, __m256i& working_code1, uint8 code_length);
	
	void _load_from_mem(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3);
	void _store_to_mem(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3);

	void _expand_code(uint32 key, uint32 data, __m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3);

	int shuffle(int addr);

	ALIGNED(32) uint8 working_code_data[128];

	ALIGNED(32) static uint8 carnival_world_checksum_mask_shuffled[128];
	ALIGNED(32) static uint8 carnival_world_data_shuffled[128];

	ALIGNED(32) static uint8 other_world_checksum_mask_shuffled[128];
	ALIGNED(32) static uint8 other_world_data_shuffled[128];

	static bool initialized;
};
#endif // TM_AVX2_R256S_8_H
