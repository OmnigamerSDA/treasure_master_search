#include <stdio.h>
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

#include "data_sizes.h"
#include "tm_avx2_r256s_8.h"

#if defined(__GNUC__)
#define _mm256_set_m128i(vh, vl) \
        _mm256_castpd_si256(_mm256_insertf128_pd(_mm256_castsi256_pd(_mm256_castsi128_si256(vl)), _mm_castsi128_pd(vh), 1))
#endif

tm_avx2_r256s_8::tm_avx2_r256s_8(RNG* rng_obj) : TM_base(rng_obj)
{
	initialize();
}

__forceinline void tm_avx2_r256s_8::initialize()
{
	if (!initialized)
	{
		shuffle_mem(carnival_world_checksum_mask, carnival_world_checksum_mask_shuffled, 256, false);
		shuffle_mem(carnival_world_data, carnival_world_data_shuffled, 256, false);

		shuffle_mem(other_world_checksum_mask, other_world_checksum_mask_shuffled, 256, false);
		shuffle_mem(other_world_data, other_world_data_shuffled, 256, false);

		rng->generate_expansion_values_256_8_shuffled();

		rng->generate_seed_forward_1();
		rng->generate_seed_forward_128();

		rng->generate_regular_rng_values_8();
		rng->generate_regular_rng_values_256_8_shuffled();

		rng->generate_alg0_values_256_8_shuffled();
		rng->generate_alg2_values_256_8();
		rng->generate_alg5_values_256_8();
		rng->generate_alg6_values_256_8_shuffled();

		initialized = true;
	}
	obj_name = "tm_avx2_r256s_8";
}

__forceinline void tm_avx2_r256s_8::_store_to_mem(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3)
{
	_mm256_store_si256((__m256i*)(working_code_data), working_code0);
	_mm256_store_si256((__m256i*)(working_code_data + 32), working_code1);
	_mm256_store_si256((__m256i*)(working_code_data + 64), working_code2);
	_mm256_store_si256((__m256i*)(working_code_data + 96), working_code3);
}

__forceinline void tm_avx2_r256s_8::_load_from_mem(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3)
{
	working_code0 = _mm256_load_si256((__m256i*)(working_code_data));
	working_code1 = _mm256_load_si256((__m256i*)(working_code_data + 32));
	working_code2 = _mm256_load_si256((__m256i*)(working_code_data + 64));
	working_code3 = _mm256_load_si256((__m256i*)(working_code_data + 96));
}

int tm_avx2_r256s_8::shuffle(int addr)
{
	return (addr / 64) * 64 + (addr % 2) * 32 + ((addr / 2) % 32);
}

void tm_avx2_r256s_8::expand(uint32 key, uint32 data)
{
	__m256i working_code0;
	__m256i working_code1;
	__m256i working_code2;
	__m256i working_code3;

	_expand_code(key, data, working_code0, working_code1, working_code2, working_code3);

	_store_to_mem(working_code0, working_code1, working_code2, working_code3);
}

__forceinline void tm_avx2_r256s_8::_expand_code(uint32 key, uint32 data, __m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3)
{
	uint64 x = ((uint64)key << 32) | data;

	__m128i a = _mm_cvtsi64_si128(x);

	__m128i lo_mask = _mm_set_epi8(1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7);
	__m128i hi_mask = _mm_set_epi8(0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6);

	__m128i lo_128 = _mm_shuffle_epi8(a, lo_mask);
	__m128i hi_128 = _mm_shuffle_epi8(a, hi_mask);

	__m256i lo = _mm256_setr_m128i(lo_128, lo_128);
	__m256i hi = _mm256_setr_m128i(hi_128, hi_128);

	working_code0 = lo;
	working_code1 = hi;
	working_code2 = lo;
	working_code3 = hi;

	uint8* rng_start = rng->expansion_values_256_8_shuffled;
	uint16 rng_seed = (key >> 16) & 0xFFFF;

	add_alg(working_code0, working_code1, working_code2, working_code3, &rng_seed, rng_start);
}

void tm_avx2_r256s_8::load_data(uint8* new_data)
{
	for (int i = 0; i < 128; i++)
	{
		((uint8*)working_code_data)[(i / 64) * 64 + (i % 2) * 32 + ((i / 2) % 32)] = new_data[i];
	}

}

void tm_avx2_r256s_8::fetch_data(uint8* new_data)
{
	for (int i = 0; i < 128; i++)
	{
		new_data[i] = ((uint8*)working_code_data)[shuffle_8(i, 256)];
	}
}

__forceinline void tm_avx2_r256s_8::run_alg(int algorithm_id, uint16* rng_seed, int iterations)
{
	__m256i working_code0 = _mm256_load_si256((__m256i*)(working_code_data));
	__m256i working_code1 = _mm256_load_si256((__m256i*)(working_code_data + 32));
	__m256i working_code2 = _mm256_load_si256((__m256i*)(working_code_data + 64));
	__m256i working_code3 = _mm256_load_si256((__m256i*)(working_code_data + 96));

	__m256i mask_FF = _mm256_set1_epi16(0xFFFF);
	__m256i mask_FE = _mm256_set1_epi16(0xFEFE);
	__m256i mask_7F = _mm256_set1_epi16(0x7F7F);
	__m256i mask_80 = _mm256_set1_epi16(0x8080);
	__m256i mask_01 = _mm256_set1_epi16(0x0101);

	__m256i mask_top_01 = _mm256_set_epi16(0x0100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	__m256i mask_top_80 = _mm256_set_epi16(0x8000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	if (algorithm_id == 0)
	{
		for (int j = 0; j < iterations; j++)
		{
			alg_0(working_code0, working_code1, working_code2, working_code3, rng_seed, mask_FE);
			*rng_seed = rng->seed_forward_128[*rng_seed];
		}
	}
	else if (algorithm_id == 1)
	{
		for (int j = 0; j < iterations; j++)
		{
			add_alg(working_code0, working_code1, working_code2, working_code3, rng_seed, rng->regular_rng_values_256_8_shuffled);
			*rng_seed = rng->seed_forward_128[*rng_seed];
		}
	}
	else if (algorithm_id == 4)
	{
		for (int j = 0; j < iterations; j++)
		{
			sub_alg(working_code0, working_code1, working_code2, working_code3, rng_seed, rng->regular_rng_values_256_8_shuffled);
			*rng_seed = rng->seed_forward_128[*rng_seed];
		}
	}
	else if (algorithm_id == 2)
	{
		for (int j = 0; j < iterations; j++)
		{
			alg_2(working_code0, working_code1, working_code2, working_code3, rng_seed, mask_top_01, mask_80, mask_7F, mask_FE, mask_01);
			*rng_seed = rng->seed_forward_1[*rng_seed];
		}
	}
	else if (algorithm_id == 3)
	{
		for (int j = 0; j < iterations; j++)
		{
			alg_3(working_code0, working_code1, working_code2, working_code3, rng_seed);
			*rng_seed = rng->seed_forward_128[*rng_seed];
		}
	}
	else if (algorithm_id == 5)
	{
		for (int j = 0; j < iterations; j++)
		{
			alg_5(working_code0, working_code1, working_code2, working_code3, rng_seed, mask_top_80, mask_80, mask_7F, mask_FE, mask_01);
			*rng_seed = rng->seed_forward_1[*rng_seed];
		}
	}
	else if (algorithm_id == 6)
	{
		for (int j = 0; j < iterations; j++)
		{
			alg_6(working_code0, working_code1, working_code2, working_code3, rng_seed, mask_7F);
			*rng_seed = rng->seed_forward_128[*rng_seed];
		}
	}
	else if (algorithm_id == 7)
	{
		for (int j = 0; j < iterations; j++)
		{
			alg_7(working_code0, working_code1, working_code2, working_code3, mask_FF);
		}
	}

	_mm256_store_si256((__m256i*)(working_code_data), working_code0);
	_mm256_store_si256((__m256i*)(working_code_data + 32), working_code1);
	_mm256_store_si256((__m256i*)(working_code_data + 64), working_code2);
	_mm256_store_si256((__m256i*)(working_code_data + 96), working_code3);
}

__forceinline void tm_avx2_r256s_8::alg_0(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint16* rng_seed, __m256i& mask_FE)
{
	uint8* rng_start = rng->alg0_values_256_8_shuffled + ((*rng_seed) * 128);

	working_code0 = _mm256_slli_epi16(working_code0, 1);
	__m256i rng_val = _mm256_load_si256((__m256i*)(rng_start));
	working_code0 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(working_code0), _mm256_castsi256_pd(mask_FE)));
	working_code0 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(working_code0), _mm256_castsi256_pd(rng_val)));

	working_code1 = _mm256_slli_epi16(working_code1, 1);
	rng_val = _mm256_load_si256((__m256i*)(rng_start + 32));
	working_code1 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(working_code1), _mm256_castsi256_pd(mask_FE)));
	working_code1 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(working_code1), _mm256_castsi256_pd(rng_val)));

	working_code2 = _mm256_slli_epi16(working_code2, 1);
	rng_val = _mm256_load_si256((__m256i*)(rng_start + 64));
	working_code2 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(working_code2), _mm256_castsi256_pd(mask_FE)));
	working_code2 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(working_code2), _mm256_castsi256_pd(rng_val)));

	working_code3 = _mm256_slli_epi16(working_code3, 1);
	rng_val = _mm256_load_si256((__m256i*)(rng_start + 96));
	working_code3 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(working_code3), _mm256_castsi256_pd(mask_FE)));
	working_code3 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(working_code3), _mm256_castsi256_pd(rng_val)));
}

__forceinline void tm_avx2_r256s_8::alg_2_sub(__m256i& working_a, __m256i& working_b, __m256i& carry, __m256i& mask_top_01, __m256i& mask_80, __m256i& mask_7F, __m256i& mask_FE, __m256i& mask_01)
{
	// Shift bytes right 1 bit
	__m256i cur_val1_most = _mm256_and_si256(_mm256_srli_epi16(working_a, 1), mask_7F);
	// Shift bytes left 1 bit
	__m256i cur_val2_most = _mm256_and_si256(_mm256_slli_epi16(working_b, 1), mask_FE);
	
	// Mask off the top bits
	__m256i cur_val2_masked = _mm256_and_si256(working_b, mask_80);

	// Shift right 1 byte
	__m256i cur_val1_bit = _mm256_and_si256(working_a, mask_01);

	__m256i mask = _mm256_permute2x128_si256(cur_val1_bit, cur_val1_bit, _MM_SHUFFLE(3, 0, 0, 3));
	__m256i cur_val1_srl = _mm256_alignr_epi8(mask, cur_val1_bit, 1);
	__m256i cur_val1_srl_w_carry = _mm256_or_si256(cur_val1_srl, carry);

	// Save the next carry
	__m256i lo_to_hi = _mm256_permute2x128_si256(cur_val1_bit, cur_val1_bit, _MM_SHUFFLE(0, 0, 3, 0));
	__m256i next_carry = _mm256_bslli_epi128(lo_to_hi, 15);

	working_a = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(cur_val1_most), _mm256_castsi256_pd(cur_val2_masked)));
	working_b = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(cur_val2_most), _mm256_castsi256_pd(cur_val1_srl_w_carry)));

	carry = next_carry;
}

__forceinline void tm_avx2_r256s_8::alg_2(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint16* rng_seed, __m256i& mask_top_01, __m256i& mask_80, __m256i& mask_7F, __m256i& mask_FE, __m256i& mask_01)
{
	__m256i carry = _mm256_load_si256((__m256i*)(rng->alg2_values_256_8 + ((*rng_seed) * 32)));

	alg_2_sub(working_code2, working_code3, carry, mask_top_01, mask_80, mask_7F, mask_FE, mask_01);
	alg_2_sub(working_code0, working_code1, carry, mask_top_01, mask_80, mask_7F, mask_FE, mask_01);
}

__forceinline void tm_avx2_r256s_8::alg_3(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint16* rng_seed)
{
	uint8* rng_start = rng->regular_rng_values_256_8_shuffled + ((*rng_seed) * 128);

	__m256i rng_val = _mm256_load_si256((__m256i*)(rng_start));
	working_code0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(working_code0), _mm256_castsi256_pd(rng_val)));

	rng_val = _mm256_load_si256((__m256i*)(rng_start + 32));
	working_code1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(working_code1), _mm256_castsi256_pd(rng_val)));

	rng_val = _mm256_load_si256((__m256i*)(rng_start + 64));
	working_code2 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(working_code2), _mm256_castsi256_pd(rng_val)));

	rng_val = _mm256_load_si256((__m256i*)(rng_start + 96));
	working_code3 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(working_code3), _mm256_castsi256_pd(rng_val)));
}

__forceinline void tm_avx2_r256s_8::alg_5_sub(__m256i& working_a, __m256i& working_b, __m256i& carry, __m256i& mask_top_80, __m256i& mask_80, __m256i& mask_7F, __m256i& mask_FE, __m256i& mask_01)
{
	// Shift bytes right 1 bit
	__m256i cur_val1_most = _mm256_and_si256(_mm256_slli_epi16(working_a, 1), mask_FE);
	// Shift bytes left 1 bit
	__m256i cur_val2_most = _mm256_and_si256(_mm256_srli_epi16(working_b, 1), mask_7F);

	// Mask off the top bits
	__m256i cur_val2_masked = _mm256_and_si256(working_b, mask_01);

	// Shift right 1 byte
	__m256i cur_val1_bit = _mm256_and_si256(working_a, mask_80);

	__m256i mask = _mm256_permute2x128_si256(cur_val1_bit, cur_val1_bit, _MM_SHUFFLE(3, 0, 0, 3));
	__m256i cur_val1_srl = _mm256_alignr_epi8(mask, cur_val1_bit, 1);
	__m256i cur_val1_srl_w_carry = _mm256_or_si256(cur_val1_srl, carry);

	// Save the next carry
	__m256i lo_to_hi = _mm256_permute2x128_si256(cur_val1_bit, cur_val1_bit, _MM_SHUFFLE(0, 0, 3, 0));
	__m256i next_carry = _mm256_bslli_epi128(lo_to_hi, 15);

	working_a = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(cur_val1_most), _mm256_castsi256_pd(cur_val2_masked)));
	working_b = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(cur_val2_most), _mm256_castsi256_pd(cur_val1_srl_w_carry)));

	carry = next_carry;
}

__forceinline void tm_avx2_r256s_8::alg_5(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint16* rng_seed, __m256i& mask_top_80, __m256i& mask_80, __m256i& mask_7F, __m256i& mask_FE, __m256i& mask_01)
{
	__m256i carry = _mm256_load_si256((__m256i*)(rng->alg5_values_256_8 + ((*rng_seed) * 32)));

	alg_5_sub(working_code2, working_code3, carry, mask_top_80, mask_80, mask_7F, mask_FE, mask_01);
	alg_5_sub(working_code0, working_code1, carry, mask_top_80, mask_80, mask_7F, mask_FE, mask_01);
}

__forceinline void tm_avx2_r256s_8::alg_6(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint16* rng_seed, __m256i& mask_7F)
{
	uint8* rng_start = rng->alg6_values_256_8_shuffled + ((*rng_seed) * 128);

	working_code0 = _mm256_srli_epi16(working_code0, 1);
	__m256i rng_val = _mm256_load_si256((__m256i*)(rng_start));
	working_code0 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(working_code0), _mm256_castsi256_pd(mask_7F)));
	working_code0 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(working_code0), _mm256_castsi256_pd(rng_val)));

	working_code1 = _mm256_srli_epi16(working_code1, 1);
	rng_val = _mm256_load_si256((__m256i*)(rng_start + 32));
	working_code1 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(working_code1), _mm256_castsi256_pd(mask_7F)));
	working_code1 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(working_code1), _mm256_castsi256_pd(rng_val)));

	working_code2 = _mm256_srli_epi16(working_code2, 1);
	rng_val = _mm256_load_si256((__m256i*)(rng_start + 64));
	working_code2 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(working_code2), _mm256_castsi256_pd(mask_7F)));
	working_code2 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(working_code2), _mm256_castsi256_pd(rng_val)));

	working_code3 = _mm256_srli_epi16(working_code3, 1);
	rng_val = _mm256_load_si256((__m256i*)(rng_start + 96));
	working_code3 = _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(working_code3), _mm256_castsi256_pd(mask_7F)));
	working_code3 = _mm256_castpd_si256(_mm256_or_pd(_mm256_castsi256_pd(working_code3), _mm256_castsi256_pd(rng_val)));
}

__forceinline void tm_avx2_r256s_8::alg_7(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, __m256i& mask_FF)
{
	working_code0 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(working_code0), _mm256_castsi256_pd(mask_FF)));
	working_code1 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(working_code1), _mm256_castsi256_pd(mask_FF)));
	working_code2 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(working_code2), _mm256_castsi256_pd(mask_FF)));
	working_code3 = _mm256_castpd_si256(_mm256_xor_pd(_mm256_castsi256_pd(working_code3), _mm256_castsi256_pd(mask_FF)));
}

__forceinline void tm_avx2_r256s_8::xor_alg(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint8* values)
{
	__m256i v = _mm256_load_si256((__m256i*)(values));
	working_code0 = _mm256_xor_si256(working_code0, v);

	v = _mm256_load_si256((__m256i*)(values + 32));
	working_code1 = _mm256_xor_si256(working_code1, v);

	v = _mm256_load_si256((__m256i*)(values + 64));
	working_code2 = _mm256_xor_si256(working_code2, v);

	v = _mm256_load_si256((__m256i*)(values + 96));
	working_code3 = _mm256_xor_si256(working_code3, v);
}

__forceinline void tm_avx2_r256s_8::add_alg(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint16* rng_seed, uint8* rng_start)
{
	rng_start = rng_start + ((*rng_seed) * 128);

	__m256i rng_val = _mm256_load_si256((__m256i*)(rng_start));
	working_code0 = _mm256_add_epi8(working_code0, rng_val);

	rng_val = _mm256_load_si256((__m256i*)(rng_start + 32));
	working_code1 = _mm256_add_epi8(working_code1, rng_val);

	rng_val = _mm256_load_si256((__m256i*)(rng_start + 64));
	working_code2 = _mm256_add_epi8(working_code2, rng_val);

	rng_val = _mm256_load_si256((__m256i*)(rng_start + 96));
	working_code3 = _mm256_add_epi8(working_code3, rng_val);
}

__forceinline void tm_avx2_r256s_8::sub_alg(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint16* rng_seed, uint8* rng_start)
{
	rng_start = rng_start + ((*rng_seed) * 128);

	__m256i rng_val = _mm256_load_si256((__m256i*)(rng_start));
	working_code0 = _mm256_sub_epi8(working_code0, rng_val);

	rng_val = _mm256_load_si256((__m256i*)(rng_start + 32));
	working_code1 = _mm256_sub_epi8(working_code1, rng_val);

	rng_val = _mm256_load_si256((__m256i*)(rng_start + 64));
	working_code2 = _mm256_sub_epi8(working_code2, rng_val);

	rng_val = _mm256_load_si256((__m256i*)(rng_start + 96));
	working_code3 = _mm256_sub_epi8(working_code3, rng_val);
}

void tm_avx2_r256s_8::run_one_map(const key_schedule::key_schedule_entry& schedule_entry)
{
	__m256i working_code0;
	__m256i working_code1;
	__m256i working_code2;
	__m256i working_code3;
	_load_from_mem(working_code0, working_code1, working_code2, working_code3);

	__m256i mask_FF = _mm256_set1_epi16(0xFFFF);
	__m256i mask_FE = _mm256_set1_epi16(0xFEFE);
	__m256i mask_7F = _mm256_set1_epi16(0x7F7F);
	__m256i mask_80 = _mm256_set1_epi16(0x8080);
	__m256i mask_01 = _mm256_set1_epi16(0x0101);
	__m256i mask_top_01 = _mm256_set_epi16(0x0100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	__m256i mask_top_80 = _mm256_set_epi16(0x8000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	_run_map_entry(working_code0, working_code1, working_code2, working_code3, schedule_entry, mask_FF, mask_FE, mask_7F, mask_80, mask_01, mask_top_01, mask_top_80);
	_store_to_mem(working_code0, working_code1, working_code2, working_code3);
}

void tm_avx2_r256s_8::run_all_maps(const key_schedule& schedule_entries)
{
	run_maps_range(schedule_entries, 0, schedule_entries.entries.size());
}

void tm_avx2_r256s_8::run_maps_range(const key_schedule& schedule_entries, std::size_t begin, std::size_t end)
{
	__m256i working_code0;
	__m256i working_code1;
	__m256i working_code2;
	__m256i working_code3;
	_load_from_mem(working_code0, working_code1, working_code2, working_code3);

	__m256i mask_FF = _mm256_set1_epi16(0xFFFF);
	__m256i mask_FE = _mm256_set1_epi16(0xFEFE);
	__m256i mask_7F = _mm256_set1_epi16(0x7F7F);
	__m256i mask_80 = _mm256_set1_epi16(0x8080);
	__m256i mask_01 = _mm256_set1_epi16(0x0101);

	__m256i mask_top_01 = _mm256_set_epi16(0x0100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	__m256i mask_top_80 = _mm256_set_epi16(0x8000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	if (end > schedule_entries.entries.size())
		end = schedule_entries.entries.size();
	for (std::size_t map_idx = begin; map_idx < end; map_idx++)
	{
		_run_map_entry(working_code0, working_code1, working_code2, working_code3, schedule_entries.entries[map_idx], mask_FF, mask_FE, mask_7F, mask_80, mask_01, mask_top_01, mask_top_80);
	}

	_store_to_mem(working_code0, working_code1, working_code2, working_code3);
}

__forceinline void tm_avx2_r256s_8::_run_all_maps(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, const key_schedule& schedule_entries, __m256i& mask_FF, __m256i& mask_FE, __m256i& mask_7F, __m256i& mask_80, __m256i& mask_01, __m256i& mask_top_01, __m256i& mask_top_80)
{
	for (std::vector<key_schedule::key_schedule_entry>::const_iterator it = schedule_entries.entries.begin(); it != schedule_entries.entries.end(); it++)
	{
		_run_map_entry(working_code0, working_code1, working_code2, working_code3, *it, mask_FF, mask_FE, mask_7F, mask_80, mask_01, mask_top_01, mask_top_80);
	}
}

__forceinline void tm_avx2_r256s_8::_run_map_entry(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, const key_schedule::key_schedule_entry& schedule_entry, __m256i& mask_FF, __m256i& mask_FE, __m256i& mask_7F, __m256i& mask_80, __m256i& mask_01, __m256i& mask_top_01, __m256i& mask_top_80)
{
	uint16 rng_seed = (schedule_entry.rng1 << 8) | schedule_entry.rng2;
	uint16 nibble_selector = schedule_entry.nibble_selector;

	// Next, the working code is processed with the same steps 16 times.
	// #pragma GCC unroll 16 enables _mm_extract_epi8 to receive compile-time
	// constant indices so we can read the algorithm-select byte directly from
	// AVX registers without a store+load round-trip. MSVC does not honor that
	// pragma, so it gets explicit immediate-index cases below.
#if defined(__GNUC__)
#pragma GCC unroll 16
#endif
	for (int i = 0; i < 16; i++)
	{
		// Get the highest bit of the nibble selector to use as a flag
		unsigned char nibble = (nibble_selector >> 15) & 0x01;
		// Shift the nibble selector up one bit
		nibble_selector = nibble_selector << 1;

		// Extract byte directly from register (avoids 2 AVX2 stores per iteration).
		// shuffle_8(i, 256) for i=0..15: even i -> byte i/2 of working_code0,
		//                                odd i -> byte (i-1)/2 of working_code1.
		unsigned char current_byte = 0;
#if defined(_MSC_VER) || defined(__clang__)
		const __m128i working_code0_low = _mm256_castsi256_si128(working_code0);
		const __m128i working_code1_low = _mm256_castsi256_si128(working_code1);
		switch (i)
		{
			case 0: current_byte = (unsigned char)_mm_extract_epi8(working_code0_low, 0); break;
			case 1: current_byte = (unsigned char)_mm_extract_epi8(working_code1_low, 0); break;
			case 2: current_byte = (unsigned char)_mm_extract_epi8(working_code0_low, 1); break;
			case 3: current_byte = (unsigned char)_mm_extract_epi8(working_code1_low, 1); break;
			case 4: current_byte = (unsigned char)_mm_extract_epi8(working_code0_low, 2); break;
			case 5: current_byte = (unsigned char)_mm_extract_epi8(working_code1_low, 2); break;
			case 6: current_byte = (unsigned char)_mm_extract_epi8(working_code0_low, 3); break;
			case 7: current_byte = (unsigned char)_mm_extract_epi8(working_code1_low, 3); break;
			case 8: current_byte = (unsigned char)_mm_extract_epi8(working_code0_low, 4); break;
			case 9: current_byte = (unsigned char)_mm_extract_epi8(working_code1_low, 4); break;
			case 10: current_byte = (unsigned char)_mm_extract_epi8(working_code0_low, 5); break;
			case 11: current_byte = (unsigned char)_mm_extract_epi8(working_code1_low, 5); break;
			case 12: current_byte = (unsigned char)_mm_extract_epi8(working_code0_low, 6); break;
			case 13: current_byte = (unsigned char)_mm_extract_epi8(working_code1_low, 6); break;
			case 14: current_byte = (unsigned char)_mm_extract_epi8(working_code0_low, 7); break;
			case 15: current_byte = (unsigned char)_mm_extract_epi8(working_code1_low, 7); break;
		}
#else
		current_byte = ((i & 1) == 0)
			? (unsigned char)_mm_extract_epi8(_mm256_castsi256_si128(working_code0), i >> 1)
			: (unsigned char)_mm_extract_epi8(_mm256_castsi256_si128(working_code1), (i - 1) >> 1);
#endif

		if (nibble == 1)
		{
			current_byte = current_byte >> 4;
		}

		// Mask off only 3 bits
		unsigned char algorithm_id = (current_byte >> 1) & 0x07;

		// Switch over consecutive 0..7 cases — gcc emits a jump table backed by
		// a single indirect branch (predicted via BTB) rather than an if/else
		// cascade with 7 conditional branches. Confirmed +20% throughput vs
		// the cascade form on Zen 5.
		switch (algorithm_id)
		{
			case 0:
				alg_0(working_code0, working_code1, working_code2, working_code3, &rng_seed, mask_FE);
				rng_seed = rng->seed_forward_128[rng_seed];
				break;
			case 1:
				add_alg(working_code0, working_code1, working_code2, working_code3, &rng_seed, rng->regular_rng_values_256_8_shuffled);
				rng_seed = rng->seed_forward_128[rng_seed];
				break;
			case 2:
				alg_2(working_code0, working_code1, working_code2, working_code3, &rng_seed, mask_top_01, mask_80, mask_7F, mask_FE, mask_01);
				rng_seed = rng->seed_forward_1[rng_seed];
				break;
			case 3:
				alg_3(working_code0, working_code1, working_code2, working_code3, &rng_seed);
				rng_seed = rng->seed_forward_128[rng_seed];
				break;
			case 4:
				sub_alg(working_code0, working_code1, working_code2, working_code3, &rng_seed, rng->regular_rng_values_256_8_shuffled);
				rng_seed = rng->seed_forward_128[rng_seed];
				break;
			case 5:
				alg_5(working_code0, working_code1, working_code2, working_code3, &rng_seed, mask_top_80, mask_80, mask_7F, mask_FE, mask_01);
				rng_seed = rng->seed_forward_1[rng_seed];
				break;
			case 6:
				alg_6(working_code0, working_code1, working_code2, working_code3, &rng_seed, mask_7F);
				rng_seed = rng->seed_forward_128[rng_seed];
				break;
			default: // case 7 — already & 0x07 in caller
				alg_7(working_code0, working_code1, working_code2, working_code3, mask_FF);
				break;
		}
	}
}

// PROTOTYPE: one schedule entry over two interleaved states. Mirrors
// _run_map_entry's per-step extract/dispatch exactly, but runs both states'
// dependency chains adjacently so the OOO engine overlaps state 0's RNG-load /
// dependent-SIMD latency with state 1's independent work. nibble_selector is
// shared (same schedule entry for both states); the two RNG seeds are separate.
// tier-1 il alg variants — build only the 1-2 needed broadcast constants
// locally so they rematerialize under register pressure. 0xFE=~0x01 and
// 0x7F=~0x80 are produced with _mm256_andnot_si256 (vpandn, same op count as
// vpand), 0xFFFF with _mm256_cmpeq_epi8 (no load), so the persistent pinned
// mask set in the 2-way loop is empty.
__forceinline void tm_avx2_r256s_8::alg_0_il(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3, uint16* rng_seed)
{
	uint8* rng_start = rng->alg0_values_256_8_shuffled + ((*rng_seed) * 128);
	const __m256i m01 = _mm256_set1_epi16(0x0101);
	w0 = _mm256_or_si256(_mm256_andnot_si256(m01, _mm256_slli_epi16(w0, 1)), _mm256_load_si256((__m256i*)(rng_start)));
	w1 = _mm256_or_si256(_mm256_andnot_si256(m01, _mm256_slli_epi16(w1, 1)), _mm256_load_si256((__m256i*)(rng_start + 32)));
	w2 = _mm256_or_si256(_mm256_andnot_si256(m01, _mm256_slli_epi16(w2, 1)), _mm256_load_si256((__m256i*)(rng_start + 64)));
	w3 = _mm256_or_si256(_mm256_andnot_si256(m01, _mm256_slli_epi16(w3, 1)), _mm256_load_si256((__m256i*)(rng_start + 96)));
}

__forceinline void tm_avx2_r256s_8::alg_6_il(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3, uint16* rng_seed)
{
	uint8* rng_start = rng->alg6_values_256_8_shuffled + ((*rng_seed) * 128);
	const __m256i m80 = _mm256_set1_epi16(0x8080);
	w0 = _mm256_or_si256(_mm256_andnot_si256(m80, _mm256_srli_epi16(w0, 1)), _mm256_load_si256((__m256i*)(rng_start)));
	w1 = _mm256_or_si256(_mm256_andnot_si256(m80, _mm256_srli_epi16(w1, 1)), _mm256_load_si256((__m256i*)(rng_start + 32)));
	w2 = _mm256_or_si256(_mm256_andnot_si256(m80, _mm256_srli_epi16(w2, 1)), _mm256_load_si256((__m256i*)(rng_start + 64)));
	w3 = _mm256_or_si256(_mm256_andnot_si256(m80, _mm256_srli_epi16(w3, 1)), _mm256_load_si256((__m256i*)(rng_start + 96)));
}

__forceinline void tm_avx2_r256s_8::alg_7_il(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3)
{
	const __m256i ones = _mm256_cmpeq_epi8(w0, w0);
	w0 = _mm256_xor_si256(w0, ones);
	w1 = _mm256_xor_si256(w1, ones);
	w2 = _mm256_xor_si256(w2, ones);
	w3 = _mm256_xor_si256(w3, ones);
}

__forceinline void tm_avx2_r256s_8::alg_2_sub_il(__m256i& working_a, __m256i& working_b, __m256i& carry)
{
	const __m256i m01 = _mm256_set1_epi16(0x0101);
	const __m256i m80 = _mm256_set1_epi16(0x8080);
	__m256i cur_val1_most = _mm256_andnot_si256(m80, _mm256_srli_epi16(working_a, 1)); // & 0x7F
	__m256i cur_val2_most = _mm256_andnot_si256(m01, _mm256_slli_epi16(working_b, 1)); // & 0xFE
	__m256i cur_val2_masked = _mm256_and_si256(working_b, m80);
	__m256i cur_val1_bit = _mm256_and_si256(working_a, m01);

	__m256i mask = _mm256_permute2x128_si256(cur_val1_bit, cur_val1_bit, _MM_SHUFFLE(3, 0, 0, 3));
	__m256i cur_val1_srl = _mm256_alignr_epi8(mask, cur_val1_bit, 1);
	__m256i cur_val1_srl_w_carry = _mm256_or_si256(cur_val1_srl, carry);

	__m256i lo_to_hi = _mm256_permute2x128_si256(cur_val1_bit, cur_val1_bit, _MM_SHUFFLE(0, 0, 3, 0));
	__m256i next_carry = _mm256_bslli_epi128(lo_to_hi, 15);

	working_a = _mm256_or_si256(cur_val1_most, cur_val2_masked);
	working_b = _mm256_or_si256(cur_val2_most, cur_val1_srl_w_carry);
	carry = next_carry;
}

__forceinline void tm_avx2_r256s_8::alg_2_il(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3, uint16* rng_seed)
{
	__m256i carry = _mm256_load_si256((__m256i*)(rng->alg2_values_256_8 + ((*rng_seed) * 32)));
	alg_2_sub_il(w2, w3, carry);
	alg_2_sub_il(w0, w1, carry);
}

__forceinline void tm_avx2_r256s_8::alg_5_sub_il(__m256i& working_a, __m256i& working_b, __m256i& carry)
{
	const __m256i m01 = _mm256_set1_epi16(0x0101);
	const __m256i m80 = _mm256_set1_epi16(0x8080);
	__m256i cur_val1_most = _mm256_andnot_si256(m01, _mm256_slli_epi16(working_a, 1)); // & 0xFE
	__m256i cur_val2_most = _mm256_andnot_si256(m80, _mm256_srli_epi16(working_b, 1)); // & 0x7F
	__m256i cur_val2_masked = _mm256_and_si256(working_b, m01);
	__m256i cur_val1_bit = _mm256_and_si256(working_a, m80);

	__m256i mask = _mm256_permute2x128_si256(cur_val1_bit, cur_val1_bit, _MM_SHUFFLE(3, 0, 0, 3));
	__m256i cur_val1_srl = _mm256_alignr_epi8(mask, cur_val1_bit, 1);
	__m256i cur_val1_srl_w_carry = _mm256_or_si256(cur_val1_srl, carry);

	__m256i lo_to_hi = _mm256_permute2x128_si256(cur_val1_bit, cur_val1_bit, _MM_SHUFFLE(0, 0, 3, 0));
	__m256i next_carry = _mm256_bslli_epi128(lo_to_hi, 15);

	working_a = _mm256_or_si256(cur_val1_most, cur_val2_masked);
	working_b = _mm256_or_si256(cur_val2_most, cur_val1_srl_w_carry);
	carry = next_carry;
}

__forceinline void tm_avx2_r256s_8::alg_5_il(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3, uint16* rng_seed)
{
	__m256i carry = _mm256_load_si256((__m256i*)(rng->alg5_values_256_8 + ((*rng_seed) * 32)));
	alg_5_sub_il(w2, w3, carry);
	alg_5_sub_il(w0, w1, carry);
}

__forceinline void tm_avx2_r256s_8::_run_map_entry_x2(
	__m256i& a0, __m256i& a1, __m256i& a2, __m256i& a3,
	__m256i& b0, __m256i& b1, __m256i& b2, __m256i& b3,
	const key_schedule::key_schedule_entry& schedule_entry)
{
	uint16 seed_a = (schedule_entry.rng1 << 8) | schedule_entry.rng2;
	uint16 seed_b = seed_a;
	uint16 nibble_selector = schedule_entry.nibble_selector;

	// Unrolled (×16). Counter-intuitively, NOT unrolling this 2-way kernel was
	// measured SLOWER: a switch(i) jump-table removes the modest icache pressure
	// (5.4M→429K L1i) but doubles branch-misses (97M→212M) on the rotating
	// 16-way indirect target, a net loss (unlike the AVX-512 4-way, whose icache
	// problem was severe enough that no-unroll won). So keep the unroll here; the
	// immediate-index extract it enables avoids the per-step branch entirely.
#if defined(__GNUC__)
#pragma GCC unroll 16
#endif
	for (int i = 0; i < 16; i++)
	{
		unsigned char nibble = (nibble_selector >> 15) & 0x01;
		nibble_selector = nibble_selector << 1;

		unsigned char byte_a = 0, byte_b = 0;
#if defined(_MSC_VER) || defined(__clang__)
		const __m128i a0_low = _mm256_castsi256_si128(a0);
		const __m128i a1_low = _mm256_castsi256_si128(a1);
		const __m128i b0_low = _mm256_castsi256_si128(b0);
		const __m128i b1_low = _mm256_castsi256_si128(b1);
#define TM_X2_EXTRACT(out, r0, r1) \
		switch (i) { \
			case 0:  out=(unsigned char)_mm_extract_epi8(r0,0); break; \
			case 1:  out=(unsigned char)_mm_extract_epi8(r1,0); break; \
			case 2:  out=(unsigned char)_mm_extract_epi8(r0,1); break; \
			case 3:  out=(unsigned char)_mm_extract_epi8(r1,1); break; \
			case 4:  out=(unsigned char)_mm_extract_epi8(r0,2); break; \
			case 5:  out=(unsigned char)_mm_extract_epi8(r1,2); break; \
			case 6:  out=(unsigned char)_mm_extract_epi8(r0,3); break; \
			case 7:  out=(unsigned char)_mm_extract_epi8(r1,3); break; \
			case 8:  out=(unsigned char)_mm_extract_epi8(r0,4); break; \
			case 9:  out=(unsigned char)_mm_extract_epi8(r1,4); break; \
			case 10: out=(unsigned char)_mm_extract_epi8(r0,5); break; \
			case 11: out=(unsigned char)_mm_extract_epi8(r1,5); break; \
			case 12: out=(unsigned char)_mm_extract_epi8(r0,6); break; \
			case 13: out=(unsigned char)_mm_extract_epi8(r1,6); break; \
			case 14: out=(unsigned char)_mm_extract_epi8(r0,7); break; \
			case 15: out=(unsigned char)_mm_extract_epi8(r1,7); break; \
		}
		TM_X2_EXTRACT(byte_a, a0_low, a1_low);
		TM_X2_EXTRACT(byte_b, b0_low, b1_low);
#undef TM_X2_EXTRACT
#else
		if ((i & 1) == 0)
		{
			byte_a = (unsigned char)_mm_extract_epi8(_mm256_castsi256_si128(a0), i >> 1);
			byte_b = (unsigned char)_mm_extract_epi8(_mm256_castsi256_si128(b0), i >> 1);
		}
		else
		{
			byte_a = (unsigned char)_mm_extract_epi8(_mm256_castsi256_si128(a1), (i - 1) >> 1);
			byte_b = (unsigned char)_mm_extract_epi8(_mm256_castsi256_si128(b1), (i - 1) >> 1);
		}
#endif
		if (nibble == 1) { byte_a = byte_a >> 4; byte_b = byte_b >> 4; }
		const unsigned char alg_a = (byte_a >> 1) & 0x07;
		const unsigned char alg_b = (byte_b >> 1) & 0x07;

		switch (alg_a)
		{
			case 0: alg_0_il(a0, a1, a2, a3, &seed_a); seed_a = rng->seed_forward_128[seed_a]; break;
			case 1: add_alg(a0, a1, a2, a3, &seed_a, rng->regular_rng_values_256_8_shuffled); seed_a = rng->seed_forward_128[seed_a]; break;
			case 2: alg_2_il(a0, a1, a2, a3, &seed_a); seed_a = rng->seed_forward_1[seed_a]; break;
			case 3: alg_3(a0, a1, a2, a3, &seed_a); seed_a = rng->seed_forward_128[seed_a]; break;
			case 4: sub_alg(a0, a1, a2, a3, &seed_a, rng->regular_rng_values_256_8_shuffled); seed_a = rng->seed_forward_128[seed_a]; break;
			case 5: alg_5_il(a0, a1, a2, a3, &seed_a); seed_a = rng->seed_forward_1[seed_a]; break;
			case 6: alg_6_il(a0, a1, a2, a3, &seed_a); seed_a = rng->seed_forward_128[seed_a]; break;
			default: alg_7_il(a0, a1, a2, a3); break;
		}
		switch (alg_b)
		{
			case 0: alg_0_il(b0, b1, b2, b3, &seed_b); seed_b = rng->seed_forward_128[seed_b]; break;
			case 1: add_alg(b0, b1, b2, b3, &seed_b, rng->regular_rng_values_256_8_shuffled); seed_b = rng->seed_forward_128[seed_b]; break;
			case 2: alg_2_il(b0, b1, b2, b3, &seed_b); seed_b = rng->seed_forward_1[seed_b]; break;
			case 3: alg_3(b0, b1, b2, b3, &seed_b); seed_b = rng->seed_forward_128[seed_b]; break;
			case 4: sub_alg(b0, b1, b2, b3, &seed_b, rng->regular_rng_values_256_8_shuffled); seed_b = rng->seed_forward_128[seed_b]; break;
			case 5: alg_5_il(b0, b1, b2, b3, &seed_b); seed_b = rng->seed_forward_1[seed_b]; break;
			case 6: alg_6_il(b0, b1, b2, b3, &seed_b); seed_b = rng->seed_forward_128[seed_b]; break;
			default: alg_7_il(b0, b1, b2, b3); break;
		}
	}
}

void tm_avx2_r256s_8::run_maps_range_x2(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
                                        const uint8* in0, const uint8* in1, uint8* out0, uint8* out1)
{
	// Pool entries are not 32-byte aligned; use unaligned load/store (no penalty
	// on aligned data on Zen, correct on unaligned).
	__m256i a0 = _mm256_loadu_si256((const __m256i*)(in0));
	__m256i a1 = _mm256_loadu_si256((const __m256i*)(in0 + 32));
	__m256i a2 = _mm256_loadu_si256((const __m256i*)(in0 + 64));
	__m256i a3 = _mm256_loadu_si256((const __m256i*)(in0 + 96));
	__m256i b0 = _mm256_loadu_si256((const __m256i*)(in1));
	__m256i b1 = _mm256_loadu_si256((const __m256i*)(in1 + 32));
	__m256i b2 = _mm256_loadu_si256((const __m256i*)(in1 + 64));
	__m256i b3 = _mm256_loadu_si256((const __m256i*)(in1 + 96));

	if (end > schedule_entries.entries.size())
		end = schedule_entries.entries.size();
	for (std::size_t map_idx = begin; map_idx < end; map_idx++)
	{
		_run_map_entry_x2(a0, a1, a2, a3, b0, b1, b2, b3,
		                  schedule_entries.entries[map_idx]);
	}

	_mm256_storeu_si256((__m256i*)(out0), a0);
	_mm256_storeu_si256((__m256i*)(out0 + 32), a1);
	_mm256_storeu_si256((__m256i*)(out0 + 64), a2);
	_mm256_storeu_si256((__m256i*)(out0 + 96), a3);
	_mm256_storeu_si256((__m256i*)(out1), b0);
	_mm256_storeu_si256((__m256i*)(out1 + 32), b1);
	_mm256_storeu_si256((__m256i*)(out1 + 64), b2);
	_mm256_storeu_si256((__m256i*)(out1 + 96), b3);
}

void tm_avx2_r256s_8::decrypt_carnival_world()
{
	__m256i working_code0;
	__m256i working_code1;
	__m256i working_code2;
	__m256i working_code3;
	_load_from_mem(working_code0, working_code1, working_code2, working_code3);

	_decrypt_carnival_world(working_code0, working_code1, working_code2, working_code3);
	_store_to_mem(working_code0, working_code1, working_code2, working_code3);
}

void tm_avx2_r256s_8::decrypt_other_world()
{
	__m256i working_code0;
	__m256i working_code1;
	__m256i working_code2;
	__m256i working_code3;
	_load_from_mem(working_code0, working_code1, working_code2, working_code3);

	_decrypt_other_world(working_code0, working_code1, working_code2, working_code3);
	_store_to_mem(working_code0, working_code1, working_code2, working_code3);
}

void tm_avx2_r256s_8::_decrypt_carnival_world(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3)
{
	xor_alg(working_code0, working_code1, working_code2, working_code3, carnival_world_data_shuffled);
}

void tm_avx2_r256s_8::_decrypt_other_world(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3)
{
	xor_alg(working_code0, working_code1, working_code2, working_code3, other_world_data_shuffled);
}

__forceinline void tm_avx2_r256s_8::mid_sum(__m128i& sum, __m256i& working_code, __m256i& sum_mask, __m128i& lo_mask)
{
	__m256i temp_masked = _mm256_and_si256(working_code, sum_mask);

	__m128i temp1_lo = _mm256_castsi256_si128(temp_masked);
	__m128i temp1_hi = _mm256_extractf128_si256(temp_masked, 1);

	__m128i temp1_lo_lo = _mm_and_si128(temp1_lo, lo_mask);
	__m128i temp1_lo_hi = _mm_srli_epi16(temp1_lo, 8);
	__m128i temp1_hi_lo = _mm_and_si128(temp1_hi, lo_mask);
	__m128i temp1_hi_hi = _mm_srli_epi16(temp1_hi, 8);

	sum = _mm_add_epi16(sum, temp1_lo_lo);
	sum = _mm_add_epi16(sum, temp1_lo_hi);
	sum = _mm_add_epi16(sum, temp1_hi_lo);
	sum = _mm_add_epi16(sum, temp1_hi_hi);
}

__forceinline uint16 tm_avx2_r256s_8::masked_checksum(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3, uint8* mask)
{
	__m128i sum = _mm_setzero_si128();
	__m128i lo_mask = _mm_set1_epi16(0x00FF);

	__m256i sum_mask = _mm256_load_si256((__m256i*)(mask));
	mid_sum(sum, working_code0, sum_mask, lo_mask);

	sum_mask = _mm256_load_si256((__m256i*)(mask + 32));
	mid_sum(sum, working_code1, sum_mask, lo_mask);

	sum_mask = _mm256_load_si256((__m256i*)(mask + 64));
	mid_sum(sum, working_code2, sum_mask, lo_mask);

	sum_mask = _mm256_load_si256((__m256i*)(mask + 96));
	mid_sum(sum, working_code3, sum_mask, lo_mask);

	uint16 code_sum = _mm_extract_epi16(sum, 0) +
		_mm_extract_epi16(sum, 1) +
		_mm_extract_epi16(sum, 2) +
		_mm_extract_epi16(sum, 3) +
		_mm_extract_epi16(sum, 4) +
		_mm_extract_epi16(sum, 5) +
		_mm_extract_epi16(sum, 6) +
		_mm_extract_epi16(sum, 7);

	return code_sum;
}

uint16 tm_avx2_r256s_8::calculate_carnival_world_checksum()
{
	__m256i working_code0;
	__m256i working_code1;
	__m256i working_code2;
	__m256i working_code3;
	_load_from_mem(working_code0, working_code1, working_code2, working_code3);

	return _calculate_carnival_world_checksum(working_code0, working_code1, working_code2, working_code3);
}

uint16 tm_avx2_r256s_8::calculate_other_world_checksum()
{
	__m256i working_code0;
	__m256i working_code1;
	__m256i working_code2;
	__m256i working_code3;
	_load_from_mem(working_code0, working_code1, working_code2, working_code3);

	return _calculate_other_world_checksum(working_code0, working_code1, working_code2, working_code3);
}

uint16 tm_avx2_r256s_8::_calculate_carnival_world_checksum(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3)
{
	return masked_checksum(working_code0, working_code1, working_code2, working_code3, carnival_world_checksum_mask_shuffled);
}

uint16 tm_avx2_r256s_8::_calculate_other_world_checksum(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3)
{
	return masked_checksum(working_code0, working_code1, working_code2, working_code3, other_world_checksum_mask_shuffled);
}

__forceinline uint16 tm_avx2_r256s_8::fetch_checksum_value(__m256i& working_code0, __m256i& working_code1, uint8 code_length)
{
	_mm256_store_si256((__m256i*)(working_code_data), working_code0);
	_mm256_store_si256((__m256i*)(working_code_data + 32), working_code1);

	unsigned char checksum_low = (uint8)((uint8*)working_code_data)[shuffle_8((127 - code_length), 256)];
	unsigned char checksum_hi = (uint8)((uint8*)working_code_data)[shuffle_8((127 - (code_length + 1)), 256)];
	return (checksum_hi << 8) | checksum_low;
}

uint16 tm_avx2_r256s_8::fetch_carnival_world_checksum_value()
{
	__m256i working_code0 = _mm256_load_si256((__m256i*)(working_code_data));
	__m256i working_code1 = _mm256_load_si256((__m256i*)(working_code_data + 32));

	return _fetch_carnival_world_checksum_value(working_code0, working_code1);
}

uint16 tm_avx2_r256s_8::fetch_other_world_checksum_value()
{
	__m256i working_code0 = _mm256_load_si256((__m256i*)(working_code_data));
	__m256i working_code1 = _mm256_load_si256((__m256i*)(working_code_data + 32));

	return _fetch_other_world_checksum_value(working_code0, working_code1);
}

__forceinline uint16 tm_avx2_r256s_8::_fetch_carnival_world_checksum_value(__m256i& working_code0, __m256i& working_code1)
{
	return fetch_checksum_value(working_code0, working_code1, CARNIVAL_WORLD_CODE_LENGTH - 2);
}

__forceinline uint16 tm_avx2_r256s_8::_fetch_other_world_checksum_value(__m256i& working_code0, __m256i& working_code1)
{
	return fetch_checksum_value(working_code0, working_code1, OTHER_WORLD_CODE_LENGTH - 2);
}

__forceinline bool tm_avx2_r256s_8::check_carnival_world_checksum(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3)
{
	return _calculate_carnival_world_checksum(working_code0, working_code1, working_code2, working_code3) == _fetch_carnival_world_checksum_value(working_code0, working_code1);
}

__forceinline bool tm_avx2_r256s_8::check_other_world_checksum(__m256i& working_code0, __m256i& working_code1, __m256i& working_code2, __m256i& working_code3)
{
	return _calculate_other_world_checksum(working_code0, working_code1, working_code2, working_code3) == _fetch_other_world_checksum_value(working_code0, working_code1);
}

void tm_avx2_r256s_8::run_bruteforce_data(uint32 key, uint32 start_data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size)
{
	__m256i working_code0;
	__m256i working_code1;
	__m256i working_code2;
	__m256i working_code3;

	__m256i mask_FF = _mm256_set1_epi16(0xFFFF);
	__m256i mask_FE = _mm256_set1_epi16(0xFEFE);
	__m256i mask_7F = _mm256_set1_epi16(0x7F7F);
	__m256i mask_80 = _mm256_set1_epi16(0x8080);
	__m256i mask_01 = _mm256_set1_epi16(0x0101);
	__m256i mask_top_01 = _mm256_set_epi16(0x0100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	__m256i mask_top_80 = _mm256_set_epi16(0x8000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	uint32 output_pos = 0;
	for (uint32 i = 0; i < amount_to_run; i++)
	{
		if ((result_max_size - output_pos) < 5)
		{
			*result_size = result_max_size;
			return;
		}

		_expand_code(key, start_data + i, working_code0, working_code1, working_code2, working_code3);
		_run_all_maps(working_code0, working_code1, working_code2, working_code3, schedule_entries, mask_FF, mask_FE, mask_7F, mask_80, mask_01, mask_top_01, mask_top_80);

		__m256i working_code0_xor = working_code0;
		__m256i working_code1_xor = working_code1;
		__m256i working_code2_xor = working_code2;
		__m256i working_code3_xor = working_code3;

		_decrypt_carnival_world(working_code0_xor, working_code1_xor, working_code2_xor, working_code3_xor);

		if (check_carnival_world_checksum(working_code0_xor, working_code1_xor, working_code2_xor, working_code3_xor))
		{
			_store_to_mem(working_code0_xor, working_code1_xor, working_code2_xor, working_code3_xor);
			*((uint32*)(&result_data[output_pos])) = i;

			uint8 unshuffled_data[128];
			unshuffle_mem(working_code_data, unshuffled_data, 256, false);

			result_data[output_pos + 4] = check_machine_code(unshuffled_data, CARNIVAL_WORLD);
			output_pos += 5;
		}
		else
		{
			working_code0_xor = working_code0;
			working_code1_xor = working_code1;
			working_code2_xor = working_code2;
			working_code3_xor = working_code3;

			_decrypt_other_world(working_code0_xor, working_code1_xor, working_code2_xor, working_code3_xor);

			if (check_other_world_checksum(working_code0_xor, working_code1_xor, working_code2_xor, working_code3_xor))
			{
				_store_to_mem(working_code0_xor, working_code1_xor, working_code2_xor, working_code3_xor);
				*((uint32*)(&result_data[output_pos])) = i;

				uint8 unshuffled_data[128];
				unshuffle_mem(working_code_data, unshuffled_data, 256, false);

				result_data[output_pos + 4] = check_machine_code(unshuffled_data, OTHER_WORLD);
				output_pos += 5;
			}
		}

		report_progress((float)(i + 1) / amount_to_run);
	}

	*result_size = output_pos;
}

bool tm_avx2_r256s_8::initialized = false;
uint8 tm_avx2_r256s_8::carnival_world_checksum_mask_shuffled[128] = {};
uint8 tm_avx2_r256s_8::carnival_world_data_shuffled[128] = {};
uint8 tm_avx2_r256s_8::other_world_checksum_mask_shuffled[128] = {};
uint8 tm_avx2_r256s_8::other_world_data_shuffled[128] = {};
