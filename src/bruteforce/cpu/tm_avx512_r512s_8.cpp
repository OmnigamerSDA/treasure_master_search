#include <stdio.h>
#include <cstdint>
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
#include "tm_avx512_r512s_8.h"

#if defined(__GNUC__)
#define _mm256_set_m128i(vh, vl) \
        _mm256_castpd_si256(_mm256_insertf128_pd(_mm256_castsi256_pd(_mm256_castsi128_si256(vl)), _mm_castsi128_pd(vh), 1))
#endif

tm_avx512_r512s_8::tm_avx512_r512s_8(RNG* rng_obj) : TM_base(rng_obj)
{
	initialize();

	// Pre-shuffle the world-decrypt operands and checksum masks into the same
	// physical 512 layout the state uses, so the screen overrides can AND/XOR per
	// byte without re-laying-out the state on every survivor.
	shuffle_mem(carnival_world_checksum_mask, carnival_world_checksum_mask_shuffled, 512, false);
	shuffle_mem(carnival_world_data,          carnival_world_data_shuffled,          512, false);
	shuffle_mem(other_world_checksum_mask,    other_world_checksum_mask_shuffled,    512, false);
	shuffle_mem(other_world_data,             other_world_data_shuffled,             512, false);
}

__forceinline void tm_avx512_r512s_8::initialize()
{
	if (!initialized)
	{
		rng->generate_expansion_values_512_8_shuffled();

		rng->generate_seed_forward_1();
		rng->generate_seed_forward_128();

		rng->generate_regular_rng_values_8();
		rng->generate_regular_rng_values_512_8_shuffled();

		rng->generate_alg0_values_512_8_shuffled();
		rng->generate_alg2_values_512_8();
		rng->generate_alg4_values_512_8_shuffled();
		rng->generate_alg5_values_512_8();
		rng->generate_alg6_values_512_8_shuffled();

		initialized = true;
	}
	obj_name = "tm_avx512_r512s_8";
}

int tm_avx512_r512s_8::shuffle(int addr)
{
	return (addr % 2) * 64 + ((addr / 2) % 64);
}

void tm_avx512_r512s_8::expand(uint32 key, uint32 data)
{
	// Vectorized expand (replaces the per-byte scalar version that perf showed
	// was ~27% of dedup instructions). Under the 512 layout shuffle(addr) =
	// (addr%2)*64 + (addr/2)%64, physical reg0 byte p = canonical byte 2p and
	// reg1 byte p = canonical byte 2p+1, where canonical byte j cycles through
	// [key>>24, key>>16, key>>8, key, data>>24, data>>16, data>>8, data]. So
	// reg0 is the period-4 pattern [key>>24, key>>8, data>>24, data>>8] and
	// reg1 is [key>>16, key, data>>16, data], each broadcast across 64 bytes
	// (vpbroadcastd), then add the pre-shuffled expansion row.
	const uint32 dword0 =
		(uint32)((key >> 24) & 0xFF)
		| ((uint32)((key >> 8) & 0xFF) << 8)
		| ((uint32)((data >> 24) & 0xFF) << 16)
		| ((uint32)((data >> 8) & 0xFF) << 24);
	const uint32 dword1 =
		(uint32)((key >> 16) & 0xFF)
		| ((uint32)(key & 0xFF) << 8)
		| ((uint32)((data >> 16) & 0xFF) << 16)
		| ((uint32)(data & 0xFF) << 24);

	__m512i working_code0 = _mm512_set1_epi32((int)dword0);
	__m512i working_code1 = _mm512_set1_epi32((int)dword1);

	const uint16 rng_seed = (key >> 16) & 0xFFFF;
	const uint8* row = rng->expansion_values_512_8_shuffled + (rng_seed * 128);
	working_code0 = _mm512_add_epi8(working_code0, _mm512_load_si512((const __m512i*)(row)));
	working_code1 = _mm512_add_epi8(working_code1, _mm512_load_si512((const __m512i*)(row + 64)));

	_mm512_store_si512((__m512i*)(working_code_data), working_code0);
	_mm512_store_si512((__m512i*)(working_code_data + 64), working_code1);
}

void tm_avx512_r512s_8::load_data(uint8* new_data)
{
	for (int i = 0; i < 128; i++)
	{
		((uint8*)working_code_data)[(i % 2) * 64 + ((i / 2) % 64)] = new_data[i];
	}

}

void tm_avx512_r512s_8::fetch_data(uint8* new_data)
{
	for (int i = 0; i < 128; i++)
	{
		new_data[i] = ((uint8*)working_code_data)[(i % 2) * 64 + ((i / 2) % 64)];
	}
}

__forceinline void tm_avx512_r512s_8::run_alg(int algorithm_id, uint16* rng_seed, int iterations)
{
	__m512i working_code0 = _mm512_load_si512((__m512i*)(working_code_data));
	__m512i working_code1 = _mm512_load_si512((__m512i*)(working_code_data + 64));

	__m512i mask_FF = _mm512_set1_epi16(0xFFFF);
	__m512i mask_FE = _mm512_set1_epi16(0xFEFE);
	__m512i mask_7F = _mm512_set1_epi16(0x7F7F);
	__m512i mask_80 = _mm512_set1_epi16(0x8080);
	__m512i mask_01 = _mm512_set1_epi16(0x0101);

	if (algorithm_id == 0)
	{
		for (int j = 0; j < iterations; j++)
		{
			alg_0(working_code0, working_code1, rng_seed, mask_FE);
			*rng_seed = rng->seed_forward_128[*rng_seed];
		}
	}
	else if (algorithm_id == 1 || algorithm_id == 4)
	{
		for (int j = 0; j < iterations; j++)
		{
			uint8* rng_start = rng->regular_rng_values_512_8_shuffled;

			if (algorithm_id == 4)
			{
				rng_start = rng->alg4_values_512_8_shuffled;
			}

			add_alg(working_code0, working_code1, rng_seed, rng_start);
			*rng_seed = rng->seed_forward_128[*rng_seed];
		}
	}
	else if (algorithm_id == 2)
	{
		for (int j = 0; j < iterations; j++)
		{
			alg_2(working_code0, working_code1, rng_seed, mask_80, mask_7F, mask_FE, mask_01);
			*rng_seed = rng->seed_forward_1[*rng_seed];
		}
	}
	else if (algorithm_id == 3)
	{
		for (int j = 0; j < iterations; j++)
		{
			alg_3(working_code0, working_code1, rng_seed);
			*rng_seed = rng->seed_forward_128[*rng_seed];
		}
	}
	else if (algorithm_id == 5)
	{
		for (int j = 0; j < iterations; j++)
		{
			alg_5(working_code0, working_code1, rng_seed, mask_80, mask_7F, mask_FE, mask_01);
			*rng_seed = rng->seed_forward_1[*rng_seed];
		}
	}
	else if (algorithm_id == 6)
	{
		for (int j = 0; j < iterations; j++)
		{
			alg_6(working_code0, working_code1, rng_seed, mask_7F);
			*rng_seed = rng->seed_forward_128[*rng_seed];
		}
	}
	else if (algorithm_id == 7)
	{
		for (int j = 0; j < iterations; j++)
		{
			alg_7(working_code0, working_code1, mask_FF);
		}
	}

	_mm512_store_si512((__m512i*)(working_code_data), working_code0);
	_mm512_store_si512((__m512i*)(working_code_data + 64), working_code1);

}

__forceinline void tm_avx512_r512s_8::alg_0(__m512i& working_code0, __m512i& working_code1, uint16* rng_seed, __m512i& mask_FE)
{
	uint8* rng_start = rng->alg0_values_512_8_shuffled + ((*rng_seed) * 128);

	// (wc<<1 & 0xFE) | rng fused into one vpternlogd (0xEA = (A&B)|C), per
	// micro500's avx512 kernels. Saves an op/alg on the latency-bound chain.
	working_code0 = _mm512_slli_epi64(working_code0, 1);
	__m512i rng_val = _mm512_load_si512((__m512i*)(rng_start));
	working_code0 = _mm512_ternarylogic_epi32(working_code0, mask_FE, rng_val, 0xEA);

	working_code1 = _mm512_slli_epi64(working_code1, 1);
	rng_val = _mm512_load_si512((__m512i*)(rng_start + 64));
	working_code1 = _mm512_ternarylogic_epi32(working_code1, mask_FE, rng_val, 0xEA);
}

__forceinline void tm_avx512_r512s_8::alg_2_sub(__m512i& working_a, __m512i& working_b, __m512i& carry, __m512i& mask_80, __m512i& mask_7F, __m512i& mask_FE, __m512i& mask_01)
{
	// Shift bytes right 1 bit
	__m512i cur_val1_most = _mm512_and_si512(_mm512_srli_epi64(working_a, 1), mask_7F);
	// Shift bytes left 1 bit
	__m512i cur_val2_most = _mm512_and_si512(_mm512_slli_epi64(working_b, 1), mask_FE);
	
	// Mask off the top bits
	__m512i cur_val2_masked = _mm512_and_si512(working_b, mask_80);

	__m512i cur_val1_bit = _mm512_and_si512(working_a, mask_01);

	// Shift right 1 byte (byte i <- byte i+1, top byte zero). Single VBMI vpermb
	// replaces the AVX-512F permutexvar_epi32 + valignr pair — the alg_2/alg_5
	// cross-lane op was the per-step critical-path cost that left a naive
	// AVX-512 dedup ~10% behind AVX2's 256-bit vperm2i128+valignr.
	const __m512i idx_shr1 = _mm512_set_epi8(
		0,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,
		48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,
		32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,
		16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);
	__m512i cur_val1_srl = _mm512_maskz_permutexvar_epi8(_cvtu64_mask64(0x7FFFFFFFFFFFFFFFull), idx_shr1, cur_val1_bit);
	// working_b = cur_val2_most | cur_val1_srl | carry, fused into one vpternlogd
	// (0xFE = A|B|C), dropping the intermediate OR.
	working_a = _mm512_or_si512(cur_val1_most, cur_val2_masked);
	working_b = _mm512_ternarylogic_epi32(cur_val2_most, cur_val1_srl, carry, 0xFE);
}
__forceinline void tm_avx512_r512s_8::alg_2(__m512i& working_code0, __m512i& working_code1, uint16* rng_seed, __m512i& mask_80, __m512i& mask_7F, __m512i& mask_FE, __m512i& mask_01)
{
	__m512i carry = _mm512_load_si512((__m512i*)(rng->alg2_values_512_8 + ((*rng_seed) * 64)));

	alg_2_sub(working_code0, working_code1, carry, mask_80, mask_7F, mask_FE, mask_01);
}

__forceinline void tm_avx512_r512s_8::alg_3(__m512i& working_code0, __m512i& working_code1, uint16* rng_seed)
{
	uint8* rng_start = rng->regular_rng_values_512_8_shuffled + ((*rng_seed) * 128);

	__m512i rng_val = _mm512_load_si512((__m512i*)(rng_start));
	working_code0 = _mm512_xor_si512(working_code0, rng_val);

	rng_val = _mm512_load_si512((__m512i*)(rng_start + 64));
	working_code1 = _mm512_xor_si512(working_code1, rng_val);
}

__forceinline void tm_avx512_r512s_8::alg_5_sub(__m512i& working_a, __m512i& working_b, __m512i& carry, __m512i& mask_80, __m512i& mask_7F, __m512i& mask_FE, __m512i& mask_01)
{
	// Shift bytes right 1 bit
	__m512i cur_val1_most = _mm512_and_si512(_mm512_slli_epi64(working_a, 1), mask_FE);
	// Shift bytes left 1 bit
	__m512i cur_val2_most = _mm512_and_si512(_mm512_srli_epi64(working_b, 1), mask_7F);

	// Mask off the top bits
	__m512i cur_val2_masked = _mm512_and_si512(working_b, mask_01);

	__m512i cur_val1_bit = _mm512_and_si512(working_a, mask_80);

	// Shift right 1 byte (byte i <- byte i+1, top byte zero). Single VBMI vpermb
	// replaces the AVX-512F permutexvar_epi32 + valignr pair — the alg_2/alg_5
	// cross-lane op was the per-step critical-path cost that left a naive
	// AVX-512 dedup ~10% behind AVX2's 256-bit vperm2i128+valignr.
	const __m512i idx_shr1 = _mm512_set_epi8(
		0,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,
		48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,
		32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,
		16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);
	__m512i cur_val1_srl = _mm512_maskz_permutexvar_epi8(_cvtu64_mask64(0x7FFFFFFFFFFFFFFFull), idx_shr1, cur_val1_bit);
	// working_b = cur_val2_most | cur_val1_srl | carry, fused into one vpternlogd
	// (0xFE = A|B|C), dropping the intermediate OR.
	working_a = _mm512_or_si512(cur_val1_most, cur_val2_masked);
	working_b = _mm512_ternarylogic_epi32(cur_val2_most, cur_val1_srl, carry, 0xFE);
}

__forceinline void tm_avx512_r512s_8::alg_5(__m512i& working_code0, __m512i& working_code1, uint16* rng_seed, __m512i& mask_80, __m512i& mask_7F, __m512i& mask_FE, __m512i& mask_01)
{
	__m512i carry = _mm512_load_si512((__m512i*)(rng->alg5_values_512_8 + ((*rng_seed) * 64)));

	alg_5_sub(working_code0, working_code1, carry, mask_80, mask_7F, mask_FE, mask_01);
}

__forceinline void tm_avx512_r512s_8::alg_6(__m512i& working_code0, __m512i& working_code1, uint16* rng_seed, __m512i& mask_7F)
{
	uint8* rng_start = rng->alg6_values_512_8_shuffled + ((*rng_seed) * 128);

	// (wc>>1 & 0x7F) | rng fused into one vpternlogd (0xEA = (A&B)|C).
	working_code0 = _mm512_srli_epi64(working_code0, 1);
	__m512i rng_val = _mm512_load_si512((__m512i*)(rng_start));
	working_code0 = _mm512_ternarylogic_epi32(working_code0, mask_7F, rng_val, 0xEA);

	working_code1 = _mm512_srli_epi64(working_code1, 1);
	rng_val = _mm512_load_si512((__m512i*)(rng_start + 64));
	working_code1 = _mm512_ternarylogic_epi32(working_code1, mask_7F, rng_val, 0xEA);
}

__forceinline void tm_avx512_r512s_8::alg_7(__m512i& working_code0, __m512i& working_code1, __m512i& mask_FF)
{
	working_code0 = _mm512_xor_si512(working_code0, mask_FF);
	working_code1 = _mm512_xor_si512(working_code1, mask_FF);
}

__forceinline void tm_avx512_r512s_8::add_alg(__m512i& working_code0, __m512i& working_code1, uint16* rng_seed, uint8* rng_start)
{
	rng_start = rng_start + ((*rng_seed) * 128);

	__m512i rng_val = _mm512_load_si512((__m512i*)(rng_start));
	working_code0 = _mm512_add_epi8(working_code0, rng_val);

	rng_val = _mm512_load_si512((__m512i*)(rng_start + 64));
	working_code1 = _mm512_add_epi8(working_code1, rng_val);
}


void tm_avx512_r512s_8::run_one_map(const key_schedule::key_schedule_entry& schedule_entry)
{
	uint16 rng_seed = (schedule_entry.rng1 << 8) | schedule_entry.rng2;
	uint16 nibble_selector = schedule_entry.nibble_selector;

	// Next, the working code is processed with the same steps 16 times:
	for (int i = 0; i < 16; i++)
	{
		// Get the highest bit of the nibble selector to use as a flag
		unsigned char nibble = (nibble_selector >> 15) & 0x01;
		// Shift the nibble selector up one bit
		nibble_selector = nibble_selector << 1;

		// working_code_data is in the impl's shuffled layout; logical byte i
		// is at physical index shuffle(i). run_all_maps reads via shuffle(i);
		// reading the raw index here produced different alg-IDs and silently
		// diverged from run_all_maps' final state (caught 2026-05-24 by
		// state_dedup parity checks).
		unsigned char current_byte = (uint8)working_code_data[shuffle(i)];

		if (nibble == 1)
		{
			current_byte = current_byte >> 4;
		}

		// Mask off only 3 bits
		unsigned char alg_id = (current_byte >> 1) & 0x07;

		run_alg(alg_id, &rng_seed, 1);
	}
}

void tm_avx512_r512s_8::run_all_maps(const key_schedule& schedule_entries)
{
	__m512i working_code0 = _mm512_load_si512((__m512i*)(working_code_data));
	__m512i working_code1 = _mm512_load_si512((__m512i*)(working_code_data + 64));

	__m512i mask_FF = _mm512_set1_epi16(0xFFFF);
	__m512i mask_FE = _mm512_set1_epi16(0xFEFE);
	__m512i mask_7F = _mm512_set1_epi16(0x7F7F);
	__m512i mask_80 = _mm512_set1_epi16(0x8080);
	__m512i mask_01 = _mm512_set1_epi16(0x0101);

	for (std::vector<key_schedule::key_schedule_entry>::const_iterator it = schedule_entries.entries.begin(); it != schedule_entries.entries.end(); it++)
	{
		key_schedule::key_schedule_entry schedule_entry = *it;

		uint16 rng_seed = (schedule_entry.rng1 << 8) | schedule_entry.rng2;
		uint16 nibble_selector = schedule_entry.nibble_selector;

		// Next, the working code is processed with the same steps 16 times:
		for (int i = 0; i < 16; i++)
		{
			_mm512_store_si512((__m512i*)(working_code_data), working_code0);
			_mm512_store_si512((__m512i*)(working_code_data + 64), working_code1);

			// Get the highest bit of the nibble selector to use as a flag
			unsigned char nibble = (nibble_selector >> 15) & 0x01;
			// Shift the nibble selector up one bit
			nibble_selector = nibble_selector << 1;

			// If the flag is a 1, get the high nibble of the current byte
			// Otherwise use the low nibble
			unsigned char current_byte = (uint8)((uint8*)working_code_data)[shuffle(i)];

			if (nibble == 1)
			{
				current_byte = current_byte >> 4;
			}

			// Mask off only 3 bits
			unsigned char algorithm_id = (current_byte >> 1) & 0x07;
			/*
			printf("%i ", algorithm_id);
			printf("%04X ", rng_seed);

			// store back to memory
			_mm256_store_si256((__m256i*)(working_code_data), working_code0);
			_mm256_store_si256((__m256i*)(working_code_data + 32), working_code1);
			_mm256_store_si256((__m256i*)(working_code_data + 64), working_code2);
			_mm256_store_si256((__m256i*)(working_code_data + 96), working_code3);

			print_working_code();
			*/

			if (algorithm_id == 0)
			{
				alg_0(working_code0, working_code1, &rng_seed, mask_FE);
				rng_seed = rng->seed_forward_128[rng_seed];
			}
			else if (algorithm_id == 1 || algorithm_id == 4)
			{
				uint8* rng_start = rng->regular_rng_values_512_8_shuffled;

				if (algorithm_id == 4)
				{
					rng_start = rng->alg4_values_512_8_shuffled;
				}

				add_alg(working_code0, working_code1, &rng_seed, rng_start);
				rng_seed = rng->seed_forward_128[rng_seed];
			}
			else if (algorithm_id == 2)
			{
				alg_2(working_code0, working_code1, &rng_seed, mask_80, mask_7F, mask_FE, mask_01);
				rng_seed = rng->seed_forward_1[rng_seed];
			}
			else if (algorithm_id == 3)
			{
				alg_3(working_code0, working_code1, &rng_seed);
				rng_seed = rng->seed_forward_128[rng_seed];
			}
			else if (algorithm_id == 5)
			{
				alg_5(working_code0, working_code1, &rng_seed, mask_80, mask_7F, mask_FE, mask_01);
				rng_seed = rng->seed_forward_1[rng_seed];
			}
			else if (algorithm_id == 6)
			{
				alg_6(working_code0, working_code1, &rng_seed, mask_7F);
				rng_seed = rng->seed_forward_128[rng_seed];
			}
			else if (algorithm_id == 7)
			{
				alg_7(working_code0, working_code1, mask_FF);
			}
		}
		/*
		printf("\n");
		if (schedule_counter == 6)
		{
			// store back to memory
			_mm256_store_si256((__m256i*)(working_code_data), working_code0);
			_mm256_store_si256((__m256i*)(working_code_data + 32), working_code1);
			_mm256_store_si256((__m256i*)(working_code_data + 64), working_code2);
			_mm256_store_si256((__m256i*)(working_code_data + 96), working_code3);

			print_working_code();
		}
		*/
	}

	// store back to memory
	_mm512_store_si512((__m512i*)(working_code_data), working_code0);
	_mm512_store_si512((__m512i*)(working_code_data + 64), working_code1);
}

// Register-resident single map entry. Mirrors run_all_maps' per-step logic but
// (a) extracts the alg-select byte directly from the registers via vpextrb on
// the low 128 bits instead of a store + scalar reload, and (b) dispatches via a
// jump-table switch. Under the 512 shuffle layout (i%2)*64 + (i/2)%64, logical
// byte i for i=0..15 is reg0[i/2] (even i) or reg1[(i-1)/2] (odd i) — both in
// the low 128 bits of their ZMM — so the same extraction trick AVX2 uses works.
__forceinline void tm_avx512_r512s_8::_run_map_entry(__m512i& working_code0, __m512i& working_code1,
	const key_schedule::key_schedule_entry& schedule_entry,
	__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01)
{
	uint16 rng_seed = (schedule_entry.rng1 << 8) | schedule_entry.rng2;
	uint16 nibble_selector = schedule_entry.nibble_selector;

#if defined(__GNUC__)
#pragma GCC unroll 16
#endif
	for (int i = 0; i < 16; i++)
	{
		unsigned char nibble = (nibble_selector >> 15) & 0x01;
		nibble_selector = nibble_selector << 1;

		unsigned char current_byte = 0;
#if defined(_MSC_VER) || defined(__clang__)
		const __m128i w0_low = _mm512_castsi512_si128(working_code0);
		const __m128i w1_low = _mm512_castsi512_si128(working_code1);
		switch (i)
		{
			case 0:  current_byte = (unsigned char)_mm_extract_epi8(w0_low, 0); break;
			case 1:  current_byte = (unsigned char)_mm_extract_epi8(w1_low, 0); break;
			case 2:  current_byte = (unsigned char)_mm_extract_epi8(w0_low, 1); break;
			case 3:  current_byte = (unsigned char)_mm_extract_epi8(w1_low, 1); break;
			case 4:  current_byte = (unsigned char)_mm_extract_epi8(w0_low, 2); break;
			case 5:  current_byte = (unsigned char)_mm_extract_epi8(w1_low, 2); break;
			case 6:  current_byte = (unsigned char)_mm_extract_epi8(w0_low, 3); break;
			case 7:  current_byte = (unsigned char)_mm_extract_epi8(w1_low, 3); break;
			case 8:  current_byte = (unsigned char)_mm_extract_epi8(w0_low, 4); break;
			case 9:  current_byte = (unsigned char)_mm_extract_epi8(w1_low, 4); break;
			case 10: current_byte = (unsigned char)_mm_extract_epi8(w0_low, 5); break;
			case 11: current_byte = (unsigned char)_mm_extract_epi8(w1_low, 5); break;
			case 12: current_byte = (unsigned char)_mm_extract_epi8(w0_low, 6); break;
			case 13: current_byte = (unsigned char)_mm_extract_epi8(w1_low, 6); break;
			case 14: current_byte = (unsigned char)_mm_extract_epi8(w0_low, 7); break;
			case 15: current_byte = (unsigned char)_mm_extract_epi8(w1_low, 7); break;
		}
#else
		current_byte = ((i & 1) == 0)
			? (unsigned char)_mm_extract_epi8(_mm512_castsi512_si128(working_code0), i >> 1)
			: (unsigned char)_mm_extract_epi8(_mm512_castsi512_si128(working_code1), (i - 1) >> 1);
#endif

		if (nibble == 1)
		{
			current_byte = current_byte >> 4;
		}

		unsigned char algorithm_id = (current_byte >> 1) & 0x07;

		switch (algorithm_id)
		{
			case 0:
				alg_0(working_code0, working_code1, &rng_seed, mask_FE);
				rng_seed = rng->seed_forward_128[rng_seed];
				break;
			case 1:
				add_alg(working_code0, working_code1, &rng_seed, rng->regular_rng_values_512_8_shuffled);
				rng_seed = rng->seed_forward_128[rng_seed];
				break;
			case 2:
				alg_2(working_code0, working_code1, &rng_seed, mask_80, mask_7F, mask_FE, mask_01);
				rng_seed = rng->seed_forward_1[rng_seed];
				break;
			case 3:
				alg_3(working_code0, working_code1, &rng_seed);
				rng_seed = rng->seed_forward_128[rng_seed];
				break;
			case 4:
				add_alg(working_code0, working_code1, &rng_seed, rng->alg4_values_512_8_shuffled);
				rng_seed = rng->seed_forward_128[rng_seed];
				break;
			case 5:
				alg_5(working_code0, working_code1, &rng_seed, mask_80, mask_7F, mask_FE, mask_01);
				rng_seed = rng->seed_forward_1[rng_seed];
				break;
			case 6:
				alg_6(working_code0, working_code1, &rng_seed, mask_7F);
				rng_seed = rng->seed_forward_128[rng_seed];
				break;
			default: // case 7 — already & 0x07 in caller
				alg_7(working_code0, working_code1, mask_FF);
				break;
		}
	}
}

void tm_avx512_r512s_8::run_maps_range(const key_schedule& schedule_entries, std::size_t begin, std::size_t end)
{
	__m512i working_code0 = _mm512_load_si512((__m512i*)(working_code_data));
	__m512i working_code1 = _mm512_load_si512((__m512i*)(working_code_data + 64));

	__m512i mask_FF = _mm512_set1_epi16(0xFFFF);
	__m512i mask_FE = _mm512_set1_epi16(0xFEFE);
	__m512i mask_7F = _mm512_set1_epi16(0x7F7F);
	__m512i mask_80 = _mm512_set1_epi16(0x8080);
	__m512i mask_01 = _mm512_set1_epi16(0x0101);

	if (end > schedule_entries.entries.size())
		end = schedule_entries.entries.size();
	for (std::size_t map_idx = begin; map_idx < end; map_idx++)
	{
		_run_map_entry(working_code0, working_code1, schedule_entries.entries[map_idx],
		               mask_FF, mask_FE, mask_7F, mask_80, mask_01);
	}

	_mm512_store_si512((__m512i*)(working_code_data), working_code0);
	_mm512_store_si512((__m512i*)(working_code_data + 64), working_code1);
}

// One alg step for a single state. Shared dispatch body; mirrors _run_map_entry's
// switch so the 4-way kernel stays parity-locked to the 1-way one.
__forceinline void tm_avx512_r512s_8::_alg_dispatch(__m512i& w0, __m512i& w1, uint16& seed, unsigned alg_id,
	__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01)
{
	switch (alg_id)
	{
		case 0: alg_0(w0, w1, &seed, mask_FE); seed = rng->seed_forward_128[seed]; break;
		case 1: add_alg(w0, w1, &seed, rng->regular_rng_values_512_8_shuffled); seed = rng->seed_forward_128[seed]; break;
		case 2: alg_2(w0, w1, &seed, mask_80, mask_7F, mask_FE, mask_01); seed = rng->seed_forward_1[seed]; break;
		case 3: alg_3(w0, w1, &seed); seed = rng->seed_forward_128[seed]; break;
		case 4: add_alg(w0, w1, &seed, rng->alg4_values_512_8_shuffled); seed = rng->seed_forward_128[seed]; break;
		case 5: alg_5(w0, w1, &seed, mask_80, mask_7F, mask_FE, mask_01); seed = rng->seed_forward_1[seed]; break;
		case 6: alg_6(w0, w1, &seed, mask_7F); seed = rng->seed_forward_128[seed]; break;
		default: alg_7(w0, w1, mask_FF); break;
	}
}

// EXPERIMENT: merged branchless dispatch for the universal kernel (port of natmap
// blmerge). Folds the 6 non-butterfly algs into 3 candidate-pairs selected by a
// register blend-tree (no data-dependent branch); butterflies {2,5} stay branched.
// Universal-specific merges (operands index seed*128 into the big shuffled tables;
// seed is always a valid uint16 index so NO address clamp is needed):
//   {0,6}: (w shifted) ternlog rng  (shift dir + mask + alg0/alg6 table selected)
//   {1,4}: w + table[seed]          (both adds here; regular_rng vs alg4 table selected)
//   {3,7}: w ^ (regular_rng or 0xFF)
// Seed advance branchless: {0,1,3,4,6} -> seed_forward_128, {7} -> unchanged (the
// butterflies use seed_forward_1 in their early-return). Bit-identical to _alg_dispatch.
__forceinline void tm_avx512_r512s_8::_alg_dispatch_blmerge(__m512i& w0, __m512i& w1, uint16& seed, unsigned alg_id,
	__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01)
{
	if (alg_id == 2) { alg_2(w0, w1, &seed, mask_80, mask_7F, mask_FE, mask_01); seed = rng->seed_forward_1[seed]; return; }
	if (alg_id == 5) { alg_5(w0, w1, &seed, mask_80, mask_7F, mask_FE, mask_01); seed = rng->seed_forward_1[seed]; return; }
	const uint32_t off = (uint32_t)seed * 128u;
	const uint8* sh_tbl  = (alg_id == 6) ? rng->alg6_values_512_8_shuffled : rng->alg0_values_512_8_shuffled;
	const uint8* add_tbl = (alg_id == 4) ? rng->alg4_values_512_8_shuffled : rng->regular_rng_values_512_8_shuffled;
	const uint8* xor_tbl = rng->regular_rng_values_512_8_shuffled;
	const __mmask64 m6   = (alg_id == 6) ? ~0ull : 0ull;
	const __mmask64 m7   = (alg_id == 7) ? ~0ull : 0ull;
	const __mmask64 madd = (alg_id == 1 || alg_id == 4) ? ~0ull : 0ull;
	const __mmask64 mxor = (alg_id == 3 || alg_id == 7) ? ~0ull : 0ull;
	{   // half 0
		const __m512i shrng=_mm512_load_si512((const __m512i*)(sh_tbl+off));
		const __m512i adrng=_mm512_load_si512((const __m512i*)(add_tbl+off));
		const __m512i xrng =_mm512_load_si512((const __m512i*)(xor_tbl+off));
		const __m512i shifted=_mm512_mask_blend_epi8(m6,_mm512_slli_epi64(w0,1),_mm512_srli_epi64(w0,1));
		const __m512i smask  =_mm512_mask_blend_epi8(m6,mask_FE,mask_7F);
		const __m512i c_sh =_mm512_ternarylogic_epi32(shifted,smask,shrng,0xEA);
		const __m512i c_add=_mm512_add_epi8(w0,adrng);
		const __m512i c_x  =_mm512_xor_si512(w0,_mm512_mask_blend_epi8(m7,xrng,mask_FF));
		__m512i res=_mm512_mask_blend_epi8(madd,c_sh,c_add);
		w0=_mm512_mask_blend_epi8(mxor,res,c_x);
	}
	{   // half 1
		const __m512i shrng=_mm512_load_si512((const __m512i*)(sh_tbl+off+64));
		const __m512i adrng=_mm512_load_si512((const __m512i*)(add_tbl+off+64));
		const __m512i xrng =_mm512_load_si512((const __m512i*)(xor_tbl+off+64));
		const __m512i shifted=_mm512_mask_blend_epi8(m6,_mm512_slli_epi64(w1,1),_mm512_srli_epi64(w1,1));
		const __m512i smask  =_mm512_mask_blend_epi8(m6,mask_FE,mask_7F);
		const __m512i c_sh =_mm512_ternarylogic_epi32(shifted,smask,shrng,0xEA);
		const __m512i c_add=_mm512_add_epi8(w1,adrng);
		const __m512i c_x  =_mm512_xor_si512(w1,_mm512_mask_blend_epi8(m7,xrng,mask_FF));
		__m512i res=_mm512_mask_blend_epi8(madd,c_sh,c_add);
		w1=_mm512_mask_blend_epi8(mxor,res,c_x);
	}
	seed = (alg_id == 7) ? seed : rng->seed_forward_128[seed];
}

// EXPERIMENT v2 (lower-load): same merged branchless dispatch, but loads only ONE rng
// table per dispatch instead of three. Key idea: select the table *pointer* by alg_id
// (the table the selected alg needs), load it once, and compute all candidate ops on
// that single operand — only the candidate whose table matches alg_id is selected by
// the blend, the others are discarded. This removes the +38% L1 load amplification that
// sank the 3-load blmerge at high thread counts on the universal kernel's 8MB tables.
__forceinline void tm_avx512_r512s_8::_alg_dispatch_blmerge2(__m512i& w0, __m512i& w1, uint16& seed, unsigned alg_id,
	__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01)
{
	if (alg_id == 2) { alg_2(w0, w1, &seed, mask_80, mask_7F, mask_FE, mask_01); seed = rng->seed_forward_1[seed]; return; }
	if (alg_id == 5) { alg_5(w0, w1, &seed, mask_80, mask_7F, mask_FE, mask_01); seed = rng->seed_forward_1[seed]; return; }
	const uint32_t off = (uint32_t)seed * 128u;
	// one table for THIS alg: 0->alg0, {1,3,7}->regular, 4->alg4, 6->alg6. Select via a
	// register cmov chain (NOT a stack array — the array form added stack loads that
	// saturated the load ports and erased the single-load benefit at high thread count).
	const uint8* tbl = rng->regular_rng_values_512_8_shuffled;
	tbl = (alg_id == 0) ? rng->alg0_values_512_8_shuffled : tbl;
	tbl = (alg_id == 6) ? rng->alg6_values_512_8_shuffled : tbl;
	tbl = (alg_id == 4) ? rng->alg4_values_512_8_shuffled : tbl;
	const __mmask64 m6   = (alg_id == 6) ? ~0ull : 0ull;
	const __mmask64 m7   = (alg_id == 7) ? ~0ull : 0ull;
	const __mmask64 madd = (alg_id == 1 || alg_id == 4) ? ~0ull : 0ull;
	const __mmask64 mxor = (alg_id == 3 || alg_id == 7) ? ~0ull : 0ull;
	{   // half 0 — single load reused by all three candidate ops
		const __m512i rngv=_mm512_load_si512((const __m512i*)(tbl+off));
		const __m512i shifted=_mm512_mask_blend_epi8(m6,_mm512_slli_epi64(w0,1),_mm512_srli_epi64(w0,1));
		const __m512i smask  =_mm512_mask_blend_epi8(m6,mask_FE,mask_7F);
		const __m512i c_sh =_mm512_ternarylogic_epi32(shifted,smask,rngv,0xEA);
		const __m512i c_add=_mm512_add_epi8(w0,rngv);
		const __m512i c_x  =_mm512_xor_si512(w0,_mm512_mask_blend_epi8(m7,rngv,mask_FF));
		__m512i res=_mm512_mask_blend_epi8(madd,c_sh,c_add);
		w0=_mm512_mask_blend_epi8(mxor,res,c_x);
	}
	{   // half 1
		const __m512i rngv=_mm512_load_si512((const __m512i*)(tbl+off+64));
		const __m512i shifted=_mm512_mask_blend_epi8(m6,_mm512_slli_epi64(w1,1),_mm512_srli_epi64(w1,1));
		const __m512i smask  =_mm512_mask_blend_epi8(m6,mask_FE,mask_7F);
		const __m512i c_sh =_mm512_ternarylogic_epi32(shifted,smask,rngv,0xEA);
		const __m512i c_add=_mm512_add_epi8(w1,rngv);
		const __m512i c_x  =_mm512_xor_si512(w1,_mm512_mask_blend_epi8(m7,rngv,mask_FF));
		__m512i res=_mm512_mask_blend_epi8(madd,c_sh,c_add);
		w1=_mm512_mask_blend_epi8(mxor,res,c_x);
	}
	seed = (alg_id == 7) ? seed : rng->seed_forward_128[seed];
}

__forceinline void tm_avx512_r512s_8::_run_map_entry_x4(
	__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
	__m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
	const key_schedule::key_schedule_entry& schedule_entry,
	__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01)
{
	const uint16 base = (schedule_entry.rng1 << 8) | schedule_entry.rng2;
	uint16 sa = base, sb = base, sc = base, sd = base;
	uint16 nibble_selector = schedule_entry.nibble_selector;

	// NOT unrolled. The switch(i) below supplies the _mm_extract_epi8 immediate
	// per case, so the loop need not be unrolled to satisfy that constraint —
	// and forcing no-unroll keeps the 4x-interleaved body ~16x smaller. That is
	// decisive: the unrolled 4-way kernel was frontend/icache-bound (perf: 629M
	// L1i misses, 52% frontend stalls). Matches the codebase's prior finding
	// that stripping map-mode loop unrolling cut L1i misses ~100x.
#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
	for (int i = 0; i < 16; i++)
	{
		unsigned char nibble = (nibble_selector >> 15) & 0x01;
		nibble_selector = nibble_selector << 1;

		const __m128i a0l = _mm512_castsi512_si128(a0), a1l = _mm512_castsi512_si128(a1);
		const __m128i b0l = _mm512_castsi512_si128(b0), b1l = _mm512_castsi512_si128(b1);
		const __m128i c0l = _mm512_castsi512_si128(c0), c1l = _mm512_castsi512_si128(c1);
		const __m128i d0l = _mm512_castsi512_si128(d0), d1l = _mm512_castsi512_si128(d1);
		unsigned char ba = 0, bb = 0, bc = 0, bd = 0;
#define TM_X4_EXTRACT(out, r0, r1) \
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
		TM_X4_EXTRACT(ba, a0l, a1l);
		TM_X4_EXTRACT(bb, b0l, b1l);
		TM_X4_EXTRACT(bc, c0l, c1l);
		TM_X4_EXTRACT(bd, d0l, d1l);
#undef TM_X4_EXTRACT
		if (nibble == 1) { ba >>= 4; bb >>= 4; bc >>= 4; bd >>= 4; }

		_alg_dispatch(a0, a1, sa, (ba >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(b0, b1, sb, (bb >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(c0, c1, sc, (bc >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(d0, d1, sd, (bd >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
	}
}

void tm_avx512_r512s_8::run_maps_range_x4(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
	const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
	uint8* out0, uint8* out1, uint8* out2, uint8* out3)
{
	// Pool entries are not 64-byte aligned; use unaligned loads/stores.
	__m512i a0 = _mm512_loadu_si512((const void*)(in0)), a1 = _mm512_loadu_si512((const void*)(in0 + 64));
	__m512i b0 = _mm512_loadu_si512((const void*)(in1)), b1 = _mm512_loadu_si512((const void*)(in1 + 64));
	__m512i c0 = _mm512_loadu_si512((const void*)(in2)), c1 = _mm512_loadu_si512((const void*)(in2 + 64));
	__m512i d0 = _mm512_loadu_si512((const void*)(in3)), d1 = _mm512_loadu_si512((const void*)(in3 + 64));

	__m512i mask_FF = _mm512_set1_epi16(0xFFFF);
	__m512i mask_FE = _mm512_set1_epi16(0xFEFE);
	__m512i mask_7F = _mm512_set1_epi16(0x7F7F);
	__m512i mask_80 = _mm512_set1_epi16(0x8080);
	__m512i mask_01 = _mm512_set1_epi16(0x0101);

	if (end > schedule_entries.entries.size())
		end = schedule_entries.entries.size();
	for (std::size_t map_idx = begin; map_idx < end; map_idx++)
	{
		_run_map_entry_x4(a0, a1, b0, b1, c0, c1, d0, d1, schedule_entries.entries[map_idx],
		                  mask_FF, mask_FE, mask_7F, mask_80, mask_01);
	}

	_mm512_storeu_si512((void*)(out0), a0); _mm512_storeu_si512((void*)(out0 + 64), a1);
	_mm512_storeu_si512((void*)(out1), b0); _mm512_storeu_si512((void*)(out1 + 64), b1);
	_mm512_storeu_si512((void*)(out2), c0); _mm512_storeu_si512((void*)(out2 + 64), c1);
	_mm512_storeu_si512((void*)(out3), d0); _mm512_storeu_si512((void*)(out3 + 64), d1);
}

// 6-way: same body as _run_map_entry_x4, six states. No-unroll (switch(i)
// supplies the _mm_extract_epi8 immediate). PreIDs: all six source bytes are
// extracted in one block before the six dispatches, so the vpextrb reads hoist
// together and the OOO engine overlaps their gather latency (the GPU
// ilp6_preids trick).
__forceinline void tm_avx512_r512s_8::_run_map_entry_x6(
	__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
	__m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
	__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1,
	const key_schedule::key_schedule_entry& schedule_entry,
	__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01)
{
	const uint16 base = (schedule_entry.rng1 << 8) | schedule_entry.rng2;
	uint16 sa = base, sb = base, sc = base, sd = base, se = base, sf = base;
	uint16 nibble_selector = schedule_entry.nibble_selector;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
	for (int i = 0; i < 16; i++)
	{
		unsigned char nibble = (nibble_selector >> 15) & 0x01;
		nibble_selector = nibble_selector << 1;

		const __m128i a0l = _mm512_castsi512_si128(a0), a1l = _mm512_castsi512_si128(a1);
		const __m128i b0l = _mm512_castsi512_si128(b0), b1l = _mm512_castsi512_si128(b1);
		const __m128i c0l = _mm512_castsi512_si128(c0), c1l = _mm512_castsi512_si128(c1);
		const __m128i d0l = _mm512_castsi512_si128(d0), d1l = _mm512_castsi512_si128(d1);
		const __m128i e0l = _mm512_castsi512_si128(e0), e1l = _mm512_castsi512_si128(e1);
		const __m128i f0l = _mm512_castsi512_si128(f0), f1l = _mm512_castsi512_si128(f1);
		unsigned char ba = 0, bb = 0, bc = 0, bd = 0, be = 0, bf = 0;
#define TM_X6_EXTRACT(out, r0, r1) \
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
		TM_X6_EXTRACT(ba, a0l, a1l);
		TM_X6_EXTRACT(bb, b0l, b1l);
		TM_X6_EXTRACT(bc, c0l, c1l);
		TM_X6_EXTRACT(bd, d0l, d1l);
		TM_X6_EXTRACT(be, e0l, e1l);
		TM_X6_EXTRACT(bf, f0l, f1l);
#undef TM_X6_EXTRACT
		if (nibble == 1) { ba >>= 4; bb >>= 4; bc >>= 4; bd >>= 4; be >>= 4; bf >>= 4; }

		_alg_dispatch(a0, a1, sa, (ba >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(b0, b1, sb, (bb >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(c0, c1, sc, (bc >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(d0, d1, sd, (bd >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(e0, e1, se, (be >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(f0, f1, sf, (bf >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
	}
}

void tm_avx512_r512s_8::run_maps_range_x6(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
	const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3, const uint8* in4, const uint8* in5,
	uint8* out0, uint8* out1, uint8* out2, uint8* out3, uint8* out4, uint8* out5)
{
	__m512i a0 = _mm512_loadu_si512((const void*)(in0)), a1 = _mm512_loadu_si512((const void*)(in0 + 64));
	__m512i b0 = _mm512_loadu_si512((const void*)(in1)), b1 = _mm512_loadu_si512((const void*)(in1 + 64));
	__m512i c0 = _mm512_loadu_si512((const void*)(in2)), c1 = _mm512_loadu_si512((const void*)(in2 + 64));
	__m512i d0 = _mm512_loadu_si512((const void*)(in3)), d1 = _mm512_loadu_si512((const void*)(in3 + 64));
	__m512i e0 = _mm512_loadu_si512((const void*)(in4)), e1 = _mm512_loadu_si512((const void*)(in4 + 64));
	__m512i f0 = _mm512_loadu_si512((const void*)(in5)), f1 = _mm512_loadu_si512((const void*)(in5 + 64));

	__m512i mask_FF = _mm512_set1_epi16(0xFFFF);
	__m512i mask_FE = _mm512_set1_epi16(0xFEFE);
	__m512i mask_7F = _mm512_set1_epi16(0x7F7F);
	__m512i mask_80 = _mm512_set1_epi16(0x8080);
	__m512i mask_01 = _mm512_set1_epi16(0x0101);

	if (end > schedule_entries.entries.size())
		end = schedule_entries.entries.size();
	for (std::size_t map_idx = begin; map_idx < end; map_idx++)
	{
		_run_map_entry_x6(a0, a1, b0, b1, c0, c1, d0, d1, e0, e1, f0, f1,
		                  schedule_entries.entries[map_idx],
		                  mask_FF, mask_FE, mask_7F, mask_80, mask_01);
	}

	_mm512_storeu_si512((void*)(out0), a0); _mm512_storeu_si512((void*)(out0 + 64), a1);
	_mm512_storeu_si512((void*)(out1), b0); _mm512_storeu_si512((void*)(out1 + 64), b1);
	_mm512_storeu_si512((void*)(out2), c0); _mm512_storeu_si512((void*)(out2 + 64), c1);
	_mm512_storeu_si512((void*)(out3), d0); _mm512_storeu_si512((void*)(out3 + 64), d1);
	_mm512_storeu_si512((void*)(out4), e0); _mm512_storeu_si512((void*)(out4 + 64), e1);
	_mm512_storeu_si512((void*)(out5), f0); _mm512_storeu_si512((void*)(out5 + 64), f1);
}

// 8-way: same body, eight states. PreIDs batched extraction before dispatch.
__forceinline void tm_avx512_r512s_8::_run_map_entry_x8(
	__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
	__m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
	__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1,
	__m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
	const key_schedule::key_schedule_entry& schedule_entry,
	__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01)
{
	const uint16 base = (schedule_entry.rng1 << 8) | schedule_entry.rng2;
	uint16 sa = base, sb = base, sc = base, sd = base, se = base, sf = base, sg = base, sh = base;
	uint16 nibble_selector = schedule_entry.nibble_selector;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
	for (int i = 0; i < 16; i++)
	{
		unsigned char nibble = (nibble_selector >> 15) & 0x01;
		nibble_selector = nibble_selector << 1;

		const __m128i a0l = _mm512_castsi512_si128(a0), a1l = _mm512_castsi512_si128(a1);
		const __m128i b0l = _mm512_castsi512_si128(b0), b1l = _mm512_castsi512_si128(b1);
		const __m128i c0l = _mm512_castsi512_si128(c0), c1l = _mm512_castsi512_si128(c1);
		const __m128i d0l = _mm512_castsi512_si128(d0), d1l = _mm512_castsi512_si128(d1);
		const __m128i e0l = _mm512_castsi512_si128(e0), e1l = _mm512_castsi512_si128(e1);
		const __m128i f0l = _mm512_castsi512_si128(f0), f1l = _mm512_castsi512_si128(f1);
		const __m128i g0l = _mm512_castsi512_si128(g0), g1l = _mm512_castsi512_si128(g1);
		const __m128i h0l = _mm512_castsi512_si128(h0), h1l = _mm512_castsi512_si128(h1);
		unsigned char ba=0, bb=0, bc=0, bd=0, be=0, bf=0, bg=0, bh=0;
#define TM_X8_EXTRACT(out, r0, r1) \
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
		TM_X8_EXTRACT(ba, a0l, a1l); TM_X8_EXTRACT(bb, b0l, b1l);
		TM_X8_EXTRACT(bc, c0l, c1l); TM_X8_EXTRACT(bd, d0l, d1l);
		TM_X8_EXTRACT(be, e0l, e1l); TM_X8_EXTRACT(bf, f0l, f1l);
		TM_X8_EXTRACT(bg, g0l, g1l); TM_X8_EXTRACT(bh, h0l, h1l);
#undef TM_X8_EXTRACT
		if (nibble == 1) { ba>>=4; bb>>=4; bc>>=4; bd>>=4; be>>=4; bf>>=4; bg>>=4; bh>>=4; }

		_alg_dispatch(a0, a1, sa, (ba >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(b0, b1, sb, (bb >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(c0, c1, sc, (bc >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(d0, d1, sd, (bd >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(e0, e1, se, (be >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(f0, f1, sf, (bf >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(g0, g1, sg, (bg >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(h0, h1, sh, (bh >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
	}
}

// EXPERIMENT: x8 entry using the merged branchless dispatch.
__forceinline void tm_avx512_r512s_8::_run_map_entry_x8_blmerge(
	__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
	__m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
	__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1,
	__m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
	const key_schedule::key_schedule_entry& schedule_entry,
	__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01)
{
	const uint16 base = (schedule_entry.rng1 << 8) | schedule_entry.rng2;
	uint16 sa = base, sb = base, sc = base, sd = base, se = base, sf = base, sg = base, sh = base;
	uint16 nibble_selector = schedule_entry.nibble_selector;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
	for (int i = 0; i < 16; i++)
	{
		unsigned char nibble = (nibble_selector >> 15) & 0x01;
		nibble_selector = nibble_selector << 1;

		const __m128i a0l = _mm512_castsi512_si128(a0), a1l = _mm512_castsi512_si128(a1);
		const __m128i b0l = _mm512_castsi512_si128(b0), b1l = _mm512_castsi512_si128(b1);
		const __m128i c0l = _mm512_castsi512_si128(c0), c1l = _mm512_castsi512_si128(c1);
		const __m128i d0l = _mm512_castsi512_si128(d0), d1l = _mm512_castsi512_si128(d1);
		const __m128i e0l = _mm512_castsi512_si128(e0), e1l = _mm512_castsi512_si128(e1);
		const __m128i f0l = _mm512_castsi512_si128(f0), f1l = _mm512_castsi512_si128(f1);
		const __m128i g0l = _mm512_castsi512_si128(g0), g1l = _mm512_castsi512_si128(g1);
		const __m128i h0l = _mm512_castsi512_si128(h0), h1l = _mm512_castsi512_si128(h1);
		unsigned char ba=0, bb=0, bc=0, bd=0, be=0, bf=0, bg=0, bh=0;
#define TM_X8_EXTRACT(out, r0, r1) \
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
		TM_X8_EXTRACT(ba, a0l, a1l); TM_X8_EXTRACT(bb, b0l, b1l);
		TM_X8_EXTRACT(bc, c0l, c1l); TM_X8_EXTRACT(bd, d0l, d1l);
		TM_X8_EXTRACT(be, e0l, e1l); TM_X8_EXTRACT(bf, f0l, f1l);
		TM_X8_EXTRACT(bg, g0l, g1l); TM_X8_EXTRACT(bh, h0l, h1l);
#undef TM_X8_EXTRACT
		if (nibble == 1) { ba>>=4; bb>>=4; bc>>=4; bd>>=4; be>>=4; bf>>=4; bg>>=4; bh>>=4; }

		_alg_dispatch_blmerge(a0, a1, sa, (ba >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch_blmerge(b0, b1, sb, (bb >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch_blmerge(c0, c1, sc, (bc >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch_blmerge(d0, d1, sd, (bd >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch_blmerge(e0, e1, se, (be >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch_blmerge(f0, f1, sf, (bf >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch_blmerge(g0, g1, sg, (bg >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch_blmerge(h0, h1, sh, (bh >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
	}
}

void tm_avx512_r512s_8::run_maps_range_x8_blmerge(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
	const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
	const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
	uint8* out0, uint8* out1, uint8* out2, uint8* out3,
	uint8* out4, uint8* out5, uint8* out6, uint8* out7)
{
	__m512i a0 = _mm512_loadu_si512((const void*)(in0)), a1 = _mm512_loadu_si512((const void*)(in0 + 64));
	__m512i b0 = _mm512_loadu_si512((const void*)(in1)), b1 = _mm512_loadu_si512((const void*)(in1 + 64));
	__m512i c0 = _mm512_loadu_si512((const void*)(in2)), c1 = _mm512_loadu_si512((const void*)(in2 + 64));
	__m512i d0 = _mm512_loadu_si512((const void*)(in3)), d1 = _mm512_loadu_si512((const void*)(in3 + 64));
	__m512i e0 = _mm512_loadu_si512((const void*)(in4)), e1 = _mm512_loadu_si512((const void*)(in4 + 64));
	__m512i f0 = _mm512_loadu_si512((const void*)(in5)), f1 = _mm512_loadu_si512((const void*)(in5 + 64));
	__m512i g0 = _mm512_loadu_si512((const void*)(in6)), g1 = _mm512_loadu_si512((const void*)(in6 + 64));
	__m512i h0 = _mm512_loadu_si512((const void*)(in7)), h1 = _mm512_loadu_si512((const void*)(in7 + 64));
	__m512i mask_FF = _mm512_set1_epi16(0xFFFF), mask_FE = _mm512_set1_epi16(0xFEFE), mask_7F = _mm512_set1_epi16(0x7F7F),
	        mask_80 = _mm512_set1_epi16(0x8080), mask_01 = _mm512_set1_epi16(0x0101);
	if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
	for (std::size_t map_idx = begin; map_idx < end; map_idx++)
		_run_map_entry_x8_blmerge(a0, a1, b0, b1, c0, c1, d0, d1, e0, e1, f0, f1, g0, g1, h0, h1,
		                          schedule_entries.entries[map_idx], mask_FF, mask_FE, mask_7F, mask_80, mask_01);
	_mm512_storeu_si512((void*)(out0), a0); _mm512_storeu_si512((void*)(out0 + 64), a1);
	_mm512_storeu_si512((void*)(out1), b0); _mm512_storeu_si512((void*)(out1 + 64), b1);
	_mm512_storeu_si512((void*)(out2), c0); _mm512_storeu_si512((void*)(out2 + 64), c1);
	_mm512_storeu_si512((void*)(out3), d0); _mm512_storeu_si512((void*)(out3 + 64), d1);
	_mm512_storeu_si512((void*)(out4), e0); _mm512_storeu_si512((void*)(out4 + 64), e1);
	_mm512_storeu_si512((void*)(out5), f0); _mm512_storeu_si512((void*)(out5 + 64), f1);
	_mm512_storeu_si512((void*)(out6), g0); _mm512_storeu_si512((void*)(out6 + 64), g1);
	_mm512_storeu_si512((void*)(out7), h0); _mm512_storeu_si512((void*)(out7 + 64), h1);
}

// EXPERIMENT v2 (lower-load): x8 entry + wrapper using _alg_dispatch_blmerge2.
__forceinline void tm_avx512_r512s_8::_run_map_entry_x8_blmerge2(
	__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
	__m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
	__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1,
	__m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
	const key_schedule::key_schedule_entry& schedule_entry,
	__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01)
{
	const uint16 base = (schedule_entry.rng1 << 8) | schedule_entry.rng2;
	uint16 sa = base, sb = base, sc = base, sd = base, se = base, sf = base, sg = base, sh = base;
	uint16 nibble_selector = schedule_entry.nibble_selector;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
	for (int i = 0; i < 16; i++)
	{
		unsigned char nibble = (nibble_selector >> 15) & 0x01;
		nibble_selector = nibble_selector << 1;

		const __m128i a0l = _mm512_castsi512_si128(a0), a1l = _mm512_castsi512_si128(a1);
		const __m128i b0l = _mm512_castsi512_si128(b0), b1l = _mm512_castsi512_si128(b1);
		const __m128i c0l = _mm512_castsi512_si128(c0), c1l = _mm512_castsi512_si128(c1);
		const __m128i d0l = _mm512_castsi512_si128(d0), d1l = _mm512_castsi512_si128(d1);
		const __m128i e0l = _mm512_castsi512_si128(e0), e1l = _mm512_castsi512_si128(e1);
		const __m128i f0l = _mm512_castsi512_si128(f0), f1l = _mm512_castsi512_si128(f1);
		const __m128i g0l = _mm512_castsi512_si128(g0), g1l = _mm512_castsi512_si128(g1);
		const __m128i h0l = _mm512_castsi512_si128(h0), h1l = _mm512_castsi512_si128(h1);
		unsigned char ba=0, bb=0, bc=0, bd=0, be=0, bf=0, bg=0, bh=0;
#define TM_X8_EXTRACT(out, r0, r1) \
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
		TM_X8_EXTRACT(ba, a0l, a1l); TM_X8_EXTRACT(bb, b0l, b1l);
		TM_X8_EXTRACT(bc, c0l, c1l); TM_X8_EXTRACT(bd, d0l, d1l);
		TM_X8_EXTRACT(be, e0l, e1l); TM_X8_EXTRACT(bf, f0l, f1l);
		TM_X8_EXTRACT(bg, g0l, g1l); TM_X8_EXTRACT(bh, h0l, h1l);
#undef TM_X8_EXTRACT
		if (nibble == 1) { ba>>=4; bb>>=4; bc>>=4; bd>>=4; be>>=4; bf>>=4; bg>>=4; bh>>=4; }

		_alg_dispatch_blmerge2(a0, a1, sa, (ba >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch_blmerge2(b0, b1, sb, (bb >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch_blmerge2(c0, c1, sc, (bc >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch_blmerge2(d0, d1, sd, (bd >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch_blmerge2(e0, e1, se, (be >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch_blmerge2(f0, f1, sf, (bf >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch_blmerge2(g0, g1, sg, (bg >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch_blmerge2(h0, h1, sh, (bh >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
	}
}

void tm_avx512_r512s_8::run_maps_range_x8_blmerge2(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
	const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
	const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
	uint8* out0, uint8* out1, uint8* out2, uint8* out3,
	uint8* out4, uint8* out5, uint8* out6, uint8* out7)
{
	__m512i a0 = _mm512_loadu_si512((const void*)(in0)), a1 = _mm512_loadu_si512((const void*)(in0 + 64));
	__m512i b0 = _mm512_loadu_si512((const void*)(in1)), b1 = _mm512_loadu_si512((const void*)(in1 + 64));
	__m512i c0 = _mm512_loadu_si512((const void*)(in2)), c1 = _mm512_loadu_si512((const void*)(in2 + 64));
	__m512i d0 = _mm512_loadu_si512((const void*)(in3)), d1 = _mm512_loadu_si512((const void*)(in3 + 64));
	__m512i e0 = _mm512_loadu_si512((const void*)(in4)), e1 = _mm512_loadu_si512((const void*)(in4 + 64));
	__m512i f0 = _mm512_loadu_si512((const void*)(in5)), f1 = _mm512_loadu_si512((const void*)(in5 + 64));
	__m512i g0 = _mm512_loadu_si512((const void*)(in6)), g1 = _mm512_loadu_si512((const void*)(in6 + 64));
	__m512i h0 = _mm512_loadu_si512((const void*)(in7)), h1 = _mm512_loadu_si512((const void*)(in7 + 64));
	__m512i mask_FF = _mm512_set1_epi16(0xFFFF), mask_FE = _mm512_set1_epi16(0xFEFE), mask_7F = _mm512_set1_epi16(0x7F7F),
	        mask_80 = _mm512_set1_epi16(0x8080), mask_01 = _mm512_set1_epi16(0x0101);
	if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
	for (std::size_t map_idx = begin; map_idx < end; map_idx++)
		_run_map_entry_x8_blmerge2(a0, a1, b0, b1, c0, c1, d0, d1, e0, e1, f0, f1, g0, g1, h0, h1,
		                           schedule_entries.entries[map_idx], mask_FF, mask_FE, mask_7F, mask_80, mask_01);
	_mm512_storeu_si512((void*)(out0), a0); _mm512_storeu_si512((void*)(out0 + 64), a1);
	_mm512_storeu_si512((void*)(out1), b0); _mm512_storeu_si512((void*)(out1 + 64), b1);
	_mm512_storeu_si512((void*)(out2), c0); _mm512_storeu_si512((void*)(out2 + 64), c1);
	_mm512_storeu_si512((void*)(out3), d0); _mm512_storeu_si512((void*)(out3 + 64), d1);
	_mm512_storeu_si512((void*)(out4), e0); _mm512_storeu_si512((void*)(out4 + 64), e1);
	_mm512_storeu_si512((void*)(out5), f0); _mm512_storeu_si512((void*)(out5 + 64), f1);
	_mm512_storeu_si512((void*)(out6), g0); _mm512_storeu_si512((void*)(out6 + 64), g1);
	_mm512_storeu_si512((void*)(out7), h0); _mm512_storeu_si512((void*)(out7 + 64), h1);
}

void tm_avx512_r512s_8::run_maps_range_x8(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
	const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
	const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
	uint8* out0, uint8* out1, uint8* out2, uint8* out3,
	uint8* out4, uint8* out5, uint8* out6, uint8* out7)
{
	__m512i a0 = _mm512_loadu_si512((const void*)(in0)), a1 = _mm512_loadu_si512((const void*)(in0 + 64));
	__m512i b0 = _mm512_loadu_si512((const void*)(in1)), b1 = _mm512_loadu_si512((const void*)(in1 + 64));
	__m512i c0 = _mm512_loadu_si512((const void*)(in2)), c1 = _mm512_loadu_si512((const void*)(in2 + 64));
	__m512i d0 = _mm512_loadu_si512((const void*)(in3)), d1 = _mm512_loadu_si512((const void*)(in3 + 64));
	__m512i e0 = _mm512_loadu_si512((const void*)(in4)), e1 = _mm512_loadu_si512((const void*)(in4 + 64));
	__m512i f0 = _mm512_loadu_si512((const void*)(in5)), f1 = _mm512_loadu_si512((const void*)(in5 + 64));
	__m512i g0 = _mm512_loadu_si512((const void*)(in6)), g1 = _mm512_loadu_si512((const void*)(in6 + 64));
	__m512i h0 = _mm512_loadu_si512((const void*)(in7)), h1 = _mm512_loadu_si512((const void*)(in7 + 64));

	__m512i mask_FF = _mm512_set1_epi16(0xFFFF);
	__m512i mask_FE = _mm512_set1_epi16(0xFEFE);
	__m512i mask_7F = _mm512_set1_epi16(0x7F7F);
	__m512i mask_80 = _mm512_set1_epi16(0x8080);
	__m512i mask_01 = _mm512_set1_epi16(0x0101);

	if (end > schedule_entries.entries.size())
		end = schedule_entries.entries.size();
	for (std::size_t map_idx = begin; map_idx < end; map_idx++)
	{
		_run_map_entry_x8(a0, a1, b0, b1, c0, c1, d0, d1, e0, e1, f0, f1, g0, g1, h0, h1,
		                  schedule_entries.entries[map_idx],
		                  mask_FF, mask_FE, mask_7F, mask_80, mask_01);
	}

	_mm512_storeu_si512((void*)(out0), a0); _mm512_storeu_si512((void*)(out0 + 64), a1);
	_mm512_storeu_si512((void*)(out1), b0); _mm512_storeu_si512((void*)(out1 + 64), b1);
	_mm512_storeu_si512((void*)(out2), c0); _mm512_storeu_si512((void*)(out2 + 64), c1);
	_mm512_storeu_si512((void*)(out3), d0); _mm512_storeu_si512((void*)(out3 + 64), d1);
	_mm512_storeu_si512((void*)(out4), e0); _mm512_storeu_si512((void*)(out4 + 64), e1);
	_mm512_storeu_si512((void*)(out5), f0); _mm512_storeu_si512((void*)(out5 + 64), f1);
	_mm512_storeu_si512((void*)(out6), g0); _mm512_storeu_si512((void*)(out6 + 64), g1);
	_mm512_storeu_si512((void*)(out7), h0); _mm512_storeu_si512((void*)(out7 + 64), h1);
}

// 10-way: ten states (20 state ZMM). At the 32-ZMM edge — objdump-checked for spills.
__forceinline void tm_avx512_r512s_8::_run_map_entry_x10(
	__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
	__m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
	__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1,
	__m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
	__m512i& i0, __m512i& i1, __m512i& j0, __m512i& j1,
	const key_schedule::key_schedule_entry& schedule_entry,
	__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01)
{
	const uint16 base = (schedule_entry.rng1 << 8) | schedule_entry.rng2;
	uint16 sa=base, sb=base, sc=base, sd=base, se=base, sf=base, sg=base, sh=base, si=base, sj=base;
	uint16 nibble_selector = schedule_entry.nibble_selector;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
	for (int it = 0; it < 16; it++)
	{
		unsigned char nibble = (nibble_selector >> 15) & 0x01;
		nibble_selector = nibble_selector << 1;

		const __m128i a0l = _mm512_castsi512_si128(a0), a1l = _mm512_castsi512_si128(a1);
		const __m128i b0l = _mm512_castsi512_si128(b0), b1l = _mm512_castsi512_si128(b1);
		const __m128i c0l = _mm512_castsi512_si128(c0), c1l = _mm512_castsi512_si128(c1);
		const __m128i d0l = _mm512_castsi512_si128(d0), d1l = _mm512_castsi512_si128(d1);
		const __m128i e0l = _mm512_castsi512_si128(e0), e1l = _mm512_castsi512_si128(e1);
		const __m128i f0l = _mm512_castsi512_si128(f0), f1l = _mm512_castsi512_si128(f1);
		const __m128i g0l = _mm512_castsi512_si128(g0), g1l = _mm512_castsi512_si128(g1);
		const __m128i h0l = _mm512_castsi512_si128(h0), h1l = _mm512_castsi512_si128(h1);
		const __m128i i0l = _mm512_castsi512_si128(i0), i1l = _mm512_castsi512_si128(i1);
		const __m128i j0l = _mm512_castsi512_si128(j0), j1l = _mm512_castsi512_si128(j1);
		unsigned char ba=0, bb=0, bc=0, bd=0, be=0, bf=0, bg=0, bh=0, bi=0, bj=0;
#define TM_X10_EXTRACT(out, r0, r1) \
		switch (it) { \
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
		TM_X10_EXTRACT(ba, a0l, a1l); TM_X10_EXTRACT(bb, b0l, b1l);
		TM_X10_EXTRACT(bc, c0l, c1l); TM_X10_EXTRACT(bd, d0l, d1l);
		TM_X10_EXTRACT(be, e0l, e1l); TM_X10_EXTRACT(bf, f0l, f1l);
		TM_X10_EXTRACT(bg, g0l, g1l); TM_X10_EXTRACT(bh, h0l, h1l);
		TM_X10_EXTRACT(bi, i0l, i1l); TM_X10_EXTRACT(bj, j0l, j1l);
#undef TM_X10_EXTRACT
		if (nibble == 1) { ba>>=4; bb>>=4; bc>>=4; bd>>=4; be>>=4; bf>>=4; bg>>=4; bh>>=4; bi>>=4; bj>>=4; }

		_alg_dispatch(a0, a1, sa, (ba >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(b0, b1, sb, (bb >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(c0, c1, sc, (bc >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(d0, d1, sd, (bd >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(e0, e1, se, (be >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(f0, f1, sf, (bf >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(g0, g1, sg, (bg >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(h0, h1, sh, (bh >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(i0, i1, si, (bi >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
		_alg_dispatch(j0, j1, sj, (bj >> 1) & 0x07, mask_FF, mask_FE, mask_7F, mask_80, mask_01);
	}
}

void tm_avx512_r512s_8::run_maps_range_x10(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
	const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3, const uint8* in4,
	const uint8* in5, const uint8* in6, const uint8* in7, const uint8* in8, const uint8* in9,
	uint8* out0, uint8* out1, uint8* out2, uint8* out3, uint8* out4,
	uint8* out5, uint8* out6, uint8* out7, uint8* out8, uint8* out9)
{
	__m512i a0 = _mm512_loadu_si512((const void*)(in0)), a1 = _mm512_loadu_si512((const void*)(in0 + 64));
	__m512i b0 = _mm512_loadu_si512((const void*)(in1)), b1 = _mm512_loadu_si512((const void*)(in1 + 64));
	__m512i c0 = _mm512_loadu_si512((const void*)(in2)), c1 = _mm512_loadu_si512((const void*)(in2 + 64));
	__m512i d0 = _mm512_loadu_si512((const void*)(in3)), d1 = _mm512_loadu_si512((const void*)(in3 + 64));
	__m512i e0 = _mm512_loadu_si512((const void*)(in4)), e1 = _mm512_loadu_si512((const void*)(in4 + 64));
	__m512i f0 = _mm512_loadu_si512((const void*)(in5)), f1 = _mm512_loadu_si512((const void*)(in5 + 64));
	__m512i g0 = _mm512_loadu_si512((const void*)(in6)), g1 = _mm512_loadu_si512((const void*)(in6 + 64));
	__m512i h0 = _mm512_loadu_si512((const void*)(in7)), h1 = _mm512_loadu_si512((const void*)(in7 + 64));
	__m512i i0 = _mm512_loadu_si512((const void*)(in8)), i1 = _mm512_loadu_si512((const void*)(in8 + 64));
	__m512i j0 = _mm512_loadu_si512((const void*)(in9)), j1 = _mm512_loadu_si512((const void*)(in9 + 64));

	__m512i mask_FF = _mm512_set1_epi16(0xFFFF);
	__m512i mask_FE = _mm512_set1_epi16(0xFEFE);
	__m512i mask_7F = _mm512_set1_epi16(0x7F7F);
	__m512i mask_80 = _mm512_set1_epi16(0x8080);
	__m512i mask_01 = _mm512_set1_epi16(0x0101);

	if (end > schedule_entries.entries.size())
		end = schedule_entries.entries.size();
	for (std::size_t map_idx = begin; map_idx < end; map_idx++)
	{
		_run_map_entry_x10(a0, a1, b0, b1, c0, c1, d0, d1, e0, e1, f0, f1, g0, g1, h0, h1, i0, i1, j0, j1,
		                   schedule_entries.entries[map_idx],
		                   mask_FF, mask_FE, mask_7F, mask_80, mask_01);
	}

	_mm512_storeu_si512((void*)(out0), a0); _mm512_storeu_si512((void*)(out0 + 64), a1);
	_mm512_storeu_si512((void*)(out1), b0); _mm512_storeu_si512((void*)(out1 + 64), b1);
	_mm512_storeu_si512((void*)(out2), c0); _mm512_storeu_si512((void*)(out2 + 64), c1);
	_mm512_storeu_si512((void*)(out3), d0); _mm512_storeu_si512((void*)(out3 + 64), d1);
	_mm512_storeu_si512((void*)(out4), e0); _mm512_storeu_si512((void*)(out4 + 64), e1);
	_mm512_storeu_si512((void*)(out5), f0); _mm512_storeu_si512((void*)(out5 + 64), f1);
	_mm512_storeu_si512((void*)(out6), g0); _mm512_storeu_si512((void*)(out6 + 64), g1);
	_mm512_storeu_si512((void*)(out7), h0); _mm512_storeu_si512((void*)(out7 + 64), h1);
	_mm512_storeu_si512((void*)(out8), i0); _mm512_storeu_si512((void*)(out8 + 64), i1);
	_mm512_storeu_si512((void*)(out9), j0); _mm512_storeu_si512((void*)(out9 + 64), j1);
}

// 12-way: twelve states (24 state ZMM). Past the 32-ZMM budget once masks+temps
// are added — expected to spill; the ceiling probe for the width study.
__forceinline void tm_avx512_r512s_8::_run_map_entry_x12(
	__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1,
	__m512i& d0, __m512i& d1, __m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1,
	__m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1, __m512i& i0, __m512i& i1,
	__m512i& j0, __m512i& j1, __m512i& k0, __m512i& k1, __m512i& l0, __m512i& l1,
	const key_schedule::key_schedule_entry& schedule_entry,
	__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01)
{
	const uint16 base = (schedule_entry.rng1 << 8) | schedule_entry.rng2;
	uint16 sa=base,sb=base,sc=base,sd=base,se=base,sf=base,sg=base,sh=base,si=base,sj=base,sk=base,sl=base;
	uint16 nibble_selector = schedule_entry.nibble_selector;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
	for (int it = 0; it < 16; it++)
	{
		unsigned char nibble = (nibble_selector >> 15) & 0x01;
		nibble_selector = nibble_selector << 1;

		const __m128i a0l = _mm512_castsi512_si128(a0), a1l = _mm512_castsi512_si128(a1);
		const __m128i b0l = _mm512_castsi512_si128(b0), b1l = _mm512_castsi512_si128(b1);
		const __m128i c0l = _mm512_castsi512_si128(c0), c1l = _mm512_castsi512_si128(c1);
		const __m128i d0l = _mm512_castsi512_si128(d0), d1l = _mm512_castsi512_si128(d1);
		const __m128i e0l = _mm512_castsi512_si128(e0), e1l = _mm512_castsi512_si128(e1);
		const __m128i f0l = _mm512_castsi512_si128(f0), f1l = _mm512_castsi512_si128(f1);
		const __m128i g0l = _mm512_castsi512_si128(g0), g1l = _mm512_castsi512_si128(g1);
		const __m128i h0l = _mm512_castsi512_si128(h0), h1l = _mm512_castsi512_si128(h1);
		const __m128i i0l = _mm512_castsi512_si128(i0), i1l = _mm512_castsi512_si128(i1);
		const __m128i j0l = _mm512_castsi512_si128(j0), j1l = _mm512_castsi512_si128(j1);
		const __m128i k0l = _mm512_castsi512_si128(k0), k1l = _mm512_castsi512_si128(k1);
		const __m128i l0l = _mm512_castsi512_si128(l0), l1l = _mm512_castsi512_si128(l1);
		unsigned char ba=0,bb=0,bc=0,bd=0,be=0,bf=0,bg=0,bh=0,bi=0,bj=0,bk=0,bl=0;
#define TM_X12_EXTRACT(out, r0, r1) \
		switch (it) { \
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
		TM_X12_EXTRACT(ba,a0l,a1l); TM_X12_EXTRACT(bb,b0l,b1l); TM_X12_EXTRACT(bc,c0l,c1l);
		TM_X12_EXTRACT(bd,d0l,d1l); TM_X12_EXTRACT(be,e0l,e1l); TM_X12_EXTRACT(bf,f0l,f1l);
		TM_X12_EXTRACT(bg,g0l,g1l); TM_X12_EXTRACT(bh,h0l,h1l); TM_X12_EXTRACT(bi,i0l,i1l);
		TM_X12_EXTRACT(bj,j0l,j1l); TM_X12_EXTRACT(bk,k0l,k1l); TM_X12_EXTRACT(bl,l0l,l1l);
#undef TM_X12_EXTRACT
		if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4;be>>=4;bf>>=4;bg>>=4;bh>>=4;bi>>=4;bj>>=4;bk>>=4;bl>>=4; }

		_alg_dispatch(a0,a1,sa,(ba>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(b0,b1,sb,(bb>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(c0,c1,sc,(bc>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(d0,d1,sd,(bd>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(e0,e1,se,(be>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(f0,f1,sf,(bf>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(g0,g1,sg,(bg>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(h0,h1,sh,(bh>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(i0,i1,si,(bi>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(j0,j1,sj,(bj>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(k0,k1,sk,(bk>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(l0,l1,sl,(bl>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
	}
}

void tm_avx512_r512s_8::run_maps_range_x12(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
	const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3, const uint8* in4, const uint8* in5,
	const uint8* in6, const uint8* in7, const uint8* in8, const uint8* in9, const uint8* in10, const uint8* in11,
	uint8* out0, uint8* out1, uint8* out2, uint8* out3, uint8* out4, uint8* out5,
	uint8* out6, uint8* out7, uint8* out8, uint8* out9, uint8* out10, uint8* out11)
{
	__m512i a0 = _mm512_loadu_si512((const void*)(in0)), a1 = _mm512_loadu_si512((const void*)(in0 + 64));
	__m512i b0 = _mm512_loadu_si512((const void*)(in1)), b1 = _mm512_loadu_si512((const void*)(in1 + 64));
	__m512i c0 = _mm512_loadu_si512((const void*)(in2)), c1 = _mm512_loadu_si512((const void*)(in2 + 64));
	__m512i d0 = _mm512_loadu_si512((const void*)(in3)), d1 = _mm512_loadu_si512((const void*)(in3 + 64));
	__m512i e0 = _mm512_loadu_si512((const void*)(in4)), e1 = _mm512_loadu_si512((const void*)(in4 + 64));
	__m512i f0 = _mm512_loadu_si512((const void*)(in5)), f1 = _mm512_loadu_si512((const void*)(in5 + 64));
	__m512i g0 = _mm512_loadu_si512((const void*)(in6)), g1 = _mm512_loadu_si512((const void*)(in6 + 64));
	__m512i h0 = _mm512_loadu_si512((const void*)(in7)), h1 = _mm512_loadu_si512((const void*)(in7 + 64));
	__m512i i0 = _mm512_loadu_si512((const void*)(in8)), i1 = _mm512_loadu_si512((const void*)(in8 + 64));
	__m512i j0 = _mm512_loadu_si512((const void*)(in9)), j1 = _mm512_loadu_si512((const void*)(in9 + 64));
	__m512i k0 = _mm512_loadu_si512((const void*)(in10)), k1 = _mm512_loadu_si512((const void*)(in10 + 64));
	__m512i l0 = _mm512_loadu_si512((const void*)(in11)), l1 = _mm512_loadu_si512((const void*)(in11 + 64));

	__m512i mask_FF = _mm512_set1_epi16(0xFFFF);
	__m512i mask_FE = _mm512_set1_epi16(0xFEFE);
	__m512i mask_7F = _mm512_set1_epi16(0x7F7F);
	__m512i mask_80 = _mm512_set1_epi16(0x8080);
	__m512i mask_01 = _mm512_set1_epi16(0x0101);

	if (end > schedule_entries.entries.size())
		end = schedule_entries.entries.size();
	for (std::size_t map_idx = begin; map_idx < end; map_idx++)
	{
		_run_map_entry_x12(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1,i0,i1,j0,j1,k0,k1,l0,l1,
		                   schedule_entries.entries[map_idx],
		                   mask_FF, mask_FE, mask_7F, mask_80, mask_01);
	}

	_mm512_storeu_si512((void*)(out0), a0); _mm512_storeu_si512((void*)(out0 + 64), a1);
	_mm512_storeu_si512((void*)(out1), b0); _mm512_storeu_si512((void*)(out1 + 64), b1);
	_mm512_storeu_si512((void*)(out2), c0); _mm512_storeu_si512((void*)(out2 + 64), c1);
	_mm512_storeu_si512((void*)(out3), d0); _mm512_storeu_si512((void*)(out3 + 64), d1);
	_mm512_storeu_si512((void*)(out4), e0); _mm512_storeu_si512((void*)(out4 + 64), e1);
	_mm512_storeu_si512((void*)(out5), f0); _mm512_storeu_si512((void*)(out5 + 64), f1);
	_mm512_storeu_si512((void*)(out6), g0); _mm512_storeu_si512((void*)(out6 + 64), g1);
	_mm512_storeu_si512((void*)(out7), h0); _mm512_storeu_si512((void*)(out7 + 64), h1);
	_mm512_storeu_si512((void*)(out8), i0); _mm512_storeu_si512((void*)(out8 + 64), i1);
	_mm512_storeu_si512((void*)(out9), j0); _mm512_storeu_si512((void*)(out9 + 64), j1);
	_mm512_storeu_si512((void*)(out10), k0); _mm512_storeu_si512((void*)(out10 + 64), k1);
	_mm512_storeu_si512((void*)(out11), l0); _mm512_storeu_si512((void*)(out11 + 64), l1);
}

// 14-way: fourteen states (28 state ZMM) — past the spill-free ceiling. Built
// only to confirm the register-wall regression vs x12.
__forceinline void tm_avx512_r512s_8::_run_map_entry_x14(
	__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
	__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
	__m512i& i0, __m512i& i1, __m512i& j0, __m512i& j1, __m512i& k0, __m512i& k1, __m512i& l0, __m512i& l1,
	__m512i& m0, __m512i& m1, __m512i& n0, __m512i& n1,
	const key_schedule::key_schedule_entry& schedule_entry,
	__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01)
{
	const uint16 base = (schedule_entry.rng1 << 8) | schedule_entry.rng2;
	uint16 sa=base,sb=base,sc=base,sd=base,se=base,sf=base,sg=base,sh=base,si=base,sj=base,sk=base,sl=base,sm=base,sn=base;
	uint16 nibble_selector = schedule_entry.nibble_selector;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
	for (int it = 0; it < 16; it++)
	{
		unsigned char nibble = (nibble_selector >> 15) & 0x01;
		nibble_selector = nibble_selector << 1;

		const __m128i a0l=_mm512_castsi512_si128(a0),a1l=_mm512_castsi512_si128(a1);
		const __m128i b0l=_mm512_castsi512_si128(b0),b1l=_mm512_castsi512_si128(b1);
		const __m128i c0l=_mm512_castsi512_si128(c0),c1l=_mm512_castsi512_si128(c1);
		const __m128i d0l=_mm512_castsi512_si128(d0),d1l=_mm512_castsi512_si128(d1);
		const __m128i e0l=_mm512_castsi512_si128(e0),e1l=_mm512_castsi512_si128(e1);
		const __m128i f0l=_mm512_castsi512_si128(f0),f1l=_mm512_castsi512_si128(f1);
		const __m128i g0l=_mm512_castsi512_si128(g0),g1l=_mm512_castsi512_si128(g1);
		const __m128i h0l=_mm512_castsi512_si128(h0),h1l=_mm512_castsi512_si128(h1);
		const __m128i i0l=_mm512_castsi512_si128(i0),i1l=_mm512_castsi512_si128(i1);
		const __m128i j0l=_mm512_castsi512_si128(j0),j1l=_mm512_castsi512_si128(j1);
		const __m128i k0l=_mm512_castsi512_si128(k0),k1l=_mm512_castsi512_si128(k1);
		const __m128i l0l=_mm512_castsi512_si128(l0),l1l=_mm512_castsi512_si128(l1);
		const __m128i m0l=_mm512_castsi512_si128(m0),m1l=_mm512_castsi512_si128(m1);
		const __m128i n0l=_mm512_castsi512_si128(n0),n1l=_mm512_castsi512_si128(n1);
		unsigned char ba=0,bb=0,bc=0,bd=0,be=0,bf=0,bg=0,bh=0,bi=0,bj=0,bk=0,bl=0,bm=0,bn=0;
#define TM_X14_EXTRACT(out, r0, r1) \
		switch (it) { \
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
		TM_X14_EXTRACT(ba,a0l,a1l); TM_X14_EXTRACT(bb,b0l,b1l); TM_X14_EXTRACT(bc,c0l,c1l); TM_X14_EXTRACT(bd,d0l,d1l);
		TM_X14_EXTRACT(be,e0l,e1l); TM_X14_EXTRACT(bf,f0l,f1l); TM_X14_EXTRACT(bg,g0l,g1l); TM_X14_EXTRACT(bh,h0l,h1l);
		TM_X14_EXTRACT(bi,i0l,i1l); TM_X14_EXTRACT(bj,j0l,j1l); TM_X14_EXTRACT(bk,k0l,k1l); TM_X14_EXTRACT(bl,l0l,l1l);
		TM_X14_EXTRACT(bm,m0l,m1l); TM_X14_EXTRACT(bn,n0l,n1l);
#undef TM_X14_EXTRACT
		if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4;be>>=4;bf>>=4;bg>>=4;bh>>=4;bi>>=4;bj>>=4;bk>>=4;bl>>=4;bm>>=4;bn>>=4; }

		_alg_dispatch(a0,a1,sa,(ba>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(b0,b1,sb,(bb>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(c0,c1,sc,(bc>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(d0,d1,sd,(bd>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(e0,e1,se,(be>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(f0,f1,sf,(bf>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(g0,g1,sg,(bg>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(h0,h1,sh,(bh>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(i0,i1,si,(bi>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(j0,j1,sj,(bj>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(k0,k1,sk,(bk>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(l0,l1,sl,(bl>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(m0,m1,sm,(bm>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
		_alg_dispatch(n0,n1,sn,(bn>>1)&0x07, mask_FF,mask_FE,mask_7F,mask_80,mask_01);
	}
}

void tm_avx512_r512s_8::run_maps_range_x14(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
	const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3, const uint8* in4, const uint8* in5, const uint8* in6,
	const uint8* in7, const uint8* in8, const uint8* in9, const uint8* in10, const uint8* in11, const uint8* in12, const uint8* in13,
	uint8* out0, uint8* out1, uint8* out2, uint8* out3, uint8* out4, uint8* out5, uint8* out6,
	uint8* out7, uint8* out8, uint8* out9, uint8* out10, uint8* out11, uint8* out12, uint8* out13)
{
	__m512i a0=_mm512_loadu_si512((const void*)in0),a1=_mm512_loadu_si512((const void*)(in0+64));
	__m512i b0=_mm512_loadu_si512((const void*)in1),b1=_mm512_loadu_si512((const void*)(in1+64));
	__m512i c0=_mm512_loadu_si512((const void*)in2),c1=_mm512_loadu_si512((const void*)(in2+64));
	__m512i d0=_mm512_loadu_si512((const void*)in3),d1=_mm512_loadu_si512((const void*)(in3+64));
	__m512i e0=_mm512_loadu_si512((const void*)in4),e1=_mm512_loadu_si512((const void*)(in4+64));
	__m512i f0=_mm512_loadu_si512((const void*)in5),f1=_mm512_loadu_si512((const void*)(in5+64));
	__m512i g0=_mm512_loadu_si512((const void*)in6),g1=_mm512_loadu_si512((const void*)(in6+64));
	__m512i h0=_mm512_loadu_si512((const void*)in7),h1=_mm512_loadu_si512((const void*)(in7+64));
	__m512i i0=_mm512_loadu_si512((const void*)in8),i1=_mm512_loadu_si512((const void*)(in8+64));
	__m512i j0=_mm512_loadu_si512((const void*)in9),j1=_mm512_loadu_si512((const void*)(in9+64));
	__m512i k0=_mm512_loadu_si512((const void*)in10),k1=_mm512_loadu_si512((const void*)(in10+64));
	__m512i l0=_mm512_loadu_si512((const void*)in11),l1=_mm512_loadu_si512((const void*)(in11+64));
	__m512i m0=_mm512_loadu_si512((const void*)in12),m1=_mm512_loadu_si512((const void*)(in12+64));
	__m512i n0=_mm512_loadu_si512((const void*)in13),n1=_mm512_loadu_si512((const void*)(in13+64));

	__m512i mask_FF = _mm512_set1_epi16(0xFFFF);
	__m512i mask_FE = _mm512_set1_epi16(0xFEFE);
	__m512i mask_7F = _mm512_set1_epi16(0x7F7F);
	__m512i mask_80 = _mm512_set1_epi16(0x8080);
	__m512i mask_01 = _mm512_set1_epi16(0x0101);

	if (end > schedule_entries.entries.size())
		end = schedule_entries.entries.size();
	for (std::size_t map_idx = begin; map_idx < end; map_idx++)
	{
		_run_map_entry_x14(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1,i0,i1,j0,j1,k0,k1,l0,l1,m0,m1,n0,n1,
		                   schedule_entries.entries[map_idx], mask_FF, mask_FE, mask_7F, mask_80, mask_01);
	}

	_mm512_storeu_si512((void*)out0,a0); _mm512_storeu_si512((void*)(out0+64),a1);
	_mm512_storeu_si512((void*)out1,b0); _mm512_storeu_si512((void*)(out1+64),b1);
	_mm512_storeu_si512((void*)out2,c0); _mm512_storeu_si512((void*)(out2+64),c1);
	_mm512_storeu_si512((void*)out3,d0); _mm512_storeu_si512((void*)(out3+64),d1);
	_mm512_storeu_si512((void*)out4,e0); _mm512_storeu_si512((void*)(out4+64),e1);
	_mm512_storeu_si512((void*)out5,f0); _mm512_storeu_si512((void*)(out5+64),f1);
	_mm512_storeu_si512((void*)out6,g0); _mm512_storeu_si512((void*)(out6+64),g1);
	_mm512_storeu_si512((void*)out7,h0); _mm512_storeu_si512((void*)(out7+64),h1);
	_mm512_storeu_si512((void*)out8,i0); _mm512_storeu_si512((void*)(out8+64),i1);
	_mm512_storeu_si512((void*)out9,j0); _mm512_storeu_si512((void*)(out9+64),j1);
	_mm512_storeu_si512((void*)out10,k0); _mm512_storeu_si512((void*)(out10+64),k1);
	_mm512_storeu_si512((void*)out11,l0); _mm512_storeu_si512((void*)(out11+64),l1);
	_mm512_storeu_si512((void*)out12,m0); _mm512_storeu_si512((void*)(out12+64),m1);
	_mm512_storeu_si512((void*)out13,n0); _mm512_storeu_si512((void*)(out13+64),n1);
}

// ---------------------------------------------------------------------------
// Native screen (decrypt + masked checksum) on the shuffled 2-ZMM layout. These
// operate on the member working_code_data (the screen path loads a survivor via
// load_state_raw, then calls decrypt -> checksum -> fetch). The masked checksum
// is a byte-sum (vpsadbw), order-independent, so it is correct on the shuffled
// state given a shuffled mask. Mirrors tm_avx2_r256s_8's screen on the 512 layout.
// ---------------------------------------------------------------------------
void tm_avx512_r512s_8::decrypt_carnival_world()
{
	__m512i w0 = _mm512_load_si512((__m512i*)(working_code_data));
	__m512i w1 = _mm512_load_si512((__m512i*)(working_code_data + 64));
	w0 = _mm512_xor_si512(w0, _mm512_load_si512((const __m512i*)(carnival_world_data_shuffled)));
	w1 = _mm512_xor_si512(w1, _mm512_load_si512((const __m512i*)(carnival_world_data_shuffled + 64)));
	_mm512_store_si512((__m512i*)(working_code_data), w0);
	_mm512_store_si512((__m512i*)(working_code_data + 64), w1);
}

void tm_avx512_r512s_8::decrypt_other_world()
{
	__m512i w0 = _mm512_load_si512((__m512i*)(working_code_data));
	__m512i w1 = _mm512_load_si512((__m512i*)(working_code_data + 64));
	w0 = _mm512_xor_si512(w0, _mm512_load_si512((const __m512i*)(other_world_data_shuffled)));
	w1 = _mm512_xor_si512(w1, _mm512_load_si512((const __m512i*)(other_world_data_shuffled + 64)));
	_mm512_store_si512((__m512i*)(working_code_data), w0);
	_mm512_store_si512((__m512i*)(working_code_data + 64), w1);
}

uint16 tm_avx512_r512s_8::masked_checksum(const uint8* mask)
{
	__m512i w0 = _mm512_load_si512((__m512i*)(working_code_data));
	__m512i w1 = _mm512_load_si512((__m512i*)(working_code_data + 64));
	__m512i m0 = _mm512_load_si512((const __m512i*)(mask));
	__m512i m1 = _mm512_load_si512((const __m512i*)(mask + 64));
	__m512i z = _mm512_setzero_si512();
	__m512i s = _mm512_add_epi64(_mm512_sad_epu8(_mm512_and_si512(w0, m0), z),
	                             _mm512_sad_epu8(_mm512_and_si512(w1, m1), z));
	// Horizontal add of the 8 x 64-bit SAD partials in one intrinsic (micro500's
	// form; replaces the store + scalar lane-sum).
	return (uint16)_mm512_reduce_add_epi64(s);
}

uint16 tm_avx512_r512s_8::calculate_carnival_world_checksum()
{
	return masked_checksum(carnival_world_checksum_mask_shuffled);
}

uint16 tm_avx512_r512s_8::calculate_other_world_checksum()
{
	return masked_checksum(other_world_checksum_mask_shuffled);
}

uint16 tm_avx512_r512s_8::fetch_checksum_value(uint8 code_length)
{
	// code_length passed as (CODE_LENGTH - 2); checksum bytes are canonical
	// indices 127-code_length (lo) and 127-(code_length+1) (hi), located at
	// shuffle(idx) in the 512 layout. working_code_data holds the decrypted state.
	uint8 lo = working_code_data[shuffle(127 - code_length)];
	uint8 hi = working_code_data[shuffle(127 - (code_length + 1))];
	return (uint16)((hi << 8) | lo);
}

uint16 tm_avx512_r512s_8::fetch_carnival_world_checksum_value()
{
	return fetch_checksum_value(CARNIVAL_WORLD_CODE_LENGTH - 2);
}

uint16 tm_avx512_r512s_8::fetch_other_world_checksum_value()
{
	return fetch_checksum_value(OTHER_WORLD_CODE_LENGTH - 2);
}

bool tm_avx512_r512s_8::initialized = false;