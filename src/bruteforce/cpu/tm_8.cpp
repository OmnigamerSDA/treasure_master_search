#include <stdio.h>
#include <iostream>
#include "data_sizes.h"
#include "tm_8.h"
#include "key_schedule.h"

tm_8::tm_8(RNG * rng_obj) : TM_base(rng_obj)
{
	initialize();
}

__forceinline void tm_8::initialize()
{
	if (!initialized)
	{
		rng->generate_expansion_values_8();

		rng->generate_seed_forward_1();
		rng->generate_seed_forward_128();

		rng->generate_regular_rng_values_8();

		rng->generate_alg0_values_8();
		rng->generate_alg2_values_8_8();
		// alg4 uses regular_rng_values_8 via subtraction; no separate table needed
		rng->generate_alg5_values_8_8();
		rng->generate_alg6_values_8();

		initialized = true;
	}
	obj_name = "tm_8";
}

void tm_8::expand(uint32 key, uint32 data)
{
	for (int i = 0; i < 128; i += 8)
	{
		working_code_data[i] = (key >> 24) & 0xFF;
		working_code_data[i + 1] = (key >> 16) & 0xFF;
		working_code_data[i + 2] = (key >> 8) & 0xFF;
		working_code_data[i + 3] = key & 0xFF;

		working_code_data[i + 4] = (data >> 24) & 0xFF;
		working_code_data[i + 5] = (data >> 16) & 0xFF;
		working_code_data[i + 6] = (data >> 8) & 0xFF;
		working_code_data[i + 7] = data & 0xFF;
	}

	uint16 rng_seed = (key >> 16) & 0xFFFF;
	for (int i = 0; i < 128; i++)
	{
		working_code_data[i] += rng->expansion_values_8[rng_seed * 128 + i];
	}
}

void tm_8::load_data(uint8* new_data)
{
	for (int i = 0; i < 128; i++)
	{
		working_code_data[i] = new_data[i];
	}
}

void tm_8::fetch_data(uint8* new_data)
{
	for (int i = 0; i < 128; i++)
	{
		new_data[i] = working_code_data[i];
	}
}

void tm_8::run_alg(int algorithm_id, uint16 * rng_seed, int iterations)
{
	if (algorithm_id == 0)
	{
		for (int i = 0; i < iterations; i++)
		{
			alg_0(*rng_seed);
			*rng_seed = rng->seed_forward_128[*rng_seed];
		}
	}
	else if (algorithm_id == 1)
	{
		for (int i = 0; i < iterations; i++)
		{
			alg_1(*rng_seed);
			*rng_seed = rng->seed_forward_128[*rng_seed];
		}
	}
	else if (algorithm_id == 2)
	{
		for (int i = 0; i < iterations; i++)
		{
			alg_2(*rng_seed);
			*rng_seed = rng->seed_forward_1[*rng_seed];
		}
	}
	else if (algorithm_id == 3)
	{
		for (int i = 0; i < iterations; i++)
		{
			alg_3(*rng_seed);
			*rng_seed = rng->seed_forward_128[*rng_seed];
		}
	}
	else if (algorithm_id == 4)
	{
		for (int i = 0; i < iterations; i++)
		{
			alg_4(*rng_seed);
			*rng_seed = rng->seed_forward_128[*rng_seed];
		}
	}
	else if (algorithm_id == 5)
	{
		for (int i = 0; i < iterations; i++)
		{
			alg_5(*rng_seed);
			*rng_seed = rng->seed_forward_1[*rng_seed];
		}
	}
	else if (algorithm_id == 6)
	{
		for (int i = 0; i < iterations; i++)
		{
			alg_6(*rng_seed);
			*rng_seed = rng->seed_forward_128[*rng_seed];
		}
	}
	else if (algorithm_id == 7)
	{
		for (int i = 0; i < iterations; i++)
		{
			alg_7();
		}
	}
}

__forceinline void tm_8::alg_0(const uint16 rng_seed)
{
	const uint8* row = rng->alg0_values_8 + rng_seed * 128;
	for (int i = 0; i < 128; i++)
		working_code_data[i] = (working_code_data[i] << 1) | row[i];
}

__forceinline void tm_8::alg_1(const uint16 rng_seed)
{
	add_alg(rng->regular_rng_values_8, rng_seed);
}

__forceinline void tm_8::alg_2( const uint16 rng_seed)
{
	uint8 carry = rng->alg2_values_8_8[rng_seed];
	for (int i = 127; i >= 0; i -= 2)
	{
		uint8 next_carry = working_code_data[i - 1] & 0x01;

		working_code_data[i - 1] = (working_code_data[i - 1] >> 1) | (working_code_data[i] & 0x80);
		working_code_data[i] = (working_code_data[i] << 1) | (carry & 0x01);

		carry = next_carry;
	}
}

__forceinline void tm_8::alg_3(const uint16 rng_seed)
{
	xor_alg(working_code_data, rng->regular_rng_values_8 + rng_seed * 128);
}

__forceinline void tm_8::alg_4(const uint16 rng_seed)
{
	// alg4_values_8[seed*128+i] == -(regular_rng_values_8[seed*128+i]) mod 256,
	// so state[i] += alg4 == state[i] -= regular.  Reuse the table already in cache.
	const uint8* row = rng->regular_rng_values_8 + rng_seed * 128;
	for (int i = 0; i < 128; i++)
		working_code_data[i] -= row[i];
}

__forceinline void tm_8::alg_5(const uint16 rng_seed)
{
	uint8 carry = rng->alg5_values_8_8[rng_seed];

	for (int i = 127; i >= 0; i -= 2)
	{
		uint8 next_carry = working_code_data[i - 1] & 0x80;

		working_code_data[i - 1] = (working_code_data[i - 1] << 1) | (working_code_data[i] & 0x01);
		working_code_data[i] = (working_code_data[i] >> 1) | carry;

		carry = next_carry;
	}
}

__forceinline void tm_8::alg_6(const uint16 rng_seed)
{
	const uint8* row = rng->alg6_values_8 + rng_seed * 128;
	for (int i = 0; i < 128; i++)
		working_code_data[i] = (working_code_data[i] >> 1) | row[i];
}

__forceinline void tm_8::alg_7()
{
	for (int i = 0; i < 128; i++)
	{
		working_code_data[i] = working_code_data[i] ^ 0xFF;
	}
}

__forceinline void tm_8::add_alg(uint8* addition_values, const uint16 rng_seed)
{
	for (int i = 0; i < 128; i++)
	{
		working_code_data[i] = working_code_data[i] + addition_values[rng_seed * 128 + i];
	}
}

__forceinline void tm_8::xor_alg(uint8* working_data, uint8* xor_values)
{
	for (int i = 0; i < 128; i++)
	{
		working_data[i] = working_data[i] ^ xor_values[i];
	}
}

void tm_8::run_one_map(const key_schedule::key_schedule_entry& schedule_entry)
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

		// If the flag is a 1, get the high nibble of the current byte
		// Otherwise use the low nibble
		unsigned char current_byte = (uint8)working_code_data[i];

		if (nibble == 1)
		{
			current_byte = current_byte >> 4;
		}

		// Mask off only 3 bits
		unsigned char alg_id = (current_byte >> 1) & 0x07;

		run_alg(alg_id, &rng_seed, 1);
	}
}

void tm_8::run_all_maps(const key_schedule& schedule_entries)
{
	for (std::vector<key_schedule::key_schedule_entry>::const_iterator it = schedule_entries.entries.begin(); it != schedule_entries.entries.end(); it++)
	{
		run_one_map(*it);
	}
}

void tm_8::run_bruteforce_data(uint32 key, uint32 start_data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size)
{
	uint32 output_pos = 0;
	for (uint32 i = 0; i < amount_to_run; i++)
	{
		if ((result_max_size - output_pos) < 5)
		{
			*result_size = result_max_size;
			return;
		}
		uint32 data = start_data + i;

		expand(key, data);
		run_all_maps(schedule_entries);

		uint8 decrypted_data[128];
		for (int i = 0; i < 128; i++)
		{
			decrypted_data[i] = working_code_data[i];
		}
		_decrypt_carnival_world(decrypted_data);

		if (check_carnival_world_checksum(decrypted_data))
		{
			*((uint32*)(&result_data[output_pos])) = i;

			result_data[output_pos + 4] = check_machine_code(decrypted_data, CARNIVAL_WORLD);
			output_pos += 5;
		}
		else
		{
			for (int i = 0; i < 128; i++)
			{
				decrypted_data[i] = working_code_data[i];
			}
			_decrypt_other_world(decrypted_data);

			if (check_other_world_checksum(decrypted_data))
			{
				*((uint32*)(&result_data[output_pos])) = i;

				result_data[output_pos + 4] = check_machine_code(decrypted_data, OTHER_WORLD);
				output_pos += 5;
			}
		}

		report_progress((float)(i + 1) / amount_to_run);
	}

	*result_size = output_pos;
}
__forceinline void tm_8::decrypt_carnival_world()
{
	_decrypt_carnival_world(working_code_data);
}

__forceinline void tm_8::decrypt_other_world()
{
	_decrypt_other_world(working_code_data);
}

__forceinline void tm_8::_decrypt_carnival_world(uint8* working_data)
{
	xor_alg(working_data, carnival_world_data);
}

__forceinline void tm_8::_decrypt_other_world(uint8* working_data)
{
	xor_alg(working_data, other_world_data);
}

__forceinline uint16 tm_8::calculate_masked_checksum(uint8* working_data, uint8* mask)
{
	uint16 sum = 0;
	for (int i = 0; i < 128; i++)
	{
		sum += working_data[i] & mask[i];
	}
	return sum;
}

__forceinline uint16 tm_8::_calculate_carnival_world_checksum(uint8* working_data)
{
	return calculate_masked_checksum(working_data, carnival_world_checksum_mask);
}

__forceinline uint16 tm_8::calculate_carnival_world_checksum()
{
	return _calculate_carnival_world_checksum(working_code_data);
}

__forceinline uint16 tm_8::_calculate_other_world_checksum(uint8* working_data)
{
	return calculate_masked_checksum(working_data, other_world_checksum_mask);
}

__forceinline uint16 tm_8::calculate_other_world_checksum()
{
	return _calculate_other_world_checksum(working_code_data);
}

__forceinline uint16 tm_8::fetch_checksum_value(uint8* working_data, int code_length)
{
	return working_data[reverse_offset(code_length - 1)] << 8 | working_data[reverse_offset(code_length - 2)];
}

__forceinline uint16 tm_8::_fetch_carnival_world_checksum_value(uint8* working_data)
{
	return fetch_checksum_value(working_data, CARNIVAL_WORLD_CODE_LENGTH);
}

__forceinline uint16 tm_8::fetch_carnival_world_checksum_value()
{
	return _fetch_carnival_world_checksum_value(working_code_data);
}

__forceinline uint16 tm_8::_fetch_other_world_checksum_value(uint8* working_data)
{
	return fetch_checksum_value(working_data, OTHER_WORLD_CODE_LENGTH);
}

__forceinline uint16 tm_8::fetch_other_world_checksum_value()
{
	return _fetch_other_world_checksum_value(working_code_data);
}

__forceinline bool tm_8::check_carnival_world_checksum(uint8* working_data)
{
	return _calculate_carnival_world_checksum(working_data) == _fetch_carnival_world_checksum_value(working_data);
}

__forceinline bool tm_8::check_other_world_checksum(uint8* working_data)
{
	return _calculate_other_world_checksum(working_data) == _fetch_other_world_checksum_value(working_data);
}


bool tm_8::initialized = false;

// ─── N-way interleaved implementation ────────────────────────────────────────
// Processes NWAY_BATCH candidates in lockstep through run_all_maps, issuing
// multiple concurrent RNG-table loads per step to hide L3 latency.

namespace
{
    // Per-state algorithm kernels.  Mirrors tm_8::alg_N but operates on an
    // externally-provided state buffer so we can keep BATCH states live in L1.

    static __forceinline void n_alg_0(uint16 seed, uint8* s, RNG* r)
    {
        const uint8* row = r->alg0_values_8 + seed * 128;
        for (int i = 0; i < 128; i++)
            s[i] = (s[i] << 1) | row[i];
    }

    static __forceinline void n_alg_1(uint16 seed, uint8* s, RNG* r)
    {
        const uint8* row = r->regular_rng_values_8 + seed * 128;
        for (int i = 0; i < 128; i++) s[i] += row[i];
    }

    static __forceinline void n_alg_2(uint16 seed, uint8* s, RNG* r)
    {
        uint8 carry = r->alg2_values_8_8[seed];
        for (int i = 127; i >= 0; i -= 2)
        {
            uint8 nc = s[i - 1] & 0x01;
            s[i - 1] = (s[i - 1] >> 1) | (s[i] & 0x80);
            s[i]     = (s[i] << 1) | (carry & 0x01);
            carry = nc;
        }
    }

    static __forceinline void n_alg_3(uint16 seed, uint8* s, RNG* r)
    {
        const uint8* row = r->regular_rng_values_8 + seed * 128;
        for (int i = 0; i < 128; i++) s[i] ^= row[i];
    }

    static __forceinline void n_alg_4(uint16 seed, uint8* s, RNG* r)
    {
        const uint8* row = r->regular_rng_values_8 + seed * 128;
        for (int i = 0; i < 128; i++) s[i] -= row[i];
    }

    static __forceinline void n_alg_5(uint16 seed, uint8* s, RNG* r)
    {
        uint8 carry = r->alg5_values_8_8[seed];
        for (int i = 127; i >= 0; i -= 2)
        {
            uint8 nc = s[i - 1] & 0x80;
            s[i - 1] = (s[i - 1] << 1) | (s[i] & 0x01);
            s[i]     = (s[i] >> 1) | carry;
            carry = nc;
        }
    }

    static __forceinline void n_alg_6(uint16 seed, uint8* s, RNG* r)
    {
        const uint8* row = r->alg6_values_8 + seed * 128;
        for (int i = 0; i < 128; i++)
            s[i] = (s[i] >> 1) | row[i];
    }

    static __forceinline void n_alg_7(uint8* s)
    {
        for (int i = 0; i < 128; i++) s[i] ^= 0xFF;
    }

    static __forceinline void n_run_alg(int id, uint16* seed, uint8* s, RNG* r)
    {
        switch (id)
        {
        case 0: n_alg_0(*seed, s, r); *seed = r->seed_forward_128[*seed]; break;
        case 1: n_alg_1(*seed, s, r); *seed = r->seed_forward_128[*seed]; break;
        case 2: n_alg_2(*seed, s, r); *seed = r->seed_forward_1[*seed];   break;
        case 3: n_alg_3(*seed, s, r); *seed = r->seed_forward_128[*seed]; break;
        case 4: n_alg_4(*seed, s, r); *seed = r->seed_forward_128[*seed]; break;
        case 5: n_alg_5(*seed, s, r); *seed = r->seed_forward_1[*seed];   break;
        case 6: n_alg_6(*seed, s, r); *seed = r->seed_forward_128[*seed]; break;
        case 7: n_alg_7(s); break;
        }
    }

    // Advance NWAY_BATCH candidates through one map entry simultaneously.
    // All candidates share the same initial seed for this entry (from schedule);
    // they diverge only if they choose algorithms with different seed-step sizes.
    static void n_run_one_map(const key_schedule::key_schedule_entry& entry,
                               uint8 (*states)[128], RNG* r)
    {
        constexpr int B = tm_8::NWAY_BATCH;
        uint16 seeds[B];
        const uint16 base = static_cast<uint16>((entry.rng1 << 8) | entry.rng2);
        for (int b = 0; b < B; b++) seeds[b] = base;
        uint16 nsel = entry.nibble_selector;

        for (int i = 0; i < 16; i++)
        {
            const uint8 hi = (nsel >> 15) & 1;
            nsel <<= 1;
            // Issue all BATCH table loads back-to-back; OOO engine overlaps latencies.
            for (int b = 0; b < B; b++)
            {
                uint8 cur = states[b][i];
                if (hi) cur >>= 4;
                n_run_alg((cur >> 1) & 7, &seeds[b], states[b], r);
            }
        }
    }
} // namespace

// expand key+data directly into an external state buffer.
void tm_8::expand_into(uint32 key, uint32 data, uint8* state)
{
    for (int i = 0; i < 128; i += 8)
    {
        state[i]     = (key  >> 24) & 0xFF;
        state[i + 1] = (key  >> 16) & 0xFF;
        state[i + 2] = (key  >>  8) & 0xFF;
        state[i + 3] =  key         & 0xFF;
        state[i + 4] = (data >> 24) & 0xFF;
        state[i + 5] = (data >> 16) & 0xFF;
        state[i + 6] = (data >>  8) & 0xFF;
        state[i + 7] =  data        & 0xFF;
    }
    uint16 rng_seed = (key >> 16) & 0xFFFF;
    for (int i = 0; i < 128; i++)
        state[i] += rng->expansion_values_8[rng_seed * 128 + i];
}

void tm_8::run_bruteforce_data_nway(
    uint32 key, uint32 start_data,
    const key_schedule& schedule_entries, uint32 amount_to_run,
    void(*report_progress)(double),
    uint8* result_data, uint32 result_max_size, uint32* result_size)
{
    constexpr int B = NWAY_BATCH;
    // All BATCH states live in L1 (B*128 = 1 KB for B=8).
    uint8 states[B][128];
    uint32 output_pos = 0;
    uint32 i = 0;

    for (; i + B <= amount_to_run; i += B)
    {
        if ((result_max_size - output_pos) < 5u * B)
        {
            *result_size = result_max_size;
            return;
        }

        for (int b = 0; b < B; b++)
            expand_into(key, start_data + i + b, states[b]);

        for (const auto& entry : schedule_entries.entries)
            n_run_one_map(entry, states, rng);

        for (int b = 0; b < B; b++)
        {
            uint8 dec[128];
            for (int k = 0; k < 128; k++) dec[k] = states[b][k];
            _decrypt_carnival_world(dec);
            if (check_carnival_world_checksum(dec))
            {
                *reinterpret_cast<uint32*>(&result_data[output_pos]) = i + b;
                result_data[output_pos + 4] = check_machine_code(dec, CARNIVAL_WORLD);
                output_pos += 5;
            }
            else
            {
                for (int k = 0; k < 128; k++) dec[k] = states[b][k];
                _decrypt_other_world(dec);
                if (check_other_world_checksum(dec))
                {
                    *reinterpret_cast<uint32*>(&result_data[output_pos]) = i + b;
                    result_data[output_pos + 4] = check_machine_code(dec, OTHER_WORLD);
                    output_pos += 5;
                }
            }
        }
        report_progress(static_cast<double>(i + B) / amount_to_run);
    }

    // Remainder (count % B) handled by the original single-candidate path.
    for (; i < amount_to_run; i++)
    {
        if ((result_max_size - output_pos) < 5u)
        {
            *result_size = result_max_size;
            return;
        }
        expand(key, start_data + i);
        run_all_maps(schedule_entries);

        uint8 dec[128];
        for (int k = 0; k < 128; k++) dec[k] = working_code_data[k];
        _decrypt_carnival_world(dec);
        if (check_carnival_world_checksum(dec))
        {
            *reinterpret_cast<uint32*>(&result_data[output_pos]) = i;
            result_data[output_pos + 4] = check_machine_code(dec, CARNIVAL_WORLD);
            output_pos += 5;
        }
        else
        {
            for (int k = 0; k < 128; k++) dec[k] = working_code_data[k];
            _decrypt_other_world(dec);
            if (check_other_world_checksum(dec))
            {
                *reinterpret_cast<uint32*>(&result_data[output_pos]) = i;
                result_data[output_pos + 4] = check_machine_code(dec, OTHER_WORLD);
                output_pos += 5;
            }
        }
        report_progress(static_cast<double>(i + 1) / amount_to_run);
    }

    *result_size = output_pos;
}