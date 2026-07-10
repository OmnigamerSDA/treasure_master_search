#ifndef MAP1_CERTIFIER_H
#define MAP1_CERTIFIER_H

#include "rng_obj.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace tm_map1_certifier
{
inline std::uint8_t smear_up(std::uint8_t mask)
{
	if (mask == 0u) return 0u;
	std::uint8_t out = 0u;
	for (int bit = 0; bit < 8; bit++)
	{
		if ((mask & (1u << bit)) == 0u) continue;
		for (int j = bit; j < 8; j++) out = static_cast<std::uint8_t>(out | (1u << j));
		break;
	}
	return out;
}

inline void apply_mask(int alg_id, std::uint8_t* mask)
{
	if (alg_id == 0)
	{
		for (int i = 0; i < 128; i++) mask[i] = static_cast<std::uint8_t>(mask[i] << 1);
	}
	else if (alg_id == 6)
	{
		for (int i = 0; i < 128; i++) mask[i] = static_cast<std::uint8_t>(mask[i] >> 1);
	}
	else if (alg_id == 1 || alg_id == 4)
	{
		for (int i = 0; i < 128; i++) mask[i] = smear_up(mask[i]);
	}
	else if (alg_id == 3 || alg_id == 7)
	{
		return;
	}
	else if (alg_id == 2)
	{
		std::uint8_t carry = 0u;
		for (int i = 127; i >= 0; i -= 2)
		{
			const std::uint8_t next_carry = static_cast<std::uint8_t>(mask[i - 1] & 0x01u);
			mask[i - 1] = static_cast<std::uint8_t>((mask[i - 1] >> 1) | (mask[i] & 0x80u));
			mask[i] = static_cast<std::uint8_t>((mask[i] << 1) | (carry & 0x01u));
			carry = next_carry;
		}
	}
	else if (alg_id == 5)
	{
		std::uint8_t carry = 0u;
		for (int i = 127; i >= 0; i -= 2)
		{
			const std::uint8_t next_carry = static_cast<std::uint8_t>(mask[i - 1] & 0x80u);
			mask[i - 1] = static_cast<std::uint8_t>((mask[i - 1] << 1) | (mask[i] & 0x01u));
			mask[i] = static_cast<std::uint8_t>((mask[i] >> 1) | carry);
			carry = next_carry;
		}
	}
}

inline bool empty_mask(const std::uint8_t* mask)
{
	for (int i = 0; i < 128; i++)
		if (mask[i] != 0u) return false;
	return true;
}

inline std::uint8_t selector_bits(bool high_nibble)
{
	return high_nibble ? static_cast<std::uint8_t>(0xE0u) : static_cast<std::uint8_t>(0x0Eu);
}

inline void run_alg_for_trace(std::uint8_t alg_id, std::uint16_t* rng_seed, std::uint8_t* state)
{
	if (alg_id == 0)
	{
		const std::uint8_t* table = RNG::alg0_values_8 + static_cast<std::uint32_t>(*rng_seed) * 128u;
		for (int i = 0; i < 128; i++) state[i] = static_cast<std::uint8_t>((state[i] << 1) | table[i]);
		*rng_seed = RNG::seed_forward_128[*rng_seed];
	}
	else if (alg_id == 1)
	{
		const std::uint8_t* table = RNG::regular_rng_values_8 + static_cast<std::uint32_t>(*rng_seed) * 128u;
		for (int i = 0; i < 128; i++) state[i] = static_cast<std::uint8_t>(state[i] + table[i]);
		*rng_seed = RNG::seed_forward_128[*rng_seed];
	}
	else if (alg_id == 2)
	{
		std::uint8_t carry = static_cast<std::uint8_t>((RNG::alg2_values_32_8[*rng_seed] >> 24) & 0x01u);
		for (int i = 127; i >= 0; i -= 2)
		{
			const std::uint8_t next_carry = static_cast<std::uint8_t>(state[i - 1] & 0x01u);
			state[i - 1] = static_cast<std::uint8_t>((state[i - 1] >> 1) | (state[i] & 0x80u));
			state[i] = static_cast<std::uint8_t>((state[i] << 1) | (carry & 0x01u));
			carry = next_carry;
		}
		*rng_seed = RNG::seed_forward_1[*rng_seed];
	}
	else if (alg_id == 3)
	{
		const std::uint8_t* table = RNG::regular_rng_values_8 + static_cast<std::uint32_t>(*rng_seed) * 128u;
		for (int i = 0; i < 128; i++) state[i] ^= table[i];
		*rng_seed = RNG::seed_forward_128[*rng_seed];
	}
	else if (alg_id == 4)
	{
		const std::uint8_t* table = RNG::regular_rng_values_8 + static_cast<std::uint32_t>(*rng_seed) * 128u;
		for (int i = 0; i < 128; i++) state[i] = static_cast<std::uint8_t>(state[i] - table[i]);
		*rng_seed = RNG::seed_forward_128[*rng_seed];
	}
	else if (alg_id == 5)
	{
		std::uint8_t carry = static_cast<std::uint8_t>((RNG::alg5_values_32_8[*rng_seed] >> 24) & 0x80u);
		for (int i = 127; i >= 0; i -= 2)
		{
			const std::uint8_t next_carry = static_cast<std::uint8_t>(state[i - 1] & 0x80u);
			state[i - 1] = static_cast<std::uint8_t>((state[i - 1] << 1) | (state[i] & 0x01u));
			state[i] = static_cast<std::uint8_t>((state[i] >> 1) | carry);
			carry = next_carry;
		}
		*rng_seed = RNG::seed_forward_1[*rng_seed];
	}
	else if (alg_id == 6)
	{
		const std::uint8_t* table = RNG::alg6_values_8 + static_cast<std::uint32_t>(*rng_seed) * 128u;
		for (int i = 0; i < 128; i++) state[i] = static_cast<std::uint8_t>((state[i] >> 1) | table[i]);
		*rng_seed = RNG::seed_forward_128[*rng_seed];
	}
	else
	{
		for (int i = 0; i < 128; i++) state[i] ^= 0xFFu;
	}
}

inline void ensure_rng_tables()
{
	static const bool initialized = []() {
		RNG rng;
		rng.generate_regular_rng_values_8();
		rng.generate_alg0_values_8();
		rng.generate_alg6_values_8();
		rng.generate_seed_forward_1();
		rng.generate_seed_forward_128();
		rng.generate_alg2_values_32_8();
		rng.generate_alg5_values_32_8();
		rng.generate_expansion_values_8();
		return true;
	}();
	(void)initialized;
}

inline std::uint32_t certified_shed_mask_from_schedule_blob(
	std::uint32_t key,
	const std::vector<std::uint8_t>& schedule_blob)
{
	ensure_rng_tables();
	if (schedule_blob.size() < 4u || (schedule_blob.size() % 4u) != 0u)
		throw std::runtime_error("MAP1 certifier requires a non-empty 4-byte schedule blob");

	std::uint8_t state[128];
	const std::uint16_t expansion_seed = static_cast<std::uint16_t>(key >> 16);
	for (int i = 0; i < 128; i += 8)
	{
		state[i + 0] = static_cast<std::uint8_t>((key >> 24) & 0xFFu);
		state[i + 1] = static_cast<std::uint8_t>((key >> 16) & 0xFFu);
		state[i + 2] = static_cast<std::uint8_t>((key >>  8) & 0xFFu);
		state[i + 3] = static_cast<std::uint8_t>( key        & 0xFFu);
		state[i + 4] = 0u;
		state[i + 5] = 0u;
		state[i + 6] = 0u;
		state[i + 7] = 0u;
	}
	const std::uint8_t* exp_table = RNG::expansion_values_8 + static_cast<std::uint32_t>(expansion_seed) * 128u;
	for (int i = 0; i < 128; i++) state[i] = static_cast<std::uint8_t>(state[i] + exp_table[i]);

	std::uint16_t rng_seed = static_cast<std::uint16_t>(
		(static_cast<std::uint16_t>(schedule_blob[0]) << 8) |
		 static_cast<std::uint16_t>(schedule_blob[1]));
	std::uint16_t nibble_selector = static_cast<std::uint16_t>(
		(static_cast<std::uint16_t>(schedule_blob[2]) << 8) |
		 static_cast<std::uint16_t>(schedule_blob[3]));

	int op[16] = {};
	int nib[16] = {};
	for (int i = 0; i < 16; i++)
	{
		const std::uint8_t n = static_cast<std::uint8_t>((nibble_selector >> 15) & 0x01u);
		nibble_selector = static_cast<std::uint16_t>(nibble_selector << 1);
		std::uint8_t cur_byte = state[i];
		if (n != 0u) cur_byte = static_cast<std::uint8_t>(cur_byte >> 4);
		const std::uint8_t alg_id = static_cast<std::uint8_t>((cur_byte >> 1) & 0x07u);
		op[i] = alg_id;
		nib[i] = n;
		run_alg_for_trace(alg_id, &rng_seed, state);
	}

	std::uint8_t taint[128];
	for (int i = 0; i < 128; i++) taint[i] = ((i & 7) >= 4) ? 0xFFu : 0x00u;
	int prefix = 16;
	for (int i = 0; i < 16; i++)
	{
		if ((taint[i] & selector_bits(nib[i] != 0)) != 0u)
		{
			prefix = i;
			break;
		}
		apply_mask(op[i], taint);
	}

	std::uint32_t shed_mask = 0u;
	for (int db = 0; db < 32; db++)
	{
		const int off = 7 - (db / 8);
		const int bit = db % 8;
		std::uint8_t mask[128];
		std::memset(mask, 0, sizeof(mask));
		const std::uint8_t initial = smear_up(static_cast<std::uint8_t>(1u << bit));
		for (int rp = off; rp < 128; rp += 8) mask[rp] = initial;
		for (int i = 0; i < prefix; i++) apply_mask(op[i], mask);
		if (empty_mask(mask)) shed_mask |= (1u << db);
	}
	return shed_mask;
}
}

#endif
