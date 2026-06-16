#ifndef TM_OPENCL_VALIDATE_H
#define TM_OPENCL_VALIDATE_H
// opencl_validate.h — host-side 6502 machine-code validation for OpenCL survivors.
// ORDERED INCLUDE of the tm_opencl_32_8_test main.cpp TU (not standalone): included inside the
// anonymous namespace, after opencl_opcodes.h (uses kOpcodeType/kOpcodeBytesUsed + the screen
// flags). Provides reverse_offset, hash_state, and check_machine_code.
	// ── Machine-code validation ──────────────────────────────────────────────────
	// Validates the 128-byte decrypted code region for each checksum survivor.
	// check_machine_code walks from known entry points and sets FIRST_ENTRY_VALID
	// / ALL_ENTRIES_VALID in the returned flags byte.

	int reverse_offset(int offset)
	{
		return 127 - offset;
	}

	uint64_t hash_state(const uint8_t* data)
	{
		uint64_t hash = 1469598103934665603ull;
		for (int i = 0; i < 128; i++)
		{
			hash ^= static_cast<uint64_t>(data[i]);
			hash *= 1099511628211ull;
		}
		return hash;
	}

	uint16_t calculate_checksum_from_decrypted(const uint8_t* data, bool other_world)
	{
		const int code_length = other_world ? 0x53 : 0x72;
		const int checksum_start = 130 - code_length;
		uint16_t sum = 0;
		for (int i = checksum_start; i < 128; i++)
		{
			sum = static_cast<uint16_t>(sum + data[i]);
		}
		return sum;
	}

	uint16_t fetch_checksum_value_from_decrypted(const uint8_t* data, bool other_world)
	{
		const int code_length = other_world ? 0x53 : 0x72;
		return static_cast<uint16_t>((static_cast<uint16_t>(data[reverse_offset(code_length - 1)]) << 8)
			| static_cast<uint16_t>(data[reverse_offset(code_length - 2)]));
	}

	uint8_t check_machine_code(const uint8_t* data, bool other_world)
	{
		uint8_t code_length = other_world ? 0x53 : 0x72;
		uint8_t entry_addrs[7];
		int entry_count = 0;
		if (!other_world)
		{
			entry_count = 4;
			entry_addrs[0] = 0x00;
			entry_addrs[1] = 0x2B;
			entry_addrs[2] = 0x33;
			entry_addrs[3] = 0x3E;
			entry_addrs[4] = 0xFF;
			entry_addrs[5] = 0xFF;
			entry_addrs[6] = 0xFF;
		}
		else
		{
			entry_count = 6;
			entry_addrs[0] = 0x00;
			entry_addrs[1] = 0x05;
			entry_addrs[2] = 0x0A;
			entry_addrs[3] = 0x28;
			entry_addrs[4] = 0x40;
			entry_addrs[5] = 0x50;
			entry_addrs[6] = 0xFF;
		}

		uint8_t active_entries[7] = { 0, 0, 0, 0, 0, 0, 0 };
		uint8_t hit_entries[7] = { 0, 0, 0, 0, 0, 0, 0 };
		uint8_t valid_entries[7] = { 0, 0, 0, 0, 0, 0, 0 };
		int last_entry = -1;
		uint8_t result = 0;
		uint8_t next_entry_addr = entry_addrs[0];

		for (int i = 0; i < code_length - 2; i++)
		{
			if (i == next_entry_addr)
			{
				last_entry++;
				hit_entries[last_entry] = 1;
				active_entries[last_entry] = 1;
				next_entry_addr = entry_addrs[last_entry + 1];
			}
			else if (i > next_entry_addr)
			{
				last_entry++;
				next_entry_addr = entry_addrs[last_entry + 1];
			}

			const uint8_t opcode = data[reverse_offset(i)];
			if (kOpcodeType[opcode] & OP_JAM)
			{
				result |= USES_JAM;
				break;
			}
			else if (kOpcodeType[opcode] & OP_ILLEGAL)
			{
				result |= USES_ILLEGAL_OPCODES;
				break;
			}
			else if (kOpcodeType[opcode] & OP_NOP2)
			{
				result |= USES_UNOFFICIAL_NOPS;
			}
			else if (kOpcodeType[opcode] & OP_NOP)
			{
				result |= USES_NOP;
			}
			else if (kOpcodeType[opcode] & OP_JUMP)
			{
				for (int j = 0; j < entry_count; j++)
				{
					if (active_entries[j] == 1)
					{
						active_entries[j] = 0;
						valid_entries[j] = 1;
					}
				}
			}

			i += kOpcodeBytesUsed[opcode] - 1;
		}

		bool all_entries_valid = true;
		for (int i = 0; i < entry_count; i++)
		{
			if (hit_entries[i] == 1)
			{
				if (valid_entries[i] != 1)
				{
					all_entries_valid = false;
					break;
				}
			}
			else if (entry_addrs[i] != 0xFF)
			{
				for (int j = entry_addrs[i]; j < code_length - 2; j++)
				{
					const uint8_t opcode = data[reverse_offset(j)];
					if ((kOpcodeType[opcode] & OP_JAM) || (kOpcodeType[opcode] & OP_ILLEGAL))
					{
						all_entries_valid = false;
						break;
					}
					else if (kOpcodeType[opcode] & OP_JUMP)
					{
						break;
					}

					j += kOpcodeBytesUsed[opcode] - 1;
				}
			}
		}

		if (all_entries_valid)
		{
			result |= ALL_ENTRIES_VALID;
		}
		if (valid_entries[0] == 1)
		{
			result |= FIRST_ENTRY_VALID;
		}

		return result;
	}

	void accumulate_validation_summary(ValidationSummary& summary, uint8_t machine_flags, bool other_world)
	{
		summary.total++;
		if (other_world)
		{
			summary.other++;
		}
		else
		{
			summary.carnival++;
		}
		if ((machine_flags & FIRST_ENTRY_VALID) != 0)
		{
			summary.first_entry_valid++;
		}
		if ((machine_flags & ALL_ENTRIES_VALID) != 0)
		{
			summary.all_entries_valid++;
		}
		if ((machine_flags & USES_NOP) != 0)
		{
			summary.uses_nop++;
		}
		if ((machine_flags & USES_UNOFFICIAL_NOPS) != 0)
		{
			summary.uses_unofficial_nops++;
		}
		if ((machine_flags & USES_ILLEGAL_OPCODES) != 0)
		{
			summary.uses_illegal++;
		}
		if ((machine_flags & USES_JAM) != 0)
		{
			summary.uses_jam++;
		}
	}

#endif // TM_OPENCL_VALIDATE_H
