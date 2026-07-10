#pragma once
#include <cstdint>

// Other-world (bonus2) target machine-code SHAPE predicates over the 128-byte other-world code buffer.
// CANONICAL definitions live in src/bruteforce/test_cuda/main.cpp (other_final_rts_shape /
// other_entry_opcode_shape / other_control_flow_shape / other_structural_shape + is_load_family /
// is_branch / is_call_or_jump). This header is a verbatim mirror so the CPU receiver
// (inspect_bonus2_survivors) reports the same final_rts/structural verdict the BFS verifier does.
// KEEP IN SYNC with main.cpp. code_byte_at reads the reverse-offset byte (127 - off), matching fetch_data.
namespace tm_other_world_shape
{
	inline std::uint8_t code_byte_at(const std::uint8_t* data, int offset)
	{
		return data[127 - offset];
	}

	inline bool is_load_family(std::uint8_t value)
	{
		switch (value)
		{
			case 0xA0: case 0xA1: case 0xA2: case 0xA4: case 0xA5: case 0xA6:
			case 0xA9: case 0xAC: case 0xAD: case 0xAE: case 0xB1: case 0xB4:
			case 0xB5: case 0xB6: case 0xB9: case 0xBC: case 0xBD: case 0xBE:
				return true;
			default:
				return false;
		}
	}

	inline bool is_branch(std::uint8_t value)
	{
		switch (value)
		{
			case 0x10: case 0x30: case 0x50: case 0x70:
			case 0x90: case 0xB0: case 0xD0: case 0xF0:
				return true;
			default:
				return false;
		}
	}

	inline bool is_call_or_jump(std::uint8_t value)
	{
		return value == 0x20 || value == 0x4C;
	}

	inline bool final_rts(const std::uint8_t* data)
	{
		return code_byte_at(data, 0x50) == 0x60;
	}

	inline bool entry_opcodes(const std::uint8_t* data)
	{
		return final_rts(data)
			&& is_load_family(code_byte_at(data, 0x00))
			&& is_load_family(code_byte_at(data, 0x05))
			&& is_load_family(code_byte_at(data, 0x0A))
			&& is_load_family(code_byte_at(data, 0x28))
			&& is_load_family(code_byte_at(data, 0x40));
	}

	inline bool control_flow(const std::uint8_t* data)
	{
		return final_rts(data)
			&& is_call_or_jump(code_byte_at(data, 0x02))
			&& is_call_or_jump(code_byte_at(data, 0x07))
			&& code_byte_at(data, 0x25) == 0x4C
			&& is_branch(code_byte_at(data, 0x2D))
			&& code_byte_at(data, 0x3A) == 0x20
			&& code_byte_at(data, 0x3D) == 0x4C
			&& is_branch(code_byte_at(data, 0x45))
			&& code_byte_at(data, 0x4D) == 0x4C;
	}

	inline bool structural(const std::uint8_t* data)
	{
		return entry_opcodes(data) && control_flow(data);
	}
}  // namespace tm_other_world_shape
