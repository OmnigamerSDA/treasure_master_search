#include "key_schedule.h"
#include "other_world_shape.h"
#include "rng_obj.h"
#include "tm_8.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
	struct SurvivorRow
	{
		uint32 key = 0;
		uint32 data = 0;
		uint32 flags = 0;
	};

	struct ScanResult
	{
		int score = 0;
		int first_bad_offset = -1;
		uint8 first_bad_opcode = 0;
		bool clean = false;
	};

	uint32 parse_u32(const std::string& text)
	{
		std::size_t consumed = 0;
		const unsigned long value = std::stoul(text, &consumed, 0);
		if (consumed != text.size())
		{
			throw std::runtime_error("Invalid integer field: " + text);
		}
		return static_cast<uint32>(value);
	}

	std::string byte_hex(uint8 value)
	{
		std::ostringstream out;
		out << "0x" << std::hex << std::uppercase << std::setfill('0') << std::setw(2)
			<< static_cast<unsigned int>(value);
		return out.str();
	}

	std::string u32_hex(uint32 value)
	{
		std::ostringstream out;
		out << "0x" << std::hex << std::uppercase << std::setfill('0') << std::setw(8) << value;
		return out.str();
	}

	std::string code_hex(const uint8* state, int code_length)
	{
		std::ostringstream out;
		out << std::hex << std::uppercase << std::setfill('0');
		for (int i = 0; i < code_length; i++)
		{
			if (i != 0)
			{
				out << ' ';
			}
			out << std::setw(2) << static_cast<unsigned int>(state[reverse_offset(i)]);
		}
		return out.str();
	}

	ScanResult linear_scan(const uint8* state)
	{
		ScanResult result;
		for (int offset = 0; offset < OTHER_WORLD_CODE_LENGTH - 2;)
		{
			const uint8 opcode = state[reverse_offset(offset)];
			if ((TM_base::opcode_type[opcode] & OP_JAM) || (TM_base::opcode_type[opcode] & OP_ILLEGAL))
			{
				result.first_bad_offset = offset;
				result.first_bad_opcode = opcode;
				result.clean = false;
				return result;
			}

			result.score++;
			const int width = static_cast<int>(TM_base::opcode_bytes_used[opcode]);
			offset += (width > 0) ? width : 1;
		}

		result.clean = true;
		return result;
	}

	int anchor_match_count(const uint8* state)
	{
		struct Anchor
		{
			int offset;
			uint8 value;
		};

		const Anchor anchors[] = {
			{ 0x00, 0xA0 }, { 0x01, 0x05 }, { 0x02, 0x4C }, { 0x03, 0x95 },
			{ 0x04, 0x85 }, { 0x07, 0x20 }, { 0x2E, 0x21 }, { 0x50, 0x60 },
		};

		int matches = 0;
		for (const Anchor& anchor : anchors)
		{
			if (state[reverse_offset(anchor.offset)] == anchor.value)
			{
				matches++;
			}
		}
		return matches;
	}

	std::vector<SurvivorRow> read_rows(const std::string& path)
	{
		std::ifstream in(path.c_str());
		if (!in)
		{
			throw std::runtime_error("Could not open input CSV: " + path);
		}

		std::vector<SurvivorRow> rows;
		std::string line;
		bool first = true;
		while (std::getline(in, line))
		{
			if (line.empty())
			{
				continue;
			}
			if (first)
			{
				first = false;
				if (line.find("key") != std::string::npos)
				{
					continue;
				}
			}

			std::stringstream ss(line);
			std::string key_text;
			std::string data_text;
			std::string flags_text;
			if (!std::getline(ss, key_text, ',') || !std::getline(ss, data_text, ',') || !std::getline(ss, flags_text, ','))
			{
				throw std::runtime_error("Malformed CSV row: " + line);
			}

			SurvivorRow row;
			row.key = parse_u32(key_text);
			row.data = parse_u32(data_text);
			row.flags = parse_u32(flags_text);
			if ((row.flags & (OTHER_WORLD | ALL_ENTRIES_VALID)) == (OTHER_WORLD | ALL_ENTRIES_VALID))
			{
				rows.push_back(row);
			}
		}
		return rows;
	}
}

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		std::cerr << "Usage: inspect_bonus2_survivors <survivor_csv> <output_csv>\n";
		return 1;
	}

	try
	{
		const std::vector<SurvivorRow> rows = read_rows(argv[1]);

		std::ofstream out(argv[2], std::ios::out | std::ios::trunc);
		if (!out)
		{
			throw std::runtime_error("Could not open output CSV: " + std::string(argv[2]));
		}

		out << "key,data,data_hex,input_flags,input_flags_hex,other_checksum,other_expected,"
			<< "other_checksum_match,machine_flags,machine_flags_hex,"
				<< "final_rts,entry_opcodes,control_flow,structural,anchor_matches,"
			<< "linear_clean,linear_score,first_bad_offset,first_bad_opcode,"
			<< "b00,b01,b02,b03,b04,b07,b2e,b50,entry00,entry05,entry0a,entry28,entry40,entry50,"
			<< "other_code_hex\n";

		RNG rng;
		tm_8 tm(&rng);
		for (const SurvivorRow& row : rows)
		{
			key_schedule schedule(row.key, key_schedule::ALL_MAPS);
			uint8 state[128] = {};
			tm.expand(row.key, row.data);
			tm.run_all_maps(schedule);
			tm.fetch_data(state);

			tm.load_data(state);
			tm.decrypt_other_world();
			tm.fetch_data(state);
			tm.load_data(state);

			const uint16 checksum = tm.calculate_other_world_checksum();
			const uint16 expected = tm.fetch_other_world_checksum_value();
			const uint8 machine_flags = tm.check_machine_code(state, OTHER_WORLD);
			const ScanResult scan = linear_scan(state);

			out << row.key << ','
				<< row.data << ','
				<< u32_hex(row.data) << ','
				<< row.flags << ','
				<< byte_hex(static_cast<uint8>(row.flags)) << ','
				<< checksum << ','
				<< expected << ','
				<< (checksum == expected ? 1 : 0) << ','
				<< static_cast<unsigned int>(machine_flags) << ','
				<< byte_hex(machine_flags) << ','
				<< (tm_other_world_shape::final_rts(state) ? 1 : 0) << ','
				<< (tm_other_world_shape::entry_opcodes(state) ? 1 : 0) << ','
				<< (tm_other_world_shape::control_flow(state) ? 1 : 0) << ','
				<< (tm_other_world_shape::structural(state) ? 1 : 0) << ','
				<< anchor_match_count(state) << ','
				<< (scan.clean ? 1 : 0) << ','
				<< scan.score << ','
				<< scan.first_bad_offset << ','
				<< (scan.first_bad_offset >= 0 ? byte_hex(scan.first_bad_opcode) : "") << ','
				<< byte_hex(state[reverse_offset(0x00)]) << ','
				<< byte_hex(state[reverse_offset(0x01)]) << ','
				<< byte_hex(state[reverse_offset(0x02)]) << ','
				<< byte_hex(state[reverse_offset(0x03)]) << ','
				<< byte_hex(state[reverse_offset(0x04)]) << ','
				<< byte_hex(state[reverse_offset(0x07)]) << ','
				<< byte_hex(state[reverse_offset(0x2E)]) << ','
				<< byte_hex(state[reverse_offset(0x50)]) << ','
				<< byte_hex(state[reverse_offset(0x00)]) << ','
				<< byte_hex(state[reverse_offset(0x05)]) << ','
				<< byte_hex(state[reverse_offset(0x0A)]) << ','
				<< byte_hex(state[reverse_offset(0x28)]) << ','
				<< byte_hex(state[reverse_offset(0x40)]) << ','
				<< byte_hex(state[reverse_offset(0x50)]) << ','
				<< '"' << code_hex(state, OTHER_WORLD_CODE_LENGTH) << '"' << "\n";
		}

		std::cout << "Wrote " << rows.size() << " all-entry-valid other-world survivor inspections to " << argv[2] << "\n";
	}
	catch (const std::exception& ex)
	{
		std::cerr << "error: " << ex.what() << "\n";
		return 1;
	}

	return 0;
}
