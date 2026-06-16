#ifndef TM_OPENCL_OPCODES_H
#define TM_OPENCL_OPCODES_H
// opencl_opcodes.h — kernel screen-flag bits + 6502 opcode tables for the OpenCL host.
// ORDERED INCLUDE of the tm_opencl_32_8_test main.cpp TU (not standalone): included inside the
// anonymous namespace. Pure constants shared by the machine-code validation pass; an identical
// copy of the opcode tables lives in the CUDA host.
	// ── Kernel screen-flag bit definitions ──────────────────────────────────────
	static const uint8_t CHECKSUM_SENTINEL = 0x08;   // set when any checksum passes
	static const uint8_t OTHER_WORLD = 0x01;         // set when other-world checksum passes
	static const uint8_t FIRST_ENTRY_VALID = 0x02;
	static const uint8_t ALL_ENTRIES_VALID = 0x04;
	static const uint8_t USES_NOP = 0x10;
	static const uint8_t USES_UNOFFICIAL_NOPS = 0x20;
	static const uint8_t USES_ILLEGAL_OPCODES = 0x40;
	static const uint8_t USES_JAM = 0x80;

	static const uint8_t OP_JAM = 0x01;
	static const uint8_t OP_ILLEGAL = 0x02;
	static const uint8_t OP_NOP2 = 0x04;
	static const uint8_t OP_NOP = 0x08;
	static const uint8_t OP_JUMP = 0x10;

	// ── Opcode tables (6502 instruction set) ─────────────────────────────────────
	// kOpcodeBytesUsed: total byte width of each opcode (0 = illegal/not-fetched).
	// kOpcodeType: OP_* category flags for the machine-code validation pass.
	// Note: an identical copy of these tables lives in the CUDA main.cpp.
	static const uint8_t kOpcodeBytesUsed[0x100] = {
		1,2,0,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0,
		3,2,0,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0,
		1,2,0,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0,
		1,2,0,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0,
		2,2,2,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,0,3,0,0,
		2,2,2,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,3,3,3,0,
		2,2,2,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0,
		2,2,2,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0
	};

	static const uint8_t kOpcodeType[0x100] = {
		0, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, OP_JUMP, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, OP_JUMP, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		OP_NOP2, 0, OP_NOP2, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, OP_NOP2, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, OP_ILLEGAL, 0, OP_ILLEGAL, OP_ILLEGAL,
		0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		0, 0, OP_NOP2, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		0, 0, OP_NOP2, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL
	};

#endif // TM_OPENCL_OPCODES_H
