// Treasure Master — OpenCL kernels for the forward search.
//
// Working state: 128 bytes per candidate, held as 32 × uint32 across a
// 32-item work-group (one lane per uint32 word).  Algorithms operate on
// all 128 bytes in parallel; work-group barriers serialise cross-lane
// operations (alg2/alg5 carry propagation, checksum reduction).
//
// Kernels (development / test harness):
//   tm_process              — single-step algorithm dispatch (alg 0..7)
//   test_expand             — IV expansion (8B → 128B via RNG accumulation)
//   test_alg                — standalone algorithm test harness
//   full_process            — end-to-end (expand + schedule + alg loop)
//
// Kernels (RESEARCH / A-B + parity-reference screening path):
//   tm_stats                — kernel-side machine-flag stats accumulator
//   tm_checksum_screen      — fast checksum-only screen (forward fair filter)
//   tm_checksum_screen_offset_ilp6 — ILP6 offset-stream screen (bit-exact parity reference)
//   tm_materialize_survivors — replay survivors: expand + decrypt + validate
//
// Kernels (RESEARCH / A-B: on-GPU compaction architecture, 2026-05-29):
//   run_span_dedup          — map a candidate span, dedup, flag alive
//   compact_survivors_ordered — block-stable ordered compaction
//   resolve_flags           — union-find chain resolution for compaction
//
// Kernels (PRODUCTION engine: bounded-wave raceway — best across throughput AND memory,
// the 2026-06-16 default; OpenCL reaches ~64% of CUDA on RTX 5090 default-precert HM):
//   raceway_boundary_cap_mark_offset   — direct offset-stream cap mark pass
//   raceway_boundary_cap_state_offset  — cap-span originate (carries boundary state)
//   raceway_span_state_liveidx_cap_offset — per-boundary cap-drain span
//
// Each algorithm case (0..7) corresponds to one byte op in the schedule.
// Cases 2 and 5 require cross-lane carries; carry-ins for the last lane
// come from precomputed alg2_values / alg5_values tables.

#define reverse_offset(x) (127 - (x))

// AMD LDS-relief prototype: the raceway inner loop can keep per-lane state in
// registers and move cross-lane data with subgroup collectives instead of the
// LDS round-trip + double work-group barrier on every algorithm step. The
// alg-id source word is uniform across the wave (read from one source_lane), so
// sub_group_broadcast fits exactly; only alg2/alg5 need the +1 neighbour byte,
// which falls back to a minimal LDS exchange (this ICD lacks cl_khr_subgroup_shuffle).
// Requires the work-group to be exactly one subgroup (local size {32,1,1}).
#ifdef RACEWAY_SUBGROUP
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

// -------------------------------------------------------------------
// Development / test harness kernels
// These operate on pre-staged code_space buffers and are used only
// for unit-testing and parity checks, not for production sweeps.
// -------------------------------------------------------------------
__kernel void tm_process(__global unsigned char* code_space, __global unsigned char * regular_rng_values, __global unsigned char * alg0_values, __global unsigned char * alg6_values, __global unsigned char * rng_forward_1, __global unsigned char * rng_forward_128, __global unsigned char * alg2_values, __global unsigned char * alg5_values)
{
	__local unsigned int working_code[32];
	__local unsigned int * cur_local_int = working_code + get_local_id(0);

	unsigned int temp1 = get_global_id(0) / 32;
	unsigned int temp2 = get_global_id(1);

	__global unsigned char * start_addr = code_space + (temp1 * 10000 + temp2) * 132;
	__global unsigned int * cur_int = (__global unsigned int *)((__global unsigned char *)start_addr + 4 + get_local_id(0) * 4);

	int algorithm_id = *start_addr;
	unsigned short rng_seed = (*(start_addr + 1) << 8) | *(start_addr + 2);

	// Copy the int out of global memory into local
	*cur_local_int = *cur_int;

	/*
	if (get_local_id(0) == 0)
	{
		*cur_int = (temp1 * 10000 + temp2);
	}
	else
	{
		*cur_int = get_local_id(0);
	}
	*/

	int temp;
	switch (algorithm_id)
	{
		case 0x00:
			temp = ((*cur_local_int << 1) & 0xFEFEFEFE) | *(__global unsigned int*)(alg0_values + rng_seed * 128 +  get_local_id(0) * 4);
			break;
		case 0x01:
			temp = ((*cur_local_int & 0x00FF00FF) + ((*(__global unsigned int*)(regular_rng_values + rng_seed * 128 +  get_local_id(0) * 4)) & 0x00FF00FF)) & 0x00FF00FF;
			temp |= ((*cur_local_int & 0xFF00FF00) + ((*(__global unsigned int*)(regular_rng_values + rng_seed * 128 +  get_local_id(0) * 4)) & 0xFF00FF00)) & 0xFF00FF00;
			break;
		// alg2: rotate-left-1 across all bytes; needs neighbor lane's low bit
		// (alg2_values carry-in for last lane).
		case 0x02:
			temp = (*cur_local_int & 0x00010000) >> 8;
			if (get_local_id(0) == 31)
			{
				temp |= *(__global unsigned int *)(alg2_values + rng_seed * 4);
			}
			else
			{
				temp |= ((*(cur_local_int + 1) & 0x000000001) << 24);
			}
			temp |= (*cur_local_int >> 1) & 0x007F007F;
			temp |= (*cur_local_int << 1) & 0xFE00FE00;
			temp |= (*cur_local_int >> 8) & 0x00800080;
			break;
		case 0x03:
			temp = *cur_local_int ^ *(__global unsigned int*)(regular_rng_values + rng_seed * 128 +  get_local_id(0) * 4);
			break;

		// alg4: subtract-from-FF then add 1 — equivalent to 0x0001 + (~rng) per
		// byte half, decomposed across the two 16-bit halves.
		case 0x04:
			temp = ((*cur_local_int & 0x00FF00FF) + (((*(__global unsigned int*)(regular_rng_values + rng_seed * 128 +  get_local_id(0) * 4)) & 0x00FF00FF) ^ 0x00FF00FF) + 0x00010001) & 0x00FF00FF;
			temp |= ((*cur_local_int & 0xFF00FF00) + (((*(__global unsigned int*)(regular_rng_values + rng_seed * 128 +  get_local_id(0) * 4)) & 0xFF00FF00) ^ 0xFF00FF00) + 0x01000100) & 0xFF00FF00;
			break;
		// alg5: rotate-right-1 across all bytes; needs neighbor lane's high bit
		// (alg5_values carry-in for last lane). Mirror of alg2.
		case 0x05:
			temp = (*cur_local_int & 0x00800000) >> 8;
			if (get_local_id(0) == 31)
			{
				temp |= *(__global unsigned int *)(alg5_values + rng_seed * 4);
			}
			else
			{
				temp |= ((*(cur_local_int + 1) & 0x000000080) << 24);
			}
			temp |= (*cur_local_int >> 1) & 0x7F007F00;
			temp |= (*cur_local_int << 1) & 0x00FE00FE;
			temp |= (*cur_local_int >> 8) & 0x00010001;
			break;
		case 0x06:
			temp = ((*cur_local_int >> 1) & 0x7F7F7F7F) | *(__global unsigned int*)(alg6_values + rng_seed * 128 +  get_local_id(0) * 4);
			break;
		case 0x07:
			temp = *cur_local_int ^ 0xFFFFFFFF;
			break;
		default:
			break;
	}
	
	// Make sure each WU is done computing
	barrier(CLK_LOCAL_MEM_FENCE);

	// Store the result back into local memory
	*cur_local_int = temp;
	
	if (get_local_id(0) == 0 && (algorithm_id == 3 || algorithm_id == 1 || algorithm_id == 4 || algorithm_id == 0 || algorithm_id == 6))
	{
		*(start_addr + 1+1) = rng_forward_128[rng_seed*2];
		*(start_addr + 1) = rng_forward_128[rng_seed*2+1];
	}
	else if (get_local_id(0) == 0 && (algorithm_id == 2 || algorithm_id == 5))
	{
		*(start_addr + 1+1) = rng_forward_1[rng_seed*2];
		*(start_addr + 1) = rng_forward_1[rng_seed*2+1];
	}
	

	// Copy the result back to global memory
	*cur_int = *cur_local_int;
}

// Expansion kernel:
//   take in IV, expand it into local memory, write local memory back to global.

unsigned int pack_be_u32(unsigned int value)
{
	return ((value & 0x000000FFU) << 24)
		| ((value & 0x0000FF00U) << 8)
		| ((value & 0x00FF0000U) >> 8)
		| ((value & 0xFF000000U) >> 24);
}

void copy_from_global(__global unsigned char* code_space, __local unsigned int * working_code, int code_index, int int_index)
{
	working_code[int_index] = ((__global unsigned int*)code_space)[code_index * 32 + int_index];
}

void copy_to_global(__global unsigned char* code_space, __local unsigned int * working_code, int code_index, int int_index)
{
	((__global unsigned int*)code_space)[code_index * 32 + int_index] = working_code[int_index]; 
}

void expand(__local unsigned int * working_code, int int_index, unsigned short rng_seed, __global unsigned char * expansion_values)
{
	unsigned int temp;
	temp = ((working_code[int_index] & 0x00FF00FF) + ((*(__global unsigned int*)(expansion_values + rng_seed * 128 + int_index * 4)) & 0x00FF00FF)) & 0x00FF00FF;
	temp |= ((working_code[int_index] & 0xFF00FF00) + ((*(__global unsigned int*)(expansion_values + rng_seed * 128 + int_index * 4)) & 0xFF00FF00)) & 0xFF00FF00;
	working_code[int_index] = temp;
}

// -------------------------------------------------------------------
// test_expand — IV expansion: interleave key/data halves, accumulate
// RNG values byte-by-byte to produce the 128-byte initial working state.
// -------------------------------------------------------------------
__kernel void test_expand(__global unsigned char* code_space, __global unsigned char* input_ivs, __global unsigned char* expansion_values)
{
	__local unsigned int working_code[32];

	unsigned int code_index = get_global_id(1);
	unsigned int int_index = get_local_id(0);

	unsigned int key_half = int_index % 2;
	working_code[int_index] = *((__global unsigned int *)(input_ivs + code_index * 8 + key_half * 4));
	unsigned short rng_seed = (*(input_ivs + code_index * 8) << 8) | *(input_ivs + code_index * 8 + 1);

	expand(working_code, int_index, rng_seed, expansion_values);
	copy_to_global(code_space, working_code, code_index, int_index);
}

void run_alg(__local unsigned int * working_code, int int_index, int alg_id, unsigned short * rng_seed, __global unsigned char * regular_rng_values, __global unsigned char * alg0_values, __global unsigned char * alg6_values, __global unsigned short * rng_forward_1, __global unsigned short * rng_forward_128, __global unsigned char * alg2_values, __global unsigned char * alg5_values)
{
	__local unsigned int * cur_local_int = working_code + int_index;
	__local unsigned char * working_bytes = (__local unsigned char*)working_code;
	int byte_index = int_index * 4;
		
	int temp;
	switch (alg_id)
	{
		case 0x00:
			temp = ((*cur_local_int << 1) & 0xFEFEFEFE) | *(__global unsigned int*)(alg0_values + *rng_seed * 128 +  get_local_id(0) * 4);
			break;
		case 0x01:
			temp = ((*cur_local_int & 0x00FF00FF) + ((*(__global unsigned int*)(regular_rng_values + *rng_seed * 128 +  get_local_id(0) * 4)) & 0x00FF00FF)) & 0x00FF00FF;
			temp |= ((*cur_local_int & 0xFF00FF00) + ((*(__global unsigned int*)(regular_rng_values + *rng_seed * 128 +  get_local_id(0) * 4)) & 0xFF00FF00)) & 0xFF00FF00;
			break;
		case 0x02:
		{
			unsigned char b0 = working_bytes[byte_index + 0];
			unsigned char b1 = working_bytes[byte_index + 1];
			unsigned char b2 = working_bytes[byte_index + 2];
			unsigned char b3 = working_bytes[byte_index + 3];
			unsigned char carry_in = (int_index == 31)
				? *(alg2_values + *rng_seed * 4 + 3)
				: (working_bytes[byte_index + 4] & 0x01);
			unsigned char out0 = (unsigned char)((b0 >> 1) | (b1 & 0x80));
			unsigned char out1 = (unsigned char)((b1 << 1) | (b2 & 0x01));
			unsigned char out2 = (unsigned char)((b2 >> 1) | (b3 & 0x80));
			unsigned char out3 = (unsigned char)((b3 << 1) | carry_in);
			temp = (int)out0 | ((int)out1 << 8) | ((int)out2 << 16) | ((int)out3 << 24);
			break;
		}
		case 0x03:
			temp = *cur_local_int ^ *(__global unsigned int*)(regular_rng_values + *rng_seed * 128 +  get_local_id(0) * 4);
			break;
	
		// alg4: subtract-from-FF then add 1; see tm_process for the rationale.
		case 0x04:
			temp = ((*cur_local_int & 0x00FF00FF) + (((*(__global unsigned int*)(regular_rng_values + *rng_seed * 128 +  get_local_id(0) * 4)) & 0x00FF00FF) ^ 0x00FF00FF) + 0x00010001) & 0x00FF00FF;
			temp |= ((*cur_local_int & 0xFF00FF00) + (((*(__global unsigned int*)(regular_rng_values + *rng_seed * 128 +  get_local_id(0) * 4)) & 0xFF00FF00) ^ 0xFF00FF00) + 0x01000100) & 0xFF00FF00;
			break;
		case 0x05:
		{
			unsigned char b0 = working_bytes[byte_index + 0];
			unsigned char b1 = working_bytes[byte_index + 1];
			unsigned char b2 = working_bytes[byte_index + 2];
			unsigned char b3 = working_bytes[byte_index + 3];
			unsigned char carry_in = (int_index == 31)
				? *(alg5_values + *rng_seed * 4 + 3)
				: (working_bytes[byte_index + 4] & 0x80);
			unsigned char out0 = (unsigned char)((b0 << 1) | (b1 & 0x01));
			unsigned char out1 = (unsigned char)((b1 >> 1) | (b2 & 0x80));
			unsigned char out2 = (unsigned char)((b2 << 1) | (b3 & 0x01));
			unsigned char out3 = (unsigned char)((b3 >> 1) | carry_in);
			temp = (int)out0 | ((int)out1 << 8) | ((int)out2 << 16) | ((int)out3 << 24);
			break;
		}
		case 0x06:
			temp = ((*cur_local_int >> 1) & 0x7F7F7F7F) | *(__global unsigned int*)(alg6_values + *rng_seed * 128 +  get_local_id(0) * 4);
			break;
		case 0x07:
			temp = *cur_local_int ^ 0xFFFFFFFF;
			break;
		default:
			temp = alg_id;
			break;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	*cur_local_int = temp;

	if (alg_id == 3 || alg_id == 1 || alg_id == 4 || alg_id == 0 || alg_id == 6)
	{
		*rng_seed = (*((__global unsigned char *)rng_forward_128 + *rng_seed * 2 + 1) << 8) | *((__global unsigned char *)rng_forward_128 + *rng_seed * 2);
	}
	else if (alg_id == 2 || alg_id == 5)
	{
		*rng_seed = (*((__global unsigned char *)rng_forward_1 + *rng_seed * 2 + 1) << 8) | *((__global unsigned char *)rng_forward_1 + *rng_seed * 2);
	}

	barrier(CLK_LOCAL_MEM_FENCE);
}

// -------------------------------------------------------------------
// test_alg — run a single algorithm step on pre-staged working state.
// -------------------------------------------------------------------
__kernel void test_alg(__global unsigned char* code_space, __global unsigned char * test_data, __global unsigned char * regular_rng_values, __global unsigned char * alg0_values, __global unsigned char * alg6_values, __global unsigned short * rng_forward_1, __global unsigned short * rng_forward_128, __global unsigned char * alg2_values, __global unsigned char * alg5_values)
{
	__local unsigned int working_code[32];
	unsigned short rng_seed;

	unsigned int code_index = get_global_id(1);
	unsigned int int_index = get_local_id(0);

	__global unsigned char * cur_test_data = test_data + code_index * 3;

	// get alg id, rng seed from test data
	int alg_id = *(cur_test_data);
	
	rng_seed = *(cur_test_data + 1) << 8 | *(cur_test_data + 2);

	copy_from_global(code_space, working_code, code_index, int_index);
	run_alg(working_code, int_index, alg_id, &rng_seed, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
	copy_to_global(code_space, working_code, code_index, int_index);
	*(cur_test_data + 1) = (rng_seed >> 8) & 0xFF;
	*(cur_test_data + 2) = rng_seed & 0xFF;
}

void run_one_map(__local unsigned int* working_code, unsigned int int_index, __global unsigned char * regular_rng_values, __global unsigned char * alg0_values, __global unsigned char * alg6_values, __global unsigned short * rng_forward_1, __global unsigned short * rng_forward_128, __global unsigned char * alg2_values, __global unsigned char * alg5_values, __global unsigned char * schedule_data, int schedule_index)
{
	unsigned short rng_seed = (*(schedule_data + schedule_index * 4) << 8) | *(schedule_data + schedule_index * 4 + 1);
	unsigned short nibble_selector = (*(schedule_data + schedule_index * 4 + 2) << 8) | *(schedule_data + schedule_index * 4 + 3);
	
	// Next, the working code is processed with the same steps 16 times:
	for (int i = 0; i < 16; i++)
	{
		// Get the highest bit of the nibble selector to use as a flag
		unsigned char nibble = (nibble_selector >> 15) & 0x01;
		// Shift the nibble selector up one bit
		nibble_selector = nibble_selector << 1;

		// If the flag is a 1, get the high nibble of the current byte
		// Otherwise use the low nibble
		unsigned char current_byte = *((__local unsigned char*)working_code + i);
		if (nibble == 1)
		{
			current_byte = current_byte >> 4;
		}

		// Mask off only 3 bits
		unsigned char alg_id = (current_byte >> 1) & 0x07;

		run_alg(working_code, int_index, alg_id, &rng_seed, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values);
	}

}


// -------------------------------------------------------------------
// full_process — end-to-end: expand IV → run 27-schedule → decrypt.
// For comparison against the screen-only path; not used in production.
// -------------------------------------------------------------------
__kernel void full_process(__global unsigned char* code_space, __global unsigned char * test_data, __global unsigned char * regular_rng_values, __global unsigned char * alg0_values, __global unsigned char * alg6_values, __global unsigned short * rng_forward_1, __global unsigned short * rng_forward_128, __global unsigned char * alg2_values, __global unsigned char * alg5_values, __global unsigned char * expansion_values, __global unsigned char * schedule_data, unsigned int key, unsigned int data_start)
{
	__local unsigned int working_code[32];
	__local unsigned short rng_seed;

	unsigned int code_index = get_global_id(1);
	unsigned int int_index = get_local_id(0);
	unsigned int cur_data = data_start + code_index;

	if (int_index % 2 == 0)
	{
		working_code[int_index] = pack_be_u32(key);
	}
	else
	{
		working_code[int_index] = pack_be_u32(cur_data);
	}

	expand(working_code, int_index, key >> 16, expansion_values);

	for (int i = 0; i < 27; i++)
	{
		run_one_map(working_code, int_index, regular_rng_values, alg0_values, alg6_values, rng_forward_1, rng_forward_128, alg2_values, alg5_values, schedule_data, i);
	}
	copy_to_global(code_space, working_code, code_index, int_index);
}

// -------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------
#define CHECKSUM_SENTINEL    0x08
#define OP_JAM               0x01
#define OP_ILLEGAL           0x02
#define OP_NOP2              0x04
#define OP_NOP_OP            0x08
#define OP_JUMP              0x10

#define OTHER_WORLD          0x01
#define FIRST_ENTRY_VALID    0x02
#define ALL_ENTRIES_VALID    0x04
#define USES_NOP             0x10
#define USES_UNOFFICIAL_NOPS 0x20
#define USES_ILLEGAL_OPCODES 0x40
#define USES_JAM             0x80

__constant unsigned char other_world_data_k[128] = {
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0xCA,0x68,0xC1,0x66,0x44,
	0xD2,0x04,0x0B,0x90,0x81,0x86,0xC7,0xF4,0xD2,0xE2,
	0xF1,0x22,0xE3,0x0C,0xD9,0x54,0xFB,0xFF,0x0A,0xCF,
	0x81,0x72,0x0A,0x94,0x9A,0x98,0xD3,0xFF,0xAB,0x80,
	0x9A,0xE5,0xB7,0x45,0x6E,0x8F,0xD2,0xF0,0x67,0xFF,
	0xB3,0xAE,0x49,0xBB,0x9C,0x06,0x12,0x40,0x49,0xA3,
	0x9A,0xDB,0x32,0x7B,0x58,0xA1,0x5A,0xB9,0x2B,0x2B,
	0x2D,0x6E,0x36,0x93,0x1C,0x1A,0x52,0x03,0x18,0xE4,
	0x5E,0xB1,0xC1,0xBD,0x44,0xFB,0xF1,0x50
};

// -------------------------------------------------------------------
// 6502 opcode tables used by check_machine_code (in tm_stats).
// opcode_bytes_used_k: instruction width (0 = JAM/ILLEGAL/not fetched).
// opcode_type_k: OP_* category flags (OP_JAM=1, OP_ILLEGAL=2, OP_NOP2=4,
//   OP_NOP=8, OP_JUMP=16).
// Identical data is in main.cpp (kOpcodeBytesUsed / kOpcodeType).
// -------------------------------------------------------------------
__constant unsigned char opcode_bytes_used_k[256] = {
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

// OP_JAM=1,OP_ILLEGAL=2,OP_NOP2=4,OP_NOP=8,OP_JUMP=16
__constant unsigned char opcode_type_k[256] = {
	 0, 0, 1, 2, 4, 0, 0, 2, 0, 0, 0, 2, 4, 0, 0, 2,
	16, 0, 1, 2, 4, 0, 0, 2, 0, 0, 4, 2, 4, 0, 0, 2,
	16, 0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2,
	16, 0, 1, 2, 4, 0, 0, 2, 0, 0, 4, 2, 4, 0, 0, 2,
	16, 0, 1, 2, 4, 0, 0, 2, 0, 0, 0, 2,16, 0, 0, 2,
	16, 0, 1, 2, 4, 0, 0, 2, 0, 0, 4, 2, 4, 0, 0, 2,
	16, 0, 1, 2, 4, 0, 0, 2, 0, 0, 0, 2,16, 0, 0, 2,
	16, 0, 1, 2, 4, 0, 0, 2, 0, 0, 4, 2, 4, 0, 0, 2,
	 4, 0, 4, 2, 0, 0, 0, 2, 0, 4, 0, 2, 0, 0, 0, 2,
	16, 0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 2, 2, 0, 2, 2,
	 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2,
	16, 0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2,
	 0, 0, 4, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2,
	16, 0, 1, 2, 4, 0, 0, 2, 0, 0, 4, 2, 4, 0, 0, 2,
	 0, 0, 4, 2, 0, 0, 0, 2, 0, 0, 8, 2, 0, 0, 0, 2,
	16, 0, 1, 2, 4, 0, 0, 2, 0, 0, 4, 2, 4, 0, 0, 2
};

__constant unsigned char carnival_world_checksum_mask_k[128] = {
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x00,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF
};

__constant unsigned char other_world_checksum_mask_k[128] = {
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF
};

// -------------------------------------------------------------------
// Checksum helpers (called by tm_stats; run from lane 0 only)
// -------------------------------------------------------------------
unsigned short masked_checksum(__local unsigned int* data_i, __constant unsigned char* mask)
{
	__local unsigned char* data = (__local unsigned char*)data_i;
	unsigned short sum = 0;
	for (int i = 0; i < 128; i++)
	{
		sum = (unsigned short)(sum + (unsigned short)(data[i] & mask[i]));
	}
	return sum;
}

unsigned short fetch_checksum_value_local(__local unsigned int* data_i, int code_length)
{
	__local unsigned char* data = (__local unsigned char*)data_i;
	return (unsigned short)(((unsigned short)data[127 - (code_length - 1)] << 8)
		| (unsigned short)data[127 - (code_length - 2)]);
}

// -------------------------------------------------------------------
// checksum_ok: run by thread 0 only
// -------------------------------------------------------------------
int checksum_ok(__local unsigned int* data_i, int length)
{
	if (length == 0x72)
	{
		return masked_checksum(data_i, carnival_world_checksum_mask_k) == fetch_checksum_value_local(data_i, length);
	}
	else if (length == 0x53)
	{
		return masked_checksum(data_i, other_world_checksum_mask_k) == fetch_checksum_value_local(data_i, length);
	}
	return 0;
}

// -------------------------------------------------------------------
// machine_code_flags: run by thread 0 only
// e0..e6 are entry addresses (0xFF = unused); entry_count excludes sentinels.
// -------------------------------------------------------------------
unsigned char machine_code_flags(__local unsigned int* data_i, int code_length,
                                  unsigned char e0, unsigned char e1, unsigned char e2,
                                  unsigned char e3, unsigned char e4, unsigned char e5,
                                  unsigned char e6, int entry_count)
{
	__local unsigned char* data = (__local unsigned char*)data_i;
	unsigned char entry_addrs[7] = { e0, e1, e2, e3, e4, e5, e6 };
	unsigned char active_entries[7]  = { 0,0,0,0,0,0,0 };
	unsigned char hit_entries[7]     = { 0,0,0,0,0,0,0 };
	unsigned char valid_entries[7]   = { 0,0,0,0,0,0,0 };
	int last_entry = -1;
	unsigned char result = 0;
	unsigned char next_entry_addr = e0;

	for (int i = 0; i < code_length - 2; i++)
	{
		if (i == (int)next_entry_addr)
		{
			last_entry++;
			hit_entries[last_entry] = 1;
			active_entries[last_entry] = 1;
			next_entry_addr = entry_addrs[last_entry + 1];
		}
		else if (i > (int)next_entry_addr)
		{
			last_entry++;
			next_entry_addr = entry_addrs[last_entry + 1];
		}

		unsigned char opcode = data[127 - i];
		unsigned char otype  = opcode_type_k[opcode];

		if (otype & OP_JAM)
		{
			result |= USES_JAM;
			break;
		}
		else if (otype & OP_ILLEGAL)
		{
			result |= USES_ILLEGAL_OPCODES;
			break;
		}
		else if (otype & OP_NOP2)
		{
			result |= USES_UNOFFICIAL_NOPS;
		}
		else if (otype & OP_NOP_OP)
		{
			result |= USES_NOP;
		}
		else if (otype & OP_JUMP)
		{
			for (int j = 0; j < entry_count; j++)
			{
				if (active_entries[j] == 1)
				{
					active_entries[j] = 0;
					valid_entries[j]  = 1;
				}
			}
		}

		i += (int)opcode_bytes_used_k[opcode] - 1;
	}

	int all_entries_valid = 1;
	for (int i = 0; i < entry_count; i++)
	{
		if (hit_entries[i] == 1)
		{
			if (valid_entries[i] != 1)
			{
				all_entries_valid = 0;
				break;
			}
		}
		else if (entry_addrs[i] == 255)
		{
			continue;
		}
		else
		{
			for (int j = (int)entry_addrs[i]; j < code_length - 2; j++)
			{
				unsigned char opcode = data[127 - j];
				unsigned char otype  = opcode_type_k[opcode];
				if ((otype & OP_JAM) || (otype & OP_ILLEGAL))
				{
					all_entries_valid = 0;
					break;
				}
				else if (otype & OP_JUMP)
				{
					break;
				}
				j += (int)opcode_bytes_used_k[opcode] - 1;
			}
		}
		if (!all_entries_valid) break;
	}

	if (all_entries_valid)       result |= ALL_ENTRIES_VALID;
	if (valid_entries[0] == 1)   result |= FIRST_ENTRY_VALID;

	return result;
}

// -------------------------------------------------------------------
// tm_stats kernel
// -------------------------------------------------------------------
__kernel void tm_stats(
	__global unsigned char*  result_data,
	__global unsigned char*  regular_rng_values,
	__global unsigned char*  alg0_values,
	__global unsigned char*  alg6_values,
	__global unsigned short* rng_forward_1,
	__global unsigned short* rng_forward_128,
	__global unsigned char*  alg2_values,
	__global unsigned char*  alg5_values,
	__global unsigned char*  expansion_values,
	__global unsigned char*  schedule_data,
	__global unsigned char*  carnival_data,
	__global unsigned char*  unused_param,
	unsigned int key,
	unsigned int data_start)
{
	__local unsigned int working_code[32];
	__local unsigned int decrypted_carnival[32];
	__local unsigned int decrypted_other[32];

	unsigned int code_index = get_global_id(1);
	unsigned int int_index  = get_local_id(0);
	unsigned int cur_data   = data_start + code_index;

	if (int_index % 2 == 0)
		working_code[int_index] = pack_be_u32(key);
	else
		working_code[int_index] = pack_be_u32(cur_data);

	expand(working_code, int_index, key >> 16, expansion_values);
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < 27; i++)
	{
		run_one_map(working_code, int_index,
		            regular_rng_values, alg0_values, alg6_values,
		            rng_forward_1, rng_forward_128,
		            alg2_values, alg5_values,
		            schedule_data, i);
	}

	// All 32 threads cooperatively decrypt both worlds
	decrypted_carnival[int_index] = working_code[int_index]
	                              ^ ((__global unsigned int*)carnival_data)[int_index];
	((__local unsigned char*)decrypted_other)[int_index * 4 + 0] = ((__local unsigned char*)working_code)[int_index * 4 + 0] ^ other_world_data_k[int_index * 4 + 0];
	((__local unsigned char*)decrypted_other)[int_index * 4 + 1] = ((__local unsigned char*)working_code)[int_index * 4 + 1] ^ other_world_data_k[int_index * 4 + 1];
	((__local unsigned char*)decrypted_other)[int_index * 4 + 2] = ((__local unsigned char*)working_code)[int_index * 4 + 2] ^ other_world_data_k[int_index * 4 + 2];
	((__local unsigned char*)decrypted_other)[int_index * 4 + 3] = ((__local unsigned char*)working_code)[int_index * 4 + 3] ^ other_world_data_k[int_index * 4 + 3];

	barrier(CLK_LOCAL_MEM_FENCE);

	if (int_index == 0)
	{
		unsigned char stats = 0;

		if (checksum_ok(decrypted_carnival, 0x72))
		{
			// Carnival world checksum passed
			stats = machine_code_flags(decrypted_carnival, 0x72,
			                           0, 0x2B, 0x33, 0x3E, 0xFF, 0xFF, 0xFF, 4);
			stats |= CHECKSUM_SENTINEL;
		}
		else if (checksum_ok(decrypted_other, 0x53))
		{
			// Other world checksum passed
			stats = machine_code_flags(decrypted_other, 0x53,
			                           0, 0x05, 0x0A, 0x28, 0x40, 0x50, 0xFF, 6);
			stats |= CHECKSUM_SENTINEL | OTHER_WORLD;
		}

		result_data[code_index] = stats;
	}
}

// -------------------------------------------------------------------
// tm_checksum_screen kernel
// GPU-only fair screen: expand, maps, decrypt, checksum.
// The host can do any later CPU-side validation on survivors.
// -------------------------------------------------------------------
__kernel void tm_checksum_screen(
	__global unsigned char*  result_data,
	__global unsigned char*  regular_rng_values,
	__global unsigned char*  alg0_values,
	__global unsigned char*  alg6_values,
	__global unsigned short* rng_forward_1,
	__global unsigned short* rng_forward_128,
	__global unsigned char*  alg2_values,
	__global unsigned char*  alg5_values,
	__global unsigned char*  expansion_values,
	__global unsigned char*  schedule_data,
	__global unsigned char*  carnival_data,
	unsigned int key,
	unsigned int data_start)
{
	__local unsigned int working_code[32];
	__local unsigned int decrypted_carnival[32];
	__local unsigned int decrypted_other[32];

	unsigned int code_index = get_global_id(1);
	unsigned int int_index = get_local_id(0);
	unsigned int cur_data = data_start + code_index;

	if (int_index % 2 == 0)
		working_code[int_index] = pack_be_u32(key);
	else
		working_code[int_index] = pack_be_u32(cur_data);

	expand(working_code, int_index, key >> 16, expansion_values);
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < 27; i++)
	{
		run_one_map(working_code, int_index,
		            regular_rng_values, alg0_values, alg6_values,
		            rng_forward_1, rng_forward_128,
		            alg2_values, alg5_values,
		            schedule_data, i);
	}

	decrypted_carnival[int_index] = working_code[int_index]
		^ ((__global unsigned int*)carnival_data)[int_index];
	((__local unsigned char*)decrypted_other)[int_index * 4 + 0] = ((__local unsigned char*)working_code)[int_index * 4 + 0] ^ other_world_data_k[int_index * 4 + 0];
	((__local unsigned char*)decrypted_other)[int_index * 4 + 1] = ((__local unsigned char*)working_code)[int_index * 4 + 1] ^ other_world_data_k[int_index * 4 + 1];
	((__local unsigned char*)decrypted_other)[int_index * 4 + 2] = ((__local unsigned char*)working_code)[int_index * 4 + 2] ^ other_world_data_k[int_index * 4 + 2];
	((__local unsigned char*)decrypted_other)[int_index * 4 + 3] = ((__local unsigned char*)working_code)[int_index * 4 + 3] ^ other_world_data_k[int_index * 4 + 3];

	barrier(CLK_LOCAL_MEM_FENCE);

	if (int_index == 0)
	{
		unsigned char screen_flags = 0;

		if (checksum_ok(decrypted_carnival, 0x72))
		{
			screen_flags = CHECKSUM_SENTINEL;
		}
		else if (checksum_ok(decrypted_other, 0x53))
		{
			screen_flags = CHECKSUM_SENTINEL | OTHER_WORLD;
		}

		result_data[code_index] = screen_flags;
	}
}

// -------------------------------------------------------------------
// tm_checksum_screen_offset_ilp6 kernel
// -------------------------------------------------------------------
// Port of the CUDA `tm_checksum_screen_offset_store_ilp6_preids_cuda`
// production kernel (May 2026 advancements):
//   * Offset-stream RNG — host precomputes per-key streams indexed by
//     (schedule_step, rng_offset_within_step); kernel does indexed reads
//     instead of walking rng_seed via rng_forward_1/128.
//   * ILP6 — 32 threads × 6 candidates per work-group, amortizing
//     work-group overhead and barrier cost over 6× more work.
//   * PreIDs — compute all 6 alg_ids before the alg dispatch so the
//     compiler can hoist all 6 reads from source_lane in one block.
//   * Per-byte store (no packed uint32 — ILP6 isn't divisible by 4 and
//     candidate_base isn't 4-aligned, matches the CUDA fallback path).
//
// Offset-stream layout (per key, computed on host):
//   offset_regular  : 27 × 2048 × 128 bytes (regular_rng_values per step×pos)
//   offset_alg0     : 27 × 2048 × 128 bytes (alg0_values per step×pos)
//   offset_alg6     : 27 × 2048 × 128 bytes (alg6_values per step×pos)
//   offset_alg2     : 27 × 2048 × 4 bytes (carry uint32 per step×pos)
//   offset_alg5     : 27 × 2048 × 4 bytes (carry uint32 per step×pos)
//
// rng_offset[j] advances by 128 per alg-0/1/3/4/6, by 1 per alg-2/5
// (matching the CUDA semantics — alg-7 doesn't advance).
__kernel void tm_checksum_screen_offset_ilp6(
	__global unsigned char*  result_data,
	__global unsigned char*  offset_regular,
	__global unsigned char*  offset_alg0,
	__global unsigned char*  offset_alg6,
	__global unsigned int*   offset_alg2,
	__global unsigned int*   offset_alg5,
	__global unsigned char*  expansion_values,
	__global unsigned char*  schedule_data,
	__global unsigned char*  carnival_data,
	unsigned int key,
	unsigned int data_start)
{
	#define ILP6_N 6

	__local unsigned int working_code[32 * ILP6_N];
	__local unsigned int decrypted_carnival[32];
	__local unsigned int decrypted_other[32];

	const unsigned int int_index = get_local_id(0);          // lane 0..31
	const unsigned int wg_index  = get_global_id(1);         // work-group index along candidate axis
	const unsigned int candidate_base = wg_index * ILP6_N;

	// Phase: init + expansion (for all 6 candidates).
	#pragma unroll
	for (int j = 0; j < ILP6_N; j++)
	{
		const unsigned int cur_data = data_start + candidate_base + (unsigned int)j;
		unsigned int word;
		if ((int_index & 1u) == 0u)
			word = pack_be_u32(key);
		else
			word = pack_be_u32(cur_data);
		working_code[j * 32 + int_index] = word;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	#pragma unroll
	for (int j = 0; j < ILP6_N; j++)
	{
		expand(working_code + j * 32, (int)int_index, (unsigned short)(key >> 16), expansion_values);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Schedule loop — 27 steps × 16 inner iters × 6 ILP candidates.
	for (int sched = 0; sched < 27; sched++)
	{
		// Per-candidate offset within this schedule step. Advances by 128
		// for alg-0/1/3/4/6, by 1 for alg-2/5, stays for alg-7.
		unsigned int rng_offset[ILP6_N];
		#pragma unroll
		for (int j = 0; j < ILP6_N; j++) rng_offset[j] = 0u;

		// Schedule nibble selectors (lane-broadcast — same for all lanes).
		const unsigned int packed_sched_b2 = (unsigned int)schedule_data[sched * 4 + 2];
		const unsigned int packed_sched_b3 = (unsigned int)schedule_data[sched * 4 + 3];
		unsigned short nibble_selector = (unsigned short)((packed_sched_b2 << 8) | packed_sched_b3);

		for (int i = 0; i < 16; i++)
		{
			const unsigned int source_lane = (unsigned int)(i >> 2);
			const unsigned int source_shift = (unsigned int)((i & 3) * 8);
			const unsigned int algorithm_shift = source_shift + 1u + (((unsigned int)nibble_selector >> 13u) & 4u);

			// Phase 1 (precompute IDs): read source_lane for all 6 candidates,
			// extract algorithm_id. Reads only — no writes to working_code yet.
			unsigned char algorithm_id[ILP6_N];
			#pragma unroll
			for (int j = 0; j < ILP6_N; j++)
			{
				const unsigned int source_word = working_code[j * 32 + source_lane];
				algorithm_id[j] = (unsigned char)((source_word >> algorithm_shift) & 0x07u);
			}

			// Phase 2 (apply): each candidate reads + writes its own lane.
			unsigned int new_values[ILP6_N];
			#pragma unroll
			for (int j = 0; j < ILP6_N; j++)
			{
				const unsigned int val = working_code[j * 32 + int_index];
				const unsigned char aid = algorithm_id[j];
				unsigned int temp = val;

				if (aid == 0u)
				{
					temp = ((val << 1) & 0xFEFEFEFEu) |
					       *(__global unsigned int*)(offset_alg0 + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					rng_offset[j] += 128u;
				}
				else if (aid == 1u)
				{
					const unsigned int r = *(__global unsigned int*)(offset_regular + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					temp = ((val & 0x00FF00FFu) + (r & 0x00FF00FFu)) & 0x00FF00FFu;
					temp |= ((val & 0xFF00FF00u) + (r & 0xFF00FF00u)) & 0xFF00FF00u;
					rng_offset[j] += 128u;
				}
				else if (aid == 3u)
				{
					temp = val ^ *(__global unsigned int*)(offset_regular + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					rng_offset[j] += 128u;
				}
				else if (aid == 4u)
				{
					const unsigned int r = *(__global unsigned int*)(offset_regular + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					temp = ((val & 0x00FF00FFu) + ((r & 0x00FF00FFu) ^ 0x00FF00FFu) + 0x00010001u) & 0x00FF00FFu;
					temp |= ((val & 0xFF00FF00u) + ((r & 0xFF00FF00u) ^ 0xFF00FF00u) + 0x01000100u) & 0xFF00FF00u;
					rng_offset[j] += 128u;
				}
				else if (aid == 6u)
				{
					temp = ((val >> 1) & 0x7F7F7F7Fu) |
					       *(__global unsigned int*)(offset_alg6 + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					rng_offset[j] += 128u;
				}
				else if (aid == 7u)
				{
					temp = val ^ 0xFFFFFFFFu;
				}
				else if (aid == 2u)
				{
					// rotate-left-1 across bytes; carry from next lane's low bit.
					unsigned int t = (val & 0x00010000u) >> 8;
					if (int_index == 31u)
					{
						t |= offset_alg2[sched * 2048u + rng_offset[j]];
					}
					else
					{
						const unsigned int next_lane_val = working_code[j * 32 + int_index + 1u];
						t |= ((next_lane_val & 0x00000001u) << 24);
					}
					t |= (val >> 1) & 0x007F007Fu;
					t |= (val << 1) & 0xFE00FE00u;
					t |= (val >> 8) & 0x00800080u;
					temp = t;
					rng_offset[j] += 1u;
				}
				else /* aid == 5 */
				{
					unsigned int t = (val & 0x00800000u) >> 8;
					if (int_index == 31u)
					{
						t |= offset_alg5[sched * 2048u + rng_offset[j]];
					}
					else
					{
						const unsigned int next_lane_val = working_code[j * 32 + int_index + 1u];
						t |= ((next_lane_val & 0x00000080u) << 24);
					}
					t |= (val >> 1) & 0x7F007F00u;
					t |= (val << 1) & 0x00FE00FEu;
					t |= (val >> 8) & 0x00010001u;
					temp = t;
					rng_offset[j] += 1u;
				}

				new_values[j] = temp;
			}

			// Barrier between read-source-lane and write-own-lane: must ensure
			// all reads complete before any thread writes (since other threads'
			// next iteration source_lane reads could observe updated values).
			barrier(CLK_LOCAL_MEM_FENCE);

			#pragma unroll
			for (int j = 0; j < ILP6_N; j++)
			{
				working_code[j * 32 + int_index] = new_values[j];
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			nibble_selector = (unsigned short)(nibble_selector << 1);
		}
	}

	// Checksum phase — sequential per candidate, reusing decrypted_carnival/other
	// scratch arrays. Lane 0 writes the screen flag per candidate.
	#pragma unroll
	for (int j = 0; j < ILP6_N; j++)
	{
		decrypted_carnival[int_index] = working_code[j * 32 + int_index]
		                              ^ ((__global unsigned int*)carnival_data)[int_index];
		((__local unsigned char*)decrypted_other)[int_index * 4 + 0] = ((__local unsigned char*)(working_code + j * 32))[int_index * 4 + 0] ^ other_world_data_k[int_index * 4 + 0];
		((__local unsigned char*)decrypted_other)[int_index * 4 + 1] = ((__local unsigned char*)(working_code + j * 32))[int_index * 4 + 1] ^ other_world_data_k[int_index * 4 + 1];
		((__local unsigned char*)decrypted_other)[int_index * 4 + 2] = ((__local unsigned char*)(working_code + j * 32))[int_index * 4 + 2] ^ other_world_data_k[int_index * 4 + 2];
		((__local unsigned char*)decrypted_other)[int_index * 4 + 3] = ((__local unsigned char*)(working_code + j * 32))[int_index * 4 + 3] ^ other_world_data_k[int_index * 4 + 3];

		barrier(CLK_LOCAL_MEM_FENCE);

		if (int_index == 0u)
		{
			unsigned char screen_flags = 0;
			if (checksum_ok(decrypted_carnival, 0x72))
			{
				screen_flags = CHECKSUM_SENTINEL;
			}
			else if (checksum_ok(decrypted_other, 0x53))
			{
				screen_flags = CHECKSUM_SENTINEL | OTHER_WORLD;
			}
			result_data[candidate_base + (unsigned int)j] = screen_flags;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	#undef ILP6_N
}

// -------------------------------------------------------------------
// tm_materialize_survivors kernel
// Replays only checksum survivors and writes decrypted bytes back out
// for CPU-side machine-code validation.
// -------------------------------------------------------------------
__kernel void tm_materialize_survivors(
	__global unsigned char*  output_data,
	__global unsigned int*   survivor_data,
	__global unsigned char*  survivor_flags,
	__global unsigned char*  regular_rng_values,
	__global unsigned char*  alg0_values,
	__global unsigned char*  alg6_values,
	__global unsigned short* rng_forward_1,
	__global unsigned short* rng_forward_128,
	__global unsigned char*  alg2_values,
	__global unsigned char*  alg5_values,
	__global unsigned char*  expansion_values,
	__global unsigned char*  schedule_data,
	__global unsigned char*  carnival_data,
	unsigned int key)
{
	__local unsigned int working_code[32];

	unsigned int survivor_index = get_global_id(1);
	unsigned int int_index = get_local_id(0);
	unsigned int cur_data = survivor_data[survivor_index];
	unsigned char screen_flags = survivor_flags[survivor_index];

	if (int_index % 2 == 0)
		working_code[int_index] = pack_be_u32(key);
	else
		working_code[int_index] = pack_be_u32(cur_data);

	expand(working_code, int_index, key >> 16, expansion_values);
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < 27; i++)
	{
		run_one_map(working_code, int_index,
		            regular_rng_values, alg0_values, alg6_values,
		            rng_forward_1, rng_forward_128,
		            alg2_values, alg5_values,
		            schedule_data, i);
	}

	unsigned int decrypted_word;
	if ((screen_flags & OTHER_WORLD) == 0)
	{
		decrypted_word = working_code[int_index]
			^ ((__global unsigned int*)carnival_data)[int_index];
	}
	else
	{
		unsigned char out0 = ((__local unsigned char*)working_code)[int_index * 4 + 0] ^ other_world_data_k[int_index * 4 + 0];
		unsigned char out1 = ((__local unsigned char*)working_code)[int_index * 4 + 1] ^ other_world_data_k[int_index * 4 + 1];
		unsigned char out2 = ((__local unsigned char*)working_code)[int_index * 4 + 2] ^ other_world_data_k[int_index * 4 + 2];
		unsigned char out3 = ((__local unsigned char*)working_code)[int_index * 4 + 3] ^ other_world_data_k[int_index * 4 + 3];
		decrypted_word = (unsigned int)out0 | ((unsigned int)out1 << 8) | ((unsigned int)out2 << 16) | ((unsigned int)out3 << 24);
	}

	((__global unsigned int*)output_data)[survivor_index * 32 + int_index] = decrypted_word;
}

// ===================================================================
// On-GPU VRAM survivor-compaction architecture (OpenCL port).
//
// Geometry: single work-group of 32 lanes handles SPAN_ILP candidates (the
// dedup window W = SPAN_ILP). State lives in __local (mirrors the ilp6 screen).
// Because only lane 0 does the per-span dedup inserts (sequentially over the
// SPAN_ILP candidates), NO atomics are needed inside the span kernel — the
// int64-local-atomic concern is sidestepped entirely. Cross-span re-blocking
// (ordered compaction) widens the effective dedup window over the run.
// ===================================================================
#ifndef SPAN_ILP
#define SPAN_ILP   8       // dedup window W = SPAN_ILP; single-warp is optimal on OpenCL
#endif
#define SPAN_NSLOT (SPAN_ILP * 4)   // power-of-two when SPAN_ILP is; load factor 4
#ifndef RACEWAY_CAP_ILP
#define RACEWAY_CAP_ILP 4
#endif

// -------------------------------------------------------------------
// LDS-resident RNG staging (rng_lds_staging_architectural_exploration_20260615.md).
// Moves the per-map RNG-row reads off the saturated VMEM unit onto the idle LDS
// unit. The ×3 offset tables (regular/alg0/alg6) collapse to regular only —
// proven bit-exact over all 65536 seeds (lds_staging_probe.cpp):
//   alg0_word[L] = (regular_word[L]    >> 7) & 0x01010101   (same lane)
//   alg6_word[L] = bswap(regular_word[31-L]) & 0x80808080   (position-reversed)
// Every touched rng_offset = 128*f + c with f+c <= 16 (f = #full-RNG steps so
// far [+128 each], c = #rotate steps [+1 each]), so the touched set is a complete
// bounded triangular superset (<=153 rows) and the LDS slot index is pure
// arithmetic — no scatter table, no cold tail (slot < STAGE_ROWS is a dead-but-
// safe guard). STAGE_ROWS is passed by the host (>=153).
#ifdef RACEWAY_LDS_RNG
#ifndef STAGE_ROWS
#define STAGE_ROWS 160
#endif
// kLdsSlotBase[f] = sum_{g<f}(17-g) = #slots reserved for full-counts below f
// (each full-count f' carries c in [0, 16-f'], i.e. 17-f' slots).
__constant unsigned int kLdsSlotBase[17] = {
	0u, 17u, 33u, 48u, 62u, 75u, 87u, 98u, 108u, 117u, 125u, 132u, 138u, 143u, 147u, 150u, 152u
};

inline unsigned int lds_slot_index(unsigned int rng_offset)
{
	return kLdsSlotBase[rng_offset >> 7] + (rng_offset & 0x7Fu);
}

inline unsigned int lds_bswap32(unsigned int v)
{
	return (v << 24) | ((v & 0x0000FF00u) << 8) | ((v & 0x00FF0000u) >> 8) | (v >> 24);
}

// Cooperatively stage this map's regular RNG rows into LDS (lane int_index loads
// its own word for every structural slot). ~153 VMEM loads/lane/map, amortized
// over RACEWAY_CAP_ILP * 16 LDS-served reads.
void stage_map_rng(__local unsigned int* staged_regular, unsigned int int_index,
                   unsigned int sched, __global unsigned char* offset_regular)
{
	for (unsigned int f = 0u; f <= 16u; f++)
	{
		const unsigned int cmax = 16u - f;
		const unsigned int slot_base = kLdsSlotBase[f];
		for (unsigned int c = 0u; c <= cmax; c++)
		{
			const unsigned int rng_offset = (f << 7) + c;
			staged_regular[(slot_base + c) * 32u + int_index] =
				*(__global unsigned int*)(offset_regular + ((sched * 2048u + rng_offset) * 128u + int_index * 4u));
		}
	}
}
#endif
#ifdef RACEWAY_CAP_FP64
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
typedef unsigned long raceway_cap_word_t;
#else
typedef unsigned int raceway_cap_word_t;
#endif

// Fingerprint the full 32-word state (lane 0 only): per-word murmur3, XOR-merged.
// Collision-safe within an SPAN_ILP=8 window at 32 bits.
unsigned int span_fingerprint(__local unsigned int* st)
{
	unsigned int h = 0u;
	for (int i = 0; i < 32; i++)
	{
		unsigned int v = st[i] + (unsigned int)i * 0x9e3779b9u;
		v ^= v >> 16; v *= 0x85ebca6bu; v ^= v >> 13; v *= 0xc2b2ae35u; v ^= v >> 16;
		h ^= v;
	}
	return h ? h : 1u;
}

#ifdef RACEWAY_CAP_FP64
unsigned long raceway_rotl64(unsigned long x, int r)
{
	return (x << r) | (x >> (64 - r));
}

unsigned long raceway_hstrong_step(unsigned long h, unsigned long w)
{
	h ^= w;
	h = raceway_rotl64(h, 31);
	h *= 0x9e3779b97f4a7c15UL;
	h ^= h >> 29;
	h *= 0xbf58476d1ce4e5b9UL;
	return h;
}

raceway_cap_word_t raceway_state_fingerprint(__local unsigned int* st)
{
	unsigned long h = 0xcbf29ce484222325UL;
	for (int i = 0; i < 16; i++)
	{
		const unsigned long lo = (unsigned long)st[i * 2];
		const unsigned long hi = (unsigned long)st[i * 2 + 1];
		h = raceway_hstrong_step(h, lo | (hi << 32));
	}
	h ^= h >> 32;
	return h ? h : 1UL;
}

unsigned int raceway_cap_bucket(raceway_cap_word_t fp, unsigned int cap_bits)
{
	return (unsigned int)((fp * 0x9e3779b97f4a7c15UL) >> (64u - cap_bits));
}

raceway_cap_word_t raceway_atomic_cmpxchg(
	volatile __global raceway_cap_word_t* slot,
	raceway_cap_word_t cmp,
	raceway_cap_word_t value)
{
	return atom_cmpxchg(slot, cmp, value);
}

raceway_cap_word_t raceway_atomic_xchg(
	volatile __global raceway_cap_word_t* slot,
	raceway_cap_word_t value)
{
	return atom_xchg(slot, value);
}
#else
raceway_cap_word_t raceway_state_fingerprint(__local unsigned int* st)
{
	return span_fingerprint(st);
}

unsigned int raceway_cap_bucket(raceway_cap_word_t fp, unsigned int cap_bits)
{
	return (fp * 0x9e3779b9u) >> (32u - cap_bits);
}

raceway_cap_word_t raceway_atomic_cmpxchg(
	volatile __global raceway_cap_word_t* slot,
	raceway_cap_word_t cmp,
	raceway_cap_word_t value)
{
	return atomic_cmpxchg(slot, cmp, value);
}

raceway_cap_word_t raceway_atomic_xchg(
	volatile __global raceway_cap_word_t* slot,
	raceway_cap_word_t value)
{
	return atomic_xchg(slot, value);
}
#endif

// Portable OpenCL 1.2 fixed-cap probe. This intentionally uses 32-bit
// fingerprints unless cl_khr_int64_base_atomics is available at build time.
int raceway_cap_probe_or_keep(
	raceway_cap_word_t fp,
	volatile __global raceway_cap_word_t* table,
	unsigned int cap_bits,
	unsigned int cap_ways)
{
	if (table == 0 || cap_bits == 0u || cap_ways == 0u) return 0;
	if (fp == (raceway_cap_word_t)0) fp = (raceway_cap_word_t)1;

	const unsigned int bucket = raceway_cap_bucket(fp, cap_bits);
	const unsigned int base = bucket * cap_ways;

	for (unsigned int w = 0u; w < cap_ways; w++)
	{
		const raceway_cap_word_t cur = table[base + w];
		if (cur == fp) return 1;
	}

#ifdef RACEWAY_CAP_FP64
	const unsigned int victim = base + (unsigned int)(((fp >> 32) ^ (fp >> 17) ^ fp) % (raceway_cap_word_t)cap_ways);
#else
	const unsigned int victim = base + (unsigned int)(((fp >> 16) ^ (fp >> 7) ^ fp) % (raceway_cap_word_t)cap_ways);
#endif
#ifdef RACEWAY_CAP_NONATOMIC
	// Benign-race plain store (lever #2): the cap is FN-safe over-keep, so a lost
	// concurrent write just re-keeps a candidate later — it never drops a true hit.
	// Drops the global-atomic-swap to a plain global store (CUDA does the same).
	table[victim] = fp;
#else
	(void)raceway_atomic_xchg(table + victim, fp);
#endif
	return 0;
}

unsigned int tm_deposit_bits32(unsigned int bits, unsigned int mask)
{
	unsigned int out = 0u;
	while (mask != 0u)
	{
		const unsigned int bit = mask & (0u - mask);
		if ((bits & 1u) != 0u) out |= bit;
		bits >>= 1u;
		mask ^= bit;
	}
	return out;
}

unsigned int raceway_precert_data_value(
	unsigned int logical,
	unsigned int fixed_value,
	unsigned int support_mask,
	unsigned int use_precert)
{
	return (use_precert != 0u)
		? (fixed_value | tm_deposit_bits32(logical, support_mask))
		: logical;
}

__kernel void raceway_boundary_cap_mark_offset(
	__global unsigned char*  alive_out,
	__global unsigned char*  drop_map_out,
	volatile __global raceway_cap_word_t* cap_table,
	unsigned int cap_bits,
	unsigned int cap_ways,
	unsigned int cap_count,
	__global unsigned char*  offset_regular,
	__global unsigned char*  offset_alg0,
	__global unsigned char*  offset_alg6,
	__global unsigned int*   offset_alg2,
	__global unsigned int*   offset_alg5,
	__global unsigned char*  expansion_values,
	__global unsigned char*  schedule_data,
	unsigned int key,
	unsigned int data_start,
	unsigned int candidate_count,
	unsigned int first_cap_map,
	__global unsigned int* cap_maps,
	__global unsigned int* work_counter,
	unsigned int persistent_queue,
	unsigned int precert_fixed_value,
	unsigned int precert_support_mask,
	unsigned int use_precert)
{
	__local unsigned int working_code[32 * RACEWAY_CAP_ILP];
	__local unsigned char alive_local[RACEWAY_CAP_ILP];
	__local unsigned char drop_local[RACEWAY_CAP_ILP];
	__local unsigned int task_base_local;
	__local unsigned int any_alive_local;

	const unsigned int int_index = get_local_id(0);
	const unsigned int static_base = get_global_id(1) * RACEWAY_CAP_ILP;
	const unsigned int last_map = (cap_count == 0u) ? 0u : (cap_maps[cap_count - 1u] + 1u);

	while (1)
	{
		if (int_index == 0u)
		{
			task_base_local = (persistent_queue != 0u && work_counter != 0)
				? atomic_add(work_counter, (unsigned int)RACEWAY_CAP_ILP)
				: static_base;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		const unsigned int base = task_base_local;
		if (base >= candidate_count) break;

		if (int_index == 0u)
		{
			for (int j = 0; j < RACEWAY_CAP_ILP; j++)
			{
				const unsigned int c = base + (unsigned int)j;
				alive_local[j] = (c < candidate_count) ? 1u : 0u;
				drop_local[j] = 0xFFu;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int j = 0; j < RACEWAY_CAP_ILP; j++)
		{
			const unsigned int logical = data_start + base + (unsigned int)j;
			const unsigned int cur_data = raceway_precert_data_value(
				logical, precert_fixed_value, precert_support_mask, use_precert);
			working_code[j * 32 + int_index] = ((int_index & 1u) == 0u) ? pack_be_u32(key) : pack_be_u32(cur_data);
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int j = 0; j < RACEWAY_CAP_ILP; j++)
		{
			expand(working_code + j * 32, (int)int_index, (unsigned short)(key >> 16), expansion_values);
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (unsigned int sched = 0u; sched < last_map; sched++)
		{
			unsigned int rng_offset[RACEWAY_CAP_ILP];
			for (int j = 0; j < RACEWAY_CAP_ILP; j++) rng_offset[j] = 0u;

			const unsigned int packed_sched_b2 = (unsigned int)schedule_data[sched * 4 + 2];
			const unsigned int packed_sched_b3 = (unsigned int)schedule_data[sched * 4 + 3];
			unsigned short nibble_selector = (unsigned short)((packed_sched_b2 << 8) | packed_sched_b3);

			for (int i = 0; i < 16; i++)
			{
				const unsigned int source_lane = (unsigned int)(i >> 2);
				const unsigned int algorithm_shift = (unsigned int)((i & 3) * 8) + 1u + (((unsigned int)nibble_selector >> 13u) & 4u);

				unsigned char algorithm_id[RACEWAY_CAP_ILP];
				for (int j = 0; j < RACEWAY_CAP_ILP; j++)
				{
					const unsigned int source_word = working_code[j * 32 + source_lane];
					algorithm_id[j] = (unsigned char)((source_word >> algorithm_shift) & 0x07u);
				}

				unsigned int new_values[RACEWAY_CAP_ILP];
				for (int j = 0; j < RACEWAY_CAP_ILP; j++)
				{
					const unsigned int val = working_code[j * 32 + int_index];
					const unsigned char aid = algorithm_id[j];
					unsigned int temp = val;
					if (alive_local[j] == 0u)
					{
						new_values[j] = temp;
						continue;
					}
					if (aid == 0u)
					{
						temp = ((val << 1) & 0xFEFEFEFEu) | *(__global unsigned int*)(offset_alg0 + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
						rng_offset[j] += 128u;
					}
					else if (aid == 1u)
					{
						const unsigned int r = *(__global unsigned int*)(offset_regular + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
						temp = ((val & 0x00FF00FFu) + (r & 0x00FF00FFu)) & 0x00FF00FFu;
						temp |= ((val & 0xFF00FF00u) + (r & 0xFF00FF00u)) & 0xFF00FF00u;
						rng_offset[j] += 128u;
					}
					else if (aid == 3u)
					{
						temp = val ^ *(__global unsigned int*)(offset_regular + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
						rng_offset[j] += 128u;
					}
					else if (aid == 4u)
					{
						const unsigned int r = *(__global unsigned int*)(offset_regular + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
						temp = ((val & 0x00FF00FFu) + ((r & 0x00FF00FFu) ^ 0x00FF00FFu) + 0x00010001u) & 0x00FF00FFu;
						temp |= ((val & 0xFF00FF00u) + ((r & 0xFF00FF00u) ^ 0xFF00FF00u) + 0x01000100u) & 0xFF00FF00u;
						rng_offset[j] += 128u;
					}
					else if (aid == 6u)
					{
						temp = ((val >> 1) & 0x7F7F7F7Fu) | *(__global unsigned int*)(offset_alg6 + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
						rng_offset[j] += 128u;
					}
					else if (aid == 7u)
					{
						temp = val ^ 0xFFFFFFFFu;
					}
					else if (aid == 2u)
					{
						unsigned int t = (val & 0x00010000u) >> 8;
						if (int_index == 31u) t |= offset_alg2[sched * 2048u + rng_offset[j]];
						else { const unsigned int nlv = working_code[j * 32 + int_index + 1u]; t |= ((nlv & 0x00000001u) << 24); }
						t |= (val >> 1) & 0x007F007Fu; t |= (val << 1) & 0xFE00FE00u; t |= (val >> 8) & 0x00800080u;
						temp = t; rng_offset[j] += 1u;
					}
					else
					{
						unsigned int t = (val & 0x00800000u) >> 8;
						if (int_index == 31u) t |= offset_alg5[sched * 2048u + rng_offset[j]];
						else { const unsigned int nlv = working_code[j * 32 + int_index + 1u]; t |= ((nlv & 0x00000080u) << 24); }
						t |= (val >> 1) & 0x7F007F00u; t |= (val << 1) & 0x00FE00FEu; t |= (val >> 8) & 0x00010001u;
						temp = t; rng_offset[j] += 1u;
					}
					new_values[j] = temp;
				}
				barrier(CLK_LOCAL_MEM_FENCE);
				for (int j = 0; j < RACEWAY_CAP_ILP; j++) working_code[j * 32 + int_index] = new_values[j];
				barrier(CLK_LOCAL_MEM_FENCE);
				nibble_selector = (unsigned short)(nibble_selector << 1);
			}

			unsigned int cap_idx = 0xFFFFFFFFu;
			for (unsigned int ci = 0u; ci < cap_count; ci++)
			{
				if (cap_maps[ci] == sched)
				{
					cap_idx = ci;
					break;
				}
			}
			if (cap_idx != 0xFFFFFFFFu)
			{
				if (int_index == 0u)
				{
					const ulong cap_slots_per_table = ((cap_bits == 0u) ? 1UL : (1UL << cap_bits)) * (ulong)cap_ways;
					volatile __global raceway_cap_word_t* cap_table_for_map = cap_table + (ulong)cap_idx * cap_slots_per_table;
					for (int j = 0; j < RACEWAY_CAP_ILP; j++)
					{
						if (alive_local[j] == 0u) continue;
						const raceway_cap_word_t fp = raceway_state_fingerprint(working_code + j * 32);
						if (raceway_cap_probe_or_keep(fp, cap_table_for_map, cap_bits, cap_ways))
						{
							alive_local[j] = 0u;
							drop_local[j] = (unsigned char)sched;
						}
					}
				}
				barrier(CLK_LOCAL_MEM_FENCE);
				if (int_index == 0u)
				{
					unsigned int any_alive = 0u;
					for (int j = 0; j < RACEWAY_CAP_ILP; j++)
					{
						any_alive |= (unsigned int)(alive_local[j] != 0u);
					}
					any_alive_local = any_alive;
				}
				barrier(CLK_LOCAL_MEM_FENCE);
				if (any_alive_local == 0u) break;
			}
		}

		if (int_index == 0u)
		{
			for (int j = 0; j < RACEWAY_CAP_ILP; j++)
			{
				const unsigned int c = base + (unsigned int)j;
				if (c >= candidate_count) continue;
				alive_out[c] = alive_local[j];
				if (drop_map_out != 0) drop_map_out[c] = drop_local[j];
			}
		}
		if (persistent_queue == 0u) break;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

void raceway_run_map_range_local(
	__local unsigned int* working_code,
	__local unsigned char* alive_local,
	unsigned int int_index,
	unsigned int start_map,
	unsigned int end_map,
	__global unsigned char* offset_regular,
	__global unsigned char* offset_alg0,
	__global unsigned char* offset_alg6,
	__global unsigned int* offset_alg2,
	__global unsigned int* offset_alg5,
	__global unsigned char* schedule_data
#ifdef RACEWAY_LDS_RNG
	, __local unsigned int* staged_regular
#endif
	)
{
#ifdef RACEWAY_LDS_RNG
	// --- LDS-resident RNG variant -----------------------------------------------
	// Stage this map's regular rows into LDS once, then serve all RNG reads from
	// LDS and derive alg0/alg6 on the fly (idle VALU). alg2/5 carries stay global
	// (tiny, lane-31 only). This is the production path under -DRACEWAY_LDS_RNG.
	for (unsigned int sched = start_map; sched <= end_map && sched < 27u; sched++)
	{
		stage_map_rng(staged_regular, int_index, sched, offset_regular);
		barrier(CLK_LOCAL_MEM_FENCE);

		unsigned int rng_offset[RACEWAY_CAP_ILP];
		for (int j = 0; j < RACEWAY_CAP_ILP; j++) rng_offset[j] = 0u;

		const unsigned int packed_sched_b2 = (unsigned int)schedule_data[sched * 4 + 2];
		const unsigned int packed_sched_b3 = (unsigned int)schedule_data[sched * 4 + 3];
		unsigned short nibble_selector = (unsigned short)((packed_sched_b2 << 8) | packed_sched_b3);

		for (int i = 0; i < 16; i++)
		{
			const unsigned int source_lane = (unsigned int)(i >> 2);
			const unsigned int algorithm_shift = (unsigned int)((i & 3) * 8) + 1u + (((unsigned int)nibble_selector >> 13u) & 4u);

			unsigned char algorithm_id[RACEWAY_CAP_ILP];
			for (int j = 0; j < RACEWAY_CAP_ILP; j++)
			{
				const unsigned int source_word = working_code[j * 32 + source_lane];
				algorithm_id[j] = (unsigned char)((source_word >> algorithm_shift) & 0x07u);
			}

			unsigned int new_values[RACEWAY_CAP_ILP];
			for (int j = 0; j < RACEWAY_CAP_ILP; j++)
			{
				const unsigned int val = working_code[j * 32 + int_index];
				const unsigned char aid = algorithm_id[j];
				unsigned int temp = val;
				if (alive_local[j] == 0u) { new_values[j] = temp; continue; }

				// Pure-LDS serving: the staged set is a complete structural superset
				// (slot < STAGE_ROWS always), so there is no global fallback in the hot
				// loop — every RNG read is an LDS read; alg0/alg6 derived on the fly.
				const unsigned int slot = lds_slot_index(rng_offset[j]);
				if (aid == 0u)
				{
					const unsigned int reg = staged_regular[slot * 32u + int_index];
					temp = ((val << 1) & 0xFEFEFEFEu) | ((reg >> 7) & 0x01010101u);
					rng_offset[j] += 128u;
				}
				else if (aid == 1u)
				{
					const unsigned int r = staged_regular[slot * 32u + int_index];
					temp = ((val & 0x00FF00FFu) + (r & 0x00FF00FFu)) & 0x00FF00FFu;
					temp |= ((val & 0xFF00FF00u) + (r & 0xFF00FF00u)) & 0xFF00FF00u;
					rng_offset[j] += 128u;
				}
				else if (aid == 3u)
				{
					const unsigned int r = staged_regular[slot * 32u + int_index];
					temp = val ^ r;
					rng_offset[j] += 128u;
				}
				else if (aid == 4u)
				{
					const unsigned int r = staged_regular[slot * 32u + int_index];
					temp = ((val & 0x00FF00FFu) + ((r & 0x00FF00FFu) ^ 0x00FF00FFu) + 0x00010001u) & 0x00FF00FFu;
					temp |= ((val & 0xFF00FF00u) + ((r & 0xFF00FF00u) ^ 0xFF00FF00u) + 0x01000100u) & 0xFF00FF00u;
					rng_offset[j] += 128u;
				}
				else if (aid == 6u)
				{
					// alg6 row is position-reversed: derive from the mirror lane's word.
					const unsigned int regm = staged_regular[slot * 32u + (31u - int_index)];
					temp = ((val >> 1) & 0x7F7F7F7Fu) | (lds_bswap32(regm) & 0x80808080u);
					rng_offset[j] += 128u;
				}
				else if (aid == 7u)
				{
					temp = val ^ 0xFFFFFFFFu;
				}
				else if (aid == 2u)
				{
					unsigned int t = (val & 0x00010000u) >> 8;
					if (int_index == 31u) t |= offset_alg2[sched * 2048u + rng_offset[j]];
					else { const unsigned int nlv = working_code[j * 32 + int_index + 1u]; t |= ((nlv & 0x00000001u) << 24); }
					t |= (val >> 1) & 0x007F007Fu; t |= (val << 1) & 0xFE00FE00u; t |= (val >> 8) & 0x00800080u;
					temp = t; rng_offset[j] += 1u;
				}
				else
				{
					unsigned int t = (val & 0x00800000u) >> 8;
					if (int_index == 31u) t |= offset_alg5[sched * 2048u + rng_offset[j]];
					else { const unsigned int nlv = working_code[j * 32 + int_index + 1u]; t |= ((nlv & 0x00000080u) << 24); }
					t |= (val >> 1) & 0x7F007F00u; t |= (val << 1) & 0x00FE00FEu; t |= (val >> 8) & 0x00010001u;
					temp = t; rng_offset[j] += 1u;
				}
				new_values[j] = temp;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			for (int j = 0; j < RACEWAY_CAP_ILP; j++) working_code[j * 32 + int_index] = new_values[j];
			barrier(CLK_LOCAL_MEM_FENCE);
			nibble_selector = (unsigned short)(nibble_selector << 1);
		}
	}
#elif defined(RACEWAY_SUBGROUP)
	// --- AMD LDS-relief variant -------------------------------------------------
	// Per-lane state lives in registers (st[]); cross-lane data moves via
	// sub_group_broadcast (uniform alg-id source_lane). Only alg2/alg5 need the
	// +1 neighbour byte, handled by a minimal LDS exchange. Lane-local algorithm
	// steps touch neither LDS nor a barrier — that is the whole win. Requires the
	// work-group to be exactly one subgroup (local size {32,1,1}); int_index is
	// the subgroup local id.
	unsigned int st[RACEWAY_CAP_ILP];
	for (int j = 0; j < RACEWAY_CAP_ILP; j++) st[j] = working_code[j * 32 + int_index];

	for (unsigned int sched = start_map; sched <= end_map && sched < 27u; sched++)
	{
		unsigned int rng_offset[RACEWAY_CAP_ILP];
		for (int j = 0; j < RACEWAY_CAP_ILP; j++) rng_offset[j] = 0u;

		const unsigned int packed_sched_b2 = (unsigned int)schedule_data[sched * 4 + 2];
		const unsigned int packed_sched_b3 = (unsigned int)schedule_data[sched * 4 + 3];
		unsigned short nibble_selector = (unsigned short)((packed_sched_b2 << 8) | packed_sched_b3);

		for (int i = 0; i < 16; i++)
		{
			const unsigned int source_lane = (unsigned int)(i >> 2);
			const unsigned int algorithm_shift = (unsigned int)((i & 3) * 8) + 1u + (((unsigned int)nibble_selector >> 13u) & 4u);

			// Alg id is read from one (uniform) source_lane for every lane -> broadcast.
			unsigned char algorithm_id[RACEWAY_CAP_ILP];
			int need_lds = 0;
			for (int j = 0; j < RACEWAY_CAP_ILP; j++)
			{
				const unsigned int source_word = sub_group_broadcast(st[j], source_lane);
				const unsigned char aid = (unsigned char)((source_word >> algorithm_shift) & 0x07u);
				algorithm_id[j] = aid;
				if (alive_local[j] != 0u && (aid == 2u || aid == 5u)) need_lds = 1;
			}

			// need_lds is uniform across the wave, so these barriers stay convergent.
			if (need_lds)
			{
				barrier(CLK_LOCAL_MEM_FENCE);
				for (int j = 0; j < RACEWAY_CAP_ILP; j++) working_code[j * 32 + int_index] = st[j];
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			for (int j = 0; j < RACEWAY_CAP_ILP; j++)
			{
				if (alive_local[j] == 0u) continue;
				const unsigned int val = st[j];
				const unsigned char aid = algorithm_id[j];
				unsigned int temp = val;
				if (aid == 0u)
				{
					temp = ((val << 1) & 0xFEFEFEFEu) | *(__global unsigned int*)(offset_alg0 + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					rng_offset[j] += 128u;
				}
				else if (aid == 1u)
				{
					const unsigned int r = *(__global unsigned int*)(offset_regular + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					temp = ((val & 0x00FF00FFu) + (r & 0x00FF00FFu)) & 0x00FF00FFu;
					temp |= ((val & 0xFF00FF00u) + (r & 0xFF00FF00u)) & 0xFF00FF00u;
					rng_offset[j] += 128u;
				}
				else if (aid == 3u)
				{
					temp = val ^ *(__global unsigned int*)(offset_regular + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					rng_offset[j] += 128u;
				}
				else if (aid == 4u)
				{
					const unsigned int r = *(__global unsigned int*)(offset_regular + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					temp = ((val & 0x00FF00FFu) + ((r & 0x00FF00FFu) ^ 0x00FF00FFu) + 0x00010001u) & 0x00FF00FFu;
					temp |= ((val & 0xFF00FF00u) + ((r & 0xFF00FF00u) ^ 0xFF00FF00u) + 0x01000100u) & 0xFF00FF00u;
					rng_offset[j] += 128u;
				}
				else if (aid == 6u)
				{
					temp = ((val >> 1) & 0x7F7F7F7Fu) | *(__global unsigned int*)(offset_alg6 + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					rng_offset[j] += 128u;
				}
				else if (aid == 7u)
				{
					temp = val ^ 0xFFFFFFFFu;
				}
				else if (aid == 2u)
				{
					unsigned int t = (val & 0x00010000u) >> 8;
					if (int_index == 31u) t |= offset_alg2[sched * 2048u + rng_offset[j]];
					else { const unsigned int nlv = working_code[j * 32 + int_index + 1u]; t |= ((nlv & 0x00000001u) << 24); }
					t |= (val >> 1) & 0x007F007Fu; t |= (val << 1) & 0xFE00FE00u; t |= (val >> 8) & 0x00800080u;
					temp = t; rng_offset[j] += 1u;
				}
				else
				{
					unsigned int t = (val & 0x00800000u) >> 8;
					if (int_index == 31u) t |= offset_alg5[sched * 2048u + rng_offset[j]];
					else { const unsigned int nlv = working_code[j * 32 + int_index + 1u]; t |= ((nlv & 0x00000080u) << 24); }
					t |= (val >> 1) & 0x7F007F00u; t |= (val << 1) & 0x00FE00FEu; t |= (val >> 8) & 0x00010001u;
					temp = t; rng_offset[j] += 1u;
				}
				st[j] = temp;
			}
			nibble_selector = (unsigned short)(nibble_selector << 1);
		}
	}

	// Publish final register state so the caller's fingerprint / state-save reads it.
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int j = 0; j < RACEWAY_CAP_ILP; j++) working_code[j * 32 + int_index] = st[j];
	barrier(CLK_LOCAL_MEM_FENCE);
#else
	for (unsigned int sched = start_map; sched <= end_map && sched < 27u; sched++)
	{
		unsigned int rng_offset[RACEWAY_CAP_ILP];
		for (int j = 0; j < RACEWAY_CAP_ILP; j++) rng_offset[j] = 0u;

		const unsigned int packed_sched_b2 = (unsigned int)schedule_data[sched * 4 + 2];
		const unsigned int packed_sched_b3 = (unsigned int)schedule_data[sched * 4 + 3];
		unsigned short nibble_selector = (unsigned short)((packed_sched_b2 << 8) | packed_sched_b3);

		for (int i = 0; i < 16; i++)
		{
			const unsigned int source_lane = (unsigned int)(i >> 2);
			const unsigned int algorithm_shift = (unsigned int)((i & 3) * 8) + 1u + (((unsigned int)nibble_selector >> 13u) & 4u);

			unsigned char algorithm_id[RACEWAY_CAP_ILP];
			for (int j = 0; j < RACEWAY_CAP_ILP; j++)
			{
				const unsigned int source_word = working_code[j * 32 + source_lane];
				algorithm_id[j] = (unsigned char)((source_word >> algorithm_shift) & 0x07u);
			}

			unsigned int new_values[RACEWAY_CAP_ILP];
			for (int j = 0; j < RACEWAY_CAP_ILP; j++)
			{
				const unsigned int val = working_code[j * 32 + int_index];
				const unsigned char aid = algorithm_id[j];
				unsigned int temp = val;
				if (alive_local[j] == 0u)
				{
					new_values[j] = temp;
					continue;
				}
				if (aid == 0u)
				{
					temp = ((val << 1) & 0xFEFEFEFEu) | *(__global unsigned int*)(offset_alg0 + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					rng_offset[j] += 128u;
				}
				else if (aid == 1u)
				{
					const unsigned int r = *(__global unsigned int*)(offset_regular + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					temp = ((val & 0x00FF00FFu) + (r & 0x00FF00FFu)) & 0x00FF00FFu;
					temp |= ((val & 0xFF00FF00u) + (r & 0xFF00FF00u)) & 0xFF00FF00u;
					rng_offset[j] += 128u;
				}
				else if (aid == 3u)
				{
					temp = val ^ *(__global unsigned int*)(offset_regular + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					rng_offset[j] += 128u;
				}
				else if (aid == 4u)
				{
					const unsigned int r = *(__global unsigned int*)(offset_regular + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					temp = ((val & 0x00FF00FFu) + ((r & 0x00FF00FFu) ^ 0x00FF00FFu) + 0x00010001u) & 0x00FF00FFu;
					temp |= ((val & 0xFF00FF00u) + ((r & 0xFF00FF00u) ^ 0xFF00FF00u) + 0x01000100u) & 0xFF00FF00u;
					rng_offset[j] += 128u;
				}
				else if (aid == 6u)
				{
					temp = ((val >> 1) & 0x7F7F7F7Fu) | *(__global unsigned int*)(offset_alg6 + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					rng_offset[j] += 128u;
				}
				else if (aid == 7u)
				{
					temp = val ^ 0xFFFFFFFFu;
				}
				else if (aid == 2u)
				{
					unsigned int t = (val & 0x00010000u) >> 8;
					if (int_index == 31u) t |= offset_alg2[sched * 2048u + rng_offset[j]];
					else { const unsigned int nlv = working_code[j * 32 + int_index + 1u]; t |= ((nlv & 0x00000001u) << 24); }
					t |= (val >> 1) & 0x007F007Fu; t |= (val << 1) & 0xFE00FE00u; t |= (val >> 8) & 0x00800080u;
					temp = t; rng_offset[j] += 1u;
				}
				else
				{
					unsigned int t = (val & 0x00800000u) >> 8;
					if (int_index == 31u) t |= offset_alg5[sched * 2048u + rng_offset[j]];
					else { const unsigned int nlv = working_code[j * 32 + int_index + 1u]; t |= ((nlv & 0x00000080u) << 24); }
					t |= (val >> 1) & 0x7F007F00u; t |= (val << 1) & 0x00FE00FEu; t |= (val >> 8) & 0x00010001u;
					temp = t; rng_offset[j] += 1u;
				}
				new_values[j] = temp;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			for (int j = 0; j < RACEWAY_CAP_ILP; j++) working_code[j * 32 + int_index] = new_values[j];
			barrier(CLK_LOCAL_MEM_FENCE);
			nibble_selector = (unsigned short)(nibble_selector << 1);
		}
	}
#endif
}

__kernel void raceway_boundary_cap_state_offset(
	__global unsigned char* alive_out,
	__global unsigned char* drop_map_out,
	__global unsigned int* state,
	volatile __global raceway_cap_word_t* cap_table,
	unsigned int cap_bits,
	unsigned int cap_ways,
	unsigned int cap_index,
	__global unsigned char* offset_regular,
	__global unsigned char* offset_alg0,
	__global unsigned char* offset_alg6,
	__global unsigned int* offset_alg2,
	__global unsigned int* offset_alg5,
	__global unsigned char* expansion_values,
	__global unsigned char* schedule_data,
	unsigned int key,
	unsigned int data_start,
	unsigned int candidate_count,
	unsigned int end_map,
	unsigned int precert_fixed_value,
	unsigned int precert_support_mask,
	unsigned int use_precert)
{
	__local unsigned int working_code[32 * RACEWAY_CAP_ILP];
	__local unsigned char alive_local[RACEWAY_CAP_ILP];
#ifdef RACEWAY_LDS_RNG
	__local unsigned int staged_regular[STAGE_ROWS * 32];
#endif
	const unsigned int int_index = get_local_id(0);
	const unsigned int base = get_global_id(1) * RACEWAY_CAP_ILP;

	if (int_index == 0u)
	{
		for (int j = 0; j < RACEWAY_CAP_ILP; j++)
		{
			const unsigned int c = base + (unsigned int)j;
			alive_local[j] = (c < candidate_count) ? 1u : 0u;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int j = 0; j < RACEWAY_CAP_ILP; j++)
	{
		const unsigned int logical = data_start + base + (unsigned int)j;
		const unsigned int cur_data = raceway_precert_data_value(
			logical, precert_fixed_value, precert_support_mask, use_precert);
		working_code[j * 32 + int_index] = ((int_index & 1u) == 0u) ? pack_be_u32(key) : pack_be_u32(cur_data);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int j = 0; j < RACEWAY_CAP_ILP; j++)
		expand(working_code + j * 32, (int)int_index, (unsigned short)(key >> 16), expansion_values);
	barrier(CLK_LOCAL_MEM_FENCE);

	raceway_run_map_range_local(working_code, alive_local, int_index, 0u, end_map,
		offset_regular, offset_alg0, offset_alg6, offset_alg2, offset_alg5, schedule_data
#ifdef RACEWAY_LDS_RNG
		, staged_regular
#endif
		);

	if (int_index == 0u)
	{
		const ulong cap_slots_per_table = ((cap_bits == 0u) ? 1UL : (1UL << cap_bits)) * (ulong)cap_ways;
		volatile __global raceway_cap_word_t* cap_table_for_map = cap_table + (ulong)cap_index * cap_slots_per_table;
		for (int j = 0; j < RACEWAY_CAP_ILP; j++)
		{
			const unsigned int c = base + (unsigned int)j;
			if (c >= candidate_count) continue;
			const raceway_cap_word_t fp = raceway_state_fingerprint(working_code + j * 32);
			if (raceway_cap_probe_or_keep(fp, cap_table_for_map, cap_bits, cap_ways))
			{
				alive_local[j] = 0u;
				if (drop_map_out != 0) drop_map_out[c] = (unsigned char)end_map;
			}
			alive_out[c] = alive_local[j];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int j = 0; j < RACEWAY_CAP_ILP; j++)
	{
		const unsigned int c = base + (unsigned int)j;
		if (c < candidate_count && alive_local[j] != 0u)
		{
			state[(size_t)c * 32u + int_index] = working_code[j * 32 + int_index];
		}
	}
}

__kernel void raceway_span_state_liveidx_cap_offset(
	__global unsigned int* live_idx,
	unsigned int live_count,
	__global unsigned int* state,
	__global unsigned char* alive_out,
	__global unsigned char* drop_map_out,
	volatile __global raceway_cap_word_t* cap_table,
	unsigned int cap_bits,
	unsigned int cap_ways,
	unsigned int cap_index,
	__global unsigned char* offset_regular,
	__global unsigned char* offset_alg0,
	__global unsigned char* offset_alg6,
	__global unsigned int* offset_alg2,
	__global unsigned int* offset_alg5,
	__global unsigned char* schedule_data,
	unsigned int start_map,
	unsigned int end_map)
{
	__local unsigned int working_code[32 * RACEWAY_CAP_ILP];
	__local unsigned char alive_local[RACEWAY_CAP_ILP];
	__local unsigned int orig_local[RACEWAY_CAP_ILP];
#ifdef RACEWAY_LDS_RNG
	__local unsigned int staged_regular[STAGE_ROWS * 32];
#endif
	const unsigned int int_index = get_local_id(0);
	const unsigned int base = get_global_id(1) * RACEWAY_CAP_ILP;

	if (int_index == 0u)
	{
		for (int j = 0; j < RACEWAY_CAP_ILP; j++)
		{
			const unsigned int s = base + (unsigned int)j;
			alive_local[j] = (s < live_count) ? 1u : 0u;
			orig_local[j] = (s < live_count) ? live_idx[s] : 0u;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int j = 0; j < RACEWAY_CAP_ILP; j++)
		working_code[j * 32 + int_index] = state[(size_t)orig_local[j] * 32u + int_index];
	barrier(CLK_LOCAL_MEM_FENCE);

	raceway_run_map_range_local(working_code, alive_local, int_index, start_map, end_map,
		offset_regular, offset_alg0, offset_alg6, offset_alg2, offset_alg5, schedule_data
#ifdef RACEWAY_LDS_RNG
		, staged_regular
#endif
		);

	if (int_index == 0u)
	{
		const ulong cap_slots_per_table = ((cap_bits == 0u) ? 1UL : (1UL << cap_bits)) * (ulong)cap_ways;
		volatile __global raceway_cap_word_t* cap_table_for_map = cap_table + (ulong)cap_index * cap_slots_per_table;
		for (int j = 0; j < RACEWAY_CAP_ILP; j++)
		{
			const unsigned int s = base + (unsigned int)j;
			if (s >= live_count) continue;
			const raceway_cap_word_t fp = raceway_state_fingerprint(working_code + j * 32);
			if (raceway_cap_probe_or_keep(fp, cap_table_for_map, cap_bits, cap_ways))
			{
				alive_local[j] = 0u;
				if (drop_map_out != 0) drop_map_out[orig_local[j]] = (unsigned char)end_map;
			}
			alive_out[s] = alive_local[j];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int j = 0; j < RACEWAY_CAP_ILP; j++)
	{
		const unsigned int s = base + (unsigned int)j;
		if (s < live_count && alive_local[j] != 0u)
		{
			state[(size_t)orig_local[j] * 32u + int_index] = working_code[j * 32 + int_index];
		}
	}
}

__kernel void run_span_dedup(
	__global unsigned int*   live_idx,        // ignored when first_span != 0
	unsigned int             M,
	__global unsigned int*   state,           // [N*32]
	__global unsigned char*  alive_out,       // [M]
	__global unsigned int*   rep_global,      // [N] union-find parent (0xFFFFFFFF = root)
	unsigned int m0, unsigned int m1, int first_span,
	__global unsigned char*  offset_regular,
	__global unsigned char*  offset_alg0,
	__global unsigned char*  offset_alg6,
	__global unsigned int*   offset_alg2,
	__global unsigned int*   offset_alg5,
	__global unsigned char*  expansion_values,
	__global unsigned char*  schedule_data,
	__global unsigned char*  carnival_data,
	__global unsigned char*  flag_out,        // mode 2 only
	unsigned int key, unsigned int data_start,
	int mode)                                  // 0=writeback, 1=dedup, 2=screen+flag
{
	// Single-warp geometry (work-group = 32 lanes, SPAN_ILP candidates, W = SPAN_ILP).
	// Only lane 0 does the per-span dedup inserts sequentially -> no atomics. The
	// multi-warp (W>SPAN_ILP) variant was tested and is SLOWER on OpenCL: its
	// work-group-wide barriers cost more than the larger dedup window saves
	// (unlike CUDA, whose barrier-free warp shuffles made W=64 the optimum).
	__local unsigned int  working_code[32 * SPAN_ILP];
	__local unsigned int  decrypted_carnival[32];
	__local unsigned int  decrypted_other[32];
	__local unsigned int  orig_local[SPAN_ILP];
	__local unsigned char alive_local[SPAN_ILP];
	__local unsigned char rep_local[SPAN_ILP];
	__local unsigned int  slot_fp[SPAN_NSLOT];
	__local unsigned char slot_rep[SPAN_NSLOT];

	const unsigned int int_index = get_local_id(0);
	const unsigned int wg_index  = get_global_id(1);
	const unsigned int base      = wg_index * SPAN_ILP;

	if (int_index == 0u)
	{
		for (int j = 0; j < SPAN_ILP; j++)
		{
			const unsigned int s = base + (unsigned int)j;
			const int active = (s < M);
			orig_local[j]  = active ? (first_span ? s : live_idx[s]) : 0xFFFFFFFFu;
			alive_local[j] = active ? 1u : 0u;
			rep_local[j]   = (unsigned char)j;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int j = 0; j < SPAN_ILP; j++)
	{
		if (alive_local[j])
		{
			if (first_span)
			{
				const unsigned int cur_data = data_start + orig_local[j];
				working_code[j * 32 + int_index] = ((int_index & 1u) == 0u) ? pack_be_u32(key) : pack_be_u32(cur_data);
			}
			else
			{
				working_code[j * 32 + int_index] = state[(size_t)orig_local[j] * 32u + int_index];
			}
		}
		else
		{
			working_code[j * 32 + int_index] = 0u;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (first_span)
	{
		for (int j = 0; j < SPAN_ILP; j++)
			expand(working_code + j * 32, (int)int_index, (unsigned short)(key >> 16), expansion_values);
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	for (unsigned int sched = m0; sched < m1; sched++)
	{
		unsigned int rng_offset[SPAN_ILP];
		for (int j = 0; j < SPAN_ILP; j++) rng_offset[j] = 0u;

		const unsigned int packed_sched_b2 = (unsigned int)schedule_data[sched * 4 + 2];
		const unsigned int packed_sched_b3 = (unsigned int)schedule_data[sched * 4 + 3];
		unsigned short nibble_selector = (unsigned short)((packed_sched_b2 << 8) | packed_sched_b3);

		for (int i = 0; i < 16; i++)
		{
			const unsigned int source_lane  = (unsigned int)(i >> 2);
			const unsigned int algorithm_shift = (unsigned int)((i & 3) * 8) + 1u + (((unsigned int)nibble_selector >> 13u) & 4u);

			unsigned char algorithm_id[SPAN_ILP];
			for (int j = 0; j < SPAN_ILP; j++)
			{
				const unsigned int source_word = working_code[j * 32 + source_lane];
				algorithm_id[j] = (unsigned char)((source_word >> algorithm_shift) & 0x07u);
			}

			unsigned int new_values[SPAN_ILP];
			for (int j = 0; j < SPAN_ILP; j++)
			{
				const unsigned int val = working_code[j * 32 + int_index];
				const unsigned char aid = algorithm_id[j];
				unsigned int temp = val;
				if (aid == 0u)
				{
					temp = ((val << 1) & 0xFEFEFEFEu) | *(__global unsigned int*)(offset_alg0 + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					rng_offset[j] += 128u;
				}
				else if (aid == 1u)
				{
					const unsigned int r = *(__global unsigned int*)(offset_regular + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					temp = ((val & 0x00FF00FFu) + (r & 0x00FF00FFu)) & 0x00FF00FFu;
					temp |= ((val & 0xFF00FF00u) + (r & 0xFF00FF00u)) & 0xFF00FF00u;
					rng_offset[j] += 128u;
				}
				else if (aid == 3u)
				{
					temp = val ^ *(__global unsigned int*)(offset_regular + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					rng_offset[j] += 128u;
				}
				else if (aid == 4u)
				{
					const unsigned int r = *(__global unsigned int*)(offset_regular + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					temp = ((val & 0x00FF00FFu) + ((r & 0x00FF00FFu) ^ 0x00FF00FFu) + 0x00010001u) & 0x00FF00FFu;
					temp |= ((val & 0xFF00FF00u) + ((r & 0xFF00FF00u) ^ 0xFF00FF00u) + 0x01000100u) & 0xFF00FF00u;
					rng_offset[j] += 128u;
				}
				else if (aid == 6u)
				{
					temp = ((val >> 1) & 0x7F7F7F7Fu) | *(__global unsigned int*)(offset_alg6 + ((sched * 2048u + rng_offset[j]) * 128u + int_index * 4u));
					rng_offset[j] += 128u;
				}
				else if (aid == 7u)
				{
					temp = val ^ 0xFFFFFFFFu;
				}
				else if (aid == 2u)
				{
					unsigned int t = (val & 0x00010000u) >> 8;
					if (int_index == 31u) t |= offset_alg2[sched * 2048u + rng_offset[j]];
					else { const unsigned int nlv = working_code[j * 32 + int_index + 1u]; t |= ((nlv & 0x00000001u) << 24); }
					t |= (val >> 1) & 0x007F007Fu; t |= (val << 1) & 0xFE00FE00u; t |= (val >> 8) & 0x00800080u;
					temp = t; rng_offset[j] += 1u;
				}
				else /* aid == 5 */
				{
					unsigned int t = (val & 0x00800000u) >> 8;
					if (int_index == 31u) t |= offset_alg5[sched * 2048u + rng_offset[j]];
					else { const unsigned int nlv = working_code[j * 32 + int_index + 1u]; t |= ((nlv & 0x00000080u) << 24); }
					t |= (val >> 1) & 0x7F007F00u; t |= (val << 1) & 0x00FE00FEu; t |= (val >> 8) & 0x00010001u;
					temp = t; rng_offset[j] += 1u;
				}
				new_values[j] = temp;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			for (int j = 0; j < SPAN_ILP; j++) working_code[j * 32 + int_index] = new_values[j];
			barrier(CLK_LOCAL_MEM_FENCE);
			nibble_selector = (unsigned short)(nibble_selector << 1);
		}
	}

	if (mode == 1)
	{
		if (int_index == 0u)
		{
			for (int sIdx = 0; sIdx < SPAN_NSLOT; sIdx++) slot_fp[sIdx] = 0u;
			for (int j = 0; j < SPAN_ILP; j++)
			{
				if (!alive_local[j]) continue;
				const unsigned int fp = span_fingerprint(working_code + j * 32);
				unsigned int idx = (fp >> 8) & (SPAN_NSLOT - 1u);
				for (int probe = 0; probe < SPAN_NSLOT; probe++)
				{
					if (slot_fp[idx] == 0u) { slot_fp[idx] = fp; slot_rep[idx] = (unsigned char)j; break; }
					if (slot_fp[idx] == fp) { rep_local[j] = slot_rep[idx]; alive_local[j] = 0u; break; }
					idx = (idx + 1u) & (SPAN_NSLOT - 1u);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (mode == 2)
	{
		for (int j = 0; j < SPAN_ILP; j++)
		{
			decrypted_carnival[int_index] = working_code[j * 32 + int_index] ^ ((__global unsigned int*)carnival_data)[int_index];
			((__local unsigned char*)decrypted_other)[int_index * 4 + 0] = ((__local unsigned char*)(working_code + j * 32))[int_index * 4 + 0] ^ other_world_data_k[int_index * 4 + 0];
			((__local unsigned char*)decrypted_other)[int_index * 4 + 1] = ((__local unsigned char*)(working_code + j * 32))[int_index * 4 + 1] ^ other_world_data_k[int_index * 4 + 1];
			((__local unsigned char*)decrypted_other)[int_index * 4 + 2] = ((__local unsigned char*)(working_code + j * 32))[int_index * 4 + 2] ^ other_world_data_k[int_index * 4 + 2];
			((__local unsigned char*)decrypted_other)[int_index * 4 + 3] = ((__local unsigned char*)(working_code + j * 32))[int_index * 4 + 3] ^ other_world_data_k[int_index * 4 + 3];
			barrier(CLK_LOCAL_MEM_FENCE);
			if (int_index == 0u && (base + (unsigned int)j) < M)
			{
				unsigned char f = 0;
				if (checksum_ok(decrypted_carnival, 0x72)) f = CHECKSUM_SENTINEL;
				else if (checksum_ok(decrypted_other, 0x53)) f = CHECKSUM_SENTINEL | OTHER_WORLD;
				flag_out[orig_local[j]] = f;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		return;
	}

	for (int j = 0; j < SPAN_ILP; j++)
	{
		const unsigned int s = base + (unsigned int)j;
		if (s >= M) continue;
		if (alive_local[j])
		{
			state[(size_t)orig_local[j] * 32u + int_index] = working_code[j * 32 + int_index];
			if (int_index == 0u) alive_out[s] = 1u;
		}
		else if (int_index == 0u)
		{
			unsigned int cur = (unsigned int)j;
			// Union-find path compression: the chain reaches a root in ~1-2 hops at
			// runtime, so do NOT fully unroll the SPAN_ILP-bounded walk — that only
			// bloats the kernel (the unrolled tail is branched over). Re-roll it.
			#pragma unroll 1
			for (int hop = 0; hop < SPAN_ILP; hop++) { unsigned int r = rep_local[cur]; if (r == cur || r >= SPAN_ILP) break; cur = r; }
			rep_global[orig_local[j]] = orig_local[cur];
			alive_out[s] = 0u;
		}
	}
}



// Block-stable ordered compaction: local prefix scan + one global atomic_add per
// work-group for the base offset, ordered writes. 1-D launch.
__kernel void compact_survivors_ordered(
	__global unsigned char*  alive_in,
	__global unsigned int*   live_idx_in,
	unsigned int             M,
	__global unsigned int*   out_live,
	__global unsigned int*   counter,
	int first_span)
{
	__local unsigned int s_scan[256];
	__local unsigned int s_base;
	const unsigned int gid = get_global_id(0);
	const unsigned int lid = get_local_id(0);
	const unsigned int lsz = get_local_size(0);
	const unsigned int a = (gid < M && alive_in[gid] != 0u) ? 1u : 0u;
	s_scan[lid] = a;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (unsigned int off = 1u; off < lsz; off <<= 1)
	{
		unsigned int v = (lid >= off) ? s_scan[lid - off] : 0u;
		barrier(CLK_LOCAL_MEM_FENCE);
		s_scan[lid] += v;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (lid == lsz - 1u) s_base = atomic_add(counter, s_scan[lid]);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (a)
	{
		const unsigned int pos = s_base + s_scan[lid] - 1u;
		out_live[pos] = first_span ? gid : live_idx_in[gid];
	}
}

// Resolve every candidate's flag via the union-find chain (0xFFFFFFFF = root).
__kernel void resolve_flags(
	__global unsigned int*  rep_global,
	__global unsigned char* flag,
	__global unsigned char* result,
	unsigned int N)
{
	const unsigned int c = get_global_id(0);
	if (c >= N) return;
	unsigned int root = c;
	// Same union-find walk as run_span_dedup: a root resolves in ~1-2 hops, so the
	// 64-bound loop should stay rolled rather than unrolling 64× of dead tail.
	#pragma unroll 1
	for (unsigned int hop = 0u; hop < 64u; hop++) { unsigned int r = rep_global[root]; if (r == 0xFFFFFFFFu) break; root = r; }
	result[c] = flag[root];
}
