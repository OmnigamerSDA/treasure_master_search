// Treasure Master — OpenCL kernels for the forward search.
//
// Kernels:
//   tm_process              — single-step algorithm dispatch (alg 0..7)
//   test_expand             — IV expansion (8B → 128B via RNG accumulation)
//   test_alg                — standalone algorithm test harness
//   full_process            — end-to-end (expand + schedule + alg loop)
//   tm_stats                — kernel-side machine-flag stats accumulator
//   tm_checksum_screen      — fast checksum-only screen (forward fair filter)
//   tm_materialize_survivors — emit survivor records to host buffer
//
// Each algorithm case (0..7) corresponds to one byte op in the schedule.
// Cases 2 and 5 require cross-lane carries; carry-ins for the last lane
// come from precomputed alg2_values / alg5_values tables.

#define reverse_offset(x) (127 - (x))

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







