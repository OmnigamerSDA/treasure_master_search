#ifndef RNG_H
#define RNG_H

#include "data_sizes.h"

void generate_rng_table(uint16 * rng_table);
uint8 run_rng(uint16 * rng_seed, uint16 * rng_table);

void generate_regular_rng_values_8(uint8 * regular_rng_values, uint16 * rng_table);
void generate_regular_rng_values_16(uint16 * regular_rng_values, uint16 * rng_table);

void generate_regular_rng_values_8_lo(uint8 * regular_rng_values_lo, uint16 * rng_table);
void generate_regular_rng_values_8_hi(uint8 * regular_rng_values_hi, uint16 * rng_table);

void generate_alg0_values_8(uint8 * alg0_values, uint16 * rng_table);
void generate_alg0_values_16(uint16 * alg0_values, uint16 * rng_table);

void generate_alg6_values_8(uint8 * alg6_values, uint16 * rng_table);
void generate_alg6_values_16(uint16 * alg6_values, uint16 * rng_table);

void generate_alg4_values_8(uint8 * alg4_values, uint16 * rng_table);
void generate_alg4_values_8_lo(uint8 * alg4_values_lo, uint16 * rng_table);
void generate_alg4_values_8_hi(uint8 * alg4_values_hi, uint16 * rng_table);

void generate_alg4_values_16(uint16 * alg4_values, uint16 * rng_table);

void generate_alg2_values_8_8(uint8 * alg2_values, uint16 * rng_table);
void generate_alg2_values_32_8(uint32 * alg2_values, uint16 * rng_table);
void generate_alg2_values_32_16(uint32 * alg2_values, uint16 * rng_table);
void generate_alg2_values_64_8(uint64 * alg2_values, uint16 * rng_table);
void generate_alg2_values_64_16(uint64 * alg2_values, uint16 * rng_table);
void generate_alg2_values_128_8(uint8 * alg2_values, uint16 * rng_table);
void generate_alg2_values_128_16(uint8 * alg2_values, uint16 * rng_table);
void generate_alg2_values_256_8(uint8 * alg2_values, uint16 * rng_table);
void generate_alg2_values_256_16(uint8 * alg2_values, uint16 * rng_table);

void generate_alg5_values_8_8(uint8 * alg5_values, uint16 * rng_table);
void generate_alg5_values_32_8(uint32 * alg5_values, uint16 * rng_table);
void generate_alg5_values_32_16(uint32 * alg5_values, uint16 * rng_table);
void generate_alg5_values_64_8(uint64 * alg5_values, uint16 * rng_table);
void generate_alg5_values_64_16(uint64 * alg5_values, uint16 * rng_table);
void generate_alg5_values_128_8(uint8 * alg5_values, uint16 * rng_table);
void generate_alg5_values_128_16(uint8 * alg5_values, uint16 * rng_table);
void generate_alg5_values_256_8(uint8 * alg5_values, uint16 * rng_table);
void generate_alg5_values_256_16(uint8 * alg5_values, uint16 * rng_table);

void generate_seed_forward_1(uint16 * values, uint16 * rng_table);
void generate_seed_forward_128(uint16 * values, uint16 * rng_table);

void generate_expansion_values_8(uint8* values, uint16* rng_table);

// Per-seed RNG-value tables, 2048 bytes per seed. Used by the map-mode forward
// kernels (e.g. tm_avx2_r256_map_8) which precompute one of these once per
// schedule entry and walk a moving pointer through it instead of indexing the
// 8 MB universal table.
//   - regular_rng_values_for_seed_8: raw run_rng bytes, written reversed
//     (out[2047-j] = j-th run_rng output). Consumed by alg 1/3/4 in 128-byte
//     chunks; also read 1 byte at a time for the carry bit in alg 2/5.
//   - alg0_values_for_seed_8: (run_rng() >> 7) & 0x01, reversed layout.
//   - alg6_values_for_seed_8: run_rng() & 0x80, forward layout (alg 6 walks the
//     buffer in the opposite direction).
void generate_regular_rng_values_for_seed_8(uint8* out, uint16 seed, uint16* rng_table);
void generate_alg0_values_for_seed_8(uint8* out, uint16 seed, uint16* rng_table);
void generate_alg6_values_for_seed_8(uint8* out, uint16 seed, uint16* rng_table);
#endif // RNG_H