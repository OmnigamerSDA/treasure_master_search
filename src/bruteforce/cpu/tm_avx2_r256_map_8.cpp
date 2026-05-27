#include <stdio.h>
#include <cstdlib>
#include <cstring>

#include "tm_avx2_r256_map_8.h"
#include "rng.h"

tm_avx2_r256_map_8::tm_avx2_r256_map_8(RNG* rng_obj)
    : TM_base(rng_obj),
      mask_FF(_mm256_set1_epi8(static_cast<int8_t>(0xFF))),
      mask_FE(_mm256_set1_epi8(static_cast<int8_t>(0xFE))),
      mask_7F(_mm256_set1_epi8(0x7F)),
      mask_80(_mm256_set1_epi8(static_cast<int8_t>(0x80))),
      mask_01(_mm256_set1_epi8(0x01)),
      mask_top_01(_mm256_set_epi16(0x0100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
      mask_top_80(_mm256_set_epi16(static_cast<int16_t>(0x8000), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
{
    initialize();
}

tm_avx2_r256_map_8::tm_avx2_r256_map_8(RNG* rng_obj, map_tables_shared::Tables* shared_tables)
    : tm_avx2_r256_map_8(rng_obj)
{
    _shared = shared_tables;
}

void tm_avx2_r256_map_8::initialize()
{
    if (!initialized)
    {
        rng->generate_rng_table();
        rng->generate_expansion_values_8();
        rng->generate_seed_forward_1();
        rng->generate_seed_forward_128();
        initialized = true;
    }
    obj_name = "tm_avx2_r256_map_8";
}

void tm_avx2_r256_map_8::_bind_key(uint32 new_key)
{
    if (new_key == _key && _expansion_for_seed_128 != nullptr) return;
    _key = new_key;
    const uint16 expansion_seed = static_cast<uint16>((new_key >> 16) & 0xFFFF);
    _expansion_for_seed_128 = rng->expansion_values_8 + static_cast<size_t>(expansion_seed) * 128;
}

void tm_avx2_r256_map_8::_bind_schedule(const key_schedule& schedule_entries)
{
    // Shared-tables mode: caller built the tables externally. Just refresh the
    // nibble_selectors view and entry-pointer base (cheap, must be a no-op for
    // perf-critical paths so the kernel sees stable shared state).
    if (_shared != nullptr) return;

    // Owned-tables mode: build/refresh per-instance.
    _owned.bind(rng, schedule_entries);
}

void tm_avx2_r256_map_8::bind_schedule(const key_schedule& schedule_entries)
{
    _bind_schedule(schedule_entries);
}

void tm_avx2_r256_map_8::bind_dedup_schedule(const key_schedule& /*schedule_entries*/)
{
    // Dedup calls run_maps_range() for every checkpoint group. In owned-table
    // mode that method binds only the requested range, avoiding an upfront
    // full-schedule table build for sparse checkpoint policies.
    if (_shared != nullptr)
    {
        // Shared-table mode still expects the caller-provided table to be bound.
        return;
    }
}

void tm_avx2_r256_map_8::bind_maps_range(const key_schedule& schedule_entries, std::size_t begin, std::size_t end)
{
    if (end > schedule_entries.entries.size())
        end = schedule_entries.entries.size();
    if (begin > end)
        begin = end;

    if (_shared == nullptr)
    {
        _owned.bind_range(rng, schedule_entries, begin, end);
    }
    else
    {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() ||
            t.entry_count != static_cast<int>(schedule_entries.entries.size()))
        {
            _bind_schedule(schedule_entries);
        }
    }
}

__forceinline void tm_avx2_r256_map_8::_load_from_mem(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3)
{
    wc0 = _mm256_load_si256((__m256i*)(working_code_data));
    wc1 = _mm256_load_si256((__m256i*)(working_code_data + 32));
    wc2 = _mm256_load_si256((__m256i*)(working_code_data + 64));
    wc3 = _mm256_load_si256((__m256i*)(working_code_data + 96));
}

__forceinline void tm_avx2_r256_map_8::_store_to_mem(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3)
{
    _mm256_store_si256((__m256i*)(working_code_data),      wc0);
    _mm256_store_si256((__m256i*)(working_code_data + 32), wc1);
    _mm256_store_si256((__m256i*)(working_code_data + 64), wc2);
    _mm256_store_si256((__m256i*)(working_code_data + 96), wc3);
}

void tm_avx2_r256_map_8::load_data(uint8* new_data)
{
    for (int i = 0; i < 128; i++) working_code_data[i] = new_data[i];
}

void tm_avx2_r256_map_8::fetch_data(uint8* new_data)
{
    for (int i = 0; i < 128; i++) new_data[i] = working_code_data[i];
}

__forceinline void tm_avx2_r256_map_8::_expand_code(uint32 data, __m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3)
{
    __m128i a = _mm_insert_epi32(_mm_cvtsi32_si128(static_cast<int32_t>(data)), static_cast<int32_t>(_key), 1);
    __m128i nat_mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7);
    __m128i pattern = _mm_shuffle_epi8(a, nat_mask);
    __m256i pat256 = _mm256_broadcastsi128_si256(pattern);

    wc0 = pat256; wc1 = pat256; wc2 = pat256; wc3 = pat256;

    const uint8* rng_start = _expansion_for_seed_128;
    wc0 = _mm256_add_epi8(wc0, _mm256_loadu_si256((const __m256i*)(rng_start)));
    wc1 = _mm256_add_epi8(wc1, _mm256_loadu_si256((const __m256i*)(rng_start + 32)));
    wc2 = _mm256_add_epi8(wc2, _mm256_loadu_si256((const __m256i*)(rng_start + 64)));
    wc3 = _mm256_add_epi8(wc3, _mm256_loadu_si256((const __m256i*)(rng_start + 96)));
}

void tm_avx2_r256_map_8::expand(uint32 key, uint32 data)
{
    _bind_key(key);

    __m256i wc0, wc1, wc2, wc3;
    _expand_code(data, wc0, wc1, wc2, wc3);
    _store_to_mem(wc0, wc1, wc2, wc3);
}

// alg_2_sub: right-rotate variant. carry bit is at byte 31, bit 0 (mask_top_01).
__forceinline void tm_avx2_r256_map_8::alg_2_sub(__m256i& wc, __m256i& carry)
{
    __m256i part_lo = _mm256_and_si256(_mm256_srli_epi16(wc, 1), _mm256_set1_epi16(0x007F));
    __m256i part_hi_bit = _mm256_and_si256(_mm256_srli_epi16(wc, 8), _mm256_set1_epi16(0x0080));
    __m256i new_lo = _mm256_or_si256(part_lo, part_hi_bit);

    __m256i low_bit0 = _mm256_and_si256(wc, _mm256_set1_epi16(0x0001));
    __m256i low_bit0_up = _mm256_slli_epi16(low_bit0, 8);
    __m256i carry_in = _mm256_or_si256(
        _mm256_alignr_epi8(
            _mm256_permute2x128_si256(low_bit0_up, _mm256_setzero_si256(), 0x81),
            low_bit0_up,
            2),
        carry);

    __m256i new_hi = _mm256_or_si256(
        _mm256_and_si256(_mm256_slli_epi16(wc, 1), _mm256_set1_epi16(static_cast<int16_t>(0xFE00))),
        carry_in);

    __m128i lo_lane = _mm256_castsi256_si128(_mm256_and_si256(wc, _mm256_set1_epi16(0x0001)));
    __m128i lo_shifted = _mm_slli_si128(lo_lane, 15);
    carry = _mm256_and_si256(
        _mm256_inserti128_si256(_mm256_setzero_si256(), lo_shifted, 1),
        mask_top_01);

    wc = _mm256_or_si256(
        _mm256_and_si256(new_lo, _mm256_set1_epi16(0x00FF)),
        _mm256_and_si256(new_hi, _mm256_set1_epi16(static_cast<int16_t>(0xFF00))));
}

// alg_5_sub: left-rotate variant. carry bit is at byte 31, bit 7 (mask_top_80).
__forceinline void tm_avx2_r256_map_8::alg_5_sub(__m256i& wc, __m256i& carry)
{
    __m256i part_lo = _mm256_and_si256(_mm256_slli_epi16(wc, 1), _mm256_set1_epi16(0x00FE));
    __m256i part_lo_bit = _mm256_and_si256(_mm256_srli_epi16(wc, 8), _mm256_set1_epi16(0x0001));
    __m256i new_lo = _mm256_or_si256(part_lo, part_lo_bit);

    __m256i low_bit7 = _mm256_and_si256(wc, _mm256_set1_epi16(0x0080));
    __m256i low_bit7_up = _mm256_slli_epi16(low_bit7, 8);
    __m256i carry_in = _mm256_or_si256(
        _mm256_alignr_epi8(
            _mm256_permute2x128_si256(low_bit7_up, _mm256_setzero_si256(), 0x81),
            low_bit7_up,
            2),
        carry);

    __m256i new_hi = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi16(wc, 1), _mm256_set1_epi16(static_cast<int16_t>(0x7F00))),
        carry_in);

    __m128i lo_lane = _mm256_castsi256_si128(_mm256_and_si256(wc, _mm256_set1_epi16(0x0080)));
    __m128i lo_shifted = _mm_slli_si128(lo_lane, 15);
    carry = _mm256_and_si256(
        _mm256_inserti128_si256(_mm256_setzero_si256(), lo_shifted, 1),
        mask_top_80);

    wc = _mm256_or_si256(
        _mm256_and_si256(new_lo, _mm256_set1_epi16(0x00FF)),
        _mm256_and_si256(new_hi, _mm256_set1_epi16(static_cast<int16_t>(0xFF00))));
}

__forceinline void tm_avx2_r256_map_8::alg_0(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, const uint8* block_start)
{
    __m256i rng_val = _mm256_loadu_si256((const __m256i*)(block_start));
    wc0 = _mm256_or_si256(_mm256_and_si256(_mm256_slli_epi16(wc0, 1), mask_FE), rng_val);
    rng_val = _mm256_loadu_si256((const __m256i*)(block_start + 32));
    wc1 = _mm256_or_si256(_mm256_and_si256(_mm256_slli_epi16(wc1, 1), mask_FE), rng_val);
    rng_val = _mm256_loadu_si256((const __m256i*)(block_start + 64));
    wc2 = _mm256_or_si256(_mm256_and_si256(_mm256_slli_epi16(wc2, 1), mask_FE), rng_val);
    rng_val = _mm256_loadu_si256((const __m256i*)(block_start + 96));
    wc3 = _mm256_or_si256(_mm256_and_si256(_mm256_slli_epi16(wc3, 1), mask_FE), rng_val);
}

__forceinline void tm_avx2_r256_map_8::alg_1(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, const uint8* block_start)
{
    wc0 = _mm256_add_epi8(wc0, _mm256_loadu_si256((const __m256i*)(block_start)));
    wc1 = _mm256_add_epi8(wc1, _mm256_loadu_si256((const __m256i*)(block_start + 32)));
    wc2 = _mm256_add_epi8(wc2, _mm256_loadu_si256((const __m256i*)(block_start + 64)));
    wc3 = _mm256_add_epi8(wc3, _mm256_loadu_si256((const __m256i*)(block_start + 96)));
}

__forceinline void tm_avx2_r256_map_8::alg_2(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, uint8 carry_byte)
{
    __m256i carry = _mm256_and_si256(_mm256_set1_epi8(static_cast<int8_t>(carry_byte)), mask_top_01);
    alg_2_sub(wc3, carry);
    alg_2_sub(wc2, carry);
    alg_2_sub(wc1, carry);
    alg_2_sub(wc0, carry);
}

__forceinline void tm_avx2_r256_map_8::alg_3(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, const uint8* block_start)
{
    wc0 = _mm256_xor_si256(wc0, _mm256_loadu_si256((const __m256i*)(block_start)));
    wc1 = _mm256_xor_si256(wc1, _mm256_loadu_si256((const __m256i*)(block_start + 32)));
    wc2 = _mm256_xor_si256(wc2, _mm256_loadu_si256((const __m256i*)(block_start + 64)));
    wc3 = _mm256_xor_si256(wc3, _mm256_loadu_si256((const __m256i*)(block_start + 96)));
}

__forceinline void tm_avx2_r256_map_8::alg_4(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, const uint8* block_start)
{
    wc0 = _mm256_sub_epi8(wc0, _mm256_loadu_si256((const __m256i*)(block_start)));
    wc1 = _mm256_sub_epi8(wc1, _mm256_loadu_si256((const __m256i*)(block_start + 32)));
    wc2 = _mm256_sub_epi8(wc2, _mm256_loadu_si256((const __m256i*)(block_start + 64)));
    wc3 = _mm256_sub_epi8(wc3, _mm256_loadu_si256((const __m256i*)(block_start + 96)));
}

__forceinline void tm_avx2_r256_map_8::alg_5(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, uint8 carry_byte)
{
    __m256i carry = _mm256_and_si256(_mm256_set1_epi8(static_cast<int8_t>(carry_byte)), mask_top_80);
    alg_5_sub(wc3, carry);
    alg_5_sub(wc2, carry);
    alg_5_sub(wc1, carry);
    alg_5_sub(wc0, carry);
}

__forceinline void tm_avx2_r256_map_8::alg_6(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, const uint8* block_start)
{
    __m256i rng_val = _mm256_loadu_si256((const __m256i*)(block_start));
    wc0 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(wc0, 1), mask_7F), rng_val);
    rng_val = _mm256_loadu_si256((const __m256i*)(block_start + 32));
    wc1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(wc1, 1), mask_7F), rng_val);
    rng_val = _mm256_loadu_si256((const __m256i*)(block_start + 64));
    wc2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(wc2, 1), mask_7F), rng_val);
    rng_val = _mm256_loadu_si256((const __m256i*)(block_start + 96));
    wc3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(wc3, 1), mask_7F), rng_val);
}

__forceinline void tm_avx2_r256_map_8::alg_7(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3)
{
    wc0 = _mm256_xor_si256(wc0, mask_FF);
    wc1 = _mm256_xor_si256(wc1, mask_FF);
    wc2 = _mm256_xor_si256(wc2, mask_FF);
    wc3 = _mm256_xor_si256(wc3, mask_FF);
}

__forceinline void tm_avx2_r256_map_8::xor_alg(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, const uint8* values)
{
    wc0 = _mm256_xor_si256(wc0, _mm256_load_si256((__m256i*)(values)));
    wc1 = _mm256_xor_si256(wc1, _mm256_load_si256((__m256i*)(values + 32)));
    wc2 = _mm256_xor_si256(wc2, _mm256_load_si256((__m256i*)(values + 64)));
    wc3 = _mm256_xor_si256(wc3, _mm256_load_si256((__m256i*)(values + 96)));
}

__forceinline void tm_avx2_r256_map_8::_run_alg(
    __m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3,
    int algorithm_id, uint16* local_pos,
    const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base)
{
    // Switch with consecutive integer cases gives gcc the option of emitting
    // a jump table — handled via an indirect branch through .rodata, predicted
    // by the BTB instead of an if/else cascade. On Zen 5 the BTB is
    // 2K-entry/8-way which handles tight indirect dispatch well.
    switch (algorithm_id)
    {
        case 0:
            alg_0(wc0, wc1, wc2, wc3, alg0_base + *local_pos - 127);
            *local_pos -= 128;
            break;
        case 1:
            alg_1(wc0, wc1, wc2, wc3, reg_base + *local_pos - 127);
            *local_pos -= 128;
            break;
        case 2:
            alg_2(wc0, wc1, wc2, wc3, reg_base[*local_pos] >> 7);
            *local_pos -= 1;
            break;
        case 3:
            alg_3(wc0, wc1, wc2, wc3, reg_base + *local_pos - 127);
            *local_pos -= 128;
            break;
        case 4:
            alg_4(wc0, wc1, wc2, wc3, reg_base + *local_pos - 127);
            *local_pos -= 128;
            break;
        case 5:
            alg_5(wc0, wc1, wc2, wc3, reg_base[*local_pos] & 0x80);
            *local_pos -= 1;
            break;
        case 6:
            alg_6(wc0, wc1, wc2, wc3, alg6_base + (2047 - *local_pos));
            *local_pos -= 128;
            break;
        default: // case 7 — already & 0x07 in caller, so 7 is the only remaining value
            alg_7(wc0, wc1, wc2, wc3);
            break;
    }
}

__forceinline void tm_avx2_r256_map_8::_run_one_map(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 local_pos = 2047;

    // Natural layout: byte i (i in 0..15) lives at wc0 byte i. Read it directly
    // out of the register via _mm_extract_epi8 (requires compile-time index, so
    // we unroll). This avoids upstream's store-spill-and-scalar-load round-trip
    // — the same optimization tm_avx2_r256s_8 uses for its dispatch byte.
#pragma GCC unroll 16
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;

        uint8 current_byte = static_cast<uint8>(_mm_extract_epi8(_mm256_castsi256_si128(wc0), i));
        if (nibble == 1) current_byte >>= 4;
        uint8 algorithm_id = static_cast<uint8>((current_byte >> 1) & 0x07);

        _run_alg(wc0, wc1, wc2, wc3, algorithm_id, &local_pos, reg_base, alg0_base, alg6_base);
    }
}

__forceinline void tm_avx2_r256_map_8::_run_maps_fixed(
    __m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3,
    int map_idx, int count)
{
    // Map-count dispatch. We keep only the count=1 specialization (used by
    // the first-dedup-maps=1 group) and fall through to the loop for all
    // other spans.
    //
    // History: an earlier version specialized counts 1,2,3,4,6,7,9 as fully
    // inlined sequences of `_run_one_map` calls. Each `_run_one_map` already
    // unrolls its 16-iter inner loop (#pragma GCC unroll 16), so a 3-map
    // span produced 3 × 16 × 8-alg-switch = ~384 inlined dispatch sites in
    // one function. `perf stat -e L1-icache-load-misses` showed 3.5 billion
    // i-cache misses on the K=3 case vs 28 million on K=5 (which fell
    // through to the loop) — a 125× difference that pushed frontend stalls
    // from 28% (loop) to 44% (unrolled). The loop reuses the same
    // instruction range across iterations, so the i-cache stays hot.
    if (count == 1)
        _run_one_map(wc0, wc1, wc2, wc3, map_idx);
    else
        for (int i = 0; i < count; i++)
            _run_one_map(wc0, wc1, wc2, wc3, map_idx + i);
}

void tm_avx2_r256_map_8::run_alg(int /*algorithm_id*/, uint16* /*rng_seed*/, int /*iterations*/)
{
    // Not used by bench_cpu / run_bruteforce_data. Left unimplemented for the port.
}

__forceinline void tm_avx2_r256_map_8::_run_all_maps(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3)
{
    const int n = _t().entry_count;
    for (int map_idx = 0; map_idx < n; map_idx++)
        _run_one_map(wc0, wc1, wc2, wc3, map_idx);
}

void tm_avx2_r256_map_8::run_one_map(const key_schedule::key_schedule_entry& schedule_entry)
{
    // Fast path: caller called bind_schedule() with the parent key_schedule, so
    // the entry's address falls inside the bound entries[] array and the index
    // is one subtraction. This is the dedup path — state_dedup binds once then
    // iterates entries, so this is the hot loop.
    int map_idx = -1;
    const auto& t = _t();
    if (t.entries_data != nullptr)
    {
        std::ptrdiff_t off = &schedule_entry - t.entries_data;
        if (off >= 0 && off < t.entry_count)
            map_idx = static_cast<int>(off);
    }

    if (map_idx < 0)
    {
        // Cold path: no schedule bound (or entry came from a different vector).
        // Rebind for just this entry. Rare; not perf-critical.
        key_schedule one_entry(0, key_schedule::ALL_MAPS);
        one_entry.entries.clear();
        one_entry.entries.push_back(schedule_entry);
        _bind_schedule(one_entry);
        map_idx = 0;
    }

    __m256i wc0, wc1, wc2, wc3;
    _load_from_mem(wc0, wc1, wc2, wc3);
    _run_one_map(wc0, wc1, wc2, wc3, map_idx);
    _store_to_mem(wc0, wc1, wc2, wc3);
}

void tm_avx2_r256_map_8::run_all_maps(const key_schedule& schedule_entries)
{
    _bind_schedule(schedule_entries);
    run_maps_range(schedule_entries, 0, schedule_entries.entries.size());
}

void tm_avx2_r256_map_8::run_maps_range(const key_schedule& schedule_entries, std::size_t begin, std::size_t end)
{
    if (end > schedule_entries.entries.size())
        end = schedule_entries.entries.size();
    if (begin > end)
        begin = end;

    std::size_t local_begin = begin;
    std::size_t local_end = end;
    if (_shared == nullptr)
    {
        if (_owned.entries_data != schedule_entries.entries.data() + begin ||
            _owned.entry_count != static_cast<int>(end - begin))
        {
            _owned.bind_range(rng, schedule_entries, begin, end);
        }
        local_begin = 0;
        local_end = end - begin;
    }
    else
    {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() ||
            t.entry_count != static_cast<int>(schedule_entries.entries.size()))
        {
            _bind_schedule(schedule_entries);
        }
    }

    __m256i wc0, wc1, wc2, wc3;
    _load_from_mem(wc0, wc1, wc2, wc3);
    _run_maps_fixed(
        wc0, wc1, wc2, wc3,
        static_cast<int>(local_begin),
        static_cast<int>(local_end - local_begin));
    _store_to_mem(wc0, wc1, wc2, wc3);
}

// Checksum helpers (natural layout — no shuffle).
__forceinline void tm_avx2_r256_map_8::mid_sum(__m128i& sum, __m256i& wc, __m256i& sum_mask, __m128i& lo_mask)
{
    __m128i masked_lo = _mm_and_si128(_mm256_castsi256_si128(wc), _mm256_castsi256_si128(sum_mask));
    __m128i masked_hi = _mm_and_si128(_mm256_extracti128_si256(wc, 1), _mm256_extracti128_si256(sum_mask, 1));
    sum = _mm_add_epi16(sum, _mm_and_si128(masked_lo, lo_mask));
    sum = _mm_add_epi16(sum, _mm_srli_epi16(masked_lo, 8));
    sum = _mm_add_epi16(sum, _mm_and_si128(masked_hi, lo_mask));
    sum = _mm_add_epi16(sum, _mm_srli_epi16(masked_hi, 8));
}

__forceinline uint16 tm_avx2_r256_map_8::masked_checksum(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, const uint8* mask)
{
    __m128i sum = _mm_setzero_si128();
    __m128i lo_mask = _mm_set1_epi16(0x00FF);
    __m256i sum_mask;
    sum_mask = _mm256_load_si256((__m256i*)(mask));      mid_sum(sum, wc0, sum_mask, lo_mask);
    sum_mask = _mm256_load_si256((__m256i*)(mask + 32)); mid_sum(sum, wc1, sum_mask, lo_mask);
    sum_mask = _mm256_load_si256((__m256i*)(mask + 64)); mid_sum(sum, wc2, sum_mask, lo_mask);
    sum_mask = _mm256_load_si256((__m256i*)(mask + 96)); mid_sum(sum, wc3, sum_mask, lo_mask);
    return static_cast<uint16>(
        _mm_extract_epi16(sum, 0) + _mm_extract_epi16(sum, 1) +
        _mm_extract_epi16(sum, 2) + _mm_extract_epi16(sum, 3) +
        _mm_extract_epi16(sum, 4) + _mm_extract_epi16(sum, 5) +
        _mm_extract_epi16(sum, 6) + _mm_extract_epi16(sum, 7));
}

__forceinline uint16 tm_avx2_r256_map_8::fetch_checksum_value(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3, uint8 code_length)
{
    _store_to_mem(wc0, wc1, wc2, wc3);
    // code_length param convention: passed as (CODE_LENGTH - 2), so we want
    // bytes 127 - code_length (high) and 127 - (code_length + 1) (low).
    uint8 hi = working_code_data[127 - (code_length + 1)];
    uint8 lo = working_code_data[127 - code_length];
    return static_cast<uint16>((hi << 8) | lo);
}

__forceinline bool tm_avx2_r256_map_8::check_carnival_world_checksum(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3)
{
    uint16 calc = masked_checksum(wc0, wc1, wc2, wc3, TM_base::carnival_world_checksum_mask);
    return calc == fetch_checksum_value(wc0, wc1, wc2, wc3, CARNIVAL_WORLD_CODE_LENGTH - 2);
}

__forceinline bool tm_avx2_r256_map_8::check_other_world_checksum(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3)
{
    uint16 calc = masked_checksum(wc0, wc1, wc2, wc3, TM_base::other_world_checksum_mask);
    return calc == fetch_checksum_value(wc0, wc1, wc2, wc3, OTHER_WORLD_CODE_LENGTH - 2);
}

__forceinline void tm_avx2_r256_map_8::_decrypt_carnival_world(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3)
{
    xor_alg(wc0, wc1, wc2, wc3, TM_base::carnival_world_data);
}

__forceinline void tm_avx2_r256_map_8::_decrypt_other_world(__m256i& wc0, __m256i& wc1, __m256i& wc2, __m256i& wc3)
{
    xor_alg(wc0, wc1, wc2, wc3, TM_base::other_world_data);
}

void tm_avx2_r256_map_8::run_bruteforce_data(uint32 key, uint32 start_data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size)
{
    _bind_key(key);
    _bind_schedule(schedule_entries);

    __m256i wc0, wc1, wc2, wc3;

    uint32 output_pos = 0;
    for (uint32 i = 0; i < amount_to_run; i++)
    {
        if ((result_max_size - output_pos) < 5)
        {
            *result_size = result_max_size;
            return;
        }

        _expand_code(start_data + i, wc0, wc1, wc2, wc3);
        _run_all_maps(wc0, wc1, wc2, wc3);

        __m256i wc0x = wc0, wc1x = wc1, wc2x = wc2, wc3x = wc3;
        _decrypt_carnival_world(wc0x, wc1x, wc2x, wc3x);

        if (check_carnival_world_checksum(wc0x, wc1x, wc2x, wc3x))
        {
            _store_to_mem(wc0x, wc1x, wc2x, wc3x);
            *((uint32*)(&result_data[output_pos])) = i;
            result_data[output_pos + 4] = check_machine_code(working_code_data, CARNIVAL_WORLD);
            output_pos += 5;
        }
        else
        {
            wc0x = wc0; wc1x = wc1; wc2x = wc2; wc3x = wc3;
            _decrypt_other_world(wc0x, wc1x, wc2x, wc3x);
            if (check_other_world_checksum(wc0x, wc1x, wc2x, wc3x))
            {
                _store_to_mem(wc0x, wc1x, wc2x, wc3x);
                *((uint32*)(&result_data[output_pos])) = i;
                result_data[output_pos + 4] = check_machine_code(working_code_data, OTHER_WORLD);
                output_pos += 5;
            }
        }

        report_progress(static_cast<double>(i + 1) / static_cast<double>(amount_to_run));
    }
    *result_size = output_pos;
}

bool tm_avx2_r256_map_8::initialized = false;
