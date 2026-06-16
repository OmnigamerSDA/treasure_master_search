#include <stdio.h>
#include <cstdlib>
#include <cstring>

#include "tm_avx512_r512s_map_8.h"
#include "rng.h"

// micro500 used _mm256_setr_m128i (lo, hi); compose it explicitly so we do not
// depend on whether the toolchain exposes that particular intrinsic.
static __forceinline __m256i tm_setr_m128i(__m128i lo, __m128i hi)
{
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

tm_avx512_r512s_map_8::tm_avx512_r512s_map_8(RNG* rng_obj)
    : TM_base(rng_obj),
      mask_FF(_mm512_set1_epi8(static_cast<int8_t>(0xFF))),
      mask_FE(_mm512_set1_epi8(static_cast<int8_t>(0xFE))),
      mask_7F(_mm512_set1_epi8(0x7F)),
      mask_80(_mm512_set1_epi8(static_cast<int8_t>(0x80))),
      mask_01(_mm512_set1_epi8(0x01)),
      mask_top_01(_mm512_set_epi16(
          0x0100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
      mask_top_80(_mm512_set_epi16(
          static_cast<int16_t>(0x8000), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
{
    initialize();

    // Pre-shuffle the world-decrypt operands and checksum masks into the same
    // physical 2-ZMM layout the state uses, so screening can AND/XOR per byte
    // without re-laying-out the state on every candidate (mirrors micro500's
    // *_shuffled members). shuffle_mem(..., 512, false) applies shuffle(addr).
    shuffle_mem(TM_base::carnival_world_data,          carnival_world_data_shuf,     512, false);
    shuffle_mem(TM_base::other_world_data,             other_world_data_shuf,        512, false);
    shuffle_mem(TM_base::carnival_world_checksum_mask, carnival_checksum_mask_shuf,  512, false);
    shuffle_mem(TM_base::other_world_checksum_mask,    other_checksum_mask_shuf,     512, false);
}

tm_avx512_r512s_map_8::tm_avx512_r512s_map_8(RNG* rng_obj, map_tables_shared::Tables* shared_tables)
    : tm_avx512_r512s_map_8(rng_obj)
{
    _shared = shared_tables;
}

void tm_avx512_r512s_map_8::initialize()
{
    if (!initialized)
    {
        // Map-mode RNG blocks (natural flat) come from map_tables_shared, which
        // needs rng_table. The shuffled expansion table is reused verbatim from
        // the universal AVX-512 kernel for expand().
        rng->generate_rng_table();
        rng->generate_expansion_values_512_8_shuffled();
        rng->generate_seed_forward_1();
        rng->generate_seed_forward_128();
        initialized = true;
    }
    obj_name = "tm_avx512_r512s_map_8";
}

int tm_avx512_r512s_map_8::shuffle(int addr)
{
    return (addr % 2) * 64 + ((addr / 2) % 64);
}

void tm_avx512_r512s_map_8::_bind_schedule(const key_schedule& schedule_entries)
{
    if (_shared != nullptr) return;
    _owned.bind(rng, schedule_entries);
}

void tm_avx512_r512s_map_8::bind_schedule(const key_schedule& schedule_entries)
{
    _bind_schedule(schedule_entries);
}

void tm_avx512_r512s_map_8::bind_dedup_schedule(const key_schedule& /*schedule_entries*/)
{
    if (_shared != nullptr)
        return;
}

void tm_avx512_r512s_map_8::bind_maps_range(const key_schedule& schedule_entries, std::size_t begin, std::size_t end)
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

__forceinline void tm_avx512_r512s_map_8::_load_from_mem(__m512i& wc0, __m512i& wc1)
{
    wc0 = _mm512_load_si512((const __m512i*)(working_code_data));
    wc1 = _mm512_load_si512((const __m512i*)(working_code_data + 64));
}

__forceinline void tm_avx512_r512s_map_8::_store_to_mem(__m512i& wc0, __m512i& wc1)
{
    _mm512_store_si512((__m512i*)(working_code_data),      wc0);
    _mm512_store_si512((__m512i*)(working_code_data + 64), wc1);
}

void tm_avx512_r512s_map_8::load_data(uint8* new_data)
{
    // Shuffled layout: canonical byte i lives at physical index shuffle(i).
    for (int i = 0; i < 128; i++)
        working_code_data[shuffle(i)] = new_data[i];
}

void tm_avx512_r512s_map_8::fetch_data(uint8* new_data)
{
    for (int i = 0; i < 128; i++)
        new_data[i] = working_code_data[shuffle(i)];
}

// Expand, verbatim from tm_avx512_r512s_8 (the universal kernel): builds the
// shuffled expanded state directly from key/data and the pre-shuffled per-seed
// expansion row. reg0 = period-4 pattern [key>>24, key>>8, data>>24, data>>8],
// reg1 = [key>>16, key, data>>16, data], each broadcast across 64 bytes, then
// add the pre-shuffled expansion row.
__forceinline void tm_avx512_r512s_map_8::_expand_code(uint32 data, __m512i& wc0, __m512i& wc1)
{
    const uint32 key = _key;
    const uint32 dword0 =
        (uint32)((key >> 24) & 0xFF)
        | ((uint32)((key >> 8) & 0xFF) << 8)
        | ((uint32)((data >> 24) & 0xFF) << 16)
        | ((uint32)((data >> 8) & 0xFF) << 24);
    const uint32 dword1 =
        (uint32)((key >> 16) & 0xFF)
        | ((uint32)(key & 0xFF) << 8)
        | ((uint32)((data >> 16) & 0xFF) << 16)
        | ((uint32)(data & 0xFF) << 24);

    wc0 = _mm512_set1_epi32((int)dword0);
    wc1 = _mm512_set1_epi32((int)dword1);

    const uint16 rng_seed = (key >> 16) & 0xFFFF;
    const uint8* row = rng->expansion_values_512_8_shuffled + (static_cast<size_t>(rng_seed) * 128);
    wc0 = _mm512_add_epi8(wc0, _mm512_load_si512((const __m512i*)(row)));
    wc1 = _mm512_add_epi8(wc1, _mm512_load_si512((const __m512i*)(row + 64)));
}

void tm_avx512_r512s_map_8::expand(uint32 key, uint32 data)
{
    _key = key;
    __m512i wc0, wc1;
    _expand_code(data, wc0, wc1);
    _store_to_mem(wc0, wc1);
}

// ---------------------------------------------------------------------------
// _load_deinterleave: turn a natural flat 128-byte RNG window into the shuffled
// r0/r1 form (even canonical bytes -> r0, odd -> r1). Ported verbatim from
// micro500's tm_avx512bwvl_r512s_map_8.
// ---------------------------------------------------------------------------
__forceinline void tm_avx512_r512s_map_8::_load_deinterleave(const uint8* block_start, __m512i& r0, __m512i& r1)
{
    __m128i sel_even = _mm_set_epi8(
        static_cast<int8_t>(0x80), static_cast<int8_t>(0x80),
        static_cast<int8_t>(0x80), static_cast<int8_t>(0x80),
        static_cast<int8_t>(0x80), static_cast<int8_t>(0x80),
        static_cast<int8_t>(0x80), static_cast<int8_t>(0x80),
        14, 12, 10, 8, 6, 4, 2, 0);
    __m128i sel_odd = _mm_set_epi8(
        static_cast<int8_t>(0x80), static_cast<int8_t>(0x80),
        static_cast<int8_t>(0x80), static_cast<int8_t>(0x80),
        static_cast<int8_t>(0x80), static_cast<int8_t>(0x80),
        static_cast<int8_t>(0x80), static_cast<int8_t>(0x80),
        15, 13, 11, 9, 7, 5, 3, 1);

    __m128i a = _mm_loadu_si128((const __m128i*)(block_start));
    __m128i b = _mm_loadu_si128((const __m128i*)(block_start + 16));
    __m128i c = _mm_loadu_si128((const __m128i*)(block_start + 32));
    __m128i d = _mm_loadu_si128((const __m128i*)(block_start + 48));
    __m128i e = _mm_loadu_si128((const __m128i*)(block_start + 64));
    __m128i f = _mm_loadu_si128((const __m128i*)(block_start + 80));
    __m128i g = _mm_loadu_si128((const __m128i*)(block_start + 96));
    __m128i h = _mm_loadu_si128((const __m128i*)(block_start + 112));

    __m256i lo_even = tm_setr_m128i(
        _mm_unpacklo_epi64(_mm_shuffle_epi8(a, sel_even), _mm_shuffle_epi8(b, sel_even)),
        _mm_unpacklo_epi64(_mm_shuffle_epi8(c, sel_even), _mm_shuffle_epi8(d, sel_even)));
    __m256i hi_even = tm_setr_m128i(
        _mm_unpacklo_epi64(_mm_shuffle_epi8(e, sel_even), _mm_shuffle_epi8(f, sel_even)),
        _mm_unpacklo_epi64(_mm_shuffle_epi8(g, sel_even), _mm_shuffle_epi8(h, sel_even)));
    r0 = _mm512_inserti64x4(_mm512_castsi256_si512(lo_even), hi_even, 1);

    __m256i lo_odd = tm_setr_m128i(
        _mm_unpacklo_epi64(_mm_shuffle_epi8(a, sel_odd), _mm_shuffle_epi8(b, sel_odd)),
        _mm_unpacklo_epi64(_mm_shuffle_epi8(c, sel_odd), _mm_shuffle_epi8(d, sel_odd)));
    __m256i hi_odd = tm_setr_m128i(
        _mm_unpacklo_epi64(_mm_shuffle_epi8(e, sel_odd), _mm_shuffle_epi8(f, sel_odd)),
        _mm_unpacklo_epi64(_mm_shuffle_epi8(g, sel_odd), _mm_shuffle_epi8(h, sel_odd)));
    r1 = _mm512_inserti64x4(_mm512_castsi256_si512(lo_odd), hi_odd, 1);
}

// ---------------------------------------------------------------------------
// alg implementations: identical math to tm_avx512_r512s_8 (shuffled layout),
// only the RNG operand source changes (deinterleaved natural block window).
// alg_2/alg_5 sub math + carry placement ported from micro500's shuffled-map
// kernel (carry comes from the block byte, not a precomputed table).
// ---------------------------------------------------------------------------
__forceinline void tm_avx512_r512s_map_8::alg_0(__m512i& wc0, __m512i& wc1, const uint8* block_start)
{
    __m512i r0, r1;
    _load_deinterleave(block_start, r0, r1);
    wc0 = _mm512_ternarylogic_epi32(_mm512_slli_epi64(wc0, 1), mask_FE, r0, 0xEA);
    wc1 = _mm512_ternarylogic_epi32(_mm512_slli_epi64(wc1, 1), mask_FE, r1, 0xEA);
}

__forceinline void tm_avx512_r512s_map_8::alg_1(__m512i& wc0, __m512i& wc1, const uint8* block_start)
{
    __m512i r0, r1;
    _load_deinterleave(block_start, r0, r1);
    wc0 = _mm512_add_epi8(wc0, r0);
    wc1 = _mm512_add_epi8(wc1, r1);
}

__forceinline void tm_avx512_r512s_map_8::alg_2_sub(__m512i& working_a, __m512i& working_b, __m512i& carry)
{
    __m512i cur_val1_most = _mm512_and_si512(_mm512_srli_epi64(working_a, 1), mask_7F);
    __m512i cur_val2_most = _mm512_and_si512(_mm512_slli_epi64(working_b, 1), mask_FE);

    __m512i cur_val2_masked = _mm512_and_si512(working_b, mask_80);

    __m512i cur_val1_bit = _mm512_and_si512(working_a, mask_01);

    __m512i mask = _mm512_maskz_permutexvar_epi32(_cvtu32_mask16(0x0FFF),
        _mm512_set_epi32(0, 0, 0, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4), cur_val1_bit);
    __m512i cur_val1_srl = _mm512_alignr_epi8(mask, cur_val1_bit, 1);
    __m512i cur_val1_srl_w_carry = _mm512_or_si512(cur_val1_srl, carry);

    working_a = _mm512_or_si512(cur_val1_most, cur_val2_masked);
    working_b = _mm512_or_si512(cur_val2_most, cur_val1_srl_w_carry);
}

__forceinline void tm_avx512_r512s_map_8::alg_2(__m512i& wc0, __m512i& wc1, uint8 carry_byte)
{
    __m512i carry = _mm512_and_si512(_mm512_set1_epi8(static_cast<int8_t>(carry_byte)), mask_top_01);
    alg_2_sub(wc0, wc1, carry);
}

__forceinline void tm_avx512_r512s_map_8::alg_3(__m512i& wc0, __m512i& wc1, const uint8* block_start)
{
    __m512i r0, r1;
    _load_deinterleave(block_start, r0, r1);
    wc0 = _mm512_xor_si512(wc0, r0);
    wc1 = _mm512_xor_si512(wc1, r1);
}

__forceinline void tm_avx512_r512s_map_8::alg_4(__m512i& wc0, __m512i& wc1, const uint8* block_start)
{
    __m512i r0, r1;
    _load_deinterleave(block_start, r0, r1);
    wc0 = _mm512_sub_epi8(wc0, r0);
    wc1 = _mm512_sub_epi8(wc1, r1);
}

__forceinline void tm_avx512_r512s_map_8::alg_5_sub(__m512i& working_a, __m512i& working_b, __m512i& carry)
{
    __m512i cur_val1_most = _mm512_and_si512(_mm512_slli_epi64(working_a, 1), mask_FE);
    __m512i cur_val2_most = _mm512_and_si512(_mm512_srli_epi64(working_b, 1), mask_7F);

    __m512i cur_val2_masked = _mm512_and_si512(working_b, mask_01);

    __m512i cur_val1_bit = _mm512_and_si512(working_a, mask_80);

    __m512i mask = _mm512_maskz_permutexvar_epi32(_cvtu32_mask16(0x0FFF),
        _mm512_set_epi32(0, 0, 0, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4), cur_val1_bit);
    __m512i cur_val1_srl = _mm512_alignr_epi8(mask, cur_val1_bit, 1);
    __m512i cur_val1_srl_w_carry = _mm512_or_si512(cur_val1_srl, carry);

    working_a = _mm512_or_si512(cur_val1_most, cur_val2_masked);
    working_b = _mm512_or_si512(cur_val2_most, cur_val1_srl_w_carry);
}

__forceinline void tm_avx512_r512s_map_8::alg_5(__m512i& wc0, __m512i& wc1, uint8 carry_byte)
{
    __m512i carry = _mm512_and_si512(_mm512_set1_epi8(static_cast<int8_t>(carry_byte)), mask_top_80);
    alg_5_sub(wc0, wc1, carry);
}

__forceinline void tm_avx512_r512s_map_8::alg_6(__m512i& wc0, __m512i& wc1, const uint8* block_start)
{
    __m512i r0, r1;
    _load_deinterleave(block_start, r0, r1);
    wc0 = _mm512_ternarylogic_epi32(_mm512_srli_epi64(wc0, 1), mask_7F, r0, 0xEA);
    wc1 = _mm512_ternarylogic_epi32(_mm512_srli_epi64(wc1, 1), mask_7F, r1, 0xEA);
}

__forceinline void tm_avx512_r512s_map_8::alg_7(__m512i& wc0, __m512i& wc1)
{
    wc0 = _mm512_xor_si512(wc0, mask_FF);
    wc1 = _mm512_xor_si512(wc1, mask_FF);
}

__forceinline void tm_avx512_r512s_map_8::_run_alg(
    __m512i& wc0, __m512i& wc1,
    int algorithm_id, uint16* local_pos,
    const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base)
{
    switch (algorithm_id)
    {
        case 0:
            alg_0(wc0, wc1, alg0_base + *local_pos - 127);
            *local_pos -= 128;
            break;
        case 1:
            alg_1(wc0, wc1, reg_base + *local_pos - 127);
            *local_pos -= 128;
            break;
        case 2:
            alg_2(wc0, wc1, reg_base[*local_pos] >> 7);
            *local_pos -= 1;
            break;
        case 3:
            alg_3(wc0, wc1, reg_base + *local_pos - 127);
            *local_pos -= 128;
            break;
        case 4:
            alg_4(wc0, wc1, reg_base + *local_pos - 127);
            *local_pos -= 128;
            break;
        case 5:
            alg_5(wc0, wc1, reg_base[*local_pos] & 0x80);
            *local_pos -= 1;
            break;
        case 6:
            alg_6(wc0, wc1, alg6_base + (2047 - *local_pos));
            *local_pos -= 128;
            break;
        default: // case 7
            alg_7(wc0, wc1);
            break;
    }
}

__forceinline void tm_avx512_r512s_map_8::_run_one_map(__m512i& wc0, __m512i& wc1, int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 local_pos = 2047;

    // Shuffled layout: logical byte i for i=0..15 is reg0[i/2] (even i) or
    // reg1[(i-1)/2] (odd i), both in the low 128 bits. Same extraction trick as
    // tm_avx512_r512s_8::_run_map_entry. switch(i) supplies the compile-time
    // _mm_extract_epi8 immediate.
    // Unrolled (×16). No-unroll was trialed for design parity with the 4-way and
    // to cut the de-interleave-inflated icache (12.6M→614K L1i), but it backfired:
    // the switch(i) jump-table added +21% instructions + branch overhead, net
    // slower (8.64s vs 8.08s) — icache was not the dominant cost here. Kept unrolled.
#if defined(__GNUC__)
#pragma GCC unroll 16
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;

        uint8 current_byte = 0;
#if defined(_MSC_VER) || defined(__clang__)
        const __m128i w0_low = _mm512_castsi512_si128(wc0);
        const __m128i w1_low = _mm512_castsi512_si128(wc1);
        switch (i)
        {
            case 0:  current_byte = (uint8)_mm_extract_epi8(w0_low, 0); break;
            case 1:  current_byte = (uint8)_mm_extract_epi8(w1_low, 0); break;
            case 2:  current_byte = (uint8)_mm_extract_epi8(w0_low, 1); break;
            case 3:  current_byte = (uint8)_mm_extract_epi8(w1_low, 1); break;
            case 4:  current_byte = (uint8)_mm_extract_epi8(w0_low, 2); break;
            case 5:  current_byte = (uint8)_mm_extract_epi8(w1_low, 2); break;
            case 6:  current_byte = (uint8)_mm_extract_epi8(w0_low, 3); break;
            case 7:  current_byte = (uint8)_mm_extract_epi8(w1_low, 3); break;
            case 8:  current_byte = (uint8)_mm_extract_epi8(w0_low, 4); break;
            case 9:  current_byte = (uint8)_mm_extract_epi8(w1_low, 4); break;
            case 10: current_byte = (uint8)_mm_extract_epi8(w0_low, 5); break;
            case 11: current_byte = (uint8)_mm_extract_epi8(w1_low, 5); break;
            case 12: current_byte = (uint8)_mm_extract_epi8(w0_low, 6); break;
            case 13: current_byte = (uint8)_mm_extract_epi8(w1_low, 6); break;
            case 14: current_byte = (uint8)_mm_extract_epi8(w0_low, 7); break;
            case 15: current_byte = (uint8)_mm_extract_epi8(w1_low, 7); break;
        }
#else
        current_byte = ((i & 1) == 0)
            ? (uint8)_mm_extract_epi8(_mm512_castsi512_si128(wc0), i >> 1)
            : (uint8)_mm_extract_epi8(_mm512_castsi512_si128(wc1), (i - 1) >> 1);
#endif
        if (nibble == 1) current_byte >>= 4;
        uint8 algorithm_id = static_cast<uint8>((current_byte >> 1) & 0x07);

        _run_alg(wc0, wc1, algorithm_id, &local_pos, reg_base, alg0_base, alg6_base);
    }
}

__forceinline void tm_avx512_r512s_map_8::_run_maps_fixed(
    __m512i& wc0, __m512i& wc1, int map_idx, int count)
{
    if (count == 1)
        _run_one_map(wc0, wc1, map_idx);
    else
        for (int i = 0; i < count; i++)
            _run_one_map(wc0, wc1, map_idx + i);
}

void tm_avx512_r512s_map_8::run_alg(int /*algorithm_id*/, uint16* /*rng_seed*/, int /*iterations*/)
{
    // Not used by bench_cpu / dedup paths.
}

void tm_avx512_r512s_map_8::run_one_map(const key_schedule::key_schedule_entry& schedule_entry)
{
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
        key_schedule one_entry(0, key_schedule::ALL_MAPS);
        one_entry.entries.clear();
        one_entry.entries.push_back(schedule_entry);
        _bind_schedule(one_entry);
        map_idx = 0;
    }

    __m512i wc0, wc1;
    _load_from_mem(wc0, wc1);
    _run_one_map(wc0, wc1, map_idx);
    _store_to_mem(wc0, wc1);
}

void tm_avx512_r512s_map_8::run_all_maps(const key_schedule& schedule_entries)
{
    _bind_schedule(schedule_entries);
    run_maps_range(schedule_entries, 0, schedule_entries.entries.size());
}

void tm_avx512_r512s_map_8::run_maps_range(const key_schedule& schedule_entries, std::size_t begin, std::size_t end)
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

    __m512i wc0, wc1;
    _load_from_mem(wc0, wc1);
    _run_maps_fixed(
        wc0, wc1,
        static_cast<int>(local_begin),
        static_cast<int>(local_end - local_begin));
    _store_to_mem(wc0, wc1);
}

// ---------------------------------------------------------------------------
// 4-way interleaved kernel
// ---------------------------------------------------------------------------
__forceinline void tm_avx512_r512s_map_8::_alg_dispatch(
    __m512i& w0, __m512i& w1, int alg_id, uint16& local_pos,
    const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base)
{
    switch (alg_id)
    {
        case 0: alg_0(w0, w1, alg0_base + local_pos - 127); local_pos -= 128; break;
        case 1: alg_1(w0, w1, reg_base + local_pos - 127);  local_pos -= 128; break;
        case 2: alg_2(w0, w1, reg_base[local_pos] >> 7);    local_pos -= 1;   break;
        case 3: alg_3(w0, w1, reg_base + local_pos - 127);  local_pos -= 128; break;
        case 4: alg_4(w0, w1, reg_base + local_pos - 127);  local_pos -= 128; break;
        case 5: alg_5(w0, w1, reg_base[local_pos] & 0x80);  local_pos -= 1;   break;
        case 6: alg_6(w0, w1, alg6_base + (2047 - local_pos)); local_pos -= 128; break;
        default: alg_7(w0, w1); break;
    }
}

__forceinline void tm_avx512_r512s_map_8::_run_map_entry_x4(
    __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
    __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
    int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];

    // Each of the 4 states needs its OWN moving local_pos (they share the
    // entry's reg/alg0/alg6 bases and nibble_selector).
    uint16 pa = 2047, pb = 2047, pc = 2047, pd = 2047;

    // NOT unrolled (mirrors tm_avx512_r512s_8::_run_map_entry_x4): the switch(i)
    // supplies the _mm_extract_epi8 immediate per case.
#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;

        const __m128i a0l = _mm512_castsi512_si128(a0), a1l = _mm512_castsi512_si128(a1);
        const __m128i b0l = _mm512_castsi512_si128(b0), b1l = _mm512_castsi512_si128(b1);
        const __m128i c0l = _mm512_castsi512_si128(c0), c1l = _mm512_castsi512_si128(c1);
        const __m128i d0l = _mm512_castsi512_si128(d0), d1l = _mm512_castsi512_si128(d1);
        uint8 ba = 0, bb = 0, bc = 0, bd = 0;
#define TM_MAP_X4_EXTRACT(out, r0, r1) \
        switch (i) { \
            case 0:  out=(uint8)_mm_extract_epi8(r0,0); break; \
            case 1:  out=(uint8)_mm_extract_epi8(r1,0); break; \
            case 2:  out=(uint8)_mm_extract_epi8(r0,1); break; \
            case 3:  out=(uint8)_mm_extract_epi8(r1,1); break; \
            case 4:  out=(uint8)_mm_extract_epi8(r0,2); break; \
            case 5:  out=(uint8)_mm_extract_epi8(r1,2); break; \
            case 6:  out=(uint8)_mm_extract_epi8(r0,3); break; \
            case 7:  out=(uint8)_mm_extract_epi8(r1,3); break; \
            case 8:  out=(uint8)_mm_extract_epi8(r0,4); break; \
            case 9:  out=(uint8)_mm_extract_epi8(r1,4); break; \
            case 10: out=(uint8)_mm_extract_epi8(r0,5); break; \
            case 11: out=(uint8)_mm_extract_epi8(r1,5); break; \
            case 12: out=(uint8)_mm_extract_epi8(r0,6); break; \
            case 13: out=(uint8)_mm_extract_epi8(r1,6); break; \
            case 14: out=(uint8)_mm_extract_epi8(r0,7); break; \
            case 15: out=(uint8)_mm_extract_epi8(r1,7); break; \
        }
        TM_MAP_X4_EXTRACT(ba, a0l, a1l);
        TM_MAP_X4_EXTRACT(bb, b0l, b1l);
        TM_MAP_X4_EXTRACT(bc, c0l, c1l);
        TM_MAP_X4_EXTRACT(bd, d0l, d1l);
#undef TM_MAP_X4_EXTRACT
        if (nibble == 1) { ba >>= 4; bb >>= 4; bc >>= 4; bd >>= 4; }

        _alg_dispatch(a0, a1, (ba >> 1) & 0x07, pa, reg_base, alg0_base, alg6_base);
        _alg_dispatch(b0, b1, (bb >> 1) & 0x07, pb, reg_base, alg0_base, alg6_base);
        _alg_dispatch(c0, c1, (bc >> 1) & 0x07, pc, reg_base, alg0_base, alg6_base);
        _alg_dispatch(d0, d1, (bd >> 1) & 0x07, pd, reg_base, alg0_base, alg6_base);
    }
}

void tm_avx512_r512s_map_8::run_maps_range_x4(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3)
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

    // Shuffled layout: in/out are shuffled 128-byte states. Pool entries are not
    // 64-byte aligned; use unaligned loads/stores.
    __m512i a0 = _mm512_loadu_si512((const void*)(in0)), a1 = _mm512_loadu_si512((const void*)(in0 + 64));
    __m512i b0 = _mm512_loadu_si512((const void*)(in1)), b1 = _mm512_loadu_si512((const void*)(in1 + 64));
    __m512i c0 = _mm512_loadu_si512((const void*)(in2)), c1 = _mm512_loadu_si512((const void*)(in2 + 64));
    __m512i d0 = _mm512_loadu_si512((const void*)(in3)), d1 = _mm512_loadu_si512((const void*)(in3 + 64));

    for (std::size_t map_idx = local_begin; map_idx < local_end; map_idx++)
    {
        _run_map_entry_x4(a0, a1, b0, b1, c0, c1, d0, d1, static_cast<int>(map_idx));
    }

    _mm512_storeu_si512((void*)(out0), a0); _mm512_storeu_si512((void*)(out0 + 64), a1);
    _mm512_storeu_si512((void*)(out1), b0); _mm512_storeu_si512((void*)(out1 + 64), b1);
    _mm512_storeu_si512((void*)(out2), c0); _mm512_storeu_si512((void*)(out2 + 64), c1);
    _mm512_storeu_si512((void*)(out3), d0); _mm512_storeu_si512((void*)(out3 + 64), d1);
}

// ---------------------------------------------------------------------------
// Native bulk forward + checksum screen (NON-dedup production path). Shuffled
// layout: masks/world-data are the pre-shuffled instance copies. The masked
// checksum is a byte-sum (vpsadbw), which is order-independent, so it is
// correct on the shuffled state as long as the mask is shuffled to match.
// ---------------------------------------------------------------------------
__forceinline void tm_avx512_r512s_map_8::_run_all_maps(__m512i& wc0, __m512i& wc1)
{
    const int n = _t().entry_count;
    for (int map_idx = 0; map_idx < n; map_idx++)
        _run_one_map(wc0, wc1, map_idx);
}

__forceinline void tm_avx512_r512s_map_8::xor_alg(__m512i& wc0, __m512i& wc1, const uint8* values)
{
    wc0 = _mm512_xor_si512(wc0, _mm512_load_si512((const __m512i*)(values)));
    wc1 = _mm512_xor_si512(wc1, _mm512_load_si512((const __m512i*)(values + 64)));
}

__forceinline void tm_avx512_r512s_map_8::_decrypt_carnival_world(__m512i& wc0, __m512i& wc1)
{
    xor_alg(wc0, wc1, carnival_world_data_shuf);
}

__forceinline void tm_avx512_r512s_map_8::_decrypt_other_world(__m512i& wc0, __m512i& wc1)
{
    xor_alg(wc0, wc1, other_world_data_shuf);
}

__forceinline uint16 tm_avx512_r512s_map_8::masked_checksum(__m512i& wc0, __m512i& wc1, const uint8* mask)
{
    __m512i m0 = _mm512_load_si512((const __m512i*)(mask));
    __m512i m1 = _mm512_load_si512((const __m512i*)(mask + 64));
    __m512i v0 = _mm512_and_si512(wc0, m0);
    __m512i v1 = _mm512_and_si512(wc1, m1);
    __m512i s = _mm512_add_epi64(_mm512_sad_epu8(v0, _mm512_setzero_si512()),
                                 _mm512_sad_epu8(v1, _mm512_setzero_si512()));
    return static_cast<uint16>(_mm512_reduce_add_epi64(s));
}

__forceinline uint16 tm_avx512_r512s_map_8::fetch_checksum_value(__m512i& wc0, __m512i& wc1, uint8 code_length)
{
    _store_to_mem(wc0, wc1);
    // code_length param convention: passed as (CODE_LENGTH - 2). The stored
    // checksum bytes are canonical indices 127-code_length (lo) and
    // 127-(code_length+1) (hi); on the shuffled state they live at shuffle(idx).
    uint8 hi = working_code_data[shuffle(127 - (code_length + 1))];
    uint8 lo = working_code_data[shuffle(127 - code_length)];
    return static_cast<uint16>((hi << 8) | lo);
}

__forceinline bool tm_avx512_r512s_map_8::check_carnival_world_checksum(__m512i& wc0, __m512i& wc1)
{
    uint16 calc = masked_checksum(wc0, wc1, carnival_checksum_mask_shuf);
    return calc == fetch_checksum_value(wc0, wc1, CARNIVAL_WORLD_CODE_LENGTH - 2);
}

__forceinline bool tm_avx512_r512s_map_8::check_other_world_checksum(__m512i& wc0, __m512i& wc1)
{
    uint16 calc = masked_checksum(wc0, wc1, other_checksum_mask_shuf);
    return calc == fetch_checksum_value(wc0, wc1, OTHER_WORLD_CODE_LENGTH - 2);
}

__forceinline void tm_avx512_r512s_map_8::_screen_emit(__m512i w0, __m512i w1, uint32 idx, uint8* result_data, uint32& output_pos)
{
    uint8 canonical[128];
    __m512i a0 = w0, a1 = w1;
    _decrypt_carnival_world(a0, a1);
    if (check_carnival_world_checksum(a0, a1))
    {
        // fetch_checksum_value already stored the decrypted state to mem.
        fetch_data(canonical);
        *((uint32*)(&result_data[output_pos])) = idx;
        result_data[output_pos + 4] = check_machine_code(canonical, CARNIVAL_WORLD);
        output_pos += 5;
        return;
    }
    a0 = w0; a1 = w1;
    _decrypt_other_world(a0, a1);
    if (check_other_world_checksum(a0, a1))
    {
        fetch_data(canonical);
        *((uint32*)(&result_data[output_pos])) = idx;
        result_data[output_pos + 4] = check_machine_code(canonical, OTHER_WORLD);
        output_pos += 5;
    }
}

void tm_avx512_r512s_map_8::run_bruteforce_data(uint32 key, uint32 start_data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size)
{
    _key = key;
    _bind_schedule(schedule_entries);

    __m512i wc0, wc1;

    uint32 output_pos = 0;
    for (uint32 i = 0; i < amount_to_run; i++)
    {
        if ((result_max_size - output_pos) < 5)
        {
            *result_size = result_max_size;
            return;
        }

        _expand_code(start_data + i, wc0, wc1);
        _run_all_maps(wc0, wc1);
        _screen_emit(wc0, wc1, i, result_data, output_pos);

        report_progress(static_cast<double>(i + 1) / static_cast<double>(amount_to_run));
    }
    *result_size = output_pos;
}

void tm_avx512_r512s_map_8::run_bruteforce_data_il(uint32 key, uint32 start_data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size)
{
    _key = key;
    _bind_schedule(schedule_entries);   // ensures tables bound (incl. the <4 tail) and avoids per-batch rebind
    const std::size_t n = static_cast<std::size_t>(_t().entry_count);

    alignas(64) uint8 in0[128], in1[128], in2[128], in3[128];
    alignas(64) uint8 out0[128], out1[128], out2[128], out3[128];

    uint32 output_pos = 0;
    uint32 i = 0;
    for (; i + 4 <= amount_to_run; i += 4)
    {
        if ((result_max_size - output_pos) < 20)   // up to 4 × 5-byte records
        {
            *result_size = output_pos;
            return;
        }

        __m512i e0, e1;
        _expand_code(start_data + i + 0, e0, e1);
        _mm512_store_si512((__m512i*)in0, e0); _mm512_store_si512((__m512i*)(in0 + 64), e1);
        _expand_code(start_data + i + 1, e0, e1);
        _mm512_store_si512((__m512i*)in1, e0); _mm512_store_si512((__m512i*)(in1 + 64), e1);
        _expand_code(start_data + i + 2, e0, e1);
        _mm512_store_si512((__m512i*)in2, e0); _mm512_store_si512((__m512i*)(in2 + 64), e1);
        _expand_code(start_data + i + 3, e0, e1);
        _mm512_store_si512((__m512i*)in3, e0); _mm512_store_si512((__m512i*)(in3 + 64), e1);

        run_maps_range_x4(schedule_entries, 0, n, in0, in1, in2, in3, out0, out1, out2, out3);

        _screen_emit(_mm512_load_si512((const __m512i*)out0), _mm512_load_si512((const __m512i*)(out0 + 64)), i + 0, result_data, output_pos);
        _screen_emit(_mm512_load_si512((const __m512i*)out1), _mm512_load_si512((const __m512i*)(out1 + 64)), i + 1, result_data, output_pos);
        _screen_emit(_mm512_load_si512((const __m512i*)out2), _mm512_load_si512((const __m512i*)(out2 + 64)), i + 2, result_data, output_pos);
        _screen_emit(_mm512_load_si512((const __m512i*)out3), _mm512_load_si512((const __m512i*)(out3 + 64)), i + 3, result_data, output_pos);

        report_progress(static_cast<double>(i + 4) / static_cast<double>(amount_to_run));
    }
    // Tail (<4 remaining): 1-way.
    for (; i < amount_to_run; i++)
    {
        if ((result_max_size - output_pos) < 5)
        {
            *result_size = output_pos;
            return;
        }
        __m512i wc0, wc1;
        _expand_code(start_data + i, wc0, wc1);
        _run_all_maps(wc0, wc1);
        _screen_emit(wc0, wc1, i, result_data, output_pos);
        report_progress(static_cast<double>(i + 1) / static_cast<double>(amount_to_run));
    }
    *result_size = output_pos;
}

bool tm_avx512_r512s_map_8::initialized = false;
