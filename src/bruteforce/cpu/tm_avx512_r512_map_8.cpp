#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <array>
#include <cstdint>

#include "tm_avx512_r512_map_8.h"
#include "routing.h"
#include "rng.h"

tm_avx512_r512_map_8::tm_avx512_r512_map_8(RNG* rng_obj)
    : TM_base(rng_obj),
      mask_FF(_mm512_set1_epi8(static_cast<int8_t>(0xFF))),
      mask_FE(_mm512_set1_epi8(static_cast<int8_t>(0xFE))),
      mask_7F(_mm512_set1_epi8(0x7F)),
      mask_80(_mm512_set1_epi8(static_cast<int8_t>(0x80))),
      mask_01(_mm512_set1_epi8(0x01))
{
    initialize();
}

tm_avx512_r512_map_8::tm_avx512_r512_map_8(RNG* rng_obj, map_tables_shared::Tables* shared_tables)
    : tm_avx512_r512_map_8(rng_obj)
{
    _shared = shared_tables;
}

void tm_avx512_r512_map_8::initialize()
{
    // Thread-safe one-time generation of the SHARED static RNG tables. These
    // generators (re-)`new` and fill RNG::rng_table and read it while building the
    // derived tables, so concurrent first-construction across threads (e.g. map1par
    // spawning Tp workers cold, with no main-thread warmup) raced on the unsynchronized
    // `initialized` guard and could observe a half-rebuilt rng_table -> a few corrupted
    // seeds -> a nondeterministic, slightly-inflated frontier. call_once serializes it.
    static std::once_flag s_rng_once;
    std::call_once(s_rng_once, [this] {
        rng->generate_rng_table();
        rng->generate_expansion_values_8();
        rng->generate_seed_forward_1();
        rng->generate_seed_forward_128();
        initialized = true;
    });
    obj_name = "tm_avx512_r512_map_8";
}

void tm_avx512_r512_map_8::_bind_key(uint32 new_key)
{
    if (new_key == _key && _expansion_for_seed_128 != nullptr) return;
    _key = new_key;
    const uint16 expansion_seed = static_cast<uint16>((new_key >> 16) & 0xFFFF);
    _expansion_for_seed_128 = rng->expansion_values_8 + static_cast<size_t>(expansion_seed) * 128;
}

void tm_avx512_r512_map_8::_bind_schedule(const key_schedule& schedule_entries)
{
    if (_shared != nullptr) return;
    _owned.bind(rng, schedule_entries);
}

void tm_avx512_r512_map_8::bind_schedule(const key_schedule& schedule_entries)
{
    _bind_schedule(schedule_entries);
}

void tm_avx512_r512_map_8::bind_dedup_schedule(const key_schedule& /*schedule_entries*/)
{
    if (_shared != nullptr)
        return;
}

void tm_avx512_r512_map_8::bind_maps_range(const key_schedule& schedule_entries, std::size_t begin, std::size_t end)
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

__forceinline void tm_avx512_r512_map_8::_load_from_mem(__m512i& wc0, __m512i& wc1)
{
    wc0 = _mm512_load_si512((const __m512i*)(working_code_data));
    wc1 = _mm512_load_si512((const __m512i*)(working_code_data + 64));
}

__forceinline void tm_avx512_r512_map_8::_store_to_mem(__m512i& wc0, __m512i& wc1)
{
    _mm512_store_si512((__m512i*)(working_code_data),      wc0);
    _mm512_store_si512((__m512i*)(working_code_data + 64), wc1);
}

void tm_avx512_r512_map_8::load_data(uint8* new_data)
{
    for (int i = 0; i < 128; i++) working_code_data[i] = new_data[i];
}

void tm_avx512_r512_map_8::fetch_data(uint8* new_data)
{
    for (int i = 0; i < 128; i++) new_data[i] = working_code_data[i];
}

__forceinline void tm_avx512_r512_map_8::_expand_code(uint32 data, __m512i& wc0, __m512i& wc1)
{
    // Natural layout: canonical byte j cycles through the 8-byte pattern
    // [key>>24, key>>16, key>>8, key, data>>24, data>>16, data>>8, data]
    // (the AVX2 map kernel's nat_mask {0,1,2,3,4,5,6,7} applied to the dword
    // [data,key] gives this 8-byte big-endian-of-each pattern). Build the 8-byte
    // pattern once and broadcast it across both ZMM (period 8 divides 64).
    const uint32 key = _key;
    uint8 pat[8];
    pat[0] = static_cast<uint8>((key >> 24) & 0xFF);
    pat[1] = static_cast<uint8>((key >> 16) & 0xFF);
    pat[2] = static_cast<uint8>((key >> 8) & 0xFF);
    pat[3] = static_cast<uint8>(key & 0xFF);
    pat[4] = static_cast<uint8>((data >> 24) & 0xFF);
    pat[5] = static_cast<uint8>((data >> 16) & 0xFF);
    pat[6] = static_cast<uint8>((data >> 8) & 0xFF);
    pat[7] = static_cast<uint8>(data & 0xFF);

    uint64_t patq;
    std::memcpy(&patq, pat, 8);
    __m512i pat512 = _mm512_set1_epi64(static_cast<long long>(patq));

    wc0 = pat512;
    wc1 = pat512;

    const uint8* rng_start = _expansion_for_seed_128;
    wc0 = _mm512_add_epi8(wc0, _mm512_loadu_si512((const __m512i*)(rng_start)));
    wc1 = _mm512_add_epi8(wc1, _mm512_loadu_si512((const __m512i*)(rng_start + 64)));
}

void tm_avx512_r512_map_8::expand(uint32 key, uint32 data)
{
    _bind_key(key);

    __m512i wc0, wc1;
    _expand_code(data, wc0, wc1);
    _store_to_mem(wc0, wc1);
}

// ---------------------------------------------------------------------------
// alg_2 / alg_5: cross-byte single-bit "butterfly" with carry threaded across
// the whole 128-byte (1024-bit) state. From the scalar reference (tm_8::alg_2 /
// alg_5), the transform is NOT a uniform byte rotate — even and odd canonical
// bytes move in opposite directions, each pulling one carry bit from its
// neighbor at canonical index +1:
//
//   alg_2:  even e' = (e>>1) | (b[e+1] & 0x80)    odd o' = (o<<1) | (b[o+1] & 0x01)
//   alg_5:  even e' = (e<<1) | (b[e+1] & 0x01)    odd o' = (o>>1) | (b[o+1] & 0x80)
//
// (the topmost byte 127, odd, takes the rng-seeded carry instead of b[128].)
//
// Because every output bit depends only on input bytes within +1, this is a
// pure stencil over the ORIGINAL state — no ripple carry. The previous port
// mirrored the AVX2 _epi16 word-pair code: ~18-20 ALU ops and TWO vpermw per
// register (one for the neighbor shift, one purely to thread the 1-bit
// inter-register carry), called once per half = 4 cross-lane shuffles per alg.
//
// This form collapses it to, per register: two per-byte shifts (qword-shift +
// mask), an even/odd blend, and one masked OR of the neighbor byte — plus a
// SINGLE two-source byte permute (vpermi2b) that produces neighbor[i]=byte[i+1]
// while pulling the cross-register byte (wc0[63]<-wc1[0]) or the seed (wc1[63])
// straight out of the second operand. ~7 ALU ops + 1 shuffle per register, 2
// cross-lane shuffles per alg (half the old count), and the dedicated
// inter-register-carry permute is gone entirely.
// ---------------------------------------------------------------------------

// neighbor index: result[i] = byte[i+1]. Values 0..63 select the first
// vpermi2b operand, value 64 selects byte0 of the second operand (used for the
// wc0<-wc1 join and the wc1<-seed join). set_epi8 lists byte63..byte0.
static const __m512i kIdxByteDown = _mm512_set_epi8(
    64,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,
    48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,
    32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,
    16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1);

static constexpr __mmask64 kOddByteMask = 0xAAAAAAAAAAAAAAAAull;                       // set on odd byte lanes

// One butterfly step over a single register. neighbor = byte[i+1] for this
// register (boundary byte already supplied by the caller's vpermi2b).
//   alg_2 (srli_first=true):  even lane <- (b>>1)&7F, odd lane <- (b<<1)&FE
//   alg_5 (srli_first=false): even lane <- (b<<1)&FE, odd lane <- (b>>1)&7F
// The per-byte carry selector (even bit7 / odd bit0 for alg_2; reversed for
// alg_5) is built inline via vpbroadcastw rather than held in a `static const`
// ZMM. The constant form was being *pinned* in a register across the whole x12
// loop (objdump: 0 in-loop vpbroadcast), and that extra live constant is exactly
// what pushed one frontier state into a stack spill at x12 (and three at x14).
// Rebuilding it on demand lets GCC rematerialize it like it does the universal
// kernel's masks, freeing the register for state.
template <bool srli_first>
static __forceinline __m512i tm_butterfly_step(__m512i wc, __m512i neighbor)
{
    const __m512i carry_mask = srli_first
        ? _mm512_set1_epi16(static_cast<short>(0x0180))   // even=0x80 (bit7), odd=0x01 (bit0)
        : _mm512_set1_epi16(static_cast<short>(0x8001));  // even=0x01, odd=0x80
    // m_7F / m_FE built inline (vpbroadcastb) for the same reason as carry_mask:
    // as pinned ZMM constants they cost state registers across the x12/x14 loop.
    const __m512i m_7F = _mm512_set1_epi8(0x7F);
    const __m512i m_FE = _mm512_set1_epi8(static_cast<char>(0xFE));
    __m512i sr = _mm512_and_si512(_mm512_srli_epi64(wc, 1), m_7F); // per-byte >>1
    __m512i sl = _mm512_and_si512(_mm512_slli_epi64(wc, 1), m_FE); // per-byte <<1
    // even lanes get the "first" shift, odd lanes the other.
    __m512i shifted = srli_first ? _mm512_mask_blend_epi8(kOddByteMask, sr, sl)
                                 : _mm512_mask_blend_epi8(kOddByteMask, sl, sr);
    __m512i carry_bits = _mm512_and_si512(neighbor, carry_mask);
    return _mm512_or_si512(shifted, carry_bits);
}

__forceinline void tm_avx512_r512_map_8::alg_2(__m512i& wc0, __m512i& wc1, uint8 carry_byte)
{
    // seed (byte 127's missing neighbor): only bit0 matters for the odd lane;
    // carry_byte already carries it in bit0 (reg_base[pos] >> 7).
    __m512i seed = _mm512_castsi128_si512(_mm_cvtsi32_si128(static_cast<int>(carry_byte)));
    // Compute+consume nb0 before nb1 is born so the two neighbor temps are never
    // simultaneously live (lower peak register pressure -> fewer state spills at
    // x12/x14). nb1 needs only the original wc1, which the wc0 store below leaves
    // intact.
    __m512i nb0 = _mm512_permutex2var_epi8(wc0, kIdxByteDown, wc1);   // wc0 neighbors, [63]<-wc1[0]
    wc0 = tm_butterfly_step<true>(wc0, nb0);
    __m512i nb1 = _mm512_permutex2var_epi8(wc1, kIdxByteDown, seed);  // wc1 neighbors, [63]<-seed
    wc1 = tm_butterfly_step<true>(wc1, nb1);
}

__forceinline void tm_avx512_r512_map_8::alg_5(__m512i& wc0, __m512i& wc1, uint8 carry_byte)
{
    // seed bit7 (odd lane mask 0x80); carry_byte = reg_base[pos] & 0x80.
    __m512i seed = _mm512_castsi128_si512(_mm_cvtsi32_si128(static_cast<int>(carry_byte)));
    __m512i nb0 = _mm512_permutex2var_epi8(wc0, kIdxByteDown, wc1);
    wc0 = tm_butterfly_step<false>(wc0, nb0);
    __m512i nb1 = _mm512_permutex2var_epi8(wc1, kIdxByteDown, seed);
    wc1 = tm_butterfly_step<false>(wc1, nb1);
}

__forceinline void tm_avx512_r512_map_8::alg_0(__m512i& wc0, __m512i& wc1, const uint8* block_start)
{
    // (wc<<1 & 0xFE) | rng, fused with vpternlogd 0xEA = (A&B)|C. Per-byte shift
    // uses slli_epi64 (the shifted-out bit is overwritten by the rng LSB).
    wc0 = _mm512_slli_epi64(wc0, 1);
    __m512i rng_val = _mm512_loadu_si512((const __m512i*)(block_start));
    wc0 = _mm512_ternarylogic_epi32(wc0, mask_FE, rng_val, 0xEA);

    wc1 = _mm512_slli_epi64(wc1, 1);
    rng_val = _mm512_loadu_si512((const __m512i*)(block_start + 64));
    wc1 = _mm512_ternarylogic_epi32(wc1, mask_FE, rng_val, 0xEA);
}

__forceinline void tm_avx512_r512_map_8::alg_1(__m512i& wc0, __m512i& wc1, const uint8* block_start)
{
    wc0 = _mm512_add_epi8(wc0, _mm512_loadu_si512((const __m512i*)(block_start)));
    wc1 = _mm512_add_epi8(wc1, _mm512_loadu_si512((const __m512i*)(block_start + 64)));
}

__forceinline void tm_avx512_r512_map_8::alg_3(__m512i& wc0, __m512i& wc1, const uint8* block_start)
{
    wc0 = _mm512_xor_si512(wc0, _mm512_loadu_si512((const __m512i*)(block_start)));
    wc1 = _mm512_xor_si512(wc1, _mm512_loadu_si512((const __m512i*)(block_start + 64)));
}

__forceinline void tm_avx512_r512_map_8::alg_4(__m512i& wc0, __m512i& wc1, const uint8* block_start)
{
    wc0 = _mm512_sub_epi8(wc0, _mm512_loadu_si512((const __m512i*)(block_start)));
    wc1 = _mm512_sub_epi8(wc1, _mm512_loadu_si512((const __m512i*)(block_start + 64)));
}

__forceinline void tm_avx512_r512_map_8::alg_6(__m512i& wc0, __m512i& wc1, const uint8* block_start)
{
    wc0 = _mm512_srli_epi64(wc0, 1);
    __m512i rng_val = _mm512_loadu_si512((const __m512i*)(block_start));
    wc0 = _mm512_ternarylogic_epi32(wc0, mask_7F, rng_val, 0xEA);

    wc1 = _mm512_srli_epi64(wc1, 1);
    rng_val = _mm512_loadu_si512((const __m512i*)(block_start + 64));
    wc1 = _mm512_ternarylogic_epi32(wc1, mask_7F, rng_val, 0xEA);
}

__forceinline void tm_avx512_r512_map_8::alg_7(__m512i& wc0, __m512i& wc1)
{
    wc0 = _mm512_xor_si512(wc0, mask_FF);
    wc1 = _mm512_xor_si512(wc1, mask_FF);
}

__forceinline void tm_avx512_r512_map_8::xor_alg(__m512i& wc0, __m512i& wc1, const uint8* values)
{
    wc0 = _mm512_xor_si512(wc0, _mm512_load_si512((const __m512i*)(values)));
    wc1 = _mm512_xor_si512(wc1, _mm512_load_si512((const __m512i*)(values + 64)));
}

__forceinline void tm_avx512_r512_map_8::_run_alg(
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

__forceinline void tm_avx512_r512_map_8::_run_one_map(__m512i& wc0, __m512i& wc1, int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 local_pos = 2047;

    // Natural layout: byte i (i in 0..15) lives at wc0 byte i. Read it directly
    // out of the low 128 bits via _mm_extract_epi8 (compile-time index, so unroll).
#if defined(__GNUC__)
#pragma GCC unroll 16
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;

        uint8 current_byte = 0;
#if defined(_MSC_VER) || defined(__clang__)   // clang requires a constant _mm_extract index
        const __m128i wc0_low = _mm512_castsi512_si128(wc0);
        switch (i)
        {
            case 0: current_byte = static_cast<uint8>(_mm_extract_epi8(wc0_low, 0)); break;
            case 1: current_byte = static_cast<uint8>(_mm_extract_epi8(wc0_low, 1)); break;
            case 2: current_byte = static_cast<uint8>(_mm_extract_epi8(wc0_low, 2)); break;
            case 3: current_byte = static_cast<uint8>(_mm_extract_epi8(wc0_low, 3)); break;
            case 4: current_byte = static_cast<uint8>(_mm_extract_epi8(wc0_low, 4)); break;
            case 5: current_byte = static_cast<uint8>(_mm_extract_epi8(wc0_low, 5)); break;
            case 6: current_byte = static_cast<uint8>(_mm_extract_epi8(wc0_low, 6)); break;
            case 7: current_byte = static_cast<uint8>(_mm_extract_epi8(wc0_low, 7)); break;
            case 8: current_byte = static_cast<uint8>(_mm_extract_epi8(wc0_low, 8)); break;
            case 9: current_byte = static_cast<uint8>(_mm_extract_epi8(wc0_low, 9)); break;
            case 10: current_byte = static_cast<uint8>(_mm_extract_epi8(wc0_low, 10)); break;
            case 11: current_byte = static_cast<uint8>(_mm_extract_epi8(wc0_low, 11)); break;
            case 12: current_byte = static_cast<uint8>(_mm_extract_epi8(wc0_low, 12)); break;
            case 13: current_byte = static_cast<uint8>(_mm_extract_epi8(wc0_low, 13)); break;
            case 14: current_byte = static_cast<uint8>(_mm_extract_epi8(wc0_low, 14)); break;
            case 15: current_byte = static_cast<uint8>(_mm_extract_epi8(wc0_low, 15)); break;
        }
#else
        current_byte = static_cast<uint8>(_mm_extract_epi8(_mm512_castsi512_si128(wc0), i));
#endif
        if (nibble == 1) current_byte >>= 4;
        uint8 algorithm_id = static_cast<uint8>((current_byte >> 1) & 0x07);

        _run_alg(wc0, wc1, algorithm_id, &local_pos, reg_base, alg0_base, alg6_base);
    }
}

__forceinline void tm_avx512_r512_map_8::_run_maps_fixed(
    __m512i& wc0, __m512i& wc1, int map_idx, int count)
{
    if (count == 1)
        _run_one_map(wc0, wc1, map_idx);
    else
        for (int i = 0; i < count; i++)
            _run_one_map(wc0, wc1, map_idx + i);
}

void tm_avx512_r512_map_8::run_alg(int /*algorithm_id*/, uint16* /*rng_seed*/, int /*iterations*/)
{
    // Not used by bench_cpu / run_bruteforce_data.
}

__forceinline void tm_avx512_r512_map_8::_run_all_maps(__m512i& wc0, __m512i& wc1)
{
    const int n = _t().entry_count;
    for (int map_idx = 0; map_idx < n; map_idx++)
        _run_one_map(wc0, wc1, map_idx);
}

void tm_avx512_r512_map_8::run_one_map(const key_schedule::key_schedule_entry& schedule_entry)
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

void tm_avx512_r512_map_8::run_all_maps(const key_schedule& schedule_entries)
{
    _bind_schedule(schedule_entries);
    run_maps_range(schedule_entries, 0, schedule_entries.entries.size());
}

void tm_avx512_r512_map_8::run_maps_range(const key_schedule& schedule_entries, std::size_t begin, std::size_t end)
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
__forceinline void tm_avx512_r512_map_8::_alg_dispatch(
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

__forceinline void tm_avx512_r512_map_8::_run_map_entry_x4(
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
    // supplies the _mm_extract_epi8 immediate per case, so no unroll is needed
    // and the small loop body stays i-cache friendly.
#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;

        const __m128i al = _mm512_castsi512_si128(a0);
        const __m128i bl = _mm512_castsi512_si128(b0);
        const __m128i cl = _mm512_castsi512_si128(c0);
        const __m128i dl = _mm512_castsi512_si128(d0);
        uint8 ba = 0, bb = 0, bc = 0, bd = 0;
#define TM_MAP_X4_EXTRACT(out, r) \
        switch (i) { \
            case 0:  out=(uint8)_mm_extract_epi8(r,0); break; \
            case 1:  out=(uint8)_mm_extract_epi8(r,1); break; \
            case 2:  out=(uint8)_mm_extract_epi8(r,2); break; \
            case 3:  out=(uint8)_mm_extract_epi8(r,3); break; \
            case 4:  out=(uint8)_mm_extract_epi8(r,4); break; \
            case 5:  out=(uint8)_mm_extract_epi8(r,5); break; \
            case 6:  out=(uint8)_mm_extract_epi8(r,6); break; \
            case 7:  out=(uint8)_mm_extract_epi8(r,7); break; \
            case 8:  out=(uint8)_mm_extract_epi8(r,8); break; \
            case 9:  out=(uint8)_mm_extract_epi8(r,9); break; \
            case 10: out=(uint8)_mm_extract_epi8(r,10); break; \
            case 11: out=(uint8)_mm_extract_epi8(r,11); break; \
            case 12: out=(uint8)_mm_extract_epi8(r,12); break; \
            case 13: out=(uint8)_mm_extract_epi8(r,13); break; \
            case 14: out=(uint8)_mm_extract_epi8(r,14); break; \
            case 15: out=(uint8)_mm_extract_epi8(r,15); break; \
        }
        TM_MAP_X4_EXTRACT(ba, al);
        TM_MAP_X4_EXTRACT(bb, bl);
        TM_MAP_X4_EXTRACT(bc, cl);
        TM_MAP_X4_EXTRACT(bd, dl);
#undef TM_MAP_X4_EXTRACT
        if (nibble == 1) { ba >>= 4; bb >>= 4; bc >>= 4; bd >>= 4; }

        _alg_dispatch(a0, a1, (ba >> 1) & 0x07, pa, reg_base, alg0_base, alg6_base);
        _alg_dispatch(b0, b1, (bb >> 1) & 0x07, pb, reg_base, alg0_base, alg6_base);
        _alg_dispatch(c0, c1, (bc >> 1) & 0x07, pc, reg_base, alg0_base, alg6_base);
        _alg_dispatch(d0, d1, (bd >> 1) & 0x07, pd, reg_base, alg0_base, alg6_base);
    }
}

void tm_avx512_r512_map_8::run_maps_range_x4(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
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

    // Natural layout: in/out are canonical 128-byte states. Pool entries are not
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

// 8-way / 12-way natural-map interleave. Same structure as x4: each state has its
// own moving local_pos; natural layout so byte i is at low128(w0) byte i.
#define TM_MAP_XN_EXTRACT(out, r, i) \
        switch (i) { \
            case 0:  out=(uint8)_mm_extract_epi8(r,0); break;  case 1:  out=(uint8)_mm_extract_epi8(r,1); break; \
            case 2:  out=(uint8)_mm_extract_epi8(r,2); break;  case 3:  out=(uint8)_mm_extract_epi8(r,3); break; \
            case 4:  out=(uint8)_mm_extract_epi8(r,4); break;  case 5:  out=(uint8)_mm_extract_epi8(r,5); break; \
            case 6:  out=(uint8)_mm_extract_epi8(r,6); break;  case 7:  out=(uint8)_mm_extract_epi8(r,7); break; \
            case 8:  out=(uint8)_mm_extract_epi8(r,8); break;  case 9:  out=(uint8)_mm_extract_epi8(r,9); break; \
            case 10: out=(uint8)_mm_extract_epi8(r,10); break; case 11: out=(uint8)_mm_extract_epi8(r,11); break; \
            case 12: out=(uint8)_mm_extract_epi8(r,12); break; case 13: out=(uint8)_mm_extract_epi8(r,13); break; \
            case 14: out=(uint8)_mm_extract_epi8(r,14); break; case 15: out=(uint8)_mm_extract_epi8(r,15); break; \
	        }

namespace
{

constexpr uint32_t MAP1_SCORE_CACHE_BITS = 20;
constexpr uint32_t MAP1_SCORE_CACHE_SIZE = 1u << MAP1_SCORE_CACHE_BITS;
constexpr uint64_t MAP1_SCORE_CACHE_EMPTY = UINT64_MAX;

struct Map1ScoreCache
{
    std::array<uint64_t, MAP1_SCORE_CACHE_SIZE> keys;
    std::array<float, MAP1_SCORE_CACHE_SIZE> values;

    Map1ScoreCache()
    {
        keys.fill(MAP1_SCORE_CACHE_EMPTY);
    }
};

static thread_local Map1ScoreCache map1_score_cache;

__forceinline uint64_t pack_map1_ops(const int op[16])
{
    uint64_t packed = 0;
    for (int i = 0; i < 16; ++i)
        packed |= static_cast<uint64_t>(op[i] & 0x07) << (i * 3);
    return packed;
}

__forceinline float cached_shed_proxy_map1(const int op[16])
{
    const uint64_t packed = pack_map1_ops(op);
    const uint32_t slot = static_cast<uint32_t>((packed * 11400714819323198485ull) >> (64 - MAP1_SCORE_CACHE_BITS));
    if (map1_score_cache.keys[slot] == packed)
        return map1_score_cache.values[slot];

    const float score = tm_routing::shed_proxy_map1(op);
    map1_score_cache.keys[slot] = packed;
    map1_score_cache.values[slot] = score;
    return score;
}

}

__forceinline void tm_avx512_r512_map_8::_run_map_entry_x8(
    __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
    __m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
    int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa=2047,pb=2047,pc=2047,pd=2047,pe=2047,pf=2047,pg=2047,ph=2047;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i al=_mm512_castsi512_si128(a0),bl=_mm512_castsi512_si128(b0),cl=_mm512_castsi512_si128(c0),dl=_mm512_castsi512_si128(d0);
        const __m128i el=_mm512_castsi512_si128(e0),fl=_mm512_castsi512_si128(f0),gl=_mm512_castsi512_si128(g0),hl=_mm512_castsi512_si128(h0);
        uint8 ba=0,bb=0,bc=0,bd=0,be=0,bf=0,bg=0,bh=0;
        TM_MAP_XN_EXTRACT(ba,al,i); TM_MAP_XN_EXTRACT(bb,bl,i); TM_MAP_XN_EXTRACT(bc,cl,i); TM_MAP_XN_EXTRACT(bd,dl,i);
        TM_MAP_XN_EXTRACT(be,el,i); TM_MAP_XN_EXTRACT(bf,fl,i); TM_MAP_XN_EXTRACT(bg,gl,i); TM_MAP_XN_EXTRACT(bh,hl,i);
        if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4;be>>=4;bf>>=4;bg>>=4;bh>>=4; }
        _alg_dispatch(a0,a1,(ba>>1)&0x07,pa,reg_base,alg0_base,alg6_base);
        _alg_dispatch(b0,b1,(bb>>1)&0x07,pb,reg_base,alg0_base,alg6_base);
        _alg_dispatch(c0,c1,(bc>>1)&0x07,pc,reg_base,alg0_base,alg6_base);
        _alg_dispatch(d0,d1,(bd>>1)&0x07,pd,reg_base,alg0_base,alg6_base);
        _alg_dispatch(e0,e1,(be>>1)&0x07,pe,reg_base,alg0_base,alg6_base);
        _alg_dispatch(f0,f1,(bf>>1)&0x07,pf,reg_base,alg0_base,alg6_base);
        _alg_dispatch(g0,g1,(bg>>1)&0x07,pg,reg_base,alg0_base,alg6_base);
        _alg_dispatch(h0,h1,(bh>>1)&0x07,ph,reg_base,alg0_base,alg6_base);
    }
}

__forceinline void tm_avx512_r512_map_8::_run_map1_entry_x8_scores(
    __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
    __m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
    int map_idx, float scores[8])
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa=2047,pb=2047,pc=2047,pd=2047,pe=2047,pf=2047,pg=2047,ph=2047;
    int opa[16], opb[16], opc[16], opd[16], ope[16], opf[16], opg[16], oph[16];

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i al=_mm512_castsi512_si128(a0),bl=_mm512_castsi512_si128(b0),cl=_mm512_castsi512_si128(c0),dl=_mm512_castsi512_si128(d0);
        const __m128i el=_mm512_castsi512_si128(e0),fl=_mm512_castsi512_si128(f0),gl=_mm512_castsi512_si128(g0),hl=_mm512_castsi512_si128(h0);
        uint8 ba=0,bb=0,bc=0,bd=0,be=0,bf=0,bg=0,bh=0;
        TM_MAP_XN_EXTRACT(ba,al,i); TM_MAP_XN_EXTRACT(bb,bl,i); TM_MAP_XN_EXTRACT(bc,cl,i); TM_MAP_XN_EXTRACT(bd,dl,i);
        TM_MAP_XN_EXTRACT(be,el,i); TM_MAP_XN_EXTRACT(bf,fl,i); TM_MAP_XN_EXTRACT(bg,gl,i); TM_MAP_XN_EXTRACT(bh,hl,i);
        if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4;be>>=4;bf>>=4;bg>>=4;bh>>=4; }
        opa[i]=(ba>>1)&0x07; opb[i]=(bb>>1)&0x07; opc[i]=(bc>>1)&0x07; opd[i]=(bd>>1)&0x07;
        ope[i]=(be>>1)&0x07; opf[i]=(bf>>1)&0x07; opg[i]=(bg>>1)&0x07; oph[i]=(bh>>1)&0x07;
        _alg_dispatch(a0,a1,opa[i],pa,reg_base,alg0_base,alg6_base);
        _alg_dispatch(b0,b1,opb[i],pb,reg_base,alg0_base,alg6_base);
        _alg_dispatch(c0,c1,opc[i],pc,reg_base,alg0_base,alg6_base);
        _alg_dispatch(d0,d1,opd[i],pd,reg_base,alg0_base,alg6_base);
        _alg_dispatch(e0,e1,ope[i],pe,reg_base,alg0_base,alg6_base);
        _alg_dispatch(f0,f1,opf[i],pf,reg_base,alg0_base,alg6_base);
        _alg_dispatch(g0,g1,opg[i],pg,reg_base,alg0_base,alg6_base);
        _alg_dispatch(h0,h1,oph[i],ph,reg_base,alg0_base,alg6_base);
    }

    scores[0] = cached_shed_proxy_map1(opa);
    scores[1] = cached_shed_proxy_map1(opb);
    scores[2] = cached_shed_proxy_map1(opc);
    scores[3] = cached_shed_proxy_map1(opd);
    scores[4] = cached_shed_proxy_map1(ope);
    scores[5] = cached_shed_proxy_map1(opf);
    scores[6] = cached_shed_proxy_map1(opg);
    scores[7] = cached_shed_proxy_map1(oph);
}

// Same as _run_map_entry_x8 but also accumulates a per-lane alg0 count as a FREE
// byproduct: the live alg_id (bX>>1)&7 is already materialized for the dispatch,
// so the count is one compare+add per dispatch, hidden in the SIMD shadow. This
// is the in-kernel replacement for the separate scalar raw_alg0_count_for_entry
// pre-pass (which re-reads the pooled state and breaks the x8 batch). NOTE the
// count is of the LIVE ops executed by THIS map (a valid, arguably better route
// signal), not the input-state proxy raw_alg0 computes; routing never drops a
// state, so the final union is unaffected by which metric is used.
__forceinline void tm_avx512_r512_map_8::_run_map_entry_x8_alg0count(
    __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
    __m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
    int map_idx, int counts[8])
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa=2047,pb=2047,pc=2047,pd=2047,pe=2047,pf=2047,pg=2047,ph=2047;
    int ca=0,cb=0,cc=0,cd=0,ce=0,cf=0,cg=0,ch=0;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i al=_mm512_castsi512_si128(a0),bl=_mm512_castsi512_si128(b0),cl=_mm512_castsi512_si128(c0),dl=_mm512_castsi512_si128(d0);
        const __m128i el=_mm512_castsi512_si128(e0),fl=_mm512_castsi512_si128(f0),gl=_mm512_castsi512_si128(g0),hl=_mm512_castsi512_si128(h0);
        uint8 ba=0,bb=0,bc=0,bd=0,be=0,bf=0,bg=0,bh=0;
        TM_MAP_XN_EXTRACT(ba,al,i); TM_MAP_XN_EXTRACT(bb,bl,i); TM_MAP_XN_EXTRACT(bc,cl,i); TM_MAP_XN_EXTRACT(bd,dl,i);
        TM_MAP_XN_EXTRACT(be,el,i); TM_MAP_XN_EXTRACT(bf,fl,i); TM_MAP_XN_EXTRACT(bg,gl,i); TM_MAP_XN_EXTRACT(bh,hl,i);
        if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4;be>>=4;bf>>=4;bg>>=4;bh>>=4; }
        int oa=(ba>>1)&0x07,ob=(bb>>1)&0x07,oc=(bc>>1)&0x07,od=(bd>>1)&0x07;
        int oe=(be>>1)&0x07,of=(bf>>1)&0x07,og=(bg>>1)&0x07,oh=(bh>>1)&0x07;
        ca+=(oa==0);cb+=(ob==0);cc+=(oc==0);cd+=(od==0);ce+=(oe==0);cf+=(of==0);cg+=(og==0);ch+=(oh==0);
        _alg_dispatch(a0,a1,oa,pa,reg_base,alg0_base,alg6_base);
        _alg_dispatch(b0,b1,ob,pb,reg_base,alg0_base,alg6_base);
        _alg_dispatch(c0,c1,oc,pc,reg_base,alg0_base,alg6_base);
        _alg_dispatch(d0,d1,od,pd,reg_base,alg0_base,alg6_base);
        _alg_dispatch(e0,e1,oe,pe,reg_base,alg0_base,alg6_base);
        _alg_dispatch(f0,f1,of,pf,reg_base,alg0_base,alg6_base);
        _alg_dispatch(g0,g1,og,pg,reg_base,alg0_base,alg6_base);
        _alg_dispatch(h0,h1,oh,ph,reg_base,alg0_base,alg6_base);
    }
    counts[0]=ca;counts[1]=cb;counts[2]=cc;counts[3]=cd;counts[4]=ce;counts[5]=cf;counts[6]=cg;counts[7]=ch;
}

// Op-tail capture entry: rolls the dispatched alg_id into a per-lane key
// (key = key<<3 | alg) over the 16 positions; low 24 bits = the last-8-op trajDens key.
// Emits the routing gate's op-tail key as a dispatch byproduct (used on the FINAL map
// of a group). Same per-dispatch accumulator shape as the alg0 counter.
__forceinline void tm_avx512_r512_map_8::_run_map_entry_x8_optail(
    __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
    __m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
    int map_idx, std::uint32_t keys[8], int counts[8])
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa=2047,pb=2047,pc=2047,pd=2047,pe=2047,pf=2047,pg=2047,ph=2047;
    std::uint32_t ka=0,kb=0,kc=0,kd=0,ke=0,kf=0,kg=0,kh=0;
    int ca=0,cb=0,cc=0,cd=0,ce=0,cf=0,cg=0,ch=0;  // per-map alg0 count (for lseA0)

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i al=_mm512_castsi512_si128(a0),bl=_mm512_castsi512_si128(b0),cl=_mm512_castsi512_si128(c0),dl=_mm512_castsi512_si128(d0);
        const __m128i el=_mm512_castsi512_si128(e0),fl=_mm512_castsi512_si128(f0),gl=_mm512_castsi512_si128(g0),hl=_mm512_castsi512_si128(h0);
        uint8 ba=0,bb=0,bc=0,bd=0,be=0,bf=0,bg=0,bh=0;
        TM_MAP_XN_EXTRACT(ba,al,i); TM_MAP_XN_EXTRACT(bb,bl,i); TM_MAP_XN_EXTRACT(bc,cl,i); TM_MAP_XN_EXTRACT(bd,dl,i);
        TM_MAP_XN_EXTRACT(be,el,i); TM_MAP_XN_EXTRACT(bf,fl,i); TM_MAP_XN_EXTRACT(bg,gl,i); TM_MAP_XN_EXTRACT(bh,hl,i);
        if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4;be>>=4;bf>>=4;bg>>=4;bh>>=4; }
        int oa=(ba>>1)&0x07,ob=(bb>>1)&0x07,oc=(bc>>1)&0x07,od=(bd>>1)&0x07;
        int oe=(be>>1)&0x07,of=(bf>>1)&0x07,og=(bg>>1)&0x07,oh=(bh>>1)&0x07;
        ka=(ka<<3)|oa; kb=(kb<<3)|ob; kc=(kc<<3)|oc; kd=(kd<<3)|od;
        ke=(ke<<3)|oe; kf=(kf<<3)|of; kg=(kg<<3)|og; kh=(kh<<3)|oh;
        ca+=(oa==0);cb+=(ob==0);cc+=(oc==0);cd+=(od==0);ce+=(oe==0);cf+=(of==0);cg+=(og==0);ch+=(oh==0);
        _alg_dispatch(a0,a1,oa,pa,reg_base,alg0_base,alg6_base);
        _alg_dispatch(b0,b1,ob,pb,reg_base,alg0_base,alg6_base);
        _alg_dispatch(c0,c1,oc,pc,reg_base,alg0_base,alg6_base);
        _alg_dispatch(d0,d1,od,pd,reg_base,alg0_base,alg6_base);
        _alg_dispatch(e0,e1,oe,pe,reg_base,alg0_base,alg6_base);
        _alg_dispatch(f0,f1,of,pf,reg_base,alg0_base,alg6_base);
        _alg_dispatch(g0,g1,og,pg,reg_base,alg0_base,alg6_base);
        _alg_dispatch(h0,h1,oh,ph,reg_base,alg0_base,alg6_base);
    }
    const std::uint32_t M=0xFFFFFFu;
    keys[0]=ka&M;keys[1]=kb&M;keys[2]=kc&M;keys[3]=kd&M;keys[4]=ke&M;keys[5]=kf&M;keys[6]=kg&M;keys[7]=kh&M;
    counts[0]=ca;counts[1]=cb;counts[2]=cc;counts[3]=cd;counts[4]=ce;counts[5]=cf;counts[6]=cg;counts[7]=ch;
}

// EXPERIMENT: branchless dispatch for the 5 arithmetic algs (0,1,3,4,6) via an
// index-select over precomputed candidates (no data-dependent branch); butterflies
// (2,5) and xor7 (7) stay branched. Goal: measure how much of the ~1.8B alg-dispatch
// mispredict this removes and what register pressure the 5 candidate ZMMs/lane add
// (the long-standing "too much pressure" dismissal). Bit-identical to _alg_dispatch.
__forceinline void tm_avx512_r512_map_8::_alg_dispatch_blarith(
    __m512i& w0, __m512i& w1, int alg_id, uint16& local_pos,
    const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base)
{
    if (alg_id == 2) { alg_2(w0, w1, reg_base[local_pos] >> 7);  local_pos -= 1; return; }
    if (alg_id == 5) { alg_5(w0, w1, reg_base[local_pos] & 0x80); local_pos -= 1; return; }
    if (alg_id == 7) { alg_7(w0, w1); return; }
    // arith {0,1,3,4,6}: compute all candidates, select by index (branchless).
    const uint8* rb = reg_base  + local_pos - 127;
    const uint8* zb = alg0_base + local_pos - 127;
    const uint8* sb = alg6_base + (2047 - local_pos);
    const __m512i R0=_mm512_loadu_si512((const void*)rb),     R1=_mm512_loadu_si512((const void*)(rb+64));
    const __m512i Z0=_mm512_loadu_si512((const void*)zb),     Z1=_mm512_loadu_si512((const void*)(zb+64));
    const __m512i S0=_mm512_loadu_si512((const void*)sb),     S1=_mm512_loadu_si512((const void*)(sb+64));
    __m512i c0[7], c1[7];
    c0[0]=_mm512_ternarylogic_epi32(_mm512_slli_epi64(w0,1), mask_FE, Z0, 0xEA);  // alg0
    c1[0]=_mm512_ternarylogic_epi32(_mm512_slli_epi64(w1,1), mask_FE, Z1, 0xEA);
    c0[1]=_mm512_add_epi8(w0,R0);  c1[1]=_mm512_add_epi8(w1,R1);                   // alg1
    c0[3]=_mm512_xor_si512(w0,R0); c1[3]=_mm512_xor_si512(w1,R1);                  // alg3
    c0[4]=_mm512_sub_epi8(w0,R0);  c1[4]=_mm512_sub_epi8(w1,R1);                   // alg4
    c0[6]=_mm512_ternarylogic_epi32(_mm512_srli_epi64(w0,1), mask_7F, S0, 0xEA);  // alg6
    c1[6]=_mm512_ternarylogic_epi32(_mm512_srli_epi64(w1,1), mask_7F, S1, 0xEA);
    w0 = c0[alg_id]; w1 = c1[alg_id];
    local_pos -= 128;
}

__forceinline void tm_avx512_r512_map_8::_run_map_entry_x8_blarith(
    __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
    __m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
    int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa=2047,pb=2047,pc=2047,pd=2047,pe=2047,pf=2047,pg=2047,ph=2047;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i al=_mm512_castsi512_si128(a0),bl=_mm512_castsi512_si128(b0),cl=_mm512_castsi512_si128(c0),dl=_mm512_castsi512_si128(d0);
        const __m128i el=_mm512_castsi512_si128(e0),fl=_mm512_castsi512_si128(f0),gl=_mm512_castsi512_si128(g0),hl=_mm512_castsi512_si128(h0);
        uint8 ba=0,bb=0,bc=0,bd=0,be=0,bf=0,bg=0,bh=0;
        TM_MAP_XN_EXTRACT(ba,al,i); TM_MAP_XN_EXTRACT(bb,bl,i); TM_MAP_XN_EXTRACT(bc,cl,i); TM_MAP_XN_EXTRACT(bd,dl,i);
        TM_MAP_XN_EXTRACT(be,el,i); TM_MAP_XN_EXTRACT(bf,fl,i); TM_MAP_XN_EXTRACT(bg,gl,i); TM_MAP_XN_EXTRACT(bh,hl,i);
        if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4;be>>=4;bf>>=4;bg>>=4;bh>>=4; }
        _alg_dispatch_blarith(a0,a1,(ba>>1)&0x07,pa,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blarith(b0,b1,(bb>>1)&0x07,pb,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blarith(c0,c1,(bc>>1)&0x07,pc,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blarith(d0,d1,(bd>>1)&0x07,pd,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blarith(e0,e1,(be>>1)&0x07,pe,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blarith(f0,f1,(bf>>1)&0x07,pf,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blarith(g0,g1,(bg>>1)&0x07,pg,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blarith(h0,h1,(bh>>1)&0x07,ph,reg_base,alg0_base,alg6_base);
    }
}

// EXPERIMENT v2: merged branchless dispatch. Exploits that the arith algs come in
// op-pairs that share work and differ only by a selected operand/parameter, folding
// 6 algs (0,1,3,4,6,7) into 3 candidates, and selects with a register blend-tree
// (no stack-indexed array) — both moves cut the register pressure that made the v1
// "compute-all-5 + array" branchless spill. Pairs:
//   {1,4}: w + (R or -R)          (add/sub, sign-selected operand)
//   {3,7}: w ^ (R or 0xFF)        (xor, operand-selected)
//   {0,6}: (w shifted) ternlog rng (shift dir + mask + rng selected)
// Butterflies (2,5) stay branched. Bit-identical to _alg_dispatch.
__forceinline void tm_avx512_r512_map_8::_alg_dispatch_blmerge(
    __m512i& w0, __m512i& w1, int alg_id, uint16& local_pos,
    const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base)
{
    if (alg_id == 2) { alg_2(w0, w1, reg_base[local_pos] >> 7);  local_pos -= 1; return; }
    if (alg_id == 5) { alg_5(w0, w1, reg_base[local_pos] & 0x80); local_pos -= 1; return; }
    const uint8* rb = reg_base  + local_pos - 127;
    const uint8* zb = alg0_base + local_pos - 127;
    const uint8* sb = alg6_base + (2047 - local_pos);
    const __mmask64 m6  = (alg_id==6) ? ~0ull : 0ull;                 // right-shift vs left
    const __mmask64 m4  = (alg_id==4) ? ~0ull : 0ull;                 // negate operand (sub)
    const __mmask64 m7  = (alg_id==7) ? ~0ull : 0ull;                 // xor with 0xFF
    const __mmask64 mas = (alg_id==1||alg_id==4) ? ~0ull : 0ull;      // pick add/sub candidate
    const __mmask64 mx  = (alg_id==3||alg_id==7) ? ~0ull : 0ull;      // pick xor candidate
    const __m512i zero = _mm512_setzero_si512();
    {   // half 0
        const __m512i R=_mm512_loadu_si512((const void*)rb), Z=_mm512_loadu_si512((const void*)zb), S=_mm512_loadu_si512((const void*)sb);
        const __m512i shifted=_mm512_mask_blend_epi8(m6,_mm512_slli_epi64(w0,1),_mm512_srli_epi64(w0,1));
        const __m512i smask  =_mm512_mask_blend_epi8(m6,mask_FE,mask_7F);
        const __m512i srng   =_mm512_mask_blend_epi8(m6,Z,S);
        const __m512i c_sh=_mm512_ternarylogic_epi32(shifted,smask,srng,0xEA);
        const __m512i c_as=_mm512_add_epi8(w0,_mm512_mask_blend_epi8(m4,R,_mm512_sub_epi8(zero,R)));
        const __m512i c_x =_mm512_xor_si512(w0,_mm512_mask_blend_epi8(m7,R,mask_FF));
        __m512i res=_mm512_mask_blend_epi8(mas,c_sh,c_as);
        w0=_mm512_mask_blend_epi8(mx,res,c_x);
    }
    {   // half 1
        const __m512i R=_mm512_loadu_si512((const void*)(rb+64)), Z=_mm512_loadu_si512((const void*)(zb+64)), S=_mm512_loadu_si512((const void*)(sb+64));
        const __m512i shifted=_mm512_mask_blend_epi8(m6,_mm512_slli_epi64(w1,1),_mm512_srli_epi64(w1,1));
        const __m512i smask  =_mm512_mask_blend_epi8(m6,mask_FE,mask_7F);
        const __m512i srng   =_mm512_mask_blend_epi8(m6,Z,S);
        const __m512i c_sh=_mm512_ternarylogic_epi32(shifted,smask,srng,0xEA);
        const __m512i c_as=_mm512_add_epi8(w1,_mm512_mask_blend_epi8(m4,R,_mm512_sub_epi8(zero,R)));
        const __m512i c_x =_mm512_xor_si512(w1,_mm512_mask_blend_epi8(m7,R,mask_FF));
        __m512i res=_mm512_mask_blend_epi8(mas,c_sh,c_as);
        w1=_mm512_mask_blend_epi8(mx,res,c_x);
    }
    // algs 0,1,3,4,6 advance pos by 128; alg_7 (folded into the xor candidate) must
    // NOT advance it. Branchless: subtract 128 unless alg_id==7.
    local_pos -= static_cast<uint16>((alg_id != 7) * 128);
}

// W6: rematerialized-mask blmerge dispatch — masks passed BY REFERENCE (address-taken) so the compiler
// reloads them from the caller's stack slot under pressure instead of pinning 3 ZMM. This is the universal
// kernel's x12 lever (frees registers for 24 state ZMM). Identical math to _alg_dispatch_blmerge.
__forceinline void tm_avx512_r512_map_8::_alg_dispatch_blmerge_rm(
    __m512i& w0, __m512i& w1, int alg_id, uint16& local_pos,
    const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base,
    __m512i& mFE, __m512i& m7F, __m512i& mFF)
{
    if (alg_id == 2) { alg_2(w0, w1, reg_base[local_pos] >> 7);  local_pos -= 1; return; }
    if (alg_id == 5) { alg_5(w0, w1, reg_base[local_pos] & 0x80); local_pos -= 1; return; }
    const uint8* rb = reg_base  + local_pos - 127;
    const uint8* zb = alg0_base + local_pos - 127;
    const uint8* sb = alg6_base + (2047 - local_pos);
    const __mmask64 m6  = (alg_id==6) ? ~0ull : 0ull;
    const __mmask64 m4  = (alg_id==4) ? ~0ull : 0ull;
    const __mmask64 m7  = (alg_id==7) ? ~0ull : 0ull;
    const __mmask64 mas = (alg_id==1||alg_id==4) ? ~0ull : 0ull;
    const __mmask64 mx  = (alg_id==3||alg_id==7) ? ~0ull : 0ull;
    const __m512i zero = _mm512_setzero_si512();
    {   // half 0
        const __m512i R=_mm512_loadu_si512((const void*)rb), Z=_mm512_loadu_si512((const void*)zb), S=_mm512_loadu_si512((const void*)sb);
        const __m512i shifted=_mm512_mask_blend_epi8(m6,_mm512_slli_epi64(w0,1),_mm512_srli_epi64(w0,1));
        const __m512i smask  =_mm512_mask_blend_epi8(m6,mFE,m7F);
        const __m512i srng   =_mm512_mask_blend_epi8(m6,Z,S);
        const __m512i c_sh=_mm512_ternarylogic_epi32(shifted,smask,srng,0xEA);
        const __m512i c_as=_mm512_add_epi8(w0,_mm512_mask_blend_epi8(m4,R,_mm512_sub_epi8(zero,R)));
        const __m512i c_x =_mm512_xor_si512(w0,_mm512_mask_blend_epi8(m7,R,mFF));
        __m512i res=_mm512_mask_blend_epi8(mas,c_sh,c_as);
        w0=_mm512_mask_blend_epi8(mx,res,c_x);
    }
    {   // half 1
        const __m512i R=_mm512_loadu_si512((const void*)(rb+64)), Z=_mm512_loadu_si512((const void*)(zb+64)), S=_mm512_loadu_si512((const void*)(sb+64));
        const __m512i shifted=_mm512_mask_blend_epi8(m6,_mm512_slli_epi64(w1,1),_mm512_srli_epi64(w1,1));
        const __m512i smask  =_mm512_mask_blend_epi8(m6,mFE,m7F);
        const __m512i srng   =_mm512_mask_blend_epi8(m6,Z,S);
        const __m512i c_sh=_mm512_ternarylogic_epi32(shifted,smask,srng,0xEA);
        const __m512i c_as=_mm512_add_epi8(w1,_mm512_mask_blend_epi8(m4,R,_mm512_sub_epi8(zero,R)));
        const __m512i c_x =_mm512_xor_si512(w1,_mm512_mask_blend_epi8(m7,R,mFF));
        __m512i res=_mm512_mask_blend_epi8(mas,c_sh,c_as);
        w1=_mm512_mask_blend_epi8(mx,res,c_x);
    }
    local_pos -= static_cast<uint16>((alg_id != 7) * 128);
}

__forceinline void tm_avx512_r512_map_8::_run_map_entry_x8_blmerge(
    __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
    __m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
    int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa=2047,pb=2047,pc=2047,pd=2047,pe=2047,pf=2047,pg=2047,ph=2047;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i al=_mm512_castsi512_si128(a0),bl=_mm512_castsi512_si128(b0),cl=_mm512_castsi512_si128(c0),dl=_mm512_castsi512_si128(d0);
        const __m128i el=_mm512_castsi512_si128(e0),fl=_mm512_castsi512_si128(f0),gl=_mm512_castsi512_si128(g0),hl=_mm512_castsi512_si128(h0);
        uint8 ba=0,bb=0,bc=0,bd=0,be=0,bf=0,bg=0,bh=0;
        TM_MAP_XN_EXTRACT(ba,al,i); TM_MAP_XN_EXTRACT(bb,bl,i); TM_MAP_XN_EXTRACT(bc,cl,i); TM_MAP_XN_EXTRACT(bd,dl,i);
        TM_MAP_XN_EXTRACT(be,el,i); TM_MAP_XN_EXTRACT(bf,fl,i); TM_MAP_XN_EXTRACT(bg,gl,i); TM_MAP_XN_EXTRACT(bh,hl,i);
        if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4;be>>=4;bf>>=4;bg>>=4;bh>>=4; }
        _alg_dispatch_blmerge(a0,a1,(ba>>1)&0x07,pa,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blmerge(b0,b1,(bb>>1)&0x07,pb,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blmerge(c0,c1,(bc>>1)&0x07,pc,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blmerge(d0,d1,(bd>>1)&0x07,pd,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blmerge(e0,e1,(be>>1)&0x07,pe,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blmerge(f0,f1,(bf>>1)&0x07,pf,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blmerge(g0,g1,(bg>>1)&0x07,pg,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blmerge(h0,h1,(bh>>1)&0x07,ph,reg_base,alg0_base,alg6_base);
    }
}

// EXPERIMENT v3: FULL all-8 branchless dispatch. Folds the butterflies {2,5} in too
// (computed on copies), so there are ZERO data-dependent branches in the dispatch.
// 5 merged candidate-pairs {sh(0,6), as(1,4), x(3,7), bf2(2), bf5(5)} + a 4-step
// blend-tree; pos advance is branchless ({2,5}->1, {7}->0, else 128). The two extra
// butterflies (cross-lane vpermi2b) are computed every dispatch, so this is the
// upper bound on branchless work/pressure. Bit-identical to _alg_dispatch.
// NOT force-inlined: folding 8 copies of this (with the butterflies + 80-ZMM spill)
// into the x8 entry produced broken codegen; a real call bounds the per-lane frame.
void tm_avx512_r512_map_8::_alg_dispatch_blall(
    __m512i& w0, __m512i& w1, int alg_id, uint16& local_pos,
    const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base)
{
    // We compute ALL candidates' operands unconditionally (the actual alg only needs
    // one), so the addresses must be clamped to valid range for the algs that are NOT
    // selected — the loaded values are discarded by the blend. Two out-of-range cases:
    //   - pos < 127 (butterfly territory): arith block base+pos-127 underflows.
    //   - pos > 2047: a prior real arith op at pos==127 did pos-=128 -> uint16 wrap to
    //     65535; a trailing non-arith op then sees the wrapped pos.
    // For the actual arith algs the schedule guarantees pos in [127,2047], so the clamp
    // is a no-op there (correctness preserved). The carry byte (used only by the real
    // butterfly) needs the true low pos, so only high-clamp it.
    const uint16 sp = local_pos < 127 ? static_cast<uint16>(127)
                    : (local_pos > 2047 ? static_cast<uint16>(2047) : local_pos);
    const uint16 cp = local_pos > 2047 ? static_cast<uint16>(2047) : local_pos;
    const uint8* rb = reg_base  + sp - 127;
    const uint8* zb = alg0_base + sp - 127;
    const uint8* sb = alg6_base + (2047 - sp);
    const uint8 carry2 = static_cast<uint8>(reg_base[cp] >> 7);
    const uint8 carry5 = static_cast<uint8>(reg_base[cp] & 0x80);
    const __mmask64 m6  = (alg_id==6) ? ~0ull : 0ull;
    const __mmask64 m4  = (alg_id==4) ? ~0ull : 0ull;
    const __mmask64 m7  = (alg_id==7) ? ~0ull : 0ull;
    const __mmask64 mas = (alg_id==1||alg_id==4) ? ~0ull : 0ull;
    const __mmask64 mx  = (alg_id==3||alg_id==7) ? ~0ull : 0ull;
    const __mmask64 m2  = (alg_id==2) ? ~0ull : 0ull;
    const __mmask64 m5  = (alg_id==5) ? ~0ull : 0ull;
    const __m512i zero = _mm512_setzero_si512();
    // butterflies on copies (couple both halves), arith candidates per half.
    __m512i bf2_0=w0, bf2_1=w1; alg_2(bf2_0, bf2_1, carry2);
    __m512i bf5_0=w0, bf5_1=w1; alg_5(bf5_0, bf5_1, carry5);
    {   // half 0
        const __m512i R=_mm512_loadu_si512((const void*)rb), Z=_mm512_loadu_si512((const void*)zb), S=_mm512_loadu_si512((const void*)sb);
        const __m512i sh=_mm512_ternarylogic_epi32(_mm512_mask_blend_epi8(m6,_mm512_slli_epi64(w0,1),_mm512_srli_epi64(w0,1)),_mm512_mask_blend_epi8(m6,mask_FE,mask_7F),_mm512_mask_blend_epi8(m6,Z,S),0xEA);
        const __m512i as=_mm512_add_epi8(w0,_mm512_mask_blend_epi8(m4,R,_mm512_sub_epi8(zero,R)));
        const __m512i xx=_mm512_xor_si512(w0,_mm512_mask_blend_epi8(m7,R,mask_FF));
        __m512i res=_mm512_mask_blend_epi8(mas,sh,as);
        res=_mm512_mask_blend_epi8(mx,res,xx);
        res=_mm512_mask_blend_epi8(m2,res,bf2_0);
        w0 =_mm512_mask_blend_epi8(m5,res,bf5_0);
    }
    {   // half 1
        const __m512i R=_mm512_loadu_si512((const void*)(rb+64)), Z=_mm512_loadu_si512((const void*)(zb+64)), S=_mm512_loadu_si512((const void*)(sb+64));
        const __m512i sh=_mm512_ternarylogic_epi32(_mm512_mask_blend_epi8(m6,_mm512_slli_epi64(w1,1),_mm512_srli_epi64(w1,1)),_mm512_mask_blend_epi8(m6,mask_FE,mask_7F),_mm512_mask_blend_epi8(m6,Z,S),0xEA);
        const __m512i as=_mm512_add_epi8(w1,_mm512_mask_blend_epi8(m4,R,_mm512_sub_epi8(zero,R)));
        const __m512i xx=_mm512_xor_si512(w1,_mm512_mask_blend_epi8(m7,R,mask_FF));
        __m512i res=_mm512_mask_blend_epi8(mas,sh,as);
        res=_mm512_mask_blend_epi8(mx,res,xx);
        res=_mm512_mask_blend_epi8(m2,res,bf2_1);
        w1 =_mm512_mask_blend_epi8(m5,res,bf5_1);
    }
    // pos advance: {2,5}->1, {7}->0, arith{0,1,3,4,6}->128. Branchless.
    const bool isbf = (alg_id==2)||(alg_id==5);
    const uint16 adv = isbf ? 1u : ((alg_id==7) ? 0u : 128u);
    local_pos -= adv;
}

__forceinline void tm_avx512_r512_map_8::_run_map_entry_x8_blall(
    __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
    __m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
    int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa=2047,pb=2047,pc=2047,pd=2047,pe=2047,pf=2047,pg=2047,ph=2047;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i al=_mm512_castsi512_si128(a0),bl=_mm512_castsi512_si128(b0),cl=_mm512_castsi512_si128(c0),dl=_mm512_castsi512_si128(d0);
        const __m128i el=_mm512_castsi512_si128(e0),fl=_mm512_castsi512_si128(f0),gl=_mm512_castsi512_si128(g0),hl=_mm512_castsi512_si128(h0);
        uint8 ba=0,bb=0,bc=0,bd=0,be=0,bf=0,bg=0,bh=0;
        TM_MAP_XN_EXTRACT(ba,al,i); TM_MAP_XN_EXTRACT(bb,bl,i); TM_MAP_XN_EXTRACT(bc,cl,i); TM_MAP_XN_EXTRACT(bd,dl,i);
        TM_MAP_XN_EXTRACT(be,el,i); TM_MAP_XN_EXTRACT(bf,fl,i); TM_MAP_XN_EXTRACT(bg,gl,i); TM_MAP_XN_EXTRACT(bh,hl,i);
        if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4;be>>=4;bf>>=4;bg>>=4;bh>>=4; }
        _alg_dispatch_blall(a0,a1,(ba>>1)&0x07,pa,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blall(b0,b1,(bb>>1)&0x07,pb,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blall(c0,c1,(bc>>1)&0x07,pc,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blall(d0,d1,(bd>>1)&0x07,pd,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blall(e0,e1,(be>>1)&0x07,pe,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blall(f0,f1,(bf>>1)&0x07,pf,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blall(g0,g1,(bg>>1)&0x07,pg,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blall(h0,h1,(bh>>1)&0x07,ph,reg_base,alg0_base,alg6_base);
    }
}

// Single-lane (x1) map-range runner, dispatcher selected at compile time. Used to
// measure dispatch variants at zero interleave (no register-pressure confound: only
// 2 state ZMM live, so even compute-all candidates fit and never spill).
template<int MODE>
__forceinline void tm_avx512_r512_map_8::_run_maps_range_x1_tpl(
    const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in, uint8* out)
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i w0=_mm512_loadu_si512((const void*)in), w1=_mm512_loadu_si512((const void*)(in+64));
    for (std::size_t map_idx = local_begin; map_idx < local_end; ++map_idx)
    {
        const auto& t = _t();
        const uint8* reg_base  = t.reg_table  + map_idx * 2048;
        const uint8* alg0_base = t.alg0_table + map_idx * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
        const uint8* alg6_base = t.alg6_table + map_idx * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
        uint16 nibble_selector = t.nibble_selectors[map_idx];
        uint16 pa = 2047;
        for (int i = 0; i < 16; i++)
        {
            uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
            nibble_selector <<= 1;
            const __m128i al=_mm512_castsi512_si128(w0);
            uint8 ba=0; TM_MAP_XN_EXTRACT(ba,al,i);
            if (nibble == 1) ba>>=4;
            const int alg=(ba>>1)&0x07;
            if constexpr (MODE==0)      _alg_dispatch(w0,w1,alg,pa,reg_base,alg0_base,alg6_base);
            else if constexpr (MODE==1) _alg_dispatch_blarith(w0,w1,alg,pa,reg_base,alg0_base,alg6_base);
            else if constexpr (MODE==2) _alg_dispatch_blmerge(w0,w1,alg,pa,reg_base,alg0_base,alg6_base);
            else                        _alg_dispatch_blall(w0,w1,alg,pa,reg_base,alg0_base,alg6_base);
        }
    }
    _mm512_storeu_si512((void*)out, w0); _mm512_storeu_si512((void*)(out+64), w1);
}

// x4 branchless-arith entry (lower interleave: 5 candidate ZMMs over only 4 lanes,
// so it should stay inside the ZMM file and avoid the x8 blarith spill).
__forceinline void tm_avx512_r512_map_8::_run_map_entry_x4_blarith(
    __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
    __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
    int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa=2047,pb=2047,pc=2047,pd=2047;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i al=_mm512_castsi512_si128(a0),bl=_mm512_castsi512_si128(b0),cl=_mm512_castsi512_si128(c0),dl=_mm512_castsi512_si128(d0);
        uint8 ba=0,bb=0,bc=0,bd=0;
        TM_MAP_XN_EXTRACT(ba,al,i); TM_MAP_XN_EXTRACT(bb,bl,i); TM_MAP_XN_EXTRACT(bc,cl,i); TM_MAP_XN_EXTRACT(bd,dl,i);
        if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4; }
        _alg_dispatch_blarith(a0,a1,(ba>>1)&0x07,pa,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blarith(b0,b1,(bb>>1)&0x07,pb,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blarith(c0,c1,(bc>>1)&0x07,pc,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blarith(d0,d1,(bd>>1)&0x07,pd,reg_base,alg0_base,alg6_base);
    }
}

// x4 alg0-count variant (lower-interleave control for the register-pressure test:
// 4 lanes => 4 local_pos + 4 counters, well inside the GP file, so it should not
// add the spills the x8 count variant does).
__forceinline void tm_avx512_r512_map_8::_run_map_entry_x4_alg0count(
    __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
    __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
    int map_idx, int counts[4])
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa=2047,pb=2047,pc=2047,pd=2047;
    int ca=0,cb=0,cc=0,cd=0;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i al=_mm512_castsi512_si128(a0),bl=_mm512_castsi512_si128(b0),cl=_mm512_castsi512_si128(c0),dl=_mm512_castsi512_si128(d0);
        uint8 ba=0,bb=0,bc=0,bd=0;
        TM_MAP_XN_EXTRACT(ba,al,i); TM_MAP_XN_EXTRACT(bb,bl,i); TM_MAP_XN_EXTRACT(bc,cl,i); TM_MAP_XN_EXTRACT(bd,dl,i);
        if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4; }
        int oa=(ba>>1)&0x07,ob=(bb>>1)&0x07,oc=(bc>>1)&0x07,od=(bd>>1)&0x07;
        ca+=(oa==0);cb+=(ob==0);cc+=(oc==0);cd+=(od==0);
        _alg_dispatch(a0,a1,oa,pa,reg_base,alg0_base,alg6_base);
        _alg_dispatch(b0,b1,ob,pb,reg_base,alg0_base,alg6_base);
        _alg_dispatch(c0,c1,oc,pc,reg_base,alg0_base,alg6_base);
        _alg_dispatch(d0,d1,od,pd,reg_base,alg0_base,alg6_base);
    }
    counts[0]=ca;counts[1]=cb;counts[2]=cc;counts[3]=cd;
}

// 10-way: 20 state ZMM — naturally spill-free even before the mask
// rematerialization (20 + alg temps fit the 32-ZMM file). Probe whether the
// spill-free middle width beats the 1-spill x12.
__forceinline void tm_avx512_r512_map_8::_run_map_entry_x10(
    __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
    __m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
    __m512i& i0, __m512i& i1, __m512i& j0, __m512i& j1,
    int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa=2047,pb=2047,pc=2047,pd=2047,pe=2047,pf=2047,pg=2047,ph=2047,pii=2047,pj=2047;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int it = 0; it < 16; it++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i a0l=_mm512_castsi512_si128(a0),b0l=_mm512_castsi512_si128(b0),c0l=_mm512_castsi512_si128(c0),d0l=_mm512_castsi512_si128(d0);
        const __m128i e0l=_mm512_castsi512_si128(e0),f0l=_mm512_castsi512_si128(f0),g0l=_mm512_castsi512_si128(g0),h0l=_mm512_castsi512_si128(h0);
        const __m128i i0l=_mm512_castsi512_si128(i0),j0l=_mm512_castsi512_si128(j0);
        uint8 ba=0,bb=0,bc=0,bd=0,be=0,bf=0,bg=0,bh=0,bi=0,bj=0;
        TM_MAP_XN_EXTRACT(ba,a0l,it); TM_MAP_XN_EXTRACT(bb,b0l,it); TM_MAP_XN_EXTRACT(bc,c0l,it); TM_MAP_XN_EXTRACT(bd,d0l,it);
        TM_MAP_XN_EXTRACT(be,e0l,it); TM_MAP_XN_EXTRACT(bf,f0l,it); TM_MAP_XN_EXTRACT(bg,g0l,it); TM_MAP_XN_EXTRACT(bh,h0l,it);
        TM_MAP_XN_EXTRACT(bi,i0l,it); TM_MAP_XN_EXTRACT(bj,j0l,it);
        if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4;be>>=4;bf>>=4;bg>>=4;bh>>=4;bi>>=4;bj>>=4; }
        _alg_dispatch(a0,a1,(ba>>1)&0x07,pa,reg_base,alg0_base,alg6_base);
        _alg_dispatch(b0,b1,(bb>>1)&0x07,pb,reg_base,alg0_base,alg6_base);
        _alg_dispatch(c0,c1,(bc>>1)&0x07,pc,reg_base,alg0_base,alg6_base);
        _alg_dispatch(d0,d1,(bd>>1)&0x07,pd,reg_base,alg0_base,alg6_base);
        _alg_dispatch(e0,e1,(be>>1)&0x07,pe,reg_base,alg0_base,alg6_base);
        _alg_dispatch(f0,f1,(bf>>1)&0x07,pf,reg_base,alg0_base,alg6_base);
        _alg_dispatch(g0,g1,(bg>>1)&0x07,pg,reg_base,alg0_base,alg6_base);
        _alg_dispatch(h0,h1,(bh>>1)&0x07,ph,reg_base,alg0_base,alg6_base);
        _alg_dispatch(i0,i1,(bi>>1)&0x07,pii,reg_base,alg0_base,alg6_base);
        _alg_dispatch(j0,j1,(bj>>1)&0x07,pj,reg_base,alg0_base,alg6_base);
    }
}

__forceinline void tm_avx512_r512_map_8::_run_map_entry_x12(
    __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
    __m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
    __m512i& i0, __m512i& i1, __m512i& j0, __m512i& j1, __m512i& k0, __m512i& k1, __m512i& l0, __m512i& l1,
    int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa=2047,pb=2047,pc=2047,pd=2047,pe=2047,pf=2047,pg=2047,ph=2047,pii=2047,pj=2047,pk=2047,pl=2047;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int it = 0; it < 16; it++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i a0l=_mm512_castsi512_si128(a0),b0l=_mm512_castsi512_si128(b0),c0l=_mm512_castsi512_si128(c0),d0l=_mm512_castsi512_si128(d0);
        const __m128i e0l=_mm512_castsi512_si128(e0),f0l=_mm512_castsi512_si128(f0),g0l=_mm512_castsi512_si128(g0),h0l=_mm512_castsi512_si128(h0);
        const __m128i i0l=_mm512_castsi512_si128(i0),j0l=_mm512_castsi512_si128(j0),k0l=_mm512_castsi512_si128(k0),l0l=_mm512_castsi512_si128(l0);
        uint8 ba=0,bb=0,bc=0,bd=0,be=0,bf=0,bg=0,bh=0,bi=0,bj=0,bk=0,bl=0;
        TM_MAP_XN_EXTRACT(ba,a0l,it); TM_MAP_XN_EXTRACT(bb,b0l,it); TM_MAP_XN_EXTRACT(bc,c0l,it); TM_MAP_XN_EXTRACT(bd,d0l,it);
        TM_MAP_XN_EXTRACT(be,e0l,it); TM_MAP_XN_EXTRACT(bf,f0l,it); TM_MAP_XN_EXTRACT(bg,g0l,it); TM_MAP_XN_EXTRACT(bh,h0l,it);
        TM_MAP_XN_EXTRACT(bi,i0l,it); TM_MAP_XN_EXTRACT(bj,j0l,it); TM_MAP_XN_EXTRACT(bk,k0l,it); TM_MAP_XN_EXTRACT(bl,l0l,it);
        if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4;be>>=4;bf>>=4;bg>>=4;bh>>=4;bi>>=4;bj>>=4;bk>>=4;bl>>=4; }
        _alg_dispatch(a0,a1,(ba>>1)&0x07,pa,reg_base,alg0_base,alg6_base);
        _alg_dispatch(b0,b1,(bb>>1)&0x07,pb,reg_base,alg0_base,alg6_base);
        _alg_dispatch(c0,c1,(bc>>1)&0x07,pc,reg_base,alg0_base,alg6_base);
        _alg_dispatch(d0,d1,(bd>>1)&0x07,pd,reg_base,alg0_base,alg6_base);
        _alg_dispatch(e0,e1,(be>>1)&0x07,pe,reg_base,alg0_base,alg6_base);
        _alg_dispatch(f0,f1,(bf>>1)&0x07,pf,reg_base,alg0_base,alg6_base);
        _alg_dispatch(g0,g1,(bg>>1)&0x07,pg,reg_base,alg0_base,alg6_base);
        _alg_dispatch(h0,h1,(bh>>1)&0x07,ph,reg_base,alg0_base,alg6_base);
        _alg_dispatch(i0,i1,(bi>>1)&0x07,pii,reg_base,alg0_base,alg6_base);
        _alg_dispatch(j0,j1,(bj>>1)&0x07,pj,reg_base,alg0_base,alg6_base);
        _alg_dispatch(k0,k1,(bk>>1)&0x07,pk,reg_base,alg0_base,alg6_base);
        _alg_dispatch(l0,l1,(bl>>1)&0x07,pl,reg_base,alg0_base,alg6_base);
    }
}

// x12 alg0-count variant (higher-interleave point for the scoring penalty test:
// 12 counters add GP pressure on top of x12's 24 state ZMM, but the wider ILP may
// hide more of the latency-bound dispatch even as the per-dispatch count cost stays).
__forceinline void tm_avx512_r512_map_8::_run_map_entry_x12_alg0count(
    __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
    __m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
    __m512i& i0, __m512i& i1, __m512i& j0, __m512i& j1, __m512i& k0, __m512i& k1, __m512i& l0, __m512i& l1,
    int map_idx, int counts[12])
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa=2047,pb=2047,pc=2047,pd=2047,pe=2047,pf=2047,pg=2047,ph=2047,pii=2047,pj=2047,pk=2047,pl=2047;
    int ca=0,cb=0,cc=0,cd=0,ce=0,cf=0,cg=0,ch=0,ci=0,cj=0,ck=0,cl=0;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int it = 0; it < 16; it++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i a0l=_mm512_castsi512_si128(a0),b0l=_mm512_castsi512_si128(b0),c0l=_mm512_castsi512_si128(c0),d0l=_mm512_castsi512_si128(d0);
        const __m128i e0l=_mm512_castsi512_si128(e0),f0l=_mm512_castsi512_si128(f0),g0l=_mm512_castsi512_si128(g0),h0l=_mm512_castsi512_si128(h0);
        const __m128i i0l=_mm512_castsi512_si128(i0),j0l=_mm512_castsi512_si128(j0),k0l=_mm512_castsi512_si128(k0),l0l=_mm512_castsi512_si128(l0);
        uint8 ba=0,bb=0,bc=0,bd=0,be=0,bf=0,bg=0,bh=0,bi=0,bj=0,bk=0,bl=0;
        TM_MAP_XN_EXTRACT(ba,a0l,it); TM_MAP_XN_EXTRACT(bb,b0l,it); TM_MAP_XN_EXTRACT(bc,c0l,it); TM_MAP_XN_EXTRACT(bd,d0l,it);
        TM_MAP_XN_EXTRACT(be,e0l,it); TM_MAP_XN_EXTRACT(bf,f0l,it); TM_MAP_XN_EXTRACT(bg,g0l,it); TM_MAP_XN_EXTRACT(bh,h0l,it);
        TM_MAP_XN_EXTRACT(bi,i0l,it); TM_MAP_XN_EXTRACT(bj,j0l,it); TM_MAP_XN_EXTRACT(bk,k0l,it); TM_MAP_XN_EXTRACT(bl,l0l,it);
        if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4;be>>=4;bf>>=4;bg>>=4;bh>>=4;bi>>=4;bj>>=4;bk>>=4;bl>>=4; }
        int oa=(ba>>1)&0x07,ob=(bb>>1)&0x07,oc=(bc>>1)&0x07,od=(bd>>1)&0x07,oe=(be>>1)&0x07,of=(bf>>1)&0x07;
        int og=(bg>>1)&0x07,oh=(bh>>1)&0x07,oi=(bi>>1)&0x07,oj=(bj>>1)&0x07,ok=(bk>>1)&0x07,ol=(bl>>1)&0x07;
        ca+=(oa==0);cb+=(ob==0);cc+=(oc==0);cd+=(od==0);ce+=(oe==0);cf+=(of==0);
        cg+=(og==0);ch+=(oh==0);ci+=(oi==0);cj+=(oj==0);ck+=(ok==0);cl+=(ol==0);
        _alg_dispatch(a0,a1,oa,pa,reg_base,alg0_base,alg6_base);
        _alg_dispatch(b0,b1,ob,pb,reg_base,alg0_base,alg6_base);
        _alg_dispatch(c0,c1,oc,pc,reg_base,alg0_base,alg6_base);
        _alg_dispatch(d0,d1,od,pd,reg_base,alg0_base,alg6_base);
        _alg_dispatch(e0,e1,oe,pe,reg_base,alg0_base,alg6_base);
        _alg_dispatch(f0,f1,of,pf,reg_base,alg0_base,alg6_base);
        _alg_dispatch(g0,g1,og,pg,reg_base,alg0_base,alg6_base);
        _alg_dispatch(h0,h1,oh,ph,reg_base,alg0_base,alg6_base);
        _alg_dispatch(i0,i1,oi,pii,reg_base,alg0_base,alg6_base);
        _alg_dispatch(j0,j1,oj,pj,reg_base,alg0_base,alg6_base);
        _alg_dispatch(k0,k1,ok,pk,reg_base,alg0_base,alg6_base);
        _alg_dispatch(l0,l1,ol,pl,reg_base,alg0_base,alg6_base);
    }
    counts[0]=ca;counts[1]=cb;counts[2]=cc;counts[3]=cd;counts[4]=ce;counts[5]=cf;
    counts[6]=cg;counts[7]=ch;counts[8]=ci;counts[9]=cj;counts[10]=ck;counts[11]=cl;
}

// 14-way: 28 state ZMM. The universal kernel spills here (28 + temps > 32), but
// the natmap's alg_2/5 butterfly rewrite cut the per-alg temp count enough that
// x14 may stay spill-free — confirm via objdump (zmm spill-stores / stack frame).
__forceinline void tm_avx512_r512_map_8::_run_map_entry_x14(
    __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
    __m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
    __m512i& i0, __m512i& i1, __m512i& j0, __m512i& j1, __m512i& k0, __m512i& k1, __m512i& l0, __m512i& l1,
    __m512i& m0, __m512i& m1, __m512i& n0, __m512i& n1,
    int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa=2047,pb=2047,pc=2047,pd=2047,pe=2047,pf=2047,pg=2047,ph=2047,pii=2047,pj=2047,pk=2047,pl=2047,pm=2047,pn=2047;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int it = 0; it < 16; it++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i a0l=_mm512_castsi512_si128(a0),b0l=_mm512_castsi512_si128(b0),c0l=_mm512_castsi512_si128(c0),d0l=_mm512_castsi512_si128(d0);
        const __m128i e0l=_mm512_castsi512_si128(e0),f0l=_mm512_castsi512_si128(f0),g0l=_mm512_castsi512_si128(g0),h0l=_mm512_castsi512_si128(h0);
        const __m128i i0l=_mm512_castsi512_si128(i0),j0l=_mm512_castsi512_si128(j0),k0l=_mm512_castsi512_si128(k0),l0l=_mm512_castsi512_si128(l0);
        const __m128i m0l=_mm512_castsi512_si128(m0),n0l=_mm512_castsi512_si128(n0);
        uint8 ba=0,bb=0,bc=0,bd=0,be=0,bf=0,bg=0,bh=0,bi=0,bj=0,bk=0,bl=0,bm=0,bn=0;
        TM_MAP_XN_EXTRACT(ba,a0l,it); TM_MAP_XN_EXTRACT(bb,b0l,it); TM_MAP_XN_EXTRACT(bc,c0l,it); TM_MAP_XN_EXTRACT(bd,d0l,it);
        TM_MAP_XN_EXTRACT(be,e0l,it); TM_MAP_XN_EXTRACT(bf,f0l,it); TM_MAP_XN_EXTRACT(bg,g0l,it); TM_MAP_XN_EXTRACT(bh,h0l,it);
        TM_MAP_XN_EXTRACT(bi,i0l,it); TM_MAP_XN_EXTRACT(bj,j0l,it); TM_MAP_XN_EXTRACT(bk,k0l,it); TM_MAP_XN_EXTRACT(bl,l0l,it);
        TM_MAP_XN_EXTRACT(bm,m0l,it); TM_MAP_XN_EXTRACT(bn,n0l,it);
        if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4;be>>=4;bf>>=4;bg>>=4;bh>>=4;bi>>=4;bj>>=4;bk>>=4;bl>>=4;bm>>=4;bn>>=4; }
        _alg_dispatch(a0,a1,(ba>>1)&0x07,pa,reg_base,alg0_base,alg6_base);
        _alg_dispatch(b0,b1,(bb>>1)&0x07,pb,reg_base,alg0_base,alg6_base);
        _alg_dispatch(c0,c1,(bc>>1)&0x07,pc,reg_base,alg0_base,alg6_base);
        _alg_dispatch(d0,d1,(bd>>1)&0x07,pd,reg_base,alg0_base,alg6_base);
        _alg_dispatch(e0,e1,(be>>1)&0x07,pe,reg_base,alg0_base,alg6_base);
        _alg_dispatch(f0,f1,(bf>>1)&0x07,pf,reg_base,alg0_base,alg6_base);
        _alg_dispatch(g0,g1,(bg>>1)&0x07,pg,reg_base,alg0_base,alg6_base);
        _alg_dispatch(h0,h1,(bh>>1)&0x07,ph,reg_base,alg0_base,alg6_base);
        _alg_dispatch(i0,i1,(bi>>1)&0x07,pii,reg_base,alg0_base,alg6_base);
        _alg_dispatch(j0,j1,(bj>>1)&0x07,pj,reg_base,alg0_base,alg6_base);
        _alg_dispatch(k0,k1,(bk>>1)&0x07,pk,reg_base,alg0_base,alg6_base);
        _alg_dispatch(l0,l1,(bl>>1)&0x07,pl,reg_base,alg0_base,alg6_base);
        _alg_dispatch(m0,m1,(bm>>1)&0x07,pm,reg_base,alg0_base,alg6_base);
        _alg_dispatch(n0,n1,(bn>>1)&0x07,pn,reg_base,alg0_base,alg6_base);
    }
}
void tm_avx512_r512_map_8::run_maps_range_x8(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
    const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3,
    uint8* out4, uint8* out5, uint8* out6, uint8* out7)
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)in0),a1=_mm512_loadu_si512((const void*)(in0+64));
    __m512i b0=_mm512_loadu_si512((const void*)in1),b1=_mm512_loadu_si512((const void*)(in1+64));
    __m512i c0=_mm512_loadu_si512((const void*)in2),c1=_mm512_loadu_si512((const void*)(in2+64));
    __m512i d0=_mm512_loadu_si512((const void*)in3),d1=_mm512_loadu_si512((const void*)(in3+64));
    __m512i e0=_mm512_loadu_si512((const void*)in4),e1=_mm512_loadu_si512((const void*)(in4+64));
    __m512i f0=_mm512_loadu_si512((const void*)in5),f1=_mm512_loadu_si512((const void*)(in5+64));
    __m512i g0=_mm512_loadu_si512((const void*)in6),g1=_mm512_loadu_si512((const void*)(in6+64));
    __m512i h0=_mm512_loadu_si512((const void*)in7),h1=_mm512_loadu_si512((const void*)(in7+64));
    for (std::size_t map_idx = local_begin; map_idx < local_end; map_idx++)
        _run_map_entry_x8(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1, static_cast<int>(map_idx));
    _mm512_storeu_si512((void*)out0,a0); _mm512_storeu_si512((void*)(out0+64),a1);
    _mm512_storeu_si512((void*)out1,b0); _mm512_storeu_si512((void*)(out1+64),b1);
    _mm512_storeu_si512((void*)out2,c0); _mm512_storeu_si512((void*)(out2+64),c1);
    _mm512_storeu_si512((void*)out3,d0); _mm512_storeu_si512((void*)(out3+64),d1);
    _mm512_storeu_si512((void*)out4,e0); _mm512_storeu_si512((void*)(out4+64),e1);
    _mm512_storeu_si512((void*)out5,f0); _mm512_storeu_si512((void*)(out5+64),f1);
    _mm512_storeu_si512((void*)out6,g0); _mm512_storeu_si512((void*)(out6+64),g1);
    _mm512_storeu_si512((void*)out7,h0); _mm512_storeu_si512((void*)(out7+64),h1);
}

void tm_avx512_r512_map_8::run_map1_range_x8_scores(const key_schedule& schedule_entries,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
    const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3,
    uint8* out4, uint8* out5, uint8* out6, uint8* out7,
    float scores[8])
{
    std::size_t local_map_idx = 0;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() || _owned.entry_count != 1)
            _owned.bind_range(rng, schedule_entries, 0, 1);
        local_map_idx = 0;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
        local_map_idx = 0;
    }

    __m512i a0=_mm512_loadu_si512((const void*)in0),a1=_mm512_loadu_si512((const void*)(in0+64));
    __m512i b0=_mm512_loadu_si512((const void*)in1),b1=_mm512_loadu_si512((const void*)(in1+64));
    __m512i c0=_mm512_loadu_si512((const void*)in2),c1=_mm512_loadu_si512((const void*)(in2+64));
    __m512i d0=_mm512_loadu_si512((const void*)in3),d1=_mm512_loadu_si512((const void*)(in3+64));
    __m512i e0=_mm512_loadu_si512((const void*)in4),e1=_mm512_loadu_si512((const void*)(in4+64));
    __m512i f0=_mm512_loadu_si512((const void*)in5),f1=_mm512_loadu_si512((const void*)(in5+64));
    __m512i g0=_mm512_loadu_si512((const void*)in6),g1=_mm512_loadu_si512((const void*)(in6+64));
    __m512i h0=_mm512_loadu_si512((const void*)in7),h1=_mm512_loadu_si512((const void*)(in7+64));
    _run_map1_entry_x8_scores(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1, static_cast<int>(local_map_idx), scores);
    _mm512_storeu_si512((void*)out0,a0); _mm512_storeu_si512((void*)(out0+64),a1);
    _mm512_storeu_si512((void*)out1,b0); _mm512_storeu_si512((void*)(out1+64),b1);
    _mm512_storeu_si512((void*)out2,c0); _mm512_storeu_si512((void*)(out2+64),c1);
    _mm512_storeu_si512((void*)out3,d0); _mm512_storeu_si512((void*)(out3+64),d1);
    _mm512_storeu_si512((void*)out4,e0); _mm512_storeu_si512((void*)(out4+64),e1);
    _mm512_storeu_si512((void*)out5,f0); _mm512_storeu_si512((void*)(out5+64),f1);
    _mm512_storeu_si512((void*)out6,g0); _mm512_storeu_si512((void*)(out6+64),g1);
    _mm512_storeu_si512((void*)out7,h0); _mm512_storeu_si512((void*)(out7+64),h1);
}

// NOTE: TM_MAP_XN_EXTRACT stays defined through _run_map_entry_x10_blmerge (W6, below); #undef moved there.

// Deep-route variant of run_maps_range_x8: runs the [begin,end) map range exactly
// like run_maps_range_x8, but the FIRST map of the range also emits a per-lane
// alg0 count into scores[8] (the K-boundary deep route score, computed in-kernel
// as a free byproduct instead of the separate scalar raw_alg0_count_for_entry
// pre-pass). For K=1 the range is one map, so this scores every map.
void tm_avx512_r512_map_8::run_maps_range_x8_dscore(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
    const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3,
    uint8* out4, uint8* out5, uint8* out6, uint8* out7,
    float scores[8])
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)in0),a1=_mm512_loadu_si512((const void*)(in0+64));
    __m512i b0=_mm512_loadu_si512((const void*)in1),b1=_mm512_loadu_si512((const void*)(in1+64));
    __m512i c0=_mm512_loadu_si512((const void*)in2),c1=_mm512_loadu_si512((const void*)(in2+64));
    __m512i d0=_mm512_loadu_si512((const void*)in3),d1=_mm512_loadu_si512((const void*)(in3+64));
    __m512i e0=_mm512_loadu_si512((const void*)in4),e1=_mm512_loadu_si512((const void*)(in4+64));
    __m512i f0=_mm512_loadu_si512((const void*)in5),f1=_mm512_loadu_si512((const void*)(in5+64));
    __m512i g0=_mm512_loadu_si512((const void*)in6),g1=_mm512_loadu_si512((const void*)(in6+64));
    __m512i h0=_mm512_loadu_si512((const void*)in7),h1=_mm512_loadu_si512((const void*)(in7+64));
    int counts[8] = {0,0,0,0,0,0,0,0};
    std::size_t map_idx = local_begin;
    if (map_idx < local_end) {
        _run_map_entry_x8_alg0count(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1, static_cast<int>(map_idx), counts);
        ++map_idx;
    }
    for (; map_idx < local_end; ++map_idx)
        _run_map_entry_x8(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1, static_cast<int>(map_idx));
    _mm512_storeu_si512((void*)out0,a0); _mm512_storeu_si512((void*)(out0+64),a1);
    _mm512_storeu_si512((void*)out1,b0); _mm512_storeu_si512((void*)(out1+64),b1);
    _mm512_storeu_si512((void*)out2,c0); _mm512_storeu_si512((void*)(out2+64),c1);
    _mm512_storeu_si512((void*)out3,d0); _mm512_storeu_si512((void*)(out3+64),d1);
    _mm512_storeu_si512((void*)out4,e0); _mm512_storeu_si512((void*)(out4+64),e1);
    _mm512_storeu_si512((void*)out5,f0); _mm512_storeu_si512((void*)(out5+64),f1);
    _mm512_storeu_si512((void*)out6,g0); _mm512_storeu_si512((void*)(out6+64),g1);
    _mm512_storeu_si512((void*)out7,h0); _mm512_storeu_si512((void*)(out7+64),h1);
    for (int k = 0; k < 8; ++k) scores[k] = static_cast<float>(counts[k]);
}

// Op-tail gate variant: runs [begin,end), capturing the FINAL map's last-8-op trajDens
// key per lane into keys[8]. The op-tail key is the routing gate's collision pre-sensor
// (sketch lookup happens at the boundary, outside the kernel loop).
void tm_avx512_r512_map_8::run_maps_range_x8_optail(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
    const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3,
    uint8* out4, uint8* out5, uint8* out6, uint8* out7,
    std::uint32_t keys[8], int counts[8])
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)in0),a1=_mm512_loadu_si512((const void*)(in0+64));
    __m512i b0=_mm512_loadu_si512((const void*)in1),b1=_mm512_loadu_si512((const void*)(in1+64));
    __m512i c0=_mm512_loadu_si512((const void*)in2),c1=_mm512_loadu_si512((const void*)(in2+64));
    __m512i d0=_mm512_loadu_si512((const void*)in3),d1=_mm512_loadu_si512((const void*)(in3+64));
    __m512i e0=_mm512_loadu_si512((const void*)in4),e1=_mm512_loadu_si512((const void*)(in4+64));
    __m512i f0=_mm512_loadu_si512((const void*)in5),f1=_mm512_loadu_si512((const void*)(in5+64));
    __m512i g0=_mm512_loadu_si512((const void*)in6),g1=_mm512_loadu_si512((const void*)(in6+64));
    __m512i h0=_mm512_loadu_si512((const void*)in7),h1=_mm512_loadu_si512((const void*)(in7+64));
    for (int k=0;k<8;++k){ keys[k]=0; counts[k]=0; }
    for (std::size_t map_idx = local_begin; map_idx < local_end; ++map_idx) {
        if (map_idx + 1 == local_end)   // final map of the group: capture op-tail key + alg0 count
            _run_map_entry_x8_optail(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1, static_cast<int>(map_idx), keys, counts);
        else
            _run_map_entry_x8(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1, static_cast<int>(map_idx));
    }
    _mm512_storeu_si512((void*)out0,a0); _mm512_storeu_si512((void*)(out0+64),a1);
    _mm512_storeu_si512((void*)out1,b0); _mm512_storeu_si512((void*)(out1+64),b1);
    _mm512_storeu_si512((void*)out2,c0); _mm512_storeu_si512((void*)(out2+64),c1);
    _mm512_storeu_si512((void*)out3,d0); _mm512_storeu_si512((void*)(out3+64),d1);
    _mm512_storeu_si512((void*)out4,e0); _mm512_storeu_si512((void*)(out4+64),e1);
    _mm512_storeu_si512((void*)out5,f0); _mm512_storeu_si512((void*)(out5+64),f1);
    _mm512_storeu_si512((void*)out6,g0); _mm512_storeu_si512((void*)(out6+64),g1);
    _mm512_storeu_si512((void*)out7,h0); _mm512_storeu_si512((void*)(out7+64),h1);
}

// Combined route signal: per-lane STICKY alg0 flag (OR over the span of "this map's alg0 count >=
// alg0_tau") + the FINAL map's op-tail key. Per-map alg0 count rides the existing _alg0count entry
// (the live alg_id is already materialized); the final map uses _optail (op-tail key + alg0 count).
void tm_avx512_r512_map_8::run_maps_range_x8_route(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
    const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3,
    uint8* out4, uint8* out5, uint8* out6, uint8* out7,
    std::uint32_t keys[8], int sticky[8], int alg0_tau)
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)in0),a1=_mm512_loadu_si512((const void*)(in0+64));
    __m512i b0=_mm512_loadu_si512((const void*)in1),b1=_mm512_loadu_si512((const void*)(in1+64));
    __m512i c0=_mm512_loadu_si512((const void*)in2),c1=_mm512_loadu_si512((const void*)(in2+64));
    __m512i d0=_mm512_loadu_si512((const void*)in3),d1=_mm512_loadu_si512((const void*)(in3+64));
    __m512i e0=_mm512_loadu_si512((const void*)in4),e1=_mm512_loadu_si512((const void*)(in4+64));
    __m512i f0=_mm512_loadu_si512((const void*)in5),f1=_mm512_loadu_si512((const void*)(in5+64));
    __m512i g0=_mm512_loadu_si512((const void*)in6),g1=_mm512_loadu_si512((const void*)(in6+64));
    __m512i h0=_mm512_loadu_si512((const void*)in7),h1=_mm512_loadu_si512((const void*)(in7+64));
    for (int k=0;k<8;++k){ keys[k]=0; sticky[k]=0; }
    int cc[8];
    for (std::size_t map_idx = local_begin; map_idx < local_end; ++map_idx) {
        if (map_idx + 1 == local_end)
            _run_map_entry_x8_optail(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1, static_cast<int>(map_idx), keys, cc);
        else
            _run_map_entry_x8_alg0count(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1, static_cast<int>(map_idx), cc);
        for (int k=0;k<8;++k) if (cc[k] >= alg0_tau) sticky[k] = 1;   // sticky OR over the span
    }
    _mm512_storeu_si512((void*)out0,a0); _mm512_storeu_si512((void*)(out0+64),a1);
    _mm512_storeu_si512((void*)out1,b0); _mm512_storeu_si512((void*)(out1+64),b1);
    _mm512_storeu_si512((void*)out2,c0); _mm512_storeu_si512((void*)(out2+64),c1);
    _mm512_storeu_si512((void*)out3,d0); _mm512_storeu_si512((void*)(out3+64),d1);
    _mm512_storeu_si512((void*)out4,e0); _mm512_storeu_si512((void*)(out4+64),e1);
    _mm512_storeu_si512((void*)out5,f0); _mm512_storeu_si512((void*)(out5+64),f1);
    _mm512_storeu_si512((void*)out6,g0); _mm512_storeu_si512((void*)(out6+64),g1);
    _mm512_storeu_si512((void*)out7,h0); _mm512_storeu_si512((void*)(out7+64),h1);
}


// EXPERIMENT wrapper: x8 with branchless-arith dispatch (no scoring). Mirrors
// run_maps_range_x8 but uses _run_map_entry_x8_blarith.
void tm_avx512_r512_map_8::run_maps_range_x8_blarith(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
    const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3,
    uint8* out4, uint8* out5, uint8* out6, uint8* out7)
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)in0),a1=_mm512_loadu_si512((const void*)(in0+64));
    __m512i b0=_mm512_loadu_si512((const void*)in1),b1=_mm512_loadu_si512((const void*)(in1+64));
    __m512i c0=_mm512_loadu_si512((const void*)in2),c1=_mm512_loadu_si512((const void*)(in2+64));
    __m512i d0=_mm512_loadu_si512((const void*)in3),d1=_mm512_loadu_si512((const void*)(in3+64));
    __m512i e0=_mm512_loadu_si512((const void*)in4),e1=_mm512_loadu_si512((const void*)(in4+64));
    __m512i f0=_mm512_loadu_si512((const void*)in5),f1=_mm512_loadu_si512((const void*)(in5+64));
    __m512i g0=_mm512_loadu_si512((const void*)in6),g1=_mm512_loadu_si512((const void*)(in6+64));
    __m512i h0=_mm512_loadu_si512((const void*)in7),h1=_mm512_loadu_si512((const void*)(in7+64));
    for (std::size_t map_idx = local_begin; map_idx < local_end; ++map_idx)
        _run_map_entry_x8_blarith(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1, static_cast<int>(map_idx));
    _mm512_storeu_si512((void*)out0,a0); _mm512_storeu_si512((void*)(out0+64),a1);
    _mm512_storeu_si512((void*)out1,b0); _mm512_storeu_si512((void*)(out1+64),b1);
    _mm512_storeu_si512((void*)out2,c0); _mm512_storeu_si512((void*)(out2+64),c1);
    _mm512_storeu_si512((void*)out3,d0); _mm512_storeu_si512((void*)(out3+64),d1);
    _mm512_storeu_si512((void*)out4,e0); _mm512_storeu_si512((void*)(out4+64),e1);
    _mm512_storeu_si512((void*)out5,f0); _mm512_storeu_si512((void*)(out5+64),f1);
    _mm512_storeu_si512((void*)out6,g0); _mm512_storeu_si512((void*)(out6+64),g1);
    _mm512_storeu_si512((void*)out7,h0); _mm512_storeu_si512((void*)(out7+64),h1);
}

// EXPERIMENT v2 wrapper: x8 with merged branchless dispatch (3 candidates + blend-tree).
void tm_avx512_r512_map_8::run_maps_range_x8_blmerge(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
    const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3,
    uint8* out4, uint8* out5, uint8* out6, uint8* out7)
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)in0),a1=_mm512_loadu_si512((const void*)(in0+64));
    __m512i b0=_mm512_loadu_si512((const void*)in1),b1=_mm512_loadu_si512((const void*)(in1+64));
    __m512i c0=_mm512_loadu_si512((const void*)in2),c1=_mm512_loadu_si512((const void*)(in2+64));
    __m512i d0=_mm512_loadu_si512((const void*)in3),d1=_mm512_loadu_si512((const void*)(in3+64));
    __m512i e0=_mm512_loadu_si512((const void*)in4),e1=_mm512_loadu_si512((const void*)(in4+64));
    __m512i f0=_mm512_loadu_si512((const void*)in5),f1=_mm512_loadu_si512((const void*)(in5+64));
    __m512i g0=_mm512_loadu_si512((const void*)in6),g1=_mm512_loadu_si512((const void*)(in6+64));
    __m512i h0=_mm512_loadu_si512((const void*)in7),h1=_mm512_loadu_si512((const void*)(in7+64));
    for (std::size_t map_idx = local_begin; map_idx < local_end; ++map_idx)
        _run_map_entry_x8_blmerge(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1, static_cast<int>(map_idx));
    _mm512_storeu_si512((void*)out0,a0); _mm512_storeu_si512((void*)(out0+64),a1);
    _mm512_storeu_si512((void*)out1,b0); _mm512_storeu_si512((void*)(out1+64),b1);
    _mm512_storeu_si512((void*)out2,c0); _mm512_storeu_si512((void*)(out2+64),c1);
    _mm512_storeu_si512((void*)out3,d0); _mm512_storeu_si512((void*)(out3+64),d1);
    _mm512_storeu_si512((void*)out4,e0); _mm512_storeu_si512((void*)(out4+64),e1);
    _mm512_storeu_si512((void*)out5,f0); _mm512_storeu_si512((void*)(out5+64),f1);
    _mm512_storeu_si512((void*)out6,g0); _mm512_storeu_si512((void*)(out6+64),g1);
    _mm512_storeu_si512((void*)out7,h0); _mm512_storeu_si512((void*)(out7+64),h1);
}

// EXPERIMENT v3 wrapper: x8 with full all-8 branchless dispatch.
void tm_avx512_r512_map_8::run_maps_range_x8_blall(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
    const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3,
    uint8* out4, uint8* out5, uint8* out6, uint8* out7)
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)in0),a1=_mm512_loadu_si512((const void*)(in0+64));
    __m512i b0=_mm512_loadu_si512((const void*)in1),b1=_mm512_loadu_si512((const void*)(in1+64));
    __m512i c0=_mm512_loadu_si512((const void*)in2),c1=_mm512_loadu_si512((const void*)(in2+64));
    __m512i d0=_mm512_loadu_si512((const void*)in3),d1=_mm512_loadu_si512((const void*)(in3+64));
    __m512i e0=_mm512_loadu_si512((const void*)in4),e1=_mm512_loadu_si512((const void*)(in4+64));
    __m512i f0=_mm512_loadu_si512((const void*)in5),f1=_mm512_loadu_si512((const void*)(in5+64));
    __m512i g0=_mm512_loadu_si512((const void*)in6),g1=_mm512_loadu_si512((const void*)(in6+64));
    __m512i h0=_mm512_loadu_si512((const void*)in7),h1=_mm512_loadu_si512((const void*)(in7+64));
    for (std::size_t map_idx = local_begin; map_idx < local_end; ++map_idx)
        _run_map_entry_x8_blall(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1, static_cast<int>(map_idx));
    _mm512_storeu_si512((void*)out0,a0); _mm512_storeu_si512((void*)(out0+64),a1);
    _mm512_storeu_si512((void*)out1,b0); _mm512_storeu_si512((void*)(out1+64),b1);
    _mm512_storeu_si512((void*)out2,c0); _mm512_storeu_si512((void*)(out2+64),c1);
    _mm512_storeu_si512((void*)out3,d0); _mm512_storeu_si512((void*)(out3+64),d1);
    _mm512_storeu_si512((void*)out4,e0); _mm512_storeu_si512((void*)(out4+64),e1);
    _mm512_storeu_si512((void*)out5,f0); _mm512_storeu_si512((void*)(out5+64),f1);
    _mm512_storeu_si512((void*)out6,g0); _mm512_storeu_si512((void*)(out6+64),g1);
    _mm512_storeu_si512((void*)out7,h0); _mm512_storeu_si512((void*)(out7+64),h1);
}

// Single-lane (x1) public entry: mode 0=branched,1=blarith,2=blmerge,3=blall.
void tm_avx512_r512_map_8::run_maps_range_x1(const key_schedule& s, std::size_t begin, std::size_t end,
    const uint8* in, uint8* out, int mode)
{
    switch (mode) {
        case 1:  _run_maps_range_x1_tpl<1>(s,begin,end,in,out); break;
        case 2:  _run_maps_range_x1_tpl<2>(s,begin,end,in,out); break;
        case 3:  _run_maps_range_x1_tpl<3>(s,begin,end,in,out); break;
        default: _run_maps_range_x1_tpl<0>(s,begin,end,in,out); break;
    }
}

// x4 branchless-arith wrapper (lower-interleave point for the blarith pressure test).
void tm_avx512_r512_map_8::run_maps_range_x4_blarith(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3)
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)in0),a1=_mm512_loadu_si512((const void*)(in0+64));
    __m512i b0=_mm512_loadu_si512((const void*)in1),b1=_mm512_loadu_si512((const void*)(in1+64));
    __m512i c0=_mm512_loadu_si512((const void*)in2),c1=_mm512_loadu_si512((const void*)(in2+64));
    __m512i d0=_mm512_loadu_si512((const void*)in3),d1=_mm512_loadu_si512((const void*)(in3+64));
    for (std::size_t map_idx = local_begin; map_idx < local_end; ++map_idx)
        _run_map_entry_x4_blarith(a0,a1,b0,b1,c0,c1,d0,d1, static_cast<int>(map_idx));
    _mm512_storeu_si512((void*)out0,a0); _mm512_storeu_si512((void*)(out0+64),a1);
    _mm512_storeu_si512((void*)out1,b0); _mm512_storeu_si512((void*)(out1+64),b1);
    _mm512_storeu_si512((void*)out2,c0); _mm512_storeu_si512((void*)(out2+64),c1);
    _mm512_storeu_si512((void*)out3,d0); _mm512_storeu_si512((void*)(out3+64),d1);
}

// x4 deep-route variant (lower-interleave control), mirrors run_maps_range_x8_dscore.
void tm_avx512_r512_map_8::run_maps_range_x4_dscore(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3,
    float scores[4])
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)in0),a1=_mm512_loadu_si512((const void*)(in0+64));
    __m512i b0=_mm512_loadu_si512((const void*)in1),b1=_mm512_loadu_si512((const void*)(in1+64));
    __m512i c0=_mm512_loadu_si512((const void*)in2),c1=_mm512_loadu_si512((const void*)(in2+64));
    __m512i d0=_mm512_loadu_si512((const void*)in3),d1=_mm512_loadu_si512((const void*)(in3+64));
    int counts[4] = {0,0,0,0};
    std::size_t map_idx = local_begin;
    if (map_idx < local_end) {
        _run_map_entry_x4_alg0count(a0,a1,b0,b1,c0,c1,d0,d1, static_cast<int>(map_idx), counts);
        ++map_idx;
    }
    for (; map_idx < local_end; ++map_idx)
        _run_map_entry_x4(a0,a1,b0,b1,c0,c1,d0,d1, static_cast<int>(map_idx));
    _mm512_storeu_si512((void*)out0,a0); _mm512_storeu_si512((void*)(out0+64),a1);
    _mm512_storeu_si512((void*)out1,b0); _mm512_storeu_si512((void*)(out1+64),b1);
    _mm512_storeu_si512((void*)out2,c0); _mm512_storeu_si512((void*)(out2+64),c1);
    _mm512_storeu_si512((void*)out3,d0); _mm512_storeu_si512((void*)(out3+64),d1);
    for (int k = 0; k < 4; ++k) scores[k] = static_cast<float>(counts[k]);
}

void tm_avx512_r512_map_8::run_maps_range_x10(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3, const uint8* in4,
    const uint8* in5, const uint8* in6, const uint8* in7, const uint8* in8, const uint8* in9,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3, uint8* out4,
    uint8* out5, uint8* out6, uint8* out7, uint8* out8, uint8* out9)
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)in0),a1=_mm512_loadu_si512((const void*)(in0+64));
    __m512i b0=_mm512_loadu_si512((const void*)in1),b1=_mm512_loadu_si512((const void*)(in1+64));
    __m512i c0=_mm512_loadu_si512((const void*)in2),c1=_mm512_loadu_si512((const void*)(in2+64));
    __m512i d0=_mm512_loadu_si512((const void*)in3),d1=_mm512_loadu_si512((const void*)(in3+64));
    __m512i e0=_mm512_loadu_si512((const void*)in4),e1=_mm512_loadu_si512((const void*)(in4+64));
    __m512i f0=_mm512_loadu_si512((const void*)in5),f1=_mm512_loadu_si512((const void*)(in5+64));
    __m512i g0=_mm512_loadu_si512((const void*)in6),g1=_mm512_loadu_si512((const void*)(in6+64));
    __m512i h0=_mm512_loadu_si512((const void*)in7),h1=_mm512_loadu_si512((const void*)(in7+64));
    __m512i i0=_mm512_loadu_si512((const void*)in8),i1=_mm512_loadu_si512((const void*)(in8+64));
    __m512i j0=_mm512_loadu_si512((const void*)in9),j1=_mm512_loadu_si512((const void*)(in9+64));
    for (std::size_t map_idx = local_begin; map_idx < local_end; map_idx++)
        _run_map_entry_x10(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1,i0,i1,j0,j1, static_cast<int>(map_idx));
    _mm512_storeu_si512((void*)out0,a0); _mm512_storeu_si512((void*)(out0+64),a1);
    _mm512_storeu_si512((void*)out1,b0); _mm512_storeu_si512((void*)(out1+64),b1);
    _mm512_storeu_si512((void*)out2,c0); _mm512_storeu_si512((void*)(out2+64),c1);
    _mm512_storeu_si512((void*)out3,d0); _mm512_storeu_si512((void*)(out3+64),d1);
    _mm512_storeu_si512((void*)out4,e0); _mm512_storeu_si512((void*)(out4+64),e1);
    _mm512_storeu_si512((void*)out5,f0); _mm512_storeu_si512((void*)(out5+64),f1);
    _mm512_storeu_si512((void*)out6,g0); _mm512_storeu_si512((void*)(out6+64),g1);
    _mm512_storeu_si512((void*)out7,h0); _mm512_storeu_si512((void*)(out7+64),h1);
    _mm512_storeu_si512((void*)out8,i0); _mm512_storeu_si512((void*)(out8+64),i1);
    _mm512_storeu_si512((void*)out9,j0); _mm512_storeu_si512((void*)(out9+64),j1);
}

void tm_avx512_r512_map_8::run_maps_range_x12(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3, const uint8* in4, const uint8* in5,
    const uint8* in6, const uint8* in7, const uint8* in8, const uint8* in9, const uint8* in10, const uint8* in11,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3, uint8* out4, uint8* out5,
    uint8* out6, uint8* out7, uint8* out8, uint8* out9, uint8* out10, uint8* out11)
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)in0),a1=_mm512_loadu_si512((const void*)(in0+64));
    __m512i b0=_mm512_loadu_si512((const void*)in1),b1=_mm512_loadu_si512((const void*)(in1+64));
    __m512i c0=_mm512_loadu_si512((const void*)in2),c1=_mm512_loadu_si512((const void*)(in2+64));
    __m512i d0=_mm512_loadu_si512((const void*)in3),d1=_mm512_loadu_si512((const void*)(in3+64));
    __m512i e0=_mm512_loadu_si512((const void*)in4),e1=_mm512_loadu_si512((const void*)(in4+64));
    __m512i f0=_mm512_loadu_si512((const void*)in5),f1=_mm512_loadu_si512((const void*)(in5+64));
    __m512i g0=_mm512_loadu_si512((const void*)in6),g1=_mm512_loadu_si512((const void*)(in6+64));
    __m512i h0=_mm512_loadu_si512((const void*)in7),h1=_mm512_loadu_si512((const void*)(in7+64));
    __m512i i0=_mm512_loadu_si512((const void*)in8),i1=_mm512_loadu_si512((const void*)(in8+64));
    __m512i j0=_mm512_loadu_si512((const void*)in9),j1=_mm512_loadu_si512((const void*)(in9+64));
    __m512i k0=_mm512_loadu_si512((const void*)in10),k1=_mm512_loadu_si512((const void*)(in10+64));
    __m512i l0=_mm512_loadu_si512((const void*)in11),l1=_mm512_loadu_si512((const void*)(in11+64));
    for (std::size_t map_idx = local_begin; map_idx < local_end; map_idx++)
        _run_map_entry_x12(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1,i0,i1,j0,j1,k0,k1,l0,l1, static_cast<int>(map_idx));
    _mm512_storeu_si512((void*)out0,a0); _mm512_storeu_si512((void*)(out0+64),a1);
    _mm512_storeu_si512((void*)out1,b0); _mm512_storeu_si512((void*)(out1+64),b1);
    _mm512_storeu_si512((void*)out2,c0); _mm512_storeu_si512((void*)(out2+64),c1);
    _mm512_storeu_si512((void*)out3,d0); _mm512_storeu_si512((void*)(out3+64),d1);
    _mm512_storeu_si512((void*)out4,e0); _mm512_storeu_si512((void*)(out4+64),e1);
    _mm512_storeu_si512((void*)out5,f0); _mm512_storeu_si512((void*)(out5+64),f1);
    _mm512_storeu_si512((void*)out6,g0); _mm512_storeu_si512((void*)(out6+64),g1);
    _mm512_storeu_si512((void*)out7,h0); _mm512_storeu_si512((void*)(out7+64),h1);
    _mm512_storeu_si512((void*)out8,i0); _mm512_storeu_si512((void*)(out8+64),i1);
    _mm512_storeu_si512((void*)out9,j0); _mm512_storeu_si512((void*)(out9+64),j1);
    _mm512_storeu_si512((void*)out10,k0); _mm512_storeu_si512((void*)(out10+64),k1);
    _mm512_storeu_si512((void*)out11,l0); _mm512_storeu_si512((void*)(out11+64),l1);
}

// x12 deep-route variant (higher-interleave point), mirrors run_maps_range_x8_dscore.
void tm_avx512_r512_map_8::run_maps_range_x12_dscore(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3, const uint8* in4, const uint8* in5,
    const uint8* in6, const uint8* in7, const uint8* in8, const uint8* in9, const uint8* in10, const uint8* in11,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3, uint8* out4, uint8* out5,
    uint8* out6, uint8* out7, uint8* out8, uint8* out9, uint8* out10, uint8* out11,
    float scores[12])
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)in0),a1=_mm512_loadu_si512((const void*)(in0+64));
    __m512i b0=_mm512_loadu_si512((const void*)in1),b1=_mm512_loadu_si512((const void*)(in1+64));
    __m512i c0=_mm512_loadu_si512((const void*)in2),c1=_mm512_loadu_si512((const void*)(in2+64));
    __m512i d0=_mm512_loadu_si512((const void*)in3),d1=_mm512_loadu_si512((const void*)(in3+64));
    __m512i e0=_mm512_loadu_si512((const void*)in4),e1=_mm512_loadu_si512((const void*)(in4+64));
    __m512i f0=_mm512_loadu_si512((const void*)in5),f1=_mm512_loadu_si512((const void*)(in5+64));
    __m512i g0=_mm512_loadu_si512((const void*)in6),g1=_mm512_loadu_si512((const void*)(in6+64));
    __m512i h0=_mm512_loadu_si512((const void*)in7),h1=_mm512_loadu_si512((const void*)(in7+64));
    __m512i i0=_mm512_loadu_si512((const void*)in8),i1=_mm512_loadu_si512((const void*)(in8+64));
    __m512i j0=_mm512_loadu_si512((const void*)in9),j1=_mm512_loadu_si512((const void*)(in9+64));
    __m512i k0=_mm512_loadu_si512((const void*)in10),k1=_mm512_loadu_si512((const void*)(in10+64));
    __m512i l0=_mm512_loadu_si512((const void*)in11),l1=_mm512_loadu_si512((const void*)(in11+64));
    int counts[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
    std::size_t map_idx = local_begin;
    if (map_idx < local_end) {
        _run_map_entry_x12_alg0count(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1,i0,i1,j0,j1,k0,k1,l0,l1, static_cast<int>(map_idx), counts);
        ++map_idx;
    }
    for (; map_idx < local_end; ++map_idx)
        _run_map_entry_x12(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1,i0,i1,j0,j1,k0,k1,l0,l1, static_cast<int>(map_idx));
    _mm512_storeu_si512((void*)out0,a0); _mm512_storeu_si512((void*)(out0+64),a1);
    _mm512_storeu_si512((void*)out1,b0); _mm512_storeu_si512((void*)(out1+64),b1);
    _mm512_storeu_si512((void*)out2,c0); _mm512_storeu_si512((void*)(out2+64),c1);
    _mm512_storeu_si512((void*)out3,d0); _mm512_storeu_si512((void*)(out3+64),d1);
    _mm512_storeu_si512((void*)out4,e0); _mm512_storeu_si512((void*)(out4+64),e1);
    _mm512_storeu_si512((void*)out5,f0); _mm512_storeu_si512((void*)(out5+64),f1);
    _mm512_storeu_si512((void*)out6,g0); _mm512_storeu_si512((void*)(out6+64),g1);
    _mm512_storeu_si512((void*)out7,h0); _mm512_storeu_si512((void*)(out7+64),h1);
    _mm512_storeu_si512((void*)out8,i0); _mm512_storeu_si512((void*)(out8+64),i1);
    _mm512_storeu_si512((void*)out9,j0); _mm512_storeu_si512((void*)(out9+64),j1);
    _mm512_storeu_si512((void*)out10,k0); _mm512_storeu_si512((void*)(out10+64),k1);
    _mm512_storeu_si512((void*)out11,l0); _mm512_storeu_si512((void*)(out11+64),l1);
    for (int kk = 0; kk < 12; ++kk) scores[kk] = static_cast<float>(counts[kk]);
}

void tm_avx512_r512_map_8::run_maps_range_x14(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3, const uint8* in4, const uint8* in5, const uint8* in6,
    const uint8* in7, const uint8* in8, const uint8* in9, const uint8* in10, const uint8* in11, const uint8* in12, const uint8* in13,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3, uint8* out4, uint8* out5, uint8* out6,
    uint8* out7, uint8* out8, uint8* out9, uint8* out10, uint8* out11, uint8* out12, uint8* out13)
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)in0),a1=_mm512_loadu_si512((const void*)(in0+64));
    __m512i b0=_mm512_loadu_si512((const void*)in1),b1=_mm512_loadu_si512((const void*)(in1+64));
    __m512i c0=_mm512_loadu_si512((const void*)in2),c1=_mm512_loadu_si512((const void*)(in2+64));
    __m512i d0=_mm512_loadu_si512((const void*)in3),d1=_mm512_loadu_si512((const void*)(in3+64));
    __m512i e0=_mm512_loadu_si512((const void*)in4),e1=_mm512_loadu_si512((const void*)(in4+64));
    __m512i f0=_mm512_loadu_si512((const void*)in5),f1=_mm512_loadu_si512((const void*)(in5+64));
    __m512i g0=_mm512_loadu_si512((const void*)in6),g1=_mm512_loadu_si512((const void*)(in6+64));
    __m512i h0=_mm512_loadu_si512((const void*)in7),h1=_mm512_loadu_si512((const void*)(in7+64));
    __m512i i0=_mm512_loadu_si512((const void*)in8),i1=_mm512_loadu_si512((const void*)(in8+64));
    __m512i j0=_mm512_loadu_si512((const void*)in9),j1=_mm512_loadu_si512((const void*)(in9+64));
    __m512i k0=_mm512_loadu_si512((const void*)in10),k1=_mm512_loadu_si512((const void*)(in10+64));
    __m512i l0=_mm512_loadu_si512((const void*)in11),l1=_mm512_loadu_si512((const void*)(in11+64));
    __m512i m0=_mm512_loadu_si512((const void*)in12),m1=_mm512_loadu_si512((const void*)(in12+64));
    __m512i n0=_mm512_loadu_si512((const void*)in13),n1=_mm512_loadu_si512((const void*)(in13+64));
    for (std::size_t map_idx = local_begin; map_idx < local_end; map_idx++)
        _run_map_entry_x14(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1,i0,i1,j0,j1,k0,k1,l0,l1,m0,m1,n0,n1, static_cast<int>(map_idx));
    _mm512_storeu_si512((void*)out0,a0); _mm512_storeu_si512((void*)(out0+64),a1);
    _mm512_storeu_si512((void*)out1,b0); _mm512_storeu_si512((void*)(out1+64),b1);
    _mm512_storeu_si512((void*)out2,c0); _mm512_storeu_si512((void*)(out2+64),c1);
    _mm512_storeu_si512((void*)out3,d0); _mm512_storeu_si512((void*)(out3+64),d1);
    _mm512_storeu_si512((void*)out4,e0); _mm512_storeu_si512((void*)(out4+64),e1);
    _mm512_storeu_si512((void*)out5,f0); _mm512_storeu_si512((void*)(out5+64),f1);
    _mm512_storeu_si512((void*)out6,g0); _mm512_storeu_si512((void*)(out6+64),g1);
    _mm512_storeu_si512((void*)out7,h0); _mm512_storeu_si512((void*)(out7+64),h1);
    _mm512_storeu_si512((void*)out8,i0); _mm512_storeu_si512((void*)(out8+64),i1);
    _mm512_storeu_si512((void*)out9,j0); _mm512_storeu_si512((void*)(out9+64),j1);
    _mm512_storeu_si512((void*)out10,k0); _mm512_storeu_si512((void*)(out10+64),k1);
    _mm512_storeu_si512((void*)out11,l0); _mm512_storeu_si512((void*)(out11+64),l1);
    _mm512_storeu_si512((void*)out12,m0); _mm512_storeu_si512((void*)(out12+64),m1);
    _mm512_storeu_si512((void*)out13,n0); _mm512_storeu_si512((void*)(out13+64),n1);
}

// ---------------------------------------------------------------------------
// W6: base-pointer interleave variants. Contiguous in[N*128]/out[N*128] bases; per-lane pointers computed
// internally (in + k*128) so they never spill as call args. Frame-safe vs the 24/28-arg forms. Bit-identical
// math (same _run_map_entry_xN). See header + docs/cpu_raceway_o2_o3_fragility (the -O2-too "frame trigger").
// ---------------------------------------------------------------------------
void tm_avx512_r512_map_8::run_maps_range_x10_arr(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in, uint8* out)
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)(in+0*128)),a1=_mm512_loadu_si512((const void*)(in+0*128+64));
    __m512i b0=_mm512_loadu_si512((const void*)(in+1*128)),b1=_mm512_loadu_si512((const void*)(in+1*128+64));
    __m512i c0=_mm512_loadu_si512((const void*)(in+2*128)),c1=_mm512_loadu_si512((const void*)(in+2*128+64));
    __m512i d0=_mm512_loadu_si512((const void*)(in+3*128)),d1=_mm512_loadu_si512((const void*)(in+3*128+64));
    __m512i e0=_mm512_loadu_si512((const void*)(in+4*128)),e1=_mm512_loadu_si512((const void*)(in+4*128+64));
    __m512i f0=_mm512_loadu_si512((const void*)(in+5*128)),f1=_mm512_loadu_si512((const void*)(in+5*128+64));
    __m512i g0=_mm512_loadu_si512((const void*)(in+6*128)),g1=_mm512_loadu_si512((const void*)(in+6*128+64));
    __m512i h0=_mm512_loadu_si512((const void*)(in+7*128)),h1=_mm512_loadu_si512((const void*)(in+7*128+64));
    __m512i i0=_mm512_loadu_si512((const void*)(in+8*128)),i1=_mm512_loadu_si512((const void*)(in+8*128+64));
    __m512i j0=_mm512_loadu_si512((const void*)(in+9*128)),j1=_mm512_loadu_si512((const void*)(in+9*128+64));
    for (std::size_t map_idx = local_begin; map_idx < local_end; map_idx++)
        _run_map_entry_x10(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1,i0,i1,j0,j1, static_cast<int>(map_idx));
    _mm512_storeu_si512((void*)(out+0*128),a0); _mm512_storeu_si512((void*)(out+0*128+64),a1);
    _mm512_storeu_si512((void*)(out+1*128),b0); _mm512_storeu_si512((void*)(out+1*128+64),b1);
    _mm512_storeu_si512((void*)(out+2*128),c0); _mm512_storeu_si512((void*)(out+2*128+64),c1);
    _mm512_storeu_si512((void*)(out+3*128),d0); _mm512_storeu_si512((void*)(out+3*128+64),d1);
    _mm512_storeu_si512((void*)(out+4*128),e0); _mm512_storeu_si512((void*)(out+4*128+64),e1);
    _mm512_storeu_si512((void*)(out+5*128),f0); _mm512_storeu_si512((void*)(out+5*128+64),f1);
    _mm512_storeu_si512((void*)(out+6*128),g0); _mm512_storeu_si512((void*)(out+6*128+64),g1);
    _mm512_storeu_si512((void*)(out+7*128),h0); _mm512_storeu_si512((void*)(out+7*128+64),h1);
    _mm512_storeu_si512((void*)(out+8*128),i0); _mm512_storeu_si512((void*)(out+8*128+64),i1);
    _mm512_storeu_si512((void*)(out+9*128),j0); _mm512_storeu_si512((void*)(out+9*128+64),j1);
}

void tm_avx512_r512_map_8::run_maps_range_x12_arr(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in, uint8* out)
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)(in+0*128)),a1=_mm512_loadu_si512((const void*)(in+0*128+64));
    __m512i b0=_mm512_loadu_si512((const void*)(in+1*128)),b1=_mm512_loadu_si512((const void*)(in+1*128+64));
    __m512i c0=_mm512_loadu_si512((const void*)(in+2*128)),c1=_mm512_loadu_si512((const void*)(in+2*128+64));
    __m512i d0=_mm512_loadu_si512((const void*)(in+3*128)),d1=_mm512_loadu_si512((const void*)(in+3*128+64));
    __m512i e0=_mm512_loadu_si512((const void*)(in+4*128)),e1=_mm512_loadu_si512((const void*)(in+4*128+64));
    __m512i f0=_mm512_loadu_si512((const void*)(in+5*128)),f1=_mm512_loadu_si512((const void*)(in+5*128+64));
    __m512i g0=_mm512_loadu_si512((const void*)(in+6*128)),g1=_mm512_loadu_si512((const void*)(in+6*128+64));
    __m512i h0=_mm512_loadu_si512((const void*)(in+7*128)),h1=_mm512_loadu_si512((const void*)(in+7*128+64));
    __m512i i0=_mm512_loadu_si512((const void*)(in+8*128)),i1=_mm512_loadu_si512((const void*)(in+8*128+64));
    __m512i j0=_mm512_loadu_si512((const void*)(in+9*128)),j1=_mm512_loadu_si512((const void*)(in+9*128+64));
    __m512i k0=_mm512_loadu_si512((const void*)(in+10*128)),k1=_mm512_loadu_si512((const void*)(in+10*128+64));
    __m512i l0=_mm512_loadu_si512((const void*)(in+11*128)),l1=_mm512_loadu_si512((const void*)(in+11*128+64));
    for (std::size_t map_idx = local_begin; map_idx < local_end; map_idx++)
        _run_map_entry_x12(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1,i0,i1,j0,j1,k0,k1,l0,l1, static_cast<int>(map_idx));
    _mm512_storeu_si512((void*)(out+0*128),a0); _mm512_storeu_si512((void*)(out+0*128+64),a1);
    _mm512_storeu_si512((void*)(out+1*128),b0); _mm512_storeu_si512((void*)(out+1*128+64),b1);
    _mm512_storeu_si512((void*)(out+2*128),c0); _mm512_storeu_si512((void*)(out+2*128+64),c1);
    _mm512_storeu_si512((void*)(out+3*128),d0); _mm512_storeu_si512((void*)(out+3*128+64),d1);
    _mm512_storeu_si512((void*)(out+4*128),e0); _mm512_storeu_si512((void*)(out+4*128+64),e1);
    _mm512_storeu_si512((void*)(out+5*128),f0); _mm512_storeu_si512((void*)(out+5*128+64),f1);
    _mm512_storeu_si512((void*)(out+6*128),g0); _mm512_storeu_si512((void*)(out+6*128+64),g1);
    _mm512_storeu_si512((void*)(out+7*128),h0); _mm512_storeu_si512((void*)(out+7*128+64),h1);
    _mm512_storeu_si512((void*)(out+8*128),i0); _mm512_storeu_si512((void*)(out+8*128+64),i1);
    _mm512_storeu_si512((void*)(out+9*128),j0); _mm512_storeu_si512((void*)(out+9*128+64),j1);
    _mm512_storeu_si512((void*)(out+10*128),k0); _mm512_storeu_si512((void*)(out+10*128+64),k1);
    _mm512_storeu_si512((void*)(out+11*128),l0); _mm512_storeu_si512((void*)(out+11*128+64),l1);
}

void tm_avx512_r512_map_8::run_maps_range_x14_arr(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in, uint8* out)
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)(in+0*128)),a1=_mm512_loadu_si512((const void*)(in+0*128+64));
    __m512i b0=_mm512_loadu_si512((const void*)(in+1*128)),b1=_mm512_loadu_si512((const void*)(in+1*128+64));
    __m512i c0=_mm512_loadu_si512((const void*)(in+2*128)),c1=_mm512_loadu_si512((const void*)(in+2*128+64));
    __m512i d0=_mm512_loadu_si512((const void*)(in+3*128)),d1=_mm512_loadu_si512((const void*)(in+3*128+64));
    __m512i e0=_mm512_loadu_si512((const void*)(in+4*128)),e1=_mm512_loadu_si512((const void*)(in+4*128+64));
    __m512i f0=_mm512_loadu_si512((const void*)(in+5*128)),f1=_mm512_loadu_si512((const void*)(in+5*128+64));
    __m512i g0=_mm512_loadu_si512((const void*)(in+6*128)),g1=_mm512_loadu_si512((const void*)(in+6*128+64));
    __m512i h0=_mm512_loadu_si512((const void*)(in+7*128)),h1=_mm512_loadu_si512((const void*)(in+7*128+64));
    __m512i i0=_mm512_loadu_si512((const void*)(in+8*128)),i1=_mm512_loadu_si512((const void*)(in+8*128+64));
    __m512i j0=_mm512_loadu_si512((const void*)(in+9*128)),j1=_mm512_loadu_si512((const void*)(in+9*128+64));
    __m512i k0=_mm512_loadu_si512((const void*)(in+10*128)),k1=_mm512_loadu_si512((const void*)(in+10*128+64));
    __m512i l0=_mm512_loadu_si512((const void*)(in+11*128)),l1=_mm512_loadu_si512((const void*)(in+11*128+64));
    __m512i m0=_mm512_loadu_si512((const void*)(in+12*128)),m1=_mm512_loadu_si512((const void*)(in+12*128+64));
    __m512i n0=_mm512_loadu_si512((const void*)(in+13*128)),n1=_mm512_loadu_si512((const void*)(in+13*128+64));
    for (std::size_t map_idx = local_begin; map_idx < local_end; map_idx++)
        _run_map_entry_x14(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1,i0,i1,j0,j1,k0,k1,l0,l1,m0,m1,n0,n1, static_cast<int>(map_idx));
    _mm512_storeu_si512((void*)(out+0*128),a0); _mm512_storeu_si512((void*)(out+0*128+64),a1);
    _mm512_storeu_si512((void*)(out+1*128),b0); _mm512_storeu_si512((void*)(out+1*128+64),b1);
    _mm512_storeu_si512((void*)(out+2*128),c0); _mm512_storeu_si512((void*)(out+2*128+64),c1);
    _mm512_storeu_si512((void*)(out+3*128),d0); _mm512_storeu_si512((void*)(out+3*128+64),d1);
    _mm512_storeu_si512((void*)(out+4*128),e0); _mm512_storeu_si512((void*)(out+4*128+64),e1);
    _mm512_storeu_si512((void*)(out+5*128),f0); _mm512_storeu_si512((void*)(out+5*128+64),f1);
    _mm512_storeu_si512((void*)(out+6*128),g0); _mm512_storeu_si512((void*)(out+6*128+64),g1);
    _mm512_storeu_si512((void*)(out+7*128),h0); _mm512_storeu_si512((void*)(out+7*128+64),h1);
    _mm512_storeu_si512((void*)(out+8*128),i0); _mm512_storeu_si512((void*)(out+8*128+64),i1);
    _mm512_storeu_si512((void*)(out+9*128),j0); _mm512_storeu_si512((void*)(out+9*128+64),j1);
    _mm512_storeu_si512((void*)(out+10*128),k0); _mm512_storeu_si512((void*)(out+10*128+64),k1);
    _mm512_storeu_si512((void*)(out+11*128),l0); _mm512_storeu_si512((void*)(out+11*128+64),l1);
    _mm512_storeu_si512((void*)(out+12*128),m0); _mm512_storeu_si512((void*)(out+12*128+64),m1);
    _mm512_storeu_si512((void*)(out+13*128),n0); _mm512_storeu_si512((void*)(out+13*128+64),n1);
}

// W6: x10 merged-branchless map entry (10 lanes; mirrors _run_map_entry_x8_blmerge). Tests blmerge×interleave-10.
__forceinline void tm_avx512_r512_map_8::_run_map_entry_x10_blmerge(
    __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
    __m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
    __m512i& i0, __m512i& i1, __m512i& j0, __m512i& j1,
    int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa=2047,pb=2047,pc=2047,pd=2047,pe=2047,pf=2047,pg=2047,ph=2047,pi=2047,pj=2047;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i al=_mm512_castsi512_si128(a0),bl=_mm512_castsi512_si128(b0),cl=_mm512_castsi512_si128(c0),dl=_mm512_castsi512_si128(d0);
        const __m128i el=_mm512_castsi512_si128(e0),fl=_mm512_castsi512_si128(f0),gl=_mm512_castsi512_si128(g0),hl=_mm512_castsi512_si128(h0);
        const __m128i il=_mm512_castsi512_si128(i0),jl=_mm512_castsi512_si128(j0);
        uint8 ba=0,bb=0,bc=0,bd=0,be=0,bf=0,bg=0,bh=0,bi=0,bj=0;
        TM_MAP_XN_EXTRACT(ba,al,i); TM_MAP_XN_EXTRACT(bb,bl,i); TM_MAP_XN_EXTRACT(bc,cl,i); TM_MAP_XN_EXTRACT(bd,dl,i);
        TM_MAP_XN_EXTRACT(be,el,i); TM_MAP_XN_EXTRACT(bf,fl,i); TM_MAP_XN_EXTRACT(bg,gl,i); TM_MAP_XN_EXTRACT(bh,hl,i);
        TM_MAP_XN_EXTRACT(bi,il,i); TM_MAP_XN_EXTRACT(bj,jl,i);
        if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4;be>>=4;bf>>=4;bg>>=4;bh>>=4;bi>>=4;bj>>=4; }
        _alg_dispatch_blmerge(a0,a1,(ba>>1)&0x07,pa,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blmerge(b0,b1,(bb>>1)&0x07,pb,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blmerge(c0,c1,(bc>>1)&0x07,pc,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blmerge(d0,d1,(bd>>1)&0x07,pd,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blmerge(e0,e1,(be>>1)&0x07,pe,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blmerge(f0,f1,(bf>>1)&0x07,pf,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blmerge(g0,g1,(bg>>1)&0x07,pg,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blmerge(h0,h1,(bh>>1)&0x07,ph,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blmerge(i0,i1,(bi>>1)&0x07,pi,reg_base,alg0_base,alg6_base);
        _alg_dispatch_blmerge(j0,j1,(bj>>1)&0x07,pj,reg_base,alg0_base,alg6_base);
    }
}

// W6: x12 merged-branchless (12 lanes = 24 state ZMM). The 3 blmerge mask consts (mask_FE/7F/FF) are the
// secondary-ZMM pressure; g++ rematerializes them on demand (vpbroadcast) under pressure → spill-free/low-spill
// (same lever that made universal x12 spill-free); clang tends to pin → may spill. Build w/ g++ for x12.
__forceinline void tm_avx512_r512_map_8::_run_map_entry_x12_blmerge(
    __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
    __m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
    __m512i& i0, __m512i& i1, __m512i& j0, __m512i& j1, __m512i& k0, __m512i& k1, __m512i& l0, __m512i& l1,
    int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa=2047,pb=2047,pc=2047,pd=2047,pe=2047,pf=2047,pg=2047,ph=2047,pi=2047,pj=2047,pk=2047,pl=2047;
    // Rematerialization lever: local, address-taken masks (passed by ref to _rm dispatch) so the compiler
    // reloads them from stack under pressure instead of pinning 3 ZMM — leaves room for 24 state ZMM.
    __m512i mFE=_mm512_set1_epi8(static_cast<int8_t>(0xFE)), m7F=_mm512_set1_epi8(0x7F), mFF=_mm512_set1_epi8(static_cast<int8_t>(0xFF));

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i al=_mm512_castsi512_si128(a0),bl=_mm512_castsi512_si128(b0),cl=_mm512_castsi512_si128(c0),dl=_mm512_castsi512_si128(d0);
        const __m128i el=_mm512_castsi512_si128(e0),fl=_mm512_castsi512_si128(f0),gl=_mm512_castsi512_si128(g0),hl=_mm512_castsi512_si128(h0);
        const __m128i il=_mm512_castsi512_si128(i0),jl=_mm512_castsi512_si128(j0),kl=_mm512_castsi512_si128(k0),ll=_mm512_castsi512_si128(l0);
        uint8 ba=0,bb=0,bc=0,bd=0,be=0,bf=0,bg=0,bh=0,bi=0,bj=0,bk=0,bm=0;
        TM_MAP_XN_EXTRACT(ba,al,i); TM_MAP_XN_EXTRACT(bb,bl,i); TM_MAP_XN_EXTRACT(bc,cl,i); TM_MAP_XN_EXTRACT(bd,dl,i);
        TM_MAP_XN_EXTRACT(be,el,i); TM_MAP_XN_EXTRACT(bf,fl,i); TM_MAP_XN_EXTRACT(bg,gl,i); TM_MAP_XN_EXTRACT(bh,hl,i);
        TM_MAP_XN_EXTRACT(bi,il,i); TM_MAP_XN_EXTRACT(bj,jl,i); TM_MAP_XN_EXTRACT(bk,kl,i); TM_MAP_XN_EXTRACT(bm,ll,i);
        if (nibble == 1) { ba>>=4;bb>>=4;bc>>=4;bd>>=4;be>>=4;bf>>=4;bg>>=4;bh>>=4;bi>>=4;bj>>=4;bk>>=4;bm>>=4; }
        _alg_dispatch_blmerge_rm(a0,a1,(ba>>1)&0x07,pa,reg_base,alg0_base,alg6_base,mFE,m7F,mFF);
        _alg_dispatch_blmerge_rm(b0,b1,(bb>>1)&0x07,pb,reg_base,alg0_base,alg6_base,mFE,m7F,mFF);
        _alg_dispatch_blmerge_rm(c0,c1,(bc>>1)&0x07,pc,reg_base,alg0_base,alg6_base,mFE,m7F,mFF);
        _alg_dispatch_blmerge_rm(d0,d1,(bd>>1)&0x07,pd,reg_base,alg0_base,alg6_base,mFE,m7F,mFF);
        _alg_dispatch_blmerge_rm(e0,e1,(be>>1)&0x07,pe,reg_base,alg0_base,alg6_base,mFE,m7F,mFF);
        _alg_dispatch_blmerge_rm(f0,f1,(bf>>1)&0x07,pf,reg_base,alg0_base,alg6_base,mFE,m7F,mFF);
        _alg_dispatch_blmerge_rm(g0,g1,(bg>>1)&0x07,pg,reg_base,alg0_base,alg6_base,mFE,m7F,mFF);
        _alg_dispatch_blmerge_rm(h0,h1,(bh>>1)&0x07,ph,reg_base,alg0_base,alg6_base,mFE,m7F,mFF);
        _alg_dispatch_blmerge_rm(i0,i1,(bi>>1)&0x07,pi,reg_base,alg0_base,alg6_base,mFE,m7F,mFF);
        _alg_dispatch_blmerge_rm(j0,j1,(bj>>1)&0x07,pj,reg_base,alg0_base,alg6_base,mFE,m7F,mFF);
        _alg_dispatch_blmerge_rm(k0,k1,(bk>>1)&0x07,pk,reg_base,alg0_base,alg6_base,mFE,m7F,mFF);
        _alg_dispatch_blmerge_rm(l0,l1,(bm>>1)&0x07,pl,reg_base,alg0_base,alg6_base,mFE,m7F,mFF);
    }
}
#undef TM_MAP_XN_EXTRACT

void tm_avx512_r512_map_8::run_maps_range_x10_blmerge_arr(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in, uint8* out)
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)(in+0*128)),a1=_mm512_loadu_si512((const void*)(in+0*128+64));
    __m512i b0=_mm512_loadu_si512((const void*)(in+1*128)),b1=_mm512_loadu_si512((const void*)(in+1*128+64));
    __m512i c0=_mm512_loadu_si512((const void*)(in+2*128)),c1=_mm512_loadu_si512((const void*)(in+2*128+64));
    __m512i d0=_mm512_loadu_si512((const void*)(in+3*128)),d1=_mm512_loadu_si512((const void*)(in+3*128+64));
    __m512i e0=_mm512_loadu_si512((const void*)(in+4*128)),e1=_mm512_loadu_si512((const void*)(in+4*128+64));
    __m512i f0=_mm512_loadu_si512((const void*)(in+5*128)),f1=_mm512_loadu_si512((const void*)(in+5*128+64));
    __m512i g0=_mm512_loadu_si512((const void*)(in+6*128)),g1=_mm512_loadu_si512((const void*)(in+6*128+64));
    __m512i h0=_mm512_loadu_si512((const void*)(in+7*128)),h1=_mm512_loadu_si512((const void*)(in+7*128+64));
    __m512i i0=_mm512_loadu_si512((const void*)(in+8*128)),i1=_mm512_loadu_si512((const void*)(in+8*128+64));
    __m512i j0=_mm512_loadu_si512((const void*)(in+9*128)),j1=_mm512_loadu_si512((const void*)(in+9*128+64));
    for (std::size_t map_idx = local_begin; map_idx < local_end; map_idx++)
        _run_map_entry_x10_blmerge(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1,i0,i1,j0,j1, static_cast<int>(map_idx));
    _mm512_storeu_si512((void*)(out+0*128),a0); _mm512_storeu_si512((void*)(out+0*128+64),a1);
    _mm512_storeu_si512((void*)(out+1*128),b0); _mm512_storeu_si512((void*)(out+1*128+64),b1);
    _mm512_storeu_si512((void*)(out+2*128),c0); _mm512_storeu_si512((void*)(out+2*128+64),c1);
    _mm512_storeu_si512((void*)(out+3*128),d0); _mm512_storeu_si512((void*)(out+3*128+64),d1);
    _mm512_storeu_si512((void*)(out+4*128),e0); _mm512_storeu_si512((void*)(out+4*128+64),e1);
    _mm512_storeu_si512((void*)(out+5*128),f0); _mm512_storeu_si512((void*)(out+5*128+64),f1);
    _mm512_storeu_si512((void*)(out+6*128),g0); _mm512_storeu_si512((void*)(out+6*128+64),g1);
    _mm512_storeu_si512((void*)(out+7*128),h0); _mm512_storeu_si512((void*)(out+7*128+64),h1);
    _mm512_storeu_si512((void*)(out+8*128),i0); _mm512_storeu_si512((void*)(out+8*128+64),i1);
    _mm512_storeu_si512((void*)(out+9*128),j0); _mm512_storeu_si512((void*)(out+9*128+64),j1);
}

void tm_avx512_r512_map_8::run_maps_range_x12_blmerge_arr(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in, uint8* out)
{
    if (end > schedule_entries.entries.size()) end = schedule_entries.entries.size();
    if (begin > end) begin = end;
    std::size_t local_begin = begin, local_end = end;
    if (_shared == nullptr) {
        if (_owned.entries_data != schedule_entries.entries.data() + begin || _owned.entry_count != static_cast<int>(end - begin))
            _owned.bind_range(rng, schedule_entries, begin, end);
        local_begin = 0; local_end = end - begin;
    } else {
        const auto& t = _t();
        if (t.entries_data != schedule_entries.entries.data() || t.entry_count != static_cast<int>(schedule_entries.entries.size()))
            _bind_schedule(schedule_entries);
    }
    __m512i a0=_mm512_loadu_si512((const void*)(in+0*128)),a1=_mm512_loadu_si512((const void*)(in+0*128+64));
    __m512i b0=_mm512_loadu_si512((const void*)(in+1*128)),b1=_mm512_loadu_si512((const void*)(in+1*128+64));
    __m512i c0=_mm512_loadu_si512((const void*)(in+2*128)),c1=_mm512_loadu_si512((const void*)(in+2*128+64));
    __m512i d0=_mm512_loadu_si512((const void*)(in+3*128)),d1=_mm512_loadu_si512((const void*)(in+3*128+64));
    __m512i e0=_mm512_loadu_si512((const void*)(in+4*128)),e1=_mm512_loadu_si512((const void*)(in+4*128+64));
    __m512i f0=_mm512_loadu_si512((const void*)(in+5*128)),f1=_mm512_loadu_si512((const void*)(in+5*128+64));
    __m512i g0=_mm512_loadu_si512((const void*)(in+6*128)),g1=_mm512_loadu_si512((const void*)(in+6*128+64));
    __m512i h0=_mm512_loadu_si512((const void*)(in+7*128)),h1=_mm512_loadu_si512((const void*)(in+7*128+64));
    __m512i i0=_mm512_loadu_si512((const void*)(in+8*128)),i1=_mm512_loadu_si512((const void*)(in+8*128+64));
    __m512i j0=_mm512_loadu_si512((const void*)(in+9*128)),j1=_mm512_loadu_si512((const void*)(in+9*128+64));
    __m512i k0=_mm512_loadu_si512((const void*)(in+10*128)),k1=_mm512_loadu_si512((const void*)(in+10*128+64));
    __m512i l0=_mm512_loadu_si512((const void*)(in+11*128)),l1=_mm512_loadu_si512((const void*)(in+11*128+64));
    for (std::size_t map_idx = local_begin; map_idx < local_end; map_idx++)
        _run_map_entry_x12_blmerge(a0,a1,b0,b1,c0,c1,d0,d1,e0,e1,f0,f1,g0,g1,h0,h1,i0,i1,j0,j1,k0,k1,l0,l1, static_cast<int>(map_idx));
    _mm512_storeu_si512((void*)(out+0*128),a0); _mm512_storeu_si512((void*)(out+0*128+64),a1);
    _mm512_storeu_si512((void*)(out+1*128),b0); _mm512_storeu_si512((void*)(out+1*128+64),b1);
    _mm512_storeu_si512((void*)(out+2*128),c0); _mm512_storeu_si512((void*)(out+2*128+64),c1);
    _mm512_storeu_si512((void*)(out+3*128),d0); _mm512_storeu_si512((void*)(out+3*128+64),d1);
    _mm512_storeu_si512((void*)(out+4*128),e0); _mm512_storeu_si512((void*)(out+4*128+64),e1);
    _mm512_storeu_si512((void*)(out+5*128),f0); _mm512_storeu_si512((void*)(out+5*128+64),f1);
    _mm512_storeu_si512((void*)(out+6*128),g0); _mm512_storeu_si512((void*)(out+6*128+64),g1);
    _mm512_storeu_si512((void*)(out+7*128),h0); _mm512_storeu_si512((void*)(out+7*128+64),h1);
    _mm512_storeu_si512((void*)(out+8*128),i0); _mm512_storeu_si512((void*)(out+8*128+64),i1);
    _mm512_storeu_si512((void*)(out+9*128),j0); _mm512_storeu_si512((void*)(out+9*128+64),j1);
    _mm512_storeu_si512((void*)(out+10*128),k0); _mm512_storeu_si512((void*)(out+10*128+64),k1);
    _mm512_storeu_si512((void*)(out+11*128),l0); _mm512_storeu_si512((void*)(out+11*128+64),l1);
}

// ---------------------------------------------------------------------------
// Checksum helpers (natural layout). Mirror tm_avx2_r256_map_8 but on 2 ZMM.
// ---------------------------------------------------------------------------
__forceinline uint16 tm_avx512_r512_map_8::masked_checksum(__m512i& wc0, __m512i& wc1, const uint8* mask)
{
    // Sum of (state[i] & mask[i]) over all 128 bytes, as 16-bit accumulation.
    __m512i m0 = _mm512_load_si512((const __m512i*)(mask));
    __m512i m1 = _mm512_load_si512((const __m512i*)(mask + 64));
    __m512i v0 = _mm512_and_si512(wc0, m0);
    __m512i v1 = _mm512_and_si512(wc1, m1);
    // sum_epu8 of each 64-bit lane -> 16 partial sums each; add lanes.
    __m512i s = _mm512_add_epi64(_mm512_sad_epu8(v0, _mm512_setzero_si512()),
                                 _mm512_sad_epu8(v1, _mm512_setzero_si512()));
    // s holds 8 x 64-bit partial byte-sums; horizontal add in one intrinsic.
    return static_cast<uint16>(_mm512_reduce_add_epi64(s));
}

__forceinline uint16 tm_avx512_r512_map_8::fetch_checksum_value(__m512i& wc0, __m512i& wc1, uint8 code_length)
{
    _store_to_mem(wc0, wc1);
    uint8 hi = working_code_data[127 - (code_length + 1)];
    uint8 lo = working_code_data[127 - code_length];
    return static_cast<uint16>((hi << 8) | lo);
}

__forceinline bool tm_avx512_r512_map_8::check_carnival_world_checksum(__m512i& wc0, __m512i& wc1)
{
    uint16 calc = masked_checksum(wc0, wc1, TM_base::carnival_world_checksum_mask);
    return calc == fetch_checksum_value(wc0, wc1, CARNIVAL_WORLD_CODE_LENGTH - 2);
}

__forceinline bool tm_avx512_r512_map_8::check_other_world_checksum(__m512i& wc0, __m512i& wc1)
{
    uint16 calc = masked_checksum(wc0, wc1, TM_base::other_world_checksum_mask);
    return calc == fetch_checksum_value(wc0, wc1, OTHER_WORLD_CODE_LENGTH - 2);
}

__forceinline void tm_avx512_r512_map_8::_decrypt_carnival_world(__m512i& wc0, __m512i& wc1)
{
    xor_alg(wc0, wc1, TM_base::carnival_world_data);
}

__forceinline void tm_avx512_r512_map_8::_decrypt_other_world(__m512i& wc0, __m512i& wc1)
{
    xor_alg(wc0, wc1, TM_base::other_world_data);
}

__forceinline void tm_avx512_r512_map_8::_screen_emit(__m512i w0, __m512i w1, uint32 idx, uint8* result_data, uint32& output_pos)
{
    __m512i a0 = w0, a1 = w1;
    _decrypt_carnival_world(a0, a1);
    if (check_carnival_world_checksum(a0, a1))
    {
        _store_to_mem(a0, a1);
        *((uint32*)(&result_data[output_pos])) = idx;
        result_data[output_pos + 4] = check_machine_code(working_code_data, CARNIVAL_WORLD);
        output_pos += 5;
        return;
    }
    a0 = w0; a1 = w1;
    _decrypt_other_world(a0, a1);
    if (check_other_world_checksum(a0, a1))
    {
        _store_to_mem(a0, a1);
        *((uint32*)(&result_data[output_pos])) = idx;
        result_data[output_pos + 4] = check_machine_code(working_code_data, OTHER_WORLD);
        output_pos += 5;
    }
}

bool tm_avx512_r512_map_8::screen_state_raw(const uint8* src, uint8& flags_out)
{
    __m512i w0 = _mm512_loadu_si512((const void*)src);
    __m512i w1 = _mm512_loadu_si512((const void*)(src + 64));
    __m512i a0 = w0, a1 = w1;
    _decrypt_carnival_world(a0, a1);
    if (check_carnival_world_checksum(a0, a1))
    {
        _store_to_mem(a0, a1);
        flags_out = check_machine_code(working_code_data, CARNIVAL_WORLD);
        return true;
    }

    a0 = w0; a1 = w1;
    _decrypt_other_world(a0, a1);
    if (check_other_world_checksum(a0, a1))
    {
        _store_to_mem(a0, a1);
        flags_out = check_machine_code(working_code_data, OTHER_WORLD);
        return true;
    }

    flags_out = 0;
    return false;
}

void tm_avx512_r512_map_8::run_bruteforce_data(uint32 key, uint32 start_data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size)
{
    _bind_key(key);
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

void tm_avx512_r512_map_8::run_bruteforce_data_il(uint32 key, uint32 start_data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size)
{
    _bind_key(key);
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

bool tm_avx512_r512_map_8::initialized = false;
