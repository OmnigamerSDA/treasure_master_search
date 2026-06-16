#include <stdio.h>
#include <cstdlib>
#include <cstring>

#include "tm_avx2_r256_map_8.h"
#include "rng.h"
#include "routing.h"   // tm_routing::shed_proxy_map1 (raceway producer routing score)

// AVX2-native raceway interleave width for the x8-signature kernels. 2 = no-spill
// half-register-file (default); 4 = full file, spills the butterfly temps.
// With the BRANCHED dispatch x4 won +17-19% (its extra ILP hid the alg-dispatch
// mispredict latency). With the production BRANCHLESS dispatch (blmerge, the
// default deep path) the mispredict wall is gone (23%->11.5%) and x2 ≈ x4 (T16 x2
// 3.59 ≥ x4 3.54 M/s) — so x2 is the better default: same throughput, zero spills,
// more robust on the older/narrower AVX2 uarchs this port targets. -DAVX2_RACEWAY_W=4
#ifndef AVX2_RACEWAY_W
#define AVX2_RACEWAY_W 2
#endif

tm_avx2_r256_map_8::tm_avx2_r256_map_8(RNG* rng_obj)
    : TM_base(rng_obj),
      mask_FF(_mm256_set1_epi8(static_cast<int8_t>(0xFF))),
      mask_FE(_mm256_set1_epi8(static_cast<int8_t>(0xFE))),
      mask_7F(_mm256_set1_epi8(0x7F)),
      mask_80(_mm256_set1_epi8(static_cast<int8_t>(0x80))),
      mask_01(_mm256_set1_epi8(0x01))
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

// ---------------------------------------------------------------------------
// alg_2 / alg_5 are an even-right / odd-left byte "butterfly", not a uniform
// rotate (ground truth: tm_8::alg_2/alg_5). Each canonical byte pulls one carry
// bit from its neighbor at index +1:
//   alg_2: even e' = (e>>1)|(b[e+1]&0x80)   odd o' = (o<<1)|(b[o+1]&0x01)
//   alg_5: even e' = (e<<1)|(b[e+1]&0x01)   odd o' = (o>>1)|(b[o+1]&0x80)
// (byte 127, odd, takes the rng-seeded carry instead of b[128].)
//
// This byte-granular form replaces the old _epi16 word-pair port (~18-20 ops +
// an alignr/permute2x128 carry per 256-bit half, called with a threaded carry).
// Because every byte's carry comes from byte i+1, feeding the NEXT 256-bit half
// into the neighbor shift supplies the cross-half boundary byte directly — so the
// inter-register carry threading is gone entirely (mirrors the AVX-512 natmap
// rework). Per half: two per-byte shifts + an even/odd byte blend + a
// masked-neighbor OR, plus one permute2x128+alignr neighbor.

// neighbor[i] = byte[i+1] across a 256-bit half, with byte31 <- next half's
// byte0. permute2x128(0x21) = [wc.hi128, next.lo128]; alignr by 1 then yields the
// byte-down-by-1 including the cross-128-lane and cross-half boundary.
static __forceinline __m256i avx2_nbr(__m256i wc, __m256i next)
{
    return _mm256_alignr_epi8(_mm256_permute2x128_si256(wc, next, 0x21), wc, 1);
}

// One butterfly half. lo byte of each 16-bit word = even canonical byte, hi =
// odd. alg_2 (srli_first): even<-(b>>1)&7F, odd<-(b<<1)&FE; alg_5 reversed.
template <bool srli_first>
static __forceinline __m256i avx2_butterfly(__m256i wc, __m256i neighbor)
{
    const __m256i m_7F = _mm256_set1_epi16(0x7F7F);
    const __m256i m_FE = _mm256_set1_epi16(static_cast<short>(0xFEFE));
    const __m256i m_00FF = _mm256_set1_epi16(0x00FF);
    const __m256i m_FF00 = _mm256_set1_epi16(static_cast<short>(0xFF00));
    const __m256i carry_mask = srli_first
        ? _mm256_set1_epi16(0x0180)                       // even=0x80(bit7), odd=0x01(bit0)
        : _mm256_set1_epi16(static_cast<short>(0x8001));  // even=0x01, odd=0x80
    __m256i sr = _mm256_and_si256(_mm256_srli_epi16(wc, 1), m_7F); // per-byte >>1
    __m256i sl = _mm256_and_si256(_mm256_slli_epi16(wc, 1), m_FE); // per-byte <<1
    __m256i shifted = srli_first
        ? _mm256_or_si256(_mm256_and_si256(sr, m_00FF), _mm256_and_si256(sl, m_FF00))  // even<-sr, odd<-sl
        : _mm256_or_si256(_mm256_and_si256(sl, m_00FF), _mm256_and_si256(sr, m_FF00)); // even<-sl, odd<-sr
    return _mm256_or_si256(shifted, _mm256_and_si256(neighbor, carry_mask));
}

// Extract canonical byte i (compile-time-indexed via switch) from a 128-bit low
// half. Shared by the interleaved map-entry kernels (mirrors the AVX-512
// TM_MAP_XN_EXTRACT). i is a runtime value here; the switch supplies the immediate.
#define TM_MAP_AVX2_EXTRACT(out, r, i) \
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
    // seed = byte 127's missing neighbor (only bit0 matters: odd lane, mask 0x01).
    const __m256i seed = _mm256_castsi128_si256(_mm_cvtsi32_si128(static_cast<int>(carry_byte)));
    // Compute each half's neighbor (from the ORIGINAL next half) before writing it.
    __m256i n0 = avx2_nbr(wc0, wc1); wc0 = avx2_butterfly<true>(wc0, n0);
    __m256i n1 = avx2_nbr(wc1, wc2); wc1 = avx2_butterfly<true>(wc1, n1);
    __m256i n2 = avx2_nbr(wc2, wc3); wc2 = avx2_butterfly<true>(wc2, n2);
    __m256i n3 = avx2_nbr(wc3, seed); wc3 = avx2_butterfly<true>(wc3, n3);
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
    // seed bit7 (odd lane mask 0x80); carry_byte = reg_base[pos] & 0x80.
    const __m256i seed = _mm256_castsi128_si256(_mm_cvtsi32_si128(static_cast<int>(carry_byte)));
    __m256i n0 = avx2_nbr(wc0, wc1); wc0 = avx2_butterfly<false>(wc0, n0);
    __m256i n1 = avx2_nbr(wc1, wc2); wc1 = avx2_butterfly<false>(wc1, n1);
    __m256i n2 = avx2_nbr(wc2, wc3); wc2 = avx2_butterfly<false>(wc2, n2);
    __m256i n3 = avx2_nbr(wc3, seed); wc3 = avx2_butterfly<false>(wc3, n3);
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
#if defined(__GNUC__)
#pragma GCC unroll 16
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;

        uint8 current_byte = 0;
#if defined(_MSC_VER) || defined(__clang__)
        const __m128i wc0_low = _mm256_castsi256_si128(wc0);
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
        current_byte = static_cast<uint8>(_mm_extract_epi8(_mm256_castsi256_si128(wc0), i));
#endif
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

__forceinline void tm_avx2_r256_map_8::_screen_emit(__m256i w0, __m256i w1, __m256i w2, __m256i w3, uint32 idx, uint8* result_data, uint32& output_pos)
{
    __m256i a0 = w0, a1 = w1, a2 = w2, a3 = w3;
    _decrypt_carnival_world(a0, a1, a2, a3);
    if (check_carnival_world_checksum(a0, a1, a2, a3))
    {
        _store_to_mem(a0, a1, a2, a3);
        *((uint32*)(&result_data[output_pos])) = idx;
        result_data[output_pos + 4] = check_machine_code(working_code_data, CARNIVAL_WORLD);
        output_pos += 5;
        return;
    }
    a0 = w0; a1 = w1; a2 = w2; a3 = w3;
    _decrypt_other_world(a0, a1, a2, a3);
    if (check_other_world_checksum(a0, a1, a2, a3))
    {
        _store_to_mem(a0, a1, a2, a3);
        *((uint32*)(&result_data[output_pos])) = idx;
        result_data[output_pos + 4] = check_machine_code(working_code_data, OTHER_WORLD);
        output_pos += 5;
    }
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
        _screen_emit(wc0, wc1, wc2, wc3, i, result_data, output_pos);

        report_progress(static_cast<double>(i + 1) / static_cast<double>(amount_to_run));
    }
    *result_size = output_pos;
}

// ---------------------------------------------------------------------------
// 2-way interleaved kernel + bulk forward.
// ---------------------------------------------------------------------------
__forceinline void tm_avx2_r256_map_8::_run_map_entry_x2(
    __m256i& a0, __m256i& a1, __m256i& a2, __m256i& a3,
    __m256i& b0, __m256i& b1, __m256i& b2, __m256i& b3, int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa = 2047, pb = 2047;

    // NOT unrolled (mirrors the 512 maps' _run_map_entry_x4): the switch(i)
    // supplies the _mm_extract_epi8 immediate per case.
#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;

        const __m128i al = _mm256_castsi256_si128(a0);
        const __m128i bl = _mm256_castsi256_si128(b0);
        uint8 ba = 0, bb = 0;
#define TM_MAP_X2_EXTRACT(out, r) \
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
        TM_MAP_X2_EXTRACT(ba, al);
        TM_MAP_X2_EXTRACT(bb, bl);
#undef TM_MAP_X2_EXTRACT
        if (nibble == 1) { ba >>= 4; bb >>= 4; }

        _run_alg(a0, a1, a2, a3, (ba >> 1) & 0x07, &pa, reg_base, alg0_base, alg6_base);
        _run_alg(b0, b1, b2, b3, (bb >> 1) & 0x07, &pb, reg_base, alg0_base, alg6_base);
    }
}

void tm_avx2_r256_map_8::run_maps_range_x2(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, uint8* out0, uint8* out1)
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

    __m256i a0 = _mm256_loadu_si256((const __m256i*)(in0));
    __m256i a1 = _mm256_loadu_si256((const __m256i*)(in0 + 32));
    __m256i a2 = _mm256_loadu_si256((const __m256i*)(in0 + 64));
    __m256i a3 = _mm256_loadu_si256((const __m256i*)(in0 + 96));
    __m256i b0 = _mm256_loadu_si256((const __m256i*)(in1));
    __m256i b1 = _mm256_loadu_si256((const __m256i*)(in1 + 32));
    __m256i b2 = _mm256_loadu_si256((const __m256i*)(in1 + 64));
    __m256i b3 = _mm256_loadu_si256((const __m256i*)(in1 + 96));

    for (std::size_t map_idx = local_begin; map_idx < local_end; map_idx++)
        _run_map_entry_x2(a0, a1, a2, a3, b0, b1, b2, b3, static_cast<int>(map_idx));

    _mm256_storeu_si256((__m256i*)(out0),      a0);
    _mm256_storeu_si256((__m256i*)(out0 + 32), a1);
    _mm256_storeu_si256((__m256i*)(out0 + 64), a2);
    _mm256_storeu_si256((__m256i*)(out0 + 96), a3);
    _mm256_storeu_si256((__m256i*)(out1),      b0);
    _mm256_storeu_si256((__m256i*)(out1 + 32), b1);
    _mm256_storeu_si256((__m256i*)(out1 + 64), b2);
    _mm256_storeu_si256((__m256i*)(out1 + 96), b3);
}

void tm_avx2_r256_map_8::run_bruteforce_data_il(uint32 key, uint32 start_data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size)
{
    _bind_key(key);
    _bind_schedule(schedule_entries);   // ensures tables bound (incl. the <2 tail) and avoids per-batch rebind
    const std::size_t n = static_cast<std::size_t>(_t().entry_count);

    alignas(32) uint8 in0[128], in1[128], out0[128], out1[128];

    uint32 output_pos = 0;
    uint32 i = 0;
    for (; i + 2 <= amount_to_run; i += 2)
    {
        if ((result_max_size - output_pos) < 10)   // up to 2 × 5-byte records
        {
            *result_size = output_pos;
            return;
        }

        __m256i e0, e1, e2, e3;
        _expand_code(start_data + i + 0, e0, e1, e2, e3);
        _mm256_store_si256((__m256i*)in0, e0); _mm256_store_si256((__m256i*)(in0 + 32), e1);
        _mm256_store_si256((__m256i*)(in0 + 64), e2); _mm256_store_si256((__m256i*)(in0 + 96), e3);
        _expand_code(start_data + i + 1, e0, e1, e2, e3);
        _mm256_store_si256((__m256i*)in1, e0); _mm256_store_si256((__m256i*)(in1 + 32), e1);
        _mm256_store_si256((__m256i*)(in1 + 64), e2); _mm256_store_si256((__m256i*)(in1 + 96), e3);

        run_maps_range_x2(schedule_entries, 0, n, in0, in1, out0, out1);

        _screen_emit(
            _mm256_load_si256((const __m256i*)out0), _mm256_load_si256((const __m256i*)(out0 + 32)),
            _mm256_load_si256((const __m256i*)(out0 + 64)), _mm256_load_si256((const __m256i*)(out0 + 96)),
            i + 0, result_data, output_pos);
        _screen_emit(
            _mm256_load_si256((const __m256i*)out1), _mm256_load_si256((const __m256i*)(out1 + 32)),
            _mm256_load_si256((const __m256i*)(out1 + 64)), _mm256_load_si256((const __m256i*)(out1 + 96)),
            i + 1, result_data, output_pos);

        report_progress(static_cast<double>(i + 2) / static_cast<double>(amount_to_run));
    }
    // Tail (<2 remaining): 1-way.
    for (; i < amount_to_run; i++)
    {
        if ((result_max_size - output_pos) < 5)
        {
            *result_size = output_pos;
            return;
        }
        __m256i wc0, wc1, wc2, wc3;
        _expand_code(start_data + i, wc0, wc1, wc2, wc3);
        _run_all_maps(wc0, wc1, wc2, wc3);
        _screen_emit(wc0, wc1, wc2, wc3, i, result_data, output_pos);
        report_progress(static_cast<double>(i + 1) / static_cast<double>(amount_to_run));
    }
    *result_size = output_pos;
}

// ---------------------------------------------------------------------------
// Raceway interleaved kernels (AVX2 port of the AVX-512 r512_map raceway).
// ---------------------------------------------------------------------------

// Shared bind preamble: resolve [begin,end) against the owned/shared tables and
// return the local (table-relative) map range. Same logic the x2 path inlines.
void tm_avx2_r256_map_8::_ileave_resolve(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    std::size_t& local_begin, std::size_t& local_end)
{
    if (end > schedule_entries.entries.size())
        end = schedule_entries.entries.size();
    if (begin > end)
        begin = end;

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
        local_begin = begin;
        local_end = end;
    }
}

// One schedule entry over four interleaved states (16 YMM). Mirrors
// _run_map_entry_x2 at 4 lanes; the AVX-512 _run_map_entry_x8 is the analog.
__forceinline void tm_avx2_r256_map_8::_run_map_entry_x4(
    __m256i& a0, __m256i& a1, __m256i& a2, __m256i& a3,
    __m256i& b0, __m256i& b1, __m256i& b2, __m256i& b3,
    __m256i& c0, __m256i& c1, __m256i& c2, __m256i& c3,
    __m256i& d0, __m256i& d1, __m256i& d2, __m256i& d3, int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa = 2047, pb = 2047, pc = 2047, pd = 2047;

#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;

        const __m128i al = _mm256_castsi256_si128(a0);
        const __m128i bl = _mm256_castsi256_si128(b0);
        const __m128i cl = _mm256_castsi256_si128(c0);
        const __m128i dl = _mm256_castsi256_si128(d0);
        uint8 ba = 0, bb = 0, bc = 0, bd = 0;
        TM_MAP_AVX2_EXTRACT(ba, al, i);
        TM_MAP_AVX2_EXTRACT(bb, bl, i);
        TM_MAP_AVX2_EXTRACT(bc, cl, i);
        TM_MAP_AVX2_EXTRACT(bd, dl, i);
        if (nibble == 1) { ba >>= 4; bb >>= 4; bc >>= 4; bd >>= 4; }

        _run_alg(a0, a1, a2, a3, (ba >> 1) & 0x07, &pa, reg_base, alg0_base, alg6_base);
        _run_alg(b0, b1, b2, b3, (bb >> 1) & 0x07, &pb, reg_base, alg0_base, alg6_base);
        _run_alg(c0, c1, c2, c3, (bc >> 1) & 0x07, &pc, reg_base, alg0_base, alg6_base);
        _run_alg(d0, d1, d2, d3, (bd >> 1) & 0x07, &pd, reg_base, alg0_base, alg6_base);
    }
}

// Register-resident 2-lane sweep over the local map range (tables pre-bound).
__forceinline void tm_avx2_r256_map_8::_sweep_x2(const uint8* in0, const uint8* in1, uint8* out0, uint8* out1,
    std::size_t lb, std::size_t le)
{
    __m256i a0 = _mm256_loadu_si256((const __m256i*)(in0));
    __m256i a1 = _mm256_loadu_si256((const __m256i*)(in0 + 32));
    __m256i a2 = _mm256_loadu_si256((const __m256i*)(in0 + 64));
    __m256i a3 = _mm256_loadu_si256((const __m256i*)(in0 + 96));
    __m256i b0 = _mm256_loadu_si256((const __m256i*)(in1));
    __m256i b1 = _mm256_loadu_si256((const __m256i*)(in1 + 32));
    __m256i b2 = _mm256_loadu_si256((const __m256i*)(in1 + 64));
    __m256i b3 = _mm256_loadu_si256((const __m256i*)(in1 + 96));
    for (std::size_t map_idx = lb; map_idx < le; map_idx++)
        _run_map_entry_x2(a0, a1, a2, a3, b0, b1, b2, b3, static_cast<int>(map_idx));
    _mm256_storeu_si256((__m256i*)(out0),      a0);
    _mm256_storeu_si256((__m256i*)(out0 + 32), a1);
    _mm256_storeu_si256((__m256i*)(out0 + 64), a2);
    _mm256_storeu_si256((__m256i*)(out0 + 96), a3);
    _mm256_storeu_si256((__m256i*)(out1),      b0);
    _mm256_storeu_si256((__m256i*)(out1 + 32), b1);
    _mm256_storeu_si256((__m256i*)(out1 + 64), b2);
    _mm256_storeu_si256((__m256i*)(out1 + 96), b3);
}

// Register-resident 4-lane sweep over the local map range (tables pre-bound).
__forceinline void tm_avx2_r256_map_8::_sweep_x4(
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3, std::size_t lb, std::size_t le)
{
    __m256i a0 = _mm256_loadu_si256((const __m256i*)(in0));
    __m256i a1 = _mm256_loadu_si256((const __m256i*)(in0 + 32));
    __m256i a2 = _mm256_loadu_si256((const __m256i*)(in0 + 64));
    __m256i a3 = _mm256_loadu_si256((const __m256i*)(in0 + 96));
    __m256i b0 = _mm256_loadu_si256((const __m256i*)(in1));
    __m256i b1 = _mm256_loadu_si256((const __m256i*)(in1 + 32));
    __m256i b2 = _mm256_loadu_si256((const __m256i*)(in1 + 64));
    __m256i b3 = _mm256_loadu_si256((const __m256i*)(in1 + 96));
    __m256i c0 = _mm256_loadu_si256((const __m256i*)(in2));
    __m256i c1 = _mm256_loadu_si256((const __m256i*)(in2 + 32));
    __m256i c2 = _mm256_loadu_si256((const __m256i*)(in2 + 64));
    __m256i c3 = _mm256_loadu_si256((const __m256i*)(in2 + 96));
    __m256i d0 = _mm256_loadu_si256((const __m256i*)(in3));
    __m256i d1 = _mm256_loadu_si256((const __m256i*)(in3 + 32));
    __m256i d2 = _mm256_loadu_si256((const __m256i*)(in3 + 64));
    __m256i d3 = _mm256_loadu_si256((const __m256i*)(in3 + 96));
    for (std::size_t map_idx = lb; map_idx < le; map_idx++)
        _run_map_entry_x4(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3, static_cast<int>(map_idx));
    _mm256_storeu_si256((__m256i*)(out0),      a0); _mm256_storeu_si256((__m256i*)(out0 + 32), a1);
    _mm256_storeu_si256((__m256i*)(out0 + 64), a2); _mm256_storeu_si256((__m256i*)(out0 + 96), a3);
    _mm256_storeu_si256((__m256i*)(out1),      b0); _mm256_storeu_si256((__m256i*)(out1 + 32), b1);
    _mm256_storeu_si256((__m256i*)(out1 + 64), b2); _mm256_storeu_si256((__m256i*)(out1 + 96), b3);
    _mm256_storeu_si256((__m256i*)(out2),      c0); _mm256_storeu_si256((__m256i*)(out2 + 32), c1);
    _mm256_storeu_si256((__m256i*)(out2 + 64), c2); _mm256_storeu_si256((__m256i*)(out2 + 96), c3);
    _mm256_storeu_si256((__m256i*)(out3),      d0); _mm256_storeu_si256((__m256i*)(out3 + 32), d1);
    _mm256_storeu_si256((__m256i*)(out3 + 64), d2); _mm256_storeu_si256((__m256i*)(out3 + 96), d3);
}

void tm_avx2_r256_map_8::run_maps_range_x4(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3)
{
    std::size_t lb, le;
    _ileave_resolve(schedule_entries, begin, end, lb, le);
    _sweep_x4(in0, in1, in2, in3, out0, out1, out2, out3, lb, le);
}

void tm_avx2_r256_map_8::run_maps_range_x8(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
    const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3,
    uint8* out4, uint8* out5, uint8* out6, uint8* out7)
{
    std::size_t lb, le;
    _ileave_resolve(schedule_entries, begin, end, lb, le);
#if AVX2_RACEWAY_W == 4
    _sweep_x4(in0, in1, in2, in3, out0, out1, out2, out3, lb, le);
    _sweep_x4(in4, in5, in6, in7, out4, out5, out6, out7, lb, le);
#else
    _sweep_x2(in0, in1, out0, out1, lb, le);
    _sweep_x2(in2, in3, out2, out3, lb, le);
    _sweep_x2(in4, in5, out4, out5, lb, le);
    _sweep_x2(in6, in7, out6, out7, lb, le);
#endif
}

// Branchless alg dispatch (AVX2 port of the AVX-512 _alg_dispatch_blmerge). Folds the
// arith algs into 3 candidates selected by a blendv tree; butterflies {2,5} branched.
//   {1,4}: w + (R or -R)         (add/sub, operand sign-selected)
//   {3,7}: w ^ (R or 0xFF)       (xor, operand-selected)
//   {0,6}: (w<<1 & FE | Z) or (w>>1 & 7F | S)   (shift dir + mask + rng selected)
// Masks are uniform 0x00/0xFF byte vectors built from the scalar alg_id (no branch);
// _mm256_blendv_epi8(a,b,m) picks b where m's high bit is set == AVX-512 mask_blend(m,a,b).
__forceinline void tm_avx2_r256_map_8::_alg_dispatch_blmerge(
    __m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3, int alg_id, uint16& local_pos,
    const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base)
{
    if (alg_id == 2) { alg_2(w0, w1, w2, w3, reg_base[local_pos] >> 7);   local_pos -= 1; return; }
    if (alg_id == 5) { alg_5(w0, w1, w2, w3, reg_base[local_pos] & 0x80); local_pos -= 1; return; }
    const uint8* rb = reg_base  + local_pos - 127;
    const uint8* zb = alg0_base + local_pos - 127;
    const uint8* sb = alg6_base + (2047 - local_pos);
    const __m256i m6  = _mm256_set1_epi8(static_cast<char>(-(int)(alg_id == 6)));            // right-shift vs left
    const __m256i m4  = _mm256_set1_epi8(static_cast<char>(-(int)(alg_id == 4)));            // negate operand (sub)
    const __m256i m7  = _mm256_set1_epi8(static_cast<char>(-(int)(alg_id == 7)));            // xor with 0xFF
    const __m256i mas = _mm256_set1_epi8(static_cast<char>(-(int)(alg_id == 1 || alg_id == 4))); // pick add/sub
    const __m256i mx  = _mm256_set1_epi8(static_cast<char>(-(int)(alg_id == 3 || alg_id == 7))); // pick xor
    const __m256i zero  = _mm256_setzero_si256();
    const __m256i smask = _mm256_blendv_epi8(mask_FE, mask_7F, m6);
#define TM_BLMERGE_QUARTER(W, OFF) do { \
        const __m256i R = _mm256_loadu_si256((const __m256i*)(rb + (OFF))); \
        const __m256i Z = _mm256_loadu_si256((const __m256i*)(zb + (OFF))); \
        const __m256i S = _mm256_loadu_si256((const __m256i*)(sb + (OFF))); \
        const __m256i shifted = _mm256_blendv_epi8(_mm256_slli_epi16(W, 1), _mm256_srli_epi16(W, 1), m6); \
        const __m256i srng    = _mm256_blendv_epi8(Z, S, m6); \
        const __m256i c_sh = _mm256_or_si256(_mm256_and_si256(shifted, smask), srng); \
        const __m256i c_as = _mm256_add_epi8(W, _mm256_blendv_epi8(R, _mm256_sub_epi8(zero, R), m4)); \
        const __m256i c_x  = _mm256_xor_si256(W, _mm256_blendv_epi8(R, mask_FF, m7)); \
        const __m256i res  = _mm256_blendv_epi8(c_sh, c_as, mas); \
        W = _mm256_blendv_epi8(res, c_x, mx); \
    } while (0)
    TM_BLMERGE_QUARTER(w0, 0);
    TM_BLMERGE_QUARTER(w1, 32);
    TM_BLMERGE_QUARTER(w2, 64);
    TM_BLMERGE_QUARTER(w3, 96);
#undef TM_BLMERGE_QUARTER
    // algs 0,1,3,4,6 advance pos by 128; alg_7 (folded into the xor candidate) must not.
    local_pos -= static_cast<uint16>((alg_id != 7) * 128);
}

__forceinline void tm_avx2_r256_map_8::_run_map_entry_x2_blmerge(
    __m256i& a0, __m256i& a1, __m256i& a2, __m256i& a3,
    __m256i& b0, __m256i& b1, __m256i& b2, __m256i& b3, int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa = 2047, pb = 2047;
#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i al = _mm256_castsi256_si128(a0);
        const __m128i bl = _mm256_castsi256_si128(b0);
        uint8 ba = 0, bb = 0;
        TM_MAP_AVX2_EXTRACT(ba, al, i);
        TM_MAP_AVX2_EXTRACT(bb, bl, i);
        if (nibble == 1) { ba >>= 4; bb >>= 4; }
        _alg_dispatch_blmerge(a0, a1, a2, a3, (ba >> 1) & 0x07, pa, reg_base, alg0_base, alg6_base);
        _alg_dispatch_blmerge(b0, b1, b2, b3, (bb >> 1) & 0x07, pb, reg_base, alg0_base, alg6_base);
    }
}

__forceinline void tm_avx2_r256_map_8::_run_map_entry_x4_blmerge(
    __m256i& a0, __m256i& a1, __m256i& a2, __m256i& a3,
    __m256i& b0, __m256i& b1, __m256i& b2, __m256i& b3,
    __m256i& c0, __m256i& c1, __m256i& c2, __m256i& c3,
    __m256i& d0, __m256i& d1, __m256i& d2, __m256i& d3, int map_idx)
{
    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    uint16 nibble_selector = t.nibble_selectors[map_idx];
    uint16 pa = 2047, pb = 2047, pc = 2047, pd = 2047;
#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (int i = 0; i < 16; i++)
    {
        uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
        nibble_selector <<= 1;
        const __m128i al = _mm256_castsi256_si128(a0);
        const __m128i bl = _mm256_castsi256_si128(b0);
        const __m128i cl = _mm256_castsi256_si128(c0);
        const __m128i dl = _mm256_castsi256_si128(d0);
        uint8 ba = 0, bb = 0, bc = 0, bd = 0;
        TM_MAP_AVX2_EXTRACT(ba, al, i);
        TM_MAP_AVX2_EXTRACT(bb, bl, i);
        TM_MAP_AVX2_EXTRACT(bc, cl, i);
        TM_MAP_AVX2_EXTRACT(bd, dl, i);
        if (nibble == 1) { ba >>= 4; bb >>= 4; bc >>= 4; bd >>= 4; }
        _alg_dispatch_blmerge(a0, a1, a2, a3, (ba >> 1) & 0x07, pa, reg_base, alg0_base, alg6_base);
        _alg_dispatch_blmerge(b0, b1, b2, b3, (bb >> 1) & 0x07, pb, reg_base, alg0_base, alg6_base);
        _alg_dispatch_blmerge(c0, c1, c2, c3, (bc >> 1) & 0x07, pc, reg_base, alg0_base, alg6_base);
        _alg_dispatch_blmerge(d0, d1, d2, d3, (bd >> 1) & 0x07, pd, reg_base, alg0_base, alg6_base);
    }
}

__forceinline void tm_avx2_r256_map_8::_sweep_x2_blmerge(const uint8* in0, const uint8* in1, uint8* out0, uint8* out1,
    std::size_t lb, std::size_t le)
{
    __m256i a0 = _mm256_loadu_si256((const __m256i*)(in0));
    __m256i a1 = _mm256_loadu_si256((const __m256i*)(in0 + 32));
    __m256i a2 = _mm256_loadu_si256((const __m256i*)(in0 + 64));
    __m256i a3 = _mm256_loadu_si256((const __m256i*)(in0 + 96));
    __m256i b0 = _mm256_loadu_si256((const __m256i*)(in1));
    __m256i b1 = _mm256_loadu_si256((const __m256i*)(in1 + 32));
    __m256i b2 = _mm256_loadu_si256((const __m256i*)(in1 + 64));
    __m256i b3 = _mm256_loadu_si256((const __m256i*)(in1 + 96));
    for (std::size_t map_idx = lb; map_idx < le; map_idx++)
        _run_map_entry_x2_blmerge(a0, a1, a2, a3, b0, b1, b2, b3, static_cast<int>(map_idx));
    _mm256_storeu_si256((__m256i*)(out0),      a0);
    _mm256_storeu_si256((__m256i*)(out0 + 32), a1);
    _mm256_storeu_si256((__m256i*)(out0 + 64), a2);
    _mm256_storeu_si256((__m256i*)(out0 + 96), a3);
    _mm256_storeu_si256((__m256i*)(out1),      b0);
    _mm256_storeu_si256((__m256i*)(out1 + 32), b1);
    _mm256_storeu_si256((__m256i*)(out1 + 64), b2);
    _mm256_storeu_si256((__m256i*)(out1 + 96), b3);
}

__forceinline void tm_avx2_r256_map_8::_sweep_x4_blmerge(
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3, std::size_t lb, std::size_t le)
{
    __m256i a0 = _mm256_loadu_si256((const __m256i*)(in0));
    __m256i a1 = _mm256_loadu_si256((const __m256i*)(in0 + 32));
    __m256i a2 = _mm256_loadu_si256((const __m256i*)(in0 + 64));
    __m256i a3 = _mm256_loadu_si256((const __m256i*)(in0 + 96));
    __m256i b0 = _mm256_loadu_si256((const __m256i*)(in1));
    __m256i b1 = _mm256_loadu_si256((const __m256i*)(in1 + 32));
    __m256i b2 = _mm256_loadu_si256((const __m256i*)(in1 + 64));
    __m256i b3 = _mm256_loadu_si256((const __m256i*)(in1 + 96));
    __m256i c0 = _mm256_loadu_si256((const __m256i*)(in2));
    __m256i c1 = _mm256_loadu_si256((const __m256i*)(in2 + 32));
    __m256i c2 = _mm256_loadu_si256((const __m256i*)(in2 + 64));
    __m256i c3 = _mm256_loadu_si256((const __m256i*)(in2 + 96));
    __m256i d0 = _mm256_loadu_si256((const __m256i*)(in3));
    __m256i d1 = _mm256_loadu_si256((const __m256i*)(in3 + 32));
    __m256i d2 = _mm256_loadu_si256((const __m256i*)(in3 + 64));
    __m256i d3 = _mm256_loadu_si256((const __m256i*)(in3 + 96));
    for (std::size_t map_idx = lb; map_idx < le; map_idx++)
        _run_map_entry_x4_blmerge(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3, static_cast<int>(map_idx));
    _mm256_storeu_si256((__m256i*)(out0),      a0); _mm256_storeu_si256((__m256i*)(out0 + 32), a1);
    _mm256_storeu_si256((__m256i*)(out0 + 64), a2); _mm256_storeu_si256((__m256i*)(out0 + 96), a3);
    _mm256_storeu_si256((__m256i*)(out1),      b0); _mm256_storeu_si256((__m256i*)(out1 + 32), b1);
    _mm256_storeu_si256((__m256i*)(out1 + 64), b2); _mm256_storeu_si256((__m256i*)(out1 + 96), b3);
    _mm256_storeu_si256((__m256i*)(out2),      c0); _mm256_storeu_si256((__m256i*)(out2 + 32), c1);
    _mm256_storeu_si256((__m256i*)(out2 + 64), c2); _mm256_storeu_si256((__m256i*)(out2 + 96), c3);
    _mm256_storeu_si256((__m256i*)(out3),      d0); _mm256_storeu_si256((__m256i*)(out3 + 32), d1);
    _mm256_storeu_si256((__m256i*)(out3 + 64), d2); _mm256_storeu_si256((__m256i*)(out3 + 96), d3);
}

// Branchless x8 (the production deep-drain dispatch). blarith/blall map to the same
// blmerge path on AVX2 (the {0,1,3,4,6,7}-folded form; no separate compute-all variant).
void tm_avx2_r256_map_8::run_maps_range_x8_blmerge(const key_schedule& s, std::size_t b, std::size_t e,
    const uint8* i0, const uint8* i1, const uint8* i2, const uint8* i3,
    const uint8* i4, const uint8* i5, const uint8* i6, const uint8* i7,
    uint8* o0, uint8* o1, uint8* o2, uint8* o3, uint8* o4, uint8* o5, uint8* o6, uint8* o7)
{
    std::size_t lb, le;
    _ileave_resolve(s, b, e, lb, le);
#if AVX2_RACEWAY_W == 4
    _sweep_x4_blmerge(i0, i1, i2, i3, o0, o1, o2, o3, lb, le);
    _sweep_x4_blmerge(i4, i5, i6, i7, o4, o5, o6, o7, lb, le);
#else
    _sweep_x2_blmerge(i0, i1, o0, o1, lb, le);
    _sweep_x2_blmerge(i2, i3, o2, o3, lb, le);
    _sweep_x2_blmerge(i4, i5, o4, o5, lb, le);
    _sweep_x2_blmerge(i6, i7, o6, o7, lb, le);
#endif
}

void tm_avx2_r256_map_8::run_maps_range_x8_blarith(const key_schedule& s, std::size_t b, std::size_t e,
    const uint8* i0, const uint8* i1, const uint8* i2, const uint8* i3,
    const uint8* i4, const uint8* i5, const uint8* i6, const uint8* i7,
    uint8* o0, uint8* o1, uint8* o2, uint8* o3, uint8* o4, uint8* o5, uint8* o6, uint8* o7)
{ run_maps_range_x8_blmerge(s, b, e, i0, i1, i2, i3, i4, i5, i6, i7, o0, o1, o2, o3, o4, o5, o6, o7); }

void tm_avx2_r256_map_8::run_maps_range_x8_blall(const key_schedule& s, std::size_t b, std::size_t e,
    const uint8* i0, const uint8* i1, const uint8* i2, const uint8* i3,
    const uint8* i4, const uint8* i5, const uint8* i6, const uint8* i7,
    uint8* o0, uint8* o1, uint8* o2, uint8* o3, uint8* o4, uint8* o5, uint8* o6, uint8* o7)
{ run_maps_range_x8_blmerge(s, b, e, i0, i1, i2, i3, i4, i5, i6, i7, o0, o1, o2, o3, o4, o5, o6, o7); }

// MAP1 (entry 0) + per-lane shed-proxy score. Processes two lanes at a time,
// capturing each lane's 16 op ids for the routing producer signal.
void tm_avx2_r256_map_8::run_map1_range_x8_scores(const key_schedule& schedule_entries,
    const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
    const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
    uint8* out0, uint8* out1, uint8* out2, uint8* out3,
    uint8* out4, uint8* out5, uint8* out6, uint8* out7,
    float scores[8])
{
    std::size_t lb, le;
    _ileave_resolve(schedule_entries, 0, 1, lb, le);
    const int map_idx = static_cast<int>(lb);

    const auto& t = _t();
    const uint8* reg_base  = t.reg_table  + static_cast<size_t>(map_idx) * 2048;
    const uint8* alg0_base = t.alg0_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;
    const uint8* alg6_base = t.alg6_table + static_cast<size_t>(map_idx) * map_tables_shared::Tables::ALG06_BYTES_PER_ENTRY;

    const uint8* ins[8]  = { in0, in1, in2, in3, in4, in5, in6, in7 };
    uint8* outs[8]       = { out0, out1, out2, out3, out4, out5, out6, out7 };

    // Two lanes at a time, capturing ops; MAP1 is a single map so this scores it.
    for (int pair = 0; pair < 8; pair += 2)
    {
        __m256i a0 = _mm256_loadu_si256((const __m256i*)(ins[pair]));
        __m256i a1 = _mm256_loadu_si256((const __m256i*)(ins[pair] + 32));
        __m256i a2 = _mm256_loadu_si256((const __m256i*)(ins[pair] + 64));
        __m256i a3 = _mm256_loadu_si256((const __m256i*)(ins[pair] + 96));
        __m256i b0 = _mm256_loadu_si256((const __m256i*)(ins[pair + 1]));
        __m256i b1 = _mm256_loadu_si256((const __m256i*)(ins[pair + 1] + 32));
        __m256i b2 = _mm256_loadu_si256((const __m256i*)(ins[pair + 1] + 64));
        __m256i b3 = _mm256_loadu_si256((const __m256i*)(ins[pair + 1] + 96));

        uint16 nibble_selector = t.nibble_selectors[map_idx];
        uint16 pa = 2047, pb = 2047;
        int opa[16], opb[16];
        for (int i = 0; i < 16; i++)
        {
            uint8 nibble = static_cast<uint8>((nibble_selector >> 15) & 0x01);
            nibble_selector <<= 1;
            const __m128i al = _mm256_castsi256_si128(a0);
            const __m128i bl = _mm256_castsi256_si128(b0);
            uint8 ba = 0, bb = 0;
            TM_MAP_AVX2_EXTRACT(ba, al, i);
            TM_MAP_AVX2_EXTRACT(bb, bl, i);
            if (nibble == 1) { ba >>= 4; bb >>= 4; }
            opa[i] = (ba >> 1) & 0x07;
            opb[i] = (bb >> 1) & 0x07;
            _run_alg(a0, a1, a2, a3, opa[i], &pa, reg_base, alg0_base, alg6_base);
            _run_alg(b0, b1, b2, b3, opb[i], &pb, reg_base, alg0_base, alg6_base);
        }
        scores[pair]     = tm_routing::shed_proxy_map1(opa);
        scores[pair + 1] = tm_routing::shed_proxy_map1(opb);

        _mm256_storeu_si256((__m256i*)(outs[pair]),      a0);
        _mm256_storeu_si256((__m256i*)(outs[pair] + 32), a1);
        _mm256_storeu_si256((__m256i*)(outs[pair] + 64), a2);
        _mm256_storeu_si256((__m256i*)(outs[pair] + 96), a3);
        _mm256_storeu_si256((__m256i*)(outs[pair + 1]),      b0);
        _mm256_storeu_si256((__m256i*)(outs[pair + 1] + 32), b1);
        _mm256_storeu_si256((__m256i*)(outs[pair + 1] + 64), b2);
        _mm256_storeu_si256((__m256i*)(outs[pair + 1] + 96), b3);
    }
}

bool tm_avx2_r256_map_8::screen_state_raw(const uint8* src, uint8& flags_out)
{
    __m256i w0 = _mm256_loadu_si256((const __m256i*)(src));
    __m256i w1 = _mm256_loadu_si256((const __m256i*)(src + 32));
    __m256i w2 = _mm256_loadu_si256((const __m256i*)(src + 64));
    __m256i w3 = _mm256_loadu_si256((const __m256i*)(src + 96));

    __m256i a0 = w0, a1 = w1, a2 = w2, a3 = w3;
    _decrypt_carnival_world(a0, a1, a2, a3);
    if (check_carnival_world_checksum(a0, a1, a2, a3))
    {
        _store_to_mem(a0, a1, a2, a3);
        flags_out = check_machine_code(working_code_data, CARNIVAL_WORLD);
        return true;
    }

    a0 = w0; a1 = w1; a2 = w2; a3 = w3;
    _decrypt_other_world(a0, a1, a2, a3);
    if (check_other_world_checksum(a0, a1, a2, a3))
    {
        _store_to_mem(a0, a1, a2, a3);
        flags_out = check_machine_code(working_code_data, OTHER_WORLD);
        return true;
    }

    flags_out = 0;
    return false;
}

#undef TM_MAP_AVX2_EXTRACT

bool tm_avx2_r256_map_8::initialized = false;
