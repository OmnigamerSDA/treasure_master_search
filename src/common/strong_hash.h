// strong_hash.h — strong avalanched fingerprint for memcmp-FREE dedup (W4B hash-set).
//
// WHY THIS EXISTS (see docs/cpu_dedup_forward_interleave_findings_20260601.md §45):
//   The production FlatTable::fingerprint (4-stream FNV + murmur finalizer) has poor
//   avalanche on the structured, low-dimensional post-MAP1 states — it behaves like
//   ~35 EFFECTIVE bits, colliding ~16x the ideal 64-bit rate (327 collisions / 5.3M
//   distinct at W256M). That is HARMLESS in production because FlatTable backstops
//   every fp-match with a 128-byte memcmp. But a memcmp-FREE hash-set treats an
//   fp-match as proof of equality, so a weak fp DROPS unlike states (false positive =
//   permanent coverage loss the final-union cannot recover).
//
//   The fix is NOT more width or a secondary salt (FNV-family variants stay correlated
//   on the same degenerate pairs); it is a single rotation per round. hstrong below
//   (xor -> rotl -> multiply -> xorshift -> multiply) gives 0 collisions at 5.3M
//   distinct. That is sufficient for smaller single-host frontiers, but full-W4B MAP1
//   frontiers can exceed 3B states, where 64-bit birthday collisions are no longer
//   negligible. Use at least 96 bits, or strong128, for billion-state hit-finding.
//   strong128 (two independent seeds) is the "secondary salt" idea done right.
//
// USE: memcmp-free hash-set dedup over 128-byte states (16 x uint64).
//   - strong64(state)        -> 1 word, performance-measurement / smaller-frontier use.
//   - strong128(state,&a,&b) -> 2 words, bulletproof. (a,b) is never (0,0).
#pragma once
#include <cstdint>

namespace w4b {

static inline std::uint64_t rotl64(std::uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

// One strong avalanched stream over the 16 uint64 words of a 128-byte state.
static inline std::uint64_t hstrong(const std::uint64_t* w, std::uint64_t seed) {
    std::uint64_t h = seed;
    for (int i = 0; i < 16; i++) {
        h ^= w[i];
        h = rotl64(h, 31);
        h *= 0x9E3779B97F4A7C15ull;
        h ^= h >> 29;
        h *= 0xBF58476D1CE4E5B9ull;
    }
    h ^= h >> 32;
    return h;
}

// 64-bit fingerprint. Never returns 0 (so callers can use 0 as an empty-slot sentinel).
static inline std::uint64_t strong64(const std::uint8_t* state) {
    const std::uint64_t* w = reinterpret_cast<const std::uint64_t*>(state);
    std::uint64_t h = hstrong(w, 0xcbf29ce484222325ull);
    return h ? h : 1;
}

// 128-bit fingerprint via two independent seeds. (h0,h1) is never (0,0).
static inline void strong128(const std::uint8_t* state, std::uint64_t& h0, std::uint64_t& h1) {
    const std::uint64_t* w = reinterpret_cast<const std::uint64_t*>(state);
    h0 = hstrong(w, 0xcbf29ce484222325ull);
    h1 = hstrong(w, 0x9E3779B97F4A7C15ull);
    if (!h0 && !h1) h0 = 1;
}

} // namespace w4b
