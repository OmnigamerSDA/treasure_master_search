#ifndef TM_ROUTING_H
#define TM_ROUTING_H

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace tm_routing
{

constexpr std::size_t STATE_BYTES = 128;
constexpr std::uint32_t TOP_CERTIFIER_TIER_MASK = 0xEE000000u;
constexpr std::uint32_t BOTTOM_CERTIFIER_TIER_MASK = 0xCC000000u;

struct RoutedState
{
    std::array<std::uint8_t, STATE_BYTES> state;
    float unused_cum = 0.0f;
    float current_alg0 = 0.0f;
    std::uint16_t hash_checks = 0u;
    std::uint32_t multiplicity = 1u;
};

inline std::uint16_t increment_hash_checks(std::uint16_t v)
{
    return v == std::numeric_limits<std::uint16_t>::max()
        ? v
        : static_cast<std::uint16_t>(v + 1u);
}

inline RoutedState mark_hash_checked(RoutedState state, float current_alg0)
{
    state.current_alg0 = current_alg0;
    state.hash_checks = increment_hash_checks(state.hash_checks);
    return state;
}

inline void merge_hash_metadata_max(RoutedState& dst, const RoutedState& src)
{
    dst.unused_cum = std::max(dst.unused_cum, src.unused_cum);
    dst.current_alg0 = std::max(dst.current_alg0, src.current_alg0);
    dst.hash_checks = std::max(dst.hash_checks, src.hash_checks);
}

inline bool is_top_certifier_tier(std::uint32_t key)
{
    return (key & TOP_CERTIFIER_TIER_MASK) == 0u;
}

inline bool is_bottom_certifier_tier(std::uint32_t key)
{
    return (key & BOTTOM_CERTIFIER_TIER_MASK) == BOTTOM_CERTIFIER_TIER_MASK;
}

inline std::uint8_t smear_up(std::uint8_t m)
{
    if (m == 0u) return 0u;
    std::uint8_t r = 0u;
    for (int b = 0; b < 8; ++b)
    {
        if ((m & (1u << b)) == 0u) continue;
        for (int j = b; j < 8; ++j) r = static_cast<std::uint8_t>(r | (1u << j));
        break;
    }
    return r;
}

inline void apply_mask(int a, std::uint8_t* m)
{
    if (a == 0)
    {
        for (int i = 0; i < 128; ++i) m[i] = static_cast<std::uint8_t>(m[i] << 1);
    }
    else if (a == 6)
    {
        for (int i = 0; i < 128; ++i) m[i] = static_cast<std::uint8_t>(m[i] >> 1);
    }
    else if (a == 1 || a == 4)
    {
        for (int i = 0; i < 128; ++i) m[i] = smear_up(m[i]);
    }
    else if (a == 3 || a == 7)
    {
    }
    else if (a == 2)
    {
        std::uint8_t c = 0u;
        for (int i = 127; i >= 0; i -= 2)
        {
            const std::uint8_t nc = static_cast<std::uint8_t>(m[i - 1] & 0x01u);
            m[i - 1] = static_cast<std::uint8_t>((m[i - 1] >> 1) | (m[i] & 0x80u));
            m[i] = static_cast<std::uint8_t>((m[i] << 1) | (c & 0x01u));
            c = nc;
        }
    }
    else if (a == 5)
    {
        std::uint8_t c = 0u;
        for (int i = 127; i >= 0; i -= 2)
        {
            const std::uint8_t nc = static_cast<std::uint8_t>(m[i - 1] & 0x80u);
            m[i - 1] = static_cast<std::uint8_t>((m[i - 1] << 1) | (m[i] & 0x01u));
            m[i] = static_cast<std::uint8_t>((m[i] >> 1) | c);
            c = nc;
        }
    }
}

// MAP1 shed proxy: one 128-byte taint propagation; count alg0/alg6 drops at
// data-tainted bits. TAINT_NORM is exactly 64 for the current expand layout:
// columns 4-7 of each 8-byte group are data columns.
inline float shed_proxy_map1(const int* op)
{
    static constexpr float TAINT_NORM = 64.0f;
    std::uint8_t m[STATE_BYTES];
    for (int i = 0; i < 128; ++i)
        m[i] = ((i & 7) >= 4) ? 0xFFu : 0x00u;

    int cnt = 0;
    for (int s = 0; s < 16; ++s)
    {
        const int a = op[s];
        if (a == 0)
        {
            for (int i = 0; i < 128; ++i) if ((m[i] & 0x80u) != 0u) ++cnt;
        }
        else if (a == 6)
        {
            for (int i = 0; i < 128; ++i) if ((m[i] & 0x01u) != 0u) ++cnt;
        }
        apply_mask(a, m);
    }
    return static_cast<float>(cnt) / TAINT_NORM;
}

} // namespace tm_routing

#endif
