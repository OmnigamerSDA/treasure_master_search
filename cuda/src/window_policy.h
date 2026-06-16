#ifndef TM_WINDOW_POLICY_H
#define TM_WINDOW_POLICY_H

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tm_window_policy
{

enum class Policy
{
    LowBits,
    HighBits,
    Backfill,
    Squeeze,
    Backfavor,
    OmitByte0,
    OmitByte1,
    OmitByte2,
    OmitByte3,
    OmitByte4,
    OmitByte5,
    OmitByte6,
    OmitByte7,
    Auto
};

inline Policy parse_policy(const std::string& policy)
{
    if (policy == "lowbits") return Policy::LowBits;
    if (policy == "highbits") return Policy::HighBits;
    if (policy == "backfill") return Policy::Backfill;
    if (policy == "squeeze") return Policy::Squeeze;
    if (policy == "backfavor") return Policy::Backfavor;
    if (policy == "auto") return Policy::Auto;
    if (policy.rfind("omitbyte", 0) == 0)
    {
        const std::string suffix = policy.substr(8);
        if (suffix.size() != 1 || suffix[0] < '0' || suffix[0] > '7')
            throw std::runtime_error("omitbyte policy must be omitbyte0..omitbyte7");
        return static_cast<Policy>(static_cast<int>(Policy::OmitByte0) + (suffix[0] - '0'));
    }
    throw std::runtime_error("unknown window mask policy: " + policy);
}

inline const char* policy_name(Policy policy)
{
    switch (policy)
    {
        case Policy::LowBits: return "lowbits";
        case Policy::HighBits: return "highbits";
        case Policy::Backfill: return "backfill";
        case Policy::Squeeze: return "squeeze";
        case Policy::Backfavor: return "backfavor";
        case Policy::OmitByte0: return "omitbyte0";
        case Policy::OmitByte1: return "omitbyte1";
        case Policy::OmitByte2: return "omitbyte2";
        case Policy::OmitByte3: return "omitbyte3";
        case Policy::OmitByte4: return "omitbyte4";
        case Policy::OmitByte5: return "omitbyte5";
        case Policy::OmitByte6: return "omitbyte6";
        case Policy::OmitByte7: return "omitbyte7";
        case Policy::Auto: return "auto";
    }
    return "unknown";
}

inline std::uint32_t lowbits_mask(std::uint32_t selected_bit_count)
{
    if (selected_bit_count == 0u || selected_bit_count > 32u)
        throw std::runtime_error("selected-bit count must be in 1..32");
    return selected_bit_count == 32u ? 0xffffffffu : ((1u << selected_bit_count) - 1u);
}

inline std::uint32_t order_mask(const std::uint32_t (&order)[8], std::uint32_t selected_bit_count)
{
    if (selected_bit_count == 0u || selected_bit_count > 32u)
        throw std::runtime_error("selected-bit count must be in 1..32");
    std::uint32_t mask = 0u;
    std::uint32_t chosen = 0u;
    for (std::uint32_t order_idx = 0u; order_idx < 8u && chosen < selected_bit_count; ++order_idx)
    {
        const std::uint32_t bit_in_byte = order[order_idx];
        for (std::uint32_t byte = 0u; byte < 4u && chosen < selected_bit_count; ++byte)
        {
            mask |= 1u << (byte * 8u + bit_in_byte);
            ++chosen;
        }
    }
    return mask;
}

inline std::uint32_t highbits_mask(std::uint32_t selected_bit_count)
{
    if (selected_bit_count == 0u || selected_bit_count > 32u)
        throw std::runtime_error("selected-bit count must be in 1..32");
    if (selected_bit_count == 32u) return 0xffffffffu;
    std::uint32_t mask = 0u;
    for (std::uint32_t i = 0u; i < selected_bit_count; ++i)
        mask |= 1u << (31u - i);
    return mask;
}

inline std::uint32_t omit_byte_mask(std::uint32_t selected_bit_count, std::uint32_t omitted_bit_in_byte)
{
    if (selected_bit_count != 28u)
        throw std::runtime_error("omitbyteN policies currently require 28 selected bits");
    if (omitted_bit_in_byte > 7u)
        throw std::runtime_error("omitbyteN bit index must be 0..7");
    std::uint32_t omitted = 0u;
    for (std::uint32_t byte = 0u; byte < 4u; ++byte)
        omitted |= 1u << (byte * 8u + omitted_bit_in_byte);
    return ~omitted;
}

inline Policy select_auto_policy(std::uint32_t selected_bit_count, std::uint32_t map_depth)
{
    (void)selected_bit_count;
    (void)map_depth;
    // Without a key-class signal, squeeze is the robust default. Backfill pays
    // mainly for known high-R/closer keys; do not infer that from width alone.
    return Policy::Squeeze;
}

inline Policy select_auto_policy(std::uint32_t selected_bit_count, std::uint32_t map_depth, bool high_r_or_closer)
{
    (void)map_depth;
    if (high_r_or_closer && selected_bit_count <= 24u) return Policy::Backfill;
    return Policy::Squeeze;
}

inline Policy resolve_policy(Policy policy, std::uint32_t selected_bit_count, std::uint32_t map_depth)
{
    return policy == Policy::Auto ? select_auto_policy(selected_bit_count, map_depth) : policy;
}

inline std::uint32_t make_bit_mask(Policy policy, std::uint32_t selected_bit_count,
                                   std::uint32_t map_depth = 27u)
{
    const Policy resolved = resolve_policy(policy, selected_bit_count, map_depth);
    static const std::uint32_t backfill_order[8] = {7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u};
    static const std::uint32_t squeeze_order[8] = {7u, 0u, 6u, 1u, 5u, 2u, 4u, 3u};
    static const std::uint32_t backfavor_order[8] = {7u, 6u, 0u, 5u, 4u, 1u, 3u, 2u};

    switch (resolved)
    {
        case Policy::LowBits: return lowbits_mask(selected_bit_count);
        case Policy::HighBits: return highbits_mask(selected_bit_count);
        case Policy::Backfill: return order_mask(backfill_order, selected_bit_count);
        case Policy::Squeeze: return order_mask(squeeze_order, selected_bit_count);
        case Policy::Backfavor: return order_mask(backfavor_order, selected_bit_count);
        case Policy::OmitByte0: return omit_byte_mask(selected_bit_count, 0u);
        case Policy::OmitByte1: return omit_byte_mask(selected_bit_count, 1u);
        case Policy::OmitByte2: return omit_byte_mask(selected_bit_count, 2u);
        case Policy::OmitByte3: return omit_byte_mask(selected_bit_count, 3u);
        case Policy::OmitByte4: return omit_byte_mask(selected_bit_count, 4u);
        case Policy::OmitByte5: return omit_byte_mask(selected_bit_count, 5u);
        case Policy::OmitByte6: return omit_byte_mask(selected_bit_count, 6u);
        case Policy::OmitByte7: return omit_byte_mask(selected_bit_count, 7u);
        case Policy::Auto: break;
    }
    return lowbits_mask(selected_bit_count);
}

inline std::uint32_t make_bit_mask(const std::string& policy, std::uint32_t selected_bit_count,
                                   std::uint32_t map_depth = 27u)
{
    return make_bit_mask(parse_policy(policy), selected_bit_count, map_depth);
}

inline std::uint32_t deposit_bits32(std::uint32_t bits, std::uint32_t mask)
{
    std::uint32_t out = 0u;
    for (std::uint32_t bit = 1u; mask != 0u; bit <<= 1u)
    {
        const std::uint32_t lsb = mask & (0u - mask);
        if ((bits & bit) != 0u) out |= lsb;
        mask ^= lsb;
    }
    return out;
}

inline std::string bit_mask_string(std::uint32_t mask)
{
    std::ostringstream out;
    bool first = true;
    for (std::uint32_t b = 0u; b < 32u; ++b)
    {
        if ((mask & (1u << b)) == 0u) continue;
        if (!first) out << ",";
        out << b;
        first = false;
    }
    return out.str();
}

} // namespace tm_window_policy

#endif
