#pragma once
// Host-side builders for the per-key raceway OFFSET STREAM.
//
// The offset stream is the ~20 MB per-key blob the CUDA cap-span/continue kernels
// consume directly: for every scheduled map, a 2048-step walk of the RNG table
// (starting from that map's seed) gathers the pre-computed map-eval vectors —
// three 128-byte streams (regular / alg0-extracted / alg6-extracted) plus two
// 32-bit carry streams (alg2 / alg5). It is a pure function of the key's
// schedule_blob and the read-only RNG tables (rng_obj.h); it holds no device
// state. Extracted from main.cpp (2026-07) so the layout lives in one small,
// self-documenting module — the raceway worker's offset-prefetch pipeline
// double-buffers the HtoD upload of exactly this blob.
//
// Layout of OffsetStreamBlob::data (offset_count = entry_count * 2048):
//   [regular 128*offset_count][alg0 128*offset_count][alg6 128*offset_count]
//   [alg2 4*offset_count][alg5 4*offset_count]
// stream_bytes = 128*offset_count, carry_bytes = 4*offset_count.

#include <cstdint>
#include <cstring>
#include <vector>

#include "rng_obj.h"

namespace tm_offset {

struct OffsetStreamBlob
{
	std::vector<uint8_t> data;
	std::size_t stream_bytes = 0;
	std::size_t carry_bytes = 0;
};

// Fill `blob` in place (resizes only when the byte count changes, so a reused blob
// avoids reallocation across keys — the persistent-worker hot path).
inline void build_offset_stream_blob_into(const std::vector<uint8_t>& schedule_blob, OffsetStreamBlob& blob)
{
	const size_t entry_count = schedule_blob.size() / 4;
	const size_t offset_count = entry_count * 2048ull;
	const size_t stream_bytes = offset_count * 128ull;
	const size_t carry_bytes = offset_count * sizeof(uint32_t);
	blob.stream_bytes = stream_bytes;
	blob.carry_bytes = carry_bytes;
	const size_t data_bytes = stream_bytes * 3ull + carry_bytes * 2ull;
	if (blob.data.size() != data_bytes)
		blob.data.resize(data_bytes);

	uint8_t* regular_stream = blob.data.data();
	uint8_t* alg0_stream = regular_stream + stream_bytes;
	uint8_t* alg6_stream = alg0_stream + stream_bytes;
	uint32_t* alg2_stream = reinterpret_cast<uint32_t*>(alg6_stream + stream_bytes);
	uint32_t* alg5_stream = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(alg2_stream) + carry_bytes);

	for (size_t m = 0; m < entry_count; ++m)
	{
		uint16_t seed = (static_cast<uint16_t>(schedule_blob[m * 4 + 0]) << 8)
		              |  static_cast<uint16_t>(schedule_blob[m * 4 + 1]);
		for (size_t pos = 0; pos < 2048ull; ++pos)
		{
			const size_t stream_offset = (m * 2048ull + pos) * 128ull;
			std::memcpy(regular_stream + stream_offset, RNG::regular_rng_values_8 + static_cast<size_t>(seed) * 128ull, 128ull);
			std::memcpy(alg0_stream + stream_offset, RNG::alg0_values_8 + static_cast<size_t>(seed) * 128ull, 128ull);
			std::memcpy(alg6_stream + stream_offset, RNG::alg6_values_8 + static_cast<size_t>(seed) * 128ull, 128ull);
			const size_t carry_offset = m * 2048ull + pos;
			alg2_stream[carry_offset] = RNG::alg2_values_32_8[seed];
			alg5_stream[carry_offset] = RNG::alg5_values_32_8[seed];
			seed = RNG::rng_table[seed];
		}
	}
}

inline OffsetStreamBlob build_offset_stream_blob(const std::vector<uint8_t>& schedule_blob)
{
	OffsetStreamBlob blob;
	build_offset_stream_blob_into(schedule_blob, blob);
	return blob;
}

// Per-map-contiguous ("interleaved") layout: each map's five streams are packed
// together in a fixed schedule_bytes stride, rather than all-regular / all-alg0 / …
// globally. Used by the offset-stream consumers that walk one map at a time.
inline std::vector<uint8_t> build_offset_stream_blob_interleaved(const std::vector<uint8_t>& schedule_blob)
{
	const size_t entry_count = schedule_blob.size() / 4;
	constexpr size_t schedule_stream_bytes = 2048ull * 128ull;
	constexpr size_t schedule_carry_bytes = 2048ull * sizeof(uint32_t);
	constexpr size_t schedule_bytes = schedule_stream_bytes * 3ull + schedule_carry_bytes * 2ull;
	std::vector<uint8_t> data(entry_count * schedule_bytes);

	for (size_t m = 0; m < entry_count; ++m)
	{
		uint8_t* schedule_base = data.data() + m * schedule_bytes;
		uint8_t* regular_stream = schedule_base;
		uint8_t* alg0_stream = regular_stream + schedule_stream_bytes;
		uint8_t* alg6_stream = alg0_stream + schedule_stream_bytes;
		uint32_t* alg2_stream = reinterpret_cast<uint32_t*>(alg6_stream + schedule_stream_bytes);
		uint32_t* alg5_stream = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(alg2_stream) + schedule_carry_bytes);

		uint16_t seed = (static_cast<uint16_t>(schedule_blob[m * 4 + 0]) << 8)
		              |  static_cast<uint16_t>(schedule_blob[m * 4 + 1]);
		for (size_t pos = 0; pos < 2048ull; ++pos)
		{
			const size_t stream_offset = pos * 128ull;
			std::memcpy(regular_stream + stream_offset, RNG::regular_rng_values_8 + static_cast<size_t>(seed) * 128ull, 128ull);
			std::memcpy(alg0_stream + stream_offset, RNG::alg0_values_8 + static_cast<size_t>(seed) * 128ull, 128ull);
			std::memcpy(alg6_stream + stream_offset, RNG::alg6_values_8 + static_cast<size_t>(seed) * 128ull, 128ull);
			alg2_stream[pos] = RNG::alg2_values_32_8[seed];
			alg5_stream[pos] = RNG::alg5_values_32_8[seed];
			seed = RNG::rng_table[seed];
		}
	}
	return data;
}

} // namespace tm_offset
