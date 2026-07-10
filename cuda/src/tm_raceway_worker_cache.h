#pragma once
// Persistent-worker device-allocation cache for the raceway engine.
//
// The persistent worker processes many keys in one process; re-allocating the
// per-key device buffers (offset stream, cap tables, wave scratch: counters,
// stats, alive/live/compact/state/flags/drop-map/occupancy/hit-packed) every
// key would dominate host latency. This struct holds those allocations across
// keys and only re-allocates when a key's shape is incompatible — matches()
// guards on the offset byte count, cap geometry (count/bits/ways), wave cap, and
// the capture-flag / flat-parity / drop-hist toggles; a mismatch triggers
// release() + realloc. release() frees everything and resets to a clean state
// (also used at worker/batch teardown).
//
// Extracted from main.cpp (2026-07). Pure CUDA-driver-API bookkeeping — no
// dependency on the worker loop's locals.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda.h>

namespace tm_raceway {

struct RacewayWorkerDeviceCache
{
	size_t offset_bytes = 0u;
	uint32_t cap_count = 0u;
	uint32_t cap_bits = 0u;
	uint32_t cap_ways = 0u;
	uint32_t wave_cap = 0u;
	bool wave_flags_enabled = false;
	bool flat_parity = false;
	bool drop_hist = false;
	CUdeviceptr d_direct_ostream = 0;
	std::vector<CUdeviceptr> cap_tables_h;
	CUdeviceptr d_cap_tables = 0;
	CUdeviceptr d_cap_bits = 0;
	CUdeviceptr d_cap_ways = 0;
	uint64_t cap_slots_total = 0ull;
	CUdeviceptr wave_work_counter = 0;
	CUdeviceptr wave_stats = 0;
	CUdeviceptr wave_alive = 0;
	CUdeviceptr wave_live_a = 0;
	CUdeviceptr wave_live_b = 0;
	CUdeviceptr wave_compact_counter = 0;
	CUdeviceptr wave_flags = 0;
	CUdeviceptr wave_flat_flags = 0;
	CUdeviceptr wave_state = 0;
	CUdeviceptr wave_drop_map = 0;
	CUdeviceptr wave_occ_counter = 0;
	CUdeviceptr wave_hit_packed = 0;

	void release()
	{
		if (d_direct_ostream) cuMemFree(d_direct_ostream);
		for (CUdeviceptr p : cap_tables_h) if (p) cuMemFree(p);
		if (d_cap_tables) cuMemFree(d_cap_tables);
		if (d_cap_bits) cuMemFree(d_cap_bits);
		if (d_cap_ways) cuMemFree(d_cap_ways);
		if (wave_work_counter) cuMemFree(wave_work_counter);
		if (wave_stats) cuMemFree(wave_stats);
		if (wave_alive) cuMemFree(wave_alive);
		if (wave_live_a) cuMemFree(wave_live_a);
		if (wave_live_b) cuMemFree(wave_live_b);
		if (wave_compact_counter) cuMemFree(wave_compact_counter);
		if (wave_flags) cuMemFree(wave_flags);
		if (wave_flat_flags) cuMemFree(wave_flat_flags);
		if (wave_state) cuMemFree(wave_state);
		if (wave_drop_map) cuMemFree(wave_drop_map);
		if (wave_occ_counter) cuMemFree(wave_occ_counter);
		if (wave_hit_packed) cuMemFree(wave_hit_packed);
		*this = RacewayWorkerDeviceCache{};
	}

	bool matches(size_t offset_bytes_in, uint32_t cap_count_in, uint32_t cap_bits_in,
		uint32_t cap_ways_in, uint32_t wave_cap_in, bool wave_flags_in,
		bool flat_parity_in, bool drop_hist_in) const
	{
		return d_direct_ostream != 0
			&& offset_bytes == offset_bytes_in
			&& cap_count == cap_count_in
			&& cap_bits == cap_bits_in
			&& cap_ways == cap_ways_in
			&& wave_cap == wave_cap_in
			&& wave_flags_enabled == wave_flags_in
			&& flat_parity == flat_parity_in
			&& drop_hist == drop_hist_in;
	}
};

} // namespace tm_raceway
