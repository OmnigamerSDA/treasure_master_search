#pragma once
// Raceway persistent-worker offset-stream staging + prefetch resources.
//
// Two concerns, bundled because they share the same parity-indexed lifetime:
//
//  1. Pinned staging (always, when overlap-setup is on): two page-locked host
//     buffers, ping-ponged by key parity, that the background prepare thread
//     fills so the per-key offset HtoD is a fast DMA (~0.4ms) instead of a
//     pageable copy (~1.6ms). The in-flight prepare writes the OTHER parity slot
//     than the one the main thread is DMA-ing, so there is no overwrite.
//
//  2. Prefetch (when active): a device double-buffer + a dedicated non-blocking
//     copy stream + two completion events. The prepare thread issues the next
//     key's offset HtoDAsync into dbuf[(idx+1)&1] on copy_stream during THIS
//     key's compute (the copy engine is independent of the SMs, so it is hidden)
//     and records ev_upload[slot]; the main thread only makes the null (compute)
//     stream wait on that event before the wave — always already signalled.
//     Non-blocking so it does not implicitly serialise against null-stream
//     compute; ordering is via the event only.
//
// Buffers are grown lazily to the (constant, per-tier) offset-stream size;
// destroy() drains the copy stream before freeing so an in-flight prefetch for a
// would-be next key (e.g. after a stop-file break) cannot be freed underneath.
// Extracted from main.cpp (2026-07).

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>

#include "tm_offset_stream.h"

namespace tm_raceway {

// Per-key blobs produced by the background prepare thread and consumed by the
// worker loop. Plain data; the prepare logic lives in main.cpp (it needs the
// key schedule + certifier).
struct RacewayWorkerPrepared
{
	uint32_t key = 0u;
	std::vector<uint8_t> sched_blob;
	tm_offset::OffsetStreamBlob osb;
	uint32_t shed_mask = 0u;
	void* pinned_ptr = nullptr;
	bool pinned_filled = false;
	double schedule_build_ms = 0.0;
	double precert_ms = 0.0;
	double offset_build_ms = 0.0;
	int upload_slot = -1; // >=0 when this key's offset stream was async-uploaded to dbuf[slot] on the copy stream (prefetch); the main thread only waits on ev_upload[slot]
};

struct RacewayOffsetStaging
{
	// Pinned host staging (overlap-setup fast-DMA source), ping-ponged by parity.
	void* pinned[2] = { nullptr, nullptr };
	size_t pinned_cap[2] = { 0u, 0u };
	// Device double-buffer + copy stream + events (prefetch only; created by init()).
	CUdeviceptr dbuf[2] = { 0, 0 };
	size_t dbuf_cap[2] = { 0u, 0u };
	CUstream copy_stream = nullptr;
	CUevent ev_upload[2] = { nullptr, nullptr };
	bool prefetch = false;

	// Create the prefetch copy stream + events when active. Pinned buffers are
	// allocated lazily by ensure_pinned regardless (they serve overlap-setup too).
	void init(bool prefetch_active)
	{
		prefetch = prefetch_active;
		if (!prefetch) return;
		check(cuStreamCreate(&copy_stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate(raceway_offset_copy)");
		check(cuEventCreate(&ev_upload[0], CU_EVENT_DISABLE_TIMING), "cuEventCreate(raceway_offset_up0)");
		check(cuEventCreate(&ev_upload[1], CU_EVENT_DISABLE_TIMING), "cuEventCreate(raceway_offset_up1)");
	}

	void ensure_pinned(int slot, size_t need)
	{
		if (need == 0u || (pinned[slot] != nullptr && pinned_cap[slot] >= need)) return;
		if (pinned[slot] != nullptr) cuMemFreeHost(pinned[slot]);
		pinned[slot] = nullptr;
		check(cuMemAllocHost(&pinned[slot], need), "cuMemAllocHost(raceway_offset_pinned)");
		pinned_cap[slot] = need;
	}

	void ensure_dbuf(int slot, size_t need)
	{
		if (need == 0u || (dbuf[slot] != 0 && dbuf_cap[slot] >= need)) return;
		if (dbuf[slot] != 0) cuMemFree(dbuf[slot]);
		dbuf[slot] = 0;
		check(cuMemAlloc(&dbuf[slot], need), "cuMemAlloc(raceway_offset_dbuf)");
		dbuf_cap[slot] = need;
	}

	void destroy()
	{
		// Drain the copy stream first: a prefetch upload for a would-be next key may
		// still be in flight (e.g. after a stop-file break that only waited on the
		// prepare future, not the async DMA it issued).
		if (copy_stream != nullptr) cuStreamSynchronize(copy_stream);
		if (dbuf[0] != 0) cuMemFree(dbuf[0]);
		if (dbuf[1] != 0) cuMemFree(dbuf[1]);
		if (ev_upload[0] != nullptr) cuEventDestroy(ev_upload[0]);
		if (ev_upload[1] != nullptr) cuEventDestroy(ev_upload[1]);
		if (copy_stream != nullptr) cuStreamDestroy(copy_stream);
		if (pinned[0] != nullptr) cuMemFreeHost(pinned[0]);
		if (pinned[1] != nullptr) cuMemFreeHost(pinned[1]);
		*this = RacewayOffsetStaging{};
	}

private:
	static void check(CUresult r, const char* what)
	{
		if (r != CUDA_SUCCESS)
		{
			const char* name = nullptr;
			cuGetErrorName(r, &name);
			throw std::runtime_error(std::string(what) + ": " + (name ? name : "CUDA error"));
		}
	}
};

} // namespace tm_raceway
