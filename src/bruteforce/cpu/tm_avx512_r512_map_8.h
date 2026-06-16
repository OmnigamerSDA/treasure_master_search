#ifndef TM_AVX512_R512_MAP_8_H
#define TM_AVX512_R512_MAP_8_H
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>

#include <cstdint>
#include <cstddef>
#include <vector>

#include "data_sizes.h"
#include "alignment2.h"
#include "rng_obj.h"
#include "tm_base.h"
#include "map_tables_shared.h"

// AVX-512 map-mode forward kernel. NATURAL (non-shuffled) byte layout in
// 2×__m512i state (wc0 = canonical bytes 0..63, wc1 = canonical bytes 64..127).
//
// Mirrors tm_avx2_r256_map_8 (the AVX2 NATURAL map kernel): per-schedule-entry
// 2048-byte RNG blocks built once at bind time, walked by a moving pointer.
// Reuses map_tables_shared::Tables and the rng.cpp generators unchanged. The
// only difference from the AVX2 kernel is the SIMD width (2 ZMM vs 4 YMM).

class tm_avx512_r512_map_8 : public TM_base
{
public:
	tm_avx512_r512_map_8(RNG* rng);
	tm_avx512_r512_map_8(RNG* rng, map_tables_shared::Tables* shared_tables);

	void load_data(uint8* new_data) override;
	void fetch_data(uint8* new_data) override;

	void expand(uint32 key, uint32 data) override;

	void run_alg(int algorithm_id, uint16* rng_seed, int iterations) override;

	void run_one_map(const key_schedule::key_schedule_entry& schedule_entry) override;
	void run_all_maps(const key_schedule& schedule_entries) override;
	void run_maps_range(const key_schedule& schedule_entries, std::size_t begin, std::size_t end);
	void bind_schedule(const key_schedule& schedule_entries) override;
	void bind_dedup_schedule(const key_schedule& schedule_entries);
	void bind_maps_range(const key_schedule& schedule_entries, std::size_t begin, std::size_t end);

	// PROTOTYPE: 4-way interleaved map-range kernel (mirrors
	// tm_avx512_r512s_8::run_maps_range_x4). Natural layout, so in/out are
	// canonical 128-byte states.
	void run_maps_range_x4(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3);

	// PROTOTYPE: 8-way / 12-way interleaved map-range kernels — does the natural
	// map's locality edge (3.4x fewer L1d misses, L2-resident 2KB blocks vs the
	// universal's scattered 8MB table) plus interleave scaling let it beat the
	// universal x12 at large windows / 16t? The natural-layout alg_2/alg_5 carry
	// is temp-heavier than the universal's, so x12 (24 state ZMM) may spill.
	void run_maps_range_x8(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		uint8* out4, uint8* out5, uint8* out6, uint8* out7);
	void run_map1_range_x8_scores(const key_schedule& schedule_entries,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		uint8* out4, uint8* out5, uint8* out6, uint8* out7,
		float scores[8]);
	// Deep-route (MAP2+ / K-boundary) variant: same maps as run_maps_range_x8, but
	// the first map emits a per-lane alg0 count into scores[8] as a free in-kernel
	// byproduct (replaces the scalar raw_alg0_count_for_entry pre-pass).
	void run_maps_range_x8_dscore(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		uint8* out4, uint8* out5, uint8* out6, uint8* out7,
		float scores[8]);
	// trajDens gate: capture the FINAL map's last-8-op key per lane (routing pre-sensor).
	void run_maps_range_x8_optail(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		uint8* out4, uint8* out5, uint8* out6, uint8* out7,
		std::uint32_t keys[8], int counts[8]);
	// Combined route signal: over [begin,end) accumulate a per-lane STICKY alg0 flag (set if ANY
	// map's live alg0 count >= alg0_tau) AND capture the FINAL map's op-tail key. One pass; sticky[k]
	// is 0/1, keys[k] is the 24-bit op-tail. Feeds the 1-pass online gate's combined (density||sticky).
	void run_maps_range_x8_route(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		uint8* out4, uint8* out5, uint8* out6, uint8* out7,
		std::uint32_t keys[8], int sticky[8], int alg0_tau);
	// EXPERIMENT: x8 with branchless-arith dispatch (validate the mispredict/pressure trade).
	void run_maps_range_x8_blarith(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		uint8* out4, uint8* out5, uint8* out6, uint8* out7);
	// EXPERIMENT v2: x8 merged branchless dispatch (3 candidates + blend-tree, lower pressure).
	void run_maps_range_x8_blmerge(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		uint8* out4, uint8* out5, uint8* out6, uint8* out7);
	// EXPERIMENT v3: x8 full all-8 branchless dispatch (butterflies folded; zero branches).
	void run_maps_range_x8_blall(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		uint8* out4, uint8* out5, uint8* out6, uint8* out7);
	// Single-lane (x1) runner for dispatch isolation: mode 0=branched,1=blarith,2=blmerge,3=blall.
	void run_maps_range_x1(const key_schedule& s, std::size_t begin, std::size_t end,
		const uint8* in, uint8* out, int mode);
	// x4 branchless-arith (lower-interleave point: fewer candidate ZMMs live -> avoid spill).
	void run_maps_range_x4_blarith(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3);
	// x4 deep-route variant (lower-interleave control for the register-pressure test).
	void run_maps_range_x4_dscore(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		float scores[4]);
	// x12 deep-route variant (higher-interleave point for the latency-recovery test).
	void run_maps_range_x12_dscore(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3, const uint8* in4, const uint8* in5,
		const uint8* in6, const uint8* in7, const uint8* in8, const uint8* in9, const uint8* in10, const uint8* in11,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3, uint8* out4, uint8* out5,
		uint8* out6, uint8* out7, uint8* out8, uint8* out9, uint8* out10, uint8* out11,
		float scores[12]);
	void run_maps_range_x10(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3, const uint8* in4,
		const uint8* in5, const uint8* in6, const uint8* in7, const uint8* in8, const uint8* in9,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3, uint8* out4,
		uint8* out5, uint8* out6, uint8* out7, uint8* out8, uint8* out9);
	void run_maps_range_x12(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3, const uint8* in4, const uint8* in5,
		const uint8* in6, const uint8* in7, const uint8* in8, const uint8* in9, const uint8* in10, const uint8* in11,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3, uint8* out4, uint8* out5,
		uint8* out6, uint8* out7, uint8* out8, uint8* out9, uint8* out10, uint8* out11);
	// x14 (28 state ZMM): probe whether the alg_2/5 butterfly rewrite freed enough
	// registers to clear the width the universal kernel spills at.
	void run_maps_range_x14(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3, const uint8* in4, const uint8* in5, const uint8* in6,
		const uint8* in7, const uint8* in8, const uint8* in9, const uint8* in10, const uint8* in11, const uint8* in12, const uint8* in13,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3, uint8* out4, uint8* out5, uint8* out6,
		uint8* out7, uint8* out8, uint8* out9, uint8* out10, uint8* out11, uint8* out12, uint8* out13);

	// W6: base-pointer variants — take CONTIGUOUS in[N*128]/out[N*128] bases (5 args incl this) and compute
	// per-lane pointers internally. The 24/28-arg x12/x14 forms spill ~18+ pointer args onto the noinline
	// AVX-512 drain frame, which corrupts them (SIGSEGV inside the kernel, even at -O2 / with -fno-builtin-
	// memset / LD_BIND_NOW — the "frame trigger" of docs/cpu_raceway_o2_o3_fragility). 5-reg-arg calls don't
	// spill, so they're frame-safe. Used by cpu_raceway's RACEWAY_NM_DEEP interleave sweep.
	void run_maps_range_x10_arr(const key_schedule&, std::size_t begin, std::size_t end, const uint8* in, uint8* out);
	void run_maps_range_x12_arr(const key_schedule&, std::size_t begin, std::size_t end, const uint8* in, uint8* out);
	void run_maps_range_x14_arr(const key_schedule&, std::size_t begin, std::size_t end, const uint8* in, uint8* out);
	// W6: x10/x12 base-pointer + merged-branchless dispatch (blmerge × higher interleave).
	void run_maps_range_x10_blmerge_arr(const key_schedule&, std::size_t begin, std::size_t end, const uint8* in, uint8* out);
	void run_maps_range_x12_blmerge_arr(const key_schedule&, std::size_t begin, std::size_t end, const uint8* in, uint8* out);

	// state_dedup interface (compile-time concept; no virtual). Natural layout,
	// so state_raw bytes are the canonical 128-byte state.
	const uint8* state_raw() const { return working_code_data; }
	void load_state_raw(const uint8* src) { for (int i = 0; i < 128; i++) working_code_data[i] = src[i]; }
	bool screen_state_raw(const uint8* src, uint8& flags_out);

	void run_bruteforce_data(uint32 key, uint32 data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size) override;

	// Interleaved (4-way) bulk forward + screen. Batches 4 independent
	// candidates (start_data+i .. +i+3) through the validated run_maps_range_x4
	// kernel, then screens each output; the tail (<4) falls back to 1-way. Bulk
	// candidates are trivially independent, so this exposes the same OOO
	// latency-hiding the dedup 4-way path uses, without a frontier pool.
	void run_bruteforce_data_il(uint32 key, uint32 data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size);

private:
	// Screen one post-maps state (natural layout) and emit a 5-byte record on a
	// checksum hit. Shared by the 1-way and interleaved bulk paths.
	void _screen_emit(__m512i w0, __m512i w1, uint32 idx, uint8* result_data, uint32& output_pos);

	void initialize();

	void _bind_schedule(const key_schedule& schedule_entries);
	void _bind_key(uint32 new_key);

	map_tables_shared::Tables& _t() { return _shared ? *_shared : _owned; }
	const map_tables_shared::Tables& _t() const { return _shared ? *_shared : _owned; }

	void alg_0(__m512i& wc0, __m512i& wc1, const uint8* block_start);
	void alg_1(__m512i& wc0, __m512i& wc1, const uint8* block_start);
	void alg_2(__m512i& wc0, __m512i& wc1, uint8 carry_byte);
	void alg_3(__m512i& wc0, __m512i& wc1, const uint8* block_start);
	void alg_4(__m512i& wc0, __m512i& wc1, const uint8* block_start);
	void alg_5(__m512i& wc0, __m512i& wc1, uint8 carry_byte);
	void alg_6(__m512i& wc0, __m512i& wc1, const uint8* block_start);
	void alg_7(__m512i& wc0, __m512i& wc1);

	void _run_alg(__m512i& wc0, __m512i& wc1,
	              int algorithm_id, uint16* local_pos,
	              const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base);

	void _run_one_map(__m512i& wc0, __m512i& wc1, int map_idx);
	void _run_maps_fixed(__m512i& wc0, __m512i& wc1, int map_idx, int count);
	void _run_all_maps(__m512i& wc0, __m512i& wc1);

	// One alg step for a single state (shared by 1-way and 4-way kernels). Uses
	// caller-supplied moving local_pos + shared schedule-entry table bases.
	void _alg_dispatch(__m512i& w0, __m512i& w1, int alg_id, uint16& local_pos,
	                   const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base);

	// One schedule entry over four interleaved states.
	void _run_map_entry_x4(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
		__m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		int map_idx);
	void _run_map_entry_x8(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		int map_idx);
	void _run_map1_entry_x8_scores(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		int map_idx, float scores[8]);
	void _run_map_entry_x8_alg0count(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		int map_idx, int counts[8]);
	void _run_map_entry_x8_optail(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		int map_idx, std::uint32_t keys[8], int counts[8]);
	void _alg_dispatch_blarith(__m512i& w0, __m512i& w1, int alg_id, uint16& local_pos,
		const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base);
	void _run_map_entry_x8_blarith(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		int map_idx);
	void _alg_dispatch_blmerge(__m512i& w0, __m512i& w1, int alg_id, uint16& local_pos,
		const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base);
	// W6: mask-by-reference variant (rematerialization lever for x12 — frees the 3 mask ZMM).
	void _alg_dispatch_blmerge_rm(__m512i& w0, __m512i& w1, int alg_id, uint16& local_pos,
		const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base,
		__m512i& mFE, __m512i& m7F, __m512i& mFF);
	void _run_map_entry_x8_blmerge(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		int map_idx);
	void _alg_dispatch_blall(__m512i& w0, __m512i& w1, int alg_id, uint16& local_pos,
		const uint8* reg_base, const uint8* alg0_base, const uint8* alg6_base);
	void _run_map_entry_x8_blall(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		int map_idx);
	template<int MODE>
	void _run_maps_range_x1_tpl(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in, uint8* out);
	void _run_map_entry_x4_blarith(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
		__m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		int map_idx);
	void _run_map_entry_x4_alg0count(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
		__m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		int map_idx, int counts[4]);
	void _run_map_entry_x12_alg0count(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		__m512i& i0, __m512i& i1, __m512i& j0, __m512i& j1, __m512i& k0, __m512i& k1, __m512i& l0, __m512i& l1,
		int map_idx, int counts[12]);
	void _run_map_entry_x10(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		__m512i& i0, __m512i& i1, __m512i& j0, __m512i& j1,
		int map_idx);
	// W6: x10 with merged-branchless dispatch (does blmerge stack with interleave>8? x8 was the historical
	// natmap ceiling). x10=20 state ZMM + blend-tree temps — tests whether it stays under 32 ZMM (no spill).
	void _run_map_entry_x10_blmerge(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		__m512i& i0, __m512i& i1, __m512i& j0, __m512i& j1,
		int map_idx);
	void _run_map_entry_x12_blmerge(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		__m512i& i0, __m512i& i1, __m512i& j0, __m512i& j1, __m512i& k0, __m512i& k1, __m512i& l0, __m512i& l1,
		int map_idx);
	void _run_map_entry_x12(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		__m512i& i0, __m512i& i1, __m512i& j0, __m512i& j1, __m512i& k0, __m512i& k1, __m512i& l0, __m512i& l1,
		int map_idx);
	void _run_map_entry_x14(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		__m512i& i0, __m512i& i1, __m512i& j0, __m512i& j1, __m512i& k0, __m512i& k1, __m512i& l0, __m512i& l1,
		__m512i& m0, __m512i& m1, __m512i& n0, __m512i& n1,
		int map_idx);

	void _load_from_mem(__m512i& wc0, __m512i& wc1);
	void _store_to_mem(__m512i& wc0, __m512i& wc1);

	void _expand_code(uint32 data, __m512i& wc0, __m512i& wc1);

	void _decrypt_carnival_world(__m512i& wc0, __m512i& wc1);
	void _decrypt_other_world(__m512i& wc0, __m512i& wc1);

	bool check_carnival_world_checksum(__m512i& wc0, __m512i& wc1);
	bool check_other_world_checksum(__m512i& wc0, __m512i& wc1);

	uint16 masked_checksum(__m512i& wc0, __m512i& wc1, const uint8* mask);
	uint16 fetch_checksum_value(__m512i& wc0, __m512i& wc1, uint8 code_length);
	void xor_alg(__m512i& wc0, __m512i& wc1, const uint8* values);

	uint32 _key = 0;

	map_tables_shared::Tables  _owned;
	map_tables_shared::Tables* _shared = nullptr;   // non-null = shared-mode

	uint8* _expansion_for_seed_128 = nullptr;       // pointer into rng->expansion_values_8 (no malloc)

	const __m512i mask_FF;
	const __m512i mask_FE;
	const __m512i mask_7F;
	const __m512i mask_80;
	const __m512i mask_01;

	ALIGNED(64) uint8 working_code_data[128];

	static bool initialized;
};
#endif // TM_AVX512_R512_MAP_8_H
