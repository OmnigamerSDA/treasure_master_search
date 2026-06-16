#ifndef TM_AVX512_R512S_8_H
#define TM_AVX512_R512S_8_H
#include <mmintrin.h>  //MMX
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSE4.2
//#include <ammintrin.h> //SSE4A
#include <immintrin.h> //AVX
//#include <zmmintrin.h> //AVX512

#include <cstddef>

#include "data_sizes.h"
#include "alignment2.h"
#include "rng_obj.h"
#include "tm_base.h"

class tm_avx512_r512s_8 : public TM_base
{
public:
	static constexpr int DEDUP_STATE_SHUFFLE = 512;

	tm_avx512_r512s_8(RNG* rng);

	virtual void load_data(uint8* new_data);
	void fetch_data(uint8* new_data);

	virtual void expand(uint32 key, uint32 data);

	virtual void run_alg(int algorithm_id, uint16* rng_seed, int iterations);

	virtual void run_one_map(const key_schedule::key_schedule_entry& schedule_entry);

	virtual void run_all_maps(const key_schedule& schedule_entries);

	// Register-resident map-range kernel for the dedup path. Holds both ZMM
	// state registers live across the whole [begin,end) group (one load at
	// entry, one store at exit) and extracts the per-step alg-select byte
	// directly from the registers (vpextrb on the low 128 bits) instead of the
	// store + scalar-reload round-trip run_all_maps/run_one_map use. Exposing
	// this makes state_dedup::run_map_group route AVX-512 onto the fast path.
	void run_maps_range(const key_schedule& schedule_entries, std::size_t begin, std::size_t end);

	// PROTOTYPE: 4-way interleaved map-range kernel. Runs [begin,end) over FOUR
	// independent states in lockstep, holding all 8 ZMM state registers live so
	// the OOO engine fills the per-step serial-chain latency of one state with
	// the other three's independent work. AVX-512's 2-ZMM-per-state footprint
	// (vs AVX2's 4 YMM) leaves vector ports under-used per state, and 32 ZMM
	// registers make 4-way spill-free — the conditions AVX2 2-way lacked. Reads
	// in0..in3 (impl shuffled layout), writes out0..out3.
	void run_maps_range_x4(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3);

	// PROTOTYPE: 6-way interleaved map-range kernel. 12 state ZMM (2/state) +
	// 5 mask + temps fits the 32-ZMM file spill-free. Profiling showed the dedup
	// forward is ~51-54% frontend-stalled on the data-dependent alg dispatch and
	// IPC drops to 0.81 on low-R/high-frontier keys (822M branch-misses) — more
	// independent states in flight hide deeper/clustered mispredict bursts that
	// 4-way can't. Expected gain concentrated on low-R keys. Reads in0..in5
	// (impl shuffled layout), writes out0..out5.
	void run_maps_range_x6(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3, const uint8* in4, const uint8* in5,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3, uint8* out4, uint8* out5);

	// PROTOTYPE: 8-way and 10-way interleaved map-range kernels (interleave-width
	// scaling study). x8 = 16 state ZMM + 5 masks + temps (spill-free); x10 = 20
	// state ZMM, right at the 32-ZMM edge (may spill — objdump-checked). Reads
	// inN (impl shuffled layout), writes outN.
	void run_maps_range_x8(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		uint8* out4, uint8* out5, uint8* out6, uint8* out7);
	// EXPERIMENT: x8 with merged branchless dispatch (universal-kernel port of natmap blmerge).
	void run_maps_range_x8_blmerge(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		uint8* out4, uint8* out5, uint8* out6, uint8* out7);
	// EXPERIMENT v2 (lower-load): one table load/dispatch instead of three.
	void run_maps_range_x8_blmerge2(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3,
		const uint8* in4, const uint8* in5, const uint8* in6, const uint8* in7,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3,
		uint8* out4, uint8* out5, uint8* out6, uint8* out7);
	void run_maps_range_x10(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3, const uint8* in4,
		const uint8* in5, const uint8* in6, const uint8* in7, const uint8* in8, const uint8* in9,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3, uint8* out4,
		uint8* out5, uint8* out6, uint8* out7, uint8* out8, uint8* out9);
	// x12 — past the 32-ZMM budget (24 state + masks), expected to spill; the
	// ceiling probe for the interleave-width study.
	void run_maps_range_x12(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3, const uint8* in4, const uint8* in5,
		const uint8* in6, const uint8* in7, const uint8* in8, const uint8* in9, const uint8* in10, const uint8* in11,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3, uint8* out4, uint8* out5,
		uint8* out6, uint8* out7, uint8* out8, uint8* out9, uint8* out10, uint8* out11);
	// x14 — 28 state ZMM; past the spill-free ceiling (masks already
	// rematerialized, nothing left to free). Built only to confirm the regression.
	void run_maps_range_x14(const key_schedule& schedule_entries, std::size_t begin, std::size_t end,
		const uint8* in0, const uint8* in1, const uint8* in2, const uint8* in3, const uint8* in4, const uint8* in5, const uint8* in6,
		const uint8* in7, const uint8* in8, const uint8* in9, const uint8* in10, const uint8* in11, const uint8* in12, const uint8* in13,
		uint8* out0, uint8* out1, uint8* out2, uint8* out3, uint8* out4, uint8* out5, uint8* out6,
		uint8* out7, uint8* out8, uint8* out9, uint8* out10, uint8* out11, uint8* out12, uint8* out13);

	// ---- streamhyb compatibility shims (UNIVERSAL build) -------------------------------------
	// Thin fallbacks so streamhyb_hash compiles against this kernel unchanged. The real x8/x12/x4
	// (and blmerge) above do the work; these degrade the natmap-only CAPTURE/score paths to plain
	// maps with zeroed outputs (unused on universal — drain is kernel-agnostic; the gate uses the
	// scalar-proxy deep_trajsp, not in-kernel capture). bind_* are no-ops (universal binds tables
	// internally in run_maps_range_*); screen_state_raw is a stub (universal runs in count mode).
	void bind_dedup_schedule(const key_schedule&) {}
	void bind_maps_range(const key_schedule&, std::size_t, std::size_t) {}
	bool screen_state_raw(const uint8*, uint8& flags) { flags = 0; return false; }
	void run_maps_range_x1(const key_schedule& s, std::size_t b, std::size_t e, const uint8* in, uint8* out, int /*disp*/){
		load_state_raw(in); run_maps_range(s,b,e); for(int i=0;i<128;i++) out[i]=working_code_data[i]; }
	void run_map1_range_x8_scores(const key_schedule& s, const uint8* i0,const uint8* i1,const uint8* i2,const uint8* i3,const uint8* i4,const uint8* i5,const uint8* i6,const uint8* i7,
		uint8* o0,uint8* o1,uint8* o2,uint8* o3,uint8* o4,uint8* o5,uint8* o6,uint8* o7, float sc[8]){
		run_maps_range_x8(s,0,1,i0,i1,i2,i3,i4,i5,i6,i7,o0,o1,o2,o3,o4,o5,o6,o7); for(int k=0;k<8;k++) sc[k]=0; }
	void run_maps_range_x8_optail(const key_schedule& s, std::size_t b, std::size_t e, const uint8* i0,const uint8* i1,const uint8* i2,const uint8* i3,const uint8* i4,const uint8* i5,const uint8* i6,const uint8* i7,
		uint8* o0,uint8* o1,uint8* o2,uint8* o3,uint8* o4,uint8* o5,uint8* o6,uint8* o7, std::uint32_t keys[8], int counts[8]){
		run_maps_range_x8(s,b,e,i0,i1,i2,i3,i4,i5,i6,i7,o0,o1,o2,o3,o4,o5,o6,o7); for(int k=0;k<8;k++){keys[k]=0;counts[k]=0;} }
	void run_maps_range_x8_route(const key_schedule& s, std::size_t b, std::size_t e, const uint8* i0,const uint8* i1,const uint8* i2,const uint8* i3,const uint8* i4,const uint8* i5,const uint8* i6,const uint8* i7,
		uint8* o0,uint8* o1,uint8* o2,uint8* o3,uint8* o4,uint8* o5,uint8* o6,uint8* o7, std::uint32_t keys[8], int sticky[8], int /*alg0_tau*/){
		run_maps_range_x8(s,b,e,i0,i1,i2,i3,i4,i5,i6,i7,o0,o1,o2,o3,o4,o5,o6,o7); for(int k=0;k<8;k++){keys[k]=0;sticky[k]=0;} }
	void run_maps_range_x8_dscore(const key_schedule& s, std::size_t b, std::size_t e, const uint8* i0,const uint8* i1,const uint8* i2,const uint8* i3,const uint8* i4,const uint8* i5,const uint8* i6,const uint8* i7,
		uint8* o0,uint8* o1,uint8* o2,uint8* o3,uint8* o4,uint8* o5,uint8* o6,uint8* o7, float sc[8]){
		run_maps_range_x8(s,b,e,i0,i1,i2,i3,i4,i5,i6,i7,o0,o1,o2,o3,o4,o5,o6,o7); for(int k=0;k<8;k++) sc[k]=0; }
	void run_maps_range_x8_blall(const key_schedule& s, std::size_t b, std::size_t e, const uint8* i0,const uint8* i1,const uint8* i2,const uint8* i3,const uint8* i4,const uint8* i5,const uint8* i6,const uint8* i7,
		uint8* o0,uint8* o1,uint8* o2,uint8* o3,uint8* o4,uint8* o5,uint8* o6,uint8* o7){
		run_maps_range_x8(s,b,e,i0,i1,i2,i3,i4,i5,i6,i7,o0,o1,o2,o3,o4,o5,o6,o7); }
	void run_maps_range_x8_blarith(const key_schedule& s, std::size_t b, std::size_t e, const uint8* i0,const uint8* i1,const uint8* i2,const uint8* i3,const uint8* i4,const uint8* i5,const uint8* i6,const uint8* i7,
		uint8* o0,uint8* o1,uint8* o2,uint8* o3,uint8* o4,uint8* o5,uint8* o6,uint8* o7){
		run_maps_range_x8(s,b,e,i0,i1,i2,i3,i4,i5,i6,i7,o0,o1,o2,o3,o4,o5,o6,o7); }
	void run_maps_range_x12_dscore(const key_schedule& s, std::size_t b, std::size_t e, const uint8* i0,const uint8* i1,const uint8* i2,const uint8* i3,const uint8* i4,const uint8* i5,const uint8* i6,const uint8* i7,const uint8* i8,const uint8* i9,const uint8* i10,const uint8* i11,
		uint8* o0,uint8* o1,uint8* o2,uint8* o3,uint8* o4,uint8* o5,uint8* o6,uint8* o7,uint8* o8,uint8* o9,uint8* o10,uint8* o11, float sc[12]){
		run_maps_range_x12(s,b,e,i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,o0,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11); for(int k=0;k<12;k++) sc[k]=0; }
	void run_maps_range_x4_dscore(const key_schedule& s, std::size_t b, std::size_t e, const uint8* i0,const uint8* i1,const uint8* i2,const uint8* i3, uint8* o0,uint8* o1,uint8* o2,uint8* o3, float sc[4]){
		run_maps_range_x4(s,b,e,i0,i1,i2,i3,o0,o1,o2,o3); for(int k=0;k<4;k++) sc[k]=0; }
	void run_maps_range_x4_blarith(const key_schedule& s, std::size_t b, std::size_t e, const uint8* i0,const uint8* i1,const uint8* i2,const uint8* i3, uint8* o0,uint8* o1,uint8* o2,uint8* o3){
		run_maps_range_x4(s,b,e,i0,i1,i2,i3,o0,o1,o2,o3); }

	// Raw internal-state accessors for state_dedup (src/common/state_dedup.h).
	const uint8* state_raw() const { return working_code_data; }
	void load_state_raw(const uint8* src) { for (int i = 0; i < 128; i++) working_code_data[i] = src[i]; }

	// Native screen (decrypt + masked checksum) overrides — operate on the member
	// working_code_data in the shuffled 2-ZMM layout. Without these the TM_base
	// defaults are no-ops/0, so the screen pass (decrypt + checksum over unique
	// survivors) would silently pass nothing. Mirrors tm_avx2_r256s_8's screen on
	// the 512 layout (pre-shuffled masks/world-data + vpsadbw checksum).
	virtual void decrypt_carnival_world();
	virtual void decrypt_other_world();
	virtual uint16 calculate_carnival_world_checksum();
	virtual uint16 calculate_other_world_checksum();
	virtual uint16 fetch_carnival_world_checksum_value();
	virtual uint16 fetch_other_world_checksum_value();

private:
	uint16 masked_checksum(const uint8* mask);
	uint16 fetch_checksum_value(uint8 code_length);

	void initialize();
	void alg_0(__m512i& working_code0, __m512i& working_code1, uint16* rng_seed, __m512i& mask_FE);
	void alg_2(__m512i& working_code0, __m512i& working_code1, uint16* rng_seed, __m512i& mask_80, __m512i& mask_7F, __m512i& mask_FE, __m512i& mask_01);
	void alg_3(__m512i& working_code0, __m512i& working_code1, uint16* rng_seed);
	void alg_5(__m512i& working_code0, __m512i& working_code1, uint16* rng_seed, __m512i& mask_80, __m512i& mask_7F, __m512i& mask_FE, __m512i& mask_01);
	void alg_6(__m512i& working_code0, __m512i& working_code1, uint16* rng_seed, __m512i& mask_7F);
	void alg_7(__m512i& working_code0, __m512i& working_code1, __m512i& mask_FF);
	void add_alg(__m512i& working_code0, __m512i& working_code1, uint16* rng_seed, uint8* rng_start);
	void alg_2_sub(__m512i& working_a, __m512i& working_b, __m512i& carry, __m512i& mask_80, __m512i& mask_7F, __m512i& mask_FE, __m512i& mask_01);
	void alg_5_sub(__m512i& working_a, __m512i& working_b, __m512i& carry, __m512i& mask_80, __m512i& mask_7F, __m512i& mask_FE, __m512i& mask_01);

	// Register-resident single map entry (16 steps) used by run_maps_range.
	void _run_map_entry(__m512i& working_code0, __m512i& working_code1,
		const key_schedule::key_schedule_entry& schedule_entry,
		__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01);

	// One alg step for a single state (the run_*_map_entry switch body), shared
	// by the 1-way and 4-way kernels.
	void _alg_dispatch(__m512i& w0, __m512i& w1, uint16& seed, unsigned alg_id,
		__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01);
	void _alg_dispatch_blmerge(__m512i& w0, __m512i& w1, uint16& seed, unsigned alg_id,
		__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01);
	void _alg_dispatch_blmerge2(__m512i& w0, __m512i& w1, uint16& seed, unsigned alg_id,
		__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01);
	void _run_map_entry_x8_blmerge(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
		__m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1,
		__m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		const key_schedule::key_schedule_entry& schedule_entry,
		__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01);
	void _run_map_entry_x8_blmerge2(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
		__m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1,
		__m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		const key_schedule::key_schedule_entry& schedule_entry,
		__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01);

	// One schedule entry over four interleaved states (see run_maps_range_x4).
	void _run_map_entry_x4(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
		__m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		const key_schedule::key_schedule_entry& schedule_entry,
		__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01);

	// One schedule entry over six interleaved states (see run_maps_range_x6).
	void _run_map_entry_x6(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
		__m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1,
		const key_schedule::key_schedule_entry& schedule_entry,
		__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01);

	// One schedule entry over eight / ten interleaved states.
	void _run_map_entry_x8(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
		__m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1,
		__m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		const key_schedule::key_schedule_entry& schedule_entry,
		__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01);
	void _run_map_entry_x10(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
		__m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1,
		__m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		__m512i& i0, __m512i& i1, __m512i& j0, __m512i& j1,
		const key_schedule::key_schedule_entry& schedule_entry,
		__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01);
	void _run_map_entry_x12(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1,
		__m512i& d0, __m512i& d1, __m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1,
		__m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1, __m512i& i0, __m512i& i1,
		__m512i& j0, __m512i& j1, __m512i& k0, __m512i& k1, __m512i& l0, __m512i& l1,
		const key_schedule::key_schedule_entry& schedule_entry,
		__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01);
	void _run_map_entry_x14(
		__m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1, __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1,
		__m512i& e0, __m512i& e1, __m512i& f0, __m512i& f1, __m512i& g0, __m512i& g1, __m512i& h0, __m512i& h1,
		__m512i& i0, __m512i& i1, __m512i& j0, __m512i& j1, __m512i& k0, __m512i& k1, __m512i& l0, __m512i& l1,
		__m512i& m0, __m512i& m1, __m512i& n0, __m512i& n1,
		const key_schedule::key_schedule_entry& schedule_entry,
		__m512i& mask_FF, __m512i& mask_FE, __m512i& mask_7F, __m512i& mask_80, __m512i& mask_01);

	int shuffle(int addr);

	ALIGNED(64) uint8 working_code_data[128];

	// Pre-shuffled (512-layout) copies of the world-decrypt operands and checksum
	// masks, built once in the constructor via shuffle_mem(..., 512, false).
	ALIGNED(64) uint8 carnival_world_data_shuffled[128];
	ALIGNED(64) uint8 other_world_data_shuffled[128];
	ALIGNED(64) uint8 carnival_world_checksum_mask_shuffled[128];
	ALIGNED(64) uint8 other_world_checksum_mask_shuffled[128];

	static bool initialized;
};
#endif // TM_AVX512_R512S_8_H
