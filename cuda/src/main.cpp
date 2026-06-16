// Treasure Master — CUDA forward-search host driver
//
// Loads the CUDA kernel bundle (tm_cuda.fatbin, built from tm_cuda.cu which
// includes tm_cuda_primitives.cuh / tm_cuda_screen.cuh / tm_cuda_dedup.cuh),
// allocates GPU assets, and dispatches the checksum-screen kernel in batches
// over the 2^32 data axis for a fixed key.  Survivors are materialised back
// to the host for CPU-side machine-code validation and statistics.
//
// PRODUCTION ENGINE (2026-06-16): the bounded-wave raceway (tm_cuda_raceway.cuh) — best
// across BOTH throughput and memory; the default for any system. Run it with
// `--raceway-direct-wave-continue-batch auto` (per-device span-ILP auto-applied from a
// `--calibrate-raceway` run). The flat screen and on-GPU compaction below are RESEARCH /
// A-B paths (the screen is also the bit-exact parity reference).
//
// Build:    make all         (compiles test_cuda binary + tm_cuda.fatbin)
// Invoke:   ./test_cuda --device <id> --key_id 0x... --workunit_size <N>
//
// Key flags:
//   --map-list all|skip-car  choose ALL_MAPS or SKIP_CAR schedule variant
//   --parity <N>             cross-check N candidates against the CPU reference
//   --hll                    enable HyperLogLog output-space cardinality estimate
//   --raceway-direct-wave-continue-batch auto   PRODUCTION bounded-wave raceway
//   --calibrate-raceway      tune raceway span-ILP/cap-bits for this device (production)
//   --screen-offsets         per-key coalesced offset-stream screen (research / parity ref)
//   --compaction-bench       on-GPU VRAM compaction architecture (research / A-B)
//
// Sections (within the anonymous namespace below):
//   - Kernel screen-flag bit definitions
//   - Validation predicates and opcode tables
//   - Data structures (Args, KernelAssets, SurvivorRecord, ValidationSummary)
//   - Argument parsing
//   - CUDA utilities (module loading, asset allocation)
//   - CPU reference implementation (parity testing)
//   - Machine-code validation (check_machine_code, linear_scan_*)
//   - Output formatting (print_summary)
//   - Kernel grid dimension helpers

#include <cuda.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <set>
#include <string>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>

// cub sort-based collapse for the wide-merge dedup (wide_merge_sort.cu, nvcc TU).
extern "C" int wms_sort_dedup(const uint64_t* h_fps, uint32_t n,
	uint32_t* h_survivor_idx, uint32_t* h_run_len, uint32_t* out_unique, float* out_sort_ms);
extern "C" int wms_sort_dedup_mult(const uint64_t* h_fps, const uint32_t* h_idx_in, const uint32_t* h_mult_in,
	uint32_t n, uint32_t* h_idx_out, uint32_t* h_mult_out, uint32_t* out_unique);
#include <vector>
#include <unistd.h>
#include <climits>
#include <cmath>

#include "rng_obj.h"
#include "key_schedule.h"
#include "window_policy.h"

namespace
{
	static const uint32_t kDefaultBatchSize = 1u << 20;
	static const uint32_t kCudaWarpSize = 32u;
	static const uint32_t kCudaWarpsPerBlock = 4u;
	static const uint32_t kCudaThreadsPerBlock = kCudaWarpSize * kCudaWarpsPerBlock;
	static const uint32_t kCudaScreenCandidatesPerWarp = 4u;
	// ALL_MAPS: 26 base entries (map 0x22 gets a duplicate via build_schedule_blob → 27 total)
	static const uint8_t kMapList[26] = {
		0x00, 0x02, 0x05, 0x04, 0x03, 0x1D, 0x1C, 0x1E, 0x1B,
		0x07, 0x08, 0x06, 0x09, 0x0C, 0x20, 0x21, 0x22, 0x23,
		0x24, 0x25, 0x26, 0x0E, 0x0F, 0x10, 0x12, 0x11
	};
	// SKIP_CAR: 25 base entries (skips map 0x06; map 0x22 gets a duplicate → 26 total)
	static const uint8_t kMapListSkipCar[25] = {
		0x00, 0x02, 0x05, 0x04, 0x03, 0x1D, 0x1C, 0x1E, 0x1B,
		0x07, 0x08, 0x09, 0x0C, 0x20, 0x21, 0x22, 0x23,
		0x24, 0x25, 0x26, 0x0E, 0x0F, 0x10, 0x12, 0x11
	};

	// ── Kernel screen-flag bit definitions ──────────────────────────────────────
	// Flags written into the per-candidate result_data[] byte by the GPU screen
	// kernel, and then re-read by the CPU materialise / validate path.
	static const uint8_t CHECKSUM_SENTINEL = 0x08;   // set when any checksum passes
	static const uint8_t OTHER_WORLD       = 0x01;   // set when other-world checksum passes
	static const uint8_t DUAL_PASS         = 0x02;  // set by CUDA when BOTH checksums pass
	static const uint8_t FIRST_ENTRY_VALID = 0x02;  // set by CPU validation (different byte: machine_flags)
	static const uint8_t ALL_ENTRIES_VALID = 0x04;
	static const uint8_t USES_NOP = 0x10;
	static const uint8_t USES_UNOFFICIAL_NOPS = 0x20;
	static const uint8_t USES_ILLEGAL_OPCODES = 0x40;
	static const uint8_t USES_JAM = 0x80;

	static const uint8_t OP_JAM = 0x01;
	static const uint8_t OP_ILLEGAL = 0x02;
	static const uint8_t OP_NOP2 = 0x04;
	static const uint8_t OP_NOP = 0x08;
	static const uint8_t OP_JUMP = 0x10;

	// ── Validation predicates and opcode tables ─────────────────────────────────
	// Structural predicates evaluated on each other-world survivor's decoded code
	// region.  Each corresponds to one expected byte sequence derived from an
	// earlier reverse-engineering audit of the game's acceptance structure.
	// kOpcodeType / kOpcodeBytesUsed encode the 6502 instruction set for the
	// machine-code validation pass (check_machine_code, linear_scan_clean).
	// Note: an identical copy of these tables lives in the OpenCL main.cpp.
	enum OtherPredicateIndex
	{
		OPRED_FINAL_RTS_50 = 0,
		OPRED_ENTRY00_LOAD,
		OPRED_ENTRY05_LOAD,
		OPRED_ENTRY0A_LOAD,
		OPRED_ENTRY28_LOAD,
		OPRED_ENTRY40_LOAD,
		OPRED_INIT_CALL_OR_JUMP_02,
		OPRED_INIT_JMP_02,
		OPRED_CALL_OR_JUMP_07,
		OPRED_JSR_07,
		OPRED_JUMP_25,
		OPRED_BRANCH_2D,
		OPRED_JSR_3A,
		OPRED_JUMP_3D,
		OPRED_BRANCH_45,
		OPRED_JUMP_4D,
		OPRED_INIT_JMP_8595,
		OPRED_ENTRY05_JSR_80B4,
		OPRED_JUMP_25_TARGET_8464,
		OPRED_JSR_3A_TARGET_8952,
		OPRED_JUMP_3D_TARGET_81EE,
		OPRED_JUMP_4D_TARGET_80B4,
		OTHER_PREDICATE_COUNT
	};

	static const char* kOtherPredicateNames[OTHER_PREDICATE_COUNT] = {
		"final_rts_50",
		"entry00_load",
		"entry05_load",
		"entry0a_load",
		"entry28_load",
		"entry40_load",
		"init_call_or_jump_02",
		"init_jmp_02",
		"call_or_jump_07",
		"jsr_07",
		"jump_25",
		"branch_2d",
		"jsr_3a",
		"jump_3d",
		"branch_45",
		"jump_4d",
		"init_jmp_8595",
		"entry05_jsr_80b4",
		"jump_25_target_8464",
		"jsr_3a_target_8952",
		"jump_3d_target_81ee",
		"jump_4d_target_80b4",
	};

	static const uint8_t kOpcodeBytesUsed[0x100] = {
		1,2,0,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0,
		3,2,0,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0,
		1,2,0,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0,
		1,2,0,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0,
		2,2,2,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,0,3,0,0,
		2,2,2,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,3,3,3,0,
		2,2,2,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0,
		2,2,2,0,2,2,2,0,1,2,1,0,3,3,3,0,
		2,2,0,0,2,2,2,0,1,3,1,0,2,3,3,0
	};

	static const uint8_t kOpcodeType[0x100] = {
		0, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, OP_JUMP, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, OP_JUMP, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		OP_NOP2, 0, OP_NOP2, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, OP_NOP2, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, OP_ILLEGAL, 0, OP_ILLEGAL, OP_ILLEGAL,
		0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		0, 0, OP_NOP2, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL,
		0, 0, OP_NOP2, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP, OP_ILLEGAL, 0, 0, 0, OP_ILLEGAL,
		OP_JUMP, 0, OP_JAM, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL, 0, 0, OP_NOP2, OP_ILLEGAL, OP_NOP2, 0, 0, OP_ILLEGAL
	};

	// ── Data structures ──────────────────────────────────────────────────────────
	struct Args
	{
		uint32_t key_id = 0x2CA5B42Du;
		uint64_t range_start = 0u;
		uint64_t workunit_size = 1u << 20;
		uint32_t batch_size = kDefaultBatchSize;
		uint32_t warmup_batches = 1u;
		uint32_t device_index = 0u;
		uint32_t parity_count = 0u;   // 0 = no parity test
		std::string output_csv_path;
		std::string map_list = "all";  // "all" or "skip-car"
		// HLL cardinality estimation is off by default — it's a profiling/diagnostic
		// feature, costs ~10% throughput, and the production goal is maximum sweep
		// rate. Opt in with --hll.
		bool hll = false;
		std::string hll_dump_path;   // if set, write raw 4096 uint32 HLL registers to this file
		bool maprng = false;  // POC: per-launch 54 KB map_rng table replaces 8 MB universal tables
		bool dedup = false;   // POC: per-boundary cooperative state dedup
		uint32_t dedup_width = 32u;  // 32 (Phase 1, 1-per-warp) or 128 (Phase 1.5/2, ILP4)
		bool dedup_skip = false;     // Phase 2: actually skip run_alg for dead duplicates (requires W=128)
		bool screen_dedup = false;   // Phase 2.5 prod: screen kernel + dedup, output 1 byte per candidate
		uint32_t screen_dedup_k = 1u; // dedup period (1 = every boundary, 2/3/6 = every Kth)
		bool screen_dedup_packed = false; // use packed 64-bit slot variant (no spin-wait)
		bool screen_dedup_fasthash = false; // cheaper warp hash (no per-lane salt, no murmur)
		bool screen_dedup_maprng = false;   // map_rng buffer instead of 8MB universal tables
		bool screen_dedup_maprng_preext = false;  // pre-extracted maprng (3 streams: raw + alg0 + alg6)
		bool screen_dedup_maprng_preext_1sync = false;  // + 1 sync/boundary instead of 2
		bool screen_dedup_maprng_preext_cstore = false; // + coalesced final store (warp-0 single 32-byte store/block)
		std::string preext_policy;  // dedup-schedule experiment on preext base: first|k2|k3|k4|k5|k6|k8|output
		std::string offset_geom;    // ILP-geometry offset dedup: ilp4w8_k4|ilp6w8_k4|ilp6w8_k1|ilp6w4_k4|ilp8w8_k4|ilp5w8_k4
		bool compaction_bench = false;  // on-GPU VRAM survivor-compaction architecture test
		bool drain_bench = false;       // WS1 drop-drain POC (Path A): ilp6 screen + dedup checkpoint at map6
		std::string drain_boundaries;   // WS1 Path B: comma list of span-boundary map idxs to GLOBAL-drain in compaction
		uint32_t drain_cap_bits = 0;    // WS1 cap: >0 ⇒ drains use a fixed 2^cap_bits-bucket inverse-bloom cap (epoch, no re-zero)
		bool drain_cap_bits_explicit = false; // true if --drain-cap-bits passed (disables --compaction-run cap-auto scaling)
		uint32_t drain_cap_ways = 4;    // cap bucket associativity
		std::string drain_cap_cas = "128"; // WS1 cap CAS scheme: 128 (atom.cas.b128, epoch) | 64a (epoch-packed 64-bit CAS, no re-zero)
		                                   //   | 64b (two-array 64-bit CAS, re-zero per boundary) | 64c (K two-array tables, zero once)
		uint32_t drain_route_tau = 0;   // WS1: >0 ⇒ the DEEP drains route (shed-proxy over the deep span): hash only
		                                //   shed>=tau into the deep cap, pass the rest un-hashed. MEMORY lever (smaller deep
		                                //   cap holds only the high-shed subset) for massive diffuse frontiers; single-table cap only.
		bool drain_route_traj = false;  // WS1: trajgate deep-routing — the CORRECT deep classifier (op-tail count-min
		                                //   trajDens + sticky alg0), not the MAP1 shed. dens_tau=drain_route_tau.
		uint32_t drain_sticky_tau = 0;  //   per-map alg0 sticky threshold (0 = sticky off; catches first-seen high-collision reps)
		uint32_t drain_mult_tau = 0;    //   multiplicity threshold OR term for deep trajgate routing (0 = off)
		uint32_t drain_traj_bits = 20;  //   count-min sketch size = 2^bits u32 cells (4 hashes)
		bool drain_traj_2pass = false;  //   two-pass: build the FULL sketch (pass1) then probe+route (pass2). Fixes the
		                                //   racy single-pass density (parallel probe-then-update); matches/beats CPU AUC.
		bool drain_cross_tile = false;  // WS1: persist the cap across sweep tiles (boundary-keyed epoch) → cross-window dedup
		uint32_t map1_route_tau = 0;    // WS1: >0 ⇒ span-0 MAP1 merge uses shed-proxy routing (hash shed>=tau into a cap, pass the rest)
		uint32_t map1_cap_bits = 22;    // span-0 routed cap size (2^bits buckets)
		uint32_t map1_cap_ways = 4;
		bool map1_table_auto = false;   // HLL-auto: presize the span-0 global table to 2x estimated MAP1 frontier
		bool drain_table_auto = false;  // B: per-drain EXACT-presize the deep drain table to 2x M (known frontier); reports LF
		bool map1_cap_auto = false;     // route+dynamic: routed-HLL pass sizes the routed MAP1 cap to 2x(routed cardinality)
		std::string compaction_spans;   // comma list of span-boundary map indices (default k4 shape)
		std::string compaction_ilp = "w8i8";  // span-kernel geometry; w8i8 is optimal with preids+ordered
		bool compaction_ilp_explicit = false; // true if --compaction-ilp was passed (overrides the calibrated geom)
		bool compaction_ordered = true;   // block-stable ordered compaction (coalesced gather + wider R); default on
		bool compaction_skip_final = true;  // skip the after-last-map hash (saves no map-work); default on
		bool compaction_fuse_final = true;  // fuse the final screen into the last span (no separate final_flag); default on
		bool compaction_sweep = false;      // run the full 2^32 data sweep (every tile), end-to-end timing + parity
		bool compaction_run = false;        // operational preset: full-key sweep with the settled host-appropriate default flags
		bool compaction_auto_tile = false;  // pick the tile size from free VRAM (portable across cards)
		bool compaction_global_span0 = false; // span-0 dedup uses a GLOBAL VRAM hash table (captures map-1 global collapse)
		uint32_t compaction_global_cap_mb = 0u; // window-cap: max VRAM (MB) for the span-0 global table; 0 = auto (2x tile)
		bool compaction_multiplicity = false; // carry per-original representative multiplicity through whole-tile dedup
		std::string sweep_hit_file;         // --sweep-hit-file PATH: write every hitting (data,flag) across the 2^32 sweep
		uint32_t sweep_hit_print = 64u;     // --sweep-hit-print N: inline-print the first N hits (0 = summary counts only)
		bool calibrate = false;             // measure host-optimal (research) screen-vs-compaction engine
		bool calibrate_raceway = false;     // measure host-optimal PRODUCTION raceway geom (span-ILP x cap-bits)
		std::string config_path = "tm_compaction.conf";  // calibration config (shared with OpenCL)
		bool screen_dedup_offsets = false; // universal-table dedup with per-key coalesced offset streams
		bool screen_coalesced = false;  // Phase 3: coalesced maprng+preext SCREEN kernel (no dedup), vs universal-table screen
		bool screen_preext = false;     // SCREEN-only 3-stream pre-extracted maprng, production geometry
		bool screen_offsets = false;    // SCREEN-only per-key coalesced offset streams, no seed-forward loads
		bool screen_offsets_interleaved = false; // SCREEN-only offset streams grouped by schedule entry
		bool screen_offset_ilp_sweep = false; // A/B screen-offset kernels at ILP 1/2/4/8
		bool byte_store = false;  // A/B/debug: use original lane-0 byte stores instead of packed uint32 stores
			// Wide-merge sort-dedup (2026-06-03): dump per-candidate 64-bit fingerprints
			// after the first FIRST_MAPS schedule entries, then count unique on host to
			// confirm real-key collapse R matches the CPU's parity-exact figure.
			bool wide_merge_dump = false;
			uint32_t wide_merge_first_maps = 1u; // merge point (1 or 2); biggest collapse at 0-1
				std::string wide_merge_points;       // periodic multi-merge schedule, e.g. "1,4,13,27"
				bool map1_frontier = false;          // W4B prototype: persistent MAP1 representative hash set
				bool map1_frontier_axis_sweep = false; // full 2^32 as independent workunit_size MAP-K windows
				bool map1_frontier_mobile_bit_sweep = false; // 32M windows as two 16M slices, mobile high-byte bit
				bool map1_frontier_backfill_sweep = false; // arbitrary build-from-byte-back selected-bit windows
				bool map1_frontier_lowbits_sweep = false; // same mask probe, but contiguous low bits
				std::string map1_frontier_mask_policy = "backfill"; // backfill, lowbits, squeeze, backfavor, highbits, omitbyteN
				uint32_t map1_frontier_table_mb = 0u; // 0 = auto (2x window, capped by free VRAM)
			uint32_t map1_frontier_chunk = 4194304u;
				bool map1_frontier_consume = false;  // emit reps and run indexed downstream screen
				bool map1_frontier_offset_only = false; // skip canonical consumer timing in broad surveys
				uint32_t map1_frontier_rep_cap = 0u; // 0 = auto for small windows; explicit cap for large windows
				bool map1_frontier_raceway = false;  // seed the warp-persistent raceway POC from MAP1 reps
				uint32_t raceway_cap_bits = 22u;     // per-boundary cap buckets = 2^bits (operational default; 2^22x4 = full-key sweet spot)
				uint32_t raceway_cap_ways = 4u;
				uint32_t raceway_cap_count = 1u;     // number of consecutive post-map caps
				uint32_t raceway_first_cap_map = 1u; // 1 => first probe after MAP2 completes
				bool raceway_flag_parity = false;    // no-cap raceway flags vs existing offset consumer
				bool raceway_drop_hist = false;      // copy drop_map_out and print per-boundary drops
				bool raceway_flat_parity = false;    // no-cap raceway reps vs full-window flat screener
				bool raceway_direct = false;         // full-window raceway: MAP1 calculation inline, no MAP1 dedup
				bool raceway_direct_flat_parity = false; // direct raceway flags vs full-window flat screener
				bool raceway_direct_offset = false;  // direct raceway using per-key offset streams instead of rng-forward tables
				uint32_t raceway_direct_ilp = 1u;    // direct raceway candidates per warp; currently 1 or offset-only 4
				bool raceway_direct_static = false;  // no-cap static grid mapping for direct raceway ILP probes
				bool raceway_direct_cap_mark = false; // boundary-cap mark pass: writes alive/drop, no final screen
				bool raceway_direct_cap_compact = false; // pack alive output from cap-mark with ordered compaction
				bool raceway_direct_cap_continue = false; // compact then recompute-screen live_idx survivors
				bool raceway_direct_cap_state_continue = false; // save boundary state, compact, continue from state
				uint32_t raceway_direct_stream_batch = 0u; // bounded origination batch size; caps persist across batches
				uint32_t raceway_direct_wave_continue_batch = 0u; // bounded mark+compact+continue wave size; "auto" sizes from free VRAM
				bool raceway_wave_auto_size = false; // --raceway-direct-wave-continue-batch auto: size wave to the VRAM budget
				uint32_t raceway_direct_wave_continue_ilp = 0u; // 0=auto: ILP4 state continuation, ILP6 recompute continuation
				uint32_t raceway_direct_wave_span_ilp = 4u; // cap-span/mark phase ILP: 4 = operational default (ILP4, +35% over ILP1, full occupancy); 1/5 also valid
				bool raceway_span_ilp_explicit = false; // true if --raceway-direct-wave-span-ilp was passed (overrides the calibrated geom)
				bool raceway_direct_wave_parity = false; // alive-only flat parity for bounded wave continuation
				bool raceway_direct_wave_state = false; // bounded wave continuation reads saved boundary state
				bool raceway_direct_wave_k_sweep = false; // sweep f1k-style completed-map drain cadences
				std::string raceway_direct_wave_k_list; // comma list of K values for f1k sweep
				std::string raceway_direct_wave_boundaries; // comma list of completed-map drain boundaries
				bool map1_frontier_deep = false;     // run local-frontier span compaction after MAP1 reps
			uint32_t map1_frontier_deep_k = 4u;  // post-MAP1 merge cadence; K4 preserves the original GPU default
			bool map1_frontier_deep_k_sweep = false;
			std::string map1_frontier_deep_k_list;
			std::string map1_frontier_stream_cuts; // explicit deep merge boundaries for stream-deep (mirror raceway cadence)
			bool map1_frontier_stream_deep = false; // W4B: deep-process each ordered MAP1 chunk
				uint32_t map1_frontier_drain_cap_bits = 0u; // Pool B: persistent cross-chunk deep cap (0 = off)
				uint32_t map1_frontier_drain_cap_ways = 4u; // ways per bucket for the producer deep cap
				bool map1_frontier_drain_cap_distinct = false; // Pool B: K distinct per-depth tables (no cross-depth eviction)
				uint32_t map1_frontier_drain_shed_tau = 0u; // shed-proxy tau at MAP1 (0=off; >0: discard MAP1 survivors with alg0/alg6 count < tau)
				uint32_t map1_frontier_drain_shed_gate_tau = 0u; // Pool B shed-proxy cap gate (0=off; low-shed states pass alive)
				uint32_t map1_frontier_drain_fp_gate_log = 0u; // Pool B final-fp sampling gate (0=off; N routes ~1/2^N)
				uint32_t map1_frontier_drain_traj_bits = 0u; // producer Pool B trajgate sketch bits (0=off)
				uint32_t map1_frontier_drain_dens_tau = 32u; // producer Pool B trajgate density threshold
				uint32_t map1_frontier_drain_alg0_tau = 3u; // producer Pool B trajgate sticky alg0 threshold
				bool map1_frontier_drain_alg0_tau_explicit = false; // explicit alg0 tau without traj bits enables sticky-only gating
				uint32_t map1_frontier_drain_mult_tau = 0u; // producer Pool B multiplicity threshold OR term (0=off)
				uint32_t map1_frontier_route_tau = 0u; // shed-routing tau for producer MAP1 (0=off; >0: measure routed cardinality + Phase 2 route insert)
				bool map1_frontier_multiplicity = false; // sidecar representative multiplicity for wide MAP1 stream-deep
				uint32_t map1_frontier_partitions = 1u; // power-of-two strong64 prefix partitions; streamed-deep only
			bool map1_frontier_wide = false;     // 128-bit fp + warp-cooperative bucketed insert (W4B-exact, 16 B/slot)
			uint32_t map1_frontier_wide_ilp = 1u; // candidates per warp for the wide insert: 1, 2, or 4 (Target A: MLP)
				uint32_t map1_frontier_depth = 1u;   // dedup at the MAP-K frontier (run K maps before fingerprint); 1=MAP1
				uint32_t map1_frontier_sample_mult = 1u; // odd Weyl multiplier: quasi-random sample across 2^32 (class spot-check); 1=contiguous
				uint32_t map1_frontier_mobile_extra_bits = 1u; // mobile-bit sweep: extra coordinated bits beyond --workunit_size
				uint32_t map1_frontier_mobile_groups = 0u; // 0 = all groups if small, otherwise sampled
				bool map1_frontier_mobile_backfill = false; // select highest available bits instead of sweeping one bit
				std::string map1_frontier_backfill_bits; // comma list of selected-bit counts; default 8,12,16,20,24
			};

	struct KernelAssets
	{
		CUdeviceptr regular_rng_values = 0;
		CUdeviceptr alg0_values = 0;
		CUdeviceptr alg6_values = 0;
		CUdeviceptr rng_seed_forward_1 = 0;
		CUdeviceptr rng_seed_forward_128 = 0;
		CUdeviceptr alg2_values = 0;
		CUdeviceptr alg5_values = 0;
		CUdeviceptr expansion_values = 0;
		CUdeviceptr schedule_data = 0;
		CUdeviceptr carnival_data = 0;
	};

		struct RacewayStatsHost
		{
			unsigned long long reps_started = 0;
			unsigned long long reps_completed = 0;
			unsigned long long reps_dropped = 0;
			unsigned long long map_evals = 0;
		};

		struct RacewayBoundaryStats
		{
			uint32_t completed_map = 0;
			uint64_t input = 0;
			uint64_t survivors = 0;
			uint64_t dropped = 0;
			uint64_t map_evals = 0;
			double span_ms = 0.0;
			double compact_ms = 0.0;
			uint32_t cap_occupied = 0;
			uint64_t cap_slots = 0;
		};

		struct OccRecord
		{
			std::string label;
			uint32_t occupied = 0;
			uint64_t slots = 0;
			uint32_t bytes_per_slot = 0;
		};

		struct SurvivorRecord
		{
			uint32_t data = 0;
			uint8_t screen_flags = 0;
		uint8_t machine_flags = 0;
	};

	struct ValidationSummary
	{
		uint64_t total = 0;
		uint64_t carnival = 0;
		uint64_t other = 0;
		uint64_t dual_pass = 0;             // passed both carnival and other-world checksums
		uint64_t first_entry_valid = 0;
		uint64_t all_entries_valid = 0;
		uint64_t all_entries_valid_carnival = 0;
		uint64_t all_entries_valid_other    = 0;
		uint64_t uses_nop = 0;
		uint64_t uses_unofficial_nops = 0;
		uint64_t uses_illegal = 0;
		uint64_t uses_jam = 0;
		// Linear-stream scan: walk entire code region as instruction stream;
		// count survivors where stream completes with no JAM/ILLEGAL encountered,
		// and track the maximum score (opcode positions walked before first bad byte
		// or until end of stream if clean) as a gradient proximity signal.
		uint64_t linear_scan_clean_carnival    = 0;
		uint64_t linear_scan_clean_other       = 0;
		uint64_t linear_scan_score_max_carnival = 0;   // max opcode steps walked (carnival path)
		uint64_t linear_scan_score_max_other    = 0;   // max opcode steps walked (other-world path)
		uint64_t other_final_rts = 0;
		uint64_t other_entry_opcodes = 0;
		uint64_t other_control_flow = 0;
		uint64_t other_structural = 0;
		uint64_t all_entries_other_final_rts = 0;
		uint64_t all_entries_other_entry_opcodes = 0;
		uint64_t all_entries_other_control_flow = 0;
		uint64_t all_entries_other_structural = 0;
		uint64_t other_predicate[OTHER_PREDICATE_COUNT] = {};
		uint64_t final_rts_other_predicate[OTHER_PREDICATE_COUNT] = {};
		uint64_t all_entries_other_predicate[OTHER_PREDICATE_COUNT] = {};
		uint64_t all_entries_final_rts_other_predicate[OTHER_PREDICATE_COUNT] = {};
		// Hash-based collision analysis: distinct 128-byte decrypted states
		// (checksum-survivor population only — biased sample of full output space).
		uint64_t unique_states_carnival = 0;
		uint64_t unique_states_other    = 0;
		// HyperLogLog estimate of distinct output states across ALL 2^32 data
		// values — the true output-space cardinality, unaffected by checksum bias.
		uint64_t hll_distinct_states    = 0;
	};

	// ── Argument parsing ─────────────────────────────────────────────────────────
	template <typename T>
	T numeric_arg(const char* value, const char* name)
	{
		char* end = nullptr;
		unsigned long parsed = std::strtoul(value, &end, 0);
		if (end == value || *end != '\0')
		{
			std::ostringstream message;
			message << "Invalid numeric value for " << name << ": " << value;
			throw std::runtime_error(message.str());
		}
		return static_cast<T>(parsed);
	}

	std::vector<uint32_t> make_post_map1_cuts(uint32_t k)
	{
		if (k == 0u || k > 26u)
			throw std::runtime_error("post-MAP1 K must be 1..26");
		std::vector<uint32_t> cuts;
		cuts.push_back(0u);
		cuts.push_back(1u);
		for (uint32_t m = 1u + k; m < 27u; m += k)
			cuts.push_back(m);
		if (cuts.back() != 27u)
			cuts.push_back(27u);
		return cuts;
	}

	// Build stream-deep cuts from an explicit deep-boundary list (e.g. "2,4,8,14,20"),
	// to mirror the raceway's drain cadence. MAP1 dedup (cut at 1) is intrinsic to the
	// producer and always present; 27 is the terminal cut.
	std::vector<uint32_t> make_post_map1_cuts_custom(const std::vector<uint32_t>& boundaries)
	{
		std::vector<uint32_t> cuts;
		cuts.push_back(0u);
		cuts.push_back(1u);
		uint32_t prev = 1u;
		for (uint32_t b : boundaries)
		{
			if (b <= prev || b >= 27u)
				throw std::runtime_error("--map1-frontier-stream-cuts must be strictly increasing, each in 2..26");
			cuts.push_back(b);
			prev = b;
		}
		cuts.push_back(27u);
		return cuts;
	}

	std::vector<uint32_t> parse_u32_list(const std::string& text, const char* name)
	{
		std::vector<uint32_t> out;
		std::stringstream ss(text);
		std::string tok;
		while (std::getline(ss, tok, ','))
		{
			if (tok.empty()) continue;
			out.push_back(static_cast<uint32_t>(std::stoul(tok)));
		}
		if (out.empty())
			throw std::runtime_error(std::string(name) + " must contain at least one value");
		return out;
	}

	uint32_t make_frontier_policy_bit_mask(const std::string& policy, uint32_t selected_bit_count, uint32_t map_depth)
	{
		return tm_window_policy::make_bit_mask(policy, selected_bit_count, map_depth);
	}

	Args parse_args(int argc, char** argv)
	{
		Args args;
		for (int i = 1; i < argc; i++)
		{
			const std::string arg(argv[i]);
				if (arg == "--key_id" && i + 1 < argc)
				{
					args.key_id = numeric_arg<uint32_t>(argv[++i], "--key_id");
				}
				else if (arg == "--range_start" && i + 1 < argc)
				{
					args.range_start = numeric_arg<uint64_t>(argv[++i], "--range_start");
				}
				else if (arg == "--workunit_size" && i + 1 < argc)
				{
					args.workunit_size = numeric_arg<uint64_t>(argv[++i], "--workunit_size");
				}
			else if (arg == "--batch_size" && i + 1 < argc)
			{
				args.batch_size = numeric_arg<uint32_t>(argv[++i], "--batch_size");
			}
			else if (arg == "--warmup_batches" && i + 1 < argc)
			{
				args.warmup_batches = numeric_arg<uint32_t>(argv[++i], "--warmup_batches");
			}
			else if (arg == "--device" && i + 1 < argc)
			{
				args.device_index = numeric_arg<uint32_t>(argv[++i], "--device");
			}
			else if (arg == "--output_csv" && i + 1 < argc)
			{
				args.output_csv_path = argv[++i];
			}
			else if (arg == "--parity" && i + 1 < argc)
			{
				args.parity_count = numeric_arg<uint32_t>(argv[++i], "--parity");
			}
			else if (arg == "--map-list" && i + 1 < argc)
			{
				args.map_list = argv[++i];
				if (args.map_list != "all" && args.map_list != "skip-car")
				{
					throw std::runtime_error("--map-list must be 'all' or 'skip-car'");
				}
			}
			else if (arg == "--hll")
			{
				args.hll = true;
			}
			else if (arg == "--dump-hll" && i + 1 < argc)
			{
				args.hll = true;
				args.hll_dump_path = argv[++i];
			}
			else if (arg == "--no-hll")
			{
				// Back-compat no-op (HLL is off by default now). Accepted silently
				// so older scripts/wrappers don't break.
				args.hll = false;
			}
			else if (arg == "--maprng")
			{
				args.maprng = true;
				args.hll = false;  // POC: maprng path doesn't implement HLL
			}
			else if (arg == "--dedup")
			{
				args.dedup = true;
			}
			else if (arg == "--dedup-w" && i + 1 < argc)
			{
				args.dedup = true;
				args.dedup_width = numeric_arg<uint32_t>(argv[++i], "--dedup-w");
				if (args.dedup_width != 32u && args.dedup_width != 64u && args.dedup_width != 128u)
				{
					throw std::runtime_error("--dedup-w must be 32, 64, or 128");
				}
			}
			else if (arg == "--dedup-skip")
			{
				args.dedup = true;
				args.dedup_skip = true;
				// dedup_width determines block size; W=32/64/128 all supported in skip mode
				if (args.dedup_width != 32u && args.dedup_width != 64u && args.dedup_width != 128u)
				{
					args.dedup_width = 128u;
				}
			}
			else if (arg == "--screen-dedup")
			{
				args.screen_dedup = true;
			}
			else if (arg == "--screen-dedup-k" && i + 1 < argc)
			{
				args.screen_dedup = true;
				args.screen_dedup_k = numeric_arg<uint32_t>(argv[++i], "--screen-dedup-k");
				if (args.screen_dedup_k != 1u && args.screen_dedup_k != 2u && args.screen_dedup_k != 3u && args.screen_dedup_k != 6u)
				{
					throw std::runtime_error("--screen-dedup-k must be 1, 2, 3, or 6");
				}
			}
			else if (arg == "--screen-dedup-packed")
			{
				args.screen_dedup = true;
				args.screen_dedup_packed = true;
			}
			else if (arg == "--screen-dedup-fasthash")
			{
				args.screen_dedup = true;
				args.screen_dedup_fasthash = true;
			}
			else if (arg == "--screen-dedup-maprng")
			{
				args.screen_dedup = true;
				args.screen_dedup_maprng = true;
			}
			else if (arg == "--screen-dedup-maprng-preext")
			{
				args.screen_dedup = true;
				args.screen_dedup_maprng_preext = true;
			}
			else if (arg == "--screen-dedup-maprng-preext-1sync")
			{
				args.screen_dedup = true;
				args.screen_dedup_maprng_preext = true;  // uses same buffer
				args.screen_dedup_maprng_preext_1sync = true;
			}
			else if (arg == "--screen-dedup-maprng-preext-cstore")
			{
				args.screen_dedup = true;
				args.screen_dedup_maprng_preext = true;  // uses same buffer
				args.screen_dedup_maprng_preext_cstore = true;
			}
			else if (arg == "--screen-dedup-preext-policy" && i + 1 < argc)
			{
				args.screen_dedup = true;
				args.screen_dedup_maprng_preext = true;  // builds the 3-stream maprng buffer
				args.preext_policy = argv[++i];
				if (args.preext_policy != "first" && args.preext_policy != "k2" &&
				    args.preext_policy != "k3" && args.preext_policy != "k4" &&
				    args.preext_policy != "k5" && args.preext_policy != "k6" &&
				    args.preext_policy != "k8" && args.preext_policy != "output")
				{
					throw std::runtime_error("--screen-dedup-preext-policy must be first|k2|k3|k4|k5|k6|k8|output");
				}
			}
			else if (arg == "--compaction-bench")
			{
				args.compaction_bench = true;
				args.screen_dedup_offsets = true;  // build per-key offset-stream buffers
			}
			else if (arg == "--drain-bench")
			{
				// WS1 drop-drain POC (Path A). Reuses the --compaction-bench harness
				// (offset streams + ilp6 ref) and runs the map-6 drain triplet, returns.
				args.drain_bench = true;
				args.compaction_bench = true;
				args.screen_dedup_offsets = true;
			}
			else if (arg == "--drain-at-boundary")
			{
				// WS1 Path B: add a GLOBAL drain at each span boundary in this comma list
				// (e.g. "6" or "6,11,16"), inside the compaction pipeline (after span-0/MAP1
				// global merge). Requires --compaction-global-span0 and --compaction-spans
				// cuts at these indices. Prints per-boundary table occupancy.
				args.drain_boundaries = argv[++i];
			}
			else if (arg == "--drain-cap-bits")
			{
				// WS1 inverse-bloom cap: drains use a FIXED 2^bits-bucket * ways table (epoch-tagged,
				// no re-zero, short probe) instead of the 2x-tile exact table. Footprint flat in tile.
				args.drain_cap_bits = static_cast<uint32_t>(std::stoul(argv[++i]));
				args.drain_cap_bits_explicit = true;  // operator pinned the cap → disable --compaction-run cap-auto
			}
			else if (arg == "--drain-cap-ways")
			{
				args.drain_cap_ways = static_cast<uint32_t>(std::stoul(argv[++i]));
			}
			else if (arg == "--drain-cap-cas")
			{
				// CAS protocol for the inverse-bloom cap: 128 (default, atom.cas.b128 + epoch),
				// 64a (epoch-packed 64-bit CAS, no re-zero), 64b (two-array 64-bit CAS + re-zero
				// per boundary), 64c (K two-array tables, one per drain site, zeroed once).
				args.drain_cap_cas = argv[++i];
			}
			else if (arg == "--map1-route-tau")
			{
				// Span-0 MAP1 merge uses shed-proxy routing: hash (dedup) only states whose
				// shed count (alg0/alg6 ops in MAP1) >= tau; pass the low-shed rest un-hashed.
				args.map1_route_tau = static_cast<uint32_t>(std::stoul(argv[++i]));
			}
			else if (arg == "--map1-cap-bits") { args.map1_cap_bits = static_cast<uint32_t>(std::stoul(argv[++i])); }
			else if (arg == "--map1-cap-ways") { args.map1_cap_ways = static_cast<uint32_t>(std::stoul(argv[++i])); }
			else if (arg == "--map1-table-auto") { args.map1_table_auto = true; }
			else if (arg == "--map1-cap-auto") { args.map1_cap_auto = true; }
			else if (arg == "--drain-table-auto") { args.drain_table_auto = true; }
			else if (arg == "--drain-route-tau") { args.drain_route_tau = static_cast<uint32_t>(std::stoul(argv[++i])); }
			else if (arg == "--drain-route-traj") { args.drain_route_traj = true; }
			else if (arg == "--drain-sticky-tau") { args.drain_sticky_tau = static_cast<uint32_t>(std::stoul(argv[++i])); }
			else if (arg == "--drain-mult-tau")
			{
				args.drain_mult_tau = static_cast<uint32_t>(std::stoul(argv[++i]));
				args.compaction_multiplicity = true;
			}
			else if (arg == "--drain-traj-bits") { args.drain_traj_bits = static_cast<uint32_t>(std::stoul(argv[++i])); }
			else if (arg == "--drain-traj-2pass") { args.drain_traj_2pass = true; }
			else if (arg == "--drain-cross-tile")
			{
				// Persist the inverse-bloom cap ACROSS sweep tiles via a boundary-keyed epoch
				// (boundary m always uses epoch m), so a later window matches earlier windows'
				// entries → cross-window dedup. Concept test: per-tile flag-parity breaks by
				// design (a recurring hit is dropped after its first window); the signal is the
				// per-tile final frontier shrinking as the cache warms.
				args.drain_cross_tile = true;
			}
			else if (arg == "--compaction-spans" && i + 1 < argc)
			{
				args.compaction_bench = true;
				args.screen_dedup_offsets = true;
				args.compaction_spans = argv[++i];  // e.g. "0,1,5,9,13,17,21,25,27"
			}
			else if (arg == "--compaction-ilp" && i + 1 < argc)
			{
				args.compaction_bench = true;
				args.screen_dedup_offsets = true;
				args.compaction_ilp = argv[++i];  // span-kernel geometry, e.g. w4i6, w8i8
				args.compaction_ilp_explicit = true;  // explicit flag wins over the calibrated config
			}
			else if (arg == "--compaction-ordered")
			{
				args.compaction_bench = true;
				args.screen_dedup_offsets = true;
				args.compaction_ordered = true;
			}
			else if (arg == "--compaction-atomic")  // disable ordered compaction (A/B)
			{
				args.compaction_bench = true;
				args.screen_dedup_offsets = true;
				args.compaction_ordered = false;
			}
			else if (arg == "--compaction-final-hash")  // keep the after-last-map hash (A/B)
			{
				args.compaction_bench = true;
				args.screen_dedup_offsets = true;
				args.compaction_skip_final = false;
			}
			else if (arg == "--compaction-no-fuse")  // separate final_flag kernel (A/B)
			{
				args.compaction_bench = true;
				args.screen_dedup_offsets = true;
				args.compaction_fuse_final = false;
			}
			else if (arg == "--compaction-sweep")  // full 2^32 end-to-end sweep
			{
				args.compaction_bench = true;
				args.screen_dedup_offsets = true;
				args.compaction_sweep = true;
				if (args.parity_count == 0u) args.parity_count = 67108864u;  // default 64M tile
			}
			else if (arg == "--compaction-run")  // operational preset: settled host-appropriate full-key sweep
			{
				args.compaction_bench = true;
				args.screen_dedup_offsets = true;
				args.compaction_sweep = true;
				args.compaction_run = true;   // post-loop: apply the settled defaults where not overridden
			}
			else if (arg == "--compaction-auto-tile")  // size the tile from free VRAM
			{
				args.compaction_bench = true;
				args.screen_dedup_offsets = true;
				args.compaction_auto_tile = true;
				if (args.parity_count == 0u) args.parity_count = 1u;  // placeholder; replaced from free VRAM
			}
			else if (arg == "--compaction-global-span0")  // span-0 dedup via global VRAM hash table
			{
				args.compaction_bench = true;
				args.screen_dedup_offsets = true;
				args.compaction_global_span0 = true;
			}
			else if (arg == "--compaction-global-cap" && i + 1 < argc)  // window-cap (MB) for the span-0 global table
			{
				args.compaction_bench = true;
				args.screen_dedup_offsets = true;
				args.compaction_global_span0 = true;
				args.compaction_global_cap_mb = numeric_arg<uint32_t>(argv[++i], "--compaction-global-cap");
			}
			else if (arg == "--compaction-multiplicity")
			{
				args.compaction_bench = true;
				args.screen_dedup_offsets = true;
				args.compaction_multiplicity = true;
			}
			else if (arg == "--sweep-hit-file" && i + 1 < argc)  // dump every hitting (data,flag) from the 2^32 sweep
			{
				args.sweep_hit_file = argv[++i];
			}
			else if (arg == "--sweep-hit-print" && i + 1 < argc)  // inline-print the first N hits (0 = counts only)
			{
				args.sweep_hit_print = numeric_arg<uint32_t>(argv[++i], "--sweep-hit-print");
			}
			else if (arg == "--calibrate")  // measure host-optimal (research) screen-vs-compaction engine
			{
				args.calibrate = true;
				args.compaction_bench = true;
				args.screen_dedup_offsets = true;
				if (args.parity_count == 0u) args.parity_count = 1048576u;  // 1M/sample
				if (i + 1 < argc && argv[i + 1][0] != '-') args.config_path = argv[++i];
			}
			else if (arg == "--calibrate-raceway")  // PRODUCTION engine: sweep raceway span-ILP x cap-bits
			{
				// Enable the production raceway (== --raceway-direct-wave-continue-batch auto):
				// direct + offset-stream + VRAM-sized wave; the cadence defaults to the locked
				// 2,5,10,16 operating point below. The sweep itself sets span-ILP / cap-bits.
				args.calibrate_raceway = true;
				args.raceway_direct = true;
				args.raceway_direct_offset = true;
				args.raceway_wave_auto_size = true;
				args.raceway_direct_wave_continue_batch = 4194304u;
				if (i + 1 < argc && argv[i + 1][0] != '-') args.config_path = argv[++i];
			}
			else if (arg == "--config" && i + 1 < argc)
			{
				args.config_path = argv[++i];
			}
			else if (arg == "--screen-dedup-offset-geom" && i + 1 < argc)
			{
				args.screen_dedup = true;
				args.screen_dedup_offsets = true;  // builds the per-key offset-stream buffers
				args.offset_geom = argv[++i];
				{
					static const std::set<std::string> ok = {
						"ilp4w8_k4","ilp6w8_k4","ilp6w8_k1","ilp6w4_k4","ilp8w8_k4","ilp5w8_k4",
						"ilp10w8_k4","ilp12w8_k4","ilp8w8_k2","ilp8w8_k3","ilp8w8_k6","ilp8w16_k4"};
					if (ok.find(args.offset_geom) == ok.end())
						throw std::runtime_error("--screen-dedup-offset-geom: unknown geometry '" + args.offset_geom + "'");
				}
			}
			else if (arg == "--screen-dedup-offsets")
			{
				args.screen_dedup = true;
				args.screen_dedup_offsets = true;
			}
			else if (arg == "--screen-coalesced")
			{
				args.screen_coalesced = true;  // sets up its own A/B test (no parity flag needed)
			}
			else if (arg == "--screen-preext")
			{
				args.screen_preext = true;
				args.hll = false;  // maprng screen path does not update HLL
			}
			else if (arg == "--screen-offsets")
			{
				args.screen_offsets = true;
				args.hll = false;
			}
			else if (arg == "--screen-offsets-interleaved")
			{
				args.screen_offsets_interleaved = true;
				args.hll = false;
			}
			else if (arg == "--screen-offset-ilp-sweep")
			{
				args.screen_offset_ilp_sweep = true;
				args.hll = false;
			}
			else if (arg == "--byte-store")
			{
				args.byte_store = true;
			}
			else if (arg == "--wide-merge-dump")
			{
				args.wide_merge_dump = true;
			}
			else if (arg == "--wide-merge-first-maps" && i + 1 < argc)
			{
				args.wide_merge_first_maps = numeric_arg<uint32_t>(argv[++i], "--wide-merge-first-maps");
				if (args.wide_merge_first_maps < 1u || args.wide_merge_first_maps > 27u)
					throw std::runtime_error("--wide-merge-first-maps must be 1..27");
			}
				else if (arg == "--wide-merge-points" && i + 1 < argc)
				{
					args.wide_merge_points = argv[++i];
				}
				else if (arg == "--map1-frontier")
				{
					args.map1_frontier = true;
				}
					else if (arg == "--map1-frontier-axis-sweep")
					{
						args.map1_frontier = true;
						args.map1_frontier_axis_sweep = true;
					}
					else if (arg == "--map1-frontier-mobile-bit-sweep")
					{
						args.map1_frontier = true;
						args.map1_frontier_wide = true;
						args.map1_frontier_mobile_bit_sweep = true;
					}
					else if (arg == "--map1-frontier-backfill-sweep")
					{
						args.map1_frontier = true;
						args.map1_frontier_wide = true;
						args.map1_frontier_backfill_sweep = true;
						args.map1_frontier_mask_policy = "backfill";
					}
					else if (arg == "--map1-frontier-lowbits-sweep")
					{
						args.map1_frontier = true;
						args.map1_frontier_wide = true;
						args.map1_frontier_backfill_sweep = true;
						args.map1_frontier_lowbits_sweep = true;
						args.map1_frontier_mask_policy = "lowbits";
					}
					else if (arg == "--map1-frontier-squeeze-sweep")
					{
						args.map1_frontier = true;
						args.map1_frontier_wide = true;
						args.map1_frontier_backfill_sweep = true;
						args.map1_frontier_mask_policy = "squeeze";
					}
					else if (arg == "--map1-frontier-backfavor-sweep")
					{
						args.map1_frontier = true;
						args.map1_frontier_wide = true;
						args.map1_frontier_backfill_sweep = true;
						args.map1_frontier_mask_policy = "backfavor";
					}
					else if (arg == "--map1-frontier-highbits-sweep")
					{
						args.map1_frontier = true;
						args.map1_frontier_wide = true;
						args.map1_frontier_backfill_sweep = true;
						args.map1_frontier_mask_policy = "highbits";
					}
					else if (arg == "--map1-frontier-mask-policy" && i + 1 < argc)
					{
						args.map1_frontier = true;
						args.map1_frontier_wide = true;
						args.map1_frontier_backfill_sweep = true;
						args.map1_frontier_mask_policy = argv[++i];
						args.map1_frontier_lowbits_sweep = (args.map1_frontier_mask_policy == "lowbits");
					}
					else if (arg == "--map1-frontier-mobile-extra-bits" && i + 1 < argc)
					{
						args.map1_frontier = true;
						args.map1_frontier_wide = true;
						args.map1_frontier_mobile_bit_sweep = true;
						args.map1_frontier_mobile_extra_bits = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-mobile-extra-bits");
						if (args.map1_frontier_mobile_extra_bits < 1u || args.map1_frontier_mobile_extra_bits > 8u)
							throw std::runtime_error("--map1-frontier-mobile-extra-bits must be in 1..8");
					}
					else if (arg == "--map1-frontier-mobile-groups" && i + 1 < argc)
					{
						args.map1_frontier = true;
						args.map1_frontier_wide = true;
						args.map1_frontier_mobile_bit_sweep = true;
						args.map1_frontier_mobile_groups = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-mobile-groups");
						if (args.map1_frontier_mobile_groups == 0u)
							throw std::runtime_error("--map1-frontier-mobile-groups must be > 0");
					}
					else if (arg == "--map1-frontier-mobile-backfill")
					{
						args.map1_frontier = true;
						args.map1_frontier_wide = true;
						args.map1_frontier_mobile_bit_sweep = true;
						args.map1_frontier_mobile_backfill = true;
					}
					else if (arg == "--map1-frontier-backfill-bits" && i + 1 < argc)
					{
						args.map1_frontier = true;
						args.map1_frontier_wide = true;
						args.map1_frontier_backfill_sweep = true;
						args.map1_frontier_backfill_bits = argv[++i];
					}
				else if (arg == "--map1-frontier-table-mb" && i + 1 < argc)
				{
					args.map1_frontier = true;
					args.map1_frontier_table_mb = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-table-mb");
				}
				else if (arg == "--map1-frontier-table-auto")
				{
					args.map1_frontier = true;
					args.map1_table_auto = true;
				}
				else if (arg == "--map1-frontier-chunk" && i + 1 < argc)
				{
					args.map1_frontier = true;
					args.map1_frontier_chunk = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-chunk");
					if (args.map1_frontier_chunk == 0u) throw std::runtime_error("--map1-frontier-chunk must be > 0");
				}
				else if (arg == "--map1-frontier-consume")
				{
					args.map1_frontier = true;
					args.map1_frontier_consume = true;
				}
				else if (arg == "--map1-frontier-offset-only")
				{
					args.map1_frontier = true;
					args.map1_frontier_consume = true;
					args.map1_frontier_offset_only = true;
				}
				else if (arg == "--map1-frontier-rep-cap" && i + 1 < argc)
				{
					args.map1_frontier = true;
					args.map1_frontier_consume = true;
					args.map1_frontier_rep_cap = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-rep-cap");
				}
				else if (arg == "--map1-frontier-raceway")
				{
					args.map1_frontier = true;
					args.map1_frontier_raceway = true;
				}
				else if (arg == "--raceway-cap-bits" && i + 1 < argc)
				{
					args.map1_frontier = true;
					args.map1_frontier_raceway = true;
					args.raceway_cap_bits = numeric_arg<uint32_t>(argv[++i], "--raceway-cap-bits");
					if (args.raceway_cap_bits > 31u)
						throw std::runtime_error("--raceway-cap-bits must be <= 31");
				}
				else if (arg == "--raceway-cap-ways" && i + 1 < argc)
				{
					args.map1_frontier = true;
					args.map1_frontier_raceway = true;
					args.raceway_cap_ways = numeric_arg<uint32_t>(argv[++i], "--raceway-cap-ways");
					if (args.raceway_cap_ways == 0u)
						throw std::runtime_error("--raceway-cap-ways must be > 0");
				}
				else if (arg == "--raceway-cap-count" && i + 1 < argc)
				{
					args.map1_frontier = true;
					args.map1_frontier_raceway = true;
					args.raceway_cap_count = numeric_arg<uint32_t>(argv[++i], "--raceway-cap-count");
				}
				else if (arg == "--raceway-rep-cap" && i + 1 < argc)
				{
					args.map1_frontier = true;
					args.map1_frontier_raceway = true;
					args.map1_frontier_rep_cap = numeric_arg<uint32_t>(argv[++i], "--raceway-rep-cap");
				}
				else if (arg == "--raceway-flag-parity")
				{
					args.map1_frontier = true;
					args.map1_frontier_raceway = true;
					args.raceway_flag_parity = true;
				}
				else if (arg == "--raceway-drop-hist")
				{
					args.map1_frontier = true;
					args.map1_frontier_raceway = true;
					args.raceway_drop_hist = true;
				}
				else if (arg == "--raceway-flat-parity")
				{
					args.map1_frontier = true;
					args.map1_frontier_raceway = true;
					args.raceway_flat_parity = true;
					args.raceway_flag_parity = true;
				}
				else if (arg == "--raceway-direct")
				{
					args.raceway_direct = true;
				}
				else if (arg == "--raceway-direct-flat-parity")
				{
					args.raceway_direct = true;
					args.raceway_direct_flat_parity = true;
					args.raceway_cap_bits = 0u;
				}
				else if (arg == "--raceway-direct-offset")
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
				}
				else if (arg == "--raceway-direct-stream-batch" && i + 1 < argc)
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_stream_batch = numeric_arg<uint32_t>(argv[++i], "--raceway-direct-stream-batch");
					if (args.raceway_direct_stream_batch == 0u)
						throw std::runtime_error("--raceway-direct-stream-batch must be > 0");
				}
				else if (arg == "--raceway-direct-wave-continue-batch" && i + 1 < argc)
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					{
						std::string v = argv[++i];
						if (v == "auto")
						{
							// Auto-size the wave to free VRAM (downstream-host adjust). Placeholder >0 so the
							// wave-cadence mode triggers; the real size is computed after cuMemGetInfo.
							args.raceway_wave_auto_size = true;
							args.raceway_direct_wave_continue_batch = 4194304u;
						}
						else
						{
							args.raceway_direct_wave_continue_batch = numeric_arg<uint32_t>(v.c_str(), "--raceway-direct-wave-continue-batch");
							if (args.raceway_direct_wave_continue_batch == 0u)
								throw std::runtime_error("--raceway-direct-wave-continue-batch must be > 0 or 'auto'");
						}
					}
				}
				else if (arg == "--raceway-direct-wave-continue-ilp" && i + 1 < argc)
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_wave_continue_ilp = numeric_arg<uint32_t>(argv[++i], "--raceway-direct-wave-continue-ilp");
					if (args.raceway_direct_wave_continue_ilp != 4u
						&& args.raceway_direct_wave_continue_ilp != 6u
						&& args.raceway_direct_wave_continue_ilp != 8u)
						throw std::runtime_error("--raceway-direct-wave-continue-ilp must be 4, 6, or 8");
				}
				else if (arg == "--raceway-direct-wave-span-ilp" && i + 1 < argc)
				{
					args.raceway_direct_wave_span_ilp = numeric_arg<uint32_t>(argv[++i], "--raceway-direct-wave-span-ilp");
					args.raceway_span_ilp_explicit = true;
					if (args.raceway_direct_wave_span_ilp != 1u
						&& args.raceway_direct_wave_span_ilp != 3u
						&& args.raceway_direct_wave_span_ilp != 4u
						&& args.raceway_direct_wave_span_ilp != 5u
						&& args.raceway_direct_wave_span_ilp != 6u)
						throw std::runtime_error("--raceway-direct-wave-span-ilp must be 1, 3, 4, 5, or 6");
				}
				else if (arg == "--raceway-direct-wave-parity")
				{
					args.raceway_direct_wave_parity = true;
				}
				else if (arg == "--raceway-direct-wave-state")
				{
					args.raceway_direct_wave_state = true;
				}
				else if (arg == "--raceway-direct-wave-k-sweep")
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_wave_k_sweep = true;
					args.raceway_direct_wave_state = true;
				}
				else if (arg == "--raceway-direct-wave-k-list" && i + 1 < argc)
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_wave_k_sweep = true;
					args.raceway_direct_wave_state = true;
					args.raceway_direct_wave_k_list = argv[++i];
				}
				else if (arg == "--raceway-direct-wave-boundaries" && i + 1 < argc)
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_wave_state = true;
					args.raceway_direct_wave_boundaries = argv[++i];
				}
				else if (arg == "--raceway-direct-cap-mark")
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_cap_mark = true;
				}
				else if (arg == "--raceway-direct-cap-compact")
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_cap_mark = true;
					args.raceway_direct_cap_compact = true;
				}
				else if (arg == "--raceway-direct-cap-continue")
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_cap_mark = true;
					args.raceway_direct_cap_compact = true;
					args.raceway_direct_cap_continue = true;
				}
				else if (arg == "--raceway-direct-cap-state-continue")
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_cap_mark = true;
					args.raceway_direct_cap_compact = true;
					args.raceway_direct_cap_continue = true;
					args.raceway_direct_cap_state_continue = true;
				}
				else if (arg == "--raceway-direct-offset-flat-parity")
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_flat_parity = true;
					args.raceway_cap_bits = 0u;
				}
				else if (arg == "--raceway-direct-offset-ilp4")
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_ilp = 4u;
				}
				else if (arg == "--raceway-direct-offset-ilp4-flat-parity")
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_ilp = 4u;
					args.raceway_direct_flat_parity = true;
					args.raceway_cap_bits = 0u;
				}
				else if (arg == "--raceway-direct-offset-ilp6")
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_ilp = 6u;
				}
				else if (arg == "--raceway-direct-offset-ilp6-flat-parity")
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_ilp = 6u;
					args.raceway_direct_flat_parity = true;
					args.raceway_cap_bits = 0u;
				}
				else if (arg == "--raceway-direct-offset-ilp6-static-flat-parity")
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_ilp = 6u;
					args.raceway_direct_static = true;
					args.raceway_direct_flat_parity = true;
					args.raceway_cap_bits = 0u;
				}
				else if (arg == "--raceway-direct-offset-ilp8")
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_ilp = 8u;
				}
				else if (arg == "--raceway-direct-offset-ilp8-flat-parity")
				{
					args.raceway_direct = true;
					args.raceway_direct_offset = true;
					args.raceway_direct_ilp = 8u;
					args.raceway_direct_flat_parity = true;
					args.raceway_cap_bits = 0u;
				}
				else if (arg == "--raceway-first-cap-map" && i + 1 < argc)
				{
					args.map1_frontier = true;
					args.map1_frontier_raceway = true;
					args.raceway_first_cap_map = numeric_arg<uint32_t>(argv[++i], "--raceway-first-cap-map");
					if (args.raceway_first_cap_map > 26u)
						throw std::runtime_error("--raceway-first-cap-map must be <= 26");
				}
				else if (arg == "--map1-frontier-deep")
				{
					args.map1_frontier = true;
					args.map1_frontier_consume = true;
					args.map1_frontier_deep = true;
				}
				else if (arg == "--map1-frontier-deep-k" && i + 1 < argc)
				{
					args.map1_frontier = true;
					args.map1_frontier_consume = true;
					args.map1_frontier_deep = true;
					args.map1_frontier_deep_k = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-deep-k");
				}
				else if (arg == "--map1-frontier-stream-cuts" && i + 1 < argc)
				{
					// Explicit deep merge boundaries for the stream-deep producer (mirror the raceway
					// cadence, e.g. 2,4,8,14,20). Does NOT set map1_frontier_deep — stream-deep only.
					args.map1_frontier_stream_cuts = argv[++i];
				}
				else if (arg == "--map1-frontier-deep-k-sweep")
				{
					args.map1_frontier = true;
					args.map1_frontier_consume = true;
					args.map1_frontier_deep = true;
					args.map1_frontier_deep_k_sweep = true;
				}
				else if (arg == "--map1-frontier-deep-k-list" && i + 1 < argc)
				{
					args.map1_frontier = true;
					args.map1_frontier_consume = true;
					args.map1_frontier_deep = true;
					args.map1_frontier_deep_k_sweep = true;
					args.map1_frontier_deep_k_list = argv[++i];
				}
				else if (arg == "--map1-frontier-stream-deep")
				{
					args.map1_frontier = true;
					args.map1_frontier_stream_deep = true;
					args.map1_frontier_deep_k = 5u;
				}
				else if (arg == "--map1-frontier-drain-cap-bits" && i + 1 < argc)
				{
					args.map1_frontier_drain_cap_bits = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-drain-cap-bits");
				}
				else if (arg == "--map1-frontier-drain-cap-ways" && i + 1 < argc)
				{
					args.map1_frontier_drain_cap_ways = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-drain-cap-ways");
				}
				else if (arg == "--map1-frontier-drain-cap-distinct")
				{
					args.map1_frontier_drain_cap_distinct = true;
				}
				else if (arg == "--map1-frontier-drain-shed-tau" && i + 1 < argc)
				{
					args.map1_frontier_drain_shed_tau = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-drain-shed-tau");
				}
				else if (arg == "--map1-frontier-drain-shed-gate-tau" && i + 1 < argc)
				{
					args.map1_frontier_drain_shed_gate_tau = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-drain-shed-gate-tau");
				}
				else if (arg == "--map1-frontier-drain-fp-gate-log" && i + 1 < argc)
				{
					args.map1_frontier_drain_fp_gate_log = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-drain-fp-gate-log");
				}
				else if (arg == "--map1-frontier-drain-traj-bits" && i + 1 < argc)
				{
					args.map1_frontier_drain_traj_bits = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-drain-traj-bits");
				}
				else if (arg == "--map1-frontier-drain-dens-tau" && i + 1 < argc)
				{
					args.map1_frontier_drain_dens_tau = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-drain-dens-tau");
				}
				else if (arg == "--map1-frontier-drain-alg0-tau" && i + 1 < argc)
				{
					args.map1_frontier_drain_alg0_tau = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-drain-alg0-tau");
					args.map1_frontier_drain_alg0_tau_explicit = true;
				}
				else if (arg == "--map1-frontier-drain-mult-tau" && i + 1 < argc)
				{
					args.map1_frontier = true;
					args.map1_frontier_stream_deep = true;
					args.map1_frontier_multiplicity = true;
					args.map1_frontier_drain_mult_tau = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-drain-mult-tau");
				}
				else if (arg == "--map1-frontier-route-tau" && i + 1 < argc)
				{
					args.map1_frontier_route_tau = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-route-tau");
				}
				else if (arg == "--map1-frontier-multiplicity")
				{
					args.map1_frontier = true;
					args.map1_frontier_stream_deep = true;
					args.map1_frontier_multiplicity = true;
				}
				else if (arg == "--map1-frontier-partitions" && i + 1 < argc)
				{
					args.map1_frontier = true;
					args.map1_frontier_stream_deep = true;
					args.map1_frontier_deep_k = 5u;
					args.map1_frontier_partitions = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-partitions");
					if (args.map1_frontier_partitions < 1u || args.map1_frontier_partitions > 256u
						|| (args.map1_frontier_partitions & (args.map1_frontier_partitions - 1u)) != 0u)
						throw std::runtime_error("--map1-frontier-partitions must be a power of two in 1..256");
				}
				else if (arg == "--map1-frontier-wide")  // 128-bit fp + warp-cooperative bucketed insert
				{
					args.map1_frontier = true;
					args.map1_frontier_wide = true;
				}
				else if (arg == "--map1-frontier-wide-ilp" && i + 1 < argc)  // Target A: candidates/warp (1,2,4)
				{
					args.map1_frontier = true;
					args.map1_frontier_wide = true;
					args.map1_frontier_wide_ilp = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-wide-ilp");
					if (args.map1_frontier_wide_ilp != 1u && args.map1_frontier_wide_ilp != 2u && args.map1_frontier_wide_ilp != 4u)
						throw std::runtime_error("--map1-frontier-wide-ilp must be 1, 2, or 4");
				}
				else if (arg == "--map1-frontier-depth" && i + 1 < argc)  // dedup at MAP-K frontier (collapse-vs-depth)
				{
					args.map1_frontier = true;
					args.map1_frontier_wide = true;
					args.map1_frontier_depth = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-depth");
					if (args.map1_frontier_depth < 1u || args.map1_frontier_depth > 27u)
						throw std::runtime_error("--map1-frontier-depth must be in 1..27");
				}
				else if (arg == "--map1-frontier-sample-mult" && i + 1 < argc)  // quasi-random sample across 2^32 (class spot-check)
				{
					args.map1_frontier = true;
					args.map1_frontier_wide = true;
					args.map1_frontier_sample_mult = numeric_arg<uint32_t>(argv[++i], "--map1-frontier-sample-mult") | 1u;  // force odd
				}
				else if (arg == "--help" || arg == "-h")
				{
					std::cout
<< "test_cuda — CUDA forward-search / checksum-screen driver\n"
						<< "\n"
						<< "Core run:\n"
						<< "  --device <index>           CUDA device ordinal (may not match nvidia-smi order;\n"
						<< "                             the resolved device name is printed in the banner)\n"
						<< "  --key_id <uint32>          Fixed key to sweep (e.g. 0x2CA5B42D)\n"
						<< "  --range_start <uint64>     First data value in the sweep (default 0)\n"
						<< "  --workunit_size <uint64>   Data candidates to sweep (range_start+size <= 2^32)\n"
						<< "  --batch_size <uint32>      Candidates per kernel launch (default 1048576)\n"
						<< "  --warmup_batches <uint32>  Untimed warm-up launches before measurement\n"
						<< "  --map-list all|skip-car    Schedule variant: 27-entry ALL_MAPS or 26-entry SKIP_CAR (default: all)\n"
						<< "  --output_csv <path>        Write per-survivor records to CSV\n"
						<< "\n"
						<< "Validation:\n"
						<< "  --parity <count>           Cross-check <count> candidates against the CPU reference\n"
						<< "                             (also sets the tile size for the --compaction-* benches)\n"
						<< "\n"
						<< "Screen kernel selection (default is the universal-table baseline):\n"
						<< "  --screen-offsets           Per-key coalesced offset-stream ILP6 screen (PRODUCTION, fastest)\n"
						<< "  --screen-offsets-interleaved Schedule-grouped offset-stream variant (A/B)\n"
						<< "  --screen-offset-ilp-sweep  Benchmark offset-stream screen at ILP 1/2/4/8 (needs --parity)\n"
						<< "  --screen-preext            3-stream pre-extracted map_rng screen kernel (A/B)\n"
						<< "  --screen-coalesced         Coalesced-load screen A/B test (self-contained)\n"
						<< "  --byte-store               Original byte-store screen kernel (A/B profiling)\n"
						<< "  --maprng                   Per-launch 54 KB map_rng table (POC; forces HLL off)\n"
						<< "\n"
						<< "On-GPU VRAM compaction (~1.3-1.9x screen on high-R keys, parity-clean):\n"
						<< "  --compaction-bench         Run compaction vs the ilp6 screen at N=--parity\n"
						<< "  --compaction-sweep         Full 2^32 end-to-end sweep, tile-by-tile (tile = --parity, default 64M)\n"
						<< "  --compaction-ilp <wWiI>    Span-kernel geometry, e.g. w8i6 / w8i8 (overrides the\n"
						<< "                             per-device geom auto-loaded from the config; default w8i8)\n"
						<< "  --compaction-spans <list>  Custom span boundaries, e.g. 0,1,5,9,13,17,21,25,27\n"
						<< "  --compaction-ordered       Block-stable prefix-scan compaction (default on)\n"
						<< "  --compaction-atomic        Disable ordered compaction (atomic-append A/B)\n"
						<< "  --compaction-no-fuse       Separate final_flag kernel instead of the fused final screen (A/B)\n"
						<< "  --compaction-final-hash    Keep the after-last-map hash (A/B; default drops it)\n"
						<< "  --drain-mult-tau N         Multiplicity OR threshold for deep trajgate routing (enables sidecar)\n"
						<< "  --compaction-global-span0  Span-0 dedup via a GLOBAL VRAM hash table (captures the\n"
						<< "                             front-loaded map-1 global collapse the within-block table misses)\n"
						<< "  --compaction-global-cap M  Window-cap: max MB for the span-0 global table (0 = auto, 2x tile)\n"
						<< "  --compaction-multiplicity  Carry uint32 representative multiplicity in whole-tile compaction\n"
							<< "  --sweep-hit-file PATH      Hit-finder: write every hitting (data,flag) across the 2^32 sweep to PATH\n"
							<< "  --sweep-hit-print N        Inline-print the first N sweep hits (default 64; 0 = summary counts only)\n"
							<< "  --compaction-run           Operational full-key sweep: settled host-appropriate defaults (global-span0,\n"
							<< "                             64a cross-tile drains 2,4,8,14,20, map1-table-auto) + VRAM auto-tile. Overridable.\n"
							<< "  --compaction-auto-tile     Size the tile from free VRAM (cuMemGetInfo)\n"
							<< "\n"
							<< "Wide-merge W4B experiments (research path):\n"
								<< "  --wide-merge-dump          Dump/collapse map-K fingerprints, then screen survivors\n"
								<< "  --wide-merge-first-maps N  Merge point for the single wide merge (1..27, default 1)\n"
								<< "  --wide-merge-points LIST   Periodic merge points, e.g. 1,4,13; re-derive prototype\n"
							<< "  --map1-frontier           Persistent MAP1 representative hash-set prototype\n"
							<< "  --map1-frontier-axis-sweep  Full 2^32 as independent --workunit_size MAP-K windows\n"
								<< "  --map1-frontier-mobile-bit-sweep  Sweep position-local extra bits within the next data byte\n"
								<< "  --map1-frontier-mobile-extra-bits N  Extra coordinated bits beyond --workunit_size (default 1)\n"
								<< "  --map1-frontier-mobile-groups N  Sample N fixed-bit groups (default exact if <=65536 else 4096)\n"
								<< "  --map1-frontier-mobile-backfill  For N>1, fill remaining selected bits from the high end of that byte\n"
								<< "  --map1-frontier-backfill-sweep  Sweep arbitrary selected-bit windows built from each data byte's MSB downward\n"
								<< "  --map1-frontier-lowbits-sweep  Same selected-bit probe, but contiguous low bits for comparison\n"
								<< "  --map1-frontier-squeeze-sweep  Selected-bit probe with per-byte high/low squeeze fill\n"
								<< "  --map1-frontier-backfavor-sweep  Selected-bit probe with two high-side fills per low-side fill\n"
								<< "  --map1-frontier-highbits-sweep  Selected-bit probe with contiguous bits 31 downward\n"
								<< "  --map1-frontier-mask-policy P  Selected-bit policy: backfill, lowbits, squeeze, backfavor, highbits, omitbyte0..omitbyte7, auto\n"
								<< "  --map1-frontier-backfill-bits L  Comma list of selected-bit counts for backfill/lowbits sweeps\n"
								<< "  --map1-frontier-table-mb M  Cap the persistent table size (0 = auto)\n"
								<< "  --map1-frontier-table-auto  HLL-size the producer table to estimated frontier\n"
								<< "  --map1-frontier-chunk N     Candidates per streamed MAP1 chunk (default 4M)\n"
								<< "  --map1-frontier-consume     Emit reps and run indexed downstream screen\n"
								<< "  --map1-frontier-offset-only  Emit reps and time only the offset consumer\n"
								<< "  --map1-frontier-rep-cap N   Max representative data values to emit\n"
								<< "  --map1-frontier-raceway    Seed warp-persistent streaming POC from MAP1 reps\n"
								<< "  --raceway-rep-cap N        Max representative data values for raceway emission\n"
								<< "  --raceway-cap-bits B       Raceway cap buckets = 2^B (0 disables caps)\n"
								<< "  --raceway-cap-ways W       Raceway cap ways per bucket (default 4)\n"
								<< "  --raceway-cap-count N      Consecutive boundary caps (default 1)\n"
								<< "  --raceway-first-cap-map M  First probed completed map index (default 1 = after MAP2)\n"
								<< "  --raceway-flag-parity     No-cap raceway flags vs offset consumer by rep value\n"
								<< "  --raceway-drop-hist       Print per-map drop histogram for capped raceway runs\n"
								<< "  --raceway-flat-parity     Small-window no-cap parity + throughput vs flat screener\n"
								<< "  --raceway-direct          Full-window streaming forward; MAP1 calculation inline, no MAP1 dedup\n"
								<< "  --raceway-direct-flat-parity  No-cap direct raceway parity + throughput vs flat screener\n"
								<< "  --raceway-direct-offset   Direct raceway using per-key offset streams\n"
								<< "  --raceway-direct-stream-batch N  Bounded origination batches; caps persist across batches\n"
								<< "  --raceway-direct-wave-continue-batch N  Bounded mark+compact+continue waves\n"
								<< "  --raceway-direct-wave-continue-ilp N  State continuation ILP for wave mode: 4, 6, or 8 (default auto)\n"
								<< "  --raceway-direct-wave-span-ilp N  Cap-span/mark phase ILP for wave-state mode: 1 (default), 4, or 5\n"
								<< "  --raceway-direct-wave-parity  Alive-only flat parity for bounded wave continuation\n"
								<< "  --raceway-direct-wave-state  Bounded wave continuation resumes from saved boundary state\n"
								<< "  --raceway-direct-wave-k-sweep  Sweep f1k completed-map drain cadences (default K=2..8)\n"
								<< "  --raceway-direct-wave-k-list L  Comma list of K values for f1k cadence sweep\n"
								<< "  --raceway-direct-wave-boundaries L  Comma list of completed-map drains, e.g. 2,4,8,15,22\n"
								<< "  --raceway-direct-cap-mark  Offset-stream boundary-cap mark pass; writes alive/drop, no final screen\n"
								<< "  --raceway-direct-cap-compact  Cap-mark pass plus ordered survivor-index compaction\n"
								<< "  --raceway-direct-cap-continue  Cap compact plus recompute-screen survivors and alive-only parity\n"
								<< "  --raceway-direct-cap-state-continue  Cap compact plus saved-state survivor continuation\n"
								<< "  --raceway-direct-offset-flat-parity  Offset-stream direct raceway parity + throughput vs flat screener\n"
								<< "  --raceway-direct-offset-ilp4  Offset-stream direct raceway with 4 candidates per warp\n"
								<< "  --raceway-direct-offset-ilp4-flat-parity  ILP4 offset direct raceway parity + throughput vs flat screener\n"
								<< "  --raceway-direct-offset-ilp6  Offset-stream direct raceway with 6 candidates per warp\n"
								<< "  --raceway-direct-offset-ilp6-flat-parity  ILP6 offset direct raceway parity + throughput vs flat screener\n"
								<< "  --raceway-direct-offset-ilp6-static-flat-parity  Static-grid ILP6 offset direct raceway parity + throughput\n"
								<< "  --raceway-direct-offset-ilp8  Offset-stream direct raceway with 8 candidates per warp\n"
								<< "  --raceway-direct-offset-ilp8-flat-parity  ILP8 offset direct raceway parity + throughput vs flat screener\n"
								<< "  --map1-frontier-deep        Run local-frontier span compaction after MAP1 reps\n"
								<< "  --map1-frontier-deep-k K    Post-MAP1 merge cadence (default 4; K5 => 1,6,11,...)\n"
								<< "  --map1-frontier-stream-cuts L  Explicit deep merge boundaries for stream-deep, e.g. 2,4,8,14,20 (mirror raceway cadence)\n"
								<< "  --map1-frontier-deep-k-sweep Sweep default post-MAP1 K values\n"
								<< "  --map1-frontier-deep-k-list L Comma list for post-MAP1 K sweep, e.g. 3,4,5,6,8\n"
								<< "  --map1-frontier-stream-deep  W4B streamed post-MAP1 deep compaction\n"
								<< "  --map1-frontier-drain-cap-bits B  Pool B: persistent cross-chunk deep cap (2^B buckets)\n"
								<< "  --map1-frontier-drain-cap-ways W  ways per bucket for the producer deep cap (default 4)\n"
								<< "  --map1-frontier-drain-cap-distinct  Pool B: K distinct per-depth tables (no cross-depth eviction)\n"
								<< "  --map1-frontier-drain-shed-gate-tau N  Pool B shed-proxy cap gate; low-shed states pass alive\n"
								<< "  --map1-frontier-drain-fp-gate-log N  Pool B final-fp sampling gate; routes about 1/2^N states\n"
								<< "  --map1-frontier-drain-traj-bits B  Producer Pool B trajgate sketch bits (online, reset per boundary/chunk)\n"
								<< "  --map1-frontier-drain-dens-tau N  Producer Pool B trajgate density threshold (default 32)\n"
									<< "  --map1-frontier-drain-alg0-tau N  Producer Pool B sticky alg0 threshold (default 3 with traj bits; without traj bits, explicit N enables sticky-only; 0 disables)\n"
									<< "  --map1-frontier-drain-mult-tau N  Producer Pool B multiplicity OR threshold (enables multiplicity sidecar)\n"
									<< "  --map1-frontier-route-tau N  Measure routed cardinality (use with --map1-frontier-table-auto); Phase 2: routes insert\n"
									<< "  --map1-frontier-multiplicity  Track representative multiplicity through wide MAP1 stream-deep\n"
									<< "  --map1-frontier-partitions P  Prefix-partition MAP1 table (power of two, streamed-deep)\n"
							<< "  --map1-frontier-wide       128-bit fp + warp-cooperative bucketed insert (W4B-exact past ~600M F1; 16 B/slot)\n"
							<< "  --map1-frontier-wide-ilp K  Wide insert candidates/warp = 1, 2, or 4 (Target A: overlap probe latency)\n"
							<< "  --map1-frontier-depth K     Dedup at the MAP-K frontier (run K maps before fingerprint, K=1..27); collapse-vs-depth\n"
							<< "  --map1-frontier-sample-mult M  Quasi-random sample across 2^32 via odd Weyl mult M (class spot-check at small --workunit_size)\n"
							<< "\n"
							<< "Calibration / config (per-device engine+geometry selection):\n"
							<< "  NOTE: raceway is the production-default architecture (best on throughput AND\n"
							<< "        memory); the screen/compaction engines below are now research/A-B only.\n"
							<< "  --calibrate-raceway [path] PRODUCTION engine: sweep raceway span-ILP{1,3,4,5,6} x\n"
						<< "                             cap-bits{25,26,27 fit}, measure wave throughput across keys,\n"
						<< "                             write engine=raceway span_ilp=N. Production raceway runs\n"
						<< "                             auto-apply the calibrated span-ILP for this device.\n"
						<< "  --calibrate [path]         (research) Sweep the screen-vs-compaction span geometries,\n"
						<< "                             pick engine/geom/tile, write a fingerprint-keyed line to the\n"
						<< "                             config (default ./tm_compaction.conf)\n"
						<< "  --config <path>            Compaction config path; --compaction-bench/-sweep auto-load\n"
						<< "                             this device's calibrated geom from it (unless --compaction-ilp given)\n"
						<< "\n"
						<< "Within-block dedup experiments (study/A-B; high-R-only, generally below the screen):\n"
						<< "  --dedup                    Enable within-block state dedup\n"
						<< "  --dedup-w <32|64|128>      Dedup window / block width\n"
						<< "  --dedup-skip               Skip-mode dedup (default width 128)\n"
						<< "  --screen-dedup             Screen+dedup kernel\n"
						<< "  --screen-dedup-k <1|2|3|6> Dedup hash period (every K boundaries)\n"
						<< "  --screen-dedup-offsets     Screen+dedup on per-key coalesced offset streams\n"
						<< "  --screen-dedup-offset-geom <g>  Offset-dedup geometry (ilp8w8_k4, ilp5w8_k4, ...)\n"
						<< "  --screen-dedup-packed      Packed-flag store variant\n"
						<< "  --screen-dedup-fasthash    Cheaper hash variant\n"
						<< "  --screen-dedup-maprng      Dedup on the map_rng base\n"
						<< "  --screen-dedup-maprng-preext         Dedup on the 3-stream pre-extracted map_rng base\n"
						<< "  --screen-dedup-maprng-preext-1sync   ...single-sync variant\n"
						<< "  --screen-dedup-maprng-preext-cstore  ...constant-store variant\n"
						<< "  --screen-dedup-preext-policy <p>  Hash schedule: first|k2|k3|k4|k5|k6|k8|output\n"
						<< "\n"
						<< "HLL / profiling:\n"
						<< "  --hll                      Enable HyperLogLog output cardinality estimate (~10% slower; off by default)\n"
						<< "  --no-hll                   Back-compat no-op (HLL already off by default)\n"
						<< "  --dump-hll <path>          Write raw 4096x uint32 HLL registers to a binary file (implies --hll)\n";
				std::exit(0);
			}
			else
			{
				std::ostringstream message;
				message << "Unknown argument: " << arg;
				throw std::runtime_error(message.str());
			}
		}

		// Operational preset: --compaction-run = the settled host-appropriate full-key sweep config.
		// Applies the validated defaults only where the operator left the research default, so explicit
		// flags (any order) still win. Validated 8/8 PASS + 8GB-fit across collapse->hard-tail.
		if (args.compaction_run)
		{
			if (args.compaction_spans.empty())   args.compaction_spans = "0,1,2,4,8,14,20,27";
			if (args.drain_boundaries.empty())   args.drain_boundaries = "2,4,8,14,20";
			if (!args.compaction_global_span0)   args.compaction_global_span0 = true;
			if (!args.drain_cross_tile)          args.drain_cross_tile = true;
			if (!args.map1_table_auto)           args.map1_table_auto = true;
			if (args.drain_cap_cas == "128")     args.drain_cap_cas = "64a";
			if (args.drain_cap_bits == 0u)       args.drain_cap_bits = 22u;
			if (!args.compaction_auto_tile)      args.compaction_auto_tile = true;
			if (args.parity_count == 0u)         args.parity_count = 16777216u; // sweep gate needs >0; auto-tile overrides
		}
		if (args.batch_size == 0u)
		{
			throw std::runtime_error("--batch_size must be greater than zero");
		}
		if (args.workunit_size == 0u)
		{
			throw std::runtime_error("--workunit_size must be greater than zero");
		}
		if (args.range_start > static_cast<uint64_t>(UINT32_MAX))
		{
			throw std::runtime_error("--range_start must fit in uint32 for the CUDA kernel interface");
		}
		if (args.workunit_size > (1ull << 32))
		{
			throw std::runtime_error("--workunit_size must be at most 2^32 candidates");
		}
		if (args.range_start + args.workunit_size > (1ull << 32))
		{
			throw std::runtime_error("--range_start + --workunit_size exceeds the 32-bit candidate space");
		}
		if (args.drain_mult_tau > 0u && !args.drain_route_traj)
		{
			throw std::runtime_error("--drain-mult-tau currently augments --drain-route-traj; enable --drain-route-traj");
		}

		return args;
		}

	// ── CUDA utilities ───────────────────────────────────────────────────────────
	void check_cuda(CUresult status, const char* what)
	{
		if (status == CUDA_SUCCESS)
		{
			return;
		}

		const char* error_name = nullptr;
		const char* error_string = nullptr;
		cuGetErrorName(status, &error_name);
		cuGetErrorString(status, &error_string);

		std::ostringstream message;
		message << what << " failed with CUDA status " << static_cast<int>(status);
		if (error_name != nullptr)
		{
			message << " (" << error_name << ")";
		}
		if (error_string != nullptr)
		{
			message << ": " << error_string;
		}
		throw std::runtime_error(message.str());
	}

	std::vector<uint8_t> read_binary_file(const std::string& path)
	{
		std::ifstream input(path, std::ios::binary);
		if (!input)
		{
			return {};
		}

		std::ostringstream buffer;
		buffer << input.rdbuf();
		const std::string data = buffer.str();
		return std::vector<uint8_t>(data.begin(), data.end());
	}

	std::string executable_dir()
	{
		char path[PATH_MAX] = {};
		const ssize_t length = readlink("/proc/self/exe", path, sizeof(path) - 1);
		if (length <= 0)
		{
			return {};
		}
		path[length] = '\0';

		std::string value(path, path + length);
		const std::size_t slash = value.find_last_of("/");
		if (slash == std::string::npos)
		{
			return {};
		}
		return value.substr(0, slash);
	}

	std::vector<uint8_t> load_module_blob()
	{
		std::vector<std::string> paths = {
			"tm_cuda.fatbin",
			"test_cuda/tm_cuda.fatbin",
			"src/bruteforce/test_cuda/tm_cuda.fatbin"
		};

		const std::string exe_dir = executable_dir();
		if (!exe_dir.empty())
		{
			paths.push_back(exe_dir + "/tm_cuda.fatbin");
		}

		for (const std::string& path : paths)
		{
			std::vector<uint8_t> blob = read_binary_file(path);
			if (!blob.empty())
			{
				return blob;
			}
		}

		throw std::runtime_error("Could not locate tm_cuda.fatbin for test_cuda");
	}

	// Build the per-launch map_rng buffer for the maprng POC kernel.
	// Layout: [SCHEDULE_COUNT][2048] bytes; entry m holds bytes
	// (1..2048)-th raw run_rng output starting from schedule entry m's seed.
	std::vector<uint8_t> build_maprng_blob(const std::vector<uint8_t>& schedule_blob)
	{
		const size_t entry_count = schedule_blob.size() / 4;
		std::vector<uint8_t> mr(entry_count * 2048);
		for (size_t m = 0; m < entry_count; ++m)
		{
			uint16_t seed = (static_cast<uint16_t>(schedule_blob[m * 4 + 0]) << 8)
			              |  static_cast<uint16_t>(schedule_blob[m * 4 + 1]);
			uint8_t* p = mr.data() + m * 2048;
			for (int k = 0; k < 2048; ++k)
			{
				uint16_t next = RNG::rng_table[seed];
				p[k] = static_cast<uint8_t>(((next >> 8) ^ next) & 0xFFu);
				seed = next;
			}
		}
		return mr;
	}

	// Coalesced 3-stream maprng buffer. Same data as build_maprng_blob_preext
	// but laid out for single-uint32 lane reads (lane L reads uint32 at offset
	// 4L in each 128-byte segment). Eliminates the 4× byte-load and lane-strided
	// access pattern that ncu flagged as 39.7% load-coalesce-loss in the prior
	// pre-extracted kernel.
	//
	// Per-segment layout (128 bytes, one alg-call's worth):
	//   raw_coalesced[4*L + i]  = original_byte[127 - 4*L - i]   (matches pack_raw assembly)
	//   alg0_coalesced[4*L + i] = (original_byte[127 - 4*L - i] >> 7) & 1
	//   alg6_coalesced[4*L + i] = original_byte[4*L + i] & 0x80   (already forward; no reorder)
	std::vector<uint8_t> build_maprng_blob_preext_coalesced(const std::vector<uint8_t>& schedule_blob)
	{
		const size_t entry_count = schedule_blob.size() / 4;
		const size_t stream_size = entry_count * 2048;
		std::vector<uint8_t> mr(stream_size * 3);
		uint8_t* raw_stream  = mr.data() + 0;
		uint8_t* alg0_stream = mr.data() + stream_size;
		uint8_t* alg6_stream = mr.data() + stream_size * 2;
		for (size_t m = 0; m < entry_count; ++m)
		{
			uint16_t seed = (static_cast<uint16_t>(schedule_blob[m * 4 + 0]) << 8)
			              |  static_cast<uint16_t>(schedule_blob[m * 4 + 1]);
			// Generate the 2048 raw bytes for this schedule entry (16 alg-call segments × 128 bytes).
			uint8_t raw[2048];
			for (int k = 0; k < 2048; ++k)
			{
				uint16_t next = RNG::rng_table[seed];
				raw[k] = static_cast<uint8_t>(((next >> 8) ^ next) & 0xFFu);
				seed = next;
			}
			// Reorder into coalesced layout per 128-byte segment.
			for (int seg = 0; seg < 2048; seg += 128)
			{
				const uint8_t* src_seg = raw + seg;
				uint8_t* dst_raw  = raw_stream  + m * 2048 + seg;
				uint8_t* dst_alg0 = alg0_stream + m * 2048 + seg;
				uint8_t* dst_alg6 = alg6_stream + m * 2048 + seg;
				for (int L = 0; L < 32; L++)
				{
					for (int i = 0; i < 4; i++)
					{
						const int src_rev = 127 - 4 * L - i;
						const uint8_t b = src_seg[src_rev];
						dst_raw [4 * L + i] = b;
						dst_alg0[4 * L + i] = static_cast<uint8_t>((b >> 7) & 1u);
						// alg6 uses forward layout — pre-extract straight from
						// src_seg without reorder.
						dst_alg6[4 * L + i] = static_cast<uint8_t>(src_seg[4 * L + i] & 0x80u);
					}
				}
			}
		}
		return mr;
	}

	// Build a triple-stream maprng buffer: raw + alg0-extracted + alg6-extracted.
	// alg0 stream: each byte = (raw_byte >> 7) & 1 — pre-shifted MSB into LSB
	//              position so the kernel's pack assembly produces alg0's
	//              expected bit-pattern with no runtime extraction.
	// alg6 stream: each byte = raw_byte & 0x80 — pre-masked MSB-in-place so
	//              pack assembly produces alg6's bit-pattern.
	// Layout: [raw 54KB][alg0 54KB][alg6 54KB] = 162 KB. Stream bases are
	//   raw    = map_rng + 0
	//   alg0   = map_rng + entry_count * 2048
	//   alg6   = map_rng + entry_count * 2048 * 2
	std::vector<uint8_t> build_maprng_blob_preext(const std::vector<uint8_t>& schedule_blob)
	{
		const size_t entry_count = schedule_blob.size() / 4;
		const size_t stream_size = entry_count * 2048;
		std::vector<uint8_t> mr(stream_size * 3);
		uint8_t* raw_stream  = mr.data() + 0;
		uint8_t* alg0_stream = mr.data() + stream_size;
		uint8_t* alg6_stream = mr.data() + stream_size * 2;
		for (size_t m = 0; m < entry_count; ++m)
		{
			uint16_t seed = (static_cast<uint16_t>(schedule_blob[m * 4 + 0]) << 8)
			              |  static_cast<uint16_t>(schedule_blob[m * 4 + 1]);
			uint8_t* p_raw  = raw_stream  + m * 2048;
			uint8_t* p_alg0 = alg0_stream + m * 2048;
			uint8_t* p_alg6 = alg6_stream + m * 2048;
			for (int k = 0; k < 2048; ++k)
			{
				uint16_t next = RNG::rng_table[seed];
				const uint8_t raw = static_cast<uint8_t>(((next >> 8) ^ next) & 0xFFu);
				p_raw[k]  = raw;
				p_alg0[k] = static_cast<uint8_t>((raw >> 7) & 1u);
				p_alg6[k] = static_cast<uint8_t>(raw & 0x80u);
				seed = next;
			}
		}
		return mr;
	}

	struct OffsetStreamBlob
	{
		std::vector<uint8_t> data;
		std::size_t stream_bytes = 0;
		std::size_t carry_bytes = 0;
	};

	OffsetStreamBlob build_offset_stream_blob(const std::vector<uint8_t>& schedule_blob)
	{
		const size_t entry_count = schedule_blob.size() / 4;
		const size_t offset_count = entry_count * 2048ull;
		const size_t stream_bytes = offset_count * 128ull;
		const size_t carry_bytes = offset_count * sizeof(uint32_t);
		OffsetStreamBlob blob;
		blob.stream_bytes = stream_bytes;
		blob.carry_bytes = carry_bytes;
		blob.data.resize(stream_bytes * 3ull + carry_bytes * 2ull);

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
		return blob;
	}

	std::vector<uint8_t> build_offset_stream_blob_interleaved(const std::vector<uint8_t>& schedule_blob)
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

	std::vector<uint8_t> build_schedule_blob(uint32_t key, const std::string& map_list = "all")
	{
		const uint8_t* map_ptr = nullptr;
		std::size_t map_count = 0;
		if (map_list == "skip-car")
		{
			map_ptr   = kMapListSkipCar;
			map_count = sizeof(kMapListSkipCar) / sizeof(kMapListSkipCar[0]);
		}
		else
		{
			map_ptr   = kMapList;
			map_count = sizeof(kMapList) / sizeof(kMapList[0]);
		}

		key_schedule_data schedule_data = {};
		schedule_data.as_uint8[0] = static_cast<uint8_t>((key >> 24) & 0xFF);
		schedule_data.as_uint8[1] = static_cast<uint8_t>((key >> 16) & 0xFF);
		schedule_data.as_uint8[2] = static_cast<uint8_t>((key >> 8) & 0xFF);
		schedule_data.as_uint8[3] = static_cast<uint8_t>(key & 0xFF);

		// Maximum blob size: map_count + 1 (for 0x22 duplicate) entries × 4 bytes each
		std::vector<uint8_t> blob((map_count + 1) * 4, 0);
		int schedule_count = 0;
		for (std::size_t i = 0; i < map_count; i++)
		{
			key_schedule_entry entry = generate_schedule_entry(map_ptr[i], &schedule_data);
			blob[schedule_count * 4 + 0] = entry.rng1;
			blob[schedule_count * 4 + 1] = entry.rng2;
			blob[schedule_count * 4 + 2] = static_cast<uint8_t>((entry.nibble_selector >> 8) & 0xFF);
			blob[schedule_count * 4 + 3] = static_cast<uint8_t>(entry.nibble_selector & 0xFF);
			schedule_count++;

			if (map_ptr[i] == 0x22)
			{
				entry = generate_schedule_entry(map_ptr[i], &schedule_data, 4);
				blob[schedule_count * 4 + 0] = entry.rng1;
				blob[schedule_count * 4 + 1] = entry.rng2;
				blob[schedule_count * 4 + 2] = static_cast<uint8_t>((entry.nibble_selector >> 8) & 0xFF);
				blob[schedule_count * 4 + 3] = static_cast<uint8_t>(entry.nibble_selector & 0xFF);
				schedule_count++;
			}
		}

		blob.resize(static_cast<std::size_t>(schedule_count) * 4);
		return blob;
	}

	// ── CPU reference implementation ─────────────────────────────────────────────
	// Host-side forward pass used for parity testing (--parity <count>).
	// Mirrors the CUDA kernel logic exactly so results can be compared
	// bit-for-bit against the GPU output.

	// Run one algorithm step using host-side RNG tables (mirrors the CUDA kernel logic exactly)
	void cpu_run_alg(uint8_t alg_id, uint16_t* rng_seed, uint8_t* state)
	{
		if (alg_id == 0)
		{
			const uint8_t* tbl = RNG::alg0_values_8 + static_cast<uint32_t>(*rng_seed) * 128u;
			for (int i = 0; i < 128; i++)
				state[i] = static_cast<uint8_t>((state[i] << 1) | tbl[i]);
			*rng_seed = RNG::seed_forward_128[*rng_seed];
		}
		else if (alg_id == 1)
		{
			const uint8_t* tbl = RNG::regular_rng_values_8 + static_cast<uint32_t>(*rng_seed) * 128u;
			for (int i = 0; i < 128; i++)
				state[i] = static_cast<uint8_t>(state[i] + tbl[i]);
			*rng_seed = RNG::seed_forward_128[*rng_seed];
		}
		else if (alg_id == 2)
		{
			// CUDA uses alg2_values_32_8; CPU equivalent: initial carry = (alg2_values_32_8[seed] >> 24) & 1
			uint8_t carry = static_cast<uint8_t>((RNG::alg2_values_32_8[*rng_seed] >> 24) & 0x01u);
			for (int i = 127; i >= 0; i -= 2)
			{
				uint8_t next_carry = state[i - 1] & 0x01u;
				state[i - 1] = static_cast<uint8_t>((state[i - 1] >> 1) | (state[i] & 0x80u));
				state[i]     = static_cast<uint8_t>((state[i] << 1) | (carry & 0x01u));
				carry = next_carry;
			}
			*rng_seed = RNG::seed_forward_1[*rng_seed];
		}
		else if (alg_id == 3)
		{
			const uint8_t* tbl = RNG::regular_rng_values_8 + static_cast<uint32_t>(*rng_seed) * 128u;
			for (int i = 0; i < 128; i++)
				state[i] ^= tbl[i];
			*rng_seed = RNG::seed_forward_128[*rng_seed];
		}
		else if (alg_id == 4)
		{
			// CUDA: vsub4(value, regular_rng_values[seed][lane])
			const uint8_t* tbl = RNG::regular_rng_values_8 + static_cast<uint32_t>(*rng_seed) * 128u;
			for (int i = 0; i < 128; i++)
				state[i] = static_cast<uint8_t>(state[i] - tbl[i]);
			*rng_seed = RNG::seed_forward_128[*rng_seed];
		}
		else if (alg_id == 5)
		{
			// CUDA uses alg5_values_32_8; initial carry = (alg5_values_32_8[seed] >> 24) & 0x80
			uint8_t carry = static_cast<uint8_t>((RNG::alg5_values_32_8[*rng_seed] >> 24) & 0x80u);
			for (int i = 127; i >= 0; i -= 2)
			{
				uint8_t next_carry = state[i - 1] & 0x80u;
				state[i - 1] = static_cast<uint8_t>((state[i - 1] << 1) | (state[i] & 0x01u));
				state[i]     = static_cast<uint8_t>((state[i] >> 1) | carry);
				carry = next_carry;
			}
			*rng_seed = RNG::seed_forward_1[*rng_seed];
		}
		else if (alg_id == 6)
		{
			const uint8_t* tbl = RNG::alg6_values_8 + static_cast<uint32_t>(*rng_seed) * 128u;
			for (int i = 0; i < 128; i++)
				state[i] = static_cast<uint8_t>((state[i] >> 1) | tbl[i]);
			*rng_seed = RNG::seed_forward_128[*rng_seed];
		}
		else  // alg 7
		{
			for (int i = 0; i < 128; i++)
				state[i] ^= 0xFFu;
			// no rng_seed advance for alg 7
		}
	}

	// CPU forward: expand + run schedule, writes 128-byte post-schedule state to out[]
	void cpu_forward(uint32_t key, uint32_t data,
	                 const std::vector<uint8_t>& schedule_blob,
	                 uint8_t* out)
	{
		// Expand: interleave key/data bytes x16
		const uint16_t expansion_seed = static_cast<uint16_t>(key >> 16);
		for (int i = 0; i < 128; i += 8)
		{
			out[i+0] = static_cast<uint8_t>((key >> 24) & 0xFF);
			out[i+1] = static_cast<uint8_t>((key >> 16) & 0xFF);
			out[i+2] = static_cast<uint8_t>((key >>  8) & 0xFF);
			out[i+3] = static_cast<uint8_t>( key        & 0xFF);
			out[i+4] = static_cast<uint8_t>((data >> 24) & 0xFF);
			out[i+5] = static_cast<uint8_t>((data >> 16) & 0xFF);
			out[i+6] = static_cast<uint8_t>((data >>  8) & 0xFF);
			out[i+7] = static_cast<uint8_t>( data        & 0xFF);
		}
		const uint8_t* exp_tbl = RNG::expansion_values_8 + static_cast<uint32_t>(expansion_seed) * 128u;
		for (int i = 0; i < 128; i++)
			out[i] = static_cast<uint8_t>(out[i] + exp_tbl[i]);

		// Run schedule entries (count derived from blob size)
		const int schedule_count_cpu = static_cast<int>(schedule_blob.size() / 4);
		for (int s = 0; s < schedule_count_cpu; s++)
		{
			uint16_t rng_seed = static_cast<uint16_t>(
				(static_cast<uint16_t>(schedule_blob[s * 4 + 0]) << 8) |
				 static_cast<uint16_t>(schedule_blob[s * 4 + 1]));
			uint16_t nibble_selector = static_cast<uint16_t>(
				(static_cast<uint16_t>(schedule_blob[s * 4 + 2]) << 8) |
				 static_cast<uint16_t>(schedule_blob[s * 4 + 3]));

			for (int i = 0; i < 16; i++)
			{
				const uint8_t nibble = static_cast<uint8_t>((nibble_selector >> 15) & 0x01u);
				nibble_selector = static_cast<uint16_t>(nibble_selector << 1);

				uint8_t cur_byte = out[i];
				if (nibble != 0u)
					cur_byte = static_cast<uint8_t>(cur_byte >> 4);
				const uint8_t alg_id = static_cast<uint8_t>((cur_byte >> 1) & 0x07u);

				cpu_run_alg(alg_id, &rng_seed, out);
			}
		}
	}

	CUdeviceptr upload_buffer(const void* data, std::size_t size)
	{
		CUdeviceptr buffer = 0;
		check_cuda(cuMemAlloc(&buffer, size), "cuMemAlloc");
		check_cuda(cuMemcpyHtoD(buffer, data, size), "cuMemcpyHtoD");
		return buffer;
	}

	KernelAssets build_kernel_assets(uint32_t key, const std::string& map_list)
	{
		KernelAssets assets;
		RNG rng;
		rng.generate_regular_rng_values_8();
		rng.generate_alg0_values_8();
		rng.generate_alg6_values_8();
		rng.generate_seed_forward_1();
		rng.generate_seed_forward_128();
		rng.generate_alg2_values_32_8();
		rng.generate_alg5_values_32_8();
		rng.generate_expansion_values_8();

		assets.regular_rng_values = upload_buffer(RNG::regular_rng_values_8, 0x10000ull * 128ull);
		assets.alg0_values = upload_buffer(RNG::alg0_values_8, 0x10000ull * 128ull);
		assets.alg6_values = upload_buffer(RNG::alg6_values_8, 0x10000ull * 128ull);
		assets.rng_seed_forward_1 = upload_buffer(RNG::seed_forward_1, 0x10000ull * sizeof(uint16_t));
		assets.rng_seed_forward_128 = upload_buffer(RNG::seed_forward_128, 0x10000ull * sizeof(uint16_t));
		assets.alg2_values = upload_buffer(RNG::alg2_values_32_8, 0x10000ull * sizeof(uint32_t));
		assets.alg5_values = upload_buffer(RNG::alg5_values_32_8, 0x10000ull * sizeof(uint32_t));
		assets.expansion_values = upload_buffer(RNG::expansion_values_8, 0x10000ull * 128ull);

		const std::vector<uint8_t> schedule_blob = build_schedule_blob(key, map_list);
		assets.schedule_data = upload_buffer(schedule_blob.data(), schedule_blob.size());

		static const uint8_t carnival_data[128] = {
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x3D, 0x5E, 0xA1, 0xA6, 0xC8, 0x23,
			0xD7, 0x6E, 0x3F, 0x7C, 0xD2, 0x46, 0x1B, 0x9F, 0xAB, 0xD2,
			0x5C, 0x9B, 0x32, 0x43, 0x67, 0x30, 0xA0, 0xA4, 0x23, 0xF3,
			0x27, 0xBF, 0xEA, 0x21, 0x0F, 0x13, 0x31, 0x1A, 0x15, 0xA1,
			0x39, 0x34, 0xE4, 0xD2, 0x52, 0x6E, 0xA6, 0xF7, 0xF6, 0x43,
			0xD1, 0x28, 0x41, 0xD8, 0xDC, 0x55, 0xE1, 0xC5, 0x49, 0xF5,
			0xD4, 0x84, 0x52, 0x1F, 0x90, 0xAB, 0x26, 0xE4, 0x2A, 0xC3,
			0xC2, 0x59, 0xAC, 0x81, 0x58, 0x35, 0x7A, 0xC3, 0x51, 0x9A,
			0x01, 0x04, 0xF5, 0xE2, 0xFB, 0xA7, 0xAE, 0x8B, 0x46, 0x9A,
			0x27, 0x41, 0xFA, 0xDD, 0x63, 0x72, 0x23, 0x7E, 0x1B, 0x44,
			0x5A, 0x0B, 0x2A, 0x3C, 0x09, 0xFA, 0xA3, 0x59, 0x3C, 0xA1,
			0xF0, 0x90, 0x4F, 0x46, 0x9E, 0xD1, 0xD7, 0xF4
		};
		assets.carnival_data = upload_buffer(carnival_data, sizeof(carnival_data));

		return assets;
	}

	void release_assets(KernelAssets& assets)
	{
		if (assets.regular_rng_values != 0) cuMemFree(assets.regular_rng_values);
		if (assets.alg0_values != 0) cuMemFree(assets.alg0_values);
		if (assets.alg6_values != 0) cuMemFree(assets.alg6_values);
		if (assets.rng_seed_forward_1 != 0) cuMemFree(assets.rng_seed_forward_1);
		if (assets.rng_seed_forward_128 != 0) cuMemFree(assets.rng_seed_forward_128);
		if (assets.alg2_values != 0) cuMemFree(assets.alg2_values);
		if (assets.alg5_values != 0) cuMemFree(assets.alg5_values);
		if (assets.expansion_values != 0) cuMemFree(assets.expansion_values);
		if (assets.schedule_data != 0) cuMemFree(assets.schedule_data);
		if (assets.carnival_data != 0) cuMemFree(assets.carnival_data);
	}

	// ── Machine-code validation ──────────────────────────────────────────────────
	// Validates the 128-byte decrypted code region produced for each checksum
	// survivor.  Two complementary passes:
	//   1. check_machine_code  — entry-point reachability walk: follows the code
	//      region from known entry points, stopping at jumps/branches. Sets the
	//      FIRST_ENTRY_VALID and ALL_ENTRIES_VALID bits in the returned flags byte.
	//   2. linear_scan_clean / linear_scan_score — width-aware linear scan of the
	//      full region; any JAM/ILLEGAL opcode fails the scan.
	// evaluate_other_predicates checks specific byte positions expected by an
	// earlier reverse-engineering audit of the game's acceptance structure.

	int reverse_offset(int offset)
	{
		return 127 - offset;
	}

	uint8_t code_byte_at(const uint8_t* data, int offset)
	{
		return data[reverse_offset(offset)];
	}

	bool is_load_family(uint8_t value)
	{
		switch (value)
		{
			case 0xA0: case 0xA1: case 0xA2: case 0xA4: case 0xA5: case 0xA6:
			case 0xA9: case 0xAC: case 0xAD: case 0xAE: case 0xB1: case 0xB4:
			case 0xB5: case 0xB6: case 0xB9: case 0xBC: case 0xBD: case 0xBE:
				return true;
			default:
				return false;
		}
	}

	bool is_branch(uint8_t value)
	{
		switch (value)
		{
			case 0x10: case 0x30: case 0x50: case 0x70:
			case 0x90: case 0xB0: case 0xD0: case 0xF0:
				return true;
			default:
				return false;
		}
	}

	bool is_call_or_jump(uint8_t value)
	{
		return value == 0x20 || value == 0x4C;
	}

	bool bytes_at(const uint8_t* data, int offset, uint8_t a, uint8_t b, uint8_t c)
	{
		return code_byte_at(data, offset) == a &&
			code_byte_at(data, offset + 1) == b &&
			code_byte_at(data, offset + 2) == c;
	}

	struct OtherPredicateResults
	{
		bool values[OTHER_PREDICATE_COUNT] = {};
	};

	OtherPredicateResults evaluate_other_predicates(const uint8_t* data)
	{
		OtherPredicateResults result;
		result.values[OPRED_FINAL_RTS_50] = code_byte_at(data, 0x50) == 0x60;
		result.values[OPRED_ENTRY00_LOAD] = is_load_family(code_byte_at(data, 0x00));
		result.values[OPRED_ENTRY05_LOAD] = is_load_family(code_byte_at(data, 0x05));
		result.values[OPRED_ENTRY0A_LOAD] = is_load_family(code_byte_at(data, 0x0A));
		result.values[OPRED_ENTRY28_LOAD] = is_load_family(code_byte_at(data, 0x28));
		result.values[OPRED_ENTRY40_LOAD] = is_load_family(code_byte_at(data, 0x40));
		result.values[OPRED_INIT_CALL_OR_JUMP_02] = is_call_or_jump(code_byte_at(data, 0x02));
		result.values[OPRED_INIT_JMP_02] = code_byte_at(data, 0x02) == 0x4C;
		result.values[OPRED_CALL_OR_JUMP_07] = is_call_or_jump(code_byte_at(data, 0x07));
		result.values[OPRED_JSR_07] = code_byte_at(data, 0x07) == 0x20;
		result.values[OPRED_JUMP_25] = code_byte_at(data, 0x25) == 0x4C;
		result.values[OPRED_BRANCH_2D] = is_branch(code_byte_at(data, 0x2D));
		result.values[OPRED_JSR_3A] = code_byte_at(data, 0x3A) == 0x20;
		result.values[OPRED_JUMP_3D] = code_byte_at(data, 0x3D) == 0x4C;
		result.values[OPRED_BRANCH_45] = is_branch(code_byte_at(data, 0x45));
		result.values[OPRED_JUMP_4D] = code_byte_at(data, 0x4D) == 0x4C;
		result.values[OPRED_INIT_JMP_8595] = bytes_at(data, 0x02, 0x4C, 0x95, 0x85);
		result.values[OPRED_ENTRY05_JSR_80B4] = bytes_at(data, 0x07, 0x20, 0xB4, 0x80);
		result.values[OPRED_JUMP_25_TARGET_8464] = bytes_at(data, 0x25, 0x4C, 0x64, 0x84);
		result.values[OPRED_JSR_3A_TARGET_8952] = bytes_at(data, 0x3A, 0x20, 0x52, 0x89);
		result.values[OPRED_JUMP_3D_TARGET_81EE] = bytes_at(data, 0x3D, 0x4C, 0xEE, 0x81);
		result.values[OPRED_JUMP_4D_TARGET_80B4] = bytes_at(data, 0x4D, 0x4C, 0xB4, 0x80);
		return result;
	}

	bool other_final_rts_shape(const uint8_t* data)
	{
		return code_byte_at(data, 0x50) == 0x60;
	}

	bool other_entry_opcode_shape(const uint8_t* data)
	{
		return other_final_rts_shape(data)
			&& is_load_family(code_byte_at(data, 0x00))
			&& is_load_family(code_byte_at(data, 0x05))
			&& is_load_family(code_byte_at(data, 0x0A))
			&& is_load_family(code_byte_at(data, 0x28))
			&& is_load_family(code_byte_at(data, 0x40));
	}

	bool other_control_flow_shape(const uint8_t* data)
	{
		return other_final_rts_shape(data)
			&& is_call_or_jump(code_byte_at(data, 0x02))
			&& is_call_or_jump(code_byte_at(data, 0x07))
			&& code_byte_at(data, 0x25) == 0x4C
			&& is_branch(code_byte_at(data, 0x2D))
			&& code_byte_at(data, 0x3A) == 0x20
			&& code_byte_at(data, 0x3D) == 0x4C
			&& is_branch(code_byte_at(data, 0x45))
			&& code_byte_at(data, 0x4D) == 0x4C;
	}

	bool other_structural_shape(const uint8_t* data)
	{
		return other_entry_opcode_shape(data) && other_control_flow_shape(data);
	}

	uint8_t check_machine_code(const uint8_t* data, bool other_world)
	{
		const uint8_t code_length = other_world ? 0x53 : 0x72;
		uint8_t entry_addrs[7];
		int entry_count = 0;
		if (!other_world)
		{
			entry_count = 4;
			entry_addrs[0] = 0x00;
			entry_addrs[1] = 0x2B;
			entry_addrs[2] = 0x33;
			entry_addrs[3] = 0x3E;
			entry_addrs[4] = 0xFF;
			entry_addrs[5] = 0xFF;
			entry_addrs[6] = 0xFF;
		}
		else
		{
			entry_count = 6;
			entry_addrs[0] = 0x00;
			entry_addrs[1] = 0x05;
			entry_addrs[2] = 0x0A;
			entry_addrs[3] = 0x28;
			entry_addrs[4] = 0x40;
			entry_addrs[5] = 0x50;
			entry_addrs[6] = 0xFF;
		}

		uint8_t active_entries[7] = { 0, 0, 0, 0, 0, 0, 0 };
		uint8_t hit_entries[7] = { 0, 0, 0, 0, 0, 0, 0 };
		uint8_t valid_entries[7] = { 0, 0, 0, 0, 0, 0, 0 };
		int last_entry = -1;
		uint8_t result = 0;
		uint8_t next_entry_addr = entry_addrs[0];

		for (int i = 0; i < code_length - 2; i++)
		{
			if (i == next_entry_addr)
			{
				last_entry++;
				hit_entries[last_entry] = 1;
				active_entries[last_entry] = 1;
				next_entry_addr = entry_addrs[last_entry + 1];
			}
			else if (i > next_entry_addr)
			{
				last_entry++;
				next_entry_addr = entry_addrs[last_entry + 1];
			}

			const uint8_t opcode = data[reverse_offset(i)];
			if (kOpcodeType[opcode] & OP_JAM)
			{
				result |= USES_JAM;
				break;
			}
			else if (kOpcodeType[opcode] & OP_ILLEGAL)
			{
				result |= USES_ILLEGAL_OPCODES;
				break;
			}
			else if (kOpcodeType[opcode] & OP_NOP2)
			{
				result |= USES_UNOFFICIAL_NOPS;
			}
			else if (kOpcodeType[opcode] & OP_NOP)
			{
				result |= USES_NOP;
			}
			else if (kOpcodeType[opcode] & OP_JUMP)
			{
				for (int j = 0; j < entry_count; j++)
				{
					if (active_entries[j] == 1)
					{
						active_entries[j] = 0;
						valid_entries[j] = 1;
					}
				}
			}

			i += kOpcodeBytesUsed[opcode] - 1;
		}

		bool all_entries_valid = true;
		for (int i = 0; i < entry_count; i++)
		{
			if (hit_entries[i] == 1)
			{
				if (valid_entries[i] != 1)
				{
					all_entries_valid = false;
					break;
				}
			}
			else if (entry_addrs[i] != 0xFF)
			{
				for (int j = entry_addrs[i]; j < code_length - 2; j++)
				{
					const uint8_t opcode = data[reverse_offset(j)];
					if ((kOpcodeType[opcode] & OP_JAM) || (kOpcodeType[opcode] & OP_ILLEGAL))
					{
						all_entries_valid = false;
						break;
					}
					else if (kOpcodeType[opcode] & OP_JUMP)
					{
						break;
					}

					j += kOpcodeBytesUsed[opcode] - 1;
				}
			}
		}

		if (all_entries_valid)
		{
			result |= ALL_ENTRIES_VALID;
		}
		if (valid_entries[0] == 1)
		{
			result |= FIRST_ENTRY_VALID;
		}

		return result;
	}

	// Standard HyperLogLog cardinality estimator for m = 4096 registers.
	// Input: 4096 uint32_t registers, each holding the max rho seen for that bucket.
	// Output: estimated number of distinct elements.
	uint64_t hll_estimate(const std::vector<uint32_t>& regs)
	{
		static const int m = 4096;
		static const double alpha_m = 0.7213 / (1.0 + 1.079 / m);

		double sum = 0.0;
		for (int i = 0; i < m; i++)
		{
			sum += std::pow(2.0, -(double)regs[i]);
		}
		double estimate = alpha_m * (double)m * (double)m / sum;

		// Small-range correction (LinearCounting) — use when estimate ≤ 2.5 m
		if (estimate <= 2.5 * m)
		{
			int zeros = 0;
			for (int i = 0; i < m; i++) if (regs[i] == 0) zeros++;
			if (zeros > 0)
			{
				estimate = (double)m * std::log((double)m / (double)zeros);
			}
		}
		// NO large-range correction: hll_update (tm_cuda_primitives.cuh) builds registers from a 64-bit
		// hash (12-bit index + 52-bit rho via __clzll), so the hash space is 2^64, not 2^32. The classic
		// 32-bit correction -2^32*ln(1-E/2^32) is WRONG here and inflates badly as E -> 2^32: it turned an
		// accurate raw ~2.68B into 4.14B (+54%) at full-key, growing from ~+1.6% (raw) to +54% as cardinality
		// rose (the bug found 2026-06-13, #4 anomaly). For our cardinalities (<= 2^32 << 2^64) no large-range
		// correction is needed; the raw estimate is accurate to ~1.6% (m=4096). Keep only the small-range
		// LinearCounting above.
		return static_cast<uint64_t>(estimate + 0.5);
	}

	// Walk the entire code region as a linear instruction stream (multi-byte aware).
	// Returns true if the stream completes without encountering any JAM or ILLEGAL opcode.
	// Operand bytes are skipped according to instruction width, so only actual opcode
	// positions are tested — this is stricter than a raw per-byte scan and avoids
	// false positives from data bytes that happen to be invalid opcode values.
	bool linear_scan_clean(const uint8_t* data, bool other_world)
	{
		const int code_length = other_world ? 0x53 : 0x72;
		for (int i = 0; i < code_length - 2; )  // last 2 bytes are the embedded checksum
		{
			const uint8_t opcode = data[reverse_offset(i)];
			if ((kOpcodeType[opcode] & OP_JAM) || (kOpcodeType[opcode] & OP_ILLEGAL))
			{
				return false;
			}
			const int width = static_cast<int>(kOpcodeBytesUsed[opcode]);
			i += (width > 0) ? width : 1;  // safety: advance at least 1 even for 0-width entries
		}
		return true;
	}

	// Count opcode positions walked before hitting JAM/ILLEGAL, or total if clean.
	// Provides a gradient: a score near the stream length means almost-valid code;
	// a score of 0 means the very first byte is already a bad opcode.
	int linear_scan_score(const uint8_t* data, bool other_world)
	{
		const int code_length = other_world ? 0x53 : 0x72;
		int count = 0;
		for (int i = 0; i < code_length - 2; )
		{
			const uint8_t opcode = data[reverse_offset(i)];
			if ((kOpcodeType[opcode] & OP_JAM) || (kOpcodeType[opcode] & OP_ILLEGAL))
			{
				return count;
			}
			count++;
			const int width = static_cast<int>(kOpcodeBytesUsed[opcode]);
			i += (width > 0) ? width : 1;
		}
		return count;  // fully clean
	}

	// FNV-1a 64-bit hash over the full 128-byte decrypted state.
	uint64_t hash_state(const uint8_t* data)
	{
		uint64_t h = 14695981039346656037ULL;
		for (int i = 0; i < 128; i++)
		{
			h ^= data[i];
			h *= 1099511628211ULL;
		}
		return h;
	}

	void accumulate_validation_summary(ValidationSummary& summary, uint8_t machine_flags, bool other_world)
	{
		summary.total++;
		if (other_world)
		{
			summary.other++;
		}
		else
		{
			summary.carnival++;
		}
		if ((machine_flags & FIRST_ENTRY_VALID) != 0)
		{
			summary.first_entry_valid++;
		}
		if ((machine_flags & ALL_ENTRIES_VALID) != 0)
		{
			summary.all_entries_valid++;
			if (other_world) summary.all_entries_valid_other++;
			else             summary.all_entries_valid_carnival++;
		}
		if ((machine_flags & USES_NOP) != 0)
		{
			summary.uses_nop++;
		}
		if ((machine_flags & USES_UNOFFICIAL_NOPS) != 0)
		{
			summary.uses_unofficial_nops++;
		}
		if ((machine_flags & USES_ILLEGAL_OPCODES) != 0)
		{
			summary.uses_illegal++;
		}
		if ((machine_flags & USES_JAM) != 0)
		{
			summary.uses_jam++;
		}
	}

	// ── Output formatting ────────────────────────────────────────────────────────
	void print_summary(
		const Args& args,
		const std::string& device_name,
		double setup_seconds,
		double warmup_seconds,
		double screen_kernel_seconds,
		double materialize_kernel_seconds,
		double validation_seconds,
		double wall_seconds,
		uint64_t carnival_hits,
		uint64_t other_hits,
		uint64_t dual_hits,
		const ValidationSummary& validation_summary)
	{
		const uint64_t total_hits = carnival_hits + other_hits;
		const double candidate_count = static_cast<double>(args.workunit_size);
		const double kernel_rate = screen_kernel_seconds == 0.0 ? 0.0 : candidate_count / screen_kernel_seconds;
		const double materialize_rate = materialize_kernel_seconds == 0.0 ? 0.0 : static_cast<double>(total_hits) / materialize_kernel_seconds;
		const double wall_rate = wall_seconds == 0.0 ? 0.0 : candidate_count / wall_seconds;

		std::cout << "CUDA checksum-screen benchmark\n";
		std::cout << "  device: " << device_name << "\n";
		std::cout << "  map_list: " << args.map_list << "\n";
		std::cout << "  screen_store: "
		          << (args.maprng ? "maprng_byte" : (args.screen_preext ? "maprng_preext_packed_u32" : (args.screen_offsets_interleaved ? "offset_stream_interleaved_packed_u32" : (args.screen_offsets ? "offset_stream_per_byte_ilp6_preids" : (args.hll ? "hll_byte" : (args.byte_store ? "byte" : "packed_u32"))))))
		          << "\n";
		std::cout << "  key_id: " << args.key_id << "\n";
		std::cout << "  range_start: " << args.range_start << "\n";
		std::cout << "  workunit_size: " << args.workunit_size << "\n";
		std::cout << "  batch_size: " << args.batch_size << "\n";
		std::cout << "  setup_s: " << std::fixed << std::setprecision(3) << setup_seconds << "\n";
		std::cout << "  warmup_s: " << std::fixed << std::setprecision(3) << warmup_seconds << "\n";
		std::cout << "  screen_kernel_s: " << std::fixed << std::setprecision(3) << screen_kernel_seconds << "\n";
		std::cout << "  materialize_kernel_s: " << std::fixed << std::setprecision(3) << materialize_kernel_seconds << "\n";
		std::cout << "  cpu_validation_s: " << std::fixed << std::setprecision(3) << validation_seconds << "\n";
		std::cout << "  wall_s: " << std::fixed << std::setprecision(3) << wall_seconds << "\n";
		std::cout << "  screen_rate: " << std::fixed << std::setprecision(0) << kernel_rate << " candidates/s\n";
		if (total_hits > 0)
		{
			std::cout << "  materialize_rate: " << std::fixed << std::setprecision(0) << materialize_rate << " survivors/s\n";
		}
		std::cout << "  wall_rate: " << std::fixed << std::setprecision(0) << wall_rate << " candidates/s\n";
		std::cout << "  checksum survivors: " << total_hits << "\n";
		std::cout << "  checksum survivors carnival: " << carnival_hits << "\n";
		std::cout << "  checksum survivors other: " << other_hits << "\n";
		std::cout << "  checksum survivors dual: " << dual_hits << "\n";
		std::cout << "  validated survivors: " << validation_summary.total << "\n";
		std::cout << "  validated carnival: " << validation_summary.carnival << "\n";
		std::cout << "  validated other: " << validation_summary.other << "\n";
		std::cout << "  validated dual_pass: " << validation_summary.dual_pass << "\n";
		std::cout << "  validated first_entry_valid: " << validation_summary.first_entry_valid << "\n";
		std::cout << "  validated all_entries_valid: " << validation_summary.all_entries_valid << "\n";
		std::cout << "  validated all_entries_valid_carnival: " << validation_summary.all_entries_valid_carnival << "\n";
		std::cout << "  validated all_entries_valid_other: " << validation_summary.all_entries_valid_other << "\n";
		std::cout << "  validated linear_scan_clean_carnival: " << validation_summary.linear_scan_clean_carnival << "\n";
		std::cout << "  validated linear_scan_clean_other: " << validation_summary.linear_scan_clean_other << "\n";
		std::cout << "  validated linear_scan_score_max_carnival: " << validation_summary.linear_scan_score_max_carnival << "\n";
		std::cout << "  validated linear_scan_score_max_other: " << validation_summary.linear_scan_score_max_other << "\n";
		std::cout << "  validated other_final_rts: " << validation_summary.other_final_rts << "\n";
		std::cout << "  validated other_entry_opcodes: " << validation_summary.other_entry_opcodes << "\n";
		std::cout << "  validated other_control_flow: " << validation_summary.other_control_flow << "\n";
		std::cout << "  validated other_structural: " << validation_summary.other_structural << "\n";
		std::cout << "  validated all_entries_other_final_rts: " << validation_summary.all_entries_other_final_rts << "\n";
		std::cout << "  validated all_entries_other_entry_opcodes: " << validation_summary.all_entries_other_entry_opcodes << "\n";
		std::cout << "  validated all_entries_other_control_flow: " << validation_summary.all_entries_other_control_flow << "\n";
		std::cout << "  validated all_entries_other_structural: " << validation_summary.all_entries_other_structural << "\n";
		for (int p = 0; p < OTHER_PREDICATE_COUNT; p++)
		{
			std::cout << "  validated other_pred_" << kOtherPredicateNames[p] << ": "
			          << validation_summary.other_predicate[p] << "\n";
			std::cout << "  validated final_rts_other_pred_" << kOtherPredicateNames[p] << ": "
			          << validation_summary.final_rts_other_predicate[p] << "\n";
			std::cout << "  validated all_entries_other_pred_" << kOtherPredicateNames[p] << ": "
			          << validation_summary.all_entries_other_predicate[p] << "\n";
			std::cout << "  validated all_entries_final_rts_other_pred_" << kOtherPredicateNames[p] << ": "
			          << validation_summary.all_entries_final_rts_other_predicate[p] << "\n";
		}
		std::cout << "  validated unique_states_carnival: " << validation_summary.unique_states_carnival << "\n";
		std::cout << "  validated unique_states_other: " << validation_summary.unique_states_other << "\n";
		std::cout << "  validated uses_nop: " << validation_summary.uses_nop << "\n";
		std::cout << "  validated uses_unofficial_nops: " << validation_summary.uses_unofficial_nops << "\n";
		std::cout << "  validated uses_illegal: " << validation_summary.uses_illegal << "\n";
		std::cout << "  validated uses_jam: " << validation_summary.uses_jam << "\n";
		if (args.hll)
		{
			std::cout << "  hll_distinct_states: " << validation_summary.hll_distinct_states << "\n";
			if (validation_summary.hll_distinct_states > 0)
			{
				const double collision_factor = static_cast<double>(args.workunit_size)
				                              / static_cast<double>(validation_summary.hll_distinct_states);
				std::cout << "  hll_collision_factor: " << std::fixed << std::setprecision(2) << collision_factor << "\n";
			}
		}
		if (!args.output_csv_path.empty())
		{
			std::cout << "  output_csv: " << args.output_csv_path << "\n";
		}
	}

	// ── Kernel grid dimension helpers ────────────────────────────────────────────
	uint32_t state_kernel_grid_x(uint32_t candidate_count)
	{
		return (candidate_count + kCudaWarpsPerBlock - 1u) / kCudaWarpsPerBlock;
	}

	uint32_t screen_kernel_grid_x(uint32_t candidate_count)
	{
		const uint32_t candidates_per_block = kCudaWarpsPerBlock * kCudaScreenCandidatesPerWarp;
		return (candidate_count + candidates_per_block - 1u) / candidates_per_block;
	}

	uint32_t screen_kernel_grid_x_ilp(uint32_t candidate_count, uint32_t ilp)
	{
		const uint32_t candidates_per_block = kCudaWarpsPerBlock * ilp;
		return (candidate_count + candidates_per_block - 1u) / candidates_per_block;
	}
}

// ── Entry point ──────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
	try
	{
		Args args = parse_args(argc, argv);  // non-const: read-back may apply the calibrated compaction geom
		// --calibrate-raceway sweeps over a representative window per key (the default 1M is launch-
		// overhead-sensitive); force >=256M so the wave amortizes, unless the user asked for more.
		if (args.calibrate_raceway && args.workunit_size < 268435456ull) args.workunit_size = 268435456ull;  // 256M
		const auto setup_begin = std::chrono::high_resolution_clock::now();

		check_cuda(cuInit(0), "cuInit");

		int device_count = 0;
		check_cuda(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
		if (device_count == 0)
		{
			throw std::runtime_error("No CUDA devices found");
		}
		if (args.device_index >= static_cast<uint32_t>(device_count))
		{
			throw std::runtime_error("Requested --device is out of range");
		}

		CUdevice device = 0;
		check_cuda(cuDeviceGet(&device, static_cast<int>(args.device_index)), "cuDeviceGet");

		char device_name_buffer[256] = {};
		check_cuda(cuDeviceGetName(device_name_buffer, static_cast<int>(sizeof(device_name_buffer)), device), "cuDeviceGetName");
		const std::string device_name(device_name_buffer);

		CUcontext context = nullptr;
		check_cuda(cuCtxCreate(&context, nullptr, 0, device), "cuCtxCreate");

		const std::vector<uint8_t> module_blob = load_module_blob();
		CUmodule module = nullptr;
		check_cuda(cuModuleLoadData(&module, module_blob.data()), "cuModuleLoadData");

		// ---- Auto-apply the calibrated compaction geometry from tm_compaction.conf ----
		// --calibrate writes this config but historically nothing read it back, so a
		// production --compaction-bench/-sweep silently used the w8i8 default unless the
		// operator remembered to pass --compaction-ilp. Here we look up the running
		// device's fingerprint (same key --calibrate writes) and apply its geom. An
		// explicit --compaction-ilp always wins; an absent/unmatched entry leaves the
		// w8i8 default untouched. Skipped during --calibrate (which sweeps all geoms).
		if (args.compaction_bench && !args.calibrate)
		{
			size_t devmem_fp = 0; cuDeviceTotalMem(&devmem_fp, device);
			int sm_fp = 0; cuDeviceGetAttribute(&sm_fp, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
			std::ostringstream fp;
			// Memory bucketed to whole GB: total VRAM can be reported a few MB apart across
			// runs/driver states, and an exact-MB key would miss the calibrated line (must
			// match the --calibrate writer below).
			fp << "cuda|" << device_name_buffer << "|" << (devmem_fp >> 30) << "GB|" << sm_fp << "SM";
			std::string cal_engine, cal_geom, cal_tile;
			std::ifstream in(args.config_path);
			for (std::string l; std::getline(in, l); )
			{
				if (l.empty() || l[0] == '#') continue;
				const size_t tab = l.find('\t');
				if (tab == std::string::npos || l.substr(0, tab) != fp.str()) continue;
				std::istringstream fields(l.substr(tab + 1));
				for (std::string tok; fields >> tok; )
				{
					const size_t eq = tok.find('=');
					if (eq == std::string::npos) continue;
					const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
					if (k == "engine") cal_engine = v;
					else if (k == "geom") cal_geom = v;
					else if (k == "tile") cal_tile = v;
				}
				break;
			}
			if (cal_geom.empty())
				std::cout << "[compaction] no calibration for this device in " << args.config_path
				          << " — using default geom=" << args.compaction_ilp
				          << " (run --calibrate to tune)\n";
			else if (args.compaction_ilp_explicit)
				std::cout << "[compaction] calibrated geom=" << cal_geom << " (engine=" << cal_engine
				          << " tile=" << cal_tile << ") overridden by --compaction-ilp "
				          << args.compaction_ilp << "\n";
			else
			{
				args.compaction_ilp = cal_geom;
				std::cout << "[compaction] applied calibrated geom=" << cal_geom << " (engine=" << cal_engine
				          << " tile=" << cal_tile << ") for " << device_name << "\n";
				if (cal_engine == "screen")
					std::cout << "[compaction] note: calibration recommends the flat screen here"
					             " (compaction below break-even); the explicit --compaction-bench/-sweep still runs compaction\n";
			}
		}


		// ---- Apply the calibrated PRODUCTION raceway span-ILP from tm_compaction.conf ----
		// --calibrate-raceway records engine=raceway span_ilp=N per device; apply it here when running
		// the production raceway (wave-cadence), unless the user passed --raceway-direct-wave-span-ilp.
		// cap-bits stays VRAM-auto (the sweep showed it is throughput-marginal -- a memory-fit knob).
		if (args.raceway_direct && args.raceway_direct_wave_continue_batch != 0u
			&& !args.calibrate_raceway && !args.raceway_span_ilp_explicit)
		{
			size_t devmem_fp = 0; cuDeviceTotalMem(&devmem_fp, device);
			int sm_fp = 0; cuDeviceGetAttribute(&sm_fp, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
			std::ostringstream fp;
			fp << "cuda|" << device_name_buffer << "|" << (devmem_fp >> 30) << "GB|" << sm_fp << "SM";
			uint32_t cal_span = 0u;
			std::ifstream in(args.config_path);
			for (std::string l; std::getline(in, l); )
			{
				if (l.empty() || l[0] == '#') continue;
				const size_t tab = l.find('\t');
				if (tab == std::string::npos || l.substr(0, tab) != fp.str()) continue;
				if (l.find("engine=raceway") == std::string::npos) continue;
				const size_t sp = l.find("span_ilp=");
				if (sp != std::string::npos) cal_span = static_cast<uint32_t>(std::stoul(l.substr(sp + 9)));
				break;
			}
			if (cal_span == 1u || cal_span == 3u || cal_span == 4u || cal_span == 5u || cal_span == 6u)
			{
				args.raceway_direct_wave_span_ilp = cal_span;
				std::cout << "[raceway] applied calibrated span-ILP" << cal_span
				          << " for " << device_name << " (cap-bits stays VRAM-auto; run --calibrate-raceway to retune)\n";
			}
		}

		const bool use_skipcar = (args.map_list == "skip-car");
		const std::string ksuffix = use_skipcar ? "_skipcar" : "";
		if (args.screen_preext && use_skipcar)
		{
			throw std::runtime_error("--screen-preext currently supports --map-list all only");
		}
		if (args.screen_offsets && use_skipcar)
		{
			throw std::runtime_error("--screen-offsets currently supports --map-list all only");
		}
		if (args.screen_offsets_interleaved && use_skipcar)
		{
			throw std::runtime_error("--screen-offsets-interleaved currently supports --map-list all only");
		}
		if (args.screen_offset_ilp_sweep && use_skipcar)
		{
			throw std::runtime_error("--screen-offset-ilp-sweep currently supports --map-list all only");
		}
		if (args.screen_offset_ilp_sweep && args.parity_count == 0)
		{
			throw std::runtime_error("--screen-offset-ilp-sweep requires --parity <count>");
		}
		if (args.screen_dedup_offsets && use_skipcar)
		{
			throw std::runtime_error("--screen-dedup-offsets currently supports --map-list all only");
		}

		CUfunction screen_kernel_byte_store = nullptr;
		check_cuda(cuModuleGetFunction(&screen_kernel_byte_store, module, ("tm_checksum_screen_cuda" + ksuffix).c_str()), "cuModuleGetFunction(tm_checksum_screen_cuda)");
		CUfunction screen_kernel_packed_store = nullptr;
		check_cuda(cuModuleGetFunction(&screen_kernel_packed_store, module, ("tm_checksum_screen_packed_store_cuda" + ksuffix).c_str()), "cuModuleGetFunction(tm_checksum_screen_packed_store_cuda)");
		CUfunction screen_kernel = args.byte_store ? screen_kernel_byte_store : screen_kernel_packed_store;
		CUfunction screen_hll_kernel = nullptr;
		check_cuda(cuModuleGetFunction(&screen_hll_kernel, module, ("tm_screen_and_hll_cuda" + ksuffix).c_str()), "cuModuleGetFunction(tm_screen_and_hll_cuda)");
		CUfunction materialize_kernel = nullptr;
		check_cuda(cuModuleGetFunction(&materialize_kernel, module, ("tm_materialize_survivors_cuda" + ksuffix).c_str()), "cuModuleGetFunction(tm_materialize_survivors_cuda)");
		CUfunction dump_kernel = nullptr;
		check_cuda(cuModuleGetFunction(&dump_kernel, module, ("tm_dump_state_cuda" + ksuffix).c_str()), "cuModuleGetFunction(tm_dump_state_cuda)");
		CUfunction dump_dedup_w32_kernel = nullptr;
		check_cuda(cuModuleGetFunction(&dump_dedup_w32_kernel, module, ("tm_dump_state_dedup_w32_cuda" + ksuffix).c_str()), "cuModuleGetFunction(tm_dump_state_dedup_w32_cuda)");
		CUfunction dump_dedup_w128_kernel = nullptr;
		check_cuda(cuModuleGetFunction(&dump_dedup_w128_kernel, module, ("tm_dump_state_dedup_w128_cuda" + ksuffix).c_str()), "cuModuleGetFunction(tm_dump_state_dedup_w128_cuda)");
		CUfunction dump_dedup_w128_skip_kernel = nullptr;
		check_cuda(cuModuleGetFunction(&dump_dedup_w128_skip_kernel, module, ("tm_dump_state_dedup_w128_skip_cuda" + ksuffix).c_str()), "cuModuleGetFunction(tm_dump_state_dedup_w128_skip_cuda)");
		// Phase 2.5 block-size sweep variants. skipcar variant not yet added — fall back to default ksuffix only.
		CUfunction dump_dedup_w32_skip_kernel = nullptr;
		CUfunction dump_dedup_w64_skip_kernel = nullptr;
		CUfunction screen_dedup_w32_kernel = nullptr;
		CUfunction screen_dedup_w32_k2_kernel = nullptr;
		CUfunction screen_dedup_w32_k3_kernel = nullptr;
		CUfunction screen_dedup_w32_k6_kernel = nullptr;
		if (!use_skipcar)
		{
			check_cuda(cuModuleGetFunction(&dump_dedup_w32_skip_kernel, module, "tm_dump_state_dedup_w32_skip_cuda"), "cuModuleGetFunction(tm_dump_state_dedup_w32_skip_cuda)");
			check_cuda(cuModuleGetFunction(&dump_dedup_w64_skip_kernel, module, "tm_dump_state_dedup_w64_skip_cuda"), "cuModuleGetFunction(tm_dump_state_dedup_w64_skip_cuda)");
			check_cuda(cuModuleGetFunction(&screen_dedup_w32_kernel, module, "tm_checksum_screen_dedup_w32_cuda"), "cuModuleGetFunction(tm_checksum_screen_dedup_w32_cuda)");
			check_cuda(cuModuleGetFunction(&screen_dedup_w32_k2_kernel, module, "tm_checksum_screen_dedup_w32_k2_cuda"), "cuModuleGetFunction(tm_checksum_screen_dedup_w32_k2_cuda)");
			check_cuda(cuModuleGetFunction(&screen_dedup_w32_k3_kernel, module, "tm_checksum_screen_dedup_w32_k3_cuda"), "cuModuleGetFunction(tm_checksum_screen_dedup_w32_k3_cuda)");
			check_cuda(cuModuleGetFunction(&screen_dedup_w32_k6_kernel, module, "tm_checksum_screen_dedup_w32_k6_cuda"), "cuModuleGetFunction(tm_checksum_screen_dedup_w32_k6_cuda)");
		}
		CUfunction screen_kernel_256 = nullptr;
		CUfunction screen_dedup_w32_packed_kernel = nullptr;
		CUfunction screen_dedup_w32_fasthash_kernel = nullptr;
		CUfunction screen_dedup_w32_maprng_kernel = nullptr;
		CUfunction screen_dedup_w32_maprng_preext_kernel = nullptr;
		CUfunction screen_dedup_w32_maprng_preext_1sync_kernel = nullptr;
		CUfunction screen_dedup_w32_maprng_preext_cstore_kernel = nullptr;
		CUfunction preext_policy_kernel = nullptr;  // selected dedup-schedule experiment kernel
		CUfunction offset_geom_kernel = nullptr;    // selected ILP-geometry offset dedup kernel
		uint32_t offset_geom_warps = 8u, offset_geom_ilp = 4u;  // W = warps*ilp candidates/block
		CUfunction screen_dedup_w32_offset_kernel = nullptr;
		CUfunction run_span_dedup_kernel = nullptr;
		CUfunction run_span_dedup_global_kernel = nullptr;  // span-0 global-table dedup (compaction_global_span0)
		CUfunction run_span_dedup_drain_kernel = nullptr;   // WS1 Path B: GLOBAL drain at a deep span boundary
		CUfunction count_nonzero_kernel = nullptr;          // counts occupied global-table slots (drain footprint)
		CUfunction count_cap_epoch_kernel = nullptr;        // counts active slots in epoch-tagged cap tables
		CUfunction count_slots16_any_kernel = nullptr;      // counts occupied 16-byte slots regardless of epoch
		CUfunction run_span_dedup_drain_cap_kernel = nullptr; // WS1 inverse-bloom cap drain (fixed-capacity, epoch)
		CUfunction cap_resolve_xtile_kernel = nullptr;        // WS1 cross-tile FN-fix: end-of-tile flag fill
		CUfunction run_span0_routed_cap_kernel = nullptr;     // WS1 shed-proxy-routed span-0 MAP1 merge cap
		CUfunction run_span_deep_trajroute_cap_kernel = nullptr; // WS1 trajgate deep-routing (op-tail count-min + sticky)
		CUfunction run_span_deep_trajbuild_kernel = nullptr;     // WS1 trajgate two-pass: pass1 (build sketch + keys)
		CUfunction run_span_deep_trajroute2_kernel = nullptr;    // WS1 trajgate two-pass: pass2 (probe complete sketch + route)
		CUfunction hash_zero_kernel = nullptr;              // zeros the span-0 global table
		uint32_t comp_warps = 8u, comp_ilp = 8u;  // span-kernel geometry; W = warps*ilp
		CUfunction compact_survivors_kernel = nullptr;
		CUfunction run_span_hll_kernel = nullptr;  // HLL-auto table sizer: map1 HLL → frontier estimate
		CUfunction final_flag_survivors_kernel = nullptr;
		CUfunction resolve_flags_kernel = nullptr;
		CUfunction screen_maprng_coalesced_kernel = nullptr;
		CUfunction screen_maprng_preext_kernel = nullptr;
		CUfunction screen_offset_kernel = nullptr;
		CUfunction screen_offset_ilp1_kernel = nullptr;
		CUfunction screen_offset_ilp2_kernel = nullptr;
		CUfunction screen_offset_ilp8_kernel = nullptr;
		CUfunction screen_offset_ilp12_kernel = nullptr;
		CUfunction screen_offset_ilp16_kernel = nullptr;
		CUfunction screen_offset_ilp8_fixed_kernel = nullptr;
		CUfunction screen_offset_ilp8_preids_kernel = nullptr;
		CUfunction screen_offset_ilp12_preids_kernel = nullptr;
		CUfunction screen_offset_ilp16_preids_kernel = nullptr;
		CUfunction screen_offset_ilp8_preids_carrysel_kernel = nullptr;
		CUfunction screen_offset_ilp8_fixed_preids_kernel = nullptr;
		CUfunction screen_offset_ilp8_preids_unroll1_kernel = nullptr;
		CUfunction screen_offset_ilp6_preids_kernel = nullptr;
		CUfunction screen_offset_ilp6_preids_unroll1_kernel = nullptr;
		CUfunction screen_offset_ilp5_preids_kernel = nullptr;
		CUfunction screen_offset_ilp5_preids_unroll1_kernel = nullptr;
		CUfunction screen_offset_ilp6_preids_ldg_kernel = nullptr;
		CUfunction screen_offset_ilp8_preids_ldg_kernel = nullptr;
		CUfunction screen_offset_ilp5_preids_prefetch_kernel = nullptr;
		CUfunction screen_offset_ilp6_preids_prefetch_kernel = nullptr;
		CUfunction screen_offset_ilp8_preids_prefetch_kernel = nullptr;
		CUfunction screen_offset_interleaved_kernel = nullptr;
		check_cuda(cuModuleGetFunction(&count_slots16_any_kernel, module, "count_slots16_any_cuda"), "cuModuleGetFunction(count_slots16_any_cuda)");
		if (!use_skipcar)
		{
			check_cuda(cuModuleGetFunction(&screen_maprng_coalesced_kernel, module, "tm_checksum_screen_maprng_coalesced_cuda"), "cuModuleGetFunction(tm_checksum_screen_maprng_coalesced_cuda)");
			check_cuda(cuModuleGetFunction(&screen_maprng_preext_kernel, module, "tm_checksum_screen_maprng_preext_cuda"), "cuModuleGetFunction(tm_checksum_screen_maprng_preext_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_kernel, module, "tm_checksum_screen_offset_store_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp1_kernel, module, "tm_checksum_screen_offset_store_ilp1_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp1_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp2_kernel, module, "tm_checksum_screen_offset_store_ilp2_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp2_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp8_kernel, module, "tm_checksum_screen_offset_store_ilp8_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp8_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp12_kernel, module, "tm_checksum_screen_offset_store_ilp12_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp12_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp16_kernel, module, "tm_checksum_screen_offset_store_ilp16_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp16_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp8_fixed_kernel, module, "tm_checksum_screen_offset_store_ilp8_fixed_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp8_fixed_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp8_preids_kernel, module, "tm_checksum_screen_offset_store_ilp8_preids_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp8_preids_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp12_preids_kernel, module, "tm_checksum_screen_offset_store_ilp12_preids_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp12_preids_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp16_preids_kernel, module, "tm_checksum_screen_offset_store_ilp16_preids_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp16_preids_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp8_preids_carrysel_kernel, module, "tm_checksum_screen_offset_store_ilp8_preids_carrysel_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp8_preids_carrysel_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp8_fixed_preids_kernel, module, "tm_checksum_screen_offset_store_ilp8_fixed_preids_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp8_fixed_preids_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp8_preids_unroll1_kernel, module, "tm_checksum_screen_offset_store_ilp8_preids_unroll1_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp8_preids_unroll1_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp6_preids_kernel, module, "tm_checksum_screen_offset_store_ilp6_preids_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp6_preids_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp6_preids_unroll1_kernel, module, "tm_checksum_screen_offset_store_ilp6_preids_unroll1_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp6_preids_unroll1_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp5_preids_kernel, module, "tm_checksum_screen_offset_store_ilp5_preids_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp5_preids_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp5_preids_unroll1_kernel, module, "tm_checksum_screen_offset_store_ilp5_preids_unroll1_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp5_preids_unroll1_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp6_preids_ldg_kernel, module, "tm_checksum_screen_offset_store_ilp6_preids_ldg_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp6_preids_ldg_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp8_preids_ldg_kernel, module, "tm_checksum_screen_offset_store_ilp8_preids_ldg_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp8_preids_ldg_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp5_preids_prefetch_kernel, module, "tm_checksum_screen_offset_store_ilp5_preids_prefetch_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp5_preids_prefetch_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp6_preids_prefetch_kernel, module, "tm_checksum_screen_offset_store_ilp6_preids_prefetch_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp6_preids_prefetch_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_ilp8_preids_prefetch_kernel, module, "tm_checksum_screen_offset_store_ilp8_preids_prefetch_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_store_ilp8_preids_prefetch_cuda)");
			check_cuda(cuModuleGetFunction(&screen_offset_interleaved_kernel, module, "tm_checksum_screen_offset_interleaved_store_cuda"), "cuModuleGetFunction(tm_checksum_screen_offset_interleaved_store_cuda)");
			check_cuda(cuModuleGetFunction(&screen_kernel_256, module, "tm_checksum_screen_cuda_256thr"), "cuModuleGetFunction(tm_checksum_screen_cuda_256thr)");
			if (args.compaction_bench)
			{
				// geometry string "wWi I" → comp_warps, comp_ilp; kernel run_span_dedup_<geom>_cuda
				const std::string& g = args.compaction_ilp;
				const size_t ip = g.find('i');
				if (g.empty() || g[0] != 'w' || ip == std::string::npos)
					throw std::runtime_error("--compaction-ilp must look like w8i8");
				comp_warps = static_cast<uint32_t>(std::stoul(g.substr(1, ip - 1)));
				comp_ilp   = static_cast<uint32_t>(std::stoul(g.substr(ip + 1)));
				const std::string fn = "run_span_dedup_" + g + "_cuda";
				check_cuda(cuModuleGetFunction(&run_span_dedup_kernel, module, fn.c_str()), ("cuModuleGetFunction(" + fn + ")").c_str());
				check_cuda(cuModuleGetFunction(&compact_survivors_kernel, module,
					args.compaction_ordered ? "compact_survivors_ordered_cuda" : "compact_survivors_cuda"),
					"cuModuleGetFunction(compact_survivors)");
				check_cuda(cuModuleGetFunction(&final_flag_survivors_kernel, module, "final_flag_survivors_cuda"), "cuModuleGetFunction(final_flag_survivors_cuda)");
				check_cuda(cuModuleGetFunction(&resolve_flags_kernel, module, "resolve_flags_cuda"), "cuModuleGetFunction(resolve_flags_cuda)");
				if (args.compaction_global_span0)
				{
					const std::string gfn = "run_span_dedup_global_" + g + "_cuda";
					check_cuda(cuModuleGetFunction(&run_span_dedup_global_kernel, module, gfn.c_str()), ("cuModuleGetFunction(" + gfn + ")").c_str());
					if (args.map1_table_auto || args.map1_cap_auto)
						check_cuda(cuModuleGetFunction(&run_span_hll_kernel, module, "run_span_hll_w8i8_cuda"), "cuModuleGetFunction(run_span_hll_w8i8_cuda)");
					check_cuda(cuModuleGetFunction(&hash_zero_kernel, module, "tm_wide_merge_hash_zero_cuda"), "cuModuleGetFunction(hash_zero)");
					check_cuda(cuModuleGetFunction(&count_nonzero_kernel, module, "count_nonzero_u64_cuda"), "cuModuleGetFunction(count_nonzero_u64_cuda)");
					check_cuda(cuModuleGetFunction(&count_cap_epoch_kernel, module, "count_cap_slots_epoch_cuda"), "cuModuleGetFunction(count_cap_slots_epoch_cuda)");
					if (!args.drain_boundaries.empty() || args.map1_route_tau > 0u)
					{
						const std::string dfn = "run_span_dedup_drain_" + g + "_cuda";
						check_cuda(cuModuleGetFunction(&run_span_dedup_drain_kernel, module, dfn.c_str()), ("cuModuleGetFunction(" + dfn + ")").c_str());
						// cap CAS-scheme suffix: ""=128(atom.cas.b128), "a"=64a(epoch-packed 64-bit),
						// "b"=64b/64c(two-array 64-bit + re-zero). 64c reuses the 64b kernel; only the
						// host table orchestration (K tables vs re-zero) differs.
						// drain_cross_tile uses the FN-safe cross-tile cap (flag-carrying, "x" kernel)
						// + an end-of-tile resolve pass; it overrides the CAS-scheme suffix.
						const std::string cap_sfx = args.drain_cross_tile ? "x"
						                           : (args.drain_cap_cas == "64a") ? "a"
						                           : (args.drain_cap_cas == "64b" || args.drain_cap_cas == "64c") ? "b"
						                           : "";
						if (args.drain_cap_bits > 0u)
						{
							const std::string cfn = "run_span_dedup_drain_cap" + cap_sfx + "_" + g + "_cuda";
							check_cuda(cuModuleGetFunction(&run_span_dedup_drain_cap_kernel, module, cfn.c_str()), ("cuModuleGetFunction(" + cfn + ")").c_str());
							if (args.drain_cross_tile)
								check_cuda(cuModuleGetFunction(&cap_resolve_xtile_kernel, module, "cap_resolve_xtile_cuda"), "cuModuleGetFunction(cap_resolve_xtile_cuda)");
						}
						if (args.map1_route_tau > 0u || args.drain_route_tau > 0u)
						{
							const std::string rfn = "run_span0_routed_cap" + cap_sfx + "_" + g + "_cuda";
							check_cuda(cuModuleGetFunction(&run_span0_routed_cap_kernel, module, rfn.c_str()), ("cuModuleGetFunction(" + rfn + ")").c_str());
						}
						if (args.drain_route_traj)
						{
							// trajgate deep-routing (op-tail count-min + sticky alg0); single-table 64a cap only.
							const std::string tfn = "run_span_deep_trajroute_cap" + cap_sfx + "_" + g + "_cuda";
							check_cuda(cuModuleGetFunction(&run_span_deep_trajroute_cap_kernel, module, tfn.c_str()), ("cuModuleGetFunction(" + tfn + ")").c_str());
							if (args.drain_traj_2pass)
							{
								const std::string bfn = "run_span_deep_trajbuild_" + g + "_cuda";
								const std::string r2fn = "run_span_deep_trajroute2_cap" + cap_sfx + "_" + g + "_cuda";
								check_cuda(cuModuleGetFunction(&run_span_deep_trajbuild_kernel, module, bfn.c_str()), ("cuModuleGetFunction(" + bfn + ")").c_str());
								check_cuda(cuModuleGetFunction(&run_span_deep_trajroute2_kernel, module, r2fn.c_str()), ("cuModuleGetFunction(" + r2fn + ")").c_str());
							}
						}
					}
				}
			}
			check_cuda(cuModuleGetFunction(&screen_dedup_w32_packed_kernel, module, "tm_checksum_screen_dedup_w32_packed_cuda"), "cuModuleGetFunction(tm_checksum_screen_dedup_w32_packed_cuda)");
			check_cuda(cuModuleGetFunction(&screen_dedup_w32_fasthash_kernel, module, "tm_checksum_screen_dedup_w32_fasthash_cuda"), "cuModuleGetFunction(tm_checksum_screen_dedup_w32_fasthash_cuda)");
			check_cuda(cuModuleGetFunction(&screen_dedup_w32_maprng_kernel, module, "tm_checksum_screen_dedup_w32_maprng_cuda"), "cuModuleGetFunction(tm_checksum_screen_dedup_w32_maprng_cuda)");
			check_cuda(cuModuleGetFunction(&screen_dedup_w32_maprng_preext_kernel, module, "tm_checksum_screen_dedup_w32_maprng_preext_cuda"), "cuModuleGetFunction(tm_checksum_screen_dedup_w32_maprng_preext_cuda)");
			check_cuda(cuModuleGetFunction(&screen_dedup_w32_maprng_preext_1sync_kernel, module, "tm_checksum_screen_dedup_w32_maprng_preext_1sync_cuda"), "cuModuleGetFunction(tm_checksum_screen_dedup_w32_maprng_preext_1sync_cuda)");
			check_cuda(cuModuleGetFunction(&screen_dedup_w32_maprng_preext_cstore_kernel, module, "tm_checksum_screen_dedup_w32_maprng_preext_cstore_cuda"), "cuModuleGetFunction(tm_checksum_screen_dedup_w32_maprng_preext_cstore_cuda)");
			if (!args.preext_policy.empty())
			{
				const std::string fn = "tm_checksum_screen_dedup_w32_maprng_preext_" + args.preext_policy + "_cuda";
				check_cuda(cuModuleGetFunction(&preext_policy_kernel, module, fn.c_str()), ("cuModuleGetFunction(" + fn + ")").c_str());
			}
			if (!args.offset_geom.empty())
			{
				const std::string fn = "tm_checksum_screen_dedup_offset_" + args.offset_geom + "_cuda";
				check_cuda(cuModuleGetFunction(&offset_geom_kernel, module, fn.c_str()), ("cuModuleGetFunction(" + fn + ")").c_str());
				// parse "ilp{ILP}w{WARPS}_..." for grid/block geometry
				offset_geom_ilp   = static_cast<uint32_t>(std::stoul(args.offset_geom.substr(3, args.offset_geom.find('w') - 3)));
				offset_geom_warps = static_cast<uint32_t>(std::stoul(args.offset_geom.substr(args.offset_geom.find('w') + 1)));
			}
			check_cuda(cuModuleGetFunction(&screen_dedup_w32_offset_kernel, module, "tm_checksum_screen_dedup_w32_offset_cuda"), "cuModuleGetFunction(tm_checksum_screen_dedup_w32_offset_cuda)");
		}

		// map_rng POC kernel: distinct from the universal-table screen kernel
		// above. Takes a per-launch 54 KB map_rng buffer instead of 8 MB tables.
		CUfunction screen_maprng_kernel = nullptr;
		check_cuda(cuModuleGetFunction(&screen_maprng_kernel, module,
			(std::string("tm_checksum_screen_cuda_maprng") + (use_skipcar ? "_skipcar" : "_27")).c_str()),
			"cuModuleGetFunction(tm_checksum_screen_cuda_maprng)");

				KernelAssets assets = build_kernel_assets(args.key_id, args.map_list);

				if (args.raceway_direct)
				{
					if (use_skipcar)
						throw std::runtime_error("--raceway-direct currently supports --map-list all only");
					// Operational default cadence: a bounded-wave run with caps but no explicit cadence uses
					// the locked operating point 2,5,10,16 (K=4 — +3% full-key vs the old 2,4,8,14,20 and one
					// fewer cap table). State
					// continuation (ILP4) is the right default for these deep boundaries.
					if (args.raceway_direct_offset
						&& args.raceway_direct_wave_continue_batch != 0u
						&& args.raceway_direct_wave_boundaries.empty()
						&& !args.raceway_direct_wave_k_sweep)
					{
						args.raceway_direct_wave_boundaries = "2,5,10,16";
						args.raceway_direct_wave_state = true;
					}
					const bool raceway_wave_cadence_mode =
						args.raceway_direct_wave_continue_batch != 0u
						&& (args.raceway_direct_wave_k_sweep || !args.raceway_direct_wave_boundaries.empty());
					if (args.workunit_size > 0xFFFFFFFFull && !raceway_wave_cadence_mode)
						throw std::runtime_error("--raceway-direct requires --workunit_size <= 2^32-1 outside bounded wave cadence mode");
					const uint32_t selected_wave_continue_ilp = (args.raceway_direct_wave_continue_ilp != 0u)
						? args.raceway_direct_wave_continue_ilp
						: (args.raceway_direct_wave_state ? 4u : 6u);

					CUfunction raceway_direct_k = nullptr, raceway_clear_k = nullptr, raceway_compact_k = nullptr;
					CUfunction raceway_continue_k = nullptr;
					CUfunction raceway_span_state_k = nullptr;
					const char* raceway_direct_name = "tm_raceway_stream_window_cuda";
					if (args.raceway_direct_offset)
					{
						if (args.raceway_direct_cap_state_continue) raceway_direct_name = "tm_raceway_boundary_cap_state_offset_cuda";
						else if (args.raceway_direct_wave_continue_batch != 0u && args.raceway_direct_wave_state)
							raceway_direct_name = (args.raceway_direct_wave_span_ilp == 3u) ? "tm_raceway_boundary_cap_state_offset_ilp3_cuda"
								: (args.raceway_direct_wave_span_ilp == 4u) ? "tm_raceway_boundary_cap_state_offset_ilp4_cuda"
								: (args.raceway_direct_wave_span_ilp == 5u) ? "tm_raceway_boundary_cap_state_offset_ilp5_cuda"
								: (args.raceway_direct_wave_span_ilp == 6u) ? "tm_raceway_boundary_cap_state_offset_ilp6_cuda"
								: "tm_raceway_boundary_cap_state_offset_cuda";
						else if (args.raceway_direct_wave_continue_batch != 0u || args.raceway_direct_cap_mark)
							raceway_direct_name = (args.raceway_direct_wave_span_ilp == 4u) ? "tm_raceway_boundary_cap_mark_offset_ilp4_cuda"
								: (args.raceway_direct_wave_span_ilp == 5u) ? "tm_raceway_boundary_cap_mark_offset_ilp5_cuda"
								: "tm_raceway_boundary_cap_mark_offset_cuda";
						else if (args.raceway_direct_static && args.raceway_direct_ilp == 6u) raceway_direct_name = "tm_raceway_stream_window_offset_ilp6_static_cuda";
						else if (args.raceway_direct_ilp == 4u) raceway_direct_name = "tm_raceway_stream_window_offset_ilp4_cuda";
						else if (args.raceway_direct_ilp == 6u) raceway_direct_name = "tm_raceway_stream_window_offset_ilp6_cuda";
						else if (args.raceway_direct_ilp == 8u) raceway_direct_name = "tm_raceway_stream_window_offset_ilp8_cuda";
						// Full-window cap path stays ILP1: its cap sits mid-stream and the ILP1 kernel breaks
						// early on a drop (skips remaining maps for the ~70% that drop at the first boundary).
						// A naive ILP carry can't early-exit individual warp lanes, so it advances all candidates
						// through all maps and is ~2x SLOWER here (measured). ILP only helps when the cap is at
						// the span END (mark/state kernels), not mid-stream.
						else raceway_direct_name = "tm_raceway_stream_window_offset_cuda";
					}
					check_cuda(cuModuleGetFunction(&raceway_direct_k, module, raceway_direct_name), "cuModuleGetFunction(raceway_direct)");
					check_cuda(cuModuleGetFunction(&raceway_clear_k, module, "tm_raceway_cap_clear_cuda"), "cuModuleGetFunction(tm_raceway_cap_clear_cuda)");
					if (args.raceway_direct_wave_k_sweep || !args.raceway_direct_wave_boundaries.empty())
						check_cuda(cuModuleGetFunction(&count_nonzero_kernel, module, "count_nonzero_u64_cuda"), "cuModuleGetFunction(count_nonzero_u64_cuda)");
					if (args.raceway_direct_cap_compact || args.raceway_direct_wave_continue_batch != 0u)
						check_cuda(cuModuleGetFunction(&raceway_compact_k, module, "compact_survivors_ordered_cuda"), "cuModuleGetFunction(raceway_compact)");
					if (args.raceway_direct_cap_continue || args.raceway_direct_wave_continue_batch != 0u)
					{
						std::string continue_name_storage;
						const bool state_continue = args.raceway_direct_cap_state_continue || args.raceway_direct_wave_state;
						if (state_continue && args.raceway_direct_wave_continue_batch != 0u)
						{
							continue_name_storage = "tm_raceway_continue_state_liveidx_offset_ilp"
								+ std::to_string(selected_wave_continue_ilp) + "_static_cuda";
						}
						else
						{
							continue_name_storage = state_continue
								? "tm_raceway_continue_state_liveidx_offset_ilp6_static_cuda"
								: "tm_raceway_continue_liveidx_offset_ilp6_static_cuda";
						}
						const char* continue_name = continue_name_storage.c_str();
						check_cuda(cuModuleGetFunction(&raceway_continue_k, module, continue_name), "cuModuleGetFunction(raceway_continue)");
					}
					if (args.raceway_direct_wave_k_sweep || !args.raceway_direct_wave_boundaries.empty())
					{
						const char* span_state_name = (args.raceway_direct_wave_span_ilp == 3u) ? "tm_raceway_span_state_liveidx_cap_offset_ilp3_cuda"
							: (args.raceway_direct_wave_span_ilp == 4u) ? "tm_raceway_span_state_liveidx_cap_offset_ilp4_cuda"
							: (args.raceway_direct_wave_span_ilp == 5u) ? "tm_raceway_span_state_liveidx_cap_offset_ilp5_cuda"
							: (args.raceway_direct_wave_span_ilp == 6u) ? "tm_raceway_span_state_liveidx_cap_offset_ilp6_cuda"
							: "tm_raceway_span_state_liveidx_cap_offset_cuda";
						check_cuda(cuModuleGetFunction(&raceway_span_state_k, module, span_state_name), "cuModuleGetFunction(raceway_span_state)");
					}

					const std::vector<uint8_t> sched_blob = build_schedule_blob(args.key_id, args.map_list);
					const uint32_t schedule_count = static_cast<uint32_t>(sched_blob.size() / 4u);
					const uint64_t direct_total = args.workunit_size;
					uint32_t direct_n = direct_total > 0xFFFFFFFFull
						? 0xFFFFFFFFu
						: static_cast<uint32_t>(direct_total);
					uint32_t direct_data_start = static_cast<uint32_t>(args.range_start);
					auto completed_maps_to_cap_indices = [&](const std::vector<uint32_t>& completed_maps, const char* flag_name) {
						std::vector<uint32_t> out;
						for (uint32_t completed : completed_maps)
						{
							if (completed == 0u || completed >= schedule_count)
							{
								std::ostringstream e;
								e << flag_name << " entries must be completed map numbers in [1,"
								  << (schedule_count - 1u) << "] so the drain saves downstream compute";
								throw std::runtime_error(e.str());
							}
							out.push_back(completed - 1u);
						}
						std::sort(out.begin(), out.end());
						out.erase(std::unique(out.begin(), out.end()), out.end());
						if (out.empty()) throw std::runtime_error(std::string(flag_name) + " produced no usable boundaries");
						return out;
					};
					auto f1k_cap_indices = [&](uint32_t k) {
						if (k == 0u) throw std::runtime_error("f1k cadence K must be > 0");
						std::vector<uint32_t> completed;
						for (uint32_t m = 1u; m < schedule_count; m += k) completed.push_back(m);
						return completed_maps_to_cap_indices(completed, "f1k cadence");
					};
					std::vector<std::pair<std::string, std::vector<uint32_t>>> wave_cadence_plans;
					if (!args.raceway_direct_wave_boundaries.empty())
					{
						wave_cadence_plans.push_back({
							"custom(" + args.raceway_direct_wave_boundaries + ")",
							completed_maps_to_cap_indices(parse_u32_list(args.raceway_direct_wave_boundaries, "--raceway-direct-wave-boundaries"),
								"--raceway-direct-wave-boundaries") });
					}
					if (args.raceway_direct_wave_k_sweep)
					{
						const std::vector<uint32_t> ks = args.raceway_direct_wave_k_list.empty()
							? std::vector<uint32_t>{2u, 3u, 4u, 5u, 6u, 7u, 8u}
							: parse_u32_list(args.raceway_direct_wave_k_list, "--raceway-direct-wave-k-list");
						for (uint32_t k : ks)
							wave_cadence_plans.push_back({ "f1k" + std::to_string(k), f1k_cap_indices(k) });
					}
					uint32_t cap_count = (args.raceway_cap_bits == 0u) ? 0u : args.raceway_cap_count;
					if (!wave_cadence_plans.empty())
					{
						cap_count = 0u;
						for (const auto& p : wave_cadence_plans)
							cap_count = std::max<uint32_t>(cap_count, static_cast<uint32_t>(p.second.size()));
					}
					if (cap_count != 0u && args.raceway_first_cap_map + cap_count > schedule_count)
						throw std::runtime_error("--raceway-first-cap-map + --raceway-cap-count exceeds schedule length");
					if (args.raceway_direct_flat_parity && cap_count != 0u)
						throw std::runtime_error("--raceway-direct-flat-parity requires caps disabled");
					if (args.raceway_direct_ilp != 1u && args.raceway_direct_ilp != 4u
						&& args.raceway_direct_ilp != 6u && args.raceway_direct_ilp != 8u)
						throw std::runtime_error("--raceway-direct ILP must be 1, 4, 6, or 8");
					if (args.raceway_direct_ilp != 1u && (!args.raceway_direct_offset || cap_count != 0u || schedule_count != 27u))
						throw std::runtime_error("--raceway-direct-offset-ilp{4,6,8} requires offset streams, --raceway-cap-bits 0, and --map-list all");
					if (args.raceway_direct_static && (args.raceway_direct_ilp != 6u || !args.raceway_direct_flat_parity))
						throw std::runtime_error("--raceway-direct-static currently supports only --raceway-direct-offset-ilp6-static-flat-parity");
					if (args.raceway_direct_cap_mark && (!args.raceway_direct_offset || args.raceway_direct_ilp != 1u || args.raceway_direct_flat_parity || args.raceway_direct_static || cap_count == 0u))
						throw std::runtime_error("--raceway-direct-cap-mark requires direct offset ILP1 with caps enabled and no flat parity/static mode");
					if (args.raceway_direct_cap_compact && !args.raceway_direct_cap_mark)
						throw std::runtime_error("--raceway-direct-cap-compact requires --raceway-direct-cap-mark");
					if (args.raceway_direct_cap_continue && !args.raceway_direct_cap_compact)
						throw std::runtime_error("--raceway-direct-cap-continue requires --raceway-direct-cap-compact");
					if (args.raceway_direct_cap_state_continue && !args.raceway_direct_cap_continue)
						throw std::runtime_error("--raceway-direct-cap-state-continue requires --raceway-direct-cap-continue");
					if (args.raceway_direct_wave_continue_batch != 0u
						&& (!args.raceway_direct_offset || args.raceway_direct_ilp != 1u || args.raceway_direct_flat_parity
							|| args.raceway_direct_cap_mark || args.raceway_direct_static || cap_count == 0u))
						throw std::runtime_error("--raceway-direct-wave-continue-batch requires direct offset ILP1 with caps enabled and no flat parity/cap-mark/static modes");
					if (args.raceway_direct_wave_parity && args.raceway_direct_wave_continue_batch == 0u)
						throw std::runtime_error("--raceway-direct-wave-parity requires --raceway-direct-wave-continue-batch");
					if (args.raceway_direct_wave_state && args.raceway_direct_wave_continue_batch == 0u)
						throw std::runtime_error("--raceway-direct-wave-state requires --raceway-direct-wave-continue-batch");
					if (args.raceway_direct_wave_continue_batch != 0u
						&& !args.raceway_direct_wave_state
						&& selected_wave_continue_ilp != 6u)
						throw std::runtime_error("--raceway-direct-wave-continue-ilp 4/8 requires --raceway-direct-wave-state");
					if ((args.raceway_direct_wave_k_sweep || !args.raceway_direct_wave_boundaries.empty())
						&& args.raceway_direct_wave_continue_batch == 0u)
						throw std::runtime_error("--raceway-direct-wave-k-sweep/--raceway-direct-wave-boundaries requires --raceway-direct-wave-continue-batch");

					size_t race_mem_start_free = 0, race_mem_total = 0, race_mem_min_free = 0;
					auto sample_race_mem = [&]() {
						size_t free_b = 0, total_b = 0;
						check_cuda(cuMemGetInfo(&free_b, &total_b), "cuMemGetInfo(raceway_direct)");
						if (race_mem_total == 0u)
						{
							race_mem_start_free = free_b;
							race_mem_total = total_b;
							race_mem_min_free = free_b;
						}
						else
						{
							race_mem_min_free = std::min(race_mem_min_free, free_b);
						}
					};
					auto mib = [](uint64_t bytes) -> double {
						return static_cast<double>(bytes) / 1048576.0;
					};
					sample_race_mem();

					// Downstream-host VRAM auto-adjust (operational bounded-wave path).
					// AUTO: wave = 4M (the throughput sweet spot, §4) and cap scaled UP with available VRAM.
					// The cap does NOT plateau on the most-diffuse worst-case keys (their frontier is ~1.3-1.6B,
					// far above any cap that fits): +~4% per cap doubling through 2^27 (+11% vs 2^24), §11.
					// Regular diffuse keys plateau by 2^25. So scale to the largest cap that fits 0.90*free, up
					// to a 2^27 sane max: 8GB->2^25, 16GB->2^26, >=24GB->2^27. Falls back automatically on
					// tighter cards; 2^25 (~4.7GB) is the 8GB-floor point.
					// Prefer the largest wave (<=ceiling) and, for each, the largest cap (<=ceiling) that fits
					// 0.90*free; shrink wave only if no cap fits at that wave. For an EXPLICIT wave/cap this only
					// ever clamps DOWN, so a smaller downstream card cannot OOM.
					if (raceway_wave_cadence_mode && cap_count != 0u && args.raceway_cap_bits != 0u)
					{
						const double budget     = static_cast<double>(race_mem_start_free) * 0.90;
						const double offset_est  = 32.0 * 1048576.0;   // offset-stream + misc fixed buffers
						const double per_state   = 144.0;              // per wave state: alive+live_idx+128B state+flags (~137 measured)
						auto caps_bytes = [&](uint32_t bits) {
							return static_cast<double>(cap_count) * static_cast<double>(1ull << bits)
							     * static_cast<double>(args.raceway_cap_ways) * 8.0;
						};
						const bool auto_mode = args.raceway_wave_auto_size;
						const uint32_t wave_ceiling = auto_mode ? 4194304u : args.raceway_direct_wave_continue_batch;
						const uint32_t cap_ceiling  = auto_mode ? 27u : args.raceway_cap_bits; // auto scales cap up to 2^27 (VRAM-bounded; falls back automatically)
						const uint32_t cap_floor    = 18u;
						uint32_t chosen_wave = 0u, chosen_bits = 0u;
						auto fits = [&](uint32_t w, uint32_t bits) {
							return caps_bytes(bits) + static_cast<double>(w) * per_state + offset_est <= budget;
						};
						auto try_wave = [&](uint32_t w) {
							if (chosen_wave != 0u || w == 0u || static_cast<uint64_t>(w) > direct_total) return;
							for (uint32_t bits = cap_ceiling; bits >= cap_floor; --bits)
								if (fits(w, bits)) { chosen_wave = w; chosen_bits = bits; return; }
						};
						for (uint32_t w = wave_ceiling; w >= 262144u && chosen_wave == 0u; w >>= 1) try_wave(w);
						if (chosen_wave == 0u && static_cast<uint64_t>(wave_ceiling) > direct_total)
							try_wave(static_cast<uint32_t>(direct_total)); // small window not on the pow2 ladder
						if (chosen_wave == 0u)
							throw std::runtime_error("raceway: caps + a 256K wave do not fit free VRAM; lower --raceway-cap-bits or free the device");
						if (auto_mode || chosen_wave != args.raceway_direct_wave_continue_batch || chosen_bits != args.raceway_cap_bits)
							std::cout << "  [raceway-auto] free=" << static_cast<uint64_t>(mib(race_mem_start_free))
							          << " MiB budget=0.90 -> wave=" << chosen_wave << " cap=2^" << chosen_bits << "x" << args.raceway_cap_ways
							          << (auto_mode ? "  (auto)" : "  (clamped to fit)") << "\n";
						args.raceway_direct_wave_continue_batch = chosen_wave;
						args.raceway_cap_bits = chosen_bits;
					}

					CUdeviceptr d_direct_ostream = 0;
					CUdeviceptr off_regular = 0, off_alg0 = 0, off_alg6 = 0, off_alg2 = 0, off_alg5 = 0;
					OffsetStreamBlob osb;
					if (args.raceway_direct_offset || args.raceway_direct_flat_parity)
					{
						osb = build_offset_stream_blob(sched_blob);
						check_cuda(cuMemAlloc(&d_direct_ostream, osb.data.size()), "cuMemAlloc(raceway_direct_ostream)");
						check_cuda(cuMemcpyHtoD(d_direct_ostream, osb.data.data(), osb.data.size()), "HtoD(raceway_direct_ostream)");
						off_regular = d_direct_ostream;
						off_alg0 = off_regular + osb.stream_bytes;
						off_alg6 = off_alg0 + osb.stream_bytes;
						off_alg2 = off_alg6 + osb.stream_bytes;
						off_alg5 = off_alg2 + osb.carry_bytes;
					}
					sample_race_mem();

					std::vector<CUdeviceptr> cap_tables_h(cap_count, 0);
					std::vector<uint32_t> cap_bits_h(cap_count, args.raceway_cap_bits);
					std::vector<uint32_t> cap_ways_h(cap_count, args.raceway_cap_ways);
					CUdeviceptr d_cap_tables = 0, d_cap_bits = 0, d_cap_ways = 0;
					uint64_t cap_slots_total = 0ull;
					for (uint32_t c = 0u; c < cap_count; ++c)
					{
						const uint64_t cap_slots = (1ull << args.raceway_cap_bits) * static_cast<uint64_t>(args.raceway_cap_ways);
						cap_slots_total += cap_slots;
						check_cuda(cuMemAlloc(&cap_tables_h[c], cap_slots * sizeof(unsigned long long)), "cuMemAlloc(raceway_direct_cap)");
						void* za[] = { &cap_tables_h[c], (void*)&cap_slots };
						check_cuda(cuLaunchKernel(raceway_clear_k, static_cast<uint32_t>((cap_slots + 255ull) / 256ull), 1, 1,
							256, 1, 1, 0, 0, za, nullptr), "launch(raceway_direct_cap_clear)");
					}
					if (cap_count != 0u)
					{
						check_cuda(cuMemAlloc(&d_cap_tables, static_cast<size_t>(cap_count) * sizeof(CUdeviceptr)), "cuMemAlloc(raceway_direct_cap_ptrs)");
						check_cuda(cuMemcpyHtoD(d_cap_tables, cap_tables_h.data(), static_cast<size_t>(cap_count) * sizeof(CUdeviceptr)), "HtoD(raceway_direct_cap_ptrs)");
						check_cuda(cuMemAlloc(&d_cap_bits, static_cast<size_t>(cap_count) * sizeof(uint32_t)), "cuMemAlloc(raceway_direct_cap_bits)");
						check_cuda(cuMemcpyHtoD(d_cap_bits, cap_bits_h.data(), static_cast<size_t>(cap_count) * sizeof(uint32_t)), "HtoD(raceway_direct_cap_bits)");
						check_cuda(cuMemAlloc(&d_cap_ways, static_cast<size_t>(cap_count) * sizeof(uint32_t)), "cuMemAlloc(raceway_direct_cap_ways)");
						check_cuda(cuMemcpyHtoD(d_cap_ways, cap_ways_h.data(), static_cast<size_t>(cap_count) * sizeof(uint32_t)), "HtoD(raceway_direct_cap_ways)");
					}
					sample_race_mem();

					if (!wave_cadence_plans.empty())
					{
						const uint32_t wave_cap = static_cast<uint32_t>(std::min<uint64_t>(
							args.raceway_direct_wave_continue_batch, direct_total));
						CUdeviceptr wave_work_counter = 0, wave_stats = 0, wave_alive = 0;
						CUdeviceptr wave_live_a = 0, wave_live_b = 0, wave_compact_counter = 0, wave_flags = 0;
						CUdeviceptr wave_flat_flags = 0, wave_state = 0, wave_drop_map = 0;
						CUdeviceptr wave_occ_counter = 0;
						check_cuda(cuMemAlloc(&wave_work_counter, sizeof(uint32_t)), "cuMemAlloc(raceway_wave_work_counter)");
						check_cuda(cuMemAlloc(&wave_stats, sizeof(RacewayStatsHost)), "cuMemAlloc(raceway_wave_stats)");
						check_cuda(cuMemAlloc(&wave_alive, wave_cap), "cuMemAlloc(raceway_wave_alive)");
						check_cuda(cuMemAlloc(&wave_live_a, static_cast<size_t>(wave_cap) * sizeof(uint32_t)), "cuMemAlloc(raceway_wave_live_a)");
						check_cuda(cuMemAlloc(&wave_live_b, static_cast<size_t>(wave_cap) * sizeof(uint32_t)), "cuMemAlloc(raceway_wave_live_b)");
						check_cuda(cuMemAlloc(&wave_compact_counter, sizeof(uint32_t)), "cuMemAlloc(raceway_wave_compact_counter)");
						check_cuda(cuMemAlloc(&wave_state, static_cast<size_t>(wave_cap) * 32u * sizeof(uint32_t)), "cuMemAlloc(raceway_wave_state)");
						if (args.raceway_direct_wave_parity)
						{
							check_cuda(cuMemAlloc(&wave_flags, wave_cap), "cuMemAlloc(raceway_wave_flags)");
							check_cuda(cuMemAlloc(&wave_flat_flags, wave_cap), "cuMemAlloc(raceway_wave_flat_flags)");
						}
						if (args.raceway_drop_hist)
							check_cuda(cuMemAlloc(&wave_drop_map, wave_cap), "cuMemAlloc(raceway_wave_drop_map)");
						check_cuda(cuMemAlloc(&wave_occ_counter, sizeof(uint32_t)), "cuMemAlloc(raceway_wave_occ_counter)");
						sample_race_mem();
						uint64_t wave_device_bytes = sizeof(uint32_t) + sizeof(RacewayStatsHost)
							+ static_cast<uint64_t>(wave_cap)
							+ static_cast<uint64_t>(wave_cap) * 2ull * sizeof(uint32_t)
							+ sizeof(uint32_t)
							+ static_cast<uint64_t>(wave_cap) * 32ull * sizeof(uint32_t);
						if (args.raceway_direct_wave_parity)
							wave_device_bytes += static_cast<uint64_t>(wave_cap) * 2ull;
						if (args.raceway_drop_hist)
							wave_device_bytes += static_cast<uint64_t>(wave_cap);
						wave_device_bytes += sizeof(uint32_t);

						auto clear_raceway_caps = [&]() {
							for (uint32_t c = 0u; c < cap_count; ++c)
							{
								const uint64_t cap_slots = (1ull << args.raceway_cap_bits) * static_cast<uint64_t>(args.raceway_cap_ways);
								void* za[] = { &cap_tables_h[c], (void*)&cap_slots };
								check_cuda(cuLaunchKernel(raceway_clear_k, static_cast<uint32_t>((cap_slots + 255ull) / 256ull), 1, 1,
									256, 1, 1, 0, 0, za, nullptr), "launch(raceway_wave_cap_clear)");
							}
							check_cuda(cuCtxSynchronize(), "sync(raceway_wave_cap_clear)");
						};
						auto completed_map_string = [](const std::vector<uint32_t>& cap_indices) {
							std::ostringstream s;
							for (size_t i = 0; i < cap_indices.size(); ++i)
							{
								if (i) s << ",";
								s << (cap_indices[i] + 1u);
							}
							return s.str();
						};
						auto count_cap_occupied = [&](CUdeviceptr table, uint64_t slots) -> uint32_t {
							if (!count_nonzero_kernel || !table || slots == 0ull || slots > static_cast<uint64_t>(UINT32_MAX))
								return 0u;
							uint32_t nslots = static_cast<uint32_t>(slots);
							check_cuda(cuMemsetD32(wave_occ_counter, 0u, 1u), "memset(raceway_wave_occ_counter)");
							void* oa[] = { &table, &nslots, &wave_occ_counter };
							check_cuda(cuLaunchKernel(count_nonzero_kernel, (nslots + 255u) / 256u, 1, 1,
								256, 1, 1, 0, 0, oa, nullptr), "launch(raceway_wave_count_cap)");
							check_cuda(cuCtxSynchronize(), "sync(raceway_wave_count_cap)");
							uint32_t occupied = 0u;
							check_cuda(cuMemcpyDtoH(&occupied, wave_occ_counter, sizeof(uint32_t)), "DtoH(raceway_wave_occ_counter)");
							return occupied;
						};

						const uint32_t race_threads_per_block = 256u;
						const uint32_t race_warps_per_block = race_threads_per_block / kCudaWarpSize;
						const uint32_t continue_threads = 256u;
						const uint32_t continue_warps = continue_threads / kCudaWarpSize;
						std::vector<uint8_t> h_cont, h_flat, h_drop;
						std::vector<uint32_t> h_live;
						if (args.raceway_direct_wave_parity)
						{
							h_cont.resize(wave_cap);
							h_flat.resize(wave_cap);
							h_live.resize(wave_cap);
						}
						if (args.raceway_drop_hist) h_drop.resize(wave_cap);

						std::cout << "[raceway-wave-cadence-sweep] device=" << device_name
						          << " key=0x" << std::hex << std::setw(8) << std::setfill('0') << args.key_id
						          << std::dec << std::setfill(' ')
						          << " mode=offset state=1"
						          << " batch=" << args.raceway_direct_wave_continue_batch
						          << " span_ilp=" << args.raceway_direct_wave_span_ilp
						          << " continue_ilp=" << selected_wave_continue_ilp
						          << " window=" << direct_total
						          << " data_start=" << direct_data_start
						          << " cap_shape=2^" << args.raceway_cap_bits << "x" << args.raceway_cap_ways
						          << " (" << (cap_slots_total * sizeof(unsigned long long) / (1ull << 20)) << " MB max)\n";
						std::cout << "  memory: observed_peak_used=" << std::fixed << std::setprecision(1)
						          << mib(static_cast<uint64_t>(race_mem_total - race_mem_min_free)) << " MiB"
						          << " delta_from_race_start="
						          << mib(static_cast<uint64_t>(race_mem_start_free - race_mem_min_free)) << " MiB"
						          << " expected_device_alloc="
						          << mib(cap_slots_total * sizeof(unsigned long long)
						                 + static_cast<uint64_t>(osb.data.size()) + wave_device_bytes) << " MiB"
						          << " (caps=" << mib(cap_slots_total * sizeof(unsigned long long))
						          << ", wave=" << mib(wave_device_bytes)
						          << ", offset_stream=" << mib(static_cast<uint64_t>(osb.data.size())) << ")\n";
						std::cout << "  Note: completed-map 27 is intentionally excluded because it cannot save downstream compute.\n";

						// ===== PRODUCTION raceway calibration: empirical span-ILP x cap-bits sweep. =====
						// Raceway is the production engine; pick its per-device geom by MEASURING the
						// wave-cadence pipeline throughput (M input/s) across representative keys. The cap is
						// FN-safe over-keep, so we rank by throughput (not bit-parity). Records engine=raceway.
						if (args.calibrate_raceway)
						{
							const uint32_t calib_keys[]    = { 0x2ca5b42du, 0x9e9d137bu, 0xdad04412u }; // mid/diffuse/most-diffuse
							const uint32_t span_ilps[]     = { 1u, 3u, 4u, 5u, 6u };
							const uint32_t capbits_cands[] = { 25u, 26u, 27u };
							const uint64_t calib_window    = std::min<uint64_t>(direct_total, 268435456ull); // 256M
							const std::vector<uint32_t>& bnd = wave_cadence_plans.front().second;
							const uint32_t continue_ilp = selected_wave_continue_ilp;
							auto NOW = []{ return std::chrono::high_resolution_clock::now(); };
							auto MS  = [](std::chrono::high_resolution_clock::time_point a, std::chrono::high_resolution_clock::time_point b){ return std::chrono::duration<double, std::milli>(b - a).count(); };
							const double budget    = static_cast<double>(race_mem_start_free) * 0.90;
							const double offset_est = static_cast<double>(osb.data.size());
							auto caps_bytes = [&](uint32_t bits){ return (double)cap_count * (double)(1ull << bits) * (double)args.raceway_cap_ways * 8.0; };
							auto fits = [&](uint32_t bits){ return caps_bytes(bits) + (double)wave_cap * 144.0 + offset_est <= budget; };
							auto suffix = [](uint32_t n){ return n == 1u ? std::string("") : ("_ilp" + std::to_string(n)); };
							auto reselect_kernels = [&](uint32_t spilp){
								const std::string dn = "tm_raceway_boundary_cap_state_offset" + suffix(spilp) + "_cuda";
								const std::string sn = "tm_raceway_span_state_liveidx_cap_offset" + suffix(spilp) + "_cuda";
								check_cuda(cuModuleGetFunction(&raceway_direct_k, module, dn.c_str()), "cal-rw-direct");
								check_cuda(cuModuleGetFunction(&raceway_span_state_k, module, sn.c_str()), "cal-rw-span");
							};
							auto realloc_caps = [&](uint32_t bits){
								for (auto& pp : cap_tables_h) if (pp) { cuMemFree(pp); pp = 0; }
								cap_slots_total = 0ull;
								const uint64_t slots = (1ull << bits) * (uint64_t)args.raceway_cap_ways;
								for (uint32_t c = 0u; c < cap_count; ++c){ cap_slots_total += slots;
									check_cuda(cuMemAlloc(&cap_tables_h[c], slots * sizeof(unsigned long long)), "cal-rw-cap"); }
								std::vector<uint32_t> cbv(cap_count, bits), cwv(cap_count, args.raceway_cap_ways);
								check_cuda(cuMemcpyHtoD(d_cap_tables, cap_tables_h.data(), (size_t)cap_count * sizeof(CUdeviceptr)), "cal-rw-capptr");
								check_cuda(cuMemcpyHtoD(d_cap_bits, cbv.data(), (size_t)cap_count * sizeof(uint32_t)), "cal-rw-capbits");
								check_cuda(cuMemcpyHtoD(d_cap_ways, cwv.data(), (size_t)cap_count * sizeof(uint32_t)), "cal-rw-capways");
								args.raceway_cap_bits = bits;
							};
							auto measure = [&]() -> double {
								clear_raceway_caps();
								double cap_ms = 0.0, comp_ms = 0.0, cont_ms = 0.0;
								for (uint64_t done = 0; done < calib_window; ){
									const uint32_t batch_n = (uint32_t)std::min<uint64_t>(args.raceway_direct_wave_continue_batch, calib_window - done);
									const uint32_t batch_data_start = (uint32_t)(direct_data_start + done);
									check_cuda(cuMemsetD8(wave_stats, 0u, sizeof(RacewayStatsHost)), "m");
									check_cuda(cuMemsetD32(wave_work_counter, 0u, 1u), "m");
									uint32_t batch_blocks = std::min<uint32_t>(8192u, std::max<uint32_t>(1u,
										(uint32_t)(((uint64_t)batch_n + race_warps_per_block - 1ull) / race_warps_per_block)));
									uint32_t one_cap = 1u, first_boundary = bnd.front(), batch_schedule_count = schedule_count;
									CUdeviceptr wave_drop_arg = 0;
									void* first_args[] = { (void*)&batch_n, (void*)&batch_data_start, &wave_work_counter, &wave_alive, &wave_drop_arg, &wave_stats,
										&wave_state, &d_cap_tables, &d_cap_bits, &d_cap_ways, &one_cap,
										&off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
										&assets.expansion_values, &assets.schedule_data, &args.key_id,
										&batch_schedule_count, &first_boundary };
									check_cuda(cuCtxSynchronize(), "m"); auto f0 = NOW();
									check_cuda(cuLaunchKernel(raceway_direct_k, batch_blocks, 1, 1, race_threads_per_block, 1, 1, 0, 0, first_args, nullptr), "cal-rw-first");
									check_cuda(cuCtxSynchronize(), "m"); cap_ms += MS(f0, NOW());
									uint32_t compacted = 0, batch_n_arg = batch_n; CUdeviceptr null_live = 0; int first_span = 1;
									check_cuda(cuMemsetD32(wave_compact_counter, 0u, 1u), "m");
									void* ca0[] = { &wave_alive, &null_live, &batch_n_arg, &wave_live_a, &wave_compact_counter, &first_span };
									auto c0 = NOW();
									check_cuda(cuLaunchKernel(raceway_compact_k, (batch_n + 255u) / 256u, 1, 1, 256, 1, 1, 0, 0, ca0, nullptr), "cal-rw-comp0");
									check_cuda(cuCtxSynchronize(), "m"); comp_ms += MS(c0, NOW());
									check_cuda(cuMemcpyDtoH(&compacted, wave_compact_counter, sizeof(uint32_t)), "m");
									CUdeviceptr cur_live = wave_live_a, next_live = wave_live_b; uint32_t cur_count = compacted, prev_boundary = first_boundary;
									for (size_t bi = 1; bi < bnd.size(); ++bi){
										if (cur_count == 0u) break;
										check_cuda(cuMemsetD32(wave_work_counter, 0u, 1u), "m");
										check_cuda(cuMemsetD32(wave_compact_counter, 0u, 1u), "m");
										uint32_t start_map = prev_boundary + 1u, end_map = bnd[bi];
										uint32_t cb1 = args.raceway_cap_bits, cw1 = args.raceway_cap_ways;
										void* sa[] = { &cur_live, &cur_count, &wave_work_counter, &wave_state, &wave_alive, &wave_drop_arg,
											&wave_stats, &cap_tables_h[bi], &cb1, &cw1,
											&off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
											&assets.schedule_data, &start_map, &end_map };
										uint32_t span_blocks = std::min<uint32_t>(8192u, std::max<uint32_t>(1u,
											(uint32_t)(((uint64_t)cur_count + race_warps_per_block - 1ull) / race_warps_per_block)));
										auto s0 = NOW();
										check_cuda(cuLaunchKernel(raceway_span_state_k, span_blocks, 1, 1, race_threads_per_block, 1, 1, 0, 0, sa, nullptr), "cal-rw-span");
										check_cuda(cuCtxSynchronize(), "m"); cap_ms += MS(s0, NOW());
										int not_first = 0;
										void* ca[] = { &wave_alive, &cur_live, &cur_count, &next_live, &wave_compact_counter, &not_first };
										auto cc0 = NOW();
										check_cuda(cuLaunchKernel(raceway_compact_k, (cur_count + 255u) / 256u, 1, 1, 256, 1, 1, 0, 0, ca, nullptr), "cal-rw-spancomp");
										check_cuda(cuCtxSynchronize(), "m"); comp_ms += MS(cc0, NOW());
										check_cuda(cuMemcpyDtoH(&cur_count, wave_compact_counter, sizeof(uint32_t)), "m");
										std::swap(cur_live, next_live); prev_boundary = end_map;
									}
									uint32_t continue_grid = std::max<uint32_t>(1u, (uint32_t)(((uint64_t)cur_count
										+ (uint64_t)continue_warps * continue_ilp - 1ull) / ((uint64_t)continue_warps * continue_ilp)));
									uint32_t continue_start_map = prev_boundary + 1u; CUdeviceptr null_flags = 0;
									void* cont_args[] = { &cur_live, &cur_count, &wave_state, &null_flags,
										&off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
										&assets.schedule_data, &assets.carnival_data, &continue_start_map };
									auto k0 = NOW();
									if (cur_count != 0u)
										check_cuda(cuLaunchKernel(raceway_continue_k, continue_grid, 1, 1, continue_threads, 1, 1, 0, 0, cont_args, nullptr), "cal-rw-cont");
									check_cuda(cuCtxSynchronize(), "m"); cont_ms += MS(k0, NOW());
									done += batch_n;
								}
								return cap_ms + comp_ms + cont_ms;
							};

							std::cout << "Calibrating raceway (PRODUCTION) on " << device_name
							          << " - 3 keys x span-ILP{1,3,4,5,6} x cap-bits{25,26,27 fit}, window=" << calib_window << "\n";
							double best_hm = 0.0; uint32_t best_sp = 4u, best_cb = 25u;
							for (uint32_t cbits : capbits_cands)
							{
								if (!fits(cbits)) { std::cout << "  cap=2^" << cbits << ": skip (does not fit VRAM budget)\n"; continue; }
								realloc_caps(cbits);
								for (uint32_t sp : span_ilps)
								{
									reselect_kernels(sp);
									double inv = 0.0; uint32_t nk = 0;
									for (uint32_t k : calib_keys)
									{
										args.key_id = k;
										std::vector<uint8_t> sb = build_schedule_blob(k, args.map_list);
										check_cuda(cuMemcpyHtoD(assets.schedule_data, sb.data(), sb.size()), "cal-rw-sched");
										const double pms = measure();
										const double rate = (double)calib_window / 1e6 / (pms / 1e3);
										inv += 1.0 / rate; ++nk;
									}
									const double hm = (double)nk / inv;
									std::cout << "  span-ILP" << sp << " cap=2^" << cbits << ": "
									          << std::fixed << std::setprecision(1) << hm << " M input/s (HM across keys)\n";
									if (hm > best_hm) { best_hm = hm; best_sp = sp; best_cb = cbits; }
								}
							}

							size_t devmem = 0; cuDeviceTotalMem(&devmem, device);
							int sm = 0; cuDeviceGetAttribute(&sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
							std::ostringstream fpkey; fpkey << "cuda|" << device_name_buffer << "|" << (devmem >> 30) << "GB|" << sm << "SM";
							std::ostringstream cline; cline << fpkey.str() << "\tengine=raceway span_ilp=" << best_sp
								<< " cap_bits=" << best_cb << " cap_ways=" << args.raceway_cap_ways
								<< " wave=" << wave_cap << " cadence=2,5,10,16 rate_Mhm="
								<< std::fixed << std::setprecision(1) << best_hm << " runtime=cuda";
							auto engine_of = [](const std::string& ss){ const size_t pe = ss.find("engine="); if (pe == std::string::npos) return std::string();
								size_t ee = ss.find_first_of(" \t", pe); return ss.substr(pe, (ee == std::string::npos ? ss.size() : ee) - pe); };
							auto identity = [](const std::string& key){ std::vector<std::string> f; std::string c; std::istringstream ks(key);
								while (std::getline(ks, c, '|')) f.push_back(c); if (f.size() < 4) return key; return f.front() + "|" + f[1] + "|" + f.back(); };
							const std::string my_id = identity(fpkey.str());
							std::vector<std::string> clines;
							{
								std::ifstream in(args.config_path); std::string l;
								while (std::getline(in, l)){
									if (l.empty() || l[0] == '#') continue;
									const size_t tab = l.find('\t');
									const bool same_dev = identity(l.substr(0, tab)) == my_id;
									const bool same_eng = engine_of(l) == "engine=raceway";
									if (!(same_dev && same_eng)) clines.push_back(l);
								}
							}
							clines.push_back(cline.str());
							{
								std::ofstream out(args.config_path, std::ios::trunc);
								out << "# tm_compaction.conf  (runtime|device-fingerprint -> engine/geom/tile)\n";
								out << "# NOTE: raceway is the production-default architecture; the screen/compaction\n";
								out << "#       engines recorded here are research/A-B only.\n";
								for (auto& l : clines) out << l << "\n";
							}
							std::cout << "Raceway calibration result: span-ILP" << best_sp << " cap=2^" << best_cb
							          << "x" << args.raceway_cap_ways << " wave=" << wave_cap << "  "
							          << std::fixed << std::setprecision(1) << best_hm << " M input/s (HM across keys)\n";
							std::cout << "  wrote " << args.config_path << "  [" << fpkey.str() << "]\n";

							for (auto pp : cap_tables_h) if (pp) cuMemFree(pp);
							cuMemFree(wave_work_counter); cuMemFree(wave_stats); cuMemFree(wave_alive);
							cuMemFree(wave_live_a); cuMemFree(wave_live_b); cuMemFree(wave_compact_counter); cuMemFree(wave_state);
							if (d_cap_tables) cuMemFree(d_cap_tables); if (d_cap_bits) cuMemFree(d_cap_bits); if (d_cap_ways) cuMemFree(d_cap_ways);
							if (d_direct_ostream) cuMemFree(d_direct_ostream);
							release_assets(assets);
							return 0;
						}

						for (const auto& plan : wave_cadence_plans)
						{
							clear_raceway_caps();
							RacewayStatsHost total_stats;
							double cap_ms_total = 0.0;
							double compact_ms_total = 0.0;
							double continue_ms_total = 0.0;
							double flat_ms_total = 0.0;
							uint64_t total_survivors = 0ull;
							uint64_t continuation_slots = 0ull;
							uint64_t parity_checked = 0ull;
							uint64_t parity_diff = 0ull;
							uint64_t cont_sum = 0ull;
							uint64_t flat_sum = 0ull;
							uint64_t hist_total = 0ull;
							std::vector<uint64_t> hist(schedule_count, 0ull);
							std::vector<RacewayBoundaryStats> boundary_stats(plan.second.size());
							for (size_t bi = 0; bi < plan.second.size(); ++bi)
							{
								boundary_stats[bi].completed_map = plan.second[bi] + 1u;
								boundary_stats[bi].cap_slots = (1ull << args.raceway_cap_bits)
									* static_cast<uint64_t>(args.raceway_cap_ways);
							}
							uint32_t min_survivors = 0xFFFFFFFFu;
							uint32_t max_survivors = 0u;
							uint32_t empty_waves = 0u;
							uint32_t batches = 0u;

							for (uint64_t done = 0u; done < direct_total; )
							{
								const uint32_t batch_n = static_cast<uint32_t>(std::min<uint64_t>(
									args.raceway_direct_wave_continue_batch, direct_total - done));
								const uint32_t batch_data_start = static_cast<uint32_t>(args.range_start + done);
								check_cuda(cuMemsetD8(wave_stats, 0u, sizeof(RacewayStatsHost)), "memset(raceway_wave_stats)");
								if (wave_flags) check_cuda(cuMemsetD8(wave_flags, 0u, batch_n), "memset(raceway_wave_flags)");
								if (wave_drop_map) check_cuda(cuMemsetD8(wave_drop_map, 0xFFu, batch_n), "memset(raceway_wave_drop_map)");

								uint32_t batch_blocks = static_cast<uint32_t>((static_cast<uint64_t>(batch_n) + race_warps_per_block - 1ull)
									/ race_warps_per_block);
								batch_blocks = std::min<uint32_t>(8192u, std::max<uint32_t>(1u, batch_blocks));

								check_cuda(cuMemsetD32(wave_work_counter, 0u, 1u), "memset(raceway_wave_work_counter)");
								uint32_t one_cap = 1u;
								uint32_t first_boundary = plan.second.front();
								uint32_t batch_schedule_count = schedule_count;
								CUdeviceptr wave_drop_arg = wave_drop_map;
								void* first_args[] = {
									(void*)&batch_n, (void*)&batch_data_start, &wave_work_counter, &wave_alive, &wave_drop_arg, &wave_stats,
									&wave_state, &d_cap_tables, &d_cap_bits, &d_cap_ways, &one_cap,
									&off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
									&assets.expansion_values, &assets.schedule_data, &args.key_id,
									&batch_schedule_count, &first_boundary };
								check_cuda(cuCtxSynchronize(), "sync(raceway_wave_first_pre)");
								const auto f0 = std::chrono::high_resolution_clock::now();
								check_cuda(cuLaunchKernel(raceway_direct_k, batch_blocks, 1, 1,
									race_threads_per_block, 1, 1, 0, 0, first_args, nullptr), "launch(raceway_wave_first)");
								check_cuda(cuCtxSynchronize(), "sync(raceway_wave_first)");
								const auto f1 = std::chrono::high_resolution_clock::now();
								const double first_span_ms = std::chrono::duration<double, std::milli>(f1 - f0).count();
								cap_ms_total += first_span_ms;

								uint32_t compacted = 0u;
								uint32_t batch_n_arg = batch_n;
								CUdeviceptr null_live_in = 0;
								int first_span = 1;
								check_cuda(cuMemsetD32(wave_compact_counter, 0u, 1u), "memset(raceway_wave_compact_counter)");
								void* ca0[] = { &wave_alive, &null_live_in, &batch_n_arg, &wave_live_a, &wave_compact_counter, &first_span };
								const auto c0 = std::chrono::high_resolution_clock::now();
								check_cuda(cuLaunchKernel(raceway_compact_k, (batch_n + 255u) / 256u, 1, 1,
									256, 1, 1, 0, 0, ca0, nullptr), "launch(raceway_wave_compact0)");
								check_cuda(cuCtxSynchronize(), "sync(raceway_wave_compact0)");
								const auto c1 = std::chrono::high_resolution_clock::now();
								const double first_compact_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
								compact_ms_total += first_compact_ms;
								check_cuda(cuMemcpyDtoH(&compacted, wave_compact_counter, sizeof(uint32_t)), "DtoH(raceway_wave_compact_counter0)");
								if (!boundary_stats.empty())
								{
									RacewayBoundaryStats& bs0 = boundary_stats[0];
									bs0.input += batch_n;
									bs0.survivors += compacted;
									bs0.dropped += static_cast<uint64_t>(batch_n) - compacted;
									bs0.map_evals += static_cast<uint64_t>(batch_n) * static_cast<uint64_t>(first_boundary + 1u);
									bs0.span_ms += first_span_ms;
									bs0.compact_ms += first_compact_ms;
								}

								CUdeviceptr cur_live = wave_live_a;
								CUdeviceptr next_live = wave_live_b;
								uint32_t cur_count = compacted;
								uint32_t prev_boundary = first_boundary;
								for (uint32_t bi = 1u; bi < plan.second.size(); ++bi)
								{
									if (cur_count == 0u) break;
									check_cuda(cuMemsetD32(wave_work_counter, 0u, 1u), "memset(raceway_wave_span_work_counter)");
									check_cuda(cuMemsetD32(wave_compact_counter, 0u, 1u), "memset(raceway_wave_span_compact_counter)");
									uint32_t start_map = prev_boundary + 1u;
									uint32_t end_map = plan.second[bi];
									uint32_t cap_bits_one = args.raceway_cap_bits;
									uint32_t cap_ways_one = args.raceway_cap_ways;
									void* sa[] = { &cur_live, &cur_count, &wave_work_counter, &wave_state, &wave_alive, &wave_drop_arg,
										&wave_stats, &cap_tables_h[bi], &cap_bits_one, &cap_ways_one,
										&off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
										&assets.schedule_data, &start_map, &end_map };
									const uint32_t span_blocks = std::min<uint32_t>(8192u, std::max<uint32_t>(1u,
										static_cast<uint32_t>((static_cast<uint64_t>(cur_count) + race_warps_per_block - 1ull) / race_warps_per_block)));
									const auto s0 = std::chrono::high_resolution_clock::now();
									check_cuda(cuLaunchKernel(raceway_span_state_k, span_blocks, 1, 1,
										race_threads_per_block, 1, 1, 0, 0, sa, nullptr), "launch(raceway_wave_span)");
									check_cuda(cuCtxSynchronize(), "sync(raceway_wave_span)");
									const auto s1 = std::chrono::high_resolution_clock::now();
									const double span_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();
									cap_ms_total += span_ms;

									int not_first_span = 0;
									const uint32_t span_input = cur_count;
									void* ca[] = { &wave_alive, &cur_live, &cur_count, &next_live, &wave_compact_counter, &not_first_span };
									const auto cc0 = std::chrono::high_resolution_clock::now();
									check_cuda(cuLaunchKernel(raceway_compact_k, (cur_count + 255u) / 256u, 1, 1,
										256, 1, 1, 0, 0, ca, nullptr), "launch(raceway_wave_span_compact)");
									check_cuda(cuCtxSynchronize(), "sync(raceway_wave_span_compact)");
									const auto cc1 = std::chrono::high_resolution_clock::now();
									const double span_compact_ms = std::chrono::duration<double, std::milli>(cc1 - cc0).count();
									compact_ms_total += span_compact_ms;
									check_cuda(cuMemcpyDtoH(&cur_count, wave_compact_counter, sizeof(uint32_t)), "DtoH(raceway_wave_span_compact_counter)");
									if (bi < boundary_stats.size())
									{
										RacewayBoundaryStats& bsb = boundary_stats[bi];
										bsb.input += span_input;
										bsb.survivors += cur_count;
										bsb.dropped += static_cast<uint64_t>(span_input) - cur_count;
										bsb.map_evals += static_cast<uint64_t>(span_input)
											* static_cast<uint64_t>(end_map - start_map + 1u);
										bsb.span_ms += span_ms;
										bsb.compact_ms += span_compact_ms;
									}
									std::swap(cur_live, next_live);
									prev_boundary = end_map;
								}

								const uint32_t continue_ilp = selected_wave_continue_ilp;
								const uint64_t cont_slots = ((static_cast<uint64_t>(cur_count) + continue_ilp - 1ull)
									/ continue_ilp) * continue_ilp;
								continuation_slots += cont_slots;
								const uint32_t continue_grid = std::max<uint32_t>(1u,
									static_cast<uint32_t>((static_cast<uint64_t>(cur_count)
										+ static_cast<uint64_t>(continue_warps) * continue_ilp - 1ull)
										/ (static_cast<uint64_t>(continue_warps) * continue_ilp)));
								uint32_t continue_start_map = prev_boundary + 1u;
								void* cont_state_args[] = { &cur_live, &cur_count, &wave_state, &wave_flags,
									&off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
									&assets.schedule_data, &assets.carnival_data, &continue_start_map };
								const auto k0 = std::chrono::high_resolution_clock::now();
								if (cur_count != 0u)
								{
									check_cuda(cuLaunchKernel(raceway_continue_k, continue_grid, 1, 1,
										continue_threads, 1, 1, 0, 0, cont_state_args, nullptr), "launch(raceway_wave_cadence_continue)");
								}
								check_cuda(cuCtxSynchronize(), "sync(raceway_wave_cadence_continue)");
								const auto k1 = std::chrono::high_resolution_clock::now();
								continue_ms_total += std::chrono::duration<double, std::milli>(k1 - k0).count();

								RacewayStatsHost bs;
								check_cuda(cuMemcpyDtoH(&bs, wave_stats, sizeof(bs)), "DtoH(raceway_wave_cadence_stats)");
								total_stats.reps_started += bs.reps_started;
								total_stats.reps_completed += bs.reps_completed;
								total_stats.reps_dropped += bs.reps_dropped;
								total_stats.map_evals += bs.map_evals;
								total_survivors += cur_count;
								min_survivors = std::min<uint32_t>(min_survivors, cur_count);
								max_survivors = std::max<uint32_t>(max_survivors, cur_count);
								if (cur_count == 0u) ++empty_waves;

								if (args.raceway_drop_hist)
								{
									check_cuda(cuMemcpyDtoH(h_drop.data(), wave_drop_map, batch_n), "DtoH(raceway_wave_cadence_drop)");
									for (uint32_t i = 0u; i < batch_n; ++i)
									{
										if (h_drop[i] == 0xFFu) continue;
										++hist_total;
										if (h_drop[i] < hist.size()) ++hist[h_drop[i]];
									}
								}
								if (args.raceway_direct_wave_parity)
								{
									check_cuda(cuMemcpyDtoH(h_live.data(), cur_live, static_cast<size_t>(cur_count) * sizeof(uint32_t)), "DtoH(raceway_wave_cadence_live)");
									check_cuda(cuMemsetD8(wave_flat_flags, 0u, batch_n), "memset(raceway_wave_cadence_flat)");
									void* flat_args[] = { &wave_flat_flags, &off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
										&assets.expansion_values, &assets.schedule_data, &assets.carnival_data,
										&args.key_id, (void*)&batch_data_start, &batch_n_arg };
									const auto pf0 = std::chrono::high_resolution_clock::now();
									check_cuda(cuLaunchKernel(screen_offset_ilp6_preids_kernel, screen_kernel_grid_x_ilp(batch_n, 6u), 1, 1,
										kCudaThreadsPerBlock, 1, 1, 0, 0, flat_args, nullptr), "launch(raceway_wave_cadence_flat_screen)");
									check_cuda(cuCtxSynchronize(), "sync(raceway_wave_cadence_flat_screen)");
									const auto pf1 = std::chrono::high_resolution_clock::now();
									flat_ms_total += std::chrono::duration<double, std::milli>(pf1 - pf0).count();
									check_cuda(cuMemcpyDtoH(h_cont.data(), wave_flags, batch_n), "DtoH(raceway_wave_cadence_flags)");
									check_cuda(cuMemcpyDtoH(h_flat.data(), wave_flat_flags, batch_n), "DtoH(raceway_wave_cadence_flat_flags)");
									for (uint32_t i = 0u; i < cur_count; ++i)
									{
										const uint32_t orig = h_live[i];
										++parity_checked;
										cont_sum += h_cont[orig];
										flat_sum += h_flat[orig];
										if (h_cont[orig] != h_flat[orig]) ++parity_diff;
									}
								}

								done += batch_n;
								++batches;
							}

							if (min_survivors == 0xFFFFFFFFu) min_survivors = 0u;
							for (size_t bi = 0; bi < boundary_stats.size(); ++bi)
								boundary_stats[bi].cap_occupied = count_cap_occupied(cap_tables_h[bi], boundary_stats[bi].cap_slots);
							const double pipeline_ms = cap_ms_total + compact_ms_total + continue_ms_total;
							const double avg_survivors = batches ? static_cast<double>(total_survivors) / static_cast<double>(batches) : 0.0;
							const double wave_fill = static_cast<double>(total_survivors) / static_cast<double>(direct_total);
							const double cont_fill = continuation_slots ? static_cast<double>(total_survivors) / static_cast<double>(continuation_slots) : 1.0;
							std::cout << "  cadence=" << plan.first
							          << " completed_maps=" << completed_map_string(plan.second)
							          << " drains=" << plan.second.size()
							          << " survivors=" << total_survivors
							          << " dropped=" << total_stats.reps_dropped
							          << " map_evals=" << total_stats.map_evals
							          << " cap_spans=" << std::fixed << std::setprecision(3) << cap_ms_total << " ms"
							          << " compact=" << compact_ms_total << " ms"
							          << " continue=" << continue_ms_total << " ms"
							          << " pipeline=" << pipeline_ms << " ms ("
							          << std::setprecision(1) << (static_cast<double>(direct_total) / 1e6 / (pipeline_ms / 1e3))
							          << " M input/s)"
							          << " avg_survivors=" << avg_survivors
							          << " min=" << min_survivors
							          << " max=" << max_survivors
							          << " empty=" << empty_waves
							          << " wave_fill=" << std::setprecision(3) << wave_fill
							          << " continuation_slot_fill=" << cont_fill;
							if (args.raceway_direct_wave_parity)
								std::cout << " parity_diff=" << parity_diff
								          << (parity_diff == 0ull ? " [MATCH]" : " [DIFFER]")
								          << " checked=" << parity_checked
								          << " flat=" << std::setprecision(3) << flat_ms_total << " ms";
							std::cout << "\n";
							std::cout << "    boundary pressure:\n";
							for (const RacewayBoundaryStats& b : boundary_stats)
							{
								const double drop_rate = b.input ? static_cast<double>(b.dropped) / static_cast<double>(b.input) : 0.0;
								const double load = b.cap_slots ? static_cast<double>(b.cap_occupied) / static_cast<double>(b.cap_slots) : 0.0;
								std::cout << "      map" << b.completed_map
								          << " input=" << b.input
								          << " dropped=" << b.dropped
								          << " survivors=" << b.survivors
								          << " drop_rate=" << std::fixed << std::setprecision(3) << drop_rate
								          << " map_evals=" << b.map_evals
								          << " span=" << b.span_ms << " ms"
								          << " compact=" << b.compact_ms << " ms"
								          << " cap_occ=" << b.cap_occupied << "/" << b.cap_slots
								          << " load=" << load << "\n";
							}
							if (args.raceway_drop_hist)
							{
								std::cout << "    drop histogram:";
								for (uint32_t m = 0u; m < hist.size(); ++m)
									if (hist[m] != 0ull) std::cout << " m" << m << "=" << hist[m];
								std::cout << " total=" << hist_total << "\n";
							}
						}

						cuMemFree(wave_work_counter);
						cuMemFree(wave_stats);
						cuMemFree(wave_alive);
						cuMemFree(wave_live_a);
						cuMemFree(wave_live_b);
						cuMemFree(wave_compact_counter);
						cuMemFree(wave_state);
						cuMemFree(wave_occ_counter);
						if (wave_flags) cuMemFree(wave_flags);
						if (wave_flat_flags) cuMemFree(wave_flat_flags);
						if (wave_drop_map) cuMemFree(wave_drop_map);
						if (d_cap_tables) cuMemFree(d_cap_tables);
						if (d_cap_bits) cuMemFree(d_cap_bits);
						if (d_cap_ways) cuMemFree(d_cap_ways);
						if (d_direct_ostream) cuMemFree(d_direct_ostream);
						for (CUdeviceptr p : cap_tables_h) if (p) cuMemFree(p);
						release_assets(assets);
						return 0;
					}

					if (args.raceway_direct_wave_continue_batch != 0u)
					{
						const uint32_t wave_cap = std::min<uint32_t>(args.raceway_direct_wave_continue_batch, direct_n);
						CUdeviceptr wave_work_counter = 0, wave_stats = 0, wave_alive = 0;
						CUdeviceptr wave_live = 0, wave_compact_counter = 0, wave_flags = 0, wave_drop_map = 0;
						CUdeviceptr wave_flat_flags = 0, wave_state = 0;
						check_cuda(cuMemAlloc(&wave_work_counter, sizeof(uint32_t)), "cuMemAlloc(raceway_wave_work_counter)");
						check_cuda(cuMemAlloc(&wave_stats, sizeof(RacewayStatsHost)), "cuMemAlloc(raceway_wave_stats)");
						check_cuda(cuMemAlloc(&wave_alive, wave_cap), "cuMemAlloc(raceway_wave_alive)");
						check_cuda(cuMemAlloc(&wave_live, static_cast<size_t>(wave_cap) * sizeof(uint32_t)), "cuMemAlloc(raceway_wave_live)");
						check_cuda(cuMemAlloc(&wave_compact_counter, sizeof(uint32_t)), "cuMemAlloc(raceway_wave_compact_counter)");
						if (args.raceway_direct_wave_parity)
						{
							check_cuda(cuMemAlloc(&wave_flags, wave_cap), "cuMemAlloc(raceway_wave_flags)");
							check_cuda(cuMemAlloc(&wave_flat_flags, wave_cap), "cuMemAlloc(raceway_wave_flat_flags)");
						}
						if (args.raceway_direct_wave_state)
							check_cuda(cuMemAlloc(&wave_state, static_cast<size_t>(wave_cap) * 32u * sizeof(uint32_t)), "cuMemAlloc(raceway_wave_state)");
						if (args.raceway_drop_hist)
							check_cuda(cuMemAlloc(&wave_drop_map, wave_cap), "cuMemAlloc(raceway_wave_drop_map)");
						sample_race_mem();
						uint64_t wave_device_bytes = sizeof(uint32_t) + sizeof(RacewayStatsHost)
							+ static_cast<uint64_t>(wave_cap)
							+ static_cast<uint64_t>(wave_cap) * sizeof(uint32_t)
							+ sizeof(uint32_t);
						if (args.raceway_direct_wave_parity)
							wave_device_bytes += static_cast<uint64_t>(wave_cap) * 2ull;
						if (args.raceway_direct_wave_state)
							wave_device_bytes += static_cast<uint64_t>(wave_cap) * 32ull * sizeof(uint32_t);
						if (args.raceway_drop_hist)
							wave_device_bytes += static_cast<uint64_t>(wave_cap);

						RacewayStatsHost total_stats;
						double mark_ms_total = 0.0;
						double compact_ms_total = 0.0;
						double continue_ms_total = 0.0;
						double flat_ms_total = 0.0;
						uint64_t total_survivors = 0ull;
						uint64_t continuation_slots = 0ull;
						uint64_t parity_checked = 0ull;
						uint64_t parity_diff = 0ull;
						uint64_t cont_sum = 0ull;
						uint64_t flat_sum = 0ull;
						uint64_t hist_total = 0ull;
						std::vector<uint64_t> hist(schedule_count, 0ull);
						uint32_t min_survivors = 0xFFFFFFFFu;
						uint32_t max_survivors = 0u;
						uint32_t empty_waves = 0u;
						uint32_t batches = 0u;
						const uint32_t race_threads_per_block = 256u;
						const uint32_t race_warps_per_block = race_threads_per_block / kCudaWarpSize;
						const uint32_t continue_threads = 256u;
						const uint32_t continue_warps = continue_threads / kCudaWarpSize;

						std::vector<uint8_t> h_alive, h_cont, h_flat, h_drop;
						if (args.raceway_direct_wave_parity)
						{
							h_alive.resize(wave_cap);
							h_cont.resize(wave_cap);
							h_flat.resize(wave_cap);
						}
						if (args.raceway_drop_hist) h_drop.resize(wave_cap);

						for (uint32_t done = 0u; done < direct_n; )
						{
							const uint32_t batch_n = std::min<uint32_t>(args.raceway_direct_wave_continue_batch, direct_n - done);
							const uint32_t batch_data_start = direct_data_start + done;
							check_cuda(cuMemsetD32(wave_work_counter, 0u, 1u), "memset(raceway_wave_work_counter)");
							check_cuda(cuMemsetD8(wave_stats, 0u, sizeof(RacewayStatsHost)), "memset(raceway_wave_stats)");
							check_cuda(cuMemsetD32(wave_compact_counter, 0u, 1u), "memset(raceway_wave_compact_counter)");
							if (wave_flags) check_cuda(cuMemsetD8(wave_flags, 0u, batch_n), "memset(raceway_wave_flags)");
							if (wave_drop_map) check_cuda(cuMemsetD8(wave_drop_map, 0xFFu, batch_n), "memset(raceway_wave_drop_map)");

							uint32_t batch_blocks = static_cast<uint32_t>((static_cast<uint64_t>(batch_n) + race_warps_per_block - 1ull)
								/ race_warps_per_block);
							batch_blocks = std::min<uint32_t>(8192u, std::max<uint32_t>(1u, batch_blocks));
							uint32_t batch_cap_count = cap_count;
							uint32_t batch_schedule_count = schedule_count;
							uint32_t batch_first_cap_map = args.raceway_first_cap_map;
							CUdeviceptr wave_drop_arg = wave_drop_map;
							void* ma_mark[] = {
								(void*)&batch_n, (void*)&batch_data_start, &wave_work_counter, &wave_alive, &wave_drop_arg, &wave_stats,
								&d_cap_tables, &d_cap_bits, &d_cap_ways, &batch_cap_count,
								&off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
								&assets.expansion_values, &assets.schedule_data, &assets.carnival_data, &args.key_id,
								&batch_schedule_count, &batch_first_cap_map };
							void* ma_state[] = {
								(void*)&batch_n, (void*)&batch_data_start, &wave_work_counter, &wave_alive, &wave_drop_arg, &wave_stats,
								&wave_state, &d_cap_tables, &d_cap_bits, &d_cap_ways, &batch_cap_count,
								&off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
								&assets.expansion_values, &assets.schedule_data, &args.key_id,
								&batch_schedule_count, &batch_first_cap_map };
							void** mark_args = args.raceway_direct_wave_state ? ma_state : ma_mark;
							check_cuda(cuCtxSynchronize(), "sync(raceway_wave_mark_pre)");
							const auto m0 = std::chrono::high_resolution_clock::now();
							check_cuda(cuLaunchKernel(raceway_direct_k, batch_blocks, 1, 1,
								race_threads_per_block, 1, 1, 0, 0, mark_args, nullptr), "launch(raceway_wave_mark)");
							check_cuda(cuCtxSynchronize(), "sync(raceway_wave_mark)");
							const auto m1 = std::chrono::high_resolution_clock::now();
							mark_ms_total += std::chrono::duration<double, std::milli>(m1 - m0).count();

							uint32_t batch_n_arg = batch_n;
							CUdeviceptr null_live_in = 0;
							int compact_first = 1;
							void* ca[] = { &wave_alive, &null_live_in, &batch_n_arg, &wave_live, &wave_compact_counter, &compact_first };
							check_cuda(cuCtxSynchronize(), "sync(raceway_wave_compact_pre)");
							const auto c0 = std::chrono::high_resolution_clock::now();
							check_cuda(cuLaunchKernel(raceway_compact_k, (batch_n + 255u) / 256u, 1, 1,
								256, 1, 1, 0, 0, ca, nullptr), "launch(raceway_wave_compact)");
							check_cuda(cuCtxSynchronize(), "sync(raceway_wave_compact)");
							const auto c1 = std::chrono::high_resolution_clock::now();
							compact_ms_total += std::chrono::duration<double, std::milli>(c1 - c0).count();

							uint32_t compacted = 0u;
							check_cuda(cuMemcpyDtoH(&compacted, wave_compact_counter, sizeof(uint32_t)), "DtoH(raceway_wave_compact_counter)");
							RacewayStatsHost bs;
							check_cuda(cuMemcpyDtoH(&bs, wave_stats, sizeof(bs)), "DtoH(raceway_wave_stats)");
							total_stats.reps_started += bs.reps_started;
							total_stats.reps_completed += bs.reps_completed;
							total_stats.reps_dropped += bs.reps_dropped;
							total_stats.map_evals += bs.map_evals;
							total_survivors += compacted;
							min_survivors = std::min<uint32_t>(min_survivors, compacted);
							max_survivors = std::max<uint32_t>(max_survivors, compacted);
							if (compacted == 0u) ++empty_waves;

							const uint32_t continue_ilp = selected_wave_continue_ilp;
							const uint64_t cont_slots = ((static_cast<uint64_t>(compacted) + continue_ilp - 1ull)
								/ continue_ilp) * continue_ilp;
							continuation_slots += cont_slots;
							const uint32_t continue_grid = std::max<uint32_t>(1u,
								static_cast<uint32_t>((static_cast<uint64_t>(compacted)
									+ static_cast<uint64_t>(continue_warps) * continue_ilp - 1ull)
									/ (static_cast<uint64_t>(continue_warps) * continue_ilp)));
							uint32_t continue_start_map = std::min<uint32_t>(batch_schedule_count, batch_first_cap_map + batch_cap_count);
							void* cont_args[] = { &wave_live, &compacted, (void*)&batch_data_start, &wave_flags,
								&off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
								&assets.expansion_values, &assets.schedule_data, &assets.carnival_data, &args.key_id };
							void* cont_state_args[] = { &wave_live, &compacted, &wave_state, &wave_flags,
								&off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
								&assets.schedule_data, &assets.carnival_data, &continue_start_map };
							void** continue_args = args.raceway_direct_wave_state ? cont_state_args : cont_args;
							check_cuda(cuCtxSynchronize(), "sync(raceway_wave_continue_pre)");
							const auto k0 = std::chrono::high_resolution_clock::now();
							if (compacted != 0u)
							{
								check_cuda(cuLaunchKernel(raceway_continue_k, continue_grid, 1, 1,
									continue_threads, 1, 1, 0, 0, continue_args, nullptr), "launch(raceway_wave_continue)");
							}
							check_cuda(cuCtxSynchronize(), "sync(raceway_wave_continue)");
							const auto k1 = std::chrono::high_resolution_clock::now();
							continue_ms_total += std::chrono::duration<double, std::milli>(k1 - k0).count();

							if (args.raceway_drop_hist)
							{
								check_cuda(cuMemcpyDtoH(h_drop.data(), wave_drop_map, batch_n), "DtoH(raceway_wave_drop)");
								for (uint32_t i = 0u; i < batch_n; ++i)
								{
									if (h_drop[i] == 0xFFu) continue;
									++hist_total;
									if (h_drop[i] < hist.size()) ++hist[h_drop[i]];
								}
							}

							if (args.raceway_direct_wave_parity)
							{
								check_cuda(cuMemsetD8(wave_flat_flags, 0u, batch_n), "memset(raceway_wave_flat_flags)");
								void* flat_args[] = { &wave_flat_flags, &off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
									&assets.expansion_values, &assets.schedule_data, &assets.carnival_data,
									&args.key_id, (void*)&batch_data_start, &batch_n_arg };
								const auto f0 = std::chrono::high_resolution_clock::now();
								check_cuda(cuLaunchKernel(screen_offset_ilp6_preids_kernel, screen_kernel_grid_x_ilp(batch_n, 6u), 1, 1,
									kCudaThreadsPerBlock, 1, 1, 0, 0, flat_args, nullptr), "launch(raceway_wave_flat_screen)");
								check_cuda(cuCtxSynchronize(), "sync(raceway_wave_flat_screen)");
								const auto f1 = std::chrono::high_resolution_clock::now();
								flat_ms_total += std::chrono::duration<double, std::milli>(f1 - f0).count();
								check_cuda(cuMemcpyDtoH(h_alive.data(), wave_alive, batch_n), "DtoH(raceway_wave_alive)");
								check_cuda(cuMemcpyDtoH(h_cont.data(), wave_flags, batch_n), "DtoH(raceway_wave_flags)");
								check_cuda(cuMemcpyDtoH(h_flat.data(), wave_flat_flags, batch_n), "DtoH(raceway_wave_flat_flags)");
								for (uint32_t i = 0u; i < batch_n; ++i)
								{
									if (h_alive[i] == 0u) continue;
									++parity_checked;
									cont_sum += h_cont[i];
									flat_sum += h_flat[i];
									if (h_cont[i] != h_flat[i]) ++parity_diff;
								}
							}

							done += batch_n;
							++batches;
						}

						if (min_survivors == 0xFFFFFFFFu) min_survivors = 0u;
						const double pipeline_ms = mark_ms_total + compact_ms_total + continue_ms_total;
						std::cout << "[raceway-wave-continue] device=" << device_name
						          << " key=0x" << std::hex << std::setw(8) << std::setfill('0') << args.key_id
						          << std::dec << std::setfill(' ')
						          << " mode=offset"
						          << (args.raceway_direct_wave_state ? " state=1" : "")
						          << " batch=" << args.raceway_direct_wave_continue_batch
						          << " batches=" << batches
						          << " window=" << direct_n
						          << " data_start=" << direct_data_start << "\n";
						std::cout << "  memory: observed_peak_used=" << std::fixed << std::setprecision(1)
						          << mib(static_cast<uint64_t>(race_mem_total - race_mem_min_free)) << " MiB"
						          << " delta_from_race_start="
						          << mib(static_cast<uint64_t>(race_mem_start_free - race_mem_min_free)) << " MiB"
						          << " expected_device_alloc="
						          << mib(cap_slots_total * sizeof(unsigned long long)
						                 + static_cast<uint64_t>(osb.data.size()) + wave_device_bytes) << " MiB"
						          << " (caps=" << mib(cap_slots_total * sizeof(unsigned long long))
						          << ", wave=" << mib(wave_device_bytes)
						          << ", offset_stream=" << mib(static_cast<uint64_t>(osb.data.size())) << ")\n";
						std::cout << "  caps=" << cap_count
						          << " first_map=" << args.raceway_first_cap_map
						          << " cap_shape=";
						if (cap_count == 0u)
							std::cout << "none";
						else
							std::cout << "2^" << args.raceway_cap_bits << "x" << args.raceway_cap_ways
							          << " (" << (cap_slots_total * sizeof(unsigned long long) / (1ull << 20)) << " MB)";
						std::cout << "  candidates=" << direct_n
						          << "  survivors=" << total_survivors
						          << "  dropped=" << total_stats.reps_dropped
						          << "  map_evals=" << total_stats.map_evals << "\n";
						std::cout << "  mark=" << std::fixed << std::setprecision(3) << mark_ms_total << " ms"
						          << "  compact=" << compact_ms_total << " ms"
						          << "  continue=" << continue_ms_total << " ms"
						          << "  pipeline=" << pipeline_ms << " ms  ("
						          << std::setprecision(1) << (static_cast<double>(direct_n) / 1e6 / (pipeline_ms / 1e3))
						          << " M input/s)\n";
						const double avg_survivors = batches ? static_cast<double>(total_survivors) / static_cast<double>(batches) : 0.0;
						const double wave_fill = static_cast<double>(total_survivors) / static_cast<double>(direct_n);
						const double cont_fill = continuation_slots ? static_cast<double>(total_survivors) / static_cast<double>(continuation_slots) : 1.0;
						std::cout << "  wave stocking: avg_survivors=" << std::fixed << std::setprecision(1) << avg_survivors
						          << " min=" << min_survivors
						          << " max=" << max_survivors
						          << " empty=" << empty_waves
						          << " wave_fill=" << std::setprecision(3) << wave_fill
						          << " continuation_slot_fill=" << cont_fill << "\n";
						if (args.raceway_direct_wave_parity)
						{
							std::cout << "  alive-parity: checked=" << parity_checked
							          << " flag_diff=" << parity_diff
							          << (parity_diff == 0ull ? " [MATCH]" : " [DIFFER]")
							          << " flag_sum(continue)=" << cont_sum
							          << " flag_sum(flat)=" << flat_sum
							          << " flat=" << std::fixed << std::setprecision(3) << flat_ms_total << " ms\n";
						}
						if (args.raceway_drop_hist)
						{
							std::cout << "  raceway-wave drop histogram:";
							for (uint32_t m = 0u; m < hist.size(); ++m)
								if (hist[m] != 0ull) std::cout << " m" << m << "=" << hist[m];
							std::cout << " total=" << hist_total << "\n";
						}

						cuMemFree(wave_work_counter);
						cuMemFree(wave_stats);
						cuMemFree(wave_alive);
						cuMemFree(wave_live);
						cuMemFree(wave_compact_counter);
						if (wave_flags) cuMemFree(wave_flags);
						if (wave_flat_flags) cuMemFree(wave_flat_flags);
						if (wave_state) cuMemFree(wave_state);
						if (wave_drop_map) cuMemFree(wave_drop_map);
						if (d_cap_tables) cuMemFree(d_cap_tables);
						if (d_cap_bits) cuMemFree(d_cap_bits);
						if (d_cap_ways) cuMemFree(d_cap_ways);
						if (d_direct_ostream) cuMemFree(d_direct_ostream);
						for (CUdeviceptr p : cap_tables_h) if (p) cuMemFree(p);
						release_assets(assets);
						return 0;
					}

					if (args.raceway_direct_stream_batch != 0u)
					{
						if (!args.raceway_direct_offset || args.raceway_direct_ilp != 1u || args.raceway_direct_flat_parity
							|| args.raceway_direct_cap_mark || args.raceway_direct_static)
							throw std::runtime_error("--raceway-direct-stream-batch requires direct offset ILP1 without flat parity/cap-mark/static modes");

						CUdeviceptr batch_work_counter = 0, batch_stats = 0;
						check_cuda(cuMemAlloc(&batch_work_counter, sizeof(uint32_t)), "cuMemAlloc(raceway_batch_work_counter)");
						check_cuda(cuMemAlloc(&batch_stats, sizeof(RacewayStatsHost)), "cuMemAlloc(raceway_batch_stats)");

						RacewayStatsHost total_stats;
						double total_ms = 0.0;
						uint32_t batches = 0u;
						const uint32_t race_threads_per_block = 256u;
						for (uint32_t done = 0u; done < direct_n; )
						{
							const uint32_t batch_n = std::min<uint32_t>(args.raceway_direct_stream_batch, direct_n - done);
							const uint32_t batch_data_start = direct_data_start + done;
							check_cuda(cuMemsetD32(batch_work_counter, 0u, 1u), "memset(raceway_batch_work_counter)");
							check_cuda(cuMemsetD8(batch_stats, 0u, sizeof(RacewayStatsHost)), "memset(raceway_batch_stats)");
							uint32_t batch_blocks = static_cast<uint32_t>((static_cast<uint64_t>(batch_n) + static_cast<uint64_t>(race_threads_per_block / kCudaWarpSize) - 1ull)
								/ static_cast<uint64_t>(race_threads_per_block / kCudaWarpSize));
							batch_blocks = std::min<uint32_t>(8192u, std::max<uint32_t>(1u, batch_blocks));
							uint32_t batch_cap_count = cap_count;
							uint32_t batch_schedule_count = schedule_count;
							uint32_t batch_first_cap_map = args.raceway_first_cap_map;
							CUdeviceptr null_u8 = 0;
							void* ba[] = {
								(void*)&batch_n, (void*)&batch_data_start, &batch_work_counter, &null_u8, &null_u8, &batch_stats,
								&d_cap_tables, &d_cap_bits, &d_cap_ways, &batch_cap_count,
								&off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
								&assets.expansion_values, &assets.schedule_data, &assets.carnival_data, &args.key_id,
								&batch_schedule_count, &batch_first_cap_map };
							check_cuda(cuCtxSynchronize(), "sync(raceway_batch_pre)");
							const auto b0 = std::chrono::high_resolution_clock::now();
							check_cuda(cuLaunchKernel(raceway_direct_k, batch_blocks, 1, 1,
								race_threads_per_block, 1, 1, 0, 0, ba, nullptr), "launch(raceway_batch)");
							check_cuda(cuCtxSynchronize(), "sync(raceway_batch)");
							const auto b1 = std::chrono::high_resolution_clock::now();
							RacewayStatsHost bs;
							check_cuda(cuMemcpyDtoH(&bs, batch_stats, sizeof(bs)), "DtoH(raceway_batch_stats)");
							total_stats.reps_started += bs.reps_started;
							total_stats.reps_completed += bs.reps_completed;
							total_stats.reps_dropped += bs.reps_dropped;
							total_stats.map_evals += bs.map_evals;
							total_ms += std::chrono::duration<double, std::milli>(b1 - b0).count();
							done += batch_n;
							++batches;
						}

						std::cout << "[raceway-stream-batch] device=" << device_name
						          << " key=0x" << std::hex << std::setw(8) << std::setfill('0') << args.key_id
						          << std::dec << std::setfill(' ')
						          << " mode=offset"
						          << " batch=" << args.raceway_direct_stream_batch
						          << " batches=" << batches
						          << " window=" << direct_n
						          << " data_start=" << direct_data_start << "\n";
						std::cout << "  caps=" << cap_count
						          << " first_map=" << args.raceway_first_cap_map
						          << " cap_shape=";
						if (cap_count == 0u)
							std::cout << "none";
						else
							std::cout << "2^" << args.raceway_cap_bits << "x" << args.raceway_cap_ways
							          << " (" << (cap_slots_total * sizeof(unsigned long long) / (1ull << 20)) << " MB)";
						std::cout << "  candidates=" << direct_n
						          << "  completed=" << total_stats.reps_completed
						          << "  dropped=" << total_stats.reps_dropped
						          << "  map_evals=" << total_stats.map_evals
						          << "  time=" << std::fixed << std::setprecision(3) << total_ms << " ms  ("
						          << std::setprecision(1) << (static_cast<double>(direct_n) / 1e6 / (total_ms / 1e3))
						          << " M cand/s)\n";

						cuMemFree(batch_work_counter);
						cuMemFree(batch_stats);
						if (d_cap_tables) cuMemFree(d_cap_tables);
						if (d_cap_bits) cuMemFree(d_cap_bits);
						if (d_cap_ways) cuMemFree(d_cap_ways);
						if (d_direct_ostream) cuMemFree(d_direct_ostream);
						for (CUdeviceptr p : cap_tables_h) if (p) cuMemFree(p);
						release_assets(assets);
						return 0;
					}

					CUdeviceptr work_counter = 0, race_stats = 0, race_flags = 0, race_drop_map = 0;
					CUdeviceptr compact_live = 0, compact_counter = 0;
					CUdeviceptr continue_flags = 0, race_state = 0;
					check_cuda(cuMemAlloc(&work_counter, sizeof(uint32_t)), "cuMemAlloc(raceway_direct_work_counter)");
					check_cuda(cuMemAlloc(&race_stats, sizeof(RacewayStatsHost)), "cuMemAlloc(raceway_direct_stats)");
					if (args.raceway_direct_flat_parity || cap_count == 0u || args.raceway_direct_cap_mark)
						check_cuda(cuMemAlloc(&race_flags, direct_n), "cuMemAlloc(raceway_direct_flags)");
					if (args.raceway_drop_hist || args.raceway_direct_cap_mark)
					{
						check_cuda(cuMemAlloc(&race_drop_map, direct_n), "cuMemAlloc(raceway_direct_drop_map)");
						check_cuda(cuMemsetD8(race_drop_map, 0xFFu, direct_n), "memset(raceway_direct_drop_map)");
					}
					if (args.raceway_direct_cap_compact)
					{
						check_cuda(cuMemAlloc(&compact_live, static_cast<size_t>(direct_n) * sizeof(uint32_t)), "cuMemAlloc(raceway_compact_live)");
						check_cuda(cuMemAlloc(&compact_counter, sizeof(uint32_t)), "cuMemAlloc(raceway_compact_counter)");
					}
					if (args.raceway_direct_cap_continue)
					{
						check_cuda(cuMemAlloc(&continue_flags, direct_n), "cuMemAlloc(raceway_continue_flags)");
						check_cuda(cuMemsetD8(continue_flags, 0u, direct_n), "memset(raceway_continue_flags)");
					}
					if (args.raceway_direct_cap_state_continue)
					{
						const size_t state_bytes = static_cast<size_t>(direct_n) * 32u * sizeof(uint32_t);
						check_cuda(cuMemAlloc(&race_state, state_bytes), "cuMemAlloc(raceway_state)");
					}
					check_cuda(cuMemsetD32(work_counter, 0u, 1u), "memset(raceway_direct_work_counter)");
					check_cuda(cuMemsetD8(race_stats, 0u, sizeof(RacewayStatsHost)), "memset(raceway_direct_stats)");

					uint32_t race_cap_count = cap_count;
					uint32_t race_schedule_count = schedule_count;
					uint32_t race_first_cap_map = args.raceway_first_cap_map;
					CUdeviceptr race_drop_arg = (args.raceway_drop_hist || args.raceway_direct_cap_mark) ? race_drop_map : 0;
					const uint32_t race_items_per_warp = args.raceway_direct_ilp;
					const uint32_t race_threads_per_block = 256u;
					const uint32_t race_warps_per_block = race_threads_per_block / kCudaWarpSize;
					uint32_t race_blocks = static_cast<uint32_t>((static_cast<uint64_t>(direct_n) + static_cast<uint64_t>(race_warps_per_block) * race_items_per_warp - 1ull)
						/ (static_cast<uint64_t>(race_warps_per_block) * race_items_per_warp));
					if (!args.raceway_direct_static)
						race_blocks = std::min<uint32_t>(8192u, race_blocks);
					race_blocks = std::max<uint32_t>(1u, race_blocks);
					void* ra_universal[] = {
						&direct_n, (void*)&direct_data_start, &work_counter, &race_flags, &race_drop_arg, &race_stats,
						&d_cap_tables, &d_cap_bits, &d_cap_ways, &race_cap_count,
						&assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
						&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
						&assets.alg2_values, &assets.alg5_values, &assets.expansion_values,
						&assets.schedule_data, &assets.carnival_data, &args.key_id,
						&race_schedule_count, &race_first_cap_map };
					void* ra_offset[] = {
						&direct_n, (void*)&direct_data_start, &work_counter, &race_flags, &race_drop_arg, &race_stats,
						&d_cap_tables, &d_cap_bits, &d_cap_ways, &race_cap_count,
						&off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
						&assets.expansion_values, &assets.schedule_data, &assets.carnival_data, &args.key_id,
						&race_schedule_count, &race_first_cap_map };
					void* ra_offset_state[] = {
						&direct_n, (void*)&direct_data_start, &work_counter, &race_flags, &race_drop_arg, &race_stats,
						&race_state, &d_cap_tables, &d_cap_bits, &d_cap_ways, &race_cap_count,
						&off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
						&assets.expansion_values, &assets.schedule_data, &args.key_id,
						&race_schedule_count, &race_first_cap_map };
					void** ra = args.raceway_direct_cap_state_continue ? ra_offset_state
						: (args.raceway_direct_offset ? ra_offset : ra_universal);
					check_cuda(cuCtxSynchronize(), "sync(raceway_direct_pre)");
					const auto r0 = std::chrono::high_resolution_clock::now();
					check_cuda(cuLaunchKernel(raceway_direct_k, race_blocks, 1, 1, race_threads_per_block, 1, 1, 0, 0, ra, nullptr), "launch(raceway_direct)");
					check_cuda(cuCtxSynchronize(), "sync(raceway_direct)");
					const auto r1 = std::chrono::high_resolution_clock::now();

					RacewayStatsHost hs;
					check_cuda(cuMemcpyDtoH(&hs, race_stats, sizeof(hs)), "DtoH(raceway_direct_stats)");
					if (args.raceway_direct_static)
					{
						hs.reps_started = direct_n;
						hs.reps_completed = direct_n;
						hs.reps_dropped = 0ull;
						hs.map_evals = static_cast<unsigned long long>(direct_n) * 27ull;
					}
					const double race_ms = std::chrono::duration<double, std::milli>(r1 - r0).count();
					std::cout << "[raceway-direct] device=" << device_name
					          << " key=0x" << std::hex << std::setw(8) << std::setfill('0') << args.key_id
					          << std::dec << std::setfill(' ')
					          << " mode=" << (args.raceway_direct_offset ? "offset" : "universal")
					          << " ilp=" << args.raceway_direct_ilp
					          << (args.raceway_direct_cap_mark ? " cap_mark=1" : "")
					          << (args.raceway_direct_cap_state_continue ? " state=1" : "")
					          << (args.raceway_direct_static ? " static=1" : "")
					          << " window=" << direct_n
					          << " data_start=" << direct_data_start << "\n";
					std::cout << "  caps=" << cap_count
					          << " first_map=" << args.raceway_first_cap_map
					          << " cap_shape=";
					if (cap_count == 0u)
						std::cout << "none";
					else
						std::cout << "2^" << args.raceway_cap_bits << "x" << args.raceway_cap_ways
						          << " (" << (cap_slots_total * sizeof(unsigned long long) / (1ull << 20)) << " MB)";
					std::cout << "  candidates=" << direct_n
					          << "  completed=" << hs.reps_completed
					          << "  dropped=" << hs.reps_dropped
					          << "  map_evals=" << hs.map_evals
					          << "  time=" << std::fixed << std::setprecision(3) << race_ms << " ms  ("
					          << std::setprecision(1) << (static_cast<double>(direct_n) / 1e6 / (race_ms / 1e3))
					          << " M cand/s)\n";

					std::vector<uint8_t> h_alive;
					uint64_t alive_count = 0ull;
					if (args.raceway_direct_cap_mark)
					{
						h_alive.resize(direct_n);
						check_cuda(cuMemcpyDtoH(h_alive.data(), race_flags, direct_n), "DtoH(raceway_direct_alive)");
						uint64_t bad_alive = 0ull;
						for (uint32_t i = 0u; i < direct_n; ++i)
						{
							if (h_alive[i] == 1u) ++alive_count;
							else if (h_alive[i] != 0u) ++bad_alive;
						}
						std::cout << "  raceway-cap-mark alive=" << alive_count
						          << " dropped=" << (static_cast<uint64_t>(direct_n) - alive_count);
						if (bad_alive != 0ull) std::cout << " bad_alive_bytes=" << bad_alive;
						std::cout << "\n";
					}

					uint32_t compacted = 0u;
					double compact_ms = 0.0;
					if (args.raceway_direct_cap_compact)
					{
						check_cuda(cuMemsetD32(compact_counter, 0u, 1u), "memset(raceway_compact_counter)");
						CUdeviceptr null_live_in = 0;
						int compact_first = 1;
						void* ca[] = { &race_flags, &null_live_in, &direct_n, &compact_live, &compact_counter, &compact_first };
						check_cuda(cuCtxSynchronize(), "sync(raceway_compact_pre)");
						const auto c0 = std::chrono::high_resolution_clock::now();
						check_cuda(cuLaunchKernel(raceway_compact_k, (direct_n + 255u) / 256u, 1, 1,
							256, 1, 1, 0, 0, ca, nullptr), "launch(raceway_compact)");
						check_cuda(cuCtxSynchronize(), "sync(raceway_compact)");
						const auto c1 = std::chrono::high_resolution_clock::now();
						check_cuda(cuMemcpyDtoH(&compacted, compact_counter, sizeof(uint32_t)), "DtoH(raceway_compact_counter)");
						compact_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
						std::cout << "  raceway-cap-compact survivors=" << compacted
						          << "  time=" << std::fixed << std::setprecision(3) << compact_ms << " ms  ("
						          << std::setprecision(1) << (static_cast<double>(direct_n) / 1e6 / (compact_ms / 1e3))
						          << " M input/s)\n";
					}

					if (args.raceway_direct_cap_continue)
					{
						const uint32_t continue_threads = 256u;
						const uint32_t continue_warps = continue_threads / kCudaWarpSize;
						const uint32_t continue_grid = std::max<uint32_t>(1u,
							static_cast<uint32_t>((static_cast<uint64_t>(compacted) + static_cast<uint64_t>(continue_warps) * 6ull - 1ull)
								/ (static_cast<uint64_t>(continue_warps) * 6ull)));
						uint32_t continue_start_map = std::min<uint32_t>(race_schedule_count, race_first_cap_map + race_cap_count);
						void* cont_args[] = { &compact_live, &compacted, (void*)&direct_data_start, &continue_flags,
							&off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
							&assets.expansion_values, &assets.schedule_data, &assets.carnival_data, &args.key_id };
						void* cont_state_args[] = { &compact_live, &compacted, &race_state, &continue_flags,
							&off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
							&assets.schedule_data, &assets.carnival_data, &continue_start_map };
						void** cont_launch_args = args.raceway_direct_cap_state_continue ? cont_state_args : cont_args;
						check_cuda(cuCtxSynchronize(), "sync(raceway_continue_pre)");
						const auto k0 = std::chrono::high_resolution_clock::now();
						check_cuda(cuLaunchKernel(raceway_continue_k, continue_grid, 1, 1,
							continue_threads, 1, 1, 0, 0, cont_launch_args, nullptr), "launch(raceway_continue)");
						check_cuda(cuCtxSynchronize(), "sync(raceway_continue)");
						const auto k1 = std::chrono::high_resolution_clock::now();
						const double continue_ms = std::chrono::duration<double, std::milli>(k1 - k0).count();
						const double pipeline_ms = race_ms + compact_ms + continue_ms;
						std::cout << "  raceway-cap-continue screened=" << compacted
						          << "  time=" << std::fixed << std::setprecision(3) << continue_ms << " ms  ("
						          << std::setprecision(1) << (static_cast<double>(compacted) / 1e6 / (continue_ms / 1e3))
						          << " M survivor/s)"
						          << "  pipeline=" << std::setprecision(3) << pipeline_ms << " ms  ("
						          << std::setprecision(1) << (static_cast<double>(direct_n) / 1e6 / (pipeline_ms / 1e3))
						          << " M input/s)\n";

						CUdeviceptr d_flat_flags = 0;
						check_cuda(cuMemAlloc(&d_flat_flags, direct_n), "cuMemAlloc(raceway_continue_flat_flags)");
						void* flat_args[] = { &d_flat_flags, &off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
							&assets.expansion_values, &assets.schedule_data, &assets.carnival_data,
							&args.key_id, (void*)&direct_data_start, &direct_n };
						const auto f0 = std::chrono::high_resolution_clock::now();
						check_cuda(cuLaunchKernel(screen_offset_ilp6_preids_kernel, screen_kernel_grid_x_ilp(direct_n, 6u), 1, 1,
							kCudaThreadsPerBlock, 1, 1, 0, 0, flat_args, nullptr), "launch(raceway_continue_flat_screen)");
						check_cuda(cuCtxSynchronize(), "sync(raceway_continue_flat_screen)");
						const auto f1 = std::chrono::high_resolution_clock::now();
						const double flat_ms = std::chrono::duration<double, std::milli>(f1 - f0).count();

						std::vector<uint8_t> h_cont(direct_n), h_flat(direct_n);
						check_cuda(cuMemcpyDtoH(h_cont.data(), continue_flags, direct_n), "DtoH(raceway_continue_flags)");
						check_cuda(cuMemcpyDtoH(h_flat.data(), d_flat_flags, direct_n), "DtoH(raceway_continue_flat_flags)");
						uint64_t alive_diff = 0ull, cont_sum = 0ull, flat_sum = 0ull;
						uint32_t first_diff = 0u;
						for (uint32_t i = 0u; i < direct_n; ++i)
						{
							if (h_alive[i] == 0u) continue;
							cont_sum += h_cont[i];
							flat_sum += h_flat[i];
							if (h_cont[i] != h_flat[i])
							{
								if (alive_diff == 0ull) first_diff = i;
								++alive_diff;
							}
						}
						std::cout << "  raceway-cap-continue alive-parity:"
						          << " checked=" << alive_count
						          << " flag_diff=" << alive_diff
						          << (alive_diff == 0ull ? " [MATCH]" : " [DIFFER]")
						          << "  flag_sum(continue)=" << cont_sum
						          << "  flag_sum(flat)=" << flat_sum
						          << "  flat=" << std::fixed << std::setprecision(3) << flat_ms << " ms";
						if (alive_diff != 0ull) std::cout << "  first_diff=" << first_diff;
						std::cout << "\n";
						cuMemFree(d_flat_flags);
					}

					if (args.raceway_drop_hist || args.raceway_direct_cap_mark)
					{
						std::vector<uint8_t> h_drop(direct_n);
						check_cuda(cuMemcpyDtoH(h_drop.data(), race_drop_map, direct_n), "DtoH(raceway_direct_drop_map)");
						std::vector<uint64_t> hist(schedule_count, 0ull);
						uint64_t hist_total = 0ull, hist_oob = 0ull;
						for (uint32_t i = 0u; i < direct_n; ++i)
						{
							if (h_drop[i] == 0xFFu) continue;
							++hist_total;
							if (h_drop[i] < hist.size()) ++hist[h_drop[i]];
							else ++hist_oob;
						}
						std::cout << "  raceway-direct drop histogram:";
						for (uint32_t m = 0u; m < hist.size(); ++m)
							if (hist[m] != 0ull) std::cout << " m" << m << "=" << hist[m];
						std::cout << "  total=" << hist_total;
						if (hist_oob != 0ull) std::cout << " oob=" << hist_oob;
						std::cout << "\n";
					}

						if (args.raceway_direct_flat_parity)
						{
							CUdeviceptr d_flat_flags = 0;
							check_cuda(cuMemAlloc(&d_flat_flags, direct_n), "cuMemAlloc(raceway_direct_flat_flags)");

							void* flat_args[] = { &d_flat_flags, &off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
								&assets.expansion_values, &assets.schedule_data, &assets.carnival_data,
							&args.key_id, (void*)&direct_data_start, &direct_n };
						const auto f0 = std::chrono::high_resolution_clock::now();
						check_cuda(cuLaunchKernel(screen_offset_ilp6_preids_kernel, screen_kernel_grid_x_ilp(direct_n, 6u), 1, 1,
							kCudaThreadsPerBlock, 1, 1, 0, 0, flat_args, nullptr), "launch(raceway_direct_flat_screen)");
						check_cuda(cuCtxSynchronize(), "sync(raceway_direct_flat_screen)");
						const auto f1 = std::chrono::high_resolution_clock::now();
						const double flat_ms = std::chrono::duration<double, std::milli>(f1 - f0).count();

						std::vector<uint8_t> h_race(direct_n), h_flat(direct_n);
						check_cuda(cuMemcpyDtoH(h_race.data(), race_flags, direct_n), "DtoH(raceway_direct_flags)");
						check_cuda(cuMemcpyDtoH(h_flat.data(), d_flat_flags, direct_n), "DtoH(raceway_direct_flat_flags)");
						uint64_t diff = 0ull, race_sum = 0ull, flat_sum = 0ull;
						uint32_t first_diff = 0u;
						for (uint32_t i = 0u; i < direct_n; ++i)
						{
							race_sum += h_race[i];
							flat_sum += h_flat[i];
							if (h_race[i] != h_flat[i])
							{
								if (diff == 0ull) first_diff = i;
								++diff;
							}
						}
						std::cout << "  raceway-direct-vs-flat:"
						          << " flat=" << std::fixed << std::setprecision(3) << flat_ms << " ms ("
						          << std::setprecision(1) << (static_cast<double>(direct_n) / 1e6 / (flat_ms / 1e3)) << " M cand/s)"
						          << "  direct=" << std::setprecision(3) << race_ms << " ms ("
						          << std::setprecision(1) << (static_cast<double>(direct_n) / 1e6 / (race_ms / 1e3)) << " M cand/s)"
						          << "\n";
						std::cout << "  raceway-direct parity:"
						          << " checked=" << direct_n
						          << " flag_diff=" << diff
						          << (diff == 0ull ? " [MATCH]" : " [DIFFER]")
						          << "  flag_sum(raceway)=" << race_sum
						          << "  flag_sum(flat)=" << flat_sum;
						if (diff != 0ull) std::cout << "  first_diff=" << first_diff;
							std::cout << "\n";

							cuMemFree(d_flat_flags);
						}

					cuMemFree(work_counter);
					cuMemFree(race_stats);
					if (race_flags) cuMemFree(race_flags);
					if (race_drop_map) cuMemFree(race_drop_map);
					if (compact_live) cuMemFree(compact_live);
					if (compact_counter) cuMemFree(compact_counter);
					if (continue_flags) cuMemFree(continue_flags);
					if (race_state) cuMemFree(race_state);
					if (d_cap_tables) cuMemFree(d_cap_tables);
						if (d_cap_bits) cuMemFree(d_cap_bits);
						if (d_cap_ways) cuMemFree(d_cap_ways);
						if (d_direct_ostream) cuMemFree(d_direct_ostream);
						for (CUdeviceptr p : cap_tables_h) if (p) cuMemFree(p);
						return 0;
					}

				// ── MAP1 frontier emitter prototype (2026-06-03) ─────────────────────────
				// Streams the data axis in chunks while keeping one persistent strong64
			// fingerprint table. This is the first GPU W4B-frontier shape: storage is
			// proportional to MAP1 representatives/table budget, not to the input tile's
			// 128-byte states.
			if (args.map1_frontier)
			{
				if (use_skipcar)
					throw std::runtime_error("--map1-frontier currently supports --map-list all only");
				if (args.map1_frontier_partitions > 1u && (args.map1_frontier_consume || args.map1_frontier_deep || args.map1_frontier_raceway))
					throw std::runtime_error("--map1-frontier-partitions currently supports --map1-frontier-stream-deep only");
				if (args.map1_frontier_raceway && (args.map1_frontier_axis_sweep || args.map1_frontier_mobile_bit_sweep || args.map1_frontier_backfill_sweep))
					throw std::runtime_error("--map1-frontier-raceway supports the normal --map1-frontier path only");
				if (args.map1_frontier_raceway && args.map1_frontier_depth != 1u)
					throw std::runtime_error("--map1-frontier-raceway currently requires --map1-frontier-depth 1");
					if (args.raceway_flag_parity && args.raceway_cap_bits != 0u)
						throw std::runtime_error("--raceway-flag-parity requires --raceway-cap-bits 0");
					if (args.raceway_flat_parity && args.workunit_size > (1ull << 24))
						throw std::runtime_error("--raceway-flat-parity is intended for small windows; use --workunit_size <= 16777216");
					if (args.map1_frontier_multiplicity)
					{
						if (!args.map1_frontier_wide || !args.map1_frontier_stream_deep
							|| args.map1_frontier_depth != 1u || args.map1_frontier_sample_mult != 1u)
							throw std::runtime_error("--map1-frontier-multiplicity currently requires wide MAP1 stream-deep with depth=1 and sample_mult=1");
						if (args.map1_frontier_partitions != 1u)
							throw std::runtime_error("--map1-frontier-multiplicity currently supports --map1-frontier-partitions 1 only");
						if (args.workunit_size > 0xFFFFFFFFull)
							throw std::runtime_error("--map1-frontier-multiplicity requires --workunit_size <= 2^32-1");
					}

				CUfunction zero_k = nullptr, insert_k = nullptr, classify_k = nullptr;
				check_cuda(cuModuleGetFunction(&zero_k, module, "tm_map1_frontier_zero_log_cuda"), "cuModuleGetFunction(map1_frontier_zero_log)");
				const char* insert_name = "tm_map1_frontier_insert_cuda";
				if (args.map1_frontier_depth > 1u || args.map1_frontier_sample_mult != 1u)
					insert_name = "tm_map1_frontier_insertK128_cuda";
				else if (args.map1_frontier_wide)
					insert_name = args.map1_frontier_wide_ilp == 4u ? "tm_map1_frontier_insert128_ilp4_cuda"
					            : args.map1_frontier_wide_ilp == 2u ? "tm_map1_frontier_insert128_ilp2_cuda"
					            : "tm_map1_frontier_insert128_cuda";
				check_cuda(cuModuleGetFunction(&insert_k, module, insert_name), "cuModuleGetFunction(map1_frontier_insert)");
				CUfunction insert_mask_k = nullptr;
				if (args.map1_frontier_backfill_sweep)
					check_cuda(cuModuleGetFunction(&insert_mask_k, module, "tm_map1_frontier_insert_mask128_cuda"), "cuModuleGetFunction(map1_frontier_insert_mask)");
				const uint32_t slot_bytes = args.map1_frontier_wide ? 16u : 8u;  // ulonglong2 vs uint64 table slot
				const uint32_t insert_ilp = (args.map1_frontier_wide && args.map1_frontier_depth == 1u) ? args.map1_frontier_wide_ilp : 1u;
				const uint32_t insert_depth = args.map1_frontier_depth;  // extra trailing iargs entries; only the depthK kernel reads them
				const uint32_t insert_sample_mult = args.map1_frontier_sample_mult;
					if (args.map1_frontier_partitions > 1u)
						check_cuda(cuModuleGetFunction(&classify_k, module, "tm_map1_frontier_classify_cuda"), "cuModuleGetFunction(map1_frontier_classify)");
					if (args.map1_table_auto)
						check_cuda(cuModuleGetFunction(&run_span_hll_kernel, module, "run_span_hll_w8i8_cuda"), "cuModuleGetFunction(run_span_hll_w8i8_cuda)");
					CUfunction ds_canon_k = nullptr, ds_off_k = nullptr;
					if (args.map1_frontier_consume || (args.map1_frontier_raceway && args.raceway_flag_parity))
					{
					if (args.map1_frontier_consume)
						check_cuda(cuModuleGetFunction(&ds_canon_k, module, "tm_wide_merge_survivor_screen_ilp6_cuda"), "cuModuleGetFunction(map1_consume_canon)");
					check_cuda(cuModuleGetFunction(&ds_off_k, module, "tm_wide_merge_survivor_screen_offset_ilp6_cuda"), "cuModuleGetFunction(map1_consume_offset)");
				}
				CUfunction raceway_k = nullptr, raceway_clear_k = nullptr;
				CUfunction raceway_repflag_build_k = nullptr, raceway_flat_parity_k = nullptr;
				if (args.map1_frontier_raceway)
				{
					check_cuda(cuModuleGetFunction(&raceway_k, module, "tm_raceway_stream_offsets_cuda"), "cuModuleGetFunction(tm_raceway_stream_offsets_cuda)");
					check_cuda(cuModuleGetFunction(&raceway_clear_k, module, "tm_raceway_cap_clear_cuda"), "cuModuleGetFunction(tm_raceway_cap_clear_cuda)");
					if (args.raceway_flat_parity)
					{
						check_cuda(cuModuleGetFunction(&raceway_repflag_build_k, module, "tm_raceway_repflag_table_build_cuda"), "cuModuleGetFunction(tm_raceway_repflag_table_build_cuda)");
						check_cuda(cuModuleGetFunction(&raceway_flat_parity_k, module, "tm_raceway_flat_parity_cuda"), "cuModuleGetFunction(tm_raceway_flat_parity_cuda)");
					}
				}

				// Build-from-the-back selected-bit windows: for a K-bit window, select
				// data bits from each byte's MSB downward (e.g. K=8 => 6,7,14,15,22,23,30,31).
				// For each sampled fixed-bit group, dedup the 2^K selected-bit subspace
				// and scale to the full 2^32 data axis.
				if (args.map1_frontier_backfill_sweep)
				{
					const uint64_t axis_total = 1ull << 32;
					if (!args.map1_frontier_wide)
						throw std::runtime_error("--map1-frontier-backfill-sweep requires --map1-frontier-wide");
					if (args.map1_frontier_consume)
						throw std::runtime_error("--map1-frontier-backfill-sweep is count-only");
					if (args.map1_frontier_partitions != 1u || args.map1_frontier_deep || args.map1_frontier_stream_deep)
						throw std::runtime_error("--map1-frontier-backfill-sweep does not support partitions or deep compaction");
					if (args.range_start != 0ull)
						throw std::runtime_error("--map1-frontier-backfill-sweep requires --range_start 0");

					std::vector<uint32_t> bit_counts = args.map1_frontier_backfill_bits.empty()
						? std::vector<uint32_t>{8u, 12u, 16u, 20u, 24u}
						: parse_u32_list(args.map1_frontier_backfill_bits, "--map1-frontier-backfill-bits");
					for (uint32_t kbits : bit_counts)
					{
						if (kbits == 0u || kbits > 31u)
							throw std::runtime_error("--map1-frontier-backfill-bits values must be in 1..31");
					}
					if (args.map1_frontier_mask_policy != "backfill" &&
					    args.map1_frontier_mask_policy != "lowbits" &&
					    args.map1_frontier_mask_policy != "squeeze" &&
					    args.map1_frontier_mask_policy != "backfavor" &&
					    args.map1_frontier_mask_policy != "highbits" &&
					    args.map1_frontier_mask_policy != "auto" &&
					    args.map1_frontier_mask_policy.rfind("omitbyte", 0) != 0)
						throw std::runtime_error("unknown --map1-frontier-mask-policy: " + args.map1_frontier_mask_policy);

					const uint32_t slot_bytes = 16u;
					const uint32_t chunk_max = args.map1_frontier_chunk;
					const uint32_t insert_depth = args.map1_frontier_depth;
					const uint32_t partition = 0u;
					const uint32_t target_window = 1u << 28; // default per-K validation work cap: 256M candidates

					std::cout << "[map1-frontier-backfill-sweep] device=" << device_name
					          << " key=0x" << std::hex << args.key_id << std::dec
					          << " K=" << insert_depth
					          << " policy=" << args.map1_frontier_mask_policy
					          << " bit_counts=" << (args.map1_frontier_backfill_bits.empty() ? "8,12,16,20,24" : args.map1_frontier_backfill_bits)
					          << " chunk=" << chunk_max << "\n";

					for (uint32_t kbits : bit_counts)
					{
						const tm_window_policy::Policy requested_policy =
							tm_window_policy::parse_policy(args.map1_frontier_mask_policy);
						const tm_window_policy::Policy resolved_policy =
							tm_window_policy::resolve_policy(requested_policy, kbits, insert_depth);
						const uint32_t selected_mask = make_frontier_policy_bit_mask(args.map1_frontier_mask_policy, kbits, insert_depth);
						const char* mask_policy = tm_window_policy::policy_name(resolved_policy);
						const uint32_t fixed_mask = ~selected_mask;
						const uint64_t selected_count = 1ull << kbits;
						const uint64_t full_groups = 1ull << (32u - kbits);
						uint64_t sample_groups = full_groups;
						if (args.map1_frontier_mobile_groups != 0u)
							sample_groups = std::min<uint64_t>(full_groups, args.map1_frontier_mobile_groups);
						else if (full_groups > 4096ull)
							sample_groups = std::max<uint64_t>(1ull, std::min<uint64_t>(4096ull, target_window / selected_count));

						const uint64_t desired_slots = selected_count * 2ull;
						uint32_t desired_logm = 1u;
						while (desired_logm < 32u && (1ull << desired_logm) < desired_slots) ++desired_logm;
						if ((1ull << desired_logm) < desired_slots)
							throw std::runtime_error("--map1-frontier-backfill-sweep table exceeds 2^32 slots");

						size_t free_b = 0, total_b = 0;
						check_cuda(cuMemGetInfo(&free_b, &total_b), "cuMemGetInfo(map1_backfill_sweep)");
						uint64_t cap_bytes = args.map1_frontier_table_mb != 0u
							? (static_cast<uint64_t>(args.map1_frontier_table_mb) << 20)
							: static_cast<uint64_t>(free_b * 0.80);
						const uint64_t cap_slots = cap_bytes / slot_bytes;
						if (cap_slots < desired_slots)
							throw std::runtime_error("--map1-frontier-backfill-sweep table budget cannot hold the requested 2x window table");

						const uint32_t logm = desired_logm;
						const uint64_t slots = 1ull << logm;
						const size_t need = static_cast<size_t>(slots) * slot_bytes;
						const size_t margin = 512ull << 20;
						if (free_b < need + margin)
						{
							std::ostringstream e;
							e << "INSUFFICIENT_VRAM for backfill table: need " << (need >> 20) << " MiB + "
							  << (margin >> 20) << " MiB margin, only " << (free_b >> 20) << " MiB free of "
							  << (total_b >> 20) << " MiB on device " << args.device_index;
							throw std::runtime_error(e.str());
						}

						CUdeviceptr table_fp = 0, d_unique = 0, d_overflow = 0, d_rep = 0;
						check_cuda(cuMemAlloc(&table_fp, static_cast<size_t>(slots) * slot_bytes), "cuMemAlloc(map1_backfill_table)");
						check_cuda(cuMemAlloc(&d_unique, sizeof(uint32_t)), "cuMemAlloc(map1_backfill_unique)");
						check_cuda(cuMemAlloc(&d_overflow, sizeof(uint32_t)), "cuMemAlloc(map1_backfill_overflow)");

						uint64_t sum_f = 0ull, sum_overflow = 0ull, max_f = 0ull;
						double zero_ms_total = 0.0, insert_ms_total = 0.0;
						for (uint64_t sample_group = 0ull; sample_group < sample_groups; ++sample_group)
						{
							const uint64_t group = (sample_groups == full_groups)
								? sample_group
								: ((sample_group * 0x9E3779B97F4A7C15ull) & (full_groups - 1ull));
							const uint32_t fixed_value = tm_window_policy::deposit_bits32(static_cast<uint32_t>(group), fixed_mask);

							const auto z0 = std::chrono::high_resolution_clock::now();
							check_cuda(cuMemsetD32(d_unique, 0u, 1u), "memset(map1_backfill_unique)");
							check_cuda(cuMemsetD32(d_overflow, 0u, 1u), "memset(map1_backfill_overflow)");
							check_cuda(cuMemsetD8(table_fp, 0u, static_cast<size_t>(slots) * slot_bytes), "memset(map1_backfill_table)");
							check_cuda(cuCtxSynchronize(), "sync(map1_backfill_zero)");
							const auto z1 = std::chrono::high_resolution_clock::now();
							zero_ms_total += std::chrono::duration<double, std::milli>(z1 - z0).count();

							const auto i0 = std::chrono::high_resolution_clock::now();
							uint64_t done = 0ull;
							while (done < selected_count)
							{
								const uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(chunk_max, selected_count - done));
								const uint32_t candidate_start = static_cast<uint32_t>(done);
								const uint32_t rep_cap = 0u;
								CUdeviceptr unique_out = 0, label_in = 0;
								void* iargs[] = {
									&table_fp, (void*)&logm, &d_unique, &d_overflow, &d_rep, (void*)&rep_cap,
									&unique_out, &label_in, (void*)&partition,
									&assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
									&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
									&assets.alg2_values, &assets.alg5_values, &assets.expansion_values,
									&assets.schedule_data, &args.key_id, (void*)&fixed_value, (void*)&candidate_start,
									(void*)&chunk, (void*)&insert_depth, (void*)&selected_mask };
								check_cuda(cuLaunchKernel(insert_mask_k, screen_kernel_grid_x_ilp(chunk, 1u), 1, 1,
									kCudaThreadsPerBlock, 1, 1, 0, 0, iargs, nullptr), "launch(map1_backfill_insert)");
								done += chunk;
							}
							check_cuda(cuCtxSynchronize(), "sync(map1_backfill_insert)");
							const auto i1 = std::chrono::high_resolution_clock::now();
							insert_ms_total += std::chrono::duration<double, std::milli>(i1 - i0).count();

							uint32_t unique = 0u, overflow = 0u;
							check_cuda(cuMemcpyDtoH(&unique, d_unique, sizeof(uint32_t)), "DtoH(map1_backfill_unique)");
							check_cuda(cuMemcpyDtoH(&overflow, d_overflow, sizeof(uint32_t)), "DtoH(map1_backfill_overflow)");
							sum_f += unique;
							sum_overflow += overflow;
							max_f = std::max<uint64_t>(max_f, unique);

							if (sample_groups <= 16ull)
							{
								std::cout << "  backfill_group policy=" << mask_policy
								          << " K=" << insert_depth
								          << " bits=" << kbits
								          << " selected_mask=0x" << std::hex << selected_mask << std::dec
								          << " group=" << group
								          << " sample_group=" << sample_group
								          << " F=" << unique
								          << " R=" << (static_cast<double>(selected_count) / static_cast<double>(unique ? unique : 1u))
								          << " overflow=" << overflow << "\n";
							}
							if (overflow != 0u)
								throw std::runtime_error("--map1-frontier-backfill-sweep overflowed its 2x table");
						}
						const double total_ms = zero_ms_total + insert_ms_total;
						const double scale = static_cast<double>(full_groups) / static_cast<double>(sample_groups ? sample_groups : 1ull);
						const uint64_t estimated_sum_f = static_cast<uint64_t>(std::llround(static_cast<double>(sum_f) * scale));
						std::cout << "  backfill_summary policy=" << mask_policy
						          << " K=" << insert_depth
						          << " bits=" << kbits
						          << " window=" << selected_count
						          << " selected_mask=0x" << std::hex << selected_mask << std::dec
						          << " selected_bits=" << tm_window_policy::bit_mask_string(selected_mask)
						          << " sampled_groups=" << sample_groups
						          << " full_groups=" << full_groups
						          << " sample_sum_F=" << sum_f
						          << " estimated_sum_F=" << estimated_sum_f
						          << " aggregate_R=" << (static_cast<double>(axis_total) / static_cast<double>(estimated_sum_f ? estimated_sum_f : 1ull))
						          << " max_F=" << max_f
						          << " overflow=" << sum_overflow
						          << " zero_ms=" << std::fixed << std::setprecision(3) << zero_ms_total
						          << " insert_ms=" << insert_ms_total
						          << " total_ms=" << total_ms
						          << " effective_Mcps=" << std::setprecision(1) << (static_cast<double>(sample_groups * selected_count) / 1e6 / (total_ms / 1e3))
						          << "\n";

						cuMemFree(table_fp); cuMemFree(d_unique); cuMemFree(d_overflow);
					}
					return 0;
				}

				// Test whether extra coordinated bits are position-local. Each group
				// dedups 2^E lower-bit slices in one table, with the selected high bits
				// varied by the mobile-bit policy.
				if (args.map1_frontier_mobile_bit_sweep)
				{
					const uint64_t axis_total = 1ull << 32;
					const uint64_t base_span = args.workunit_size;
					if (!args.map1_frontier_wide)
						throw std::runtime_error("--map1-frontier-mobile-bit-sweep requires --map1-frontier-wide");
					if (args.map1_frontier_consume)
						throw std::runtime_error("--map1-frontier-mobile-bit-sweep is count-only");
					if (args.map1_frontier_partitions != 1u || args.map1_frontier_deep || args.map1_frontier_stream_deep)
						throw std::runtime_error("--map1-frontier-mobile-bit-sweep does not support partitions or deep compaction");
					if (args.range_start != 0ull)
						throw std::runtime_error("--map1-frontier-mobile-bit-sweep requires --range_start 0");
					if (base_span == 0ull || (base_span & (base_span - 1ull)) != 0ull || base_span > (1ull << 31))
						throw std::runtime_error("--map1-frontier-mobile-bit-sweep requires power-of-two --workunit_size <= 2^31");

					uint32_t base_bits = 0u;
					while ((1ull << base_bits) < base_span) ++base_bits;
					const uint32_t extra_bits = args.map1_frontier_mobile_extra_bits;
					const uint32_t byte_hi = ((base_bits / 8u) * 8u) + 7u;
					if (base_bits > 31u || byte_hi > 31u || extra_bits > (byte_hi - base_bits + 1u))
						throw std::runtime_error("--map1-frontier-mobile-bit-sweep extra bits must fit in the byte containing the first excluded bit");
					const uint64_t legs = 1ull << extra_bits;
					const uint64_t effective_window = base_span * legs;

					const uint32_t slot_bytes = 16u;
					const uint64_t desired_slots = effective_window * 2ull;
					uint32_t desired_logm = 1u;
					while (desired_logm < 32u && (1ull << desired_logm) < desired_slots) ++desired_logm;
					if ((1ull << desired_logm) < desired_slots)
						throw std::runtime_error("--map1-frontier-mobile-bit-sweep table exceeds 2^32 slots");

					size_t free_b = 0, total_b = 0;
					check_cuda(cuMemGetInfo(&free_b, &total_b), "cuMemGetInfo(map1_mobile_bit_sweep)");
					uint64_t cap_bytes = 0ull;
					if (args.map1_frontier_table_mb != 0u)
						cap_bytes = static_cast<uint64_t>(args.map1_frontier_table_mb) << 20;
					else
						cap_bytes = static_cast<uint64_t>(free_b * 0.80);
					const uint64_t cap_slots = cap_bytes / slot_bytes;
					if (cap_slots < desired_slots)
						throw std::runtime_error("--map1-frontier-mobile-bit-sweep table budget cannot hold the requested 2x effective window table");

					const uint32_t logm = desired_logm;
					const uint64_t slots = 1ull << logm;
					const uint32_t chunk_max = args.map1_frontier_chunk;
					const uint32_t insert_ilp = (args.map1_frontier_depth == 1u) ? args.map1_frontier_wide_ilp : 1u;
					const uint32_t insert_depth = args.map1_frontier_depth;
					const uint32_t partition = 0u;

					CUdeviceptr table_fp = 0, d_unique = 0, d_overflow = 0, d_rep = 0;
					const size_t need = static_cast<size_t>(slots) * slot_bytes;
					const size_t margin = 512ull << 20;
					if (free_b < need + margin)
					{
						std::ostringstream e;
						e << "INSUFFICIENT_VRAM for mobile-bit table: need " << (need >> 20) << " MiB + "
						  << (margin >> 20) << " MiB margin, only " << (free_b >> 20) << " MiB free of "
						  << (total_b >> 20) << " MiB on device " << args.device_index;
						throw std::runtime_error(e.str());
					}
					check_cuda(cuMemAlloc(&table_fp, static_cast<size_t>(slots) * slot_bytes), "cuMemAlloc(map1_mobile_table)");
					check_cuda(cuMemAlloc(&d_unique, sizeof(uint32_t)), "cuMemAlloc(map1_mobile_unique)");
					check_cuda(cuMemAlloc(&d_overflow, sizeof(uint32_t)), "cuMemAlloc(map1_mobile_overflow)");

					std::cout << "[map1-frontier-mobile-bit-sweep] device=" << device_name
					          << " key=0x" << std::hex << args.key_id << std::dec
					          << " K=" << insert_depth
					          << " base_window=" << base_span
					          << " effective_window=" << effective_window
					          << " extra_bits=" << extra_bits
					          << " candidate_bits=" << base_bits << ".." << byte_hi
					          << " table_slots=" << slots
					          << " chunk=" << chunk_max << "\n";

					for (uint32_t mobile_bit = base_bits; mobile_bit <= byte_hi; ++mobile_bit)
					{
						std::vector<uint32_t> selected_bits;
						selected_bits.push_back(mobile_bit);
						if (extra_bits > 1u)
						{
							if (args.map1_frontier_mobile_backfill)
							{
								for (uint32_t b = byte_hi + 1u; b-- > base_bits && selected_bits.size() < extra_bits;)
								{
									if (b == mobile_bit) continue;
									selected_bits.push_back(b);
								}
							}
							else
							{
								for (uint32_t b = mobile_bit + 1u; b <= byte_hi && selected_bits.size() < extra_bits; ++b)
									selected_bits.push_back(b);
								if (selected_bits.size() < extra_bits) continue;
							}
						}
						if (selected_bits.size() != extra_bits)
							throw std::runtime_error("--map1-frontier-mobile-bit-sweep could not construct selected bit set");
						uint32_t selected_mask = 0u;
						for (uint32_t b : selected_bits) selected_mask |= 1u << b;

						std::vector<uint32_t> fixed_bits;
						for (uint32_t b = base_bits; b < 32u; ++b)
						{
							if ((selected_mask & (1u << b)) == 0u) fixed_bits.push_back(b);
						}
						const uint64_t full_groups = 1ull << fixed_bits.size();
						uint64_t sample_groups = full_groups;
						if (args.map1_frontier_mobile_groups != 0u)
							sample_groups = std::min<uint64_t>(full_groups, args.map1_frontier_mobile_groups);
						else if (full_groups > 65536ull)
							sample_groups = 4096ull;

						uint64_t sum_f = 0ull, sum_overflow = 0ull, max_f = 0ull;
						double zero_ms_total = 0.0, insert_ms_total = 0.0;
						for (uint64_t sample_group = 0ull; sample_group < sample_groups; ++sample_group)
						{
							const uint64_t group = (sample_groups == full_groups)
								? sample_group
								: ((sample_group * 0x9E3779B97F4A7C15ull) & (full_groups - 1ull));
							uint32_t fixed_value = 0u;
							uint64_t bits = group;
							for (uint32_t b : fixed_bits)
							{
								if ((bits & 1ull) != 0ull) fixed_value |= 1u << b;
								bits >>= 1u;
							}

							const auto z0 = std::chrono::high_resolution_clock::now();
							check_cuda(cuMemsetD32(d_unique, 0u, 1u), "memset(map1_mobile_unique)");
							check_cuda(cuMemsetD32(d_overflow, 0u, 1u), "memset(map1_mobile_overflow)");
							check_cuda(cuMemsetD8(table_fp, 0u, static_cast<size_t>(slots) * slot_bytes), "memset(map1_mobile_table)");
							check_cuda(cuCtxSynchronize(), "sync(map1_mobile_zero)");
							const auto z1 = std::chrono::high_resolution_clock::now();
							zero_ms_total += std::chrono::duration<double, std::milli>(z1 - z0).count();

							const auto i0 = std::chrono::high_resolution_clock::now();
							for (uint64_t leg = 0ull; leg < legs; ++leg)
							{
								uint32_t selected_value = 0u;
								for (uint32_t j = 0u; j < selected_bits.size(); ++j)
									if ((leg & (1ull << j)) != 0ull) selected_value |= 1u << selected_bits[j];
								const uint32_t leg_start = fixed_value | selected_value;
								uint64_t done = 0ull;
								while (done < base_span)
								{
									const uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(chunk_max, base_span - done));
									const uint32_t data_start = leg_start + static_cast<uint32_t>(done);
										const uint32_t rep_cap = 0u;
										CUdeviceptr unique_out = 0, label_in = 0;
										const uint32_t insert_route_tau_m = 0u;
										CUdeviceptr null_u32 = 0;
										const uint32_t rep_mult_cap = 0u;
										void* iargs[] = {
											&table_fp, (void*)&logm, &d_unique, &d_overflow, &d_rep, (void*)&rep_cap,
											&unique_out, &label_in, (void*)&partition,
											&assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
											&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
											&assets.alg2_values, &assets.alg5_values, &assets.expansion_values,
											&assets.schedule_data, &args.key_id, (void*)&data_start, (void*)&chunk,
											(void*)&insert_route_tau_m, (void*)&insert_depth, (void*)&insert_sample_mult,
											&null_u32, &null_u32, (void*)&rep_mult_cap, &null_u32 };
									check_cuda(cuLaunchKernel(insert_k, screen_kernel_grid_x_ilp(chunk, insert_ilp), 1, 1,
										kCudaThreadsPerBlock, 1, 1, 0, 0, iargs, nullptr), "launch(map1_mobile_insert)");
									done += chunk;
								}
							}
							check_cuda(cuCtxSynchronize(), "sync(map1_mobile_insert)");
							const auto i1 = std::chrono::high_resolution_clock::now();
							insert_ms_total += std::chrono::duration<double, std::milli>(i1 - i0).count();

							uint32_t unique = 0u, overflow = 0u;
							check_cuda(cuMemcpyDtoH(&unique, d_unique, sizeof(uint32_t)), "DtoH(map1_mobile_unique)");
							check_cuda(cuMemcpyDtoH(&overflow, d_overflow, sizeof(uint32_t)), "DtoH(map1_mobile_overflow)");
							sum_f += unique;
							sum_overflow += overflow;
							max_f = std::max<uint64_t>(max_f, unique);

							std::cout << "  mobile_bit_group K=" << insert_depth
							          << " mobile_bit=" << mobile_bit
							          << " selected_mask=" << selected_mask
							          << " group=" << group
							          << " sample_group=" << sample_group
							          << " F=" << unique
							          << " R=" << (static_cast<double>(effective_window) / static_cast<double>(unique ? unique : 1u))
							          << " overflow=" << overflow << "\n";
							if (overflow != 0u)
								throw std::runtime_error("--map1-frontier-mobile-bit-sweep overflowed its 2x table");
						}
						const double total_ms = zero_ms_total + insert_ms_total;
						const double scale = static_cast<double>(full_groups) / static_cast<double>(sample_groups ? sample_groups : 1ull);
						const uint64_t estimated_sum_f = static_cast<uint64_t>(std::llround(static_cast<double>(sum_f) * scale));
						std::cout << "  mobile_bit_summary K=" << insert_depth
						          << " base_window=" << base_span
						          << " effective_window=" << effective_window
						          << " extra_bits=" << extra_bits
						          << " mobile_bit=" << mobile_bit
						          << " selected_mask=" << selected_mask
						          << " sampled_groups=" << sample_groups
						          << " full_groups=" << full_groups
						          << " sample_sum_F=" << sum_f
						          << " estimated_sum_F=" << estimated_sum_f
						          << " aggregate_R=" << (static_cast<double>(axis_total) / static_cast<double>(estimated_sum_f ? estimated_sum_f : 1ull))
						          << " max_F=" << max_f
						          << " overflow=" << sum_overflow
						          << " zero_ms=" << std::fixed << std::setprecision(3) << zero_ms_total
						          << " insert_ms=" << insert_ms_total
						          << " total_ms=" << total_ms
						          << " effective_Mcps=" << std::setprecision(1) << (static_cast<double>(axis_total) / 1e6 / (total_ms / 1e3))
						          << "\n";
					}

					cuMemFree(table_fp); cuMemFree(d_unique); cuMemFree(d_overflow);
					return 0;
				}

				// Measure the cost of settling for independent lower-magnitude MAP-K
				// windows while still covering the full data axis. This keeps one CUDA
				// context and allocation alive, but clears the identity table between
				// windows so the summed frontier exposes the lost cross-window collapse.
				if (args.map1_frontier_axis_sweep)
				{
					const uint64_t axis_total = 1ull << 32;
					const uint64_t tile = args.workunit_size;
					if (!args.map1_frontier_wide)
						throw std::runtime_error("--map1-frontier-axis-sweep requires --map1-frontier-wide");
					if (args.map1_frontier_consume && args.map1_frontier_depth != 1u)
						throw std::runtime_error("--map1-frontier-axis-sweep consumer currently supports MAP1 only");
					if (args.map1_frontier_partitions != 1u || args.map1_frontier_deep || args.map1_frontier_stream_deep)
						throw std::runtime_error("--map1-frontier-axis-sweep does not support partitions or deep compaction");
					if (args.range_start != 0ull)
						throw std::runtime_error("--map1-frontier-axis-sweep requires --range_start 0");
					if (tile == 0ull || tile > axis_total || (axis_total % tile) != 0ull)
						throw std::runtime_error("--map1-frontier-axis-sweep needs --workunit_size dividing 2^32");
					if (args.map1_frontier_consume && tile > 0xffffffffull)
						throw std::runtime_error("--map1-frontier-axis-sweep consumer needs --workunit_size <= 2^32-1");

					const uint32_t slot_bytes = 16u;
					const uint64_t desired_slots = tile * 2ull;
					uint32_t desired_logm = 1u;
					while (desired_logm < 32u && (1ull << desired_logm) < desired_slots) ++desired_logm;
					if ((1ull << desired_logm) < desired_slots)
						throw std::runtime_error("--map1-frontier-axis-sweep table exceeds 2^32 slots");

					size_t free_b = 0, total_b = 0;
					check_cuda(cuMemGetInfo(&free_b, &total_b), "cuMemGetInfo(map1_axis_sweep)");
					uint64_t cap_bytes = 0ull;
					if (args.map1_frontier_table_mb != 0u)
						cap_bytes = static_cast<uint64_t>(args.map1_frontier_table_mb) << 20;
					else
						cap_bytes = static_cast<uint64_t>(free_b * 0.80);
					const uint64_t cap_slots = cap_bytes / slot_bytes;
					if (cap_slots < desired_slots)
						throw std::runtime_error("--map1-frontier-axis-sweep table budget cannot hold a 2x window table");

					const uint32_t logm = desired_logm;
					const uint64_t slots = 1ull << logm;
					const uint32_t chunk_max = args.map1_frontier_chunk;
					const uint32_t insert_ilp = (args.map1_frontier_depth == 1u) ? args.map1_frontier_wide_ilp : 1u;
					const uint32_t insert_depth = args.map1_frontier_depth;
					const uint32_t partition = 0u;
					const uint64_t ntiles = axis_total / tile;

					CUdeviceptr table_fp = 0, d_unique = 0, d_overflow = 0, d_rep = 0, d_flag = 0, d_ostream = 0;
					{	// VRAM guard (see map1_frontier path): fail loudly, not as empty output, under contention.
						size_t guard_free = 0, guard_total = 0;
						check_cuda(cuMemGetInfo(&guard_free, &guard_total), "cuMemGetInfo(map1_axis_guard)");
						const size_t need = static_cast<size_t>(slots) * slot_bytes;
						const size_t margin = 512ull << 20;
						if (guard_free < need + margin)
						{
							std::ostringstream e;
							e << "INSUFFICIENT_VRAM for MAP1 axis table: need " << (need >> 20) << " MiB + "
							  << (margin >> 20) << " MiB margin, only " << (guard_free >> 20) << " MiB free of "
							  << (guard_total >> 20) << " MiB on device " << args.device_index
							  << " (another process may be using this GPU). Free the GPU or lower --map1-frontier-table-mb.";
							throw std::runtime_error(e.str());
						}
					}
					check_cuda(cuMemAlloc(&table_fp, static_cast<size_t>(slots) * slot_bytes), "cuMemAlloc(map1_axis_table)");
					check_cuda(cuMemAlloc(&d_unique, sizeof(uint32_t)), "cuMemAlloc(map1_axis_unique)");
					check_cuda(cuMemAlloc(&d_overflow, sizeof(uint32_t)), "cuMemAlloc(map1_axis_overflow)");

					uint32_t rep_cap = 0u;
					CUdeviceptr off_regular = 0, off_alg0 = 0, off_alg6 = 0, off_alg2 = 0, off_alg5 = 0;
					if (args.map1_frontier_consume)
					{
						rep_cap = static_cast<uint32_t>(tile);
						check_cuda(cuMemAlloc(&d_rep, static_cast<size_t>(rep_cap) * sizeof(uint32_t)), "cuMemAlloc(map1_axis_rep)");
						check_cuda(cuMemAlloc(&d_flag, static_cast<size_t>(rep_cap)), "cuMemAlloc(map1_axis_flag)");
						const std::vector<uint8_t> sched_blob = build_schedule_blob(args.key_id, args.map_list);
						OffsetStreamBlob osb = build_offset_stream_blob(sched_blob);
						check_cuda(cuMemAlloc(&d_ostream, osb.data.size()), "cuMemAlloc(map1_axis_ostream)");
						check_cuda(cuMemcpyHtoD(d_ostream, osb.data.data(), osb.data.size()), "HtoD(map1_axis_ostream)");
						off_regular = d_ostream;
						off_alg0 = off_regular + osb.stream_bytes;
						off_alg6 = off_alg0 + osb.stream_bytes;
						off_alg2 = off_alg6 + osb.stream_bytes;
						off_alg5 = off_alg2 + osb.carry_bytes;
					}

					uint64_t sum_f1 = 0ull, sum_overflow = 0ull, max_f1 = 0ull;
					double zero_ms_total = 0.0, insert_ms_total = 0.0, consumer_ms_total = 0.0;
						std::cout << "[map1-frontier-axis-sweep] device=" << device_name
						          << " key=0x" << std::hex << args.key_id << std::dec
						          << " K=" << insert_depth
						          << " window=" << tile << " tiles=" << ntiles
						          << " table_slots=" << slots << " chunk=" << chunk_max
						          << " consumer=" << (args.map1_frontier_consume ? "offset" : "none") << "\n";

					for (uint64_t t = 0ull; t < ntiles; ++t)
					{
						const uint64_t tile_start64 = t * tile;
						const auto z0 = std::chrono::high_resolution_clock::now();
						check_cuda(cuMemsetD32(d_unique, 0u, 1u), "memset(map1_axis_unique)");
						check_cuda(cuMemsetD32(d_overflow, 0u, 1u), "memset(map1_axis_overflow)");
						check_cuda(cuMemsetD8(table_fp, 0u, static_cast<size_t>(slots) * slot_bytes), "memset(map1_axis_table)");
						check_cuda(cuCtxSynchronize(), "sync(map1_axis_zero)");
						const auto z1 = std::chrono::high_resolution_clock::now();
						const double zero_ms = std::chrono::duration<double, std::milli>(z1 - z0).count();
						zero_ms_total += zero_ms;

						const auto i0 = std::chrono::high_resolution_clock::now();
						uint64_t done = 0ull;
						while (done < tile)
						{
							const uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(chunk_max, tile - done));
							const uint32_t data_start = static_cast<uint32_t>(tile_start64 + done);
								CUdeviceptr unique_out = 0, label_in = 0;
								const uint32_t insert_route_tau_a = 0u;
								CUdeviceptr null_u32 = 0;
								const uint32_t rep_mult_cap = 0u;
									void* iargs[] = {
										&table_fp, (void*)&logm, &d_unique, &d_overflow, &d_rep, (void*)&rep_cap,
										&unique_out, &label_in, (void*)&partition,
										&assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
										&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
										&assets.alg2_values, &assets.alg5_values, &assets.expansion_values,
										&assets.schedule_data, &args.key_id, (void*)&data_start, (void*)&chunk,
										(void*)&insert_route_tau_a, (void*)&insert_depth, (void*)&insert_sample_mult,
										&null_u32, &null_u32, (void*)&rep_mult_cap, &null_u32 };
								check_cuda(cuLaunchKernel(insert_k, screen_kernel_grid_x_ilp(chunk, 1u), 1, 1,
									kCudaThreadsPerBlock, 1, 1, 0, 0, iargs, nullptr), "launch(map1_axis_insert)");
							done += chunk;
						}
						check_cuda(cuCtxSynchronize(), "sync(map1_axis_insert)");
						const auto i1 = std::chrono::high_resolution_clock::now();
						const double insert_ms = std::chrono::duration<double, std::milli>(i1 - i0).count();
						insert_ms_total += insert_ms;

						uint32_t unique = 0u, overflow = 0u;
						check_cuda(cuMemcpyDtoH(&unique, d_unique, sizeof(uint32_t)), "DtoH(map1_axis_unique)");
						check_cuda(cuMemcpyDtoH(&overflow, d_overflow, sizeof(uint32_t)), "DtoH(map1_axis_overflow)");
						sum_f1 += unique;
						sum_overflow += overflow;
						max_f1 = std::max<uint64_t>(max_f1, unique);

						double consumer_ms = 0.0;
						if (args.map1_frontier_consume)
						{
							uint32_t nrep = unique;
							uint32_t data0 = 0u;
							void* oa[] = { &d_rep, &nrep, &off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
								&assets.expansion_values, &assets.schedule_data, &assets.carnival_data, &args.key_id, &data0, &d_flag };
							const auto o0 = std::chrono::high_resolution_clock::now();
							check_cuda(cuLaunchKernel(ds_off_k, screen_kernel_grid_x_ilp(nrep, 6u), 1, 1,
								kCudaThreadsPerBlock, 1, 1, 0, 0, oa, nullptr), "launch(map1_axis_consumer)");
							check_cuda(cuCtxSynchronize(), "sync(map1_axis_consumer)");
							const auto o1 = std::chrono::high_resolution_clock::now();
							consumer_ms = std::chrono::duration<double, std::milli>(o1 - o0).count();
							consumer_ms_total += consumer_ms;
						}

							std::cout << "  axis_window=" << t
							          << " data_start=" << tile_start64
							          << " K=" << insert_depth
							          << " F=" << unique
							          << " R=" << (static_cast<double>(tile) / static_cast<double>(unique ? unique : 1u))
							          << " overflow=" << overflow
							          << " zero_ms=" << std::fixed << std::setprecision(3) << zero_ms
						          << " insert_ms=" << insert_ms;
						if (args.map1_frontier_consume) std::cout << " consumer_ms=" << consumer_ms;
						std::cout << "\n";
						if (overflow != 0u)
							throw std::runtime_error("--map1-frontier-axis-sweep overflowed its 2x table");
					}

						const double total_ms = zero_ms_total + insert_ms_total + consumer_ms_total;
						std::cout << "  axis_summary K=" << insert_depth
						          << " window=" << tile
						          << " tiles=" << ntiles
						          << " sum_F=" << sum_f1
						          << " aggregate_R=" << (static_cast<double>(axis_total) / static_cast<double>(sum_f1 ? sum_f1 : 1ull))
						          << " max_F=" << max_f1
					          << " overflow=" << sum_overflow
					          << " zero_ms=" << std::fixed << std::setprecision(3) << zero_ms_total
					          << " insert_ms=" << insert_ms_total;
					if (args.map1_frontier_consume) std::cout << " consumer_ms=" << consumer_ms_total;
					std::cout << " total_ms=" << total_ms
					          << " effective_Mcps=" << std::setprecision(1) << (static_cast<double>(axis_total) / 1e6 / (total_ms / 1e3))
					          << "\n";

					cuMemFree(table_fp); cuMemFree(d_unique); cuMemFree(d_overflow);
					if (d_rep) cuMemFree(d_rep);
					if (d_flag) cuMemFree(d_flag);
					if (d_ostream) cuMemFree(d_ostream);
					return 0;
				}

					CUfunction frontier_span_k = nullptr, frontier_compact_k = nullptr, frontier_compact_data_k = nullptr, frontier_compact_data_mult_k = nullptr;
				if (args.map1_frontier_deep || args.map1_frontier_stream_deep)
				{
					check_cuda(cuModuleGetFunction(&frontier_span_k, module, "run_frontier_span_local_w8i10_cuda"), "cuModuleGetFunction(frontier_span_local)");
					check_cuda(cuModuleGetFunction(&frontier_compact_k, module, "compact_survivors_ordered_cuda"), "cuModuleGetFunction(frontier_compact)");
				}
					if (args.map1_frontier_stream_deep)
						check_cuda(cuModuleGetFunction(&frontier_compact_data_k, module, "tm_map1_frontier_compact_data_cuda"), "cuModuleGetFunction(frontier_compact_data)");
					if (args.map1_frontier_multiplicity)
						check_cuda(cuModuleGetFunction(&frontier_compact_data_mult_k, module, "tm_map1_frontier_compact_data_mult_cuda"), "cuModuleGetFunction(frontier_compact_data_mult)");
				// Pool B: persistent cross-chunk deep cap (producer path). Cap kernel = the deep
				// span kernel with a cap-probe at mode==1; cap table allocated + zeroed ONCE below.
				CUfunction frontier_span_cap_k = nullptr;
				const bool use_frontier_cap = args.map1_frontier_stream_deep && args.map1_frontier_drain_cap_bits > 0u;
				if (use_frontier_cap && args.map1_frontier_deep)
					throw std::runtime_error("--map1-frontier-drain-cap-bits is for the stream-deep path; do not combine with --map1-frontier-deep");
				if (use_frontier_cap)
					check_cuda(cuModuleGetFunction(&frontier_span_cap_k, module, "run_frontier_span_local_cap_w8i10_cuda"), "cuModuleGetFunction(frontier_span_local_cap)");
				CUfunction frontier_span_shed_k = nullptr;
				const bool use_frontier_shed = use_frontier_cap && args.map1_frontier_drain_shed_tau > 0u;
				if (use_frontier_shed)
					check_cuda(cuModuleGetFunction(&frontier_span_shed_k, module, "run_frontier_span_local_shed_w8i10_cuda"), "cuModuleGetFunction(frontier_span_local_shed)");
				CUfunction frontier_span_shed_gate_k = nullptr;
				const bool use_frontier_shed_gate = use_frontier_cap && args.map1_frontier_drain_shed_gate_tau > 0u;
				if (use_frontier_shed_gate)
					check_cuda(cuModuleGetFunction(&frontier_span_shed_gate_k, module, "run_frontier_span_local_shed_gate_w8i10_cuda"), "cuModuleGetFunction(frontier_span_local_shed_gate)");
				CUfunction frontier_span_fp_gate_k = nullptr;
				const bool use_frontier_fp_gate = use_frontier_cap && args.map1_frontier_drain_fp_gate_log > 0u;
				if (args.map1_frontier_drain_fp_gate_log > 30u)
					throw std::runtime_error("--map1-frontier-drain-fp-gate-log must be <= 30");
				if (use_frontier_fp_gate)
					check_cuda(cuModuleGetFunction(&frontier_span_fp_gate_k, module, "run_frontier_span_local_fp_gate_w8i10_cuda"), "cuModuleGetFunction(frontier_span_local_fp_gate)");
				CUfunction frontier_span_traj_k = nullptr;
				const bool use_frontier_sticky_only = use_frontier_cap
					&& args.map1_frontier_drain_traj_bits == 0u
					&& args.map1_frontier_drain_alg0_tau_explicit
					&& args.map1_frontier_drain_alg0_tau > 0u;
				const bool use_frontier_mult_gate = use_frontier_cap && args.map1_frontier_drain_mult_tau > 0u;
				const bool use_frontier_traj = use_frontier_cap
					&& (args.map1_frontier_drain_traj_bits > 0u || use_frontier_sticky_only || use_frontier_mult_gate);
				if (args.map1_frontier_drain_traj_bits > 28u)
					throw std::runtime_error("--map1-frontier-drain-traj-bits must be <= 28");
				if ((use_frontier_shed ? 1u : 0u) + (use_frontier_shed_gate ? 1u : 0u)
						+ (use_frontier_fp_gate ? 1u : 0u) + (use_frontier_traj ? 1u : 0u) > 1u)
					throw std::runtime_error("producer Pool B gating modes are mutually exclusive");
				if ((args.map1_frontier_drain_traj_bits > 0u || args.map1_frontier_drain_alg0_tau_explicit
						|| args.map1_frontier_drain_mult_tau > 0u
						|| args.map1_frontier_drain_shed_gate_tau > 0u || args.map1_frontier_drain_fp_gate_log > 0u) && !use_frontier_cap)
					throw std::runtime_error("producer Pool B gate flags require --map1-frontier-drain-cap-bits");
				if (args.map1_frontier_drain_mult_tau > 0u && !args.map1_frontier_multiplicity)
					throw std::runtime_error("--map1-frontier-drain-mult-tau requires --map1-frontier-multiplicity");
				if (use_frontier_traj)
					check_cuda(cuModuleGetFunction(&frontier_span_traj_k, module, "run_frontier_span_local_traj_w8i10_cuda"), "cuModuleGetFunction(frontier_span_local_traj)");

				const uint64_t total = args.workunit_size;
				const uint32_t chunk_max = args.map1_frontier_chunk;
				size_t free_b = 0, total_b = 0;
				check_cuda(cuMemGetInfo(&free_b, &total_b), "cuMemGetInfo(map1_frontier)");

					const uint32_t partitions = args.map1_frontier_partitions;
					uint32_t partition_log = 0u;
					while ((1u << partition_log) < partitions) ++partition_log;
					uint64_t target_slots_total = total * 2ull; // default: 2x window
					if (args.map1_table_auto)
					{
						if (args.map1_frontier_sample_mult != 1u)
							throw std::runtime_error("--map1-frontier-table-auto requires --map1-frontier-sample-mult 1");
						CUdeviceptr hll_regs = 0, hll_ostream = 0;
						check_cuda(cuMemAlloc(&hll_regs, 4096u * sizeof(uint32_t)), "cuMemAlloc(map1_frontier_hll_regs)");
						check_cuda(cuMemsetD8(hll_regs, 0, 4096u * sizeof(uint32_t)), "memset(map1_frontier_hll_regs)");
						const std::vector<uint8_t> sched_blob = build_schedule_blob(args.key_id, args.map_list);
						OffsetStreamBlob osb = build_offset_stream_blob(sched_blob);
						check_cuda(cuMemAlloc(&hll_ostream, osb.data.size()), "cuMemAlloc(map1_frontier_hll_ostream)");
						check_cuda(cuMemcpyHtoD(hll_ostream, osb.data.data(), osb.data.size()), "HtoD(map1_frontier_hll_ostream)");
						CUdeviceptr hll_regular = hll_ostream;
						CUdeviceptr hll_alg0 = hll_regular + osb.stream_bytes;
						CUdeviceptr hll_alg6 = hll_alg0 + osb.stream_bytes;
						CUdeviceptr hll_alg2 = hll_alg6 + osb.stream_bytes;
						CUdeviceptr hll_alg5 = hll_alg2 + osb.carry_bytes;
						auto run_producer_hll = [&](uint32_t tau, const char* label) -> double
						{
							uint64_t remaining = total;
							uint64_t start64 = args.range_start;
							uint32_t hm0 = 0u, hm1 = args.map1_frontier_depth, htau = tau;
							check_cuda(cuCtxSynchronize(), "sync(map1_frontier_hll_pre)");
							const auto h0 = std::chrono::steady_clock::now();
							while (remaining > 0ull)
							{
								const uint64_t hchunk64 = std::min<uint64_t>(remaining, 1ull << 31);
								const uint32_t hMN = static_cast<uint32_t>(hchunk64);
								const uint32_t hds = static_cast<uint32_t>(start64);
								void* hargs[] = { (void*)&hMN, &hm0, &hm1, &hll_regular, &hll_alg0, &hll_alg6,
									&hll_alg2, &hll_alg5, &assets.expansion_values, &assets.schedule_data,
									&args.key_id, (void*)&hds, &hll_regs, &htau };
								// 64-bit grid math: hMN can be large enough that hMN + 63 overflows uint32.
								const uint32_t hgrid = static_cast<uint32_t>((hchunk64 + 63ull) / 64ull);
								check_cuda(cuLaunchKernel(run_span_hll_kernel, hgrid,1,1,256,1,1,0,0,hargs,nullptr), label);
								remaining -= hchunk64;
								start64 += hchunk64;
							}
							check_cuda(cuCtxSynchronize(), label);
							return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - h0).count();
						};
						const double hll_ms = run_producer_hll(0u, "launch(map1_frontier_hll)");
						std::vector<uint32_t> regs(4096);
						check_cuda(cuMemcpyDtoH(regs.data(), hll_regs, 4096u * sizeof(uint32_t)), "DtoH(map1_frontier_hll_regs)");
						const uint64_t frontier_est = hll_estimate(regs);
						target_slots_total = args.map1_frontier_wide
							? std::max<uint64_t>(1024ull, frontier_est * 5ull / 4ull)
							: std::max<uint64_t>(1024ull, frontier_est * 2ull);
						std::cout << "  producer HLL-auto: frontier est=" << frontier_est
						          << " (" << (frontier_est >> 20) << "M, " << std::fixed << std::setprecision(2)
						          << (total ? (double)frontier_est / (double)total : 0.0) << "x window)  pass="
						          << std::setprecision(1) << hll_ms << " ms  -> target "
						          << target_slots_total << " slots before pow2/partition/cap"
						          << (args.map1_frontier_wide ? " (1.25x wide)" : " (2x legacy)") << "\n";
						if (args.map1_frontier_route_tau > 0u)
						{
							// Second HLL pass with tau=N: measure routed cardinality, then
							// re-size the table to 2x(routed cardinality) — the insert kernel
							// will only hash high-shed states (shed>=tau), so the table needs
							// to hold the ROUTED unique states, not the full frontier.
							check_cuda(cuMemsetD8(hll_regs, 0, 4096u * sizeof(uint32_t)), "memset(map1_frontier_hll_regs_routed)");
							const double rhl_ms = run_producer_hll(args.map1_frontier_route_tau, "launch(map1_frontier_hll_routed)");
							check_cuda(cuMemcpyDtoH(regs.data(), hll_regs, 4096u * sizeof(uint32_t)), "DtoH(map1_frontier_hll_regs_routed)");
							const uint64_t routed_est = hll_estimate(regs);
							target_slots_total = args.map1_frontier_wide
								? std::max<uint64_t>(1024ull, routed_est * 5ull / 4ull)
								: std::max<uint64_t>(1024ull, routed_est * 2ull);
							uint32_t routed_logm = 1u;
							while (routed_logm < 32u && (1ull << routed_logm) < ((target_slots_total + partitions - 1u) / partitions)) ++routed_logm;
							const uint64_t routed_mb = ((1ull << routed_logm) * partitions * slot_bytes) >> 20;
							std::cout << "  routed-HLL (tau=" << args.map1_frontier_route_tau << "): routed est=" << routed_est
							          << " (" << (routed_est >> 20) << "M, " << std::fixed << std::setprecision(3)
							          << (frontier_est ? (double)routed_est / (double)frontier_est : 0.0) << "x frontier)  pass="
							          << std::setprecision(1) << rhl_ms << " ms"
							          << "  -> routed table 2^" << routed_logm << " x " << partitions
							          << " = " << routed_mb << " MB (1.25x routed est)" << "\n";
						}
						cuMemFree(hll_regs);
						cuMemFree(hll_ostream);
					}
					const uint64_t desired_slots = std::max<uint64_t>(1024ull, (target_slots_total + partitions - 1u) / partitions);
					uint32_t desired_logm = 1u;
					while (desired_logm < 32u && (1ull << desired_logm) < desired_slots) ++desired_logm;
				uint64_t cap_bytes = 0;
				if (args.map1_frontier_table_mb != 0u)
				{
					cap_bytes = static_cast<uint64_t>(args.map1_frontier_table_mb) << 20;
				}
				else
				{
					const uint64_t reserve = 1024ull << 20; // leave room for context/assets/driver bookkeeping
					const uint64_t label_bytes = partitions > 1u ? total : 0ull;
					cap_bytes = free_b > reserve + label_bytes
						? static_cast<uint64_t>((free_b - reserve - label_bytes) * 0.70)
						: 0ull;
				}
				uint32_t cap_logm = 1u;
				const uint64_t cap_slots = cap_bytes / slot_bytes;
				while (cap_logm < 32u && (1ull << (cap_logm + 1u)) <= cap_slots) ++cap_logm;
				const uint32_t logm = std::min(desired_logm, cap_logm);
				const uint64_t slots = 1ull << logm;
				if (slots < 1024ull)
					throw std::runtime_error("--map1-frontier table budget too small");

				uint32_t rep_cap = 0u;
				if (args.map1_frontier_consume || args.map1_frontier_raceway)
				{
					if (args.map1_frontier_rep_cap != 0u)
					{
						rep_cap = args.map1_frontier_rep_cap;
					}
					else
					{
						const uint64_t auto_cap_limit = 268435456ull; // 1 GiB rep list; explicit cap for larger jobs
						if (total > auto_cap_limit)
							throw std::runtime_error("large MAP1 representative emission needs --map1-frontier-rep-cap or --raceway-rep-cap");
						rep_cap = static_cast<uint32_t>(total);
					}
				}

					const uint32_t producer_mult_cap = args.map1_frontier_multiplicity ? static_cast<uint32_t>(total) : 0u;
					CUdeviceptr table_fp = 0, table_owner = 0, d_unique = 0, d_overflow = 0, d_rep = 0, d_rep_mult = 0, partition_labels = 0;
				// VRAM guard: a contended GPU (another process) can leave too little free for the
				// MAP1 table. A bare cuMemAlloc OOM is easy to swallow in scripts (empty F1), so
				// fail LOUDLY first with a distinctive, greppable token.
				{
					size_t guard_free = 0, guard_total = 0;
					check_cuda(cuMemGetInfo(&guard_free, &guard_total), "cuMemGetInfo(map1_table_guard)");
					const size_t need = static_cast<size_t>(slots) * slot_bytes;
					const size_t margin = 512ull << 20;  // counters + chunk buffers + driver headroom
					if (guard_free < need + margin)
					{
						std::ostringstream e;
						e << "INSUFFICIENT_VRAM for MAP1 table: need " << (need >> 20) << " MiB + "
						  << (margin >> 20) << " MiB margin, only " << (guard_free >> 20) << " MiB free of "
						  << (guard_total >> 20) << " MiB on device " << args.device_index
						  << " (another process may be using this GPU). Free the GPU or lower --map1-frontier-table-mb.";
						throw std::runtime_error(e.str());
					}
				}
					check_cuda(cuMemAlloc(&table_fp, static_cast<size_t>(slots) * slot_bytes), "cuMemAlloc(map1_table_fp)");
					if (args.map1_frontier_multiplicity)
						check_cuda(cuMemAlloc(&table_owner, static_cast<size_t>(slots) * sizeof(uint32_t)), "cuMemAlloc(map1_table_owner)");
					check_cuda(cuMemAlloc(&d_unique, sizeof(uint32_t)), "cuMemAlloc(map1_unique)");
					check_cuda(cuMemAlloc(&d_overflow, sizeof(uint32_t)), "cuMemAlloc(map1_overflow)");
					if (rep_cap != 0u)
						check_cuda(cuMemAlloc(&d_rep, static_cast<size_t>(rep_cap) * sizeof(uint32_t)), "cuMemAlloc(map1_rep)");
					if (args.map1_frontier_multiplicity)
						check_cuda(cuMemAlloc(&d_rep_mult, static_cast<size_t>(producer_mult_cap) * sizeof(uint32_t)), "cuMemAlloc(map1_rep_mult)");
					if (partitions > 1u)
						check_cuda(cuMemAlloc(&partition_labels, static_cast<size_t>(total)), "cuMemAlloc(map1_partition_labels)");

						CUdeviceptr stream_unique_flags = 0, stream_rep = 0, stream_rep_idx = 0, stream_mult = 0, stream_state = 0, stream_alive = 0, stream_traj_sketch = 0;
						CUdeviceptr frontier_route_stats = 0;
					CUdeviceptr producer_occ_counter = 0;
				CUdeviceptr stream_live_a = 0, stream_live_b = 0, stream_counter = 0, stream_flag = 0, stream_ostream = 0;
				CUdeviceptr stream_off_regular = 0, stream_off_alg0 = 0, stream_off_alg6 = 0, stream_off_alg2 = 0, stream_off_alg5 = 0;
				std::vector<uint32_t> stream_cuts;
				uint64_t stream_final_frontier = 0ull;
					if (count_slots16_any_kernel)
						check_cuda(cuMemAlloc(&producer_occ_counter, sizeof(uint32_t)), "cuMemAlloc(producer_occ_counter)");
					if (use_frontier_traj)
						check_cuda(cuMemAlloc(&frontier_route_stats, 5ull * sizeof(unsigned long long)), "cuMemAlloc(frontier_route_stats)");
				auto count_slots16_any = [&](CUdeviceptr table, uint64_t nslots) -> uint32_t
				{
					if (!producer_occ_counter || !count_slots16_any_kernel || !table || nslots == 0ull || nslots > static_cast<uint64_t>(UINT32_MAX))
						return 0u;
					check_cuda(cuMemsetD32(producer_occ_counter, 0u, 1u), "memset(producer_occ_counter)");
					uint32_t n = static_cast<uint32_t>(nslots);
					void* oa[] = { &table, &n, &producer_occ_counter };
					check_cuda(cuLaunchKernel(count_slots16_any_kernel, (n + 255u) / 256u, 1, 1,
						256, 1, 1, 0, 0, oa, nullptr), "launch(count_slots16_any)");
					check_cuda(cuCtxSynchronize(), "sync(count_slots16_any)");
					uint32_t occupied = 0u;
					check_cuda(cuMemcpyDtoH(&occupied, producer_occ_counter, sizeof(uint32_t)), "DtoH(producer_occ_counter)");
					return occupied;
				};
					// Pool B: persistent cross-chunk deep cap, allocated + zeroed ONCE so it
					// accumulates across the whole window's chunks. 16 B/slot (epoch16|fp48 in
					// word0), benign-race, FN-safe over-keep. Lives across the partition/chunk loop.
					CUdeviceptr frontier_cap_table = 0;
					uint64_t frontier_cap_slots = 0ull;
					std::vector<CUdeviceptr> frontier_cap_tables;   // K-distinct: one dedicated table per deep boundary
					uint32_t frontier_cap_ntab = 1u;
					if (use_frontier_cap)
					{
						frontier_cap_slots = ((uint64_t)1u << args.map1_frontier_drain_cap_bits) * args.map1_frontier_drain_cap_ways;
						if (args.map1_frontier_drain_cap_distinct)
						{
							// One table per mode==1 boundary (= cuts.size()-3). Each table only ever
							// holds one depth ⇒ no cross-depth eviction (epoch stays constant within it).
							const auto cuts_tmp = make_post_map1_cuts(args.map1_frontier_deep_k);
							frontier_cap_ntab = cuts_tmp.size() >= 4u ? (uint32_t)(cuts_tmp.size() - 3u) : 1u;
						}
						frontier_cap_tables.resize(frontier_cap_ntab, 0);
						for (uint32_t ti = 0u; ti < frontier_cap_ntab; ti++)
						{
							check_cuda(cuMemAlloc(&frontier_cap_tables[ti], frontier_cap_slots * 16ull), "cuMemAlloc(frontier_cap)");
							check_cuda(cuMemsetD8(frontier_cap_tables[ti], 0, frontier_cap_slots * 16ull), "memset(frontier_cap once)");
						}
						frontier_cap_table = frontier_cap_tables[0];
						std::cout << "  producer deep cap (Pool B): 2^" << args.map1_frontier_drain_cap_bits << " x "
						          << args.map1_frontier_drain_cap_ways << " = " << frontier_cap_slots << " slots = "
						          << ((frontier_cap_slots * 16ull) >> 20) << " MB/table x " << frontier_cap_ntab
						          << (args.map1_frontier_drain_cap_distinct ? " (distinct per-depth)" : "")
						          << " = " << (((frontier_cap_slots * 16ull) * frontier_cap_ntab) >> 20)
						          << " MB total (persists across chunks)  cas=64a\n";
							// Route through the experimental gate kernels or the plain cap kernel.
							frontier_span_k = use_frontier_traj ? frontier_span_traj_k
								: (use_frontier_shed_gate ? frontier_span_shed_gate_k
									: (use_frontier_fp_gate ? frontier_span_fp_gate_k
										: (use_frontier_shed ? frontier_span_shed_k : frontier_span_cap_k)));
							if (use_frontier_shed)
								std::cout << "  producer MAP1 shed-proxy: shed-tau=" << args.map1_frontier_drain_shed_tau
								          << " (alg0/alg6 count at MAP1; low-shed survivors discarded before drain)\n";
							if (use_frontier_shed_gate)
								std::cout << "  producer Pool B shed gate: shed_tau=" << args.map1_frontier_drain_shed_gate_tau
								          << " (alg0/alg6 count over each deep span; low-shed states pass alive)\n";
							if (use_frontier_fp_gate)
								std::cout << "  producer Pool B fp-sampling gate: log=" << args.map1_frontier_drain_fp_gate_log
								          << " (routes about 1/" << (1u << args.map1_frontier_drain_fp_gate_log)
								          << " states; classifier-overhead floor test)\n";
							if (use_frontier_traj)
							{
								if (args.map1_frontier_drain_traj_bits > 0u)
									std::cout << "  producer Pool B trajgate: sketch=2^" << args.map1_frontier_drain_traj_bits
									          << " x 4 u32 rows  dens_tau=" << args.map1_frontier_drain_dens_tau
									          << " alg0_tau=" << args.map1_frontier_drain_alg0_tau
									          << " (online atomic-return; reset per drain boundary/chunk)\n";
								else if (use_frontier_sticky_only)
									std::cout << "  producer Pool B sticky-only gate: alg0_tau="
									          << args.map1_frontier_drain_alg0_tau
									          << " (no trajDens sketch)\n";
								if (args.map1_frontier_drain_mult_tau > 0u)
									std::cout << "  producer Pool B multiplicity gate: mult_tau="
									          << args.map1_frontier_drain_mult_tau
									          << " (OR term with traj/sticky; requires carried representative counts)\n";
							}
							}
				double stream_deep_ms = 0.0;
				if (args.map1_frontier_stream_deep)
				{
						check_cuda(cuMemAlloc(&stream_unique_flags, chunk_max), "cuMemAlloc(stream_unique_flags)");
						check_cuda(cuMemAlloc(&stream_rep, static_cast<size_t>(chunk_max) * sizeof(uint32_t)), "cuMemAlloc(stream_rep)");
						if (args.map1_frontier_multiplicity)
						{
							check_cuda(cuMemAlloc(&stream_rep_idx, static_cast<size_t>(chunk_max) * sizeof(uint32_t)), "cuMemAlloc(stream_rep_idx)");
							check_cuda(cuMemAlloc(&stream_mult, static_cast<size_t>(chunk_max) * sizeof(uint32_t)), "cuMemAlloc(stream_mult)");
						}
						check_cuda(cuMemAlloc(&stream_state, static_cast<size_t>(chunk_max) * 32u * sizeof(uint32_t)), "cuMemAlloc(stream_state)");
					check_cuda(cuMemAlloc(&stream_alive, chunk_max), "cuMemAlloc(stream_alive)");
					check_cuda(cuMemAlloc(&stream_live_a, static_cast<size_t>(chunk_max) * sizeof(uint32_t)), "cuMemAlloc(stream_live_a)");
						check_cuda(cuMemAlloc(&stream_live_b, static_cast<size_t>(chunk_max) * sizeof(uint32_t)), "cuMemAlloc(stream_live_b)");
						check_cuda(cuMemAlloc(&stream_counter, sizeof(uint32_t)), "cuMemAlloc(stream_counter)");
						check_cuda(cuMemAlloc(&stream_flag, chunk_max), "cuMemAlloc(stream_flag)");
						if (use_frontier_traj && args.map1_frontier_drain_traj_bits > 0u)
						{
							const uint64_t traj_cells = 1ull << args.map1_frontier_drain_traj_bits;
							check_cuda(cuMemAlloc(&stream_traj_sketch, traj_cells * 4ull * sizeof(uint32_t)), "cuMemAlloc(stream_traj_sketch)");
						}
						const std::vector<uint8_t> sched_blob = build_schedule_blob(args.key_id, args.map_list);
					OffsetStreamBlob osb = build_offset_stream_blob(sched_blob);
					check_cuda(cuMemAlloc(&stream_ostream, osb.data.size()), "cuMemAlloc(stream_ostream)");
					check_cuda(cuMemcpyHtoD(stream_ostream, osb.data.data(), osb.data.size()), "HtoD(stream_ostream)");
					stream_off_regular = stream_ostream;
					stream_off_alg0 = stream_off_regular + osb.stream_bytes;
					stream_off_alg6 = stream_off_alg0 + osb.stream_bytes;
					stream_off_alg2 = stream_off_alg6 + osb.stream_bytes;
					stream_off_alg5 = stream_off_alg2 + osb.carry_bytes;
					stream_cuts = args.map1_frontier_stream_cuts.empty()
						? make_post_map1_cuts(args.map1_frontier_deep_k)
						: make_post_map1_cuts_custom(parse_u32_list(args.map1_frontier_stream_cuts, "--map1-frontier-stream-cuts"));
				}

				double classify_ms = 0.0;
				if (partitions > 1u)
				{
					const auto c0 = std::chrono::high_resolution_clock::now();
					uint64_t classified = 0ull;
					while (classified < total)
					{
						const uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(chunk_max, total - classified));
						const uint32_t data_start = static_cast<uint32_t>(args.range_start + classified);
						CUdeviceptr label_out = partition_labels + classified;
						void* cargs[] = {
							&label_out, &partition_log,
							&assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
							&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
							&assets.alg2_values, &assets.alg5_values, &assets.expansion_values,
							&assets.schedule_data, &args.key_id, (void*)&data_start, (void*)&chunk };
						check_cuda(cuLaunchKernel(classify_k, screen_kernel_grid_x_ilp(chunk, 1u), 1, 1,
							kCudaThreadsPerBlock, 1, 1, 0, 0, cargs, nullptr), "launch(map1_frontier_classify)");
						classified += chunk;
					}
					check_cuda(cuCtxSynchronize(), "sync(map1_frontier_classify)");
					const auto c1 = std::chrono::high_resolution_clock::now();
					classify_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
				}

				const auto t0 = std::chrono::high_resolution_clock::now();
				if (frontier_route_stats)
					check_cuda(cuMemsetD8(frontier_route_stats, 0u, 5ull * sizeof(unsigned long long)), "memset(frontier_route_stats)");
				double stream_insert_ms = 0.0;
				double table_zero_ms = 0.0;
				uint64_t unique_total = 0ull, overflow_total = 0ull;
				std::vector<uint32_t> partition_unique(partitions, 0u);
				std::vector<uint32_t> partition_overflow(partitions, 0u);
				for (uint32_t partition = 0u; partition < partitions; ++partition)
				{
					const auto z0 = std::chrono::high_resolution_clock::now();
					check_cuda(cuMemsetD32(d_unique, 0u, 1u), "memset(map1_unique)");
					check_cuda(cuMemsetD32(d_overflow, 0u, 1u), "memset(map1_overflow)");
						if (args.map1_frontier_wide)
						{
							// Empty slot == (0,0); a plain byte-zero clears the whole 16 B/slot table.
							check_cuda(cuMemsetD8(table_fp, 0u, static_cast<size_t>(slots) * slot_bytes), "memset(map1_table128)");
							if (table_owner)
								check_cuda(cuMemsetD8(table_owner, 0xFFu, static_cast<size_t>(slots) * sizeof(uint32_t)), "memset(map1_table_owner)");
						}
					else
					{
						void* zargs[] = { &table_fp, (void*)&logm };
						check_cuda(cuLaunchKernel(zero_k, static_cast<uint32_t>((slots + 255ull) / 256ull), 1, 1, 256, 1, 1, 0, 0, zargs, nullptr), "launch(map1_frontier_zero)");
					}
					check_cuda(cuCtxSynchronize(), "sync(map1_frontier_zero)");
					const auto z1 = std::chrono::high_resolution_clock::now();
					table_zero_ms += std::chrono::duration<double, std::milli>(z1 - z0).count();

					uint64_t done = 0;
					while (done < total)
					{
						const uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(chunk_max, total - done));
						const uint32_t data_start = static_cast<uint32_t>(args.range_start + done);
							if (args.map1_frontier_stream_deep)
								check_cuda(cuMemsetD8(stream_unique_flags, 0u, chunk), "memset(stream_unique_flags)");
							CUdeviceptr unique_out = args.map1_frontier_stream_deep ? stream_unique_flags : 0;
							CUdeviceptr label_in = partitions > 1u ? partition_labels + done : 0;
							CUdeviceptr rep_idx_out = args.map1_frontier_multiplicity ? stream_rep_idx : 0;
							CUdeviceptr rep_mult_arg = args.map1_frontier_multiplicity ? d_rep_mult : 0;
							const auto si0 = std::chrono::high_resolution_clock::now();
							const uint32_t insert_route_tau = (args.map1_frontier_wide && args.map1_frontier_depth == 1u && args.map1_frontier_sample_mult == 1u) ? args.map1_frontier_route_tau : 0u;
								void* iargs[] = {
									&table_fp, (void*)&logm, &d_unique, &d_overflow, &d_rep, (void*)&rep_cap,
								&unique_out, &label_in, &partition,
								&assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
									&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
									&assets.alg2_values, &assets.alg5_values, &assets.expansion_values,
									&assets.schedule_data, &args.key_id, (void*)&data_start, (void*)&chunk,
									(void*)&insert_route_tau, (void*)&insert_depth, (void*)&insert_sample_mult,
									&table_owner, &rep_mult_arg, (void*)&producer_mult_cap, &rep_idx_out };
							check_cuda(cuLaunchKernel(insert_k, screen_kernel_grid_x_ilp(chunk, 1u), 1, 1,
								kCudaThreadsPerBlock, 1, 1, 0, 0, iargs, nullptr), "launch(map1_frontier_insert)");
						if (args.map1_frontier_stream_deep)
						{
							check_cuda(cuCtxSynchronize(), "sync(stream_map1_frontier_insert)");
							const auto si1 = std::chrono::high_resolution_clock::now();
							stream_insert_ms += std::chrono::duration<double, std::milli>(si1 - si0).count();

								const auto sd0 = std::chrono::high_resolution_clock::now();
								check_cuda(cuMemsetD32(stream_counter, 0u, 1u), "memset(stream_pack_counter)");
								if (args.map1_frontier_multiplicity)
								{
									void* pa[] = { &stream_unique_flags, (void*)&chunk, (void*)&data_start,
										&stream_rep_idx, &d_rep_mult, (void*)&producer_mult_cap,
										&stream_rep, &stream_mult, &stream_counter };
									check_cuda(cuLaunchKernel(frontier_compact_data_mult_k, (chunk + 255u) / 256u, 1, 1,
										256, 1, 1, 0, 0, pa, nullptr), "launch(stream_compact_data_mult)");
								}
								else
								{
									void* pa[] = { &stream_unique_flags, (void*)&chunk, (void*)&data_start, &stream_rep, &stream_counter };
									check_cuda(cuLaunchKernel(frontier_compact_data_k, (chunk + 255u) / 256u, 1, 1,
										256, 1, 1, 0, 0, pa, nullptr), "launch(stream_compact_data)");
								}
							check_cuda(cuCtxSynchronize(), "sync(stream_compact_data)");
							uint32_t M = 0u;
							check_cuda(cuMemcpyDtoH(&M, stream_counter, sizeof(uint32_t)), "DtoH(stream_pack_counter)");

							if (M != 0u)
							{
								constexpr uint32_t F_WARPS = 8u;
								constexpr uint32_t F_ILP = 10u;
								constexpr uint32_t F_W = F_WARPS * F_ILP;
								constexpr uint32_t F_BLOCK = F_WARPS * 32u;
									CUdeviceptr cur_live = 0;
									CUdeviceptr next_live = stream_live_a;
									CUdeviceptr span_mult = args.map1_frontier_multiplicity ? stream_mult : 0;
									int first = 1;
								for (uint32_t sp = 0; sp + 1 < stream_cuts.size(); ++sp)
								{
									if (M == 0u) break;
									uint32_t m0 = stream_cuts[sp], m1 = stream_cuts[sp + 1u];
									const bool last = (sp + 2u == stream_cuts.size());
										// K-distinct: route this boundary's probe to its dedicated table
										// (sa[] holds &frontier_cap_table, so update the value before the launch).
										if (use_frontier_cap && args.map1_frontier_drain_cap_distinct
											&& sp >= 1u && (uint32_t)(sp - 1u) < frontier_cap_ntab)
											frontier_cap_table = frontier_cap_tables[sp - 1u];
										const int mode = (sp == 0u) ? 0 : (last ? 2 : 1);
										if (use_frontier_traj)
										{
												if (mode == 1 && stream_traj_sketch)
												{
													const uint64_t traj_cells = 1ull << args.map1_frontier_drain_traj_bits;
													check_cuda(cuMemsetD32(stream_traj_sketch, 0u, traj_cells * 4ull), "memset(stream_traj_sketch)");
												}
												uint32_t frontier_alg0_tau_arg = (args.map1_frontier_drain_traj_bits > 0u || use_frontier_sticky_only)
													? args.map1_frontier_drain_alg0_tau : 0u;
												void* sa[] = { &cur_live, &M, &stream_rep, &stream_state, &stream_alive, &m0, &m1, &first,
													&stream_off_regular, &stream_off_alg0, &stream_off_alg6, &stream_off_alg2, &stream_off_alg5,
													&assets.expansion_values, &assets.schedule_data, &args.key_id,
													&assets.carnival_data, &stream_flag, (void*)&mode,
													&span_mult,
													&frontier_cap_table, &args.map1_frontier_drain_cap_bits, &args.map1_frontier_drain_cap_ways,
													&stream_traj_sketch, &args.map1_frontier_drain_traj_bits,
													&args.map1_frontier_drain_dens_tau, &frontier_alg0_tau_arg,
													&args.map1_frontier_drain_mult_tau, &frontier_route_stats };
											check_cuda(cuLaunchKernel(frontier_span_k, (M + F_W - 1u) / F_W, 1, 1,
												F_BLOCK, 1, 1, 0, 0, sa, nullptr), "launch(stream_frontier_span_traj)");
										}
										else
										{
											uint32_t gate_arg = use_frontier_shed_gate
												? args.map1_frontier_drain_shed_gate_tau
												: (use_frontier_fp_gate ? args.map1_frontier_drain_fp_gate_log : args.map1_frontier_drain_shed_tau);
												void* sa[] = { &cur_live, &M, &stream_rep, &stream_state, &stream_alive, &m0, &m1, &first,
													&stream_off_regular, &stream_off_alg0, &stream_off_alg6, &stream_off_alg2, &stream_off_alg5,
													&assets.expansion_values, &assets.schedule_data, &args.key_id,
													&assets.carnival_data, &stream_flag, (void*)&mode,
													&span_mult,
													&frontier_cap_table, &args.map1_frontier_drain_cap_bits, &args.map1_frontier_drain_cap_ways,
													&gate_arg };
											check_cuda(cuLaunchKernel(frontier_span_k, (M + F_W - 1u) / F_W, 1, 1,
												F_BLOCK, 1, 1, 0, 0, sa, nullptr), "launch(stream_frontier_span)");
										}
										if (mode == 1)
										{
										check_cuda(cuMemsetD32(stream_counter, 0u, 1u), "memset(stream_frontier_counter)");
										int compact_first = (cur_live == 0) ? 1 : 0;
										void* ca[] = { &stream_alive, &cur_live, &M, &next_live, &stream_counter, &compact_first };
										check_cuda(cuLaunchKernel(frontier_compact_k, (M + 255u) / 256u, 1, 1,
											256, 1, 1, 0, 0, ca, nullptr), "launch(stream_frontier_compact)");
										check_cuda(cuCtxSynchronize(), "sync(stream_frontier_compact)");
										check_cuda(cuMemcpyDtoH(&M, stream_counter, sizeof(uint32_t)), "DtoH(stream_frontier_counter)");
										cur_live = next_live;
										next_live = (next_live == stream_live_a) ? stream_live_b : stream_live_a;
									}
									first = 0;
								}
							}
							check_cuda(cuCtxSynchronize(), "sync(stream_frontier_deep)");
							stream_final_frontier += M;
							const auto sd1 = std::chrono::high_resolution_clock::now();
							stream_deep_ms += std::chrono::duration<double, std::milli>(sd1 - sd0).count();
						}
						done += chunk;
					}
					check_cuda(cuCtxSynchronize(), "sync(map1_frontier_partition)");
					check_cuda(cuMemcpyDtoH(&partition_unique[partition], d_unique, sizeof(uint32_t)), "DtoH(map1_partition_unique)");
					check_cuda(cuMemcpyDtoH(&partition_overflow[partition], d_overflow, sizeof(uint32_t)), "DtoH(map1_partition_overflow)");
					unique_total += partition_unique[partition];
					overflow_total += partition_overflow[partition];
				}
				check_cuda(cuCtxSynchronize(), "sync(map1_frontier_insert)");
				const auto t1 = std::chrono::high_resolution_clock::now();

				const uint32_t unique = partition_unique[0];
				uint64_t map1_table_occupied = 0ull;
				if (args.map1_frontier_wide && partitions == 1u)
					map1_table_occupied = count_slots16_any(table_fp, slots);
				uint64_t poolb_occupied = 0ull;
				if (use_frontier_cap)
				{
					for (CUdeviceptr fct : frontier_cap_tables)
						poolb_occupied += count_slots16_any(fct, frontier_cap_slots);
				}
				const uint64_t poolb_slots_total = use_frontier_cap ? frontier_cap_slots * frontier_cap_ntab : 0ull;
				const double ms = args.map1_frontier_stream_deep
					? stream_insert_ms
					: std::chrono::duration<double, std::milli>(t1 - t0).count();
				const uint32_t max_partition_unique = *std::max_element(partition_unique.begin(), partition_unique.end());
				const double load = static_cast<double>(max_partition_unique) / static_cast<double>(slots);
				std::cout << "[map1-frontier] device=" << device_name
				          << " key=0x" << std::hex << args.key_id << std::dec
				          << " window=" << total << " data_start=" << args.range_start << "\n";
				std::cout << "  F1=" << unique_total << "  R_map1=" << (static_cast<double>(total) / static_cast<double>(unique_total ? unique_total : 1ull))
				          << "  overflow=" << overflow_total << "\n";
				std::cout << "  table=" << slots << " slots (" << (static_cast<double>(slots) * slot_bytes / (1u << 20))
				          << " MB, max_load=" << std::fixed << std::setprecision(3) << load << ")"
				          << "  chunk=" << chunk_max << "\n";
					if (map1_table_occupied != 0ull)
					{
						const double table_lf = static_cast<double>(map1_table_occupied) / static_cast<double>(slots);
						std::cout << "  table_occ=" << map1_table_occupied << "/" << slots
						          << " slots  LF=" << std::fixed << std::setprecision(3) << table_lf;
						if (unique_total > map1_table_occupied)
							std::cout << "  bypassed_or_duplicate_reps~=" << (unique_total - map1_table_occupied);
						std::cout << "\n";
					}
					if (partitions > 1u)
					{
						std::cout << "  partitions=" << partitions << "  labels="
						          << (static_cast<double>(total) / (1u << 20)) << " MB  partition_F1:";
						for (uint32_t p = 0u; p < partitions; ++p)
							std::cout << " " << p << "=" << partition_unique[p];
						std::cout << "\n";
						std::cout << "  classify: " << std::fixed << std::setprecision(3) << classify_ms << " ms  ("
						          << std::setprecision(1) << (static_cast<double>(total) / 1e6 / (classify_ms / 1e3))
						          << " M cand/s)\n";
					}
					std::cout << "  map1+insert: " << std::fixed << std::setprecision(3) << ms << " ms  ("
					          << std::setprecision(1) << (static_cast<double>(total) / 1e6 / (ms / 1e3)) << " M cand/s)\n";
					if (args.map1_frontier_stream_deep)
					{
						const double stream_total_ms = classify_ms + table_zero_ms + ms + stream_deep_ms;
						std::cout << "  stream-deep postK=" << args.map1_frontier_deep_k
						          << ": final_frontier=" << stream_final_frontier
						          << "  R_total=" << std::fixed << std::setprecision(3)
						          << (static_cast<double>(total) / static_cast<double>(stream_final_frontier ? stream_final_frontier : 1ull))
						          << "  R_after_MAP1="
						          << (static_cast<double>(unique_total) / static_cast<double>(stream_final_frontier ? stream_final_frontier : 1ull))
						          << "  zero=" << std::fixed << std::setprecision(3) << table_zero_ms << " ms"
						          << "  deep=" << std::fixed << std::setprecision(3) << stream_deep_ms << " ms"
						          << "  total=" << stream_total_ms << " ms  ("
						          << std::setprecision(1) << (static_cast<double>(total) / 1e6 / (stream_total_ms / 1e3))
						          << " M cand/s)\n";
							if (use_frontier_cap && poolb_slots_total != 0ull)
							{
								const double poolb_lf = static_cast<double>(poolb_occupied) / static_cast<double>(poolb_slots_total);
								std::cout << "  PoolB_occ=" << poolb_occupied << "/" << poolb_slots_total
								          << " slots  LF=" << std::fixed << std::setprecision(3) << poolb_lf
								          << "  alloc=" << ((poolb_slots_total * 16ull) >> 20) << " MB";
								if (frontier_cap_ntab > 1u) std::cout << " across " << frontier_cap_ntab << " tables";
								std::cout << "\n";
							}
							if (frontier_route_stats)
							{
								unsigned long long frs[5] = {};
								check_cuda(cuMemcpyDtoH(frs, frontier_route_stats, sizeof(frs)), "DtoH(frontier_route_stats)");
								const unsigned long long total_decisions = frs[0] + frs[1];
								std::cout << "  PoolB_gate: routed=" << frs[0]
								          << "  passed=" << frs[1]
								          << "  route_rate=" << std::fixed << std::setprecision(1)
								          << (total_decisions ? 100.0 * static_cast<double>(frs[0]) / static_cast<double>(total_decisions) : 0.0)
								          << "%  sticky-saved=" << frs[2]
								          << "  mult-only=" << frs[3]
								          << "  cap_drops=" << frs[4] << "\n";
							}
						}

					if (args.map1_frontier_raceway)
					{
						if (unique > rep_cap)
							throw std::runtime_error("MAP1 representative list exceeded --raceway-rep-cap");
						const std::vector<uint8_t> sched_blob = build_schedule_blob(args.key_id, args.map_list);
						const uint32_t schedule_count = static_cast<uint32_t>(sched_blob.size() / 4u);
						const uint32_t cap_count = (args.raceway_cap_bits == 0u) ? 0u : args.raceway_cap_count;
						if (cap_count != 0u && args.raceway_first_cap_map + cap_count > schedule_count)
							throw std::runtime_error("--raceway-first-cap-map + --raceway-cap-count exceeds schedule length");
						if (args.raceway_flag_parity && cap_count != 0u)
							throw std::runtime_error("--raceway-flag-parity requires --raceway-cap-bits 0");
						if (args.raceway_flag_parity && unique > (1u << 24))
							throw std::runtime_error("--raceway-flag-parity is capped at 16M reps");

						std::vector<CUdeviceptr> cap_tables_h(cap_count, 0);
						std::vector<uint32_t> cap_bits_h(cap_count, args.raceway_cap_bits);
						std::vector<uint32_t> cap_ways_h(cap_count, args.raceway_cap_ways);
						CUdeviceptr d_cap_tables = 0, d_cap_bits = 0, d_cap_ways = 0;
						uint64_t cap_slots_total = 0ull;
						for (uint32_t c = 0u; c < cap_count; ++c)
						{
							const uint64_t cap_slots = (1ull << args.raceway_cap_bits) * static_cast<uint64_t>(args.raceway_cap_ways);
							cap_slots_total += cap_slots;
							check_cuda(cuMemAlloc(&cap_tables_h[c], cap_slots * sizeof(unsigned long long)), "cuMemAlloc(raceway_cap)");
							void* za[] = { &cap_tables_h[c], (void*)&cap_slots };
							check_cuda(cuLaunchKernel(raceway_clear_k, static_cast<uint32_t>((cap_slots + 255ull) / 256ull), 1, 1,
								256, 1, 1, 0, 0, za, nullptr), "launch(raceway_cap_clear)");
						}
						if (cap_count != 0u)
						{
							check_cuda(cuMemAlloc(&d_cap_tables, static_cast<size_t>(cap_count) * sizeof(CUdeviceptr)), "cuMemAlloc(raceway_cap_ptrs)");
							check_cuda(cuMemcpyHtoD(d_cap_tables, cap_tables_h.data(), static_cast<size_t>(cap_count) * sizeof(CUdeviceptr)), "HtoD(raceway_cap_ptrs)");
							check_cuda(cuMemAlloc(&d_cap_bits, static_cast<size_t>(cap_count) * sizeof(uint32_t)), "cuMemAlloc(raceway_cap_bits)");
							check_cuda(cuMemcpyHtoD(d_cap_bits, cap_bits_h.data(), static_cast<size_t>(cap_count) * sizeof(uint32_t)), "HtoD(raceway_cap_bits)");
							check_cuda(cuMemAlloc(&d_cap_ways, static_cast<size_t>(cap_count) * sizeof(uint32_t)), "cuMemAlloc(raceway_cap_ways)");
							check_cuda(cuMemcpyHtoD(d_cap_ways, cap_ways_h.data(), static_cast<size_t>(cap_count) * sizeof(uint32_t)), "HtoD(raceway_cap_ways)");
						}

						CUdeviceptr work_counter = 0, survivor_count = 0, race_stats = 0;
						CUdeviceptr race_survivor_offsets = 0, race_survivor_flags = 0;
						CUdeviceptr race_drop_map = 0;
						check_cuda(cuMemAlloc(&work_counter, sizeof(uint32_t)), "cuMemAlloc(raceway_work_counter)");
						check_cuda(cuMemAlloc(&survivor_count, sizeof(uint32_t)), "cuMemAlloc(raceway_survivor_count)");
						check_cuda(cuMemAlloc(&race_stats, sizeof(RacewayStatsHost)), "cuMemAlloc(raceway_stats)");
						if (args.raceway_flag_parity)
						{
							check_cuda(cuMemAlloc(&race_survivor_offsets, static_cast<size_t>(unique) * sizeof(uint32_t)), "cuMemAlloc(raceway_survivor_offsets)");
							check_cuda(cuMemAlloc(&race_survivor_flags, unique), "cuMemAlloc(raceway_survivor_flags)");
						}
						if (args.raceway_drop_hist)
						{
							check_cuda(cuMemAlloc(&race_drop_map, unique), "cuMemAlloc(raceway_drop_map)");
							check_cuda(cuMemsetD8(race_drop_map, 0xFFu, unique), "memset(raceway_drop_map)");
						}
						check_cuda(cuMemsetD32(work_counter, 0u, 1u), "memset(raceway_work_counter)");
						check_cuda(cuMemsetD32(survivor_count, 0u, 1u), "memset(raceway_survivor_count)");
						check_cuda(cuMemsetD8(race_stats, 0u, sizeof(RacewayStatsHost)), "memset(raceway_stats)");

						CUdeviceptr null_u32 = 0, null_u8 = 0;
						uint32_t race_rep_count = unique;
						uint32_t race_cap_count = cap_count;
						uint32_t race_schedule_count = schedule_count;
						uint32_t race_first_cap_map = args.raceway_first_cap_map;
						const uint32_t race_data_start = static_cast<uint32_t>(args.range_start);
						uint32_t rep_values_are_absolute = 1u;
						const uint32_t race_blocks = std::max<uint32_t>(1u,
							std::min<uint32_t>(8192u, static_cast<uint32_t>((static_cast<uint64_t>(unique) + 7ull) / 8ull)));
						CUdeviceptr race_out_offsets_arg = args.raceway_flag_parity ? race_survivor_offsets : null_u32;
						CUdeviceptr race_out_flags_arg = args.raceway_flag_parity ? race_survivor_flags : null_u8;
						CUdeviceptr race_drop_map_arg = args.raceway_drop_hist ? race_drop_map : null_u8;
						void* ra[] = {
							&d_rep, &race_rep_count, &work_counter, &race_out_offsets_arg, &survivor_count, &race_out_flags_arg, &race_drop_map_arg, &race_stats,
							&d_cap_tables, &d_cap_bits, &d_cap_ways, &race_cap_count,
							&assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
							&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
							&assets.alg2_values, &assets.alg5_values, &assets.expansion_values,
							&assets.schedule_data, &assets.carnival_data, &args.key_id,
							(void*)&race_data_start, &rep_values_are_absolute,
							&race_schedule_count, &race_first_cap_map };
						check_cuda(cuCtxSynchronize(), "sync(raceway_pre)");
						const auto r0 = std::chrono::high_resolution_clock::now();
						check_cuda(cuLaunchKernel(raceway_k, race_blocks, 1, 1, 256, 1, 1, 0, 0, ra, nullptr), "launch(raceway_stream)");
						check_cuda(cuCtxSynchronize(), "sync(raceway_stream)");
						const auto r1 = std::chrono::high_resolution_clock::now();

						uint32_t race_survivors = 0u;
						RacewayStatsHost hs;
						check_cuda(cuMemcpyDtoH(&race_survivors, survivor_count, sizeof(uint32_t)), "DtoH(raceway_survivor_count)");
						check_cuda(cuMemcpyDtoH(&hs, race_stats, sizeof(hs)), "DtoH(raceway_stats)");
						const double race_ms = std::chrono::duration<double, std::milli>(r1 - r0).count();
						std::cout << "  raceway caps=" << cap_count
						          << " first_map=" << args.raceway_first_cap_map
						          << " cap_shape=";
						if (cap_count == 0u)
							std::cout << "none";
						else
							std::cout << "2^" << args.raceway_cap_bits << "x" << args.raceway_cap_ways
							          << " (" << (cap_slots_total * sizeof(unsigned long long) / (1ull << 20)) << " MB)";
						std::cout << "  reps=" << unique
						          << "  completed=" << race_survivors
						          << "  dropped=" << hs.reps_dropped
						          << "  map_evals=" << hs.map_evals
						          << "  time=" << std::fixed << std::setprecision(3) << race_ms << " ms  ("
						          << std::setprecision(1) << (static_cast<double>(unique) / 1e6 / (race_ms / 1e3))
						          << " M rep/s)\n";

						if (args.raceway_drop_hist)
						{
							std::vector<uint8_t> h_drop(unique);
							check_cuda(cuMemcpyDtoH(h_drop.data(), race_drop_map, unique), "DtoH(raceway_drop_map)");
							std::vector<uint64_t> hist(schedule_count, 0ull);
							uint64_t hist_total = 0ull, hist_oob = 0ull;
							for (uint32_t i = 0u; i < unique; ++i)
							{
								if (h_drop[i] == 0xFFu) continue;
								++hist_total;
								if (h_drop[i] < hist.size()) ++hist[h_drop[i]];
								else ++hist_oob;
							}
							std::cout << "  raceway drop histogram:";
							for (uint32_t m = 0u; m < hist.size(); ++m)
								if (hist[m] != 0ull) std::cout << " m" << m << "=" << hist[m];
							std::cout << "  total=" << hist_total;
							if (hist_oob != 0ull) std::cout << " oob=" << hist_oob;
							std::cout << "\n";
						}

						if (args.raceway_flag_parity)
						{
							CUdeviceptr d_ref_flags = 0, d_ostream = 0;
							check_cuda(cuMemAlloc(&d_ref_flags, unique), "cuMemAlloc(raceway_ref_flags)");
							OffsetStreamBlob osb = build_offset_stream_blob(sched_blob);
							check_cuda(cuMemAlloc(&d_ostream, osb.data.size()), "cuMemAlloc(raceway_ref_ostream)");
							check_cuda(cuMemcpyHtoD(d_ostream, osb.data.data(), osb.data.size()), "HtoD(raceway_ref_ostream)");
							CUdeviceptr off_regular = d_ostream;
							CUdeviceptr off_alg0 = off_regular + osb.stream_bytes;
							CUdeviceptr off_alg6 = off_alg0 + osb.stream_bytes;
							CUdeviceptr off_alg2 = off_alg6 + osb.stream_bytes;
							CUdeviceptr off_alg5 = off_alg2 + osb.carry_bytes;
							uint32_t nrep = unique;
							uint32_t data0 = 0u;
							void* oa[] = { &d_rep, &nrep, &off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
								&assets.expansion_values, &assets.schedule_data, &assets.carnival_data, &args.key_id, &data0, &d_ref_flags };
							check_cuda(cuLaunchKernel(ds_off_k, screen_kernel_grid_x_ilp(nrep, 6u), 1, 1,
								kCudaThreadsPerBlock, 1, 1, 0, 0, oa, nullptr), "launch(raceway_ref_offset)");
							check_cuda(cuCtxSynchronize(), "sync(raceway_ref_offset)");

							std::vector<uint32_t> h_race_offsets(race_survivors), h_reps(unique);
							std::vector<uint8_t> h_race_flags(race_survivors), h_ref_flags(unique);
							check_cuda(cuMemcpyDtoH(h_race_offsets.data(), race_survivor_offsets, static_cast<size_t>(race_survivors) * sizeof(uint32_t)), "DtoH(raceway_survivor_offsets)");
							check_cuda(cuMemcpyDtoH(h_race_flags.data(), race_survivor_flags, race_survivors), "DtoH(raceway_survivor_flags)");
							check_cuda(cuMemcpyDtoH(h_reps.data(), d_rep, static_cast<size_t>(unique) * sizeof(uint32_t)), "DtoH(raceway_reps)");
							check_cuda(cuMemcpyDtoH(h_ref_flags.data(), d_ref_flags, unique), "DtoH(raceway_ref_flags)");

							std::unordered_map<uint32_t, uint8_t> ref_by_data;
							ref_by_data.reserve(static_cast<size_t>(unique) * 2u);
							uint64_t ref_sum = 0ull;
							for (uint32_t i = 0u; i < unique; ++i)
							{
								ref_by_data.emplace(h_reps[i], h_ref_flags[i]);
								ref_sum += h_ref_flags[i];
							}
							std::unordered_set<uint32_t> seen_race;
							seen_race.reserve(static_cast<size_t>(race_survivors) * 2u);
							uint64_t missing = 0ull, diff = 0ull, dup = 0ull, race_sum = 0ull;
							for (uint32_t i = 0u; i < race_survivors; ++i)
							{
								race_sum += h_race_flags[i];
								if (!seen_race.insert(h_race_offsets[i]).second) ++dup;
								const auto it = ref_by_data.find(h_race_offsets[i]);
								if (it == ref_by_data.end()) { ++missing; continue; }
								if (it->second != h_race_flags[i]) ++diff;
							}
							const uint64_t count_mismatch = (race_survivors == unique) ? 0ull
								: static_cast<uint64_t>((race_survivors > unique) ? (race_survivors - unique) : (unique - race_survivors));
							std::cout << "  raceway flag parity: count_delta=" << count_mismatch
							          << " missing=" << missing
							          << " duplicate_outputs=" << dup
							          << " flag_diff=" << diff
							          << (count_mismatch == 0 && missing == 0 && dup == 0 && diff == 0 ? " [MATCH]" : " [DIFFER]")
							          << "  flag_sum(raceway)=" << race_sum
							          << "  flag_sum(ref)=" << ref_sum << "\n";

							cuMemFree(d_ref_flags);
							cuMemFree(d_ostream);
						}

						if (args.raceway_flat_parity)
						{
							uint32_t flat_n = static_cast<uint32_t>(total);
							uint32_t flat_data_start = static_cast<uint32_t>(args.range_start);
							const std::vector<uint8_t> sb = build_schedule_blob(args.key_id, args.map_list);
							OffsetStreamBlob osb = build_offset_stream_blob(sb);
							CUdeviceptr d_ostream = 0, d_flat_flags = 0;
							check_cuda(cuMemAlloc(&d_ostream, osb.data.size()), "cuMemAlloc(raceway_flat_ostream)");
							check_cuda(cuMemcpyHtoD(d_ostream, osb.data.data(), osb.data.size()), "HtoD(raceway_flat_ostream)");
							check_cuda(cuMemAlloc(&d_flat_flags, flat_n), "cuMemAlloc(raceway_flat_flags)");
							CUdeviceptr off_regular = d_ostream;
							CUdeviceptr off_alg0 = off_regular + osb.stream_bytes;
							CUdeviceptr off_alg6 = off_alg0 + osb.stream_bytes;
							CUdeviceptr off_alg2 = off_alg6 + osb.stream_bytes;
							CUdeviceptr off_alg5 = off_alg2 + osb.carry_bytes;

							void* flat_args[] = { &d_flat_flags, &off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
								&assets.expansion_values, &assets.schedule_data, &assets.carnival_data,
								&args.key_id, (void*)&flat_data_start, &flat_n };
							const auto f0 = std::chrono::high_resolution_clock::now();
							check_cuda(cuLaunchKernel(screen_offset_ilp6_preids_kernel, screen_kernel_grid_x_ilp(flat_n, 6u), 1, 1,
								kCudaThreadsPerBlock, 1, 1, 0, 0, flat_args, nullptr), "launch(raceway_flat_screen)");
							check_cuda(cuCtxSynchronize(), "sync(raceway_flat_screen)");
							const auto f1 = std::chrono::high_resolution_clock::now();
							const double flat_ms = std::chrono::duration<double, std::milli>(f1 - f0).count();

							uint32_t repflag_logm = 1u;
							const uint64_t repflag_target = std::max<uint64_t>(1024ull, static_cast<uint64_t>(race_survivors) * 2ull);
							while (repflag_logm < 31u && (1ull << repflag_logm) < repflag_target) ++repflag_logm;
							const uint64_t repflag_slots = 1ull << repflag_logm;
							CUdeviceptr repflag_fp = 0, repflag_flag = 0, build_counters = 0, parity_counters = 0;
							check_cuda(cuMemAlloc(&repflag_fp, repflag_slots * sizeof(uint64_t)), "cuMemAlloc(raceway_repflag_fp)");
							check_cuda(cuMemAlloc(&repflag_flag, repflag_slots), "cuMemAlloc(raceway_repflag_flag)");
							check_cuda(cuMemAlloc(&build_counters, 4ull * sizeof(unsigned long long)), "cuMemAlloc(raceway_repflag_build_counters)");
							check_cuda(cuMemAlloc(&parity_counters, 4ull * sizeof(unsigned long long)), "cuMemAlloc(raceway_flat_parity_counters)");
							void* zt[] = { &repflag_fp, (void*)&repflag_slots };
							check_cuda(cuLaunchKernel(raceway_clear_k, static_cast<uint32_t>((repflag_slots + 255ull) / 256ull), 1, 1,
								256, 1, 1, 0, 0, zt, nullptr), "launch(raceway_repflag_clear)");
							check_cuda(cuMemsetD8(build_counters, 0u, 4ull * sizeof(unsigned long long)), "memset(raceway_build_counters)");
							check_cuda(cuMemsetD8(parity_counters, 0u, 4ull * sizeof(unsigned long long)), "memset(raceway_parity_counters)");

							uint32_t abs_reps = 1u;
							uint32_t repflag_count = race_survivors;
							void* ba[] = { &repflag_fp, &repflag_flag, &repflag_logm,
								&race_survivor_offsets, &race_survivor_flags, &repflag_count,
								&assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
								&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
								&assets.alg2_values, &assets.alg5_values, &assets.expansion_values,
								&assets.schedule_data, &args.key_id, (void*)&flat_data_start, &abs_reps, &build_counters };
							const auto p0 = std::chrono::high_resolution_clock::now();
							check_cuda(cuLaunchKernel(raceway_repflag_build_k, screen_kernel_grid_x_ilp(repflag_count, 1u), 1, 1,
								kCudaThreadsPerBlock, 1, 1, 0, 0, ba, nullptr), "launch(raceway_repflag_build)");

							void* pa[] = { &d_flat_flags, &flat_n, &repflag_fp, &repflag_flag, &repflag_logm,
								&assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
								&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
								&assets.alg2_values, &assets.alg5_values, &assets.expansion_values,
								&assets.schedule_data, &args.key_id, (void*)&flat_data_start, &parity_counters };
							check_cuda(cuLaunchKernel(raceway_flat_parity_k, screen_kernel_grid_x_ilp(flat_n, 1u), 1, 1,
								kCudaThreadsPerBlock, 1, 1, 0, 0, pa, nullptr), "launch(raceway_flat_parity)");
							check_cuda(cuCtxSynchronize(), "sync(raceway_flat_parity)");
							const auto p1 = std::chrono::high_resolution_clock::now();
							const double parity_ms = std::chrono::duration<double, std::milli>(p1 - p0).count();

							unsigned long long bc[4] = {}, pc[4] = {};
							check_cuda(cuMemcpyDtoH(bc, build_counters, 4ull * sizeof(unsigned long long)), "DtoH(raceway_build_counters)");
							check_cuda(cuMemcpyDtoH(pc, parity_counters, 4ull * sizeof(unsigned long long)), "DtoH(raceway_parity_counters)");
							const double race_pipeline_ms = ms + race_ms;
							std::cout << "  raceway-vs-flat:"
							          << " flat=" << std::fixed << std::setprecision(3) << flat_ms << " ms ("
							          << std::setprecision(1) << (static_cast<double>(flat_n) / 1e6 / (flat_ms / 1e3)) << " M cand/s)"
							          << "  race_pipeline=" << std::setprecision(3) << race_pipeline_ms << " ms ("
							          << std::setprecision(1) << (static_cast<double>(flat_n) / 1e6 / (race_pipeline_ms / 1e3)) << " M cand/s eff)"
							          << "  parity_build+check=" << std::setprecision(3) << parity_ms << " ms\n";
							std::cout << "  raceway-vs-flat parity:"
							          << " checked=" << pc[0]
							          << " missing_rep=" << pc[1]
							          << " flag_diff=" << pc[2]
							          << " build_inserted=" << bc[0]
							          << " build_overflow=" << bc[1]
							          << (pc[1] == 0ull && pc[2] == 0ull && bc[1] == 0ull ? " [MATCH]" : " [DIFFER]")
							          << "\n";

							cuMemFree(d_ostream);
							cuMemFree(d_flat_flags);
							cuMemFree(repflag_fp);
							cuMemFree(repflag_flag);
							cuMemFree(build_counters);
							cuMemFree(parity_counters);
						}

						cuMemFree(work_counter);
						cuMemFree(survivor_count);
						cuMemFree(race_stats);
						if (race_survivor_offsets) cuMemFree(race_survivor_offsets);
						if (race_survivor_flags) cuMemFree(race_survivor_flags);
						if (race_drop_map) cuMemFree(race_drop_map);
						if (d_cap_tables) cuMemFree(d_cap_tables);
						if (d_cap_bits) cuMemFree(d_cap_bits);
						if (d_cap_ways) cuMemFree(d_cap_ways);
						for (CUdeviceptr p : cap_tables_h) if (p) cuMemFree(p);
					}

					if (args.map1_frontier_deep)
					{
						if (unique > rep_cap)
							throw std::runtime_error("MAP1 representative list exceeded --map1-frontier-rep-cap");
						const auto rs0 = std::chrono::high_resolution_clock::now();
						std::vector<uint32_t> h_rep(unique);
						check_cuda(cuMemcpyDtoH(h_rep.data(), d_rep, static_cast<size_t>(unique) * sizeof(uint32_t)), "DtoH(frontier_reps)");
						std::sort(h_rep.begin(), h_rep.end());
						check_cuda(cuMemcpyHtoD(d_rep, h_rep.data(), static_cast<size_t>(unique) * sizeof(uint32_t)), "HtoD(frontier_reps_sorted)");
						const auto rs1 = std::chrono::high_resolution_clock::now();
						const double rep_sort_ms = std::chrono::duration<double, std::milli>(rs1 - rs0).count();
						std::cout << "  rep-order: sorted by data for deep compaction (" << std::fixed << std::setprecision(3)
						          << rep_sort_ms << " ms host round-trip)\n";
						cuMemFree(table_fp); table_fp = 0;

						const std::vector<uint8_t> sched_blob = build_schedule_blob(args.key_id, args.map_list);
						OffsetStreamBlob osb = build_offset_stream_blob(sched_blob);
						CUdeviceptr d_ostream = 0;
						check_cuda(cuMemAlloc(&d_ostream, osb.data.size()), "cuMemAlloc(frontier_deep_ostream)");
						check_cuda(cuMemcpyHtoD(d_ostream, osb.data.data(), osb.data.size()), "HtoD(frontier_deep_ostream)");
						CUdeviceptr off_regular = d_ostream;
						CUdeviceptr off_alg0 = off_regular + osb.stream_bytes;
						CUdeviceptr off_alg6 = off_alg0 + osb.stream_bytes;
						CUdeviceptr off_alg2 = off_alg6 + osb.stream_bytes;
						CUdeviceptr off_alg5 = off_alg2 + osb.carry_bytes;

						CUdeviceptr state = 0, alive = 0, live_a = 0, live_b = 0, counter = 0, flag = 0;
						check_cuda(cuMemAlloc(&state, static_cast<size_t>(unique) * 32u * sizeof(uint32_t)), "cuMemAlloc(frontier_state)");
						check_cuda(cuMemAlloc(&alive, unique), "cuMemAlloc(frontier_alive)");
						check_cuda(cuMemAlloc(&live_a, static_cast<size_t>(unique) * sizeof(uint32_t)), "cuMemAlloc(frontier_live_a)");
						check_cuda(cuMemAlloc(&live_b, static_cast<size_t>(unique) * sizeof(uint32_t)), "cuMemAlloc(frontier_live_b)");
						check_cuda(cuMemAlloc(&counter, sizeof(uint32_t)), "cuMemAlloc(frontier_counter)");
						check_cuda(cuMemAlloc(&flag, unique), "cuMemAlloc(frontier_flag)");

						constexpr uint32_t F_WARPS = 8u;
						constexpr uint32_t F_ILP = 10u;
						constexpr uint32_t F_W = F_WARPS * F_ILP;
						constexpr uint32_t F_BLOCK = F_WARPS * 32u;
						std::vector<uint32_t> deep_ks;
						if (args.map1_frontier_deep_k_sweep)
						{
							deep_ks = args.map1_frontier_deep_k_list.empty()
								? std::vector<uint32_t>{2u,3u,4u,5u,6u,7u,8u,10u,13u,26u}
								: parse_u32_list(args.map1_frontier_deep_k_list, "--map1-frontier-deep-k-list");
						}
						else
						{
							deep_ks.push_back(args.map1_frontier_deep_k);
						}

						for (uint32_t post_k : deep_ks)
						{
							std::vector<uint32_t> cuts = make_post_map1_cuts(post_k);
							std::vector<uint32_t> fm(cuts.size(), 0u);
							uint32_t M = unique;
							CUdeviceptr cur_live = 0;
							CUdeviceptr next_live = live_a;
							CUdeviceptr no_mult = 0;
							int first = 1;
							fm[0] = M;
							const auto d0 = std::chrono::high_resolution_clock::now();
							for (uint32_t sp = 0; sp + 1 < cuts.size(); ++sp)
							{
								uint32_t m0 = cuts[sp], m1 = cuts[sp + 1u];
								const bool last = (sp + 2u == cuts.size());
								const int mode = (sp == 0u) ? 0 : (last ? 2 : 1);
									void* sa[] = { &cur_live, &M, &d_rep, &state, &alive, &m0, &m1, &first,
										&off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
										&assets.expansion_values, &assets.schedule_data, &args.key_id,
										&assets.carnival_data, &flag, (void*)&mode, &no_mult };
								check_cuda(cuLaunchKernel(frontier_span_k, (M + F_W - 1u) / F_W, 1, 1,
									F_BLOCK, 1, 1, 0, 0, sa, nullptr), "launch(frontier_span_local)");
								if (mode == 1)
								{
									check_cuda(cuMemsetD32(counter, 0u, 1u), "memset(frontier_counter)");
									int compact_first = (cur_live == 0) ? 1 : 0;
									void* ca[] = { &alive, &cur_live, &M, &next_live, &counter, &compact_first };
									check_cuda(cuLaunchKernel(frontier_compact_k, (M + 255u) / 256u, 1, 1,
										256, 1, 1, 0, 0, ca, nullptr), "launch(frontier_compact)");
									check_cuda(cuCtxSynchronize(), "sync(frontier_compact)");
									check_cuda(cuMemcpyDtoH(&M, counter, sizeof(uint32_t)), "DtoH(frontier_counter)");
									cur_live = next_live;
									next_live = (next_live == live_a) ? live_b : live_a;
								}
								first = 0;
								fm[sp + 1u] = M;
							}
							check_cuda(cuCtxSynchronize(), "sync(frontier_deep)");
							const auto d1 = std::chrono::high_resolution_clock::now();
							const double deep_ms = std::chrono::duration<double, std::milli>(d1 - d0).count();

							std::cout << "  deep-compaction postK=" << post_k
							          << ": final_frontier=" << M
							          << "  R_total=" << (static_cast<double>(total) / static_cast<double>(M ? M : 1u))
							          << "  R_after_MAP1=" << (static_cast<double>(unique) / static_cast<double>(M ? M : 1u))
							          << "  time=" << std::fixed << std::setprecision(3) << deep_ms << " ms\n";
							std::cout << "  deep frontier postK=" << post_k << ":";
							for (uint32_t i = 0; i < fm.size(); ++i)
								std::cout << " " << (i ? ("m" + std::to_string(cuts[i]) + "=") : "F1=") << fm[i];
							std::cout << "\n";
						}

						cuMemFree(state); cuMemFree(alive); cuMemFree(live_a); cuMemFree(live_b);
						cuMemFree(counter); cuMemFree(flag); cuMemFree(d_ostream);
					}

					if (args.map1_frontier_consume)
					{
					if (unique > rep_cap)
						throw std::runtime_error("MAP1 representative list exceeded --map1-frontier-rep-cap");

					CUdeviceptr d_flag_canon = 0, d_flag_off = 0;
					if (!args.map1_frontier_offset_only)
						check_cuda(cuMemAlloc(&d_flag_canon, unique), "cuMemAlloc(map1_flag_canon)");
					check_cuda(cuMemAlloc(&d_flag_off, unique), "cuMemAlloc(map1_flag_off)");

					const std::vector<uint8_t> sched_blob = build_schedule_blob(args.key_id, args.map_list);
					OffsetStreamBlob osb = build_offset_stream_blob(sched_blob);
					CUdeviceptr d_ostream = 0;
					check_cuda(cuMemAlloc(&d_ostream, osb.data.size()), "cuMemAlloc(map1_ostream)");
					check_cuda(cuMemcpyHtoD(d_ostream, osb.data.data(), osb.data.size()), "HtoD(map1_ostream)");
					CUdeviceptr off_regular = d_ostream;
					CUdeviceptr off_alg0 = off_regular + osb.stream_bytes;
					CUdeviceptr off_alg6 = off_alg0 + osb.stream_bytes;
					CUdeviceptr off_alg2 = off_alg6 + osb.stream_bytes;
					CUdeviceptr off_alg5 = off_alg2 + osb.carry_bytes;

					uint32_t nrep = unique;
					uint32_t data0 = 0u; // d_rep stores absolute 32-bit data values.
					void* oa[] = { &d_rep, &nrep, &off_regular, &off_alg0, &off_alg6, &off_alg2, &off_alg5,
						&assets.expansion_values, &assets.schedule_data, &assets.carnival_data, &args.key_id, &data0, &d_flag_off };

					double canon_ms = 0.0;
					if (!args.map1_frontier_offset_only)
					{
						void* ca[] = { &d_rep, &nrep, &assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
							&assets.rng_seed_forward_1, &assets.rng_seed_forward_128, &assets.alg2_values, &assets.alg5_values,
							&assets.expansion_values, &assets.schedule_data, &assets.carnival_data, &args.key_id, &data0, &d_flag_canon };
						const auto c0 = std::chrono::high_resolution_clock::now();
						check_cuda(cuLaunchKernel(ds_canon_k, screen_kernel_grid_x_ilp(nrep, 6u), 1, 1,
							kCudaThreadsPerBlock, 1, 1, 0, 0, ca, nullptr), "launch(map1_consume_canon)");
						check_cuda(cuCtxSynchronize(), "sync(map1_consume_canon)");
						const auto c1 = std::chrono::high_resolution_clock::now();
						canon_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
					}
					const auto o0 = std::chrono::high_resolution_clock::now();
					check_cuda(cuLaunchKernel(ds_off_k, screen_kernel_grid_x_ilp(nrep, 6u), 1, 1,
						kCudaThreadsPerBlock, 1, 1, 0, 0, oa, nullptr), "launch(map1_consume_offset)");
					check_cuda(cuCtxSynchronize(), "sync(map1_consume_offset)");
					const auto o1 = std::chrono::high_resolution_clock::now();

					const double off_ms = std::chrono::duration<double, std::milli>(o1 - o0).count();
					std::cout << "  consumer reps=" << unique;
					if (!args.map1_frontier_offset_only)
						std::cout << "  canonical=" << std::fixed << std::setprecision(3) << canon_ms << " ms ("
						          << std::setprecision(1) << (static_cast<double>(unique) / 1e6 / (canon_ms / 1e3)) << " M rep/s)";
					std::cout << "  offset=" << std::fixed << std::setprecision(3) << off_ms << " ms ("
					          << std::setprecision(1) << (static_cast<double>(unique) / 1e6 / (off_ms / 1e3)) << " M rep/s)\n";

					if (args.map1_frontier_offset_only)
					{
						std::cout << "  consumer parity: skipped (--map1-frontier-offset-only)\n";
					}
					else if (unique <= 268435456u)
					{
						std::vector<uint8_t> hc(unique), ho(unique);
						check_cuda(cuMemcpyDtoH(hc.data(), d_flag_canon, unique), "DtoH(map1_flag_canon)");
						check_cuda(cuMemcpyDtoH(ho.data(), d_flag_off, unique), "DtoH(map1_flag_off)");
						uint64_t diff = 0, sum_c = 0, sum_o = 0;
						for (uint32_t i = 0; i < unique; ++i)
						{
							sum_c += hc[i];
							sum_o += ho[i];
							if (hc[i] != ho[i]) ++diff;
						}
						std::cout << "  consumer parity: offset-vs-canonical diff=" << diff
						          << (diff == 0 ? " [IDENTICAL]" : " [DIFFER]")
						          << "  flag_sum(canon)=" << sum_c << "  flag_sum(offset)=" << sum_o << "\n";
					}
					else
					{
						std::cout << "  consumer parity: skipped host flag compare for large rep count\n";
					}

					if (d_flag_canon) cuMemFree(d_flag_canon);
					cuMemFree(d_flag_off); cuMemFree(d_ostream);
				}

					if (table_fp) cuMemFree(table_fp);
					if (table_owner) cuMemFree(table_owner);
					cuMemFree(d_unique); cuMemFree(d_overflow);
					if (d_rep) cuMemFree(d_rep);
					if (d_rep_mult) cuMemFree(d_rep_mult);
					if (partition_labels) cuMemFree(partition_labels);
					if (stream_unique_flags) cuMemFree(stream_unique_flags);
					if (stream_rep) cuMemFree(stream_rep);
					if (stream_rep_idx) cuMemFree(stream_rep_idx);
					if (stream_mult) cuMemFree(stream_mult);
					if (stream_state) cuMemFree(stream_state);
						if (stream_alive) cuMemFree(stream_alive);
						if (stream_traj_sketch) cuMemFree(stream_traj_sketch);
						if (stream_live_a) cuMemFree(stream_live_a);
						if (stream_live_b) cuMemFree(stream_live_b);
						if (stream_counter) cuMemFree(stream_counter);
						if (stream_flag) cuMemFree(stream_flag);
						if (stream_ostream) cuMemFree(stream_ostream);
						if (frontier_route_stats) cuMemFree(frontier_route_stats);
						if (producer_occ_counter) cuMemFree(producer_occ_counter);
					for (CUdeviceptr fct : frontier_cap_tables) if (fct) cuMemFree(fct);
				return 0;
			}

			// ── Wide-merge fingerprint dump (2026-06-03) ───────────────────────────────
			// Step 1 of the GPU wide first-merge sort-dedup build: launch the fp-dump
		// kernel over a whole-byte window, copy the per-candidate 64-bit fingerprints
		// back, and count unique on the host to confirm real-key collapse R matches
		// the CPU's parity-exact figure (R≈5.09 @ W65536, R≈22.6 @ W16M). No cub yet —
		// this validates the foundation kernel before the driver↔runtime interop.
		if (args.wide_merge_dump)
		{
			uint32_t count = static_cast<uint32_t>(args.workunit_size);
			uint32_t key = args.key_id;
			uint32_t data_start = static_cast<uint32_t>(args.range_start);

			uint32_t first_maps = args.wide_merge_first_maps;

			// ── PERIODIC MULTI-MERGE (B increment): re-derive collapse at a schedule of
			// merge points, composing multiplicity, then offset-screen the final frontier.
			if (!args.wide_merge_points.empty())
			{
				std::vector<uint32_t> mpts; { std::stringstream ss(args.wide_merge_points); std::string t;
					while (std::getline(ss,t,',')) if(!t.empty()) mpts.push_back((uint32_t)std::stoul(t)); }
				CUfunction fp_idx_k=nullptr, ds_off_k=nullptr, base_k=nullptr;
				check_cuda(cuModuleGetFunction(&fp_idx_k, module, "tm_wide_merge_fp_dump_indexed_cuda"), "getfn(fp_idx)");
				check_cuda(cuModuleGetFunction(&ds_off_k, module, "tm_wide_merge_survivor_screen_offset_ilp6_cuda"), "getfn(ds_off)");
				check_cuda(cuModuleGetFunction(&base_k, module, "tm_wide_merge_survivor_screen_ilp6_cuda"), "getfn(base)");
				// offset streams for the final downstream
				const std::vector<uint8_t> sb = build_schedule_blob(args.key_id, args.map_list);
				OffsetStreamBlob osb = build_offset_stream_blob(sb);
				CUdeviceptr d_os=0; check_cuda(cuMemAlloc(&d_os, osb.data.size()), "alloc(os)");
				check_cuda(cuMemcpyHtoD(d_os, osb.data.data(), osb.data.size()), "h2d(os)");
				CUdeviceptr o_reg=d_os, o_a0=o_reg+osb.stream_bytes, o_a6=o_a0+osb.stream_bytes, o_a2=o_a6+osb.stream_bytes, o_a5=o_a2+osb.carry_bytes;
				// survivors = all data values; mult = 1
				std::vector<uint32_t> surv(count), mult(count, 1u);
				for (uint32_t i=0;i<count;++i) surv[i]=i;
				CUdeviceptr d_surv=0, d_fp=0;
				check_cuda(cuMemAlloc(&d_surv, (size_t)count*4), "alloc(surv)");
				check_cuda(cuMemAlloc(&d_fp, (size_t)count*8), "alloc(mmfp)");
				uint32_t cur_n = count;
				double fp_ms_total=0, coll_ms_total=0;
				std::cout << "  [multi-merge points=" << args.wide_merge_points << "]\n";
				for (size_t mi=0; mi<mpts.size(); ++mi) {
					uint32_t mp = mpts[mi];
					check_cuda(cuMemcpyHtoD(d_surv, surv.data(), (size_t)cur_n*4), "h2d(surv)");
					void* fa[] = { &d_fp, &d_surv, &assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
						&assets.rng_seed_forward_1, &assets.rng_seed_forward_128, &assets.alg2_values, &assets.alg5_values,
						&assets.expansion_values, &assets.schedule_data, &key, &data_start, &cur_n, &mp };
					const auto f0=std::chrono::high_resolution_clock::now();
					check_cuda(cuLaunchKernel(fp_idx_k, screen_kernel_grid_x_ilp(cur_n,1u),1,1, kCudaThreadsPerBlock,1,1, 0,0, fa,nullptr), "launch(fp_idx)");
					check_cuda(cuCtxSynchronize(), "sync(fp_idx)");
					const auto f1=std::chrono::high_resolution_clock::now();
					fp_ms_total += std::chrono::duration<double,std::milli>(f1-f0).count();
					std::vector<uint64_t> mfps(cur_n);
					check_cuda(cuMemcpyDtoH(mfps.data(), d_fp, (size_t)cur_n*8), "d2h(mmfp)");
					std::vector<uint32_t> so(cur_n), mo(cur_n); uint32_t U=0;
					const auto c0=std::chrono::high_resolution_clock::now();
					const int rc = wms_sort_dedup_mult(mfps.data(), surv.data(), mult.data(), cur_n, so.data(), mo.data(), &U);
					const auto c1=std::chrono::high_resolution_clock::now();
					coll_ms_total += std::chrono::duration<double,std::milli>(c1-c0).count();
					if (rc!=0) { std::cout << "    multi-merge collapse FAILED\n"; break; }
					surv.assign(so.begin(), so.begin()+U); mult.assign(mo.begin(), mo.begin()+U);
					std::cout << "    merge@" << mp << ": " << cur_n << " -> " << U << "  (R_cum=" << ((double)count/U) << ")\n";
					cur_n = U;
				}
				// final offset-screen on the frontier survivors
				check_cuda(cuMemcpyHtoD(d_surv, surv.data(), (size_t)cur_n*4), "h2d(survF)");
				CUdeviceptr d_fsurv=0; check_cuda(cuMemAlloc(&d_fsurv, cur_n), "alloc(fsurv)");
				void* dsa[] = { &d_surv, &cur_n, &o_reg, &o_a0, &o_a6, &o_a2, &o_a5,
					&assets.expansion_values, &assets.schedule_data, &assets.carnival_data, &key, &data_start, &d_fsurv };
				const auto d0=std::chrono::high_resolution_clock::now();
				check_cuda(cuLaunchKernel(ds_off_k, screen_kernel_grid_x_ilp(cur_n,6u),1,1, kCudaThreadsPerBlock,1,1, 0,0, dsa,nullptr), "launch(dsF)");
				check_cuda(cuCtxSynchronize(), "sync(dsF)");
				const auto d1=std::chrono::high_resolution_clock::now();
				const double ds_ms=std::chrono::duration<double,std::milli>(d1-d0).count();
				std::vector<uint8_t> fsurv(cur_n); check_cuda(cuMemcpyDtoH(fsurv.data(), d_fsurv, cur_n), "d2h(fsurv)");
				// canonical baseline over all N (parity + speedup ref), min-of-5
				std::vector<uint32_t> iota(count); for(uint32_t i=0;i<count;++i) iota[i]=i;
				CUdeviceptr d_iota=0,d_fbase=0; check_cuda(cuMemAlloc(&d_iota,(size_t)count*4),"alloc(iota)"); check_cuda(cuMemAlloc(&d_fbase,count),"alloc(fbase)");
				check_cuda(cuMemcpyHtoD(d_iota, iota.data(), (size_t)count*4), "h2d(iota)");
				uint32_t cn=count; void* ba[] = { &d_iota, &cn, &assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
					&assets.rng_seed_forward_1, &assets.rng_seed_forward_128, &assets.alg2_values, &assets.alg5_values,
					&assets.expansion_values, &assets.schedule_data, &assets.carnival_data, &key, &data_start, &d_fbase };
				double base_ms=1e9; for(int r=0;r<5;++r){ const auto b0=std::chrono::high_resolution_clock::now();
					check_cuda(cuLaunchKernel(base_k, screen_kernel_grid_x_ilp(count,6u),1,1, kCudaThreadsPerBlock,1,1, 0,0, ba,nullptr), "launch(base)");
					check_cuda(cuCtxSynchronize(), "sync(base)"); const auto b1=std::chrono::high_resolution_clock::now();
					base_ms=std::min(base_ms,std::chrono::duration<double,std::milli>(b1-b0).count()); }
				std::vector<uint8_t> fbase(count); check_cuda(cuMemcpyDtoH(fbase.data(), d_fbase, count), "d2h(fbase)");
				uint64_t sb_=0, sw_=0; for(uint32_t i=0;i<count;++i) sb_+=fbase[i];
				for(uint32_t u=0;u<cur_n;++u) sw_+=(uint64_t)fsurv[u]*mult[u];
				const double e2e=fp_ms_total+coll_ms_total+ds_ms;
				std::cout << "    final frontier=" << cur_n << "  fp=" << fp_ms_total << " + collapse=" << coll_ms_total
				          << " + downstream=" << ds_ms << " = " << e2e << " ms\n";
				std::cout << "    baseline(canon ILP6 all-N)=" << base_ms << " ms  speedup=" << (base_ms/e2e) << "x ("
				          << (static_cast<double>(count)/1e6/(e2e/1e3)) << " M/s eff)\n";
				std::cout << "    parity: base=" << sb_ << " surv_w=" << sw_ << (sb_==sw_? "  [MATCH]":"  [MISMATCH!]") << "\n";
				cuMemFree(d_os);cuMemFree(d_surv);cuMemFree(d_fp);cuMemFree(d_fsurv);cuMemFree(d_iota);cuMemFree(d_fbase);
				return 0;
			}
			CUfunction fp_dump_kernel = nullptr;
			check_cuda(cuModuleGetFunction(&fp_dump_kernel, module, "tm_wide_merge_fp_dump_n_cuda"),
				"cuModuleGetFunction(wide_merge_fp_dump)");

			// -- MAP1 data-variance diagnostic (small sample): which of 32 state words
			// are data-influenced after first_maps, and is the alg-dispatch sequence
			// data-invariant? Bounds the MAP1-emitter opts (hoist prefix / incr fp).
			{
				const uint32_t ns = 4096u;
				CUfunction diag_k = nullptr;
				check_cuda(cuModuleGetFunction(&diag_k, module, "tm_wide_merge_state_alg_dump_cuda"), "cuModuleGetFunction(state_alg_dump)");
				CUdeviceptr d_state = 0, d_alg = 0;
				check_cuda(cuMemAlloc(&d_state, (size_t)ns * 32u * sizeof(uint32_t)), "cuMemAlloc(diag_state)");
				check_cuda(cuMemAlloc(&d_alg, (size_t)ns * 16u), "cuMemAlloc(diag_alg)");
				void* da[] = { &d_state, &d_alg, &assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
					&assets.rng_seed_forward_1, &assets.rng_seed_forward_128, &assets.alg2_values, &assets.alg5_values,
					&assets.expansion_values, &assets.schedule_data, &key, &data_start, (void*)&ns, &first_maps };
				check_cuda(cuLaunchKernel(diag_k, screen_kernel_grid_x_ilp(ns, 1u),1,1, kCudaThreadsPerBlock,1,1, 0,0, da,nullptr), "launch(diag)");
				check_cuda(cuCtxSynchronize(), "sync(diag)");
				std::vector<uint32_t> hstate((size_t)ns*32u); std::vector<uint8_t> halg((size_t)ns*16u);
				check_cuda(cuMemcpyDtoH(hstate.data(), d_state, (size_t)ns*32u*sizeof(uint32_t)), "DtoH(diag_state)");
				check_cuda(cuMemcpyDtoH(halg.data(), d_alg, (size_t)ns*16u), "DtoH(diag_alg)");
				int vw = 0; std::string wmask;
				for (uint32_t w = 0; w < 32u; ++w) { uint32_t v0 = hstate[w]; bool v = false;
					for (uint32_t c = 1; c < ns; ++c) if (hstate[(size_t)c*32u+w] != v0) { v = true; break; }
					vw += v; wmask += (v ? '1' : '.'); }
				int vs = 0; std::string amask;
				for (uint32_t s = 0; s < 16u; ++s) { uint8_t a0 = halg[s]; bool v = false;
					for (uint32_t c = 1; c < ns; ++c) if (halg[(size_t)c*16u+s] != a0) { v = true; break; }
					vs += v; amask += (v ? '1' : '.'); }
				std::cout << "  [MAP1 diag, first_maps=" << first_maps << ", " << ns << " data values]\n";
				std::cout << "    state words data-varying: " << vw << "/32  mask=" << wmask << "\n";
				std::cout << "    alg-dispatch data-varying: " << vs << "/16  mask=" << amask << (vs==0 ? "  [alg seq DATA-INVARIANT]" : "  [alg seq DATA-DEPENDENT]") << "\n";
				cuMemFree(d_state); cuMemFree(d_alg);
			}

			CUdeviceptr d_fp = 0;
			check_cuda(cuMemAlloc(&d_fp, static_cast<size_t>(count) * sizeof(uint64_t)),
				"cuMemAlloc(wide_merge_fp)");

			void* kargs[] = {
				&d_fp, &assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
				&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
				&assets.alg2_values, &assets.alg5_values, &assets.expansion_values,
				&assets.schedule_data, &key, &data_start, &count, &first_maps };

			const uint32_t grid = screen_kernel_grid_x_ilp(count, 1u);
			// warmup
			check_cuda(cuLaunchKernel(fp_dump_kernel, grid, 1, 1, kCudaThreadsPerBlock, 1, 1,
				0, 0, kargs, nullptr), "launch(wide_merge_fp warmup)");
			check_cuda(cuCtxSynchronize(), "sync(wide_merge_fp warmup)");

			// Min-of-N timing. Short bursty kernels leave the GPU in P1 (~75% clocks);
			// looping ~30 launches ramps it to P0 and the min is the boosted-clock,
			// steady-state cost (comparable to the doc's full-sweep references).
			double kernel_ms = 1e9;
			for (int rep = 0; rep < 30; ++rep) {
				const auto t0 = std::chrono::high_resolution_clock::now();
				check_cuda(cuLaunchKernel(fp_dump_kernel, grid, 1, 1, kCudaThreadsPerBlock, 1, 1,
					0, 0, kargs, nullptr), "launch(wide_merge_fp)");
				check_cuda(cuCtxSynchronize(), "sync(wide_merge_fp)");
				const auto t1 = std::chrono::high_resolution_clock::now();
				kernel_ms = std::min(kernel_ms, std::chrono::duration<double, std::milli>(t1 - t0).count());
			}

			std::vector<uint64_t> fps(count);
			check_cuda(cuMemcpyDtoH(fps.data(), d_fp, static_cast<size_t>(count) * sizeof(uint64_t)),
				"cuMemcpyDtoH(wide_merge_fp)");
			// NB: keep d_fp alive — the global-hash collapse below reads it on-device.

			// Ground-truth unique count via host sort (validates the cub collapse).
			std::vector<uint64_t> fps_sorted(fps);
			std::sort(fps_sorted.begin(), fps_sorted.end());
			const size_t unique = static_cast<size_t>(
				std::unique(fps_sorted.begin(), fps_sorted.end()) - fps_sorted.begin());
			const double R = static_cast<double>(count) / static_cast<double>(unique ? unique : 1);

			// Hash-quality check (CPU 45): strong-hash unique vs warp_hash unique. If strong>weak,
			// warp_hash false-merges distinct states (= dropped state = coverage loss for the search).
			{
				CUfunction fp_strong_k = nullptr;
				check_cuda(cuModuleGetFunction(&fp_strong_k, module, "tm_wide_merge_fp_dump_strong_cuda"), "cuModuleGetFunction(fp_strong)");
				CUdeviceptr d_fps2 = 0;
				check_cuda(cuMemAlloc(&d_fps2, static_cast<size_t>(count) * sizeof(uint64_t)), "cuMemAlloc(fp_strong)");
				void* sa[] = { &d_fps2, &assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
					&assets.rng_seed_forward_1, &assets.rng_seed_forward_128, &assets.alg2_values, &assets.alg5_values,
					&assets.expansion_values, &assets.schedule_data, &key, &data_start, &count, &first_maps };
				check_cuda(cuLaunchKernel(fp_strong_k, screen_kernel_grid_x_ilp(count, 1u),1,1, kCudaThreadsPerBlock,1,1, 0,0, sa,nullptr), "launch(fp_strong)");
				check_cuda(cuCtxSynchronize(), "sync(fp_strong)");
				std::vector<uint64_t> fps2(count);
				check_cuda(cuMemcpyDtoH(fps2.data(), d_fps2, static_cast<size_t>(count) * sizeof(uint64_t)), "DtoH(fp_strong)");
				cuMemFree(d_fps2);
				std::sort(fps2.begin(), fps2.end());
				const size_t unique_strong = static_cast<size_t>(std::unique(fps2.begin(), fps2.end()) - fps2.begin());
				const long drop = (long)unique_strong - (long)unique;
				std::cout << "  [hash-quality] warp_hash unique=" << unique << "  strong unique=" << unique_strong
				          << "  warp_hash dropped=" << drop << (drop == 0 ? "  [warp_hash EXACT]" : "  [warp_hash FALSE-MERGES]") << "\n";
			}

			// cub sort-collapse: survivor indices (run reps) + run lengths (multiplicity).
			std::vector<uint32_t> survivor_idx(count), run_len(count);
			uint32_t cub_unique = 0; float sort_ms = 0.0f;
			const int wms_rc = wms_sort_dedup(fps.data(), count,
				survivor_idx.data(), run_len.data(), &cub_unique, &sort_ms);

			std::cout << "[wide-merge-dump] device=" << device_name
			          << " key=0x" << std::hex << key << std::dec
			          << " first_maps=" << args.wide_merge_first_maps
			          << " window=" << count
			          << " data_start=" << data_start << "\n";
			std::cout << "  unique(host)=" << unique << "  R=" << R
			          << "  (map-" << args.wide_merge_first_maps << " merge point)\n";
			std::cout << "  fp-dump kernel: " << kernel_ms << " ms  ("
			          << (static_cast<double>(count) / 1e6 / (kernel_ms / 1e3)) << " M cand/s)\n";
			if (wms_rc == 0)
			{
				uint64_t mult_sum = 0;
				for (uint32_t r = 0; r < cub_unique; ++r) mult_sum += run_len[r];
				std::cout << "  cub sort-collapse: unique=" << cub_unique
				          << (cub_unique == unique ? "  [MATCH host]" : "  [MISMATCH!]")
				          << "  sort+rle: " << sort_ms << " ms  ("
				          << (static_cast<double>(count) / 1e6 / (sort_ms / 1e3)) << " M cand/s)\n";
				std::cout << "  multiplicity-sum=" << mult_sum
				          << (mult_sum == count ? "  [== window, conservative]" : "  [LEAK!]") << "\n";
			}
			else
			{
				std::cout << "  cub sort-collapse: FAILED (rc=" << wms_rc << ")\n";
			}

			// Global-VRAM-hash collapse (measured baseline vs sort). Pure fatbin
			// kernels on the driver-allocated fp buffer — no cub/runtime interop.
			{
				uint32_t logm = 1u;
				while ((1u << logm) < (count + (count >> 1))) ++logm; // ~1.5x load headroom
				if (logm > 31u) logm = 31u;
				const uint32_t slots = 1u << logm;

				CUfunction zero_k = nullptr, collapse_k = nullptr;
				check_cuda(cuModuleGetFunction(&zero_k, module, "tm_wide_merge_hash_zero_cuda"),
					"cuModuleGetFunction(hash_zero)");
				check_cuda(cuModuleGetFunction(&collapse_k, module, "tm_wide_merge_hash_collapse_cuda"),
					"cuModuleGetFunction(hash_collapse)");

				CUdeviceptr d_tfp = 0, d_trep = 0, d_sflag = 0, d_rep = 0, d_uniq = 0;
				check_cuda(cuMemAlloc(&d_tfp, (size_t)slots * sizeof(uint64_t)), "cuMemAlloc(table_fp)");
				check_cuda(cuMemAlloc(&d_trep, (size_t)slots * sizeof(uint32_t)), "cuMemAlloc(table_rep)");
				check_cuda(cuMemAlloc(&d_sflag, count), "cuMemAlloc(survivor_flag)");
				check_cuda(cuMemAlloc(&d_rep, (size_t)count * sizeof(uint32_t)), "cuMemAlloc(rep_out)");
				check_cuda(cuMemAlloc(&d_uniq, sizeof(uint32_t)), "cuMemAlloc(unique_counter)");

				void* zargs[] = { &d_tfp, &d_trep, (void*)&slots };
				void* cargs[] = { &d_fp, &count, &d_tfp, &d_trep, &logm, &d_sflag, &d_rep, &d_uniq };
				const uint32_t zgrid = (slots + 255u) / 256u;
				const uint32_t cgrid = (count + 255u) / 256u;

				// warmup (zero + collapse)
				check_cuda(cuMemsetD32(d_uniq, 0u, 1), "memset(uniq)");
				check_cuda(cuLaunchKernel(zero_k, zgrid, 1, 1, 256, 1, 1, 0, 0, zargs, nullptr), "launch(hash_zero warmup)");
				check_cuda(cuLaunchKernel(collapse_k, cgrid, 1, 1, 256, 1, 1, 0, 0, cargs, nullptr), "launch(hash_collapse warmup)");
				check_cuda(cuCtxSynchronize(), "sync(hash warmup)");

				check_cuda(cuMemsetD32(d_uniq, 0u, 1), "memset(uniq)");
				check_cuda(cuLaunchKernel(zero_k, zgrid, 1, 1, 256, 1, 1, 0, 0, zargs, nullptr), "launch(hash_zero)");
				const auto h0 = std::chrono::high_resolution_clock::now();
				check_cuda(cuLaunchKernel(collapse_k, cgrid, 1, 1, 256, 1, 1, 0, 0, cargs, nullptr), "launch(hash_collapse)");
				check_cuda(cuCtxSynchronize(), "sync(hash_collapse)");
				const auto h1 = std::chrono::high_resolution_clock::now();
				const double hash_ms = std::chrono::duration<double, std::milli>(h1 - h0).count();

				uint32_t hash_unique = 0;
				check_cuda(cuMemcpyDtoH(&hash_unique, d_uniq, sizeof(uint32_t)), "copy(unique_counter)");
				std::cout << "  global-hash collapse: unique=" << hash_unique
				          << (hash_unique == unique ? "  [MATCH host]" : "  [MISMATCH!]")
				          << "  collapse: " << hash_ms << " ms  ("
				          << (static_cast<double>(count) / 1e6 / (hash_ms / 1e3)) << " M cand/s)"
				          << "  table=" << slots << " slots (" << ((double)slots * 12.0 / (1u<<20)) << " MB)\n";

				cuMemFree(d_tfp); cuMemFree(d_trep); cuMemFree(d_sflag); cuMemFree(d_rep); cuMemFree(d_uniq);
			}

			// ── End-to-end: canonical ILP6 baseline (fair) + production-offset ref ────
			// The wide-merge groups by CANONICAL map-K state, so the downstream MUST
			// screen canonically (the offset path is a different forward representation
			// → invalid to mix). Fair baseline = the SAME canonical ILP6 kernel over all
			// N. The production offset screener is timed separately as an absolute-rate
			// reference only (it is a faster-per-candidate but distinct representation).
			if (wms_rc == 0 && cub_unique > 0)
			{
				CUfunction ds_ilp6_k = nullptr, ds_off_k = nullptr, prod_off_k = nullptr;
				check_cuda(cuModuleGetFunction(&ds_ilp6_k, module,
					"tm_wide_merge_survivor_screen_ilp6_cuda"), "cuModuleGetFunction(survivor_ilp6)");
				check_cuda(cuModuleGetFunction(&ds_off_k, module,
					"tm_wide_merge_survivor_screen_offset_ilp6_cuda"), "cuModuleGetFunction(survivor_offset_ilp6)");
				check_cuda(cuModuleGetFunction(&prod_off_k, module,
					"tm_checksum_screen_offset_store_ilp6_preids_cuda"), "cuModuleGetFunction(ilp6_preids)");

				std::vector<uint32_t> iota(count);
				for (uint32_t i = 0; i < count; ++i) iota[i] = i;
				CUdeviceptr d_surv = 0, d_iota = 0, d_fsurv = 0, d_fsurv_off = 0, d_fbase = 0, d_foff = 0;
				check_cuda(cuMemAlloc(&d_surv, (size_t)cub_unique * sizeof(uint32_t)), "cuMemAlloc(d_surv)");
				check_cuda(cuMemAlloc(&d_iota, (size_t)count * sizeof(uint32_t)), "cuMemAlloc(d_iota)");
				check_cuda(cuMemAlloc(&d_fsurv, cub_unique), "cuMemAlloc(d_fsurv)");
				check_cuda(cuMemAlloc(&d_fsurv_off, cub_unique), "cuMemAlloc(d_fsurv_off)");
				check_cuda(cuMemAlloc(&d_fbase, count), "cuMemAlloc(d_fbase)");
				check_cuda(cuMemAlloc(&d_foff, count), "cuMemAlloc(d_foff)");
				check_cuda(cuMemcpyHtoD(d_surv, survivor_idx.data(), (size_t)cub_unique * sizeof(uint32_t)), "HtoD(surv)");
				check_cuda(cuMemcpyHtoD(d_iota, iota.data(), (size_t)count * sizeof(uint32_t)), "HtoD(iota)");

				// ilp6_preids requires OFFSET-STREAM tables (build_offset_stream_blob
				// precomputes the true per-entry seed walk in [entry][2048][lane] layout),
				// NOT the canonical assets.* tables. Build + upload them so the offset
				// reference is invoked correctly (this is what production does).
				const std::vector<uint8_t> sched_blob = build_schedule_blob(args.key_id, args.map_list);
				OffsetStreamBlob osb = build_offset_stream_blob(sched_blob);
				CUdeviceptr d_ostream = 0;
				check_cuda(cuMemAlloc(&d_ostream, osb.data.size()), "cuMemAlloc(ostream)");
				check_cuda(cuMemcpyHtoD(d_ostream, osb.data.data(), osb.data.size()), "HtoD(ostream)");
				CUdeviceptr off_regular = d_ostream;
				CUdeviceptr off_alg0 = off_regular + osb.stream_bytes;
				CUdeviceptr off_alg6 = off_alg0 + osb.stream_bytes;
				CUdeviceptr off_alg2 = off_alg6 + osb.stream_bytes;
				CUdeviceptr off_alg5 = off_alg2 + osb.carry_bytes;

				auto launch_canon = [&](CUdeviceptr idx, uint32_t n, CUdeviceptr flags) {
					uint32_t nn = n;
					void* a[] = { &idx, &nn, &assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
						&assets.rng_seed_forward_1, &assets.rng_seed_forward_128, &assets.alg2_values, &assets.alg5_values,
						&assets.expansion_values, &assets.schedule_data, &assets.carnival_data, &key, &data_start, &flags };
					check_cuda(cuLaunchKernel(ds_ilp6_k, screen_kernel_grid_x_ilp(n, 6u),1,1, kCudaThreadsPerBlock,1,1, 0,0, a,nullptr), "canon");
				};
				// Min-of-N boosted-clock timing (see fp-dump note). The repeated launches
				// also keep the GPU in P0 across all three measurements.
				auto time_min = [&](const std::function<void()>& launch, int reps) {
					double best = 1e9;
					for (int r = 0; r < reps; ++r) {
						const auto a = std::chrono::high_resolution_clock::now();
						launch(); check_cuda(cuCtxSynchronize(), "sync");
						const auto b = std::chrono::high_resolution_clock::now();
						best = std::min(best, std::chrono::duration<double, std::milli>(b - a).count());
					}
					return best;
				};

				void* off_args[] = { &d_foff, &off_regular, &off_alg0, &off_alg6,
					&off_alg2, &off_alg5, &assets.expansion_values, &assets.schedule_data,
					&assets.carnival_data, &key, &data_start, &count };
				const uint32_t off_grid = screen_kernel_grid_x_ilp(count, 6u);

				// Fair canonical-ILP6 baseline over all N (also ramps clocks first).
				const double base_ms = time_min([&]{ launch_canon(d_iota, count, d_fbase); }, 30);
				// Downstream canonical-ILP6 over U survivors (correctness reference).
				const double ds_ms = time_min([&]{ launch_canon(d_surv, cub_unique, d_fsurv); }, 30);
				// Downstream OFFSET-ILP6 over U survivors (the fast production path).
				uint32_t nsurv = cub_unique;
				void* ds_off_args[] = { &d_surv, &nsurv, &off_regular, &off_alg0, &off_alg6,
					&off_alg2, &off_alg5, &assets.expansion_values, &assets.schedule_data,
					&assets.carnival_data, &key, &data_start, &d_fsurv_off };
				const uint32_t ds_off_grid = screen_kernel_grid_x_ilp(cub_unique, 6u);
				const double ds_off_ms = time_min([&]{
					check_cuda(cuLaunchKernel(ds_off_k, ds_off_grid,1,1, kCudaThreadsPerBlock,1,1, 0,0, ds_off_args,nullptr), "ds_off"); }, 30);
				// Production offset screener over all N — absolute-rate reference only.
				const double off_ms = time_min([&]{
					check_cuda(cuLaunchKernel(prod_off_k, off_grid,1,1, kCudaThreadsPerBlock,1,1, 0,0, off_args,nullptr), "off"); }, 30);

				// parity: Σ flag_base over N == Σ flag_surv*mult over U; and offset downstream == canonical downstream.
				std::vector<uint8_t> fbase(count), fsurv(cub_unique), fsurv_off(cub_unique);
				check_cuda(cuMemcpyDtoH(fbase.data(), d_fbase, count), "DtoH(fbase)");
				check_cuda(cuMemcpyDtoH(fsurv.data(), d_fsurv, cub_unique), "DtoH(fsurv)");
				check_cuda(cuMemcpyDtoH(fsurv_off.data(), d_fsurv_off, cub_unique), "DtoH(fsurv_off)");
					uint64_t sum_base = 0, sum_surv_w = 0, sum_surv_off_w = 0, ds_diff = 0;
					for (uint32_t i = 0; i < count; ++i) sum_base += fbase[i];
					for (uint32_t u = 0; u < cub_unique; ++u) {
						sum_surv_w += (uint64_t)fsurv[u] * run_len[u];
						sum_surv_off_w += (uint64_t)fsurv_off[u] * run_len[u];
						if (fsurv[u] != fsurv_off[u]) ++ds_diff;
					}

				const double e2e_canon = kernel_ms + sort_ms + ds_ms;
				const double e2e_off   = kernel_ms + sort_ms + ds_off_ms;
				std::cout << "  ── end-to-end (single wide merge @ map " << args.wide_merge_first_maps << ") ──\n";
				std::cout << "    canonical ILP6 baseline all-N: " << base_ms << " ms  ("
				          << (static_cast<double>(count) / 1e6 / (base_ms / 1e3)) << " M/s);  prod offset all-N: "
				          << off_ms << " ms (" << (static_cast<double>(count) / 1e6 / (off_ms / 1e3)) << " M/s)\n";
				std::cout << "    downstream: canonical " << ds_ms << " ms  vs  offset " << ds_off_ms << " ms  ("
				          << (ds_ms / (ds_off_ms > 0 ? ds_off_ms : 1)) << "x faster), offset-vs-canon flags differ=" << ds_diff
				          << (ds_diff == 0 ? " [IDENTICAL]" : " [DIFFER]") << "\n";
				std::cout << "    wide-merge CANON: " << e2e_canon << " ms -> " << (base_ms / e2e_canon) << "x vs base, "
				          << (off_ms / e2e_canon) << "x vs prod-offset\n";
				std::cout << "    wide-merge OFFSET: fpdump " << kernel_ms << " + collapse " << sort_ms
				          << " + downstream " << ds_off_ms << " = " << e2e_off << " ms -> "
				          << (off_ms / e2e_off) << "x vs prod-offset  ("
				          << (static_cast<double>(count) / 1e6 / (e2e_off / 1e3)) << " M/s eff)\n";
					std::cout << "    parity (canonical weighted flag sum): base=" << sum_base << " surv_w=" << sum_surv_w
					          << (sum_base == sum_surv_w ? "  [MATCH]" : "  [MISMATCH!]") << "\n";
					std::cout << "    parity (offset weighted flag sum): base=" << sum_base << " surv_off_w=" << sum_surv_off_w
					          << (sum_base == sum_surv_off_w ? "  [MATCH]" : "  [MISMATCH!]") << "\n";

				cuMemFree(d_surv); cuMemFree(d_iota); cuMemFree(d_fsurv); cuMemFree(d_fsurv_off); cuMemFree(d_fbase); cuMemFree(d_foff); cuMemFree(d_ostream);
			}
			cuMemFree(d_fp);
			return 0;
		}

		// map_rng POC: build & upload the per-launch buffer when requested.
		// Regular maprng = 54 KB raw. Pre-extracted maprng = 162 KB (raw+alg0+alg6).
		CUdeviceptr map_rng_buffer = 0;
		std::vector<uint8_t> map_rng_host;
		if (args.maprng || args.screen_preext || args.screen_dedup_maprng || args.screen_dedup_maprng_preext || args.screen_coalesced)
		{
			const std::vector<uint8_t> schedule_blob = build_schedule_blob(args.key_id, args.map_list);
			if (args.screen_coalesced)
				map_rng_host = build_maprng_blob_preext_coalesced(schedule_blob);
			else if (args.screen_preext || args.screen_dedup_maprng_preext)
				map_rng_host = build_maprng_blob_preext(schedule_blob);
			else
				map_rng_host = build_maprng_blob(schedule_blob);
			check_cuda(cuMemAlloc(&map_rng_buffer, map_rng_host.size()), "cuMemAlloc(map_rng_buffer)");
			check_cuda(cuMemcpyHtoD(map_rng_buffer, map_rng_host.data(), map_rng_host.size()), "cuMemcpyHtoD(map_rng_buffer)");
			std::cout << "  maprng_buffer_bytes: " << map_rng_host.size() << "\n";
		}

		CUdeviceptr offset_stream_buffer = 0;
		std::size_t offset_stream_bytes = 0;
		std::size_t offset_carry_bytes = 0;
		if (args.screen_offsets || args.screen_offsets_interleaved || args.screen_offset_ilp_sweep || args.screen_dedup_offsets)
		{
			const std::vector<uint8_t> schedule_blob = build_schedule_blob(args.key_id, args.map_list);
			if (args.screen_offsets_interleaved)
			{
				std::vector<uint8_t> offset_blob = build_offset_stream_blob_interleaved(schedule_blob);
				check_cuda(cuMemAlloc(&offset_stream_buffer, offset_blob.size()), "cuMemAlloc(offset_stream_buffer_interleaved)");
				check_cuda(cuMemcpyHtoD(offset_stream_buffer, offset_blob.data(), offset_blob.size()), "cuMemcpyHtoD(offset_stream_buffer_interleaved)");
				std::cout << "  offset_stream_interleaved_buffer_bytes: " << offset_blob.size() << "\n";
			}
			else
			{
				OffsetStreamBlob offset_blob = build_offset_stream_blob(schedule_blob);
				offset_stream_bytes = offset_blob.stream_bytes;
				offset_carry_bytes = offset_blob.carry_bytes;
				check_cuda(cuMemAlloc(&offset_stream_buffer, offset_blob.data.size()), "cuMemAlloc(offset_stream_buffer)");
				check_cuda(cuMemcpyHtoD(offset_stream_buffer, offset_blob.data.data(), offset_blob.data.size()), "cuMemcpyHtoD(offset_stream_buffer)");
				std::cout << "  offset_stream_buffer_bytes: " << offset_blob.data.size() << "\n";
			}
		}
		CUdeviceptr offset_regular = offset_stream_buffer;
		CUdeviceptr offset_alg0 = offset_regular + offset_stream_bytes;
		CUdeviceptr offset_alg6 = offset_alg0 + offset_stream_bytes;
		CUdeviceptr offset_alg2 = offset_alg6 + offset_stream_bytes;
		CUdeviceptr offset_alg5 = offset_alg2 + offset_carry_bytes;

		CUdeviceptr result_buffer = 0;
		check_cuda(cuMemAlloc(&result_buffer, args.batch_size), "cuMemAlloc(result_buffer)");
		CUdeviceptr survivor_data_buffer = 0;
		check_cuda(cuMemAlloc(&survivor_data_buffer, args.batch_size * sizeof(uint32_t)), "cuMemAlloc(survivor_data_buffer)");
		CUdeviceptr survivor_flag_buffer = 0;
		check_cuda(cuMemAlloc(&survivor_flag_buffer, args.batch_size), "cuMemAlloc(survivor_flag_buffer)");
		CUdeviceptr materialized_state_buffer = 0;
		check_cuda(cuMemAlloc(&materialized_state_buffer, args.batch_size * 128ull), "cuMemAlloc(materialized_state_buffer)");
		// HLL: 4096 uint32_t registers = 16 KB, zeroed before the sweep
		static const uint32_t kHllM = 4096u;
		CUdeviceptr hll_buffer = 0;
		check_cuda(cuMemAlloc(&hll_buffer, kHllM * sizeof(uint32_t)), "cuMemAlloc(hll_buffer)");
		check_cuda(cuMemsetD32(hll_buffer, 0u, kHllM), "cuMemsetD32(hll_buffer)");

		CUstream stream = nullptr;
		check_cuda(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate");

		CUevent kernel_start = nullptr;
		CUevent kernel_end = nullptr;
		check_cuda(cuEventCreate(&kernel_start, CU_EVENT_DEFAULT), "cuEventCreate(start)");
		check_cuda(cuEventCreate(&kernel_end, CU_EVENT_DEFAULT), "cuEventCreate(end)");

		std::vector<uint8_t> host_results(args.batch_size, 0);
		std::vector<SurvivorRecord> survivors;
		const auto setup_end = std::chrono::high_resolution_clock::now();

		// ---- SCREEN-only A/B: universal-table screen vs per-key offset streams ----
		// Keeps coalesced uint32 loads, but CPU-precomputes each schedule entry's
		// 2048 RNG offsets so the kernel can update a small offset instead of
		// loading seed_forward_1/128[seed] after every algorithm.
		if ((args.screen_offsets || args.screen_offsets_interleaved) && args.parity_count > 0)
		{
			const uint32_t N = args.parity_count;
			CUdeviceptr buf_ref = 0, buf_offset = 0;
			check_cuda(cuMemAlloc(&buf_ref, N), "cuMemAlloc(screen_ref)");
			check_cuda(cuMemAlloc(&buf_offset, N), "cuMemAlloc(screen_offset)");
			uint32_t s = static_cast<uint32_t>(args.range_start);
			uint32_t n = N;

			CUdeviceptr offset_regular = offset_stream_buffer;
			CUdeviceptr offset_alg0 = offset_regular + offset_stream_bytes;
			CUdeviceptr offset_alg6 = offset_alg0 + offset_stream_bytes;
			CUdeviceptr offset_alg2 = offset_alg6 + offset_stream_bytes;
			CUdeviceptr offset_alg5 = offset_alg2 + offset_carry_bytes;

			void* ref_args[] = {
				&buf_ref, &assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
				&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
				&assets.alg2_values, &assets.alg5_values, &assets.expansion_values,
				&assets.schedule_data, &assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id), &s, &n
			};
			void* offset_args[] = {
				&buf_offset, &offset_regular, &offset_alg0, &offset_alg6,
				&offset_alg2, &offset_alg5, &assets.expansion_values,
				&assets.schedule_data, &assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id), &s, &n
			};
			void* offset_interleaved_args[] = {
				&buf_offset, &offset_stream_buffer, &assets.expansion_values,
				&assets.schedule_data, &assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id), &s, &n
			};
			// ILP6 promoted over ILP8 on 2026-05-25 — +2.1% mean across 5 keys
			// at 16M-candidate workload. Mechanism: 25% smaller inlined-inner
			// code drops the no_instruction stall 34% (3.67 → 2.42 warps/issue)
			// and lifts IPC 3.14 → 3.26. ILP5 ties ILP6 — no further headroom.
			CUfunction active_offset_kernel = args.screen_offsets_interleaved ? screen_offset_interleaved_kernel : screen_offset_ilp6_preids_kernel;
			void** active_offset_args = args.screen_offsets_interleaved ? offset_interleaved_args : offset_args;

			float ref_ms = 0.0f, offset_ms = 0.0f;
			const uint32_t grid = screen_kernel_grid_x(n);
			const uint32_t offset_grid = args.screen_offsets_interleaved ? screen_kernel_grid_x(n) : screen_kernel_grid_x_ilp(n, 6u);
			check_cuda(cuLaunchKernel(screen_kernel, grid, 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, ref_args, nullptr), "warmup(screen_ref)");
			check_cuda(cuStreamSynchronize(stream), "sync");
			check_cuda(cuEventRecord(kernel_start, stream), "rec start");
			check_cuda(cuLaunchKernel(screen_kernel, grid, 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, ref_args, nullptr), "launch(screen_ref)");
			check_cuda(cuEventRecord(kernel_end, stream), "rec end");
			check_cuda(cuStreamSynchronize(stream), "sync");
			check_cuda(cuEventElapsedTime(&ref_ms, kernel_start, kernel_end), "elapsed");

			check_cuda(cuLaunchKernel(active_offset_kernel, offset_grid, 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, active_offset_args, nullptr), "warmup(screen_offset)");
			check_cuda(cuStreamSynchronize(stream), "sync");
			check_cuda(cuEventRecord(kernel_start, stream), "rec start");
			check_cuda(cuLaunchKernel(active_offset_kernel, offset_grid, 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, active_offset_args, nullptr), "launch(screen_offset)");
			check_cuda(cuEventRecord(kernel_end, stream), "rec end");
			check_cuda(cuStreamSynchronize(stream), "sync");
			check_cuda(cuEventElapsedTime(&offset_ms, kernel_start, kernel_end), "elapsed");

			std::vector<uint8_t> ref(N), offset(N);
			check_cuda(cuMemcpyDtoH(ref.data(), buf_ref, N), "copy ref");
			check_cuda(cuMemcpyDtoH(offset.data(), buf_offset, N), "copy offset");
			cuMemFree(buf_ref); cuMemFree(buf_offset);

			std::size_t mismatches = 0;
			std::size_t first_mismatch = 0;
			for (std::size_t i = 0; i < N; i++) if (ref[i] != offset[i]) { if (mismatches == 0) first_mismatch = i; mismatches++; }

			const double ref_cps = static_cast<double>(N) / (ref_ms / 1000.0);
			const double offset_cps = static_cast<double>(N) / (offset_ms / 1000.0);
			std::cout << "Screen vs Screen+Offset-Streams" << (args.screen_offsets_interleaved ? "-Interleaved" : "") << ": " << N << " candidates, key=0x"
			          << std::hex << std::setw(8) << std::setfill('0') << args.key_id << std::dec << std::setfill(' ') << "\n";
			std::cout << "  screen_universal: " << std::fixed << std::setprecision(3) << ref_ms << " ms   "
			          << std::fixed << std::setprecision(0) << ref_cps << " c/s\n";
			std::cout << "  " << (args.screen_offsets_interleaved ? "screen_offsets_interleaved" : "screen_offsets") << ": "
			          << std::fixed << std::setprecision(3) << offset_ms << " ms   "
			          << std::fixed << std::setprecision(0) << offset_cps << " c/s\n";
			std::cout << "  speedup vs universal: " << std::fixed << std::setprecision(3) << (ref_ms / offset_ms) << "x\n";
			if (mismatches == 0) std::cout << "  PASS - all " << N << " flag bytes match\n";
			else std::cout << "  FAIL - " << mismatches << "/" << N << " mismatches; first at candidate " << first_mismatch
			               << " ref=0x" << std::hex << (unsigned)ref[first_mismatch]
			               << " offset=0x" << (unsigned)offset[first_mismatch] << std::dec << "\n";
			return 0;
			}

			if (args.screen_offset_ilp_sweep && args.parity_count > 0)
			{
				const uint32_t N = args.parity_count;
				uint32_t s = static_cast<uint32_t>(args.range_start);
				uint32_t n = N;

				CUdeviceptr buf_ref = 0;
				check_cuda(cuMemAlloc(&buf_ref, N), "cuMemAlloc(screen_offset_ilp_ref)");
				void* ref_args[] = {
					&buf_ref, &offset_regular, &offset_alg0, &offset_alg6,
					&offset_alg2, &offset_alg5, &assets.expansion_values,
					&assets.schedule_data, &assets.carnival_data,
					const_cast<uint32_t*>(&args.key_id), &s, &n
				};
				check_cuda(cuLaunchKernel(screen_offset_kernel, screen_kernel_grid_x_ilp(n, 4u), 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, ref_args, nullptr), "warmup(screen_offset_ilp4_ref)");
				check_cuda(cuStreamSynchronize(stream), "sync");
				check_cuda(cuLaunchKernel(screen_offset_kernel, screen_kernel_grid_x_ilp(n, 4u), 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, ref_args, nullptr), "launch(screen_offset_ilp4_ref)");
				check_cuda(cuStreamSynchronize(stream), "sync");
				std::vector<uint8_t> ref(N);
				check_cuda(cuMemcpyDtoH(ref.data(), buf_ref, N), "copy screen_offset_ilp_ref");
				cuMemFree(buf_ref);

				struct IlpCase
				{
					const char* label;
					uint32_t ilp;
					CUfunction kernel;
				};
				IlpCase cases[] = {
					{"ilp1", 1u, screen_offset_ilp1_kernel},
					{"ilp2", 2u, screen_offset_ilp2_kernel},
					{"ilp4", 4u, screen_offset_kernel},
					{"ilp8", 8u, screen_offset_ilp8_kernel},
					{"ilp12", 12u, screen_offset_ilp12_kernel},
					{"ilp16", 16u, screen_offset_ilp16_kernel},
					{"ilp8_fixed", 8u, screen_offset_ilp8_fixed_kernel},
					{"ilp8_preids", 8u, screen_offset_ilp8_preids_kernel},
					{"ilp12_preids", 12u, screen_offset_ilp12_preids_kernel},
					{"ilp16_preids", 16u, screen_offset_ilp16_preids_kernel},
					{"ilp8_preids_carrysel", 8u, screen_offset_ilp8_preids_carrysel_kernel},
					{"ilp8_fixed_preids", 8u, screen_offset_ilp8_fixed_preids_kernel},
					{"ilp8_preids_unroll1", 8u, screen_offset_ilp8_preids_unroll1_kernel},
					{"ilp6_preids", 6u, screen_offset_ilp6_preids_kernel},
					{"ilp6_preids_unroll1", 6u, screen_offset_ilp6_preids_unroll1_kernel},
					{"ilp5_preids", 5u, screen_offset_ilp5_preids_kernel},
					{"ilp5_preids_unroll1", 5u, screen_offset_ilp5_preids_unroll1_kernel},
					{"ilp6_preids_ldg", 6u, screen_offset_ilp6_preids_ldg_kernel},
					{"ilp8_preids_ldg", 8u, screen_offset_ilp8_preids_ldg_kernel},
					{"ilp5_preids_prefetch", 5u, screen_offset_ilp5_preids_prefetch_kernel},
					{"ilp6_preids_prefetch", 6u, screen_offset_ilp6_preids_prefetch_kernel},
					{"ilp8_preids_prefetch", 8u, screen_offset_ilp8_preids_prefetch_kernel},
				};

				std::cout << "Screen offset ILP sweep: " << N << " candidates, key=0x"
				          << std::hex << std::setw(8) << std::setfill('0') << args.key_id
				          << std::dec << std::setfill(' ') << "\n";
				for (const IlpCase& test_case : cases)
				{
					CUdeviceptr buf_out = 0;
					check_cuda(cuMemAlloc(&buf_out, N), "cuMemAlloc(screen_offset_ilp_out)");
					void* kernel_args_ilp[] = {
						&buf_out, &offset_regular, &offset_alg0, &offset_alg6,
						&offset_alg2, &offset_alg5, &assets.expansion_values,
						&assets.schedule_data, &assets.carnival_data,
						const_cast<uint32_t*>(&args.key_id), &s, &n
					};

					const uint32_t grid = screen_kernel_grid_x_ilp(n, test_case.ilp);
					float elapsed_ms = 0.0f;
					check_cuda(cuLaunchKernel(test_case.kernel, grid, 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, kernel_args_ilp, nullptr), "warmup(screen_offset_ilp)");
					check_cuda(cuStreamSynchronize(stream), "sync");
					check_cuda(cuEventRecord(kernel_start, stream), "rec start");
					check_cuda(cuLaunchKernel(test_case.kernel, grid, 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, kernel_args_ilp, nullptr), "launch(screen_offset_ilp)");
					check_cuda(cuEventRecord(kernel_end, stream), "rec end");
					check_cuda(cuStreamSynchronize(stream), "sync");
					check_cuda(cuEventElapsedTime(&elapsed_ms, kernel_start, kernel_end), "elapsed");

					std::vector<uint8_t> out(N);
					check_cuda(cuMemcpyDtoH(out.data(), buf_out, N), "copy screen_offset_ilp_out");
					cuMemFree(buf_out);

					std::size_t mismatches = 0;
					std::size_t first_mismatch = 0;
					for (std::size_t i = 0; i < N; i++)
					{
						if (ref[i] != out[i])
						{
							if (mismatches == 0) first_mismatch = i;
							mismatches++;
						}
					}

					const double cps = static_cast<double>(N) / (elapsed_ms / 1000.0);
					std::cout << "  " << test_case.label << ": " << std::fixed << std::setprecision(3)
					          << elapsed_ms << " ms   " << std::fixed << std::setprecision(0) << cps << " c/s";
					if (mismatches == 0)
					{
						std::cout << "   PASS\n";
					}
					else
					{
						std::cout << "   FAIL " << mismatches << "/" << N << " first=" << first_mismatch
						          << " ref=0x" << std::hex << static_cast<unsigned>(ref[first_mismatch])
						          << " out=0x" << static_cast<unsigned>(out[first_mismatch]) << std::dec << "\n";
					}
				}
				return 0;
			}

			// ---- SCREEN-only A/B: universal-table screen vs pre-extracted maprng screen ----
		// Same 128-thread / 4-warp / ILP4 production geometry, no dedup. This
		// isolates whether the later pre-extracted maprng layout beats the
		// universal table path once runtime bit extraction is removed.
		if (args.screen_preext && args.parity_count > 0)
		{
			const uint32_t N = args.parity_count;
			CUdeviceptr buf_ref = 0, buf_preext = 0;
			check_cuda(cuMemAlloc(&buf_ref, N), "cuMemAlloc(screen_ref)");
			check_cuda(cuMemAlloc(&buf_preext, N), "cuMemAlloc(screen_preext)");
			uint32_t s = static_cast<uint32_t>(args.range_start);
			uint32_t n = N;

			void* ref_args[] = {
				&buf_ref, &assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
				&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
				&assets.alg2_values, &assets.alg5_values, &assets.expansion_values,
				&assets.schedule_data, &assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id), &s, &n
			};
			void* preext_args[] = {
				&buf_preext, &assets.expansion_values, &assets.schedule_data, &assets.carnival_data,
				&map_rng_buffer,
				const_cast<uint32_t*>(&args.key_id), &s, &n
			};

			float ref_ms = 0.0f, preext_ms = 0.0f;
			const uint32_t grid = screen_kernel_grid_x(n);
			check_cuda(cuLaunchKernel(screen_kernel, grid, 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, ref_args, nullptr), "warmup(screen_ref)");
			check_cuda(cuStreamSynchronize(stream), "sync");
			check_cuda(cuEventRecord(kernel_start, stream), "rec start");
			check_cuda(cuLaunchKernel(screen_kernel, grid, 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, ref_args, nullptr), "launch(screen_ref)");
			check_cuda(cuEventRecord(kernel_end, stream), "rec end");
			check_cuda(cuStreamSynchronize(stream), "sync");
			check_cuda(cuEventElapsedTime(&ref_ms, kernel_start, kernel_end), "elapsed");

			check_cuda(cuLaunchKernel(screen_maprng_preext_kernel, grid, 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, preext_args, nullptr), "warmup(screen_preext)");
			check_cuda(cuStreamSynchronize(stream), "sync");
			check_cuda(cuEventRecord(kernel_start, stream), "rec start");
			check_cuda(cuLaunchKernel(screen_maprng_preext_kernel, grid, 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, preext_args, nullptr), "launch(screen_preext)");
			check_cuda(cuEventRecord(kernel_end, stream), "rec end");
			check_cuda(cuStreamSynchronize(stream), "sync");
			check_cuda(cuEventElapsedTime(&preext_ms, kernel_start, kernel_end), "elapsed");

			std::vector<uint8_t> ref(N), preext(N);
			check_cuda(cuMemcpyDtoH(ref.data(), buf_ref, N), "copy ref");
			check_cuda(cuMemcpyDtoH(preext.data(), buf_preext, N), "copy preext");
			cuMemFree(buf_ref); cuMemFree(buf_preext);

			std::size_t mismatches = 0;
			std::size_t first_mismatch = 0;
			for (std::size_t i = 0; i < N; i++) if (ref[i] != preext[i]) { if (mismatches == 0) first_mismatch = i; mismatches++; }

			const double ref_cps = static_cast<double>(N) / (ref_ms / 1000.0);
			const double preext_cps = static_cast<double>(N) / (preext_ms / 1000.0);
			std::cout << "Screen vs Screen+Maprng-Preext: " << N << " candidates, key=0x"
			          << std::hex << std::setw(8) << std::setfill('0') << args.key_id << std::dec << std::setfill(' ') << "\n";
			std::cout << "  screen_universal: " << std::fixed << std::setprecision(3) << ref_ms << " ms   "
			          << std::fixed << std::setprecision(0) << ref_cps << " c/s\n";
			std::cout << "  screen_preext:    " << std::fixed << std::setprecision(3) << preext_ms << " ms   "
			          << std::fixed << std::setprecision(0) << preext_cps << " c/s\n";
			std::cout << "  speedup vs universal: " << std::fixed << std::setprecision(3) << (ref_ms / preext_ms) << "x\n";
			if (mismatches == 0) std::cout << "  PASS - all " << N << " flag bytes match\n";
			else std::cout << "  FAIL - " << mismatches << "/" << N << " mismatches; first at candidate " << first_mismatch
			               << " ref=0x" << std::hex << (unsigned)ref[first_mismatch]
			               << " preext=0x" << (unsigned)preext[first_mismatch] << std::dec << "\n";
			return 0;
		}

		// ---- Phase 3 SCREEN-only A/B: universal-table screen vs coalesced+preext maprng screen ----
		// No dedup. This isolates the cache-locality + coalesce optimization
		// on the production screen-kernel path that's used on 92% of keys
		// per the R-distribution measurement.
		if (args.screen_coalesced && args.parity_count > 0)
		{
			const uint32_t N = args.parity_count;
			CUdeviceptr buf_ref = 0, buf_coal = 0;
			check_cuda(cuMemAlloc(&buf_ref,  N), "cuMemAlloc(screen_ref)");
			check_cuda(cuMemAlloc(&buf_coal, N), "cuMemAlloc(screen_coal)");
			uint32_t s = static_cast<uint32_t>(args.range_start);
			uint32_t n = N;

			void* ref_args[] = {
				&buf_ref, &assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
				&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
				&assets.alg2_values, &assets.alg5_values, &assets.expansion_values,
				&assets.schedule_data, &assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id), &s, &n
			};
			void* coal_args[] = {
				&buf_coal, &assets.expansion_values, &assets.schedule_data, &assets.carnival_data,
				&map_rng_buffer,
				const_cast<uint32_t*>(&args.key_id), &s, &n
			};

			float ref_ms = 0.0f, coal_ms = 0.0f;
			check_cuda(cuLaunchKernel(screen_kernel, screen_kernel_grid_x(n), 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, ref_args, nullptr), "warmup(screen_ref)");
			check_cuda(cuStreamSynchronize(stream), "sync");
			check_cuda(cuEventRecord(kernel_start, stream), "rec start");
			check_cuda(cuLaunchKernel(screen_kernel, screen_kernel_grid_x(n), 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, ref_args, nullptr), "launch(screen_ref)");
			check_cuda(cuEventRecord(kernel_end, stream), "rec end");
			check_cuda(cuStreamSynchronize(stream), "sync");
			check_cuda(cuEventElapsedTime(&ref_ms, kernel_start, kernel_end), "elapsed");

			const uint32_t coal_grid = (n + 16u - 1u) / 16u;  // same as screen kernel: 4 warps × 4 ILP = 16 cand/block
			check_cuda(cuLaunchKernel(screen_maprng_coalesced_kernel, coal_grid, 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, coal_args, nullptr), "warmup(screen_coal)");
			check_cuda(cuStreamSynchronize(stream), "sync");
			check_cuda(cuEventRecord(kernel_start, stream), "rec start");
			check_cuda(cuLaunchKernel(screen_maprng_coalesced_kernel, coal_grid, 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, coal_args, nullptr), "launch(screen_coal)");
			check_cuda(cuEventRecord(kernel_end, stream), "rec end");
			check_cuda(cuStreamSynchronize(stream), "sync");
			check_cuda(cuEventElapsedTime(&coal_ms, kernel_start, kernel_end), "elapsed");

			std::vector<uint8_t> ref(N), coal(N);
			check_cuda(cuMemcpyDtoH(ref.data(),  buf_ref,  N), "copy ref");
			check_cuda(cuMemcpyDtoH(coal.data(), buf_coal, N), "copy coal");
			cuMemFree(buf_ref); cuMemFree(buf_coal);

			std::size_t mismatches = 0;
			std::size_t first_mismatch = 0;
			for (std::size_t i = 0; i < N; i++) if (ref[i] != coal[i]) { if (mismatches == 0) first_mismatch = i; mismatches++; }

			const double ref_cps  = static_cast<double>(N) / (ref_ms  / 1000.0);
			const double coal_cps = static_cast<double>(N) / (coal_ms / 1000.0);
			std::cout << "Screen vs Screen+Coalesced-Maprng-Preext: " << N << " candidates, key=0x"
			          << std::hex << std::setw(8) << std::setfill('0') << args.key_id << std::dec << std::setfill(' ') << "\n";
			std::cout << "  screen_universal:  " << std::fixed << std::setprecision(3) << ref_ms  << " ms   "
			          << std::fixed << std::setprecision(0) << ref_cps  << " c/s\n";
			std::cout << "  screen_coalesced:  " << std::fixed << std::setprecision(3) << coal_ms << " ms   "
			          << std::fixed << std::setprecision(0) << coal_cps << " c/s\n";
			std::cout << "  speedup vs universal: " << std::fixed << std::setprecision(3) << (ref_ms / coal_ms) << "x\n";
			if (mismatches == 0) std::cout << "  PASS — all " << N << " flag bytes match\n";
			else std::cout << "  FAIL — " << mismatches << "/" << N << " mismatches; first at candidate " << first_mismatch
			               << " ref=0x" << std::hex << (unsigned)ref[first_mismatch]
			               << " coal=0x" << (unsigned)coal[first_mismatch] << std::dec << "\n";
			return 0;
		}

		// ---- On-GPU VRAM survivor-compaction architecture bench ----
		// Multi-pass: run a map SPAN on the live frontier (state resident in VRAM),
		// within-block dedup at the span end, then compact survivors so the next
		// span's grid shrinks. Parity-checked against the production ilp6 screen.
		if (args.compaction_bench && args.parity_count > 0)
		{
			uint32_t N = args.parity_count;
			if (args.compaction_auto_tile)
			{
				// Pick the largest tile that fits free VRAM, WITH a cap that scales with the tile. Bigger
				// tiles are faster (fewer tile-boundary overheads) ONLY if the drain cap scales: a fixed cap
				// saturates on a big tile (LF->1.0) and over-keep tanks throughput (measured: 64M tile at a
				// fixed 2^22 cap = 156 M/s LF 0.996, but at a MATCHED 2^24 cap = 226 M/s LF 0.57, beating
				// 16M+2^22's 207). So under cap-auto (the --compaction-run preset, unless the operator set an
				// explicit --drain-cap-bits) we set cap_bits = log2(tile) - 2 (keeps cap LF ~0.6) and size the
				// tile from a per-candidate cost that INCLUDES that matched cap.
				// Per-candidate device bytes at PEAK: 144 base (state128+rep4+alive1+live8+flag1+buf2); the
				// production config (global-span0 + map1-table-auto + deep drains) ~doubles it on diffuse
				// tiles ⇒ 288 B/cand; the tile-scaled cap adds ndrains*ways*16/4 = ndrains*ways*4 B/cand.
				size_t free_b = 0, total_b = 0;
				check_cuda(cuMemGetInfo(&free_b, &total_b), "cuMemGetInfo");
				uint32_t ndrains = args.drain_boundaries.empty() ? 0u : 1u;
				for (char c : args.drain_boundaries) if (c == ',') ndrains++;
				// cap-auto: the --compaction-run preset scales the drain cap with the tile (cap_bits =
				// log2(tile)-2, LF ~0.6) unless the operator pinned --drain-cap-bits. Then the cap is part of
				// the per-candidate peak (ndrains*ways*16/4 B/cand); a pinned cap is a fixed reserve instead.
				const bool cap_auto = args.compaction_run && !args.drain_cap_bits_explicit
				                   && args.drain_cross_tile && ndrains > 0u;
				uint64_t per_cand = (args.compaction_global_span0 || args.map1_table_auto) ? 288ull : 144ull;
				uint64_t cap_reserve = 0ull;
				if (cap_auto)
					per_cand += static_cast<uint64_t>(ndrains) * args.drain_cap_ways * 4ull;   // tile-scaled cap
				else if (args.drain_cross_tile && args.drain_cap_bits > 0u && ndrains > 0u)
					cap_reserve = static_cast<uint64_t>(ndrains) * (1ull << args.drain_cap_bits)
					            * static_cast<uint64_t>(args.drain_cap_ways) * 16ull;       // fixed cap
				const size_t reserve = 512ull * 1024 * 1024 + cap_reserve;
				const size_t budget = (free_b > reserve) ? static_cast<size_t>((free_b - reserve) * 0.90) : 0;
				const uint64_t nmax = budget / per_cand;
				uint32_t tile = 4u * 1024u * 1024u;  // 4M floor
				while ((static_cast<uint64_t>(tile) << 1) <= nmax && (tile << 1) <= 134217728u) tile <<= 1;  // 128M ceiling
				N = tile;
				if (cap_auto)
				{
					uint32_t lg = 0u; for (uint32_t t = N; t > 1u; t >>= 1) lg++;            // log2(tile)
					uint32_t cb = (lg > 2u) ? lg - 2u : 20u;
					if (cb < 20u) cb = 20u; if (cb > 26u) cb = 26u;
					args.drain_cap_bits = cb;                                                // matched cap
				}
				std::cout << "  auto-tile: free " << (free_b >> 20) << " MB -> tile " << (N >> 20)
				          << "M, drain cap 2^" << args.drain_cap_bits << "x" << args.drain_cap_ways
				          << (cap_auto ? " (cap-auto, LF~0.6)" : "") << "\n";
			}
			uint32_t s = static_cast<uint32_t>(args.range_start);
			uint32_t cur_key = args.key_id;  // mutable so --calibrate can vary the key per sample

			CUdeviceptr offset_regular = offset_stream_buffer;
			CUdeviceptr offset_alg0 = offset_regular + offset_stream_bytes;
			CUdeviceptr offset_alg6 = offset_alg0 + offset_stream_bytes;
			CUdeviceptr offset_alg2 = offset_alg6 + offset_stream_bytes;
			CUdeviceptr offset_alg5 = offset_alg2 + offset_carry_bytes;

			// Reference: production ilp6 offset screen (parity + baseline rate).
			CUdeviceptr buf_ref = 0; check_cuda(cuMemAlloc(&buf_ref, N), "alloc ref");
			uint32_t n = N;
			void* ref_args[] = { &buf_ref, &offset_regular, &offset_alg0, &offset_alg6,
				&offset_alg2, &offset_alg5, &assets.expansion_values, &assets.schedule_data,
				&assets.carnival_data, &cur_key, &s, &n };
			const uint32_t ref_grid = screen_kernel_grid_x_ilp(n, 6u);
			auto launch_ref = [&](){ check_cuda(cuLaunchKernel(screen_offset_ilp6_preids_kernel, ref_grid,1,1,kCudaThreadsPerBlock,1,1,0,stream,ref_args,nullptr),"ref"); };
			launch_ref(); check_cuda(cuStreamSynchronize(stream),"s");
			auto rt0 = std::chrono::steady_clock::now();
			launch_ref(); check_cuda(cuStreamSynchronize(stream),"s");
			auto rt1 = std::chrono::steady_clock::now();
			const double ref_ms = std::chrono::duration<double,std::milli>(rt1-rt0).count();

			if (args.drain_bench)
			{
				// ── WS1 drop-drain POC: ilp6 screen (ref) vs +within-block-dedup@6 vs +global-drain@6 ──
				// Path A isolation: (global − within-block) drops = the cross-block dups only the
				// VRAM reach can see (the bandwidth-reframe test). Exact table ⇒ FN-safe by construction.
				auto count_hits = [&](CUdeviceptr buf)->size_t {
					std::vector<uint8_t> h(N);
					check_cuda(cuMemcpyDtoH(h.data(), buf, N), "dtoh hits");
					size_t c = 0; for (uint8_t x : h) if (x) ++c; return c;
				};
				const size_t ref_hits = count_hits(buf_ref);
				const double ref_cps = (double)N / (ref_ms / 1000.0);

				CUdeviceptr buf_drn = 0, dropc = 0;
				check_cuda(cuMemAlloc(&buf_drn, N), "alloc buf_drn");
				check_cuda(cuMemAlloc(&dropc, sizeof(unsigned long long)), "alloc dropc");
				uint32_t d_logm = 1u; while ((1ull << d_logm) < (uint64_t)N * 2ull) ++d_logm; if (d_logm > 31u) d_logm = 31u;
				const uint32_t d_slots = 1u << d_logm;
				CUdeviceptr d_table = 0; check_cuda(cuMemAlloc(&d_table, (size_t)d_slots * sizeof(uint64_t)), "alloc d_table");

				CUfunction k_block = nullptr, k_global = nullptr;
				check_cuda(cuModuleGetFunction(&k_block, module, "tm_screen_drain_block_map6_ilp6_cuda"), "get k_block");
				check_cuda(cuModuleGetFunction(&k_global, module, "tm_screen_drain_global_map6_ilp6_cuda"), "get k_global");
				const uint32_t d_grid = screen_kernel_grid_x_ilp(n, 6u);
				CUdeviceptr null_table = 0; uint32_t null_logm = 0u;

				auto run_drain = [&](CUfunction k, CUdeviceptr table, uint32_t logm, bool zero_table,
				                     double& out_ms, unsigned long long& out_drops, size_t& out_hits) {
					void* a[] = { &buf_drn, &offset_regular, &offset_alg0, &offset_alg6, &offset_alg2, &offset_alg5,
						&assets.expansion_values, &assets.schedule_data, &assets.carnival_data, &cur_key, &s, &n,
						&table, &logm, &dropc };
					auto once = [&]()->double {
						check_cuda(cuMemsetD8(dropc, 0, sizeof(unsigned long long)), "zero dropc");
						if (zero_table && table) check_cuda(cuMemsetD8(table, 0, (size_t)d_slots * sizeof(uint64_t)), "zero d_table");
						check_cuda(cuStreamSynchronize(stream), "presync");
						auto t0 = std::chrono::steady_clock::now();
						check_cuda(cuLaunchKernel(k, d_grid, 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, a, nullptr), "drain launch");
						check_cuda(cuStreamSynchronize(stream), "postsync");
						auto t1 = std::chrono::steady_clock::now();
						return std::chrono::duration<double, std::milli>(t1 - t0).count();
					};
					once();                 // warmup
					out_ms = once();        // timed
					check_cuda(cuMemcpyDtoH(&out_drops, dropc, sizeof(out_drops)), "dtoh drops");
					out_hits = count_hits(buf_drn);
				};

				double blk_ms = 0, glb_ms = 0; unsigned long long blk_drops = 0, glb_drops = 0; size_t blk_hits = 0, glb_hits = 0;
				run_drain(k_block,  null_table, null_logm, false, blk_ms, blk_drops, blk_hits);
				run_drain(k_global, d_table,   d_logm,    true,  glb_ms, glb_drops, glb_hits);

				const double blk_cps = (double)N / (blk_ms / 1000.0);
				const double glb_cps = (double)N / (glb_ms / 1000.0);
				std::cout << "\nWS1 drop-drain POC @ map6 — key=0x" << std::hex << cur_key << std::dec
				          << "  N=" << N << " (" << (N >> 20) << "M)  global_table=" << (((size_t)d_slots * 8) >> 20) << " MB\n";
				std::cout << std::fixed << std::setprecision(1);
				std::cout << "  ilp6 screen (ref)      : " << ref_ms << " ms  " << (ref_cps / 1e6) << " M/s   hits=" << ref_hits << "\n";
				std::cout << "  + within-block dedup@6 : " << blk_ms << " ms  " << (blk_cps / 1e6) << " M/s   hits=" << blk_hits
				          << "   drops=" << blk_drops << " (" << std::setprecision(2) << (100.0 * (double)blk_drops / N) << "%)"
				          << std::setprecision(3) << "   " << (blk_cps / ref_cps) << "x ref\n" << std::setprecision(1);
				std::cout << "  + global drain@6       : " << glb_ms << " ms  " << (glb_cps / 1e6) << " M/s   hits=" << glb_hits
				          << "   drops=" << glb_drops << " (" << std::setprecision(2) << (100.0 * (double)glb_drops / N) << "%)"
				          << std::setprecision(3) << "   " << (glb_cps / ref_cps) << "x ref\n" << std::setprecision(1);
				std::cout << "  reach delta (global-block drops) = " << ((long long)glb_drops - (long long)blk_drops)
				          << "   (cross-block dups only the VRAM reach sees)\n";
				return 0;
			}

			// Compaction pipeline buffers.
			uint32_t W = comp_warps * comp_ilp;     // candidates/block (mutable: calibrate sweeps geometry)
			uint32_t SPAN_BLOCK = comp_warps * 32u;
			CUdeviceptr state=0, rep_global=0, mult=0, alive=0, live_a=0, live_b=0, counter=0, flag=0, buf_comp=0;
			check_cuda(cuMemAlloc(&state, (size_t)N*32u*sizeof(uint32_t)), "alloc state");
			check_cuda(cuMemAlloc(&rep_global, (size_t)N*sizeof(uint32_t)), "alloc rep");
			if (args.compaction_multiplicity)
				check_cuda(cuMemAlloc(&mult, (size_t)N*sizeof(uint32_t)), "alloc mult");
			check_cuda(cuMemAlloc(&alive, (size_t)N), "alloc alive");
			check_cuda(cuMemAlloc(&live_a, (size_t)N*sizeof(uint32_t)), "alloc live_a");
			check_cuda(cuMemAlloc(&live_b, (size_t)N*sizeof(uint32_t)), "alloc live_b");
			check_cuda(cuMemAlloc(&counter, sizeof(uint32_t)), "alloc counter");
			check_cuda(cuMemAlloc(&flag, (size_t)N), "alloc flag");
			check_cuda(cuMemAlloc(&buf_comp, (size_t)N), "alloc result");
			if (args.compaction_multiplicity)
				std::cout << "  multiplicity sidecar: " << (N >> 20) << "M uint32 entries ("
				          << (((uint64_t)N * 4ull) >> 20) << " MB), current-tile owner merges only\n";

			// Span-0 global dedup table (window-cap). Default 2x tile (load factor 0.5);
			// --compaction-global-cap MB caps it. When the frontier exceeds the table's
			// safe occupancy the linear probe simply leaves late candidates as their own
			// survivors (correctness-safe: an un-merged dup just runs separately), so the
			// cap degrades dedup effectiveness gracefully rather than failing — the
			// keyclass-survey reality that only ~12% of keys one-shot W4B.
			CUdeviceptr g_table_fp = 0, g_table_rep = 0;
			uint32_t g_logm = 0u, g_slots = 0u;
			// The exact 2x-tile global table is only NEEDED when something actually uses it:
			// the EXACT span-0 merge (no MAP1 routing) or an EXACT deep drain (drains present,
			// no drain cap). When the span-0 merge is capped (--map1-route-tau) AND the drains
			// are capped (or absent), g_table is never touched — skip the alloc so the capped
			// MAP1 merge realises its fixed-footprint "fits VRAM" win (g_table = 6 GB @256M tile).
			std::vector<uint32_t> g_drain_vec;
			{ std::stringstream gs(args.drain_boundaries); std::string gt;
			  while (std::getline(gs, gt, ',')) if (!gt.empty()) g_drain_vec.push_back((uint32_t)std::stoul(gt)); }
			const bool exact_span0 = (args.map1_route_tau == 0u);
			const bool exact_drains = (!g_drain_vec.empty() && args.drain_cap_bits == 0u);
			const bool need_g_table = args.compaction_global_span0 && (exact_span0 || exact_drains);
			if (args.compaction_global_span0 && need_g_table)
			{
				// HLL-auto: a single map-1 HLL pass estimates the distinct MAP1 frontier so the
				// table is sized to 2x FRONTIER (not 2x tile). The pass cost is the viability
				// question — it is timed and reported. (No dedup table; just the walk + 4096-reg HLL.)
				uint64_t auto_target_slots = (uint64_t)N * 2ull;   // default 2x tile
					uint64_t est_frontier_for_lf = N;   // honest LF basis: frontier est (auto) or tile (default)
				if (args.map1_table_auto && run_span_hll_kernel)
				{
					CUdeviceptr hll_regs = 0;
					check_cuda(cuMemAlloc(&hll_regs, 4096u * sizeof(uint32_t)), "alloc hll_regs");
					check_cuda(cuMemsetD8(hll_regs, 0, 4096u * sizeof(uint32_t)), "zero hll_regs");
					uint32_t hm0 = 0u, hm1 = 1u, hMN = N, hds = 0u, htau = 0u;   // tau=0 → HLL all (full frontier)
					void* hargs[] = { &hMN, &hm0, &hm1, &offset_regular, &offset_alg0, &offset_alg6,
						&offset_alg2, &offset_alg5, &assets.expansion_values, &assets.schedule_data,
						&cur_key, &hds, &hll_regs, &htau };
					uint32_t hgrid = static_cast<uint32_t>((static_cast<uint64_t>(N) + 63ull) / 64ull);   // run_span_hll_w8i8: W=64 (64-bit grid math: N up to 2^32-1)
					check_cuda(cuStreamSynchronize(stream), "presync hll");
					auto h0 = std::chrono::steady_clock::now();
					check_cuda(cuLaunchKernel(run_span_hll_kernel, hgrid,1,1,256,1,1,0,stream,hargs,nullptr), "span_hll");
					check_cuda(cuStreamSynchronize(stream), "postsync hll");
					double hll_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-h0).count();
					std::vector<uint32_t> regs(4096);
					check_cuda(cuMemcpyDtoH(regs.data(), hll_regs, 4096u * sizeof(uint32_t)), "dtoh hll");
					cuMemFree(hll_regs);
					const uint64_t f1_est = hll_estimate(regs);
					// Target the SMALLEST power-of-2 that holds the frontier at LF ~0.8 (×1.25 margin for
					// HLL error). 2×frontier (LF 0.5) would round up to the same pow2 as 2×tile = no saving;
					// the soft-cap + multi-drain cadence heal any slight overflow (FN-safe), so LF ~0.7-0.9
					// is the throughput-safe operating point (the validated 768MB/LF0.72 = −2.2% case).
					auto_target_slots = f1_est * 5ull / 4ull;
						est_frontier_for_lf = f1_est;
					std::cout << "  HLL-auto: MAP1 frontier est=" << f1_est << " (" << (f1_est >> 20) << "M, "
					          << std::fixed << std::setprecision(2) << ((double)f1_est / (double)N) << "x tile)  pass="
					          << std::setprecision(1) << hll_ms << " ms (" << ((double)N / (hll_ms/1000.0) / 1e6)
					          << " M/s)  -> table 1.25x est (smallest pow2 holding it)\n";
				}
				g_logm = 1u;
				while ((1ull << g_logm) < auto_target_slots) ++g_logm;
				if (args.compaction_global_cap_mb)
				{
					const uint64_t cap_slots = ((uint64_t)args.compaction_global_cap_mb << 20) / 12ull;  // 8+4 B/slot
					uint32_t cap_logm = 1u;
					while ((1ull << (cap_logm + 1u)) <= cap_slots) ++cap_logm;
					if (cap_logm < g_logm) g_logm = cap_logm;
				}
				if (g_logm > 31u) g_logm = 31u;
				g_slots = 1u << g_logm;
				check_cuda(cuMemAlloc(&g_table_fp, (size_t)g_slots * sizeof(uint64_t)), "alloc g_table_fp");
				check_cuda(cuMemAlloc(&g_table_rep, (size_t)g_slots * sizeof(uint32_t)), "alloc g_table_rep");
				std::cout << "  span-0 global table: " << g_slots << " slots ("
				          << (((size_t)g_slots * 12u) >> 20) << " MB, est LF "
				          << std::fixed << std::setprecision(2) << ((double)est_frontier_for_lf / (double)g_slots)
				          << " = frontier/slots)\n";
			}

			std::vector<uint32_t> cuts;  // span boundaries; default = k4 shape
			if (args.compaction_spans.empty())
				cuts = {0u,1u,5u,9u,13u,17u,21u,25u,27u};
			else
			{
				std::stringstream ss(args.compaction_spans); std::string tok;
				while (std::getline(ss, tok, ',')) if (!tok.empty()) cuts.push_back(static_cast<uint32_t>(std::stoul(tok)));
				if (cuts.size() < 2u || cuts.front() != 0u || cuts.back() != 27u)
					throw std::runtime_error("--compaction-spans must start at 0, end at 27, e.g. 0,1,9,27");
			}
			const uint32_t NSPANS = static_cast<uint32_t>(cuts.size() - 1u);
			std::vector<uint32_t> span_M(NSPANS+1u, 0u);

			// Run the whole pipeline once; returns final survivor count. Used for
			// both warm-up and the timed run (re-inits rep_global each call).
			std::vector<uint32_t> drain_vec;
				if (!args.drain_boundaries.empty() || args.map1_route_tau > 0u)
				{
					std::stringstream ds(args.drain_boundaries); std::string t;
					while (std::getline(ds, t, ',')) if (!t.empty()) drain_vec.push_back(static_cast<uint32_t>(std::stoul(t)));
				}
				auto is_drain = [&](uint32_t m)->bool { for (uint32_t x : drain_vec) if (x == m) return true; return false; };
				bool instrument_occ = false;
				std::vector<OccRecord> occ_records;
				// WS1 inverse-bloom cap. Two layouts share one set of drain boundaries:
				//   128 / 64a : one 16-byte-slot epoch-tagged table (cap_table), zeroed ONCE.
				//   64b / 64c : two-array exact protocol (cap_fp[u64] + cap_rep[u32]). 64b re-zeros
				//               ONE table per boundary; 64c uses K distinct tables (one per drain
				//               site, indexed by position in drain_vec) zeroed once per pipeline.
				const bool cas_b  = (args.drain_cap_cas == "64b" || args.drain_cap_cas == "64c");
				const bool cas_kc = (args.drain_cap_cas == "64c");
				const uint32_t cap_ntab = cas_kc ? (drain_vec.empty() ? 1u : (uint32_t)drain_vec.size()) : 1u;
				auto drain_tab_idx = [&](uint32_t m)->uint32_t {   // 64c: boundary → its dedicated table
					for (uint32_t k = 0u; k < drain_vec.size(); k++) if (drain_vec[k] == m) return k;
					return 0u;
				};
				CUdeviceptr cap_table = 0; uint32_t cap_epoch = 1u;
				CUdeviceptr cap_fp = 0, cap_rep = 0;          // two-array cap (64b/64c)
				const uint64_t cap_slots = ((uint64_t)1u << args.drain_cap_bits) * args.drain_cap_ways;
				const bool use_cap = !drain_vec.empty() && args.drain_cap_bits > 0u && run_span_dedup_drain_cap_kernel;
				// Deep-routing: the DEEP drains route into the single-table cap — a MEMORY lever (the cap
				// holds only the likely-dup subset). Single-table cap only. Two classifiers:
				//   --drain-route-tau  : shed proxy over the span (MAP1-style; NOT the right deep classifier).
				//   --drain-route-traj : trajgate (op-tail count-min trajDens + sticky alg0) = the deep classifier.
				const bool use_deep_traj  = args.drain_route_traj && use_cap && !cas_b && run_span_deep_trajroute_cap_kernel;
				const bool use_deep_route = args.drain_route_tau > 0u && !use_deep_traj && use_cap && !cas_b && run_span0_routed_cap_kernel;
				CUdeviceptr traj_sketch = 0, traj_keybuf = 0;
				const uint64_t traj_cells = (uint64_t)1u << args.drain_traj_bits;
				const bool use_traj_2pass = use_deep_traj && args.drain_traj_2pass
				                            && run_span_deep_trajbuild_kernel && run_span_deep_trajroute2_kernel;
				// (2026-06-13) --drain-route-traj + --drain-cross-tile previously HUNG (GPU fault): the
				// trajroute/trajroute2 cap kernels called cap_probe_insert<MODE> without dispatching to the
				// xtile probe for CAP_CAS_XTILE, so XTILE fell through to CAS64A on a 128-bit slot. Fixed in
				// tm_cuda_dedup.cuh (cap_probe_insert<CAP_CAS_XTILE> now delegates to cap_probe_insert_xtile).
				if (use_deep_traj)
					check_cuda(cuMemAlloc(&traj_sketch, traj_cells * sizeof(uint32_t)), "alloc traj_sketch");
				if (use_traj_2pass)
					check_cuda(cuMemAlloc(&traj_keybuf, (size_t)N * sizeof(uint32_t)), "alloc traj_keybuf");
				if (use_cap && !cas_b)
				{
					check_cuda(cuMemAlloc(&cap_table, cap_slots * 16ull), "alloc cap_table");
					check_cuda(cuMemsetD8(cap_table, 0, cap_slots * 16ull), "zero cap_table (once)");
					std::cout << "  inverse-bloom drain cap: 2^" << args.drain_cap_bits << " x " << args.drain_cap_ways
					          << " = " << cap_slots << " slots = " << ((cap_slots * 16ull) >> 20) << " MB (FLAT in tile)"
					          << "  cas=" << args.drain_cap_cas << "\n";
				}
				else if (use_cap && cas_b)
				{
					const uint64_t tot = cap_slots * cap_ntab;
					check_cuda(cuMemAlloc(&cap_fp,  tot * 8ull), "alloc cap_fp");
					check_cuda(cuMemAlloc(&cap_rep, tot * 4ull), "alloc cap_rep");
					check_cuda(cuMemsetD8 (cap_fp,  0, tot * 8ull), "zero cap_fp");
					check_cuda(cuMemsetD32(cap_rep, 0xFFFFFFFFu, tot), "init cap_rep sentinel");
					std::cout << "  inverse-bloom drain cap: 2^" << args.drain_cap_bits << " x " << args.drain_cap_ways
					          << " = " << cap_slots << " slots = " << ((cap_slots * 12ull) >> 20) << " MB/table x "
					          << cap_ntab << " (FLAT in tile)  cas=" << args.drain_cap_cas << "\n";
				}

				// WS1 shed-proxy-routed span-0 MAP1 merge: its own cap + routing-stats counters.
				// (Single table — span 0 is launched once per pipeline — so 64b and 64c are identical
				// here: zero the table before the span-0 launch each pipeline.)
				CUdeviceptr m1_cap = 0, m1_fp = 0, m1_rep = 0, route_stats = 0; uint32_t m1_epoch = 1u;
				// route+dynamic: a routed-HLL pass (HLL only shed>=tau states) estimates the ROUTED
				// cardinality so the MAP1 cap is sized to 2x(routed cardinality) — the route+dynamic
				// compose. Multiplicative table shrink over the full-frontier size (only the high-shed
				// subset is hashed). One-time pass (amortized in a sweep).
				if (args.map1_cap_auto && args.map1_route_tau > 0u && args.compaction_global_span0
				    && !args.calibrate && run_span_hll_kernel)
				{
					CUdeviceptr hr = 0;
					check_cuda(cuMemAlloc(&hr, 4096u * sizeof(uint32_t)), "alloc routed-hll");
					check_cuda(cuMemsetD8(hr, 0, 4096u * sizeof(uint32_t)), "zero routed-hll");
					uint32_t rm0 = 0u, rm1 = 1u, rMN = N, rds = 0u, rtau = args.map1_route_tau;
					void* rargs[] = { &rMN, &rm0, &rm1, &offset_regular, &offset_alg0, &offset_alg6,
						&offset_alg2, &offset_alg5, &assets.expansion_values, &assets.schedule_data,
						&cur_key, &rds, &hr, &rtau };
					uint32_t rgrid = static_cast<uint32_t>((static_cast<uint64_t>(N) + 63ull) / 64ull); // 64-bit grid math: N up to 2^32-1
					check_cuda(cuStreamSynchronize(stream), "presync routed-hll");
					auto rt0 = std::chrono::steady_clock::now();
					check_cuda(cuLaunchKernel(run_span_hll_kernel, rgrid,1,1,256,1,1,0,stream,rargs,nullptr), "routed_hll");
					check_cuda(cuStreamSynchronize(stream), "postsync routed-hll");
					double rms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-rt0).count();
					std::vector<uint32_t> rr(4096);
					check_cuda(cuMemcpyDtoH(rr.data(), hr, 4096u * sizeof(uint32_t)), "dtoh routed-hll");
					cuMemFree(hr);
					const uint64_t routed_card = hll_estimate(rr);
					const uint64_t want_slots = routed_card * 2ull;   // 2x routed cardinality (LF 0.5)
					uint32_t bits = 1u; while (((uint64_t)1u << bits) * args.map1_cap_ways < want_slots) ++bits;
					if (bits < 10u) bits = 10u; if (bits > 28u) bits = 28u;
					args.map1_cap_bits = bits;
					std::cout << "  routed-HLL (tau=" << args.map1_route_tau << "): routed card est=" << routed_card
					          << " (" << (routed_card >> 20) << "M, " << std::fixed << std::setprecision(2)
					          << ((double)routed_card / (double)N) << "x tile)  pass=" << std::setprecision(1) << rms
					          << " ms  -> cap 2^" << bits << "x" << args.map1_cap_ways << " ("
					          << ((((uint64_t)1u << bits) * args.map1_cap_ways * 16ull) >> 20) << " MB)\n";
				}
				const uint64_t m1slots = ((uint64_t)1u << args.map1_cap_bits) * args.map1_cap_ways;
				const bool use_routed_span0 = args.map1_route_tau > 0u && args.compaction_global_span0
				                              && !args.calibrate && run_span0_routed_cap_kernel;
				if (use_routed_span0)
				{
					if (!cas_b)
					{
						check_cuda(cuMemAlloc(&m1_cap, m1slots * 16ull), "alloc m1_cap");
						check_cuda(cuMemsetD8(m1_cap, 0, m1slots * 16ull), "zero m1_cap (once)");
					}
					else
					{
						check_cuda(cuMemAlloc(&m1_fp,  m1slots * 8ull), "alloc m1_fp");
						check_cuda(cuMemAlloc(&m1_rep, m1slots * 4ull), "alloc m1_rep");
					}
					check_cuda(cuMemAlloc(&route_stats, 4ull * sizeof(unsigned long long)), "alloc route_stats");  // 4 slots: shared with deep-traj
					std::cout << "  routed span-0 MAP1 cap: tau=" << args.map1_route_tau << "  2^" << args.map1_cap_bits
					          << "x" << args.map1_cap_ways << " = " << ((m1slots * (cas_b?12ull:16ull)) >> 20)
					          << " MB (vs exact 2x-tile)  cas=" << args.drain_cap_cas << "\n";
				}
				if ((use_deep_route || use_deep_traj) && !route_stats)
					check_cuda(cuMemAlloc(&route_stats, 4ull * sizeof(unsigned long long)), "alloc route_stats (deep)");
				if (use_deep_route)
					std::cout << "  DEEP-ROUTED drains: shed tau=" << args.drain_route_tau
					          << " (hash shed>=tau into the drain cap, pass the rest = smaller deep cap)\n";
				if (use_deep_traj)
					std::cout << "  DEEP-TRAJGATE drains: dens_tau=" << args.drain_route_tau
					          << " sticky_alg0_tau=" << args.drain_sticky_tau
					          << " mult_tau=" << args.drain_mult_tau
					          << " sketch=2^" << args.drain_traj_bits << " u32 ("
					          << ((traj_cells * 4ull) >> 20) << " MB)  [op-tail count-min + sticky]\n";

				CUdeviceptr occ_counter = 0;
				if (count_nonzero_kernel && (need_g_table || use_routed_span0 || use_cap))
					check_cuda(cuMemAlloc(&occ_counter, sizeof(uint32_t)), "alloc occ_counter");
				auto record_exact_occ = [&](const std::string& label, CUdeviceptr table, uint64_t slots, uint32_t bytes_per_slot)
				{
					if (!instrument_occ || !occ_counter || !count_nonzero_kernel || !table || slots == 0ull) return;
					check_cuda(cuMemsetD32(occ_counter, 0u, 1u), "zero occ");
					uint32_t nslots = static_cast<uint32_t>(slots);
					uint32_t ogrid = (nslots + 255u) / 256u;
					void* oargs[] = { &table, &nslots, &occ_counter };
					check_cuda(cuLaunchKernel(count_nonzero_kernel, ogrid,1,1,256,1,1,0,stream,oargs,nullptr), "count_occ");
					check_cuda(cuStreamSynchronize(stream), "count_occ_sync");
					uint32_t occ = 0; check_cuda(cuMemcpyDtoH(&occ, occ_counter, sizeof(uint32_t)), "read occ");
					occ_records.push_back({ label, occ, slots, bytes_per_slot });
				};
				auto record_epoch_cap_occ = [&](const std::string& label, CUdeviceptr table, uint64_t slots,
				                                uint32_t bytes_per_slot, uint32_t mode, uint32_t epoch)
				{
					if (!instrument_occ || !occ_counter || !count_cap_epoch_kernel || !table || slots == 0ull) return;
					check_cuda(cuMemsetD32(occ_counter, 0u, 1u), "zero cap occ");
					uint32_t nslots = static_cast<uint32_t>(slots);
					uint32_t ogrid = (nslots + 255u) / 256u;
					void* oargs[] = { &table, &nslots, &mode, &epoch, &occ_counter };
					check_cuda(cuLaunchKernel(count_cap_epoch_kernel, ogrid,1,1,256,1,1,0,stream,oargs,nullptr), "count_cap_occ");
					check_cuda(cuStreamSynchronize(stream), "count_cap_occ_sync");
					uint32_t occ = 0; check_cuda(cuMemcpyDtoH(&occ, occ_counter, sizeof(uint32_t)), "read cap occ");
					occ_records.push_back({ label, occ, slots, bytes_per_slot });
				};

				auto run_pipeline = [&](bool record_M) -> uint32_t {
				check_cuda(cuMemsetD32(rep_global, 0xFFFFFFFFu, N), "memset rep");
				if (mult) check_cuda(cuMemsetD32(mult, 1u, N), "init mult");
				// 64c: zero the K dedicated drain tables ONCE per pipeline (each boundary then uses
				// its own clean table with no per-boundary re-zero, no cross-boundary contention).
				if (use_cap && cas_kc)
				{
					const uint64_t tot = cap_slots * cap_ntab;
					check_cuda(cuMemsetD8 (cap_fp,  0, tot * 8ull), "zero cap_fp (64c)");
					check_cuda(cuMemsetD32(cap_rep, 0xFFFFFFFFu, tot), "init cap_rep (64c)");
				}
				if ((use_deep_traj || use_deep_route) && route_stats)
					check_cuda(cuMemsetD8(route_stats, 0, 4ull * sizeof(unsigned long long)), "zero route_stats (deep)");
				uint32_t M = N;
				CUdeviceptr cur_live = 0;             // null = identity for first span
				CUdeviceptr next_live = live_a;
				int first = 1;
				if (record_M) span_M[0] = N;
				for (uint32_t sp = 0u; sp < NSPANS; sp++)
				{
					uint32_t m0 = cuts[sp], m1 = cuts[sp+1];
					const bool last = (sp == NSPANS - 1u && sp > 0u);
					// Final span (mode 2): fuse the screen into the span — advance through
					// the last maps and write flags in-register, no state writeback / no
					// separate final_flag kernel. mode 1 = dedup; mode 0 unused here.
					int mode = (last && args.compaction_fuse_final) ? 2
					         : (args.compaction_skip_final && last) ? 0 : 1;
					uint32_t span_grid = (M + W - 1u) / W;
					if (span_grid == 0u) span_grid = 1u;  // M==0 (tile fully collapsed before this boundary): keep launch valid; kernels guard cand<M so it is a no-op
					// Calibrate sweeps geometry by reloading only the within-block kernel, so
					// the global kernel (fixed geom) would mismatch — skip global under calibrate.
					const bool use_global = args.compaction_global_span0 && !args.calibrate && sp == 0u && NSPANS > 1u;
						if (use_routed_span0 && sp == 0u && NSPANS > 1u)
						{
							// Span 0 = shed-proxy-ROUTED MAP1 merge: dedup only high-shed (>=tau) states
							// into a small cap, pass the low-shed rest un-hashed (the in-place MAP1 pre-pass).
							check_cuda(cuMemsetD8(route_stats, 0, 2ull * sizeof(unsigned long long)), "zero route_stats");
								if (!cas_b)
								{
									uint32_t this_m1_epoch = m1_epoch;
									void* rargs[] = { &cur_live, &M, &state, &alive, &rep_global, &mult, &m0, &m1, &first,
										&offset_regular, &offset_alg0, &offset_alg6, &offset_alg2, &offset_alg5,
										&assets.expansion_values, &assets.schedule_data,
										&cur_key, &s, &m1_cap, &args.map1_cap_bits, &args.map1_cap_ways, &this_m1_epoch,
										&args.map1_route_tau, &route_stats };
									check_cuda(cuLaunchKernel(run_span0_routed_cap_kernel, span_grid,1,1,SPAN_BLOCK,1,1,0,stream,rargs,nullptr),"span0_routed");
									const uint32_t cap_mode = (args.drain_cap_cas == "64a") ? 1u : 0u;
									record_epoch_cap_occ("MAP1 routed cap", m1_cap, m1slots, 16u, cap_mode, this_m1_epoch);
								}
								else
								{
									// two-array exact protocol: re-zero the single MAP1 cap before this launch.
									check_cuda(cuMemsetD8 (m1_fp,  0, m1slots * 8ull), "zero m1_fp");
								check_cuda(cuMemsetD32(m1_rep, 0xFFFFFFFFu, m1slots), "init m1_rep");
								void* rargs[] = { &cur_live, &M, &state, &alive, &rep_global, &mult, &m0, &m1, &first,
									&offset_regular, &offset_alg0, &offset_alg6, &offset_alg2, &offset_alg5,
									&assets.expansion_values, &assets.schedule_data,
										&cur_key, &s, &m1_fp, &m1_rep, &args.map1_cap_bits, &args.map1_cap_ways,
										&args.map1_route_tau, &route_stats };
									check_cuda(cuLaunchKernel(run_span0_routed_cap_kernel, span_grid,1,1,SPAN_BLOCK,1,1,0,stream,rargs,nullptr),"span0_routed_b");
									record_exact_occ("MAP1 routed cap", m1_fp, m1slots, 12u);
								}
								m1_epoch++;
						}
						else
					if (use_global)
					{
						// Span 0: GLOBAL VRAM table catches the front-loaded map-1 collapse
						// that the within-block table (W<=96/block) misses. Zero table, dedup
						// the whole window globally, then the normal compaction packs survivors.
						uint32_t zgrid = (g_slots + 255u) / 256u;
						void* zargs[] = { &g_table_fp, &g_table_rep, &g_slots };
						check_cuda(cuLaunchKernel(hash_zero_kernel, zgrid,1,1,256,1,1,0,stream,zargs,nullptr),"hash_zero");
							void* gargs[] = { &M, &state, &alive, &rep_global, &mult, &m0, &m1,
								&offset_regular, &offset_alg0, &offset_alg6, &offset_alg2, &offset_alg5,
								&assets.expansion_values, &assets.schedule_data,
								&cur_key, &s, &g_table_fp, &g_table_rep, &g_logm };
							check_cuda(cuLaunchKernel(run_span_dedup_global_kernel, span_grid,1,1,SPAN_BLOCK,1,1,0,stream,gargs,nullptr),"span_global");
							record_exact_occ("MAP1 exact g_table", g_table_fp, g_slots, 12u);
						}
					else
					{
						void* span_args[] = { &cur_live, &M, &state, &alive, &rep_global, &mult, &m0, &m1, &first,
							&offset_regular, &offset_alg0, &offset_alg6, &offset_alg2, &offset_alg5,
							&assets.expansion_values, &assets.schedule_data,
							&cur_key, &s, &assets.carnival_data, &flag, &mode };
						if (use_cap && !args.calibrate && sp > 0u && is_drain(m1) && !cas_b)
							{
								// WS1 inverse-bloom cap drain: fixed-capacity, epoch-tagged (NO re-zero), short probe.
								// cross-tile: boundary-keyed epoch (m1) so a later window matches earlier windows.
								uint32_t this_epoch = args.drain_cross_tile ? m1 : cap_epoch;
								if (use_traj_2pass)
								{
									// TWO-PASS trajgate: pass1 builds the FULL sketch + per-state key/sticky (no
									// routing, all kept); pass2 probes the COMPLETE sketch + routes. Fixes the racy
									// single-pass density (parallel probe-then-update). No re-walk in pass2.
									check_cuda(cuMemsetD32(traj_sketch, 0, traj_cells), "zero traj_sketch");
									void* p1[] = { &cur_live, &M, &state, &alive, &m0, &m1, &first,
										&offset_regular, &offset_alg0, &offset_alg6, &offset_alg2, &offset_alg5,
										&assets.expansion_values, &assets.schedule_data,
										&cur_key, &s, &traj_keybuf, &traj_sketch, &args.drain_traj_bits, &args.drain_sticky_tau };
									check_cuda(cuLaunchKernel(run_span_deep_trajbuild_kernel, span_grid,1,1,SPAN_BLOCK,1,1,0,stream,p1,nullptr),"deep_trajbuild");
									void* p2[] = { &cur_live, &M, &state, &alive, &rep_global, &mult, &first,
										&cap_table, &args.drain_cap_bits, &args.drain_cap_ways, &this_epoch,
										&args.drain_route_tau, &traj_keybuf, &traj_sketch, &args.drain_traj_bits, &route_stats,
										&args.drain_mult_tau };
									check_cuda(cuLaunchKernel(run_span_deep_trajroute2_kernel, span_grid,1,1,SPAN_BLOCK,1,1,0,stream,p2,nullptr),"deep_trajroute2");
								}
								else if (use_deep_traj)
								{
									// DEEP-TRAJGATE drain (single-pass): op-tail count-min trajDens + sticky alg0 over
									// THIS deep span → hash only likely-dups into the cap, pass the rest. Reset the
									// sketch per boundary. NOTE: the single-pass density is racy under parallelism
									// (use --drain-traj-2pass for a correct, complete sketch).
									check_cuda(cuMemsetD32(traj_sketch, 0, traj_cells), "zero traj_sketch");
									void* targs[] = { &cur_live, &M, &state, &alive, &rep_global, &mult, &m0, &m1, &first,
										&offset_regular, &offset_alg0, &offset_alg6, &offset_alg2, &offset_alg5,
										&assets.expansion_values, &assets.schedule_data,
										&cur_key, &s, &cap_table, &args.drain_cap_bits, &args.drain_cap_ways, &this_epoch,
										&args.drain_route_tau, &args.drain_sticky_tau, &traj_sketch, &args.drain_traj_bits,
										&route_stats, &args.drain_mult_tau };
									check_cuda(cuLaunchKernel(run_span_deep_trajroute_cap_kernel, span_grid,1,1,SPAN_BLOCK,1,1,0,stream,targs,nullptr),"deep_trajroute");
								}
								else if (use_deep_route)
								{
									// DEEP-ROUTED drain: shed-proxy over THIS deep span [m0,m1) → hash only
									// shed>=tau into the deep cap, pass the rest un-hashed. The same routed
									// kernel as span-0 (it generalises to any span); cap holds only the
									// high-shed subset = the MEMORY lever for massive deep diffuse frontiers.
									void* rargs[] = { &cur_live, &M, &state, &alive, &rep_global, &mult, &m0, &m1, &first,
										&offset_regular, &offset_alg0, &offset_alg6, &offset_alg2, &offset_alg5,
										&assets.expansion_values, &assets.schedule_data,
										&cur_key, &s, &cap_table, &args.drain_cap_bits, &args.drain_cap_ways, &this_epoch,
										&args.drain_route_tau, &route_stats };
									check_cuda(cuLaunchKernel(run_span0_routed_cap_kernel, span_grid,1,1,SPAN_BLOCK,1,1,0,stream,rargs,nullptr),"deep_routed_drain");
								}
								else
									{
										void* cargs[] = { &cur_live, &M, &state, &alive, &rep_global, &mult, &m0, &m1, &first,
											&offset_regular, &offset_alg0, &offset_alg6, &offset_alg2, &offset_alg5,
											&assets.expansion_values, &assets.schedule_data,
											&cur_key, &s, &cap_table, &args.drain_cap_bits, &args.drain_cap_ways, &this_epoch };
										check_cuda(cuLaunchKernel(run_span_dedup_drain_cap_kernel, span_grid,1,1,SPAN_BLOCK,1,1,0,stream,cargs,nullptr),"span_drain_cap");
									}
									const uint32_t cap_mode = args.drain_cross_tile ? 2u : (args.drain_cap_cas == "64a" ? 1u : 0u);
									record_epoch_cap_occ("drain cap @map" + std::to_string(m1), cap_table, cap_slots, 16u, cap_mode, this_epoch);
									if (!args.drain_cross_tile) cap_epoch++;
								}
							else if (use_cap && !args.calibrate && sp > 0u && is_drain(m1) && cas_b)
							{
								// two-array exact 64-bit cap. 64b: one table, re-zero per boundary.
								// 64c: K tables (zeroed once per pipeline), this boundary's dedicated table.
								const uint32_t tk = cas_kc ? drain_tab_idx(m1) : 0u;
								CUdeviceptr d_fp  = cap_fp  + (CUdeviceptr)((uint64_t)tk * cap_slots * 8ull);
								CUdeviceptr d_rep = cap_rep + (CUdeviceptr)((uint64_t)tk * cap_slots * 4ull);
								if (!cas_kc)   // 64b: re-zero this single table before use
								{
									check_cuda(cuMemsetD8 (d_fp,  0, cap_slots * 8ull), "zero cap_fp (64b)");
									check_cuda(cuMemsetD32(d_rep, 0xFFFFFFFFu, cap_slots), "init cap_rep (64b)");
								}
									void* cargs[] = { &cur_live, &M, &state, &alive, &rep_global, &mult, &m0, &m1, &first,
										&offset_regular, &offset_alg0, &offset_alg6, &offset_alg2, &offset_alg5,
										&assets.expansion_values, &assets.schedule_data,
										&cur_key, &s, &d_fp, &d_rep, &args.drain_cap_bits, &args.drain_cap_ways };
									check_cuda(cuLaunchKernel(run_span_dedup_drain_cap_kernel, span_grid,1,1,SPAN_BLOCK,1,1,0,stream,cargs,nullptr),"span_drain_capb");
									record_exact_occ("drain cap @map" + std::to_string(m1), d_fp, cap_slots, 12u);
								}
							else if (!drain_vec.empty() && !args.calibrate && sp > 0u && is_drain(m1) && run_span_dedup_drain_kernel)
							{
								// WS1 Path B: GLOBAL drain at a deep span boundary (after span-0/MAP1
								// merge). Re-zero the global table (span 0 done with it), dedup this
								// span's frontier globally; the following compaction re-densifies so
								// the extra drops convert to throughput.
								// --drain-table-auto: EXACT-presize this drain to 2x M (the KNOWN frontier
								// entering it), since the deep frontier shrinks hard with depth. Re-zero +
								// probe only 2^drain_logm slots → less zero cost + better locality (the deeper
								// drains get aggressively smaller tables). M is exact (no estimate/rehash).
								uint32_t drain_logm = g_logm;
								if (args.drain_table_auto)
								{
									drain_logm = 1u; while ((1ull << drain_logm) < (uint64_t)M * 2ull) ++drain_logm;  // 2x M = LF 0.5
									if (drain_logm > g_logm) drain_logm = g_logm; if (drain_logm < 10u) drain_logm = 10u;
									const uint32_t dslots = 1u << drain_logm;
									std::cout << "  [drain-auto m" << m1 << "] frontier M=" << M << " -> table 2^" << drain_logm
									          << " (" << (((uint64_t)dslots * 12u) >> 20) << " MB, LF "
									          << std::fixed << std::setprecision(3) << ((double)M / (double)dslots) << ")\n";
								}
								const uint32_t dz_slots = 1u << drain_logm;
								uint32_t dz_slots_arg = dz_slots;
								uint32_t zgrid = (dz_slots + 255u) / 256u;
								void* zargs[] = { &g_table_fp, &g_table_rep, &dz_slots_arg };
								check_cuda(cuLaunchKernel(hash_zero_kernel, zgrid,1,1,256,1,1,0,stream,zargs,nullptr),"hash_zero_drain");
								void* dargs[] = { &cur_live, &M, &state, &alive, &rep_global, &mult, &m0, &m1, &first,
									&offset_regular, &offset_alg0, &offset_alg6, &offset_alg2, &offset_alg5,
									&assets.expansion_values, &assets.schedule_data,
									&cur_key, &s, &g_table_fp, &g_table_rep, &drain_logm };
								check_cuda(cuLaunchKernel(run_span_dedup_drain_kernel, span_grid,1,1,SPAN_BLOCK,1,1,0,stream,dargs,nullptr),"span_drain");
									// Count occupied slots = the drain's ACTUAL footprint at this boundary.
									record_exact_occ("exact drain @map" + std::to_string(m1), g_table_fp, dz_slots, 12u);
								}
							else
							{
								check_cuda(cuLaunchKernel(run_span_dedup_kernel, span_grid,1,1,SPAN_BLOCK,1,1,0,stream,span_args,nullptr),"span");
							}
					}
					if (mode == 1)
					{
						check_cuda(cuMemsetD32(counter, 0u, 1u), "memset counter");
						uint32_t cgrid = (M + 255u) / 256u;
						if (cgrid == 0u) cgrid = 1u;  // M==0: compaction over zero survivors is a no-op (counter stays 0), but the launch grid must be >=1
						void* comp_args[] = { &alive, &cur_live, &M, &next_live, &counter, &first };
						check_cuda(cuLaunchKernel(compact_survivors_kernel, cgrid,1,1,256,1,1,0,stream,comp_args,nullptr),"compact");
						check_cuda(cuStreamSynchronize(stream),"s");
						check_cuda(cuMemcpyDtoH(&M, counter, sizeof(uint32_t)), "read counter");
						cur_live = next_live;
						next_live = (next_live == live_a) ? live_b : live_a;
					}
					// else: cur_live / M stay = this span's input frontier (now advanced
					// through the final maps in state[]); final_flag screens it directly.
					first = 0;
					if (record_M) span_M[sp+1] = M;
				}
				// Final flags for survivors (unless the last span already fused the
				// screen in mode 2), then resolve every candidate.
				const bool fused = args.compaction_fuse_final && NSPANS > 1u;
				if (!fused)
				{
					uint32_t fgrid = (M*32u + SPAN_BLOCK - 1u) / SPAN_BLOCK;
					void* ff_args[] = { &cur_live, &M, &state, &flag, &assets.carnival_data };
					check_cuda(cuLaunchKernel(final_flag_survivors_kernel, fgrid?fgrid:1u,1,1,SPAN_BLOCK,1,1,0,stream,ff_args,nullptr),"final_flag");
				}
				uint32_t rgrid = (N + 255u) / 256u;
				uint32_t Nn = N;
				void* rs_args[] = { &rep_global, &flag, &buf_comp, &Nn };
				check_cuda(cuLaunchKernel(resolve_flags_kernel, rgrid,1,1,256,1,1,0,stream,rs_args,nullptr),"resolve");
				// Cross-tile FN-fix: AFTER resolve_flags, fill each THIS-TILE cap slot's resolved flag
				// from buf_comp (the fully-resolved per-candidate flag — correct even for a rep that was
				// itself dropped at a deeper boundary, where raw flag[] would be stale). The NEXT tile's
				// cross-tile dups then adopt the correct flag instead of a stale index.
				if (args.drain_cross_tile && use_cap && cap_resolve_xtile_kernel)
				{
					uint64_t nslots = cap_slots;
					uint32_t crgrid = (uint32_t)((nslots + 255ull) / 256ull);
					void* cr_args[] = { &cap_table, &nslots, &buf_comp };
					check_cuda(cuLaunchKernel(cap_resolve_xtile_kernel, crgrid,1,1,256,1,1,0,stream,cr_args,nullptr),"cap_resolve_xtile");
				}
				check_cuda(cuStreamSynchronize(stream),"s");
				return M;
			};

			// ---- Calibration: measure host-optimal engine, write shared config ----
			if (args.calibrate)
			{
				auto splitmix = [](uint64_t x) -> uint32_t {
					x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
					x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return (uint32_t)(x ^ (x >> 31));
				};
				const uint32_t windows[3] = { 0u, 0x55555555u, 0xAAAAAAAAu };
				std::vector<std::pair<uint32_t,uint32_t>> samples;
				for (uint32_t ki = 0; ki < 4u; ki++)
					for (uint32_t wi = 0; wi < 3u; wi++)
						samples.push_back({ splitmix(0xC0FFEEu + ki), windows[wi] });

				std::vector<uint8_t> hr(N), hc(N);
				// Sweep span geometries (all precompiled in the fatbin — no rebuild, unlike
				// OpenCL). w8i8 is the Blackwell optimum but the register-cliff location and
				// small-L2 behavior can shift across NVIDIA gens, so measure per host.
				const char* geoms[] = { "w8i5", "w8i6", "w8i8", "w8i10", "w8i4", "w4i8" };
				std::cout << "Calibrating " << device_name_buffer << " — " << samples.size()
				          << " samples x " << (sizeof(geoms)/sizeof(geoms[0])) << " geometries (CUDA)...\n";
				double best_agg = 0.0; std::string best_geom = "w8i8"; std::size_t total_mism = 0;
				for (const char* geom : geoms)
				{
					std::string g(geom); const size_t ip = g.find('i');
					comp_warps = (uint32_t)std::stoul(g.substr(1, ip - 1));
					comp_ilp   = (uint32_t)std::stoul(g.substr(ip + 1));
					W = comp_warps * comp_ilp; SPAN_BLOCK = comp_warps * 32u;
					const std::string fn = "run_span_dedup_" + g + "_cuda";
					check_cuda(cuModuleGetFunction(&run_span_dedup_kernel, module, fn.c_str()), ("cal:" + fn).c_str());
					cur_key = samples[0].first; s = samples[0].second;
					launch_ref(); check_cuda(cuStreamSynchronize(stream), "s"); run_pipeline(false);  // warm-up
					double inv_sum = 0.0;
					for (auto& smp : samples)
					{
						cur_key = smp.first; s = smp.second;
						check_cuda(cuStreamSynchronize(stream), "s");
						auto a0 = std::chrono::steady_clock::now();
						launch_ref(); check_cuda(cuStreamSynchronize(stream), "s");
						auto a1 = std::chrono::steady_clock::now();
						run_pipeline(false);
						auto a2 = std::chrono::steady_clock::now();
						const double s_ms = std::chrono::duration<double,std::milli>(a1 - a0).count();
						const double c_ms = std::chrono::duration<double,std::milli>(a2 - a1).count();
						check_cuda(cuMemcpyDtoH(hr.data(), buf_ref, N), "cal ref");
						check_cuda(cuMemcpyDtoH(hc.data(), buf_comp, N), "cal comp");
						for (uint32_t i = 0; i < N; i++) if (hr[i] != hc[i]) total_mism++;
						inv_sum += c_ms / s_ms;
					}
					const double agg = (double)samples.size() / inv_sum;
					std::cout << "  geom=" << geom << "  aggregate vs screen: " << std::fixed << std::setprecision(3) << agg << "x\n";
					if (agg > best_agg) { best_agg = agg; best_geom = g; }
				}

				// Production tile from free VRAM (cap 128M = plateau + index safety + divides 2^32).
				size_t free_b = 0, total_b = 0; check_cuda(cuMemGetInfo(&free_b, &total_b), "cuMemGetInfo");
				const size_t reserve = 128ull * 1024 * 1024;
				const size_t budget = (free_b > reserve) ? (size_t)((free_b - reserve) * 0.92) : 0;
				const uint64_t nmax = budget / 144ull;
				uint32_t tile = 4u * 1024u * 1024u;
				while (((uint64_t)tile << 1) <= nmax && (tile << 1) <= 134217728u) tile <<= 1;
				const char* engine = (best_agg >= 1.0) ? "compaction" : "screen";

				size_t devmem = 0; cuDeviceTotalMem(&devmem, device);
				int sm = 0; cuDeviceGetAttribute(&sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
				std::ostringstream fpkey;
				// GB-bucketed memory (see the reader above): stable across the few-MB total-VRAM
				// drift that otherwise leaves stale duplicate lines / misses the lookup.
				fpkey << "cuda|" << device_name_buffer << "|" << (devmem >> 30) << "GB|" << sm << "SM";
				std::ostringstream line;
				line << fpkey.str() << "\tengine=" << engine << " geom=" << best_geom
				     << " tile=" << (tile >> 20) << "M aggregate=" << std::fixed << std::setprecision(3) << best_agg
				     << " runtime=cuda";

				// Rewrite the config: drop any prior line for THIS device (matched on identity =
				// runtime|name|SM, ignoring the memory token so an old MB-keyed or drifted line is
				// replaced, not duplicated), then append the fresh result.
				auto identity = [](const std::string& key) -> std::string {
					std::vector<std::string> f; std::string cur; std::istringstream ks(key);
					while (std::getline(ks, cur, '|')) f.push_back(cur);
					if (f.size() < 4) return key;                      // unknown format: exact match
					return f.front() + "|" + f[1] + "|" + f.back();    // runtime|name|<NN>SM
				};
				const std::string my_id = identity(fpkey.str());
				std::vector<std::string> lines;
				{
					std::ifstream in(args.config_path); std::string l;
					while (std::getline(in, l)) {
						if (l.empty() || l[0] == '#') continue;
						if (identity(l.substr(0, l.find('\t'))) != my_id) lines.push_back(l);
					}
				}
				lines.push_back(line.str());
				{
					std::ofstream out(args.config_path, std::ios::trunc);
					out << "# tm_compaction.conf  (runtime|device-fingerprint -> engine/geom/tile)\n";
					out << "# NOTE: raceway is the production-default architecture; the screen/compaction\n";
					out << "#       engines recorded here are research/A-B only.\n";
					for (auto& l : lines) out << l << "\n";
				}

				std::cout << "Calibration result: engine=" << engine << "  geom=" << best_geom
				          << "  tile=" << (tile >> 20) << "M  aggregate=" << std::fixed << std::setprecision(3) << best_agg << "x\n";
				std::cout << (total_mism == 0 ? "  parity: PASS (all samples match the screen)\n"
				             : ("  parity: FAIL (" + std::to_string(total_mism) + " mismatches)\n"));
				std::cout << "  wrote " << args.config_path << "  [" << fpkey.str() << "]\n";
				if (best_agg < 1.0)
					std::cout << "  note: compaction below break-even here (likely small-L2) -> flat screen chosen.\n";

				cuMemFree(buf_ref); cuMemFree(state); cuMemFree(rep_global); if (mult) cuMemFree(mult); cuMemFree(alive);
				cuMemFree(live_a); cuMemFree(live_b); cuMemFree(counter); cuMemFree(flag); cuMemFree(buf_comp);
				if (g_table_fp) cuMemFree(g_table_fp);
				if (g_table_rep) cuMemFree(g_table_rep);
				return 0;
			}

			// ---- Full 2^32 end-to-end sweep: every tile, real timing + full parity ----
			if (args.compaction_sweep)
			{
				const uint64_t TOTAL = (uint64_t)1u << 32;
				if ((TOTAL % (uint64_t)N) != 0ull)
					throw std::runtime_error("--compaction-sweep needs a tile size (--parity) dividing 2^32 (use a power of 2, e.g. 67108864)");
				const uint64_t ntiles = TOTAL / (uint64_t)N;
				std::vector<uint8_t> ref(N), comp(N);
				s = 0u; launch_ref(); check_cuda(cuStreamSynchronize(stream),"s"); run_pipeline(false);  // warm-up
				double screen_total_ms = 0.0, comp_total_ms = 0.0;
				uint64_t total_mism = 0ull, first_mism_tile = ntiles, total_final_frontier = 0ull;
				std::cout << "Full-key sweep (2^32) key=0x" << std::hex << std::setw(8) << std::setfill('0')
				          << args.key_id << std::dec << std::setfill(' ') << "  tile=" << N
				          << " (" << (static_cast<double>(N) / 1048576.0) << "M)  "
				          << ntiles << " tiles  geom=" << args.compaction_ilp << "\n";
				// Hit-finder: tally every screen pass across the full key. This is the
				// screen-and-discard hit report — each tile is screened then discarded
				// (buffers reused); only the rare hits are retained. resolve_flags has
				// already propagated each root's flag to every duplicate, so a nonzero
				// comp[i] counts EVERY hitting data value (weighted / all-data), which is
				// the actionable candidate list for downstream verification.
				uint64_t hits_total = 0ull, hits_carnival = 0ull, hits_other = 0ull, hits_dual = 0ull;
				std::vector<std::pair<uint32_t,uint8_t>> hit_samples;
				std::ofstream hit_out;
				if (!args.sweep_hit_file.empty())
				{
					hit_out.open(args.sweep_hit_file, std::ios::trunc);
					if (!hit_out) throw std::runtime_error("cannot open --sweep-hit-file: " + args.sweep_hit_file);
					hit_out << "# data_value\tflag  key=0x" << std::hex << std::setw(8) << std::setfill('0')
					        << args.key_id << std::dec << std::setfill(' ')
					        << "  flag: 0x08=carnival 0x09=other-world 0x0B=dual\n";
				}
				for (uint64_t t = 0; t < ntiles; t++)
				{
					s = (uint32_t)(t * (uint64_t)N);
					check_cuda(cuStreamSynchronize(stream),"s");
					auto a0 = std::chrono::steady_clock::now();
					launch_ref(); check_cuda(cuStreamSynchronize(stream),"s");
					auto a1 = std::chrono::steady_clock::now();
					screen_total_ms += std::chrono::duration<double,std::milli>(a1-a0).count();
					auto b0 = std::chrono::steady_clock::now();
					uint64_t tile_frontier = run_pipeline(false);  // internally syncs at end
					total_final_frontier += tile_frontier;
					auto b1 = std::chrono::steady_clock::now();
					comp_total_ms += std::chrono::duration<double,std::milli>(b1-b0).count();
					check_cuda(cuMemcpyDtoH(ref.data(), buf_ref, N), "copy ref");
					check_cuda(cuMemcpyDtoH(comp.data(), buf_comp, N), "copy comp");
					uint64_t tile_mism = 0;
					for (size_t i = 0; i < N; i++)
					{
						if (ref[i] != comp[i]) tile_mism++;
						const uint8_t f = comp[i];
						if (!f) continue;
						hits_total++;
						if      (f == 0x0Bu) hits_dual++;
						else if (f == 0x09u) hits_other++;
						else                 hits_carnival++;  // 0x08 carnival-only (or any unexpected nonzero)
						const uint32_t data = s + (uint32_t)i;
						if (hit_samples.size() < (size_t)args.sweep_hit_print) hit_samples.emplace_back(data, f);
						if (hit_out.is_open())
							hit_out << "0x" << std::hex << std::setw(8) << std::setfill('0') << data
							        << "\t0x" << std::setw(2) << (uint32_t)f << std::dec << std::setfill(' ') << "\n";
					}
					if (tile_mism && first_mism_tile == ntiles) first_mism_tile = t;
					total_mism += tile_mism;
					std::cout << "  tile " << (t+1) << "/" << ntiles << "  comp "
					          << std::fixed << std::setprecision(1) << comp_total_ms/1000.0 << "s  tile_frontier "
					          << tile_frontier << "  mism " << total_mism << "\n" << std::flush;
				}
				const double comp_s = comp_total_ms/1000.0, screen_s = screen_total_ms/1000.0;
				std::cout << "\n  compaction total : " << std::fixed << std::setprecision(2) << comp_s << " s   "
				          << std::setprecision(0) << (double)TOTAL/comp_s << " c/s\n";
				std::cout << "  ilp6 screen total: " << std::fixed << std::setprecision(2) << screen_s << " s   "
				          << std::setprecision(0) << (double)TOTAL/screen_s << " c/s\n";
				std::cout << "  speedup (full key): " << std::fixed << std::setprecision(3) << screen_s/comp_s << "x\n";
				std::cout << "  aggregate final frontier: " << total_final_frontier << "/" << TOTAL
				          << "  R=" << std::fixed << std::setprecision(3)
				          << ((total_final_frontier > 0ull) ? (static_cast<double>(TOTAL) / static_cast<double>(total_final_frontier)) : 0.0)
				          << "\n";
				std::cout << "  screen hits (all data values): " << hits_total
				          << "  [carnival=" << hits_carnival << " other-world=" << hits_other << " dual=" << hits_dual << "]"
				          << "  other-world-reachable=" << (hits_other + hits_dual) << "\n";
				if (hit_out.is_open()) { hit_out.close(); std::cout << "  wrote all " << hits_total << " hits to " << args.sweep_hit_file << "\n"; }
				if (!hit_samples.empty())
				{
					std::cout << "  first " << hit_samples.size() << " hits (data:flag):";
					for (auto& h : hit_samples)
						std::cout << " 0x" << std::hex << std::setw(8) << std::setfill('0') << h.first
						          << ":0x" << std::setw(2) << (uint32_t)h.second << std::dec << std::setfill(' ');
					std::cout << "\n";
				}
				std::cout << (total_mism==0 ? "  PASS - all 2^32 flags match the screen across every tile\n"
				             : ("  FAIL - " + std::to_string(total_mism) + " mismatches; first in tile " + std::to_string(first_mism_tile) + "\n"));
				cuMemFree(buf_ref); cuMemFree(state); cuMemFree(rep_global); if (mult) cuMemFree(mult); cuMemFree(alive);
				cuMemFree(live_a); cuMemFree(live_b); cuMemFree(counter); cuMemFree(flag); cuMemFree(buf_comp);
				if (g_table_fp) cuMemFree(g_table_fp);
				if (g_table_rep) cuMemFree(g_table_rep);
				return 0;
			}

			run_pipeline(false);  // warm-up
			auto ct0 = std::chrono::steady_clock::now();
			uint32_t M_final = run_pipeline(true);
			auto ct1 = std::chrono::steady_clock::now();
			const double comp_ms = std::chrono::duration<double,std::milli>(ct1-ct0).count();

			std::vector<uint8_t> ref(N), comp(N);
			check_cuda(cuMemcpyDtoH(ref.data(),  buf_ref,  N), "copy ref");
			check_cuda(cuMemcpyDtoH(comp.data(), buf_comp, N), "copy comp");
			size_t mism = 0, firstm = 0;
			for (size_t i = 0; i < N; i++) if (ref[i] != comp[i]) { if (!mism) firstm = i; mism++; }

			const double ref_cps  = (double)N / (ref_ms  / 1000.0);
			const double comp_cps = (double)N / (comp_ms / 1000.0);
			std::cout << "Compaction architecture: " << N << " candidates, key=0x"
			          << std::hex << std::setw(8) << std::setfill('0') << args.key_id << std::dec << std::setfill(' ')
			          << " data_start=0x" << std::hex << (uint32_t)args.range_start << std::dec
			          << "  geom=" << args.compaction_ilp << " (W=" << W << ", " << SPAN_BLOCK << "t)"
			          << " spans=" << NSPANS << "\n";
			std::cout << "  ilp6 screen  : " << std::fixed << std::setprecision(3) << ref_ms  << " ms   " << std::setprecision(0) << ref_cps  << " c/s\n";
			std::cout << "  compaction   : " << std::fixed << std::setprecision(3) << comp_ms << " ms   " << std::setprecision(0) << comp_cps << " c/s\n";
			std::cout << "  compaction vs ilp6 screen: " << std::fixed << std::setprecision(3) << (ref_ms / comp_ms) << "x\n";
			std::cout << "  overall R: " << std::fixed << std::setprecision(3) << ((M_final>0)?((double)N/(double)M_final):0.0)
			          << "   final survivors: " << M_final << "/" << N << "\n";
				if (use_routed_span0)
				{
					unsigned long long rs[2] = {0,0};
					check_cuda(cuMemcpyDtoH(rs, route_stats, 2ull * sizeof(unsigned long long)), "read route_stats");
					const double tot = (double)(rs[0] + rs[1]);
					std::cout << "  routed MAP1 (tau=" << args.map1_route_tau << "): hashed " << rs[0]
					          << "  passed " << rs[1] << "  (" << std::fixed << std::setprecision(1)
					          << (tot > 0 ? 100.0 * (double)rs[1] / tot : 0.0) << "% passed un-hashed -> cap holds only the hashed subset)\n";
				}
				if (use_deep_traj && route_stats)
				{
					unsigned long long rs[4] = {0,0,0,0};
					check_cuda(cuMemcpyDtoH(rs, route_stats, 4ull * sizeof(unsigned long long)), "read route_stats (deep)");
					const double tot = (double)(rs[0] + rs[1]);
					std::cout << "  deep-trajgate (all deep drains): hashed " << rs[0] << "  passed " << rs[1]
					          << "  (" << std::fixed << std::setprecision(1)
					          << (tot > 0 ? 100.0 * (double)rs[1] / tot : 0.0) << "% passed un-hashed -> deep cap holds only the routed subset)"
					          << "  sticky-saved " << rs[2]
					          << "  mult-saved " << rs[3]
					          << " (routed despite low density)\n";
				}
				if (occ_counter && (need_g_table || use_routed_span0 || use_cap))
					{
						instrument_occ = true; occ_records.clear();
						run_pipeline(false);  // untimed pass: records active table occupancy
						instrument_occ = false;
						std::cout << "  table occupancy:\n";
						for (auto& r : occ_records)
						{
							const double mb_used = (double)r.occupied * (double)r.bytes_per_slot / 1048576.0;
							const double mb_alloc = (double)r.slots * (double)r.bytes_per_slot / 1048576.0;
							const double lf = r.slots ? (double)r.occupied / (double)r.slots : 0.0;
							std::cout << "    " << r.label << ": occupied " << r.occupied << "/" << r.slots
							          << " slots = " << std::fixed << std::setprecision(1) << mb_used << "/"
							          << mb_alloc << " MB  (LF " << std::setprecision(3) << lf << ")\n";
						}
						if (occ_counter) cuMemFree(occ_counter);
					}
				std::cout << "  frontier:";
			for (uint32_t sp = 0u; sp <= NSPANS; sp++)
				std::cout << " " << (sp?("m"+std::to_string(cuts[sp])+"="):"N=") << span_M[sp];
			std::cout << "\n";
			std::cout << (mism==0 ? "  PASS — all flag bytes match the ilp6 screen\n"
			             : ("  FAIL — " + std::to_string(mism) + " mismatches; first @ " + std::to_string(firstm) + "\n"));

			cuMemFree(buf_ref); cuMemFree(state); cuMemFree(rep_global); if (mult) cuMemFree(mult); cuMemFree(alive);
			cuMemFree(live_a); cuMemFree(live_b); cuMemFree(counter); cuMemFree(flag); cuMemFree(buf_comp);
				if (g_table_fp) cuMemFree(g_table_fp);
				if (g_table_rep) cuMemFree(g_table_rep);
				if (cap_table) cuMemFree(cap_table);
				if (cap_fp) cuMemFree(cap_fp);
				if (cap_rep) cuMemFree(cap_rep);
				if (m1_cap) cuMemFree(m1_cap);
				if (m1_fp) cuMemFree(m1_fp);
				if (m1_rep) cuMemFree(m1_rep);
				if (route_stats) cuMemFree(route_stats);
				if (traj_sketch) cuMemFree(traj_sketch);
				if (traj_keybuf) cuMemFree(traj_keybuf);
			return 0;
		}

		// ---- Screen-dedup validation + bench (Phase 2.5 production-shape) ----
		// Compares tm_checksum_screen_dedup_w32_cuda against the production
		// tm_checksum_screen_cuda. Both produce 1-byte-per-candidate flags.
		// Validates flag-by-flag equality and reports throughput for both.
		if (args.screen_dedup && args.parity_count > 0)
		{
			const uint32_t N = args.parity_count;
			CUdeviceptr buf_ref = 0;
			CUdeviceptr buf_dedup = 0;
			check_cuda(cuMemAlloc(&buf_ref,   N), "cuMemAlloc(screen_ref)");
			check_cuda(cuMemAlloc(&buf_dedup, N), "cuMemAlloc(screen_dedup)");
			uint32_t s = static_cast<uint32_t>(args.range_start);
			uint32_t n = N;

			void* ref_args[] = {
				&buf_ref, &assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
				&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
				&assets.alg2_values, &assets.alg5_values, &assets.expansion_values,
				&assets.schedule_data, &assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id), &s, &n
			};
			void* dedup_args[] = {
				&buf_dedup, &assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
				&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
				&assets.alg2_values, &assets.alg5_values, &assets.expansion_values,
				&assets.schedule_data, &assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id), &s, &n
			};
			void* offset_dedup_args[] = {
				&buf_dedup,
				&offset_regular,
				&offset_alg0,
				&offset_alg6,
				&offset_alg2,
				&offset_alg5,
				&assets.expansion_values,
				&assets.schedule_data,
				&assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id),
				&s,
				&n
			};

			// Warm-up + timed for both.
			float ref_ms = 0.0f, dedup_ms = 0.0f;
			check_cuda(cuLaunchKernel(screen_kernel, screen_kernel_grid_x(n), 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, ref_args, nullptr), "warmup(screen_ref)");
			check_cuda(cuStreamSynchronize(stream), "sync");
			check_cuda(cuEventRecord(kernel_start, stream), "rec start");
			check_cuda(cuLaunchKernel(screen_kernel, screen_kernel_grid_x(n), 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, ref_args, nullptr), "launch(screen_ref)");
			check_cuda(cuEventRecord(kernel_end, stream), "rec end");
			check_cuda(cuStreamSynchronize(stream), "sync");
			check_cuda(cuEventElapsedTime(&ref_ms, kernel_start, kernel_end), "elapsed");

			uint32_t dedup_grid = (n + 31u) / 32u;
			uint32_t dedup_block = 256u;
			CUfunction dedup_kernel = screen_dedup_w32_kernel;
			if      (args.screen_dedup_offsets) dedup_kernel = screen_dedup_w32_offset_kernel;
			else if (args.screen_dedup_maprng_preext_cstore) dedup_kernel = screen_dedup_w32_maprng_preext_cstore_kernel;
			else if (args.screen_dedup_maprng_preext_1sync) dedup_kernel = screen_dedup_w32_maprng_preext_1sync_kernel;
			else if (args.screen_dedup_maprng_preext) dedup_kernel = screen_dedup_w32_maprng_preext_kernel;
			else if (args.screen_dedup_maprng)   dedup_kernel = screen_dedup_w32_maprng_kernel;
			else if (args.screen_dedup_fasthash) dedup_kernel = screen_dedup_w32_fasthash_kernel;
			else if (args.screen_dedup_packed)   dedup_kernel = screen_dedup_w32_packed_kernel;
			else if (args.screen_dedup_k == 2u)  dedup_kernel = screen_dedup_w32_k2_kernel;
			else if (args.screen_dedup_k == 3u)  dedup_kernel = screen_dedup_w32_k3_kernel;
			else if (args.screen_dedup_k == 6u)  dedup_kernel = screen_dedup_w32_k6_kernel;
			// maprng variant has a different arg list (no RNG tables, has map_rng buffer).
			void* maprng_dedup_args[] = {
				&buf_dedup, &assets.expansion_values, &assets.schedule_data, &assets.carnival_data,
					&map_rng_buffer,
					const_cast<uint32_t*>(&args.key_id), &s, &n
				};
				// dedup-schedule experiment kernels have an extra survivor_count param
				// (within-block unique-state counter) inserted before key.
				CUdeviceptr buf_survivors = 0;
				check_cuda(cuMemAlloc(&buf_survivors, sizeof(unsigned long long)), "cuMemAlloc(survivors)");
				void* preext_sched_args[] = {
					&buf_dedup, &assets.expansion_values, &assets.schedule_data, &assets.carnival_data,
					&map_rng_buffer, &buf_survivors,
					const_cast<uint32_t*>(&args.key_id), &s, &n
				};
				// ILP-geometry offset dedup: offset streams + survivor_count, custom grid/block.
				void* offset_geom_args[] = {
					&buf_dedup, &offset_regular, &offset_alg0, &offset_alg6, &offset_alg2, &offset_alg5,
					&assets.expansion_values, &assets.schedule_data, &assets.carnival_data, &buf_survivors,
					const_cast<uint32_t*>(&args.key_id), &s, &n
				};
				if (!args.preext_policy.empty()) dedup_kernel = preext_policy_kernel;
				if (!args.offset_geom.empty())
				{
					dedup_kernel = offset_geom_kernel;
					const uint32_t W = offset_geom_warps * offset_geom_ilp;
					dedup_block = offset_geom_warps * 32u;
					dedup_grid = (n + W - 1u) / W;
				}
				void** launch_args = !args.offset_geom.empty() ? offset_geom_args
					: (!args.preext_policy.empty() ? preext_sched_args
					: (args.screen_dedup_offsets ? offset_dedup_args : ((args.screen_dedup_maprng || args.screen_dedup_maprng_preext) ? maprng_dedup_args : dedup_args)));
			check_cuda(cuLaunchKernel(dedup_kernel, dedup_grid, 1, 1, dedup_block, 1, 1, 0, stream, launch_args, nullptr), "warmup(screen_dedup)");
			check_cuda(cuStreamSynchronize(stream), "sync");
			const unsigned long long zero_u64 = 0ull;
			check_cuda(cuMemcpyHtoD(buf_survivors, &zero_u64, sizeof(zero_u64)), "zero survivors");  // discard warmup's count
			check_cuda(cuEventRecord(kernel_start, stream), "rec start");
			check_cuda(cuLaunchKernel(dedup_kernel, dedup_grid, 1, 1, dedup_block, 1, 1, 0, stream, launch_args, nullptr), "launch(screen_dedup)");
			check_cuda(cuEventRecord(kernel_end, stream), "rec end");
			check_cuda(cuStreamSynchronize(stream), "sync");
			check_cuda(cuEventElapsedTime(&dedup_ms, kernel_start, kernel_end), "elapsed");
			unsigned long long survivors = 0ull;
			if (!args.preext_policy.empty() || !args.offset_geom.empty())
			{
				check_cuda(cuMemcpyDtoH(&survivors, buf_survivors, sizeof(survivors)), "copy survivors");
			}
			cuMemFree(buf_survivors);

			std::vector<uint8_t> ref(N), dedup(N);
			check_cuda(cuMemcpyDtoH(ref.data(),   buf_ref,   N), "copy ref");
			check_cuda(cuMemcpyDtoH(dedup.data(), buf_dedup, N), "copy dedup");
			cuMemFree(buf_ref);
			cuMemFree(buf_dedup);

			std::size_t mismatches = 0;
			std::size_t first_mismatch = 0;
			for (std::size_t i = 0; i < N; i++)
			{
				if (ref[i] != dedup[i])
				{
					if (mismatches == 0) first_mismatch = i;
					mismatches++;
				}
			}

			const double ref_cps   = static_cast<double>(N) / (ref_ms   / 1000.0);
			const double dedup_cps = static_cast<double>(N) / (dedup_ms / 1000.0);
			// Also time the 256-thread variant of the screen kernel (no dedup).
			// This isolates "256-thread block geometry" from "dedup work" so we
			// can tell if the dedup tax is the bottleneck vs the block size.
			float screen256_ms = 0.0f;
			{
				CUdeviceptr buf_screen256 = 0;
				check_cuda(cuMemAlloc(&buf_screen256, N), "cuMemAlloc(screen256)");
				void* args256[] = {
					&buf_screen256, &assets.regular_rng_values, &assets.alg0_values, &assets.alg6_values,
					&assets.rng_seed_forward_1, &assets.rng_seed_forward_128,
					&assets.alg2_values, &assets.alg5_values, &assets.expansion_values,
					&assets.schedule_data, &assets.carnival_data,
					const_cast<uint32_t*>(&args.key_id), &s, &n
				};
				const uint32_t grid_256 = (n + 31u) / 32u;  // 256 threads = 8 warps × 4 ILP = 32 cand/block
				check_cuda(cuLaunchKernel(screen_kernel_256, grid_256, 1, 1, 256, 1, 1, 0, stream, args256, nullptr), "warmup(screen256)");
				check_cuda(cuStreamSynchronize(stream), "sync");
				check_cuda(cuEventRecord(kernel_start, stream), "rec start");
				check_cuda(cuLaunchKernel(screen_kernel_256, grid_256, 1, 1, 256, 1, 1, 0, stream, args256, nullptr), "launch(screen256)");
				check_cuda(cuEventRecord(kernel_end, stream), "rec end");
				check_cuda(cuStreamSynchronize(stream), "sync");
				check_cuda(cuEventElapsedTime(&screen256_ms, kernel_start, kernel_end), "elapsed");
				cuMemFree(buf_screen256);
			}
			const double screen256_cps = static_cast<double>(N) / (screen256_ms / 1000.0);

			std::cout << "Screen vs Screen+Dedup-W32 (K=" << args.screen_dedup_k << "): " << N << " candidates, key=0x"
			          << std::hex << std::setw(8) << std::setfill('0') << args.key_id << std::dec << std::setfill(' ')
			          << " data_start=0x" << std::hex << static_cast<uint32_t>(args.range_start) << std::dec << "\n";
			std::cout << "  screen_128t  :  " << std::fixed << std::setprecision(3) << ref_ms       << " ms   "
			          << std::fixed << std::setprecision(0) << ref_cps       << " c/s\n";
			std::cout << "  screen_256t  :  " << std::fixed << std::setprecision(3) << screen256_ms << " ms   "
			          << std::fixed << std::setprecision(0) << screen256_cps << " c/s   (no dedup)\n";
			std::cout << "  screen_dedup :  " << std::fixed << std::setprecision(3) << dedup_ms     << " ms   "
			          << std::fixed << std::setprecision(0) << dedup_cps     << " c/s\n";
			std::cout << "  dedup vs 128t screen: " << std::fixed << std::setprecision(3) << (ref_ms / dedup_ms)       << "x\n";
			std::cout << "  dedup vs 256t screen: " << std::fixed << std::setprecision(3) << (screen256_ms / dedup_ms) << "x  (isolates dedup tax)\n";
			std::cout << "  256t vs 128t screen : " << std::fixed << std::setprecision(3) << (ref_ms / screen256_ms)   << "x  (isolates block size)\n";
			if (!args.preext_policy.empty() || !args.offset_geom.empty())
			{
				const double block_local_R = survivors > 0ull ? static_cast<double>(N) / static_cast<double>(survivors) : 0.0;
				std::cout << "  policy: " << (args.offset_geom.empty() ? ("preext-" + args.preext_policy) : ("offset-" + args.offset_geom))
				          << "   within-block unique survivors: " << survivors << "/" << N
				          << "   block-local R: " << std::fixed << std::setprecision(3) << block_local_R << "\n";
			}
			if (mismatches == 0)
			{
				std::cout << "  PASS — all " << N << " flag bytes match\n";
			}
			else
			{
				std::cout << "  FAIL — " << mismatches << "/" << N << " mismatches; first at candidate "
				          << first_mismatch << " ref=0x" << std::hex << (unsigned)ref[first_mismatch]
				          << " dedup=0x" << (unsigned)dedup[first_mismatch] << std::dec << "\n";
			}
			return 0;
		}

		// ---- Parity test ----
		if (args.parity_count > 0)
		{
			const uint32_t N = args.parity_count;
			const std::vector<uint8_t> schedule_blob = build_schedule_blob(args.key_id, args.map_list);

			CUdeviceptr gpu_dump = 0;
			check_cuda(cuMemAlloc(&gpu_dump, static_cast<std::size_t>(N) * 128u), "cuMemAlloc(parity_dump)");
			CUdeviceptr gpu_rep = 0;
			if (args.dedup)
			{
				check_cuda(cuMemAlloc(&gpu_rep, static_cast<std::size_t>(N) * sizeof(uint32_t)), "cuMemAlloc(parity_rep)");
			}

			uint32_t parity_start = static_cast<uint32_t>(args.range_start);
			uint32_t parity_n = N;

			// Time the kernel using CUDA events. Run a warm-up pass first to
			// dodge first-launch overhead, then take the timed measurement.
			float kernel_ms = 0.0f;
			if (args.dedup)
			{
				CUfunction dedup_kernel = nullptr;
				uint32_t dedup_block_threads = 1024u;
				if (args.dedup_skip)
				{
					if      (args.dedup_width == 32u)  { dedup_kernel = dump_dedup_w32_skip_kernel;  dedup_block_threads = 256u;  }
					else if (args.dedup_width == 64u)  { dedup_kernel = dump_dedup_w64_skip_kernel;  dedup_block_threads = 512u;  }
					else                                 { dedup_kernel = dump_dedup_w128_skip_kernel; dedup_block_threads = 1024u; }
				}
				else
				{
					dedup_kernel = (args.dedup_width == 128u) ? dump_dedup_w128_kernel : dump_dedup_w32_kernel;
					dedup_block_threads = 1024u;  // no-skip kernels are always 1024 threads
				}
				const uint32_t dedup_grid_x = (parity_n + args.dedup_width - 1u) / args.dedup_width;
				void* dedup_args[] = {
					&gpu_dump,
					&gpu_rep,
					&assets.regular_rng_values,
					&assets.alg0_values,
					&assets.alg6_values,
					&assets.rng_seed_forward_1,
					&assets.rng_seed_forward_128,
					&assets.alg2_values,
					&assets.alg5_values,
					&assets.expansion_values,
					&assets.schedule_data,
					const_cast<uint32_t*>(&args.key_id),
					&parity_start,
					&parity_n
				};
				// Warm-up
				check_cuda(cuLaunchKernel(dedup_kernel, dedup_grid_x, 1, 1, dedup_block_threads, 1, 1, 0, stream, dedup_args, nullptr), "cuLaunchKernel(dump_dedup warmup)");
				check_cuda(cuStreamSynchronize(stream), "cuStreamSynchronize(warmup)");
				// Timed
				check_cuda(cuEventRecord(kernel_start, stream), "cuEventRecord(start)");
				check_cuda(cuLaunchKernel(dedup_kernel, dedup_grid_x, 1, 1, dedup_block_threads, 1, 1, 0, stream, dedup_args, nullptr), "cuLaunchKernel(dump_dedup)");
				check_cuda(cuEventRecord(kernel_end, stream), "cuEventRecord(end)");
				check_cuda(cuStreamSynchronize(stream), "cuStreamSynchronize(dump_dedup)");
				check_cuda(cuEventElapsedTime(&kernel_ms, kernel_start, kernel_end), "cuEventElapsedTime");
			}
			else
			{
				void* dump_args[] = {
					&gpu_dump,
					&assets.regular_rng_values,
					&assets.alg0_values,
					&assets.alg6_values,
					&assets.rng_seed_forward_1,
					&assets.rng_seed_forward_128,
					&assets.alg2_values,
					&assets.alg5_values,
					&assets.expansion_values,
					&assets.schedule_data,
					const_cast<uint32_t*>(&args.key_id),
					&parity_start,
					&parity_n
				};
				check_cuda(cuLaunchKernel(dump_kernel, state_kernel_grid_x(parity_n), 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, dump_args, nullptr), "cuLaunchKernel(dump warmup)");
				check_cuda(cuStreamSynchronize(stream), "cuStreamSynchronize(warmup)");
				check_cuda(cuEventRecord(kernel_start, stream), "cuEventRecord(start)");
				check_cuda(cuLaunchKernel(dump_kernel, state_kernel_grid_x(parity_n), 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, dump_args, nullptr), "cuLaunchKernel(dump)");
				check_cuda(cuEventRecord(kernel_end, stream), "cuEventRecord(end)");
				check_cuda(cuStreamSynchronize(stream), "cuStreamSynchronize(dump)");
				check_cuda(cuEventElapsedTime(&kernel_ms, kernel_start, kernel_end), "cuEventElapsedTime");
			}
			const double kernel_seconds = kernel_ms / 1000.0;
			const double cps = parity_n / kernel_seconds;

			std::vector<uint8_t> gpu_state(static_cast<std::size_t>(parity_n) * 128u);
			check_cuda(cuMemcpyDtoH(gpu_state.data(), gpu_dump, gpu_state.size()), "cuMemcpyDtoH(parity_dump)");
			cuMemFree(gpu_dump);

			std::vector<uint32_t> dedup_rep;
			if (args.dedup)
			{
				dedup_rep.resize(parity_n);
				check_cuda(cuMemcpyDtoH(dedup_rep.data(), gpu_rep, dedup_rep.size() * sizeof(uint32_t)), "cuMemcpyDtoH(parity_rep)");
				cuMemFree(gpu_rep);
			}

			// In skip mode, dead candidates have stale final state in gpu_state.
			// Resolve their state by following the (block-local) rep chain to the
			// alive representative whose state IS correct.
			if (args.dedup && args.dedup_skip)
			{
				const std::size_t W = args.dedup_width;
				std::vector<uint8_t> resolved(gpu_state.size());
				for (std::size_t c = 0; c < parity_n; c++)
				{
					const std::size_t block = c / W;
					const std::size_t local = c % W;
					std::size_t cur_local = local;
					// Follow rep chain within the block — capped at W hops.
					for (uint32_t hop = 0; hop < W; hop++)
					{
						const uint32_t rep_local = dedup_rep[block * W + cur_local];
						if (rep_local >= W || rep_local == cur_local) break;
						cur_local = rep_local;
					}
					const std::size_t src = block * W + cur_local;
					std::memcpy(&resolved[c * 128u], &gpu_state[src * 128u], 128u);
				}
				gpu_state.swap(resolved);
			}

			uint32_t mismatch_count = 0;
			uint32_t first_mismatch_candidate = 0;
			int      first_mismatch_byte = -1;
			uint8_t  first_cpu_byte = 0;
			uint8_t  first_gpu_byte = 0;

			std::vector<uint8_t> cpu_state(128);
			for (uint32_t c = 0; c < parity_n; c++)
			{
					cpu_forward(args.key_id, static_cast<uint32_t>(args.range_start + c), schedule_blob, cpu_state.data());

				const uint8_t* gpu_c = gpu_state.data() + static_cast<std::size_t>(c) * 128u;
				for (int b = 0; b < 128; b++)
				{
					if (cpu_state[b] != gpu_c[b])
					{
						if (mismatch_count == 0)
						{
							first_mismatch_candidate = c;
							first_mismatch_byte      = b;
							first_cpu_byte           = cpu_state[b];
							first_gpu_byte           = gpu_c[b];
						}
						mismatch_count++;
						break;   // count one mismatch per candidate
					}
				}
			}

			std::cout << "Parity test (";
			if (args.dedup) std::cout << "dedup W=" << args.dedup_width
				<< (args.dedup_skip ? " + SKIP" : " no-skip") << " per block";
			else            std::cout << "baseline dump_kernel";
			std::cout << "): " << parity_n << " candidates  key=0x"
			          << std::hex << std::setw(8) << std::setfill('0') << args.key_id
			          << "  data_start=0x" << std::setw(8) << static_cast<uint32_t>(args.range_start) << std::dec << "\n";
			std::cout << "  kernel_ms: " << std::fixed << std::setprecision(3) << kernel_ms
			          << "    candidates/s: " << std::fixed << std::setprecision(0) << cps << "\n";
			if (mismatch_count == 0)
			{
				std::cout << "  PASS — all " << parity_n << " candidates match CPU\n";
			}
			else
			{
				std::cout << "  FAIL — " << mismatch_count << "/" << N << " candidates have mismatches\n";
				std::cout << "  First mismatch: candidate " << first_mismatch_candidate
				          << "  byte[" << first_mismatch_byte << "]"
				          << "  cpu=0x" << std::hex << static_cast<unsigned>(first_cpu_byte)
				          << "  gpu=0x" << static_cast<unsigned>(first_gpu_byte) << std::dec << "\n";
			}

			if (args.dedup)
			{
				// Per-block dedup metadata: count unique reps per W-wide block.
				std::size_t total_unique = 0;
				std::size_t total_dupes = 0;
				const std::size_t W = args.dedup_width;
				const std::size_t n_blocks = (parity_n + W - 1u) / W;
				std::vector<bool> seen(W, false);
				for (std::size_t b = 0; b < n_blocks; b++)
				{
					std::fill(seen.begin(), seen.end(), false);
					const std::size_t base = b * W;
					const std::size_t end = std::min<std::size_t>(base + W, parity_n);
					for (std::size_t i = base; i < end; i++)
					{
						const uint32_t r = dedup_rep[i];
						if (r < W) seen[r] = true;
					}
					std::size_t u = 0;
					for (std::size_t k = 0; k < W; k++) if (seen[k]) u++;
					total_unique += u;
					total_dupes  += (end - base) - u;
				}
				const double avg_unique = static_cast<double>(total_unique) / static_cast<double>(n_blocks);
				const double compression = total_unique > 0
					? static_cast<double>(total_unique + total_dupes) / static_cast<double>(total_unique)
					: 0.0;
				std::cout << "  dedup metadata at final boundary:\n"
				          << "    blocks:           " << n_blocks << "\n"
				          << "    avg unique reps:  " << std::fixed << std::setprecision(2) << avg_unique << " / " << W << "\n"
				          << "    total dupes:      " << total_dupes << " (out of " << parity_n << ")\n"
				          << "    compression R:    " << std::fixed << std::setprecision(2) << compression << "x at W=" << W << "\n";
			}
			return 0;
		}
		// ---- End parity test ----

		double warmup_seconds = 0.0;
		for (uint32_t i = 0; i < args.warmup_batches; i++)
		{
			uint32_t warmup_start = static_cast<uint32_t>(args.range_start + (static_cast<uint64_t>(i) * args.batch_size));
			uint32_t warmup_count = args.batch_size;
			const auto warmup_begin = std::chrono::high_resolution_clock::now();

			void* kernel_args[] = {
				&result_buffer,
				&assets.regular_rng_values,
				&assets.alg0_values,
				&assets.alg6_values,
				&assets.rng_seed_forward_1,
				&assets.rng_seed_forward_128,
				&assets.alg2_values,
				&assets.alg5_values,
				&assets.expansion_values,
				&assets.schedule_data,
				&assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id),
				&warmup_start,
				&warmup_count
			};
			void* kernel_args_maprng[] = {
				&result_buffer,
				&assets.expansion_values,
				&assets.schedule_data,
				&assets.carnival_data,
				&map_rng_buffer,
				const_cast<uint32_t*>(&args.key_id),
				&warmup_start,
				&warmup_count
			};
			void* kernel_args_offsets[] = {
				&result_buffer,
				&offset_regular,
				&offset_alg0,
				&offset_alg6,
				&offset_alg2,
				&offset_alg5,
				&assets.expansion_values,
				&assets.schedule_data,
				&assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id),
				&warmup_start,
				&warmup_count
			};
			void* kernel_args_offsets_interleaved[] = {
				&result_buffer,
				&offset_stream_buffer,
				&assets.expansion_values,
				&assets.schedule_data,
				&assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id),
				&warmup_start,
				&warmup_count
			};

			CUfunction warmup_kernel = args.screen_offsets_interleaved ? screen_offset_interleaved_kernel : (args.screen_offsets ? screen_offset_ilp8_preids_kernel : ((args.maprng || args.screen_preext) ? (args.screen_preext ? screen_maprng_preext_kernel : screen_maprng_kernel) : screen_kernel));
			void** warmup_args = args.screen_offsets_interleaved ? kernel_args_offsets_interleaved : (args.screen_offsets ? kernel_args_offsets : ((args.maprng || args.screen_preext) ? kernel_args_maprng : kernel_args));
			const uint32_t warmup_grid = (args.screen_offsets && !args.screen_offsets_interleaved) ? screen_kernel_grid_x_ilp(warmup_count, 8u) : screen_kernel_grid_x(warmup_count);
			check_cuda(cuLaunchKernel(warmup_kernel, warmup_grid, 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, warmup_args, nullptr), "cuLaunchKernel(warmup)");
			check_cuda(cuStreamSynchronize(stream), "cuStreamSynchronize(warmup)");
			check_cuda(cuMemcpyDtoH(host_results.data(), result_buffer, args.batch_size), "cuMemcpyDtoH(warmup)");

			const auto warmup_end = std::chrono::high_resolution_clock::now();
			warmup_seconds += std::chrono::duration<double>(warmup_end - warmup_begin).count();
		}

		uint64_t carnival_hits = 0;
		uint64_t other_hits = 0;
		uint64_t dual_hits = 0;
		double screen_kernel_seconds = 0.0;
		double materialize_kernel_seconds = 0.0;
		const auto wall_begin = std::chrono::high_resolution_clock::now();

		for (uint64_t offset = 0; offset < args.workunit_size; offset += args.batch_size)
		{
			uint64_t chunk = args.workunit_size - offset;
			if (chunk > args.batch_size)
			{
				chunk = args.batch_size;
			}

			const uint32_t data_start = static_cast<uint32_t>(args.range_start + offset);
			uint32_t chunk_u32 = static_cast<uint32_t>(chunk);

			void* kernel_args_nohll[] = {
				&result_buffer,
				&assets.regular_rng_values,
				&assets.alg0_values,
				&assets.alg6_values,
				&assets.rng_seed_forward_1,
				&assets.rng_seed_forward_128,
				&assets.alg2_values,
				&assets.alg5_values,
				&assets.expansion_values,
				&assets.schedule_data,
				&assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id),
				const_cast<uint32_t*>(&data_start),
				&chunk_u32
			};
			void* kernel_args_hll[] = {
				&result_buffer,
				&hll_buffer,
				&assets.regular_rng_values,
				&assets.alg0_values,
				&assets.alg6_values,
				&assets.rng_seed_forward_1,
				&assets.rng_seed_forward_128,
				&assets.alg2_values,
				&assets.alg5_values,
				&assets.expansion_values,
				&assets.schedule_data,
				&assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id),
				const_cast<uint32_t*>(&data_start),
				&chunk_u32
			};
			void* kernel_args_maprng_loop[] = {
				&result_buffer,
				&assets.expansion_values,
				&assets.schedule_data,
				&assets.carnival_data,
				&map_rng_buffer,
				const_cast<uint32_t*>(&args.key_id),
				const_cast<uint32_t*>(&data_start),
				&chunk_u32
			};
			void* kernel_args_offsets_loop[] = {
				&result_buffer,
				&offset_regular,
				&offset_alg0,
				&offset_alg6,
				&offset_alg2,
				&offset_alg5,
				&assets.expansion_values,
				&assets.schedule_data,
				&assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id),
				const_cast<uint32_t*>(&data_start),
				&chunk_u32
			};
			void* kernel_args_offsets_interleaved_loop[] = {
				&result_buffer,
				&offset_stream_buffer,
				&assets.expansion_values,
				&assets.schedule_data,
				&assets.carnival_data,
				const_cast<uint32_t*>(&args.key_id),
				const_cast<uint32_t*>(&data_start),
				&chunk_u32
			};

			void** kernel_args;
			CUfunction active_screen;
			if (args.screen_offsets_interleaved)
			{
				kernel_args = kernel_args_offsets_interleaved_loop;
				active_screen = screen_offset_interleaved_kernel;
			}
			else if (args.screen_offsets)
			{
				kernel_args = kernel_args_offsets_loop;
				active_screen = screen_offset_ilp8_preids_kernel;
			}
			else if (args.maprng || args.screen_preext)
			{
				kernel_args   = kernel_args_maprng_loop;
				active_screen = args.screen_preext ? screen_maprng_preext_kernel : screen_maprng_kernel;
			}
			else if (args.hll)
			{
				kernel_args   = kernel_args_hll;
				active_screen = screen_hll_kernel;
			}
			else
			{
				kernel_args   = kernel_args_nohll;
				active_screen = screen_kernel;
			}

			const uint32_t active_grid = (args.screen_offsets && !args.screen_offsets_interleaved) ? screen_kernel_grid_x_ilp(chunk_u32, 8u) : screen_kernel_grid_x(chunk_u32);
			check_cuda(cuEventRecord(kernel_start, stream), "cuEventRecord(start)");
			check_cuda(cuLaunchKernel(active_screen, active_grid, 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, kernel_args, nullptr), "cuLaunchKernel");
			check_cuda(cuEventRecord(kernel_end, stream), "cuEventRecord(end)");
			check_cuda(cuEventSynchronize(kernel_end), "cuEventSynchronize(end)");

			float elapsed_ms = 0.0f;
			check_cuda(cuEventElapsedTime(&elapsed_ms, kernel_start, kernel_end), "cuEventElapsedTime");
			screen_kernel_seconds += static_cast<double>(elapsed_ms) / 1000.0;

			check_cuda(cuMemcpyDtoH(host_results.data(), result_buffer, chunk_u32), "cuMemcpyDtoH");

			for (uint32_t i = 0; i < chunk_u32; i++)
			{
				const uint8_t flags = host_results[i];
				if ((flags & CHECKSUM_SENTINEL) == 0)
				{
					continue;
				}

				if ((flags & OTHER_WORLD) != 0)
				{
					other_hits++;
					if ((flags & DUAL_PASS) != 0)
					{
						dual_hits++;
					}
				}
				else
				{
					carnival_hits++;
				}

				SurvivorRecord survivor = {};
				survivor.data = data_start + i;
				survivor.screen_flags = flags;
				survivors.push_back(survivor);
			}
		}

		ValidationSummary validation_summary = {};
		double cpu_validation_seconds = 0.0;
		if (!survivors.empty())
		{
			std::vector<uint32_t> survivor_data_host(args.batch_size, 0);
			std::vector<uint8_t> survivor_flags_host(args.batch_size, 0);
			std::vector<uint8_t> decrypted_state_host(args.batch_size * 128, 0);

			for (std::size_t offset = 0; offset < survivors.size(); offset += args.batch_size)
			{
				std::size_t chunk = survivors.size() - offset;
				if (chunk > args.batch_size)
				{
					chunk = args.batch_size;
				}

				for (std::size_t i = 0; i < chunk; i++)
				{
					survivor_data_host[i] = survivors[offset + i].data;
					survivor_flags_host[i] = survivors[offset + i].screen_flags;
				}

				check_cuda(cuMemcpyHtoD(survivor_data_buffer, survivor_data_host.data(), chunk * sizeof(uint32_t)), "cuMemcpyHtoD(survivor_data)");
				check_cuda(cuMemcpyHtoD(survivor_flag_buffer, survivor_flags_host.data(), chunk), "cuMemcpyHtoD(survivor_flags)");

				uint32_t chunk_u32 = static_cast<uint32_t>(chunk);
				void* materialize_args[] = {
					&materialized_state_buffer,
					&survivor_data_buffer,
					&survivor_flag_buffer,
					&assets.regular_rng_values,
					&assets.alg0_values,
					&assets.alg6_values,
					&assets.rng_seed_forward_1,
					&assets.rng_seed_forward_128,
					&assets.alg2_values,
					&assets.alg5_values,
					&assets.expansion_values,
					&assets.schedule_data,
					&assets.carnival_data,
					const_cast<uint32_t*>(&args.key_id),
					&chunk_u32
				};

				check_cuda(cuEventRecord(kernel_start, stream), "cuEventRecord(materialize.start)");
				check_cuda(cuLaunchKernel(materialize_kernel, state_kernel_grid_x(chunk_u32), 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, materialize_args, nullptr), "cuLaunchKernel(materialize)");
				check_cuda(cuEventRecord(kernel_end, stream), "cuEventRecord(materialize.end)");
				check_cuda(cuEventSynchronize(kernel_end), "cuEventSynchronize(materialize.end)");

				float elapsed_ms = 0.0f;
				check_cuda(cuEventElapsedTime(&elapsed_ms, kernel_start, kernel_end), "cuEventElapsedTime(materialize)");
				materialize_kernel_seconds += static_cast<double>(elapsed_ms) / 1000.0;

				check_cuda(cuMemcpyDtoH(decrypted_state_host.data(), materialized_state_buffer, chunk * 128ull), "cuMemcpyDtoH(materialized_state)");

				const auto cpu_validation_begin = std::chrono::high_resolution_clock::now();
				for (std::size_t i = 0; i < chunk; i++)
				{
					const bool other_world = (survivor_flags_host[i] & OTHER_WORLD) != 0;
					const bool dual         = (survivor_flags_host[i] & DUAL_PASS)  != 0;
					const uint8_t* state    = &decrypted_state_host[i * 128];

					const uint8_t machine_flags = check_machine_code(state, other_world);
					survivors[offset + i].machine_flags = static_cast<uint8_t>(machine_flags | (other_world ? OTHER_WORLD : 0));
					accumulate_validation_summary(validation_summary, machine_flags, other_world);

					if (dual) validation_summary.dual_pass++;

					if (other_world)
					{
						const OtherPredicateResults predicates = evaluate_other_predicates(state);
						const bool final_rts = other_final_rts_shape(state);
						const bool entry_opcodes = other_entry_opcode_shape(state);
						const bool control_flow = other_control_flow_shape(state);
						const bool structural = entry_opcodes && control_flow;
						const bool all_entries = (machine_flags & ALL_ENTRIES_VALID) != 0;

						if (final_rts) validation_summary.other_final_rts++;
						if (entry_opcodes) validation_summary.other_entry_opcodes++;
						if (control_flow) validation_summary.other_control_flow++;
						if (structural) validation_summary.other_structural++;

						for (int p = 0; p < OTHER_PREDICATE_COUNT; p++)
						{
							if (!predicates.values[p])
							{
								continue;
							}
							validation_summary.other_predicate[p]++;
							if (final_rts)
							{
								validation_summary.final_rts_other_predicate[p]++;
							}
							if (all_entries)
							{
								validation_summary.all_entries_other_predicate[p]++;
								if (final_rts)
								{
									validation_summary.all_entries_final_rts_other_predicate[p]++;
								}
							}
						}

						if (all_entries)
						{
							if (final_rts) validation_summary.all_entries_other_final_rts++;
							if (entry_opcodes) validation_summary.all_entries_other_entry_opcodes++;
							if (control_flow) validation_summary.all_entries_other_control_flow++;
							if (structural) validation_summary.all_entries_other_structural++;
						}
					}

					// Linear stream scan: clean flag + gradient score
					const int lscore = linear_scan_score(state, other_world);
					const bool lclean = linear_scan_clean(state, other_world);
					if (lclean)
					{
						if (other_world) validation_summary.linear_scan_clean_other++;
						else             validation_summary.linear_scan_clean_carnival++;
					}
					if (other_world)
					{
						if (static_cast<uint64_t>(lscore) > validation_summary.linear_scan_score_max_other)
							validation_summary.linear_scan_score_max_other = static_cast<uint64_t>(lscore);
					}
					else
					{
						if (static_cast<uint64_t>(lscore) > validation_summary.linear_scan_score_max_carnival)
							validation_summary.linear_scan_score_max_carnival = static_cast<uint64_t>(lscore);
					}

					// Hash-based unique state counting
					const uint64_t h = hash_state(state);
					if (other_world) validation_summary.unique_states_other    += 1;  // placeholder, set below
					else             validation_summary.unique_states_carnival += 1;  // placeholder, set below
					(void)h;  // suppress unused warning; set is built separately below
				}
				const auto cpu_validation_end = std::chrono::high_resolution_clock::now();
				cpu_validation_seconds += std::chrono::duration<double>(cpu_validation_end - cpu_validation_begin).count();
			}

			// Count distinct decrypted states using hash sets (after all batches are done above)
			{
				std::unordered_set<uint64_t> hashes_carnival;
				std::unordered_set<uint64_t> hashes_other;
				hashes_carnival.reserve(static_cast<std::size_t>(other_hits + carnival_hits));
				hashes_other.reserve(static_cast<std::size_t>(other_hits));

				// Re-materialize in chunks to hash; reuse existing GPU buffers
				for (std::size_t offset = 0; offset < survivors.size(); offset += args.batch_size)
				{
					std::size_t chunk = survivors.size() - offset;
					if (chunk > args.batch_size) chunk = args.batch_size;

					for (std::size_t i = 0; i < chunk; i++)
					{
						survivor_data_host[i]  = survivors[offset + i].data;
						survivor_flags_host[i] = survivors[offset + i].screen_flags;
					}

					check_cuda(cuMemcpyHtoD(survivor_data_buffer,  survivor_data_host.data(),  chunk * sizeof(uint32_t)), "cuMemcpyHtoD(hash_data)");
					check_cuda(cuMemcpyHtoD(survivor_flag_buffer,  survivor_flags_host.data(), chunk),                    "cuMemcpyHtoD(hash_flags)");

					uint32_t chunk_u32 = static_cast<uint32_t>(chunk);
					void* mat_args[] = {
						&materialized_state_buffer,
						&survivor_data_buffer,
						&survivor_flag_buffer,
						&assets.regular_rng_values,
						&assets.alg0_values,
						&assets.alg6_values,
						&assets.rng_seed_forward_1,
						&assets.rng_seed_forward_128,
						&assets.alg2_values,
						&assets.alg5_values,
						&assets.expansion_values,
						&assets.schedule_data,
						&assets.carnival_data,
						const_cast<uint32_t*>(&args.key_id),
						&chunk_u32
					};
					check_cuda(cuLaunchKernel(materialize_kernel, state_kernel_grid_x(chunk_u32), 1, 1, kCudaThreadsPerBlock, 1, 1, 0, stream, mat_args, nullptr), "cuLaunchKernel(hash_mat)");
					check_cuda(cuStreamSynchronize(stream), "cuStreamSynchronize(hash_mat)");
					check_cuda(cuMemcpyDtoH(decrypted_state_host.data(), materialized_state_buffer, chunk * 128ull), "cuMemcpyDtoH(hash_state)");

					for (std::size_t i = 0; i < chunk; i++)
					{
						const bool other_world = (survivor_flags_host[i] & OTHER_WORLD) != 0;
						const uint64_t h = hash_state(&decrypted_state_host[i * 128]);
						if (other_world) hashes_other.insert(h);
						else             hashes_carnival.insert(h);
					}
				}
				// Overwrite the placeholder counts with real unique counts
				validation_summary.unique_states_carnival = hashes_carnival.size();
				validation_summary.unique_states_other    = hashes_other.size();
			}
		}

		const auto wall_end = std::chrono::high_resolution_clock::now();
		const double setup_seconds = std::chrono::duration<double>(setup_end - setup_begin).count();
		const double wall_seconds = std::chrono::duration<double>(wall_end - wall_begin).count();

		if (args.hll)
		{
			std::vector<uint32_t> hll_host(kHllM, 0);
			check_cuda(cuMemcpyDtoH(hll_host.data(), hll_buffer, kHllM * sizeof(uint32_t)), "cuMemcpyDtoH(hll)");
			validation_summary.hll_distinct_states = hll_estimate(hll_host);
			if (!args.hll_dump_path.empty())
			{
				std::ofstream hf(args.hll_dump_path, std::ios::binary | std::ios::trunc);
				if (!hf) throw std::runtime_error("Cannot open HLL dump path: " + args.hll_dump_path);
				hf.write(reinterpret_cast<const char*>(hll_host.data()), kHllM * sizeof(uint32_t));
			}
		}

		if (!args.output_csv_path.empty())
		{
			std::ofstream output_csv(args.output_csv_path.c_str(), std::ios::out | std::ios::trunc);
			if (!output_csv)
			{
				throw std::runtime_error("Could not open output CSV path: " + args.output_csv_path);
			}

			output_csv << "key,data,flags\n";
			for (std::size_t i = 0; i < survivors.size(); i++)
			{
				output_csv << args.key_id << ","
					<< survivors[i].data << ","
					<< static_cast<unsigned int>(survivors[i].machine_flags) << "\n";
			}
		}

		print_summary(args, device_name, setup_seconds, warmup_seconds, screen_kernel_seconds, materialize_kernel_seconds, cpu_validation_seconds, wall_seconds, carnival_hits, other_hits, dual_hits, validation_summary);

		cuEventDestroy(kernel_end);
		cuEventDestroy(kernel_start);
		cuStreamDestroy(stream);
		cuMemFree(hll_buffer);
		cuMemFree(materialized_state_buffer);
		cuMemFree(survivor_flag_buffer);
		cuMemFree(survivor_data_buffer);
		cuMemFree(result_buffer);
		if (map_rng_buffer != 0) cuMemFree(map_rng_buffer);
		release_assets(assets);
		cuFuncSetCacheConfig(materialize_kernel, CU_FUNC_CACHE_PREFER_NONE);
		cuModuleUnload(module);
		cuCtxDestroy(context);
		return 0;
	}
	catch (const std::exception& ex)
	{
		std::cerr << ex.what() << "\n";
		return 1;
	}
}
