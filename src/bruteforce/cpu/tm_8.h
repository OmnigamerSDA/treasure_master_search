#ifndef TM_8_H
#define TM_8_H
#include "data_sizes.h"
#include "rng_obj.h"
#include "tm_base.h"

class tm_8: public TM_base
{
public:
	tm_8(RNG *rng);

	virtual void load_data(uint8* new_data);
	void fetch_data(uint8* new_data);

	virtual void expand(uint32 key, uint32 data);

	virtual void run_alg(int algorithm_id, uint16 * rng_seed, int iterations);

	virtual void run_one_map(const key_schedule::key_schedule_entry& schedule_entry);

	virtual void run_all_maps(const key_schedule& schedule_entries);

	virtual void decrypt_carnival_world();
	virtual void decrypt_other_world();
	virtual uint16 calculate_carnival_world_checksum();
	virtual uint16 calculate_other_world_checksum();
	virtual uint16 fetch_carnival_world_checksum_value();
	virtual uint16 fetch_other_world_checksum_value();

	void run_bruteforce_data(uint32 key, uint32 start_data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size);

	// N-way interleaved variant: processes NWAY_BATCH candidates in lockstep to
	// overlap cache-miss latency for per-step RNG table loads.
	static constexpr int NWAY_BATCH = 16;
	void run_bruteforce_data_nway(uint32 key, uint32 start_data, const key_schedule& schedule_entries, uint32 amount_to_run, void(*report_progress)(double), uint8* result_data, uint32 result_max_size, uint32* result_size);

	// Read-only accessor over the internal working state. Used by Row A POC
	// in src/bruteforce/row_a_bench/ to extract per-step alg-IDs for a
	// custom run_one_map; not part of TM_base's contract.
	const uint8* working_state() const { return working_code_data; }

	// Raw internal-state accessors for dedup POCs. For tm_8 the internal
	// layout IS canonical, so these are aliases for working_state()/load_data().
	// SIMD impls that store state in a shuffled layout expose their shuffled
	// bytes through the same interface — hashing and replay stay within the
	// impl's native representation, no shuffle/unshuffle in the hot path.
	const uint8* state_raw() const { return working_code_data; }
	void load_state_raw(const uint8* src) { for (int i = 0; i < 128; i++) working_code_data[i] = src[i]; }

private:
	void initialize();

	// Expand key+data into an externally-supplied 128-byte state buffer.
	void expand_into(uint32 key, uint32 data, uint8* state);

	void add_alg(uint8* addition_values, const uint16 rng_seed);
	void xor_alg(uint8* working_data, uint8* xor_values);
	void alg_0(const uint16 rng_seed);
	void alg_1(const uint16 rng_seed);
	void alg_2(const uint16 rng_seed);
	void alg_3(const uint16 rng_seed);
	void alg_4(const uint16 rng_seed);
	void alg_5(const uint16 rng_seed);
	void alg_6(const uint16 rng_seed);
	void alg_7();

	uint16 calculate_masked_checksum(uint8* working_data, uint8* mask);
	uint16 fetch_checksum_value(uint8* working_data, int code_length);

	uint16 _calculate_carnival_world_checksum(uint8* working_data);
	uint16 _calculate_other_world_checksum(uint8* working_data);
	bool check_carnival_world_checksum(uint8* working_data);
	bool check_other_world_checksum(uint8* working_data);
	uint16 _fetch_carnival_world_checksum_value(uint8* working_data);
	uint16 _fetch_other_world_checksum_value(uint8* working_data);

	void _decrypt_carnival_world(uint8* working_data);
	void _decrypt_other_world(uint8* working_data);

	uint8 working_code_data[128 * 2];

	static bool initialized;
};

#endif // TM_8_H