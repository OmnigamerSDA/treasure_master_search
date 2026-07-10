// Full-forward fixture for the END-TO-END parity gate: for a fixed key, emits
// the 27-map schedule, the expansion vector, and (data -> post-27-maps state)
// vectors from the canonical scalar kernel tm_8. The VHDL TB reproduces
// expand + 27 maps (rng_gen + map_engine) and checks the 128-byte state, so it
// validates the WHOLE forward chain (not just one map) against ground truth.
//
//   g++ -O2 -I../../common -I../../bruteforce/cpu dump_forward.cpp \
//       ../../bruteforce/cpu/tm_8.cpp ../../common/rng.cpp ../../common/rng_obj.cpp \
//       ../../common/tm_base.cpp ../../common/key_schedule.cpp ../../common/alignment2.cpp -o dump_forward
//   ./dump_forward            # writes forward_vectors.txt
//
// Format:
//   <KEY hex8> <NMAPS dec>
//   NMAPS lines: <seed hex4> <nibble hex4>
//   <exp_vec : 256 hex chars>
//   <NDATA dec>
//   NDATA * { <data hex8> ; <state_out 256 hex> }
#include <cstdio>
#include <cstdint>
#include "data_sizes.h"
#include "rng_obj.h"
#include "tm_8.h"
#include "key_schedule.h"

int main()
{
    RNG rng;
    tm_8 t(&rng);

    const uint32 KEY = 0x2ca5b42d;
    const uint32 DBASE = 0x00000000;
    const int NDATA = 24;

    key_schedule sched(KEY, key_schedule::ALL_MAPS);
    const int NMAPS = (int)sched.entries.size();

    FILE* f = fopen("forward_vectors.txt", "w");
    if (!f) { perror("forward_vectors.txt"); return 1; }

    fprintf(f, "%08X %d\n", KEY, NMAPS);
    for (int m = 0; m < NMAPS; m++)
    {
        const auto& e = sched.entries[m];
        uint16 seed = (uint16)((e.rng1 << 8) | e.rng2);
        fprintf(f, "%04X %04X\n", seed, e.nibble_selector);
    }

    const uint16 es = (uint16)((KEY >> 16) & 0xFFFF);
    for (int i = 0; i < 128; i++) fprintf(f, "%02X", rng.expansion_values_8[es * 128 + i]);
    fprintf(f, "\n");

    fprintf(f, "%d\n", NDATA);
    for (int v = 0; v < NDATA; v++)
    {
        uint32 data = DBASE + v;
        t.expand(KEY, data);
        t.run_all_maps(sched);
        uint8 st[128];
        t.fetch_data(st);
        fprintf(f, "%08X\n", data);
        for (int i = 0; i < 128; i++) fprintf(f, "%02X", st[i]);
        fprintf(f, "\n");
    }
    fclose(f);
    fprintf(stderr, "wrote forward_vectors.txt (KEY=%08X NMAPS=%d NDATA=%d)\n", KEY, NMAPS, NDATA);
    return 0;
}
