// Reference dumper for the VALUE-PARITY gate: emits (seed, nibble, state_in,
// state_out) vectors for run_one_map from the canonical scalar kernel tm_8 --
// the repo's ground-truth implementation. The VHDL parity TB regenerates the
// window from `seed` (via the already-validated rng_gen) and checks
// tm_map_engine(state_in, window, nibble) == state_out, bit for bit. This
// closes the alg2/alg5 derivation + window-indexing + byte-order gate.
//
//   g++ -O2 -I../../common -I../../bruteforce/cpu dump_map.cpp \
//       ../../bruteforce/cpu/tm_8.cpp ../../common/rng.cpp ../../common/rng_obj.cpp \
//       ../../common/tm_base.cpp ../../common/key_schedule.cpp -o dump_map
//   ./dump_map         # writes map_vectors.txt
//
// File format, per vector: 3 lines
//   <SEED hex4> <NIBBLE hex4>
//   <state_in : 256 hex chars (128 bytes, byte 0 first)>
//   <state_out: 256 hex chars>
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

    const uint16 seeds[] = { 0x1234, 0x2CA5, 0xABCD, 0x0001, 0xFFFF, 0x8000, 0x00FF, 0x5A5A };
    const uint16 nibs[]  = { 0x0000, 0xFFFF, 0xA5A5, 0x9E37, 0x1357, 0x8001, 0x4242, 0xC3C3 };
    const int NS = (int)(sizeof(seeds)/sizeof(seeds[0]));
    const int NN = (int)(sizeof(nibs)/sizeof(nibs[0]));
    const int NV = 128;

    FILE* f = fopen("map_vectors.txt", "w");
    if (!f) { perror("map_vectors.txt"); return 1; }

    uint32_t lc = 0x12345678u;
    for (int v = 0; v < NV; v++)
    {
        uint16 S   = seeds[v % NS];
        uint16 nib = nibs[(v / NS) % NN];

        uint8 st[128];
        for (int i = 0; i < 128; i++) { lc = lc * 1103515245u + 12345u; st[i] = (uint8)((lc >> 16) & 0xFF); }

        t.load_data(st);
        key_schedule::key_schedule_entry e;
        e.rng1 = (uint8)((S >> 8) & 0xFF);
        e.rng2 = (uint8)(S & 0xFF);
        e.nibble_selector = nib;
        t.run_one_map(e);

        uint8 out[128];
        t.fetch_data(out);

        fprintf(f, "%04X %04X\n", S, nib);
        for (int i = 0; i < 128; i++) fprintf(f, "%02X", st[i]);  fprintf(f, "\n");
        for (int i = 0; i < 128; i++) fprintf(f, "%02X", out[i]); fprintf(f, "\n");
    }
    fclose(f);
    fprintf(stderr, "wrote map_vectors.txt (%d vectors)\n", NV);
    return 0;
}
