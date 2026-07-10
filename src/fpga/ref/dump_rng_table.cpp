// Reference dumper: emits the ACTUAL repo rng_table (and run_rng outputs) so
// the VHDL rng_step can be validated exhaustively against ground truth rather
// than against a re-transcription. Links the real src/common/rng.cpp.
//
//   g++ -O2 -I../../common dump_rng_table.cpp ../../common/rng.cpp -o dump_rng_table
//   ./dump_rng_table            # writes rng_table.bin (65536 x uint16, LE)
//
// rng_table.bin[seed] = rng_table[seed] = next seed for run_rng.
// run_rng output byte = ((rng_table[seed]>>8) ^ rng_table[seed]) & 0xFF, so the
// table alone validates both nseed and obyte of the VHDL rng_step.
#include <cstdio>
#include "data_sizes.h"
#include "rng.h"

int main()
{
    static uint16 table[0x10000];
    generate_rng_table(table);

    FILE* f = fopen("rng_table.bin", "wb");
    if (!f) { perror("rng_table.bin"); return 1; }
    // little-endian uint16 per entry, seed order 0..65535
    for (int s = 0; s < 0x10000; s++) {
        unsigned char lo = (unsigned char)(table[s] & 0xFF);
        unsigned char hi = (unsigned char)((table[s] >> 8) & 0xFF);
        fputc(lo, f);
        fputc(hi, f);
    }
    fclose(f);

    // text hex (one 4-digit value per line, seed order) for GHDL textio.
    FILE* h = fopen("rng_table.hex", "w");
    if (!h) { perror("rng_table.hex"); return 1; }
    for (int s = 0; s < 0x10000; s++) fprintf(h, "%04X\n", table[s]);
    fclose(h);

    // reference run_rng stream from a fixed seed, for the generator TB.
    // stream[n] = n-th run_rng output byte walking from START_SEED.
    const uint16 START_SEED = 0x1234;
    uint16 seed = START_SEED;
    FILE* sf = fopen("rng_stream_1234.hex", "w");
    if (!sf) { perror("rng_stream_1234.hex"); return 1; }
    for (int n = 0; n < 2048; n++) {
        uint8 b = run_rng(&seed, table);
        fprintf(sf, "%02X\n", b);
    }
    fclose(sf);

    fprintf(stderr, "wrote rng_table.bin + rng_table.hex + rng_stream_1234.hex\n");
    return 0;
}
