# CPU raceway — bounded-wave forward dedup

A single-host CPU forward search over the full 2³² key window that adapts the GPU
*bounded-wave raceway* (persistent cap-drain + wave-local compaction) to the CPU's two
shared-resource walls: cross-core memory bandwidth and per-core vector-ALU ports.

Each worker thread shards the data axis, originates the MAP1 frontier, packs distinct
representatives into an LLC-sized **wave**, and drains the wave through the span schedule
with all reps advancing lockstep (x10 on AVX-512, x2 on AVX2). A persistent, LLC-resident
**cap** at each boundary depth drops resident duplicates between spans. The cap is
**FN-only**: under pressure it *over-keeps* (merged out by the exact final union), it never
false-drops a distinct final.

## Build

```sh
./raceway_autoconfig.sh            # detect ISA/compiler/topology, print the plan
./raceway_autoconfig.sh --build    # build the target this host should use
```

or pick a build directly:

| target | ISA | interleave | compiler | when |
|---|---|---|---|---|
| `cpu_raceway`      | AVX-512 natmap    | x10 / blmerge | clang≥19 (recommended) | primary, AVX-512 hosts |
| `cpu_raceway_u`    | AVX-512 universal | x12 branched   | g++ | fallback / cross-check |
| `cpu_raceway_avx2` | AVX2 natmap       | x2 / blmerge   | clang≥19 (recommended) | hosts without AVX-512 |

```sh
make cpu_raceway                       # prefers clang++-21 when available
make                 cpu_raceway_u
```

clang≥19 gives +14–22% over g++ on the deep-bound keys for the natmap builds. All targets
build at `-O3` with `-fno-builtin-memset` (roots out an `-O3` vectorized-memset SIGSEGV).
Linux only (peak-RSS reporting reads `/proc` + `<sys/resource.h>`).

## Run

```
./cpu_raceway <key> <window> <T> [K] [count|screen] [wave_N] [cap_bits] [cap_ways] [fp64|fp96|fp128]
```

- `window` = `0` means the full 2³² space. A full-key scan needs the producer cap
  (`PRODUCER_CAP=1 PCAP_BITS=24`) or the MAP1 set grows unbounded.
- Size `wave_N` so the live wave (`2 × wave_N × 128 B`) fits the LLC — `131072` (~32 MiB/buf)
  suits a ≥32 MiB-L3 host. `raceway_autoconfig.sh` derives this from the detected cache.
- Pin to physical cores for the SMT-off thread count, then add the siblings for SMT-on. The
  sibling layout is host-specific; the autoconfig script prints both pin sets.

Useful env knobs: `RACEWAY_CAP=0|1` (cap off/on, default on), `DEEP_DISP=branched|blmerge`,
`PRODUCER_CAP=exact|shadow|1`, `CONT=carry|recompute`, `SHARD_CHUNK=N`,
`PRECERT=0` (disable default MAP1 certified-shed pre-exclusion).

`PRECERT` is enabled by default. When a key has certified-shed MAP1 data bits,
the producer scans only the logical support axis and fixes the shed bits before
MAP1. For zero-cert keys it is a no-op. The initial bridge backs off
automatically for incompatible window policies; explicit `PRECERT=1` turns those
shape mismatches into errors.

## Self-check (built-in parity)

The cap path is exact on the final distinct count, so cap-off and cap-on must agree:

```sh
PRECERT=0 RACEWAY_CAP=0 ./cpu_raceway 0x9e9d137b 1048576 8 5 count   # cap-off (exact ceiling)
PRECERT=0 RACEWAY_CAP=1 ./cpu_raceway 0x9e9d137b 1048576 8 5 count   # cap-on  (same UNION_final)
```

Both print the same `UNION_final` — the cap dropped only true duplicates. Leave
`PRECERT` at its default for production; `PRECERT=0` is useful when you want raw
contiguous-window parity.

## Source layout

`cpu_raceway.cpp` is split into ordered includes (they assume the file's include order; they
are **not** standalone headers):

- `raceway_kernel_select.h` — ISA/kernel typedef (`Kern`) + interleave width (`g_ilp`)
- `raceway_cap.h` — boundary caps, the W1 producer cap, run-config, and the MAP1 dedup set
- `raceway_deep.h` — the deep-drain kernel dispatch
