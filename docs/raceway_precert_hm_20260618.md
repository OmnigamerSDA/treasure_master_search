# Raceway Precert Harmonic Mean, 2026-06-18

Default MAP1 precert enabled, represented window `W=268435456`, cadence
`2,5,10,16`, 8-key representative set:

`0xb2e8c4ac`, `0x65a37ee7`, `0x2ca5b42d`, `0x9e3779b9`,
`0x6f2c81a4`, `0x9e9d137b`, `0xceadd933`, `0xdeadbeef`.

Rates below are represented candidates per second. For no-cert keys this equals
logical candidates per second; for certified keys it multiplies the logical scan
rate by `2^certified_bits`.

Development-tree raw logs and CSV summaries:

- `results/raceway_precert_hm_20260618/summary.csv`
- `results/raceway_precert_hm_20260618/aggregate.csv`

## Aggregate

| Backend | Hardware/config | 8-key HM | 60% diffuse-weight HM | Collapse HM | Mid HM | Diffuse HM |
|---|---|---:|---:|---:|---:|---:|
| CUDA | RTX 5090, production wave-state raceway | 415.5 M/s | 367.7 M/s | 45.7 B/s | 599.5 M/s | 252.0 M/s |
| OpenCL | RTX 5090, wave-cap raceway | 264.5 M/s | 233.3 M/s | 30.2 B/s | 403.5 M/s | 158.6 M/s |
| CPU | Ryzen 9 9900X, AVX-512, 24 threads | 22.9 M/s | 20.1 M/s | 1.34 B/s | 39.0 M/s | 13.5 M/s |

## Per-Key Rates

| Key | Class | Cert bits | CUDA 5090 | OpenCL 5090 | CPU 9900X 24t |
|---|---|---:|---:|---:|---:|
| `0xb2e8c4ac` | collapse | 8 | 41.7 B/s | 27.5 B/s | 1.23 B/s |
| `0x65a37ee7` | collapse | 8 | 50.4 B/s | 33.6 B/s | 1.47 B/s |
| `0x2ca5b42d` | mid | 0 | 460.8 M/s | 321.0 M/s | 31.0 M/s |
| `0x9e3779b9` | diffuse | 0 | 276.0 M/s | 176.7 M/s | 13.8 M/s |
| `0x6f2c81a4` | diffuse | 0 | 299.4 M/s | 192.4 M/s | 15.4 M/s |
| `0x9e9d137b` | diffuse | 0 | 227.3 M/s | 147.5 M/s | 13.3 M/s |
| `0xceadd933` | diffuse | 0 | 221.6 M/s | 131.8 M/s | 12.0 M/s |
| `0xdeadbeef` | mid | 2 | 857.8 M/s | 542.9 M/s | 52.8 M/s |

## Reading

Precert changes the workload shape sharply for certified collapse keys. The two
8-bit certified keys run at tens of billions represented candidates per second
on GPU and above one billion represented candidates per second on CPU. The mixed
harmonic mean still remains controlled by no-cert diffuse keys, so this does not
remove the need for the diffuse-friendly raceway path.
