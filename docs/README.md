# docs/

Forward-search references for the public CPU/CUDA/OpenCL release.

The **production engine on every backend is the bounded-wave raceway** (best across
throughput AND memory; FN-safe). The flat checksum screen / compaction / state-dedup
paths are kept as a research baseline and the bit-exact parity reference. Start with
`PROJECT_STATUS.md`.

| Doc | Topic |
|---|---|
| `PROJECT_STATUS.md` | Public forward-release status and measured raceway rates |
| `forward_release_candidate_20260525.md` | Historical (2026-05-25) screen/dedup RC snapshot — superseded by the raceway |
| `gpu_forward_benchmark_notes.md` | GPU screen-benchmark methodology + history (research baseline) |
| `decryption_execution_trace_reference.md` | Forward algorithm walkthrough |
| `password_conversion_algorithm.md` | 24-character password to key/data conversion |

Reverse engines, CNF/SAT tooling, daily research notes, and large benchmark
corpora are intentionally omitted from this forward-only public package.
