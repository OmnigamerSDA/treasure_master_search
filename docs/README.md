# docs/

Forward-search references for the public CPU / CUDA / OpenCL / FPGA release.

The **production engine on every backend is the bounded-wave raceway** (best across
throughput AND memory; FN-safe). The flat checksum screen / compaction / state-dedup
paths are kept as a research baseline and the bit-exact parity reference. Start with
`PROJECT_STATUS.md`.

Supported production launches also enable the MAP1 certifier by default. It is
disabled with `--no-precert` on CUDA/OpenCL or `PRECERT=0` on the CPU raceway.

| Doc | Topic |
|---|---|
| `PROJECT_STATUS.md` | Public forward-release status and measured raceway rates |
| `raceway_precert_hm_20260618.md` | Default-precert represented-throughput harmonic means |
| `forward_release_candidate_20260525.md` | Historical (2026-05-25) screen/dedup RC snapshot — superseded by the raceway |
| `gpu_forward_benchmark_notes.md` | GPU screen-benchmark methodology + history (research baseline) |
| `decryption_execution_trace_reference.md` | Forward algorithm walkthrough |
| `password_conversion_algorithm.md` | 24-character password to key/data conversion |

The **FPGA** map/RNG/screen engine (VHDL + testbenches + OOC synthesis) has its own
guide at `src/fpga/README.md`. The durable cert-tier **operator** that clears a
keyspace tier by tier (`scripts/cert_tier_ops.py`, with a runnable cert16 sample DB)
is documented in the repo-root README's "Clearing a keyspace (the operator)" section.

Reverse engines, CNF/SAT tooling, daily research notes, and large benchmark
corpora are intentionally omitted from this forward-only public package.
