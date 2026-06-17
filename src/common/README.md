# common/ - Shared Forward Primitives

Code shared by the public CPU forward tools: RNG, key schedule, target
tables, state-dedup helpers, and forward-algorithm scaffolding. CUDA and
OpenCL keep self-contained copies of the pieces they need under their own
top-level directories.

## Files

| Module                            | Purpose                                    |
|-----------------------------------|--------------------------------------------|
| `rng.{cpp,h}` / `rng_obj.{cpp,h}` | C-style and object-style PRNG              |
| `key_schedule.{cpp,h}`            | Forward key schedule                       |
| `carnival_targets.{cpp,h}`        | Hardcoded target hashes per world          |
| `alignment2.{cpp,h}`              | Memory-alignment helpers                   |
| `tm_base.{cpp,h}`                 | Forward algorithm reference implementation |
| `state_dedup*.h`                  | CPU final-state dedup helpers (raceway + research benches) |
| `strong_hash.h`                   | Avalanched fingerprint for memcmp-free dedup |
| `window_policy.h`                 | Window/tile index mapping (raceway)        |
| `routing.h`                       | Shed-proxy routing helpers                 |
| `key_file.h`                      | CSV/key-list parsing helper                |
| `data_sizes.h`                    | Compile-time size constants                |

## Caveats

- This directory is a curated subset of the larger development tree. Files
  needed only by reverse search, CNF generation, or legacy OpenCL host glue
  are not included in the forward-only public package.
- The CPU tools have no external dependencies beyond the C++ standard
  library and the compiler intrinsics enabled by their Makefiles.
