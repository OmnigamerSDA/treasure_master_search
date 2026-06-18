# GPU Forward Benchmark Notes

This note tracks the current forward-path GPU benchmark shape and the latest measured CUDA results.

> **Production engine (2026-06-16): the bounded-wave raceway** (best across throughput AND memory; see
> `docs/forward_engines_operating_guide_20260614.md`, tuned per device by `test_cuda --calibrate-raceway`).
> The flat **checksum screen** measured below is the throughput-methodology baseline and the bit-exact
> parity reference — not the production engine.

## Scope

Chosen scope for the GPU benchmark:

- GPU:
  - expand
  - all maps
  - decrypt both worlds
  - checksum screen
- CPU:
  - later validation of checksum survivors
  - no machine-code acceptance on GPU in this pass

Why this shape:

- it keeps the GPU benchmark aligned with the "fair forward screen" notes
- it avoids moving the branch-heavy opcode validator into the kernel too early
- it gives a benchmarkable unit with clear throughput and survivor-count outputs

## Implementations

OpenCL benchmark path:

- host benchmark: `src/bruteforce/tm_opencl_32_8_test/main.cpp`
- checksum-screen kernel: `src/bruteforce/tm_opencl_32_8_test/tm.cl`
- survivor materialization kernel: `src/bruteforce/tm_opencl_32_8_test/tm.cl`

CUDA benchmark path:

- host benchmark: `src/bruteforce/test_cuda/main.cpp`
- checksum-screen kernel: `src/bruteforce/test_cuda/tm_cuda_screen.cuh`
- survivor materialization kernel: `src/bruteforce/test_cuda/tm_cuda_screen.cuh`

Current CUDA benchmark output focuses on:

- setup time
- warmup time
- screen-kernel time
- survivor-materialization kernel time
- CPU validation time
- end-to-end wall time
- candidate throughput
- checksum survivor counts split by world
- CPU machine-code flag counts over survivors

## Current CUDA Design

The current CUDA path is the best measured point from the recent optimization pass:

- one block = `128` threads = `4` warps
- checksum-screen kernel processes `4` candidates per warp
- dump/materialize kernels stay at `1` candidate per warp
- the schedule loop is warp-synchronous
- `alg2` and `alg5` use warp shuffles instead of shared-memory handoff
- checksum accumulation is done as a warp reduction
- other-world and checksum-mask constants are packed as `uint32_t` constant-memory words

Relevant current constants:

- `kCudaWarpsPerBlock = 4` in `src/bruteforce/test_cuda/main.cpp`
- `kCudaScreenCandidatesPerWarp = 4` in `src/bruteforce/test_cuda/main.cpp`

## Optimization Summary

Recent CUDA work that materially improved throughput:

1. Removed block-wide synchronization from the hot schedule loop.
2. Rewrote `alg2` and `alg5` to use warp shuffles.
3. Moved from `1` warp/block to `4` warps/block.
4. Parallelized checksum reduction across the warp.
5. Packed other-world and checksum-mask data into `uint32_t` constant tables.
6. Interleaved independent candidates inside the checksum-screen warp.

Measured outcome:

- the structural warp-synchronous rewrite produced the first large gain
- `2x` interleaving was a major additional win
- `4x` interleaving improved further and is the current best measured point
- `8x` interleaving was correct but regressed throughput, so it was not kept

## Profiling Findings

Nsight Systems and Nsight Compute now agree on the remaining bottleneck:

- the kernel is no longer occupancy-limited in the original sense
- host memcpy time is small relative to kernel time
- survivor materialization is negligible in the benchmark configuration
- the dominant remaining issue is the dependent table-load/use chain inside `run_alg()`

Hot sites repeatedly identified by source-correlated `ncu` runs (line numbers
from the original monolithic `tm_cuda.cu`; the relevant code now lives in
`src/bruteforce/test_cuda/tm_cuda_primitives.cuh` after the 2026-05-30 split):

- `tm_cuda.cu:92` → `run_alg()` alg-2 LDR path
- `tm_cuda.cu:96` → `run_alg()` alg-5 LDR path
- `tm_cuda.cu:100` → `run_alg()` alg-0 bit-extract
- `tm_cuda.cu:104` → `run_alg()` alg-6 bit-extract
- `tm_cuda.cu:108` → `run_alg()` alg-1/3/4 table load

What did not show a compelling standalone gain:

- adding `__restrict__` on hot-path pointers
- moving `schedule_data` to CUDA constant memory
- small loop-unroll tweaks

Conclusion from profiling:

- the obvious structural wins are already captured
- remaining changes are likely to be smaller, harder, or more speculative

## Measured CUDA Results

All results below use:

- `key_id = 0x2CA5B42D`
- `range_start = 0`
- `batch_size = 1048576`
- `warmup_batches = 1`

Short benchmark point after the final ILP4 kernel:

- `NVIDIA GeForce RTX 5090`
  - `screen_rate`: about `102.46M candidates/s`
  - `wall_rate`: about `84.58M candidates/s`
- `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`
  - representative short-run point: about `82.46M candidates/s` screen-rate
  - representative short-run point: about `71.85M candidates/s` wall-rate

Full `2^32` data sweep for one fixed key:

- command shape:
  - `./test_cuda --device <id> --key_id 0x2CA5B42D --range_start 0 --workunit_size 4294967296 --batch_size 1048576 --warmup_batches 1`
- `--device 1` = `NVIDIA GeForce RTX 5090`
  - `screen_kernel_s: 46.152`
  - `wall_s: 47.529`
  - `screen_rate: 93,061,603 candidates/s`
  - `wall_rate: 90,365,959 candidates/s`
- `--device 0` = `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`
  - `screen_kernel_s: 57.449`
  - `wall_s: 58.988`
  - `screen_rate: 74,761,553 candidates/s`
  - `wall_rate: 72,810,562 candidates/s`

Steady-state interpretation on this machine:

- one full `2^32` data sweep is about `47.5s` on the 5090
- one full `2^32` data sweep is about `59.0s` on the RTX PRO 6000

The full-sweep survivor totals matched across both GPUs:

- checksum survivors: `152,049`
- carnival: `92,538`
- other: `59,511`

## Benchmarking and Profiling Commands

Build:

```bash
cd src/bruteforce/test_cuda
make all
```

Quick parity check:

```bash
./test_cuda --device 1 --parity 256 --batch_size 256 --workunit_size 256 --warmup_batches 0
```

Short throughput benchmark (baseline kernel, historical reference):

```bash
./test_cuda --device 1 --key_id 0x2CA5B42D --range_start 0 --workunit_size 16777216 --batch_size 1048576 --warmup_batches 1
```

Short throughput benchmark (production offset-stream + ILP6 kernel, current):

```bash
./test_cuda --device 1 --key_id 0x2CA5B42D --range_start 0 --workunit_size 16777216 --batch_size 1048576 --warmup_batches 1 --screen-offsets
```

Full `2^32` sweep (production):

```bash
./test_cuda --device 1 --key_id 0x2CA5B42D --range_start 0 --workunit_size 4294967296 --batch_size 1048576 --warmup_batches 1 --screen-offsets
```

Nsight Systems:

```bash
/usr/local/cuda-13.2/bin/nsys profile --trace=cuda,osrt --sample=none -o nsys_cuda_5090_ilp4 ./test_cuda --device 1 --key_id 0x2CA5B42D --range_start 0 --workunit_size 16777216 --batch_size 1048576 --warmup_batches 1
```

Nsight Compute:

```bash
/usr/local/cuda-13.2/bin/ncu --set basic --target-processes all --kernel-name regex:tm_checksum_screen_cuda --launch-skip 1 --launch-count 1 ./test_cuda --device 1 --key_id 0x2CA5B42D --range_start 0 --workunit_size 1048576 --batch_size 1048576 --warmup_batches 1
```

## Operating Notes and Caveats

- `test_cuda` device numbering does not match `nvidia-smi` ordering on this machine:
  - `--device 1` is the `RTX 5090`
  - `--device 0` is the `RTX PRO 6000`
- short microbenchmarks overstate long-run steady-state throughput slightly; use the full `2^32` sweep when planning search throughput
- if other compute is active on the same GPU, both benchmark rates and replay-based profiler results can move noticeably
- `ncu` requires NVIDIA profiling counters to be enabled for the running user
- on Linux, setting `NVreg_RestrictProfilingToAdminUsers=0` did not take effect until reboot
- if `ncu` fails with `ERR_NVGPUCTRPERM`, re-check `/proc/driver/nvidia/params`
- the benchmark host now accepts `uint64_t` `--range_start` and `--workunit_size`, but the CUDA kernel interface is still 32-bit candidate indexed:
  - `--range_start` must still fit in `uint32`
  - `--workunit_size` may be up to `2^32`
  - `--range_start + --workunit_size` must stay within the `2^32` candidate space

## Profiling Artifacts

Recent saved profiler artifacts in `src/bruteforce/test_cuda/`:

- `ncu_source_5090.ncu-rep`
- `ncu_source_5090_pass2.ncu-rep`
- `ncu_source_5090_ilp2.ncu-rep`
- `ncu_source_5090_ilp4_final.ncu-rep`
- `nsys_cuda_5090.nsys-rep`
- `nsys_cuda_5090_opt.nsys-rep`
- `nsys_cuda_5090_pass2.nsys-rep`
- `nsys_cuda_5090_ilp2.nsys-rep`
- `nsys_cuda_5090_ilp4.nsys-rep`

## Practical Conclusion

The current CUDA checksum-screen path is already strong enough to materially change the search budget:

- the code is roughly two orders of magnitude faster than the default CPU forward path
- a full `2^32` data sweep for one key is now comfortably measured in under a minute on either GPU in this machine
- further changes are expected to be incremental unless a deeper algorithmic simplification is found
