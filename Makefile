# Convenience build targets for the public forward-search release.
#
# Each implementation keeps its own Makefile under src/bruteforce/. These
# targets are thin wrappers so a fresh checkout has obvious entry points.

CUDA_DIR ?= $(if $(wildcard cuda/Makefile),cuda,release_staging/tm_cuda)
OPENCL_DIR ?= $(if $(wildcard opencl/Makefile),opencl,release_staging/opencl_public)

.PHONY: all cpu cpu-dedup raceway cuda opencl receiver clean help

all: cpu

# CPU receiver that verifies raceway checksum survivors (machine-code gate). Required by the
# cert-tier operator's raceway path (scripts/cert_tier_ops.py --engine raceway).
receiver:
	$(MAKE) -C src/bruteforce inspect_bonus2_survivors

cpu:
	$(MAKE) -C src/bruteforce/bench_cpu all

# Bounded-wave CPU raceway (single-host full-2^32 forward dedup). Builds the AVX-512
# natmap target by default; src/bruteforce/cpu_raceway/raceway_autoconfig.sh --build
# auto-selects the best ISA/compiler for the host (AVX-512 vs AVX2 legacy).
raceway:
	$(MAKE) -C src/bruteforce/cpu_raceway all

cpu-dedup:
	$(MAKE) -C src/bruteforce/state_dedup_speedup_bench all
	$(MAKE) -C src/bruteforce/state_dedup_matrix_bench all
	$(MAKE) -C src/bruteforce/state_dedup_screen_bench all
	$(MAKE) -C src/bruteforce/state_dedup_origin_bench all
	@# map1_prefilter_bench is not part of the public allowlist; build it only if present.
	@if [ -d src/bruteforce/map1_prefilter_bench ]; then $(MAKE) -C src/bruteforce/map1_prefilter_bench all; fi

cuda:
	$(MAKE) -C $(CUDA_DIR) all

opencl:
	$(MAKE) -C $(OPENCL_DIR) all

clean:
	$(MAKE) -C src/bruteforce/bench_cpu clean
	$(MAKE) -C src/bruteforce/cpu_raceway clean
	$(MAKE) -C src/bruteforce/state_dedup_speedup_bench clean
	$(MAKE) -C src/bruteforce/state_dedup_matrix_bench clean
	$(MAKE) -C src/bruteforce/state_dedup_screen_bench clean
	$(MAKE) -C src/bruteforce/state_dedup_origin_bench clean
	@if [ -d src/bruteforce/map1_prefilter_bench ]; then $(MAKE) -C src/bruteforce/map1_prefilter_bench clean; fi
	$(MAKE) -C $(CUDA_DIR) clean
	$(MAKE) -C $(OPENCL_DIR) clean
	$(MAKE) -C src/bruteforce clean

help:
	@echo "Targets:"
	@echo "  make cpu        Build native CPU AVX/SIMD forward benchmark"
	@echo "  make cpu-dedup  Build CPU state-dedup / MAP1 prefilter characterization tools"
	@echo "  make raceway    Build the bounded-wave CPU raceway (full-2^32 forward dedup)"
	@echo "  make cuda       Build CUDA forward verifier"
	@echo "  make opencl     Build OpenCL forward verifier"
	@echo "  make receiver   Build the CPU hit-verifier used by the cert-tier operator"
	@echo "  make clean      Remove generated binaries from those targets"
