# Convenience build targets for the public forward-search release.
#
# Each implementation keeps its own Makefile under src/bruteforce/. These
# targets are thin wrappers so a fresh checkout has obvious entry points.

CUDA_DIR ?= $(if $(wildcard cuda/Makefile),cuda,release_staging/tm_cuda)
OPENCL_DIR ?= $(if $(wildcard opencl/Makefile),opencl,release_staging/opencl_public)

.PHONY: all cpu cpu-dedup cuda opencl clean help

all: cpu

cpu:
	$(MAKE) -C src/bruteforce/bench_cpu all

cpu-dedup:
	$(MAKE) -C src/bruteforce/state_dedup_speedup_bench all
	$(MAKE) -C src/bruteforce/state_dedup_matrix_bench all
	$(MAKE) -C src/bruteforce/state_dedup_screen_bench all
	$(MAKE) -C src/bruteforce/state_dedup_origin_bench all

cuda:
	$(MAKE) -C $(CUDA_DIR) all

opencl:
	$(MAKE) -C $(OPENCL_DIR) all

clean:
	$(MAKE) -C src/bruteforce/bench_cpu clean
	$(MAKE) -C src/bruteforce/state_dedup_speedup_bench clean
	$(MAKE) -C src/bruteforce/state_dedup_matrix_bench clean
	$(MAKE) -C src/bruteforce/state_dedup_screen_bench clean
	$(MAKE) -C src/bruteforce/state_dedup_origin_bench clean
	$(MAKE) -C $(CUDA_DIR) clean
	$(MAKE) -C $(OPENCL_DIR) clean

help:
	@echo "Targets:"
	@echo "  make cpu        Build native CPU AVX/SIMD forward benchmark"
	@echo "  make cpu-dedup  Build CPU state-dedup characterization tools"
	@echo "  make cuda       Build CUDA forward verifier"
	@echo "  make opencl     Build OpenCL forward verifier"
	@echo "  make clean      Remove generated binaries from those targets"
