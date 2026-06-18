# Pinned to CUDA 12.x/13.2 rather than the bare `/usr/local/cuda` symlink: ptxas
# 13.3 emits ~8-13% slower SASS for the Blackwell forward/screen kernels than 13.2
# (an instruction-scheduling regression on the dependent LDG->SHFL load chain — not
# occupancy/registers, and not recoverable via ptxas flags). Override CUDA_PATH only
# if your toolchain predates 13.3 or once the regression is resolved. Falls back to
# the `/usr/local/cuda` symlink if /usr/local/cuda-13.2 is absent.
CUDA_PATH ?= $(firstword $(wildcard /usr/local/cuda-13.2) /usr/local/cuda)
NVCC      = $(CUDA_PATH)/bin/nvcc
CXX       = g++

# Default sm_120 (RTX 5090 / RTX PRO 6000 Blackwell). Override with e.g.
# `make GENCODE='-gencode arch=compute_89,code=sm_89'` for Ada-class GPUs.
GENCODE  ?= -gencode arch=compute_120,code=sm_120 \
            -gencode arch=compute_120,code=compute_120

SRC_DIR   = src
INCLUDES  = -I$(CUDA_PATH)/include -I$(SRC_DIR)
CXXFLAGS  = -std=c++17 -O2 -m64 $(INCLUDES)

# wide_merge_sort.cu is a real host+device object (cub radix-sort / RLE used by the
# raceway/compaction wide-merge path), NOT a fatbin. cub/CCCL headers live under
# targets/.../include/cccl in CUDA 13.x. WMS_ARCH = the build's sm_ arch.
CCCL_INC      = $(CUDA_PATH)/targets/x86_64-linux/include/cccl
WMS_ARCH     ?= 120
NVCC_OBJFLAGS = -O3 -std=c++17 -arch=sm_$(WMS_ARCH) -I$(CCCL_INC)

CPP_OBJS  = main.o rng_obj.o alignment2.o key_schedule.o
OBJS      = $(CPP_OBJS) wide_merge_sort.o

all: tm_cuda tm_cuda.fatbin

tm_cuda.fatbin: $(SRC_DIR)/tm_cuda.cu
	$(NVCC) --fatbin -lineinfo $(GENCODE) $< -o $@

%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

main.o: $(SRC_DIR)/map1_certifier.h $(SRC_DIR)/rng_obj.h $(SRC_DIR)/alignment2.h

wide_merge_sort.o: $(SRC_DIR)/wide_merge_sort.cu
	$(NVCC) $(NVCC_OBJFLAGS) -c $< -o $@

tm_cuda: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ \
	    -L$(CUDA_PATH)/lib64/stubs -lcuda \
	    -L$(CUDA_PATH)/lib64 -lcudart \
	    -Wl,-rpath,$(CUDA_PATH)/lib64

clean:
	rm -f *.o tm_cuda tm_cuda.fatbin

.PHONY: all clean
