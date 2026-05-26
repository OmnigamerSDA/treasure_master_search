CUDA_PATH ?= /usr/local/cuda
NVCC      = $(CUDA_PATH)/bin/nvcc
CXX       = g++

# Default sm_120 (RTX 5090 / RTX PRO 6000 Blackwell). Override with e.g.
# `make GENCODE='-gencode arch=compute_89,code=sm_89'` for Ada-class GPUs.
GENCODE  ?= -gencode arch=compute_120,code=sm_120 \
            -gencode arch=compute_120,code=compute_120

SRC_DIR   = src
INCLUDES  = -I$(CUDA_PATH)/include -I$(SRC_DIR)
CXXFLAGS  = -std=c++17 -O2 -m64 $(INCLUDES)

CPP_OBJS  = main.o rng_obj.o alignment2.o key_schedule.o

all: tm_cuda tm_cuda.fatbin

tm_cuda.fatbin: $(SRC_DIR)/tm_cuda.cu
	$(NVCC) --fatbin -lineinfo $(GENCODE) $< -o $@

%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

tm_cuda: $(CPP_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ \
	    -L$(CUDA_PATH)/lib64/stubs -lcuda \
	    -Wl,-rpath,$(CUDA_PATH)/lib64

clean:
	rm -f *.o tm_cuda tm_cuda.fatbin

.PHONY: all clean
