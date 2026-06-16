# Makefile — CPU raceway forward (bounded-wave dedup), public package.
#
# Three builds (see cpu_raceway.cpp header block for the full operating notes):
#   make CXX=clang++-21 cpu_raceway        # AVX-512 natmap   (x10/blmerge) — primary
#   make                 cpu_raceway_u     # AVX-512 universal (g++, x12)    — fallback
#   make CXX=clang++-21 cpu_raceway_avx2   # AVX2 natmap (x2/blmerge)        — older ISA
#
# Or let the host-detector pick: ./raceway_autoconfig.sh --build
#
# clang>=19 is recommended for the natmap builds (+14-22% vs g++ on deep-bound keys).
# All targets pass -fno-builtin-memset (roots out an -O3 vectorized-memset SIGSEGV).
# Linux only (reads /proc + <sys/resource.h> for peak-RSS reporting).

CXX        := g++
CXXFLAGS   := -O3 -std=c++20 -m64 -march=native -mtune=native -I../../common -I.. -I../cpu -pthread
CXXFLAGS_AVX2 := $(subst -march=native,-march=x86-64-v3,$(CXXFLAGS))
SRC_COMMON := ../../common

COMMON_BASE := $(SRC_COMMON)/rng.cpp \
               $(SRC_COMMON)/rng_obj.cpp \
               $(SRC_COMMON)/alignment2.cpp \
               $(SRC_COMMON)/key_schedule.cpp \
               $(SRC_COMMON)/tm_base.cpp
COMMON_MAP  := $(COMMON_BASE) ../cpu/tm_avx512_r512_map_8.cpp
COMMON_UNIV := $(COMMON_BASE) ../cpu/tm_avx512_r512s_8.cpp
COMMON_AVX2 := $(COMMON_BASE) ../cpu/tm_avx2_r256_map_8.cpp

HDRS := raceway_kernel_select.h raceway_cap.h raceway_deep.h

AVX2_W ?= 2

all: cpu_raceway

cpu_raceway:      cpu_raceway.cpp $(HDRS) $(COMMON_MAP)
	$(CXX) $(CXXFLAGS) -fno-builtin-memset -o $@ $< $(COMMON_MAP)

cpu_raceway_u:    cpu_raceway.cpp $(HDRS) $(COMMON_UNIV)
	$(CXX) $(CXXFLAGS) -DUNIVERSAL -o $@ $< $(COMMON_UNIV)

cpu_raceway_avx2: cpu_raceway.cpp $(HDRS) $(COMMON_AVX2)
	$(CXX) $(CXXFLAGS_AVX2) -DAVX2_NATMAP -DAVX2_RACEWAY_W=$(AVX2_W) -fno-builtin-memset -o $@ $< $(COMMON_AVX2)

clean:
	rm -f cpu_raceway cpu_raceway_u cpu_raceway_avx2

.PHONY: all clean
