CXX      := g++
CXXFLAGS := -O3 -std=c++20 -m64 -march=native -mtune=native -I../../common -I..
SRC_COMMON := ../../common

SRCS := state_dedup_matrix_bench.cpp \
        $(SRC_COMMON)/rng.cpp \
        $(SRC_COMMON)/rng_obj.cpp \
        $(SRC_COMMON)/alignment2.cpp \
        $(SRC_COMMON)/key_schedule.cpp \
        $(SRC_COMMON)/tm_base.cpp \
        ../cpu/tm_8.cpp \
        ../cpu/tm_avx_r128s_8.cpp \
        ../cpu/tm_avx_r256s_8.cpp \
        ../cpu/tm_avx2_r256s_8.cpp \
        ../cpu/tm_avx2_r256_map_8.cpp \
        ../cpu/tm_avx512_r512s_8.cpp

TARGET := state_dedup_matrix_bench

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

.PHONY: all clean
