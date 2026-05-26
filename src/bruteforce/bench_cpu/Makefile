CXX      := g++
# -march=native enables the full ISA available on the build host (AVX2,
# AVX-512, etc.) so all CPU variants linked below can compile. The bench tool
# is for the build host; not a redistributable binary.
CXXFLAGS := -O3 -std=c++20 -m64 -march=native -mtune=native -I../../common -I.. -pthread
SRC_COMMON := ../../common

SRCS := bench_cpu.cpp \
        $(SRC_COMMON)/rng.cpp \
        $(SRC_COMMON)/rng_obj.cpp \
        $(SRC_COMMON)/alignment2.cpp \
        $(SRC_COMMON)/key_schedule.cpp \
        $(SRC_COMMON)/tm_base.cpp \
        $(SRC_COMMON)/carnival_targets.cpp \
        ../cpu/tm_8.cpp \
        ../cpu/tm_avx_r128s_8.cpp \
        ../cpu/tm_avx_r256s_8.cpp \
        ../cpu/tm_avx2_r256s_8.cpp \
        ../cpu/tm_avx2_r256_map_8.cpp \
        ../cpu/tm_avx512_r512s_8.cpp

TARGET := bench_cpu
PROFILE_DIR ?= .pgo
PGO_IMPL ?= avx2_r256_map_8

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -pthread

pgo:
	rm -rf $(PROFILE_DIR)
	mkdir -p $(PROFILE_DIR)
	$(CXX) $(CXXFLAGS) -fprofile-generate=$(PROFILE_DIR) -o $(TARGET) $(SRCS) -pthread
	./$(TARGET) --impl $(PGO_IMPL) --threads 1 --workunit_size 1048576 --warmup 262144
	./$(TARGET) --impl $(PGO_IMPL) --threads 1 --range_start 4047122432 --workunit_size 1048576 --warmup 65536
	$(CXX) $(CXXFLAGS) -fprofile-use=$(PROFILE_DIR) -fprofile-correction -o $(TARGET) $(SRCS) -pthread

clean:
	rm -f $(TARGET)
	rm -rf $(PROFILE_DIR)

.PHONY: all pgo clean
