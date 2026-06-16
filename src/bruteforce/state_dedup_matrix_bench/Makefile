CXX      := g++
CXXFLAGS := -O3 -std=c++20 -m64 -march=native -mtune=native -I../../common -I.. -pthread
SRC_COMMON := ../../common
HEADERS := ../../common/state_dedup.h ../../common/state_dedup_il.h ../../common/key_schedule.h

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
        ../cpu/tm_avx512_r512s_8.cpp \
        ../cpu/tm_avx512_r512_map_8.cpp \
        ../cpu/tm_avx512_r512s_map_8.cpp

SURVEY_SRCS := map1_fpset_survey.cpp \
        $(SRC_COMMON)/rng.cpp \
        $(SRC_COMMON)/rng_obj.cpp \
        $(SRC_COMMON)/alignment2.cpp \
        $(SRC_COMMON)/key_schedule.cpp \
        $(SRC_COMMON)/tm_base.cpp \
        ../cpu/tm_avx2_r256s_8.cpp \
        ../cpu/tm_avx512_r512s_8.cpp \
        ../cpu/tm_avx512_r512_map_8.cpp

TARGET := state_dedup_matrix_bench

all: $(TARGET) map1_fpset_survey

$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS) -pthread

map1_fpset_survey: $(SURVEY_SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SURVEY_SRCS) -pthread

clean:
	rm -f $(TARGET) map1_fpset_survey

.PHONY: all clean
