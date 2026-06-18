CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++11 -DCL_TARGET_OPENCL_VERSION=120

# OpenCL headers and the ICD loader library. Override on the command line
# if either is in a non-standard location, e.g.:
#
#   # Headers from a system package:
#   make
#
#   # Headers shipped alongside CUDA (when no system OpenCL SDK is installed):
#   make OPENCL_HEADERS=/usr/local/cuda/include
#
#   # The CUDA ICD loader has the SONAME but no unversioned symlink, so
#   # `-lOpenCL` fails. Link the versioned library directly:
#   make OPENCL_LIB=/usr/local/cuda/targets/x86_64-linux/lib/libOpenCL.so.1
#
#   # macOS uses the OpenCL framework:
#   make OPENCL_LIB='-framework OpenCL'
OPENCL_HEADERS ?= /usr/include
OPENCL_LIB     ?= -lOpenCL

SRC_DIR   = src
INCLUDES  = -I$(OPENCL_HEADERS) -I$(SRC_DIR)
LDFLAGS   = $(OPENCL_LIB)

TARGET    = tm_opencl_forward
SRCS      = $(SRC_DIR)/main.cpp $(SRC_DIR)/key_schedule.cpp $(SRC_DIR)/rng.cpp \
            $(SRC_DIR)/rng_obj.cpp $(SRC_DIR)/alignment2.cpp
OBJS      = $(notdir $(SRCS:.cpp=.o))

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

main.o: $(SRC_DIR)/map1_certifier.h $(SRC_DIR)/rng_obj.h $(SRC_DIR)/alignment2.h

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
