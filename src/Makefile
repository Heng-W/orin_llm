CXX       ?= g++
CXXFLAGS  ?= -std=c++17 -O3 -fno-exceptions -fno-rtti -Wall -Wextra
CXXFLAGS  += -I../include -fPIC

BUILD      ?= ../build

STATIC_LIB = $(BUILD)/liborin_llm.a
SHARED_LIB = $(BUILD)/liborin_llm.so

COMMON_SRCS = gguf.cpp tokenizer.cpp sampler.cpp engine.cpp model_common.cpp

USE_CPU_ONLY ?= 0
ifeq ($(USE_CPU_ONLY),1)
    SRCS = $(COMMON_SRCS) model_cpu.cpp
    OBJS = $(patsubst %.cpp,$(BUILD)/%.o,$(SRCS))
else
    NVCC      ?= nvcc
    CUDAFLAGS  = -std=c++17 -arch=sm_87 -O3 -use_fast_math -extended-lambda \
                 -Xcompiler -fno-exceptions -Xcompiler -fno-rtti -Xcompiler -fPIC \
                 -I../include
    CUDA_SRCS  = kernels.cu model_gpu.cu
    CUDA_OBJS  = $(patsubst %.cu,$(BUILD)/%.o,$(CUDA_SRCS))
    CXXFLAGS  += -DLLM_USE_CUDA
    SRCS = $(COMMON_SRCS)
    OBJS = $(patsubst %.cpp,$(BUILD)/%.o,$(SRCS)) $(CUDA_OBJS)
endif

all: $(STATIC_LIB)

static: $(STATIC_LIB)
shared: $(SHARED_LIB)

$(STATIC_LIB): $(OBJS) | $(BUILD)
	ar rcs $@ $^

$(SHARED_LIB): $(OBJS) | $(BUILD)
ifeq ($(USE_CPU_ONLY),1)
	$(CXX) -shared -o $@ $^ -lm
else
	$(NVCC) $(CUDAFLAGS) -shared -o $@ $^ -lm
endif

$(BUILD)/%.o: %.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD)/%.o: %.cu | $(BUILD)
	$(NVCC) $(CUDAFLAGS) -c -o $@ $<

$(BUILD):
	mkdir -p $@

clean:
	rm -f $(OBJS) $(STATIC_LIB) $(SHARED_LIB)

.PHONY: all static shared clean
