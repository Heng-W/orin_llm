
USE_CPU_ONLY ?= 0  # 0: CUDA, 1: CPU

BUILD  = build
LIB    = $(BUILD)/liborin_llm.a
TARGET = chat_cli

all: $(TARGET)

$(LIB):
	$(MAKE) -C src BUILD=$(CURDIR)/$(BUILD) USE_CPU_ONLY=$(USE_CPU_ONLY)

$(TARGET): chat_cli.cpp $(LIB)
ifeq ($(USE_CPU_ONLY),1)
	$(CXX) -std=c++17 -O3 -Iinclude -o $@ $< -L$(BUILD) -lorin_llm -lm
else
	nvcc -std=c++17 -arch=sm_87 -O3 -Iinclude -o $@ $< -L$(BUILD) -lorin_llm -lm
endif

$(BUILD):
	mkdir -p $@

clean:
	$(MAKE) -C src BUILD=$(CURDIR)/$(BUILD) clean
	rm -rf $(BUILD) $(TARGET)

.PHONY: all clean $(LIB)
