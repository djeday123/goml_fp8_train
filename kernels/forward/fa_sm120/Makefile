## libfa_sm120 — FlashAttention FP8 forward for sm_120a (Blackwell consumer).
## make lib        → libfa_sm120.so (shared) + libfa_sm120.a (static)
## make test       → build + run dispatcher unit test
## make clean

CUDA      ?= /usr/local/cuda-13.1
NVCC      := $(CUDA)/bin/nvcc
CXX       := g++
ARCH      := sm_120a

INCLUDE   := -Iinclude
NVCC_FLAGS := -O3 -gencode arch=compute_120a,code=$(ARCH) -std=c++17 \
              -Xcompiler "-fPIC -fvisibility=hidden" \
              -Xptxas=-v

CXX_FLAGS := -O3 -fPIC -fvisibility=hidden -std=c++17 $(INCLUDE)

BUILD     := build
SRCS_CXX  := src/fa_dispatch.cpp
SRCS_CU   := src/_v121r_kernel_full.cu src/fa_ctx.cu

OBJS_CXX  := $(SRCS_CXX:%.cpp=$(BUILD)/%.o)
OBJS_CU   := $(SRCS_CU:%.cu=$(BUILD)/%.o)

LIB_SO    := libfa_sm120.so
LIB_A     := libfa_sm120.a

.PHONY: all lib test clean

all: lib

## Step 1: concat v121r kernel + launcher wrapper into single TU
src/_v121r_kernel_full.cu: src/_v121r_kernel.cu src/_v121r_launcher.cuh
	cat src/_v121r_kernel.cu src/_v121r_launcher.cuh > $@

lib: $(LIB_SO) $(LIB_A)

$(LIB_SO): $(OBJS_CXX) $(OBJS_CU)
	$(NVCC) -shared -o $@ $^ $(NVCC_FLAGS) -lcudart
	@echo ""
	@echo "== Built $(LIB_SO) =="
	@nm -D --defined-only $(LIB_SO) | grep -E "fa_(create|destroy|forward|version|status|kernel|dispatch|last)" || true

$(LIB_A): $(OBJS_CXX) $(OBJS_CU)
	ar rcs $@ $(OBJS_CXX) $(OBJS_CU)

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE) -c $< -o $@

test: lib
	$(CXX) -O2 -Iinclude test/test_dispatch.c -o $(BUILD)/test_dispatch \
	    -L. -lfa_sm120 -Wl,-rpath,'$$ORIGIN' -Wl,-rpath,$(PWD)
	@echo ""
	@echo "== Running dispatcher unit test =="
	@LD_LIBRARY_PATH=$(PWD):$$LD_LIBRARY_PATH $(BUILD)/test_dispatch

test_readme: lib
	$(CXX) -O2 -Iinclude test/test_readme_c.c -o $(BUILD)/test_readme_c \
	    -L. -lfa_sm120 -L$(CUDA)/lib64 -lcudart -lm \
	    -Wl,-rpath,'$$ORIGIN' -Wl,-rpath,$(PWD)
	@echo ""
	@echo "== Running verbatim README C example =="
	@LD_LIBRARY_PATH=$(PWD):$(CUDA)/lib64:$$LD_LIBRARY_PATH $(BUILD)/test_readme_c

clean:
	rm -rf $(BUILD) $(LIB_SO) $(LIB_A) src/_v121r_kernel_full.cu
