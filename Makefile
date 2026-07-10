# goml_fp8_train – FP8 Training Makefile
#
# Targets
# -------
#   make run         – CPU reference build + run (no GPU needed)
#   make build-cuda  – Compile CUDA shared library + Go binary with FP8 path
#   make test        – Run all Go unit tests (CPU reference)
#   make bench       – Go benchmark suite
#   make clean       – Remove build artefacts

.PHONY: run build-cuda test bench clean

# Default dims chosen to show meaningful TFLOPS numbers on GPU.
STEPS  ?= 50
BATCH  ?= 2048
HIDDEN ?= 4096
LAYERS ?= 3

# ── CPU reference (no CUDA required) ────────────────────────────────────────
run:
	go run . -steps $(STEPS) -batch $(BATCH) -hidden $(HIDDEN) -layers $(LAYERS)

# ── CUDA FP8 path (requires CUDA 12.1+ and Hopper GPU) ──────────────────────
CUDA_ARCH ?= sm_90a   # Hopper (H100 / H200); set sm_89 for Ada Lovelace
CUDA_DIR  ?= /usr/local/cuda
NVCC      := $(CUDA_DIR)/bin/nvcc

cuda/libfp8gemm.so: cuda/fp8_gemm.cu cuda/fp8_gemm.h
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -shared -fPIC \
		-o $@ cuda/fp8_gemm.cu \
		-lcublas -lcublasLt -lcudart

build-cuda: cuda/libfp8gemm.so
	CGO_LDFLAGS="-L$(CURDIR)/cuda -Wl,-rpath,$(CURDIR)/cuda -lfp8gemm \
	             -lcublas -lcublasLt -lcudart -lstdc++" \
	go build -tags cuda -o goml_fp8_train_cuda .
	@echo "Built: ./goml_fp8_train_cuda"
	@echo "Run:   ./goml_fp8_train_cuda -steps $(STEPS) -batch $(BATCH) \
	             -hidden $(HIDDEN) -layers $(LAYERS)"

# ── Tests ───────────────────────────────────────────────────────────────────
test:
	go test ./... -count=1

bench:
	go test ./train/... -run='^$$' -bench=. -benchtime=10s -count=3

clean:
	rm -f cuda/libfp8gemm.so goml_fp8_train_cuda
