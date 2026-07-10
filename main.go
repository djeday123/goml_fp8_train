// goml_fp8_train – FP8 training on GoTorch with real FP8 acceleration.
//
// Performance targets (NVIDIA H100 SXM5 80 GB, batch=2048, hidden=4096):
//   Forward:  ~652 TFLOPS
//   Backward: ~285 TFLOPS
//
// Usage:
//
//	# CPU reference (for development / CI):
//	go run . [-steps 20] [-batch 2048] [-hidden 4096]
//
//	# CUDA FP8 path (requires Hopper GPU + CUDA 12.1+):
//	go build -tags cuda . && ./goml_fp8_train
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/djeday123/goml_fp8_train/fp8"
	"github.com/djeday123/goml_fp8_train/train"
)

func main() {
	steps  := flag.Int("steps",  20,   "number of training steps to benchmark")
	batch  := flag.Int("batch",  2048, "batch size")
	hidden := flag.Int("hidden", 4096, "hidden dimension")
	layers := flag.Int("layers", 3,    "number of layer dimensions (creates layers-1 linear layers)")
	flag.Parse()

	if *steps <= 0 || *batch <= 0 || *hidden <= 0 || *layers < 2 {
		fmt.Fprintln(os.Stderr, "steps, batch, hidden must be > 0; layers >= 2")
		os.Exit(1)
	}

	backend := fp8.Backend()
	fmt.Printf("goml_fp8_train\n")
	fmt.Printf("  GEMM backend : %s\n", backendName(backend))
	fmt.Printf("  Batch size   : %d\n", *batch)
	fmt.Printf("  Hidden dim   : %d\n", *hidden)
	fmt.Printf("  Layers       : %d linear layers\n", *layers-1)
	fmt.Printf("  Steps        : %d\n\n", *steps)

	dims := make([]int, *layers)
	for i := range dims {
		dims[i] = *hidden
	}

	cfg := train.Config{
		LayerDims:    dims,
		BatchSize:    *batch,
		LearningRate: 1e-3,
		WeightDecay:  1e-2,
		Beta1:        0.9,
		Beta2:        0.95,
		Epsilon:      1e-8,
		LogInterval:  max(*steps/5, 1),
	}

	result, err := train.Benchmark(cfg, *steps)
	if err != nil {
		log.Fatalf("benchmark failed: %v", err)
	}

	fmt.Println(result)
	fmt.Printf("\n  Forward  : %s\n", formatFLOPS(result.ForwardTFLOPS))
	fmt.Printf("  Backward : %s\n", formatFLOPS(result.BackwardTFLOPS))
	fmt.Printf("  Step time: %.3f ms\n",
		result.Duration.Seconds()/float64(result.TotalSteps)*1e3)
	fmt.Println()

	if backend == fp8.BackendCUDA {
		fmt.Printf("  Target forward  : ~652 TFLOPS\n")
		fmt.Printf("  Target backward : ~285 TFLOPS\n")
	} else {
		fmt.Printf("  ⚠  Running on CPU reference backend.\n")
		fmt.Printf("  Build with -tags cuda on an H100 GPU to reach\n")
		fmt.Printf("  the target performance of 652 / 285 TFLOPS.\n")
	}
}

func formatFLOPS(tflops float64) string {
	switch {
	case tflops >= 0.1:
		return fmt.Sprintf("%.1f TFLOPS", tflops)
	case tflops >= 1e-4:
		return fmt.Sprintf("%.1f GFLOPS", tflops*1e3)
	default:
		return fmt.Sprintf("%.1f MFLOPS", tflops*1e6)
	}
}

func backendName(b fp8.GEMMBackend) string {
	switch b {
	case fp8.BackendCUDA:
		return "CUDA (cuBLASLt FP8)"
	default:
		return "CPU (float32 reference)"
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
