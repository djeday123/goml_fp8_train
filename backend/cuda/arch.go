package cuda

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ─────────────────────────────────────────────────────────────────────────────
// Compute-capability helpers and library lookup.
//
// goml's PTX kernels target sm_80 and run on every Ampere/Ada/Hopper/Blackwell
// GPU via JIT compilation in the driver. The compiled .so files (libfp8gemm,
// libtransformer, libflash_attention_*) are produced as fat binaries
// containing native SASS for several architectures plus PTX for forward
// compatibility (see scripts/build_cuda.sh).
//
// At runtime we don't gate behaviour on the SM; we just log it so the user
// can confirm the GPU is what they expect, and we extend the .so search
// path to include $GOML_LIBS_DIR and the project-local "libs/" directory.
// ─────────────────────────────────────────────────────────────────────────────

// archName maps a (major, minor) compute capability to a human-readable
// architecture family name.
func archName(major, minor int) string {
	cc := major*10 + minor
	switch {
	case cc >= 120:
		return "Blackwell" // RTX PRO 6000, RTX 5090, RTX 5080 (sm_120+)
	case cc == 100:
		return "Blackwell DC" // B100/B200
	case cc >= 90:
		return "Hopper" // H100, H200
	case cc == 89:
		return "Ada Lovelace" // RTX 4090, L40
	case cc >= 86:
		return "Ampere" // RTX 3090, A40
	case cc == 80:
		return "Ampere" // A100, A30
	case cc >= 75:
		return "Turing"
	case cc >= 70:
		return "Volta"
	case cc >= 60:
		return "Pascal"
	default:
		return fmt.Sprintf("sm_%d.%d", major, minor)
	}
}

// SMTag returns a short tag like "sm_120" for the given compute capability.
func SMTag(major, minor int) string {
	return fmt.Sprintf("sm_%d%d", major, minor)
}

// LibsDir returns the directory to look in for prebuilt .so files. Priority:
//   1. $GOML_LIBS_DIR (explicit override)
//   2. ./libs (working directory)
//   3. <executable_dir>/libs
//   4. /usr/local/lib/goml
// First existing directory wins; empty string if nothing found.
func LibsDir() string {
	candidates := []string{os.Getenv("GOML_LIBS_DIR")}
	if cwd, err := os.Getwd(); err == nil {
		candidates = append(candidates, filepath.Join(cwd, "libs"))
	}
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Join(filepath.Dir(exe), "libs"))
	}
	candidates = append(candidates, "/usr/local/lib/goml")

	for _, d := range candidates {
		if d == "" {
			continue
		}
		if fi, err := os.Stat(d); err == nil && fi.IsDir() {
			return d
		}
	}
	return ""
}

// resolveLib returns the full path to a library, or just its base name (so
// the OS dynamic loader can find it via LD_LIBRARY_PATH / rpath).
//
// Example: resolveLib("libfp8gemm.so") tries:
//   $GOML_LIBS_DIR/libfp8gemm.so
//   ./libs/libfp8gemm.so
//   <exe_dir>/libs/libfp8gemm.so
//   libfp8gemm.so (last resort — relies on standard loader path)
func resolveLib(name string) string {
	dir := LibsDir()
	if dir != "" {
		full := filepath.Join(dir, name)
		if _, err := os.Stat(full); err == nil {
			return full
		}
	}
	return name
}

// DescribeArch returns a multi-line string describing the GPU and the SM
// targets baked into goml's prebuilt libraries — useful for debugging
// installation issues.
func DescribeArch(info *DeviceInfo) string {
	if info == nil {
		return "CUDA backend not yet initialized"
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "GPU:        %s\n", info.Name)
	fmt.Fprintf(&sb, "Compute:    sm_%d.%d (%s)\n",
		info.ComputeMaj, info.ComputeMin, archName(info.ComputeMaj, info.ComputeMin))
	fmt.Fprintf(&sb, "Driver:     loaded via libcuda.so (purego)\n")
	if d := LibsDir(); d != "" {
		fmt.Fprintf(&sb, "Libs dir:   %s\n", d)
	} else {
		fmt.Fprintf(&sb, "Libs dir:   (not found — using OS loader path)\n")
	}
	return sb.String()
}
