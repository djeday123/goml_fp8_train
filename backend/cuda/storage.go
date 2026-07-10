package cuda

import (
	"fmt"
	"unsafe"

	"github.com/djeday123/goml/backend"
)

// Storage represents a GPU memory buffer.
// Implements backend.Storage interface.
type Storage struct {
	ptr     uintptr // CUDA device pointer (not a Go pointer — just a numeric handle)
	byteLen int
	device  backend.Device
}

// Alloc allocates GPU memory.
func Alloc(byteLen int, dev backend.Device) (*Storage, error) {
	s := &Storage{byteLen: byteLen, device: dev}
	if r := cuMemAlloc(&s.ptr, uint64(byteLen)); r != CUDA_SUCCESS {
		return nil, fmt.Errorf("cuMemAlloc(%d bytes): %s", byteLen, r.Error())
	}
	return s, nil
}

func (s *Storage) Device() backend.Device { return s.device }
func (s *Storage) Ptr() unsafe.Pointer    { return unsafe.Pointer(s.ptr) }
func (s *Storage) Bytes() []byte          { return nil } // GPU memory — no direct access
func (s *Storage) ByteLen() int           { return s.byteLen }

func (s *Storage) Free() {
	if s.ptr != 0 {
		cuMemFree(s.ptr)
		s.ptr = 0
	}
}

// DevicePtr returns the raw uintptr for CUDA API calls (cuMemcpy, cuLaunchKernel).
func (s *Storage) DevicePtr() uintptr { return s.ptr }

// ──────────────────────────────────────────────────────────
// Host <-> Device transfers
// ──────────────────────────────────────────────────────────

// CopyHtoD copies from host (Go slice) to device (GPU).
func CopyHtoD(dst *Storage, src []byte) error {
	if len(src) > dst.byteLen {
		return fmt.Errorf("CopyHtoD: src (%d) > dst (%d)", len(src), dst.byteLen)
	}
	r := cuMemcpyHtoD(dst.ptr, unsafe.Pointer(&src[0]), uint64(len(src)))
	if r != CUDA_SUCCESS {
		return fmt.Errorf("cuMemcpyHtoD: %s", r.Error())
	}
	return nil
}

// CopyDtoH copies from device (GPU) to host (Go slice).
func CopyDtoH(dst []byte, src *Storage) error {
	if len(dst) < src.byteLen {
		return fmt.Errorf("CopyDtoH: dst (%d) < src (%d)", len(dst), src.byteLen)
	}
	r := cuMemcpyDtoH(unsafe.Pointer(&dst[0]), src.ptr, uint64(src.byteLen))
	if r != CUDA_SUCCESS {
		return fmt.Errorf("cuMemcpyDtoH: %s", r.Error())
	}
	return nil
}

// CopyDtoD copies between device buffers.
func CopyDtoD(dst, src *Storage, byteLen int) error {
	r := cuMemcpyDtoD(dst.ptr, src.ptr, uint64(byteLen))
	if r != CUDA_SUCCESS {
		return fmt.Errorf("cuMemcpyDtoD: %s", r.Error())
	}
	return nil
}

// Zero fills device memory with zeros.
func Zero(s *Storage) error {
	r := cuMemsetD8(s.ptr, 0, uint64(s.byteLen))
	if r != CUDA_SUCCESS {
		return fmt.Errorf("cuMemsetD8: %s", r.Error())
	}
	return nil
}
