package cpu

import (
	"unsafe"

	"github.com/djeday123/goml/backend"
)

// storage is a CPU memory buffer backed by a Go byte slice.
type storage struct {
	data []byte
}

func newStorage(byteLen int) *storage {
	return &storage{data: make([]byte, byteLen)}
}

func (s *storage) Device() backend.Device { return backend.CPU0 }

func (s *storage) Ptr() unsafe.Pointer {
	if len(s.data) == 0 {
		return nil
	}
	return unsafe.Pointer(&s.data[0])
}

func (s *storage) ByteLen() int { return len(s.data) }

func (s *storage) Bytes() []byte { return s.data }

func (s *storage) Free() {
	s.data = nil
}
