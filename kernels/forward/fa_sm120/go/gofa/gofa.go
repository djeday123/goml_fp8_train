// Package gofa is a CGO binding for libfa_sm120 (FlashAttention FP8 fwd on sm_120a).
//
// Build: ensure libfa_sm120.so is on the dynamic loader path (LD_LIBRARY_PATH
// or system lib dir), then `go build` / `go test` as usual.
package gofa

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo LDFLAGS: -L${SRCDIR}/../.. -lfa_sm120 -Wl,-rpath,${SRCDIR}/../..
#include "fa_sm120.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Status mirrors fa_status_t.
type Status int

const (
	OK                  Status = 0
	ErrInvalidArg       Status = 1
	ErrUnsupportedArch  Status = 2
	ErrUnsupportedHd    Status = 3
	ErrUnsupportedShape Status = 4
	ErrCuda             Status = 5
	ErrOOM              Status = 6
	ErrInternal         Status = 7
)

// Ctx wraps fa_ctx_t opaque pointer.
type Ctx struct {
	ptr *C.fa_ctx_t
}

// Error implements error for non-OK statuses.
type Error struct {
	Status   Status
	Message  string
	CudaInfo string
}

func (e *Error) Error() string {
	if e.CudaInfo != "" {
		return fmt.Sprintf("fa_sm120 status=%d: %s (%s)", e.Status, e.Message, e.CudaInfo)
	}
	return fmt.Sprintf("fa_sm120 status=%d: %s", e.Status, e.Message)
}

func makeError(s Status, ctx *Ctx) error {
	if s == OK {
		return nil
	}
	msg := C.GoString(C.fa_status_str(C.fa_status_t(s)))
	cuda := ""
	if ctx != nil && ctx.ptr != nil {
		cuda = C.GoString(C.fa_last_cuda_error(ctx.ptr))
	}
	return &Error{Status: s, Message: msg, CudaInfo: cuda}
}

// Version returns the build version string of libfa_sm120.
func Version() string {
	return C.GoString(C.fa_version())
}

// Create probes the device and prepares a context. Returns ErrUnsupportedArch
// on non-sm_120a cards.
func Create() (*Ctx, error) {
	c := &Ctx{}
	s := Status(C.fa_create(&c.ptr))
	if s != OK {
		return c, makeError(s, c)
	}
	return c, nil
}

// Destroy releases context. Safe on nil.
func (c *Ctx) Destroy() {
	if c == nil || c.ptr == nil {
		return
	}
	C.fa_destroy(c.ptr)
	c.ptr = nil
}

// Forward dispatches a single FA forward call.
//
// q, k, v: device pointers to FP8 e4m3 (uint8) data, layout [bh, sl, hd], row-major.
// o:       device pointer to FP16 output, same layout.
// scale:   typically 1/sqrt(hd).
// causal:  0=no mask, 1=causal.
// window:  0=no window, >0=sliding window length.
// stream:  CUDA stream pointer (0=default).
func (c *Ctx) Forward(
	q, k, v, o unsafe.Pointer,
	bh, sl, hd, causal, window int,
	scale float32, stream unsafe.Pointer,
) error {
	s := Status(C.fa_forward(
		c.ptr,
		q, k, v, o,
		C.int(bh), C.int(sl), C.int(hd),
		C.int(causal), C.int(window),
		C.float(scale),
		C.fa_stream_t(stream),
	))
	return makeError(s, c)
}

// DispatchSelect returns which kernel would run for the given config without launching.
// Useful for tests and autotuner integration.
func DispatchSelect(bh, sl, hd, causal, window int) (kernelID int, name string) {
	kid := C.fa_dispatch_select(
		C.int(bh), C.int(sl), C.int(hd), C.int(causal), C.int(window))
	return int(kid), C.GoString(C.fa_kernel_name(kid))
}
