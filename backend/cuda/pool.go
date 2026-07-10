package cuda

// Pool caches freed GPU memory buffers by size for reuse.
// Avoids expensive cuMemAlloc/cuMemFree in the training loop hot path.
//
// GPU memory allocation is slow (~5-50μs per call). In a training step
// we allocate dozens of temporary buffers for activations, gradients, etc.
// The pool eliminates this overhead by recycling buffers.
//
// Design:
//   - Buckets keyed by 256-byte-aligned size
//   - Get() returns cached buffer or allocates new
//   - Put() returns buffer to pool (no cuMemFree)
//   - FreeAll() releases everything at shutdown
//   - Thread-safe via mutex (goroutine-safe for compound arch)
//
// Usage:
//   pool := NewPool(device)
//   s, _ := pool.Get(4096)   // alloc or reuse
//   // ... use s.Ptr() for GPU ops ...
//   pool.Put(s)              // return to pool
//   pool.FreeAll()           // cleanup at shutdown

import (
	"sync"

	"github.com/djeday123/goml/backend"
)

type Pool struct {
	mu      sync.Mutex
	device  backend.Device
	buckets map[int][]*Storage // aligned size -> available buffers
	stats   PoolStats
}

type PoolStats struct {
	Hits       int64 // reused from pool
	Misses     int64 // new allocation
	AllocBytes int64 // total allocated
	FreeBytes  int64 // total freed
	PoolSize   int   // current buffers in pool
}

func NewPool(dev backend.Device) *Pool {
	return &Pool{
		device:  dev,
		buckets: make(map[int][]*Storage),
	}
}

// alignSize rounds up to 256-byte boundary for cache-friendly reuse.
// Also prevents fragmentation from many similar-but-not-identical sizes.
func alignSize(byteLen int) int {
	return ((byteLen + 255) / 256) * 256
}

// Get returns a buffer of at least byteLen bytes.
// Tries to reuse a cached buffer first (O(1) lookup by aligned size).
func (p *Pool) Get(byteLen int) (*Storage, error) {
	aligned := alignSize(byteLen)

	p.mu.Lock()
	if bufs, ok := p.buckets[aligned]; ok && len(bufs) > 0 {
		s := bufs[len(bufs)-1]
		p.buckets[aligned] = bufs[:len(bufs)-1]
		p.stats.Hits++
		p.stats.PoolSize--
		p.mu.Unlock()
		return s, nil
	}
	p.stats.Misses++
	p.mu.Unlock()

	// Cache miss — allocate new GPU memory
	s, err := Alloc(aligned, p.device)
	if err != nil {
		return nil, err
	}

	p.mu.Lock()
	p.stats.AllocBytes += int64(aligned)
	p.mu.Unlock()
	return s, nil
}

// Put returns a buffer to the pool for reuse.
// The buffer is NOT freed — it stays allocated on GPU for next Get().
func (p *Pool) Put(s *Storage) {
	if s == nil || s.ptr == 0 {
		return
	}
	p.mu.Lock()
	p.buckets[s.byteLen] = append(p.buckets[s.byteLen], s)
	p.stats.PoolSize++
	p.mu.Unlock()
}

// FreeAll releases all cached buffers back to the GPU driver.
// Call at shutdown or when switching models.
func (p *Pool) FreeAll() {
	p.mu.Lock()
	defer p.mu.Unlock()
	for size, bufs := range p.buckets {
		for _, s := range bufs {
			p.stats.FreeBytes += int64(s.byteLen)
			p.stats.PoolSize--
			s.Free()
		}
		delete(p.buckets, size)
	}
}

// Stats returns current pool statistics (thread-safe snapshot).
func (p *Pool) Stats() PoolStats {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.stats
}

// Trim releases buffers that haven't been used, keeping at most maxPerBucket
// buffers of each size. Useful for reducing memory after a spike.
func (p *Pool) Trim(maxPerBucket int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for size, bufs := range p.buckets {
		if len(bufs) > maxPerBucket {
			// Free excess
			for _, s := range bufs[maxPerBucket:] {
				p.stats.FreeBytes += int64(s.byteLen)
				p.stats.PoolSize--
				s.Free()
			}
			p.buckets[size] = bufs[:maxPerBucket]
		}
	}
}
