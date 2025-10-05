package utils

import (
	"sync"

	"github.com/valyala/bytebufferpool"
)

// BufferPool provides efficient buffer pooling for streaming operations
// Uses bytebufferpool for automatic size-class management and anti-fragmentation
type BufferPool struct {
	pool *bytebufferpool.Pool
}

// Global buffer pool instance
var (
	globalPool     *BufferPool
	globalPoolOnce sync.Once
)

// NewBufferPool creates a new buffer pool
func NewBufferPool() *BufferPool {
	return &BufferPool{
		pool: &bytebufferpool.Pool{},
	}
}

// Get retrieves a buffer from the pool
func (bp *BufferPool) Get() *bytebufferpool.ByteBuffer {
	return bp.pool.Get()
}

// Put returns a buffer to the pool
func (bp *BufferPool) Put(buf *bytebufferpool.ByteBuffer) {
	bp.pool.Put(buf)
}

// Global returns the global buffer pool instance
func Global() *BufferPool {
	globalPoolOnce.Do(func() {
		globalPool = NewBufferPool()
	})
	return globalPool
}

// Get is a convenience function that uses the global pool
func Get() *bytebufferpool.ByteBuffer {
	return Global().Get()
}

// Put is a convenience function that uses the global pool
func Put(buf *bytebufferpool.ByteBuffer) {
	Global().Put(buf)
}
