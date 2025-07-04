package cache

import (
	"sync"

	"adaptive-backend/internal/services/metrics"
)

// BufferPool provides a memory pool for reusable byte buffers
// This reduces garbage collection pressure during high-throughput streaming operations
type BufferPool struct {
	pool sync.Pool
	size int
}

// NewBufferPool creates a new buffer pool with the specified buffer size
func NewBufferPool(size int) *BufferPool {
	if size <= 0 {
		size = 4096 // Default 4KB buffers
	}

	return &BufferPool{
		pool: sync.Pool{
			New: func() any {
				buf := make([]byte, size)
				return &buf
			},
		},
		size: size,
	}
}

// Get retrieves a buffer from the pool
func (bp *BufferPool) Get() []byte {
	return *bp.pool.Get().(*[]byte)
}

// Put returns a buffer to the pool for reuse
func (bp *BufferPool) Put(buf []byte) {
	if cap(buf) != bp.size {
		// Don't pool buffers of wrong size
		return
	}

	// Reset the buffer length but keep capacity
	buf = buf[:cap(buf)]
	bp.pool.Put(&buf)
}

// GetSize returns the buffer size used by this pool
func (bp *BufferPool) GetSize() int {
	return bp.size
}

// StreamBufferManager manages different sized buffer pools for various use cases
type StreamBufferManager struct {
	small   *BufferPool // 1KB buffers for small responses
	medium  *BufferPool // 4KB buffers for typical streaming
	large   *BufferPool // 16KB buffers for large responses
	metrics *metrics.StreamingMetrics
}

// NewStreamBufferManager creates a new buffer manager with optimized pool sizes
func NewStreamBufferManager() *StreamBufferManager {
	return &StreamBufferManager{
		small:  NewBufferPool(1024),  // 1KB
		medium: NewBufferPool(4096),  // 4KB
		large:  NewBufferPool(16384), // 16KB
	}
}

// NewStreamBufferManagerWithMetrics creates a new buffer manager with metrics
func NewStreamBufferManagerWithMetrics(metrics *metrics.StreamingMetrics) *StreamBufferManager {
	manager := &StreamBufferManager{
		small:   NewBufferPool(1024),  // 1KB
		medium:  NewBufferPool(4096),  // 4KB
		large:   NewBufferPool(16384), // 16KB
		metrics: metrics,
	}

	// Update initial pool sizes
	manager.updatePoolSizeMetrics()

	return manager
}

// GetBuffer returns an appropriately sized buffer based on the hint
func (sbm *StreamBufferManager) GetBuffer(sizeHint int) []byte {
	var buffer []byte
	var poolType string

	switch {
	case sizeHint <= 1024:
		buffer = sbm.small.Get()
		poolType = "small"
	case sizeHint <= 4096:
		buffer = sbm.medium.Get()
		poolType = "medium"
	default:
		buffer = sbm.large.Get()
		poolType = "large"
	}

	if sbm.metrics != nil {
		sbm.metrics.RecordBufferPoolHit(poolType)
	}

	return buffer
}

// PutBuffer returns a buffer to the appropriate pool
func (sbm *StreamBufferManager) PutBuffer(buf []byte) {
	size := cap(buf)
	var poolType string

	switch size {
	case 1024:
		sbm.small.Put(buf)
		poolType = "small"
	case 4096:
		sbm.medium.Put(buf)
		poolType = "medium"
	case 16384:
		sbm.large.Put(buf)
		poolType = "large"
	default:
		// Don't pool buffers of unexpected sizes
		if sbm.metrics != nil {
			sbm.metrics.RecordBufferPoolMiss("unknown")
		}
		return
	}

	if sbm.metrics != nil && poolType != "" {
		sbm.updatePoolSizeMetrics()
	}
}

// GetOptimalBuffer returns a buffer based on expected data size
func (sbm *StreamBufferManager) GetOptimalBuffer(expectedSize int) []byte {
	var buffer []byte
	var poolType string

	// Choose buffer size that minimizes allocations while avoiding waste
	switch {
	case expectedSize <= 512:
		buffer = sbm.small.Get()[:expectedSize]
		poolType = "small"
	case expectedSize <= 2048:
		buffer = sbm.medium.Get()[:expectedSize]
		poolType = "medium"
	case expectedSize <= 8192:
		buffer = sbm.large.Get()[:expectedSize]
		poolType = "large"
	default:
		// For very large expected sizes, allocate directly
		buffer = make([]byte, expectedSize)
		if sbm.metrics != nil {
			sbm.metrics.RecordBufferPoolMiss("oversized")
		}
		return buffer
	}

	if sbm.metrics != nil {
		sbm.metrics.RecordBufferPoolHit(poolType)
	}

	return buffer
}

// Global buffer manager instance
var (
	globalBufferManager *StreamBufferManager
	bufferManagerOnce   sync.Once
)

// GetGlobalBufferManager returns the singleton buffer manager
func GetGlobalBufferManager() *StreamBufferManager {
	bufferManagerOnce.Do(func() {
		globalBufferManager = NewStreamBufferManager()
	})
	return globalBufferManager
}

// SetGlobalBufferManagerMetrics sets metrics for the global buffer manager
func SetGlobalBufferManagerMetrics(metrics *metrics.StreamingMetrics) {
	manager := GetGlobalBufferManager()
	manager.metrics = metrics
	manager.updatePoolSizeMetrics()
}

// updatePoolSizeMetrics updates the Prometheus metrics for pool sizes
func (sbm *StreamBufferManager) updatePoolSizeMetrics() {
	if sbm.metrics == nil {
		return
	}

	// Note: We can't easily get the current pool sizes from sync.Pool
	// So we'll track this differently or estimate based on usage patterns
	// For now, we'll just update when buffers are returned
}
