package metrics

import (
	"sync"
	
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// StreamingMetrics holds all Prometheus metrics related to streaming operations
type StreamingMetrics struct {
	StreamDuration        *prometheus.HistogramVec
	StreamsActive         prometheus.Gauge
	StreamsTotal          *prometheus.CounterVec
	BytesStreamed         *prometheus.CounterVec
	StreamErrors          *prometheus.CounterVec
	BufferPoolSize        *prometheus.GaugeVec
	BufferPoolHits        *prometheus.CounterVec
	BufferPoolMisses      *prometheus.CounterVec
	ChunkSize             prometheus.Histogram
	FlushFrequency        prometheus.Histogram
	ConnectionDrops       *prometheus.CounterVec
	BackpressureEvents    prometheus.Counter
	MemoryUsage           prometheus.Gauge
}

var (
	streamingMetricsInstance *StreamingMetrics
	streamingMetricsOnce     sync.Once
)

// GetStreamingMetrics returns the singleton instance of StreamingMetrics
func GetStreamingMetrics() *StreamingMetrics {
	streamingMetricsOnce.Do(func() {
		streamingMetricsInstance = &StreamingMetrics{
			StreamDuration: promauto.NewHistogramVec(prometheus.HistogramOpts{
				Name:    "streaming_duration_seconds",
				Help:    "Duration of streaming operations",
				Buckets: prometheus.ExponentialBuckets(0.1, 2, 12), // 0.1s to ~400s
			}, []string{"provider", "model", "status"}),

			StreamsActive: promauto.NewGauge(prometheus.GaugeOpts{
				Name: "streaming_active_connections",
				Help: "Current number of active streaming connections",
			}),

			StreamsTotal: promauto.NewCounterVec(prometheus.CounterOpts{
				Name: "streaming_connections_total",
				Help: "Total number of streaming connections",
			}, []string{"provider", "model", "status"}),

			BytesStreamed: promauto.NewCounterVec(prometheus.CounterOpts{
				Name: "streaming_bytes_total",
				Help: "Total bytes streamed",
			}, []string{"provider", "direction"}), // direction: "sent", "received"

			StreamErrors: promauto.NewCounterVec(prometheus.CounterOpts{
				Name: "streaming_errors_total",
				Help: "Total number of streaming errors",
			}, []string{"error_type", "provider"}),

			BufferPoolSize: promauto.NewGaugeVec(prometheus.GaugeOpts{
				Name: "streaming_buffer_pool_size",
				Help: "Current size of streaming buffer pools",
			}, []string{"pool_type"}), // "small", "medium", "large"

			BufferPoolHits: promauto.NewCounterVec(prometheus.CounterOpts{
				Name: "streaming_buffer_pool_hits_total",
				Help: "Total number of buffer pool hits",
			}, []string{"pool_type"}),

			BufferPoolMisses: promauto.NewCounterVec(prometheus.CounterOpts{
				Name: "streaming_buffer_pool_misses_total",
				Help: "Total number of buffer pool misses",
			}, []string{"pool_type"}),

			ChunkSize: promauto.NewHistogram(prometheus.HistogramOpts{
				Name:    "streaming_chunk_size_bytes",
				Help:    "Size of streaming chunks in bytes",
				Buckets: prometheus.ExponentialBuckets(64, 2, 12), // 64B to 256KB
			}),

			FlushFrequency: promauto.NewHistogram(prometheus.HistogramOpts{
				Name:    "streaming_flush_interval_seconds",
				Help:    "Time between streaming flushes",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 12), // 1ms to ~4s
			}),

			ConnectionDrops: promauto.NewCounterVec(prometheus.CounterOpts{
				Name: "streaming_connection_drops_total",
				Help: "Total number of dropped streaming connections",
			}, []string{"reason"}), // "timeout", "client_disconnect", "error"

			BackpressureEvents: promauto.NewCounter(prometheus.CounterOpts{
				Name: "streaming_backpressure_events_total",
				Help: "Total number of backpressure events during streaming",
			}),

			MemoryUsage: promauto.NewGauge(prometheus.GaugeOpts{
				Name: "streaming_memory_usage_bytes",
				Help: "Current memory usage by streaming operations",
			}),
		}
	})
	return streamingMetricsInstance
}

// NewStreamingMetrics is deprecated, use GetStreamingMetrics() instead
// Keeping for backward compatibility
func NewStreamingMetrics() *StreamingMetrics {
	return GetStreamingMetrics()
}

// RecordStreamStart records the start of a streaming operation
func (m *StreamingMetrics) RecordStreamStart(provider, model string) {
	m.StreamsActive.Inc()
	m.StreamsTotal.WithLabelValues(provider, model, "started").Inc()
}

// RecordStreamEnd records the end of a streaming operation
func (m *StreamingMetrics) RecordStreamEnd(provider, model, status string, duration float64, bytesStreamed int64) {
	m.StreamsActive.Dec()
	m.StreamsTotal.WithLabelValues(provider, model, status).Inc()
	m.StreamDuration.WithLabelValues(provider, model, status).Observe(duration)
	m.BytesStreamed.WithLabelValues(provider, "sent").Add(float64(bytesStreamed))
}

// RecordError records a streaming error
func (m *StreamingMetrics) RecordError(errorType, provider string) {
	m.StreamErrors.WithLabelValues(errorType, provider).Inc()
}

// RecordBufferPoolHit records a buffer pool hit
func (m *StreamingMetrics) RecordBufferPoolHit(poolType string) {
	m.BufferPoolHits.WithLabelValues(poolType).Inc()
}

// RecordBufferPoolMiss records a buffer pool miss
func (m *StreamingMetrics) RecordBufferPoolMiss(poolType string) {
	m.BufferPoolMisses.WithLabelValues(poolType).Inc()
}

// UpdateBufferPoolSize updates the buffer pool size
func (m *StreamingMetrics) UpdateBufferPoolSize(poolType string, size int) {
	m.BufferPoolSize.WithLabelValues(poolType).Set(float64(size))
}

// RecordChunkSize records the size of a streaming chunk
func (m *StreamingMetrics) RecordChunkSize(size int) {
	m.ChunkSize.Observe(float64(size))
}

// RecordFlushInterval records the time between flushes
func (m *StreamingMetrics) RecordFlushInterval(interval float64) {
	m.FlushFrequency.Observe(interval)
}

// RecordConnectionDrop records a dropped connection
func (m *StreamingMetrics) RecordConnectionDrop(reason string) {
	m.ConnectionDrops.WithLabelValues(reason).Inc()
}

// RecordBackpressure records a backpressure event
func (m *StreamingMetrics) RecordBackpressure() {
	m.BackpressureEvents.Inc()
}

// UpdateMemoryUsage updates the current memory usage
func (m *StreamingMetrics) UpdateMemoryUsage(bytes int64) {
	m.MemoryUsage.Set(float64(bytes))
}