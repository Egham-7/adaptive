package metrics

import (
	"runtime"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// SystemMetrics holds all Prometheus metrics related to system-level operations
type SystemMetrics struct {
	MemoryUsage          *prometheus.GaugeVec
	GoroutineCount       prometheus.Gauge
	GCDuration           prometheus.Histogram
	GCFrequency          prometheus.Counter
	CPUUsage             prometheus.Gauge
	HeapObjects          prometheus.Gauge
	StackInuse           prometheus.Gauge
	NextGC               prometheus.Gauge
	LastGC               prometheus.Gauge
	Uptime               prometheus.Gauge
	GoVersion            *prometheus.GaugeVec
	NumCPU               prometheus.Gauge
	CGOCalls             prometheus.Counter
	ThreadsCreated       prometheus.Counter
	MemoryAllocations    prometheus.Counter
	MemoryDeallocations  prometheus.Counter
	LookupLatency        prometheus.Histogram
	MutexContention      prometheus.Counter
}

var startTime = time.Now()

// NewSystemMetrics initializes and registers all system-related Prometheus metrics
func NewSystemMetrics() *SystemMetrics {
	return &SystemMetrics{
		MemoryUsage: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: "system_memory_bytes",
			Help: "Current memory usage by type",
		}, []string{"type"}), // "alloc", "total_alloc", "sys", "heap_alloc", "heap_sys", "heap_idle", "heap_inuse", "stack_inuse"

		GoroutineCount: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "system_goroutines",
			Help: "Current number of goroutines",
		}),

		GCDuration: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "system_gc_duration_seconds",
			Help:    "Duration of garbage collection cycles",
			Buckets: prometheus.ExponentialBuckets(0.00001, 2, 20), // 10μs to ~10s
		}),

		GCFrequency: promauto.NewCounter(prometheus.CounterOpts{
			Name: "system_gc_cycles_total",
			Help: "Total number of garbage collection cycles",
		}),

		CPUUsage: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "system_cpu_usage_percent",
			Help: "Current CPU usage percentage",
		}),

		HeapObjects: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "system_heap_objects",
			Help: "Current number of heap objects",
		}),

		StackInuse: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "system_stack_inuse_bytes",
			Help: "Current stack memory in use",
		}),

		NextGC: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "system_next_gc_bytes",
			Help: "Target heap size for next garbage collection",
		}),

		LastGC: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "system_last_gc_timestamp",
			Help: "Timestamp of last garbage collection",
		}),

		Uptime: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "system_uptime_seconds",
			Help: "System uptime in seconds",
		}),

		GoVersion: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: "system_go_info",
			Help: "Go runtime version information",
		}, []string{"version"}),

		NumCPU: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "system_cpu_cores",
			Help: "Number of CPU cores available",
		}),

		CGOCalls: promauto.NewCounter(prometheus.CounterOpts{
			Name: "system_cgo_calls_total",
			Help: "Total number of CGO calls",
		}),

		ThreadsCreated: promauto.NewCounter(prometheus.CounterOpts{
			Name: "system_threads_created_total",
			Help: "Total number of OS threads created",
		}),

		MemoryAllocations: promauto.NewCounter(prometheus.CounterOpts{
			Name: "system_memory_allocations_total",
			Help: "Total number of memory allocations",
		}),

		MemoryDeallocations: promauto.NewCounter(prometheus.CounterOpts{
			Name: "system_memory_deallocations_total",
			Help: "Total number of memory deallocations",
		}),

		LookupLatency: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "system_dns_lookup_duration_seconds",
			Help:    "DNS lookup latency",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 12), // 1ms to ~4s
		}),

		MutexContention: promauto.NewCounter(prometheus.CounterOpts{
			Name: "system_mutex_contention_total",
			Help: "Total number of mutex contention events",
		}),
	}
}

// UpdateMemoryStats updates all memory-related metrics
func (m *SystemMetrics) UpdateMemoryStats() {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	m.MemoryUsage.WithLabelValues("alloc").Set(float64(memStats.Alloc))
	m.MemoryUsage.WithLabelValues("total_alloc").Set(float64(memStats.TotalAlloc))
	m.MemoryUsage.WithLabelValues("sys").Set(float64(memStats.Sys))
	m.MemoryUsage.WithLabelValues("heap_alloc").Set(float64(memStats.HeapAlloc))
	m.MemoryUsage.WithLabelValues("heap_sys").Set(float64(memStats.HeapSys))
	m.MemoryUsage.WithLabelValues("heap_idle").Set(float64(memStats.HeapIdle))
	m.MemoryUsage.WithLabelValues("heap_inuse").Set(float64(memStats.HeapInuse))
	m.MemoryUsage.WithLabelValues("stack_inuse").Set(float64(memStats.StackInuse))

	m.HeapObjects.Set(float64(memStats.HeapObjects))
	m.StackInuse.Set(float64(memStats.StackInuse))
	m.NextGC.Set(float64(memStats.NextGC))
	m.LastGC.Set(float64(memStats.LastGC))

	// Update allocation counters (delta since last update would be ideal, but this is simpler)
	m.MemoryAllocations.Add(float64(memStats.Mallocs))
	m.MemoryDeallocations.Add(float64(memStats.Frees))

	// Update GC metrics
	m.GCFrequency.Add(float64(memStats.NumGC))
}

// UpdateRuntimeStats updates runtime-related metrics
func (m *SystemMetrics) UpdateRuntimeStats() {
	m.GoroutineCount.Set(float64(runtime.NumGoroutine()))
	m.NumCPU.Set(float64(runtime.NumCPU()))
	m.Uptime.Set(time.Since(startTime).Seconds())
	m.GoVersion.WithLabelValues(runtime.Version()).Set(1)
}

// RecordGCDuration records a garbage collection duration
func (m *SystemMetrics) RecordGCDuration(duration time.Duration) {
	m.GCDuration.Observe(duration.Seconds())
}

// RecordCGOCall records a CGO call
func (m *SystemMetrics) RecordCGOCall() {
	m.CGOCalls.Inc()
}

// RecordThreadCreation records OS thread creation
func (m *SystemMetrics) RecordThreadCreation() {
	m.ThreadsCreated.Inc()
}

// UpdateCPUUsage updates the CPU usage percentage
func (m *SystemMetrics) UpdateCPUUsage(percent float64) {
	m.CPUUsage.Set(percent)
}

// RecordDNSLookupLatency records DNS lookup latency
func (m *SystemMetrics) RecordDNSLookupLatency(duration time.Duration) {
	m.LookupLatency.Observe(duration.Seconds())
}

// RecordMutexContention records mutex contention event
func (m *SystemMetrics) RecordMutexContention() {
	m.MutexContention.Inc()
}

// StartPeriodicUpdates starts a goroutine that periodically updates system metrics
func (m *SystemMetrics) StartPeriodicUpdates(interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for range ticker.C {
			m.UpdateMemoryStats()
			m.UpdateRuntimeStats()
		}
	}()
}