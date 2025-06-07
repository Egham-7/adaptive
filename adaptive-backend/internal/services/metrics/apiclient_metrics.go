package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// APIClientMetrics holds all Prometheus metrics related to API client operations
type APIClientMetrics struct {
	RequestsTotal     *prometheus.CounterVec
	RequestDuration   *prometheus.HistogramVec
	RequestSize       prometheus.Histogram
	ResponseSize      prometheus.Histogram
	ErrorsTotal       *prometheus.CounterVec
	RetriesTotal      *prometheus.CounterVec
	ConnectionsActive prometheus.Gauge
	ConnectionsTotal  prometheus.Counter
}

// NewAPIClientMetrics initializes and registers all API client-related Prometheus metrics
func NewAPIClientMetrics() *APIClientMetrics {
	return &APIClientMetrics{
		RequestsTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "apiclient_requests_total",
			Help: "Total number of API client requests",
		}, []string{"method", "status"}),

		RequestDuration: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "apiclient_request_duration_seconds",
			Help:    "Duration of API client requests",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 15), // 1ms to ~32s
		}, []string{"method", "status"}),

		RequestSize: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "apiclient_request_size_bytes",
			Help:    "Size of API client request bodies",
			Buckets: prometheus.ExponentialBuckets(64, 2, 12), // 64B to 256KB
		}),

		ResponseSize: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "apiclient_response_size_bytes",
			Help:    "Size of API client response bodies",
			Buckets: prometheus.ExponentialBuckets(64, 2, 12), // 64B to 256KB
		}),

		ErrorsTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "apiclient_errors_total",
			Help: "Total number of API client errors",
		}, []string{"method", "error_type"}),

		RetriesTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "apiclient_retries_total",
			Help: "Total number of API client retries",
		}, []string{"method", "reason"}),

		ConnectionsActive: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "apiclient_connections_active",
			Help: "Current number of active API client connections",
		}),

		ConnectionsTotal: promauto.NewCounter(prometheus.CounterOpts{
			Name: "apiclient_connections_total",
			Help: "Total number of API client connections created",
		}),
	}
}

// RecordRequest records a completed request
func (m *APIClientMetrics) RecordRequest(method, status string, duration float64) {
	m.RequestsTotal.WithLabelValues(method, status).Inc()
	m.RequestDuration.WithLabelValues(method, status).Observe(duration)
}

// RecordResponse records response metrics
func (m *APIClientMetrics) RecordResponse(method, status string) {
	// This is handled in RecordRequest for now
}

// RecordRequestSize records the size of request body
func (m *APIClientMetrics) RecordRequestSize(size int64) {
	m.RequestSize.Observe(float64(size))
}

// RecordResponseSize records the size of response body
func (m *APIClientMetrics) RecordResponseSize(size int64) {
	m.ResponseSize.Observe(float64(size))
}

// RecordError records an error
func (m *APIClientMetrics) RecordError(method, errorType string) {
	m.ErrorsTotal.WithLabelValues(method, errorType).Inc()
}

// RecordRetry records a retry attempt
func (m *APIClientMetrics) RecordRetry(method, reason string) {
	m.RetriesTotal.WithLabelValues(method, reason).Inc()
}

// UpdateActiveConnections updates the active connections gauge
func (m *APIClientMetrics) UpdateActiveConnections(count int) {
	m.ConnectionsActive.Set(float64(count))
}

// RecordNewConnection records a new connection
func (m *APIClientMetrics) RecordNewConnection() {
	m.ConnectionsTotal.Inc()
}