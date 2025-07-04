package metrics

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

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

var (
	apiClientMetricsInstance *APIClientMetrics
	onceAPIClientMetrics     sync.Once
)

func NewAPIClientMetrics() *APIClientMetrics {
	onceAPIClientMetrics.Do(func() {
		apiClientMetricsInstance = &APIClientMetrics{
			RequestsTotal: promauto.NewCounterVec(prometheus.CounterOpts{
				Name: "apiclient_requests_total",
				Help: "Total number of API client requests",
			}, []string{"method", "status"}),

			RequestDuration: promauto.NewHistogramVec(prometheus.HistogramOpts{
				Name:    "apiclient_request_duration_seconds",
				Help:    "Duration of API client requests",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
			}, []string{"method", "status"}),

			RequestSize: promauto.NewHistogram(prometheus.HistogramOpts{
				Name:    "apiclient_request_size_bytes",
				Help:    "Size of API client request bodies",
				Buckets: prometheus.ExponentialBuckets(64, 2, 12),
			}),

			ResponseSize: promauto.NewHistogram(prometheus.HistogramOpts{
				Name:    "apiclient_response_size_bytes",
				Help:    "Size of API client response bodies",
				Buckets: prometheus.ExponentialBuckets(64, 2, 12),
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
	})
	return apiClientMetricsInstance
}

func (m *APIClientMetrics) RecordRequest(method, status string, duration float64) {
	m.RequestsTotal.WithLabelValues(method, status).Inc()
	m.RequestDuration.WithLabelValues(method, status).Observe(duration)
}

func (m *APIClientMetrics) RecordResponse(method, status string) {
}

func (m *APIClientMetrics) RecordRequestSize(size int64) {
	m.RequestSize.Observe(float64(size))
}

func (m *APIClientMetrics) RecordResponseSize(size int64) {
	m.ResponseSize.Observe(float64(size))
}

func (m *APIClientMetrics) RecordError(method, errorType string) {
	m.ErrorsTotal.WithLabelValues(method, errorType).Inc()
}

func (m *APIClientMetrics) RecordRetry(method, reason string) {
	m.RetriesTotal.WithLabelValues(method, reason).Inc()
}

func (m *APIClientMetrics) UpdateActiveConnections(count int) {
	m.ConnectionsActive.Set(float64(count))
}

func (m *APIClientMetrics) RecordNewConnection() {
	m.ConnectionsTotal.Inc()
}
