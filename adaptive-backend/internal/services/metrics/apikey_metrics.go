package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// APIKeyMetrics holds all Prometheus metrics related to API key operations
type APIKeyMetrics struct {
	VerificationDuration *prometheus.HistogramVec
	VerificationTotal    *prometheus.CounterVec
	CacheHits           *prometheus.CounterVec
	CacheMisses         *prometheus.CounterVec
	CacheSize           prometheus.Gauge
	CacheEvictions      prometheus.Counter
	KeyCreationTotal    prometheus.Counter
	KeyDeletionTotal    prometheus.Counter
	KeyUpdateTotal      prometheus.Counter
	ActiveKeys          prometheus.Gauge
	ExpiredKeys         prometheus.Counter
}

// NewAPIKeyMetrics initializes and registers all API key-related Prometheus metrics
func NewAPIKeyMetrics() *APIKeyMetrics {
	return &APIKeyMetrics{
		VerificationDuration: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "apikey_verification_duration_seconds",
			Help:    "Duration of API key verification operations",
			Buckets: prometheus.ExponentialBuckets(0.0001, 2, 15), // 0.1ms to ~3.2s
		}, []string{"result"}), // "success", "failed", "expired", "invalid"

		VerificationTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "apikey_verification_total",
			Help: "Total number of API key verification attempts",
		}, []string{"result", "source"}), // result: success/failed, source: cache/database

		CacheHits: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "apikey_cache_hits_total",
			Help: "Total number of API key cache hits",
		}, []string{"cache_type"}), // "valid", "invalid"

		CacheMisses: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "apikey_cache_misses_total",
			Help: "Total number of API key cache misses",
		}, []string{"reason"}), // "not_found", "expired", "invalid_format"

		CacheSize: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "apikey_cache_size",
			Help: "Current number of entries in API key cache",
		}),

		CacheEvictions: promauto.NewCounter(prometheus.CounterOpts{
			Name: "apikey_cache_evictions_total",
			Help: "Total number of API key cache evictions",
		}),

		KeyCreationTotal: promauto.NewCounter(prometheus.CounterOpts{
			Name: "apikey_creation_total",
			Help: "Total number of API keys created",
		}),

		KeyDeletionTotal: promauto.NewCounter(prometheus.CounterOpts{
			Name: "apikey_deletion_total",
			Help: "Total number of API keys deleted",
		}),

		KeyUpdateTotal: promauto.NewCounter(prometheus.CounterOpts{
			Name: "apikey_update_total",
			Help: "Total number of API key updates",
		}),

		ActiveKeys: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "apikey_active_total",
			Help: "Current number of active API keys",
		}),

		ExpiredKeys: promauto.NewCounter(prometheus.CounterOpts{
			Name: "apikey_expired_total",
			Help: "Total number of expired API keys encountered",
		}),
	}
}