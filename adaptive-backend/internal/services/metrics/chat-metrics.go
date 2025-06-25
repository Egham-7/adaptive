package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// ChatMetrics holds all Prometheus metrics related to chat completions
type ChatMetrics struct {
	RequestDuration    *prometheus.HistogramVec
	CacheHits          *prometheus.CounterVec
	ProtocolSelections *prometheus.CounterVec
	CacheLookups       *prometheus.CounterVec
}

// NewChatMetrics initializes and registers all chat-related Prometheus metrics
func NewChatMetrics() *ChatMetrics {
	return &ChatMetrics{
		RequestDuration: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Name: "chat_completion_duration_seconds",
			Help: "Duration of chat completion requests",
		}, []string{"method_type", "status", "provider"}),

		CacheHits: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "chat_completion_cache_hits_total",
			Help: "Number of cache hits",
		}, []string{"cache_type", "provider"}),

		ProtocolSelections: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "chat_completion_protocol_selections_total",
			Help: "Number of times each protocol was selected",
		}, []string{"model", "provider"}),
		CacheLookups: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "chat_completion_cache_lookups_total",
			Help: "Number of cache lookups by type (user/global/miss)",
		}, []string{"cache_type", "provider"}),
	}
}
