package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// DatabaseMetrics holds all Prometheus metrics related to database operations
type DatabaseMetrics struct {
	ConnectionPoolStats   *prometheus.GaugeVec
	QueryDuration        *prometheus.HistogramVec
	ActiveConnections    prometheus.Gauge
	IdleConnections      prometheus.Gauge
	WaitCount            prometheus.Counter
	WaitDuration         prometheus.Histogram
	MaxIdleClosed        prometheus.Counter
	MaxLifetimeClosed    prometheus.Counter
	QueryErrors          *prometheus.CounterVec
	TransactionDuration  *prometheus.HistogramVec
}

// NewDatabaseMetrics initializes and registers all database-related Prometheus metrics
func NewDatabaseMetrics() *DatabaseMetrics {
	return &DatabaseMetrics{
		ConnectionPoolStats: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: "database_connection_pool",
			Help: "Database connection pool statistics",
		}, []string{"stat_type"}),

		QueryDuration: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "database_query_duration_seconds",
			Help:    "Duration of database queries",
			Buckets: prometheus.DefBuckets,
		}, []string{"operation", "table"}),

		ActiveConnections: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "database_connections_active",
			Help: "Number of active database connections",
		}),

		IdleConnections: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "database_connections_idle",
			Help: "Number of idle database connections",
		}),

		WaitCount: promauto.NewCounter(prometheus.CounterOpts{
			Name: "database_connection_wait_total",
			Help: "Total number of times waited for a database connection",
		}),

		WaitDuration: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "database_connection_wait_duration_seconds",
			Help:    "Time spent waiting for database connections",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
		}),

		MaxIdleClosed: promauto.NewCounter(prometheus.CounterOpts{
			Name: "database_connections_closed_max_idle_total",
			Help: "Total number of connections closed due to max idle",
		}),

		MaxLifetimeClosed: promauto.NewCounter(prometheus.CounterOpts{
			Name: "database_connections_closed_max_lifetime_total",
			Help: "Total number of connections closed due to max lifetime",
		}),

		QueryErrors: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "database_query_errors_total",
			Help: "Total number of database query errors",
		}, []string{"operation", "error_type"}),

		TransactionDuration: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "database_transaction_duration_seconds",
			Help:    "Duration of database transactions",
			Buckets: prometheus.DefBuckets,
		}, []string{"operation"}),
	}
}

// UpdateConnectionPoolStats updates the connection pool statistics
func (m *DatabaseMetrics) UpdateConnectionPoolStats(maxOpen, open, inUse, idle int, waitCount, maxIdleClosed, maxLifetimeClosed int64) {
	m.ConnectionPoolStats.WithLabelValues("max_open").Set(float64(maxOpen))
	m.ConnectionPoolStats.WithLabelValues("open").Set(float64(open))
	m.ConnectionPoolStats.WithLabelValues("in_use").Set(float64(inUse))
	m.ConnectionPoolStats.WithLabelValues("idle").Set(float64(idle))
	
	m.ActiveConnections.Set(float64(inUse))
	m.IdleConnections.Set(float64(idle))
}