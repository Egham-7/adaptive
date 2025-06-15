package metrics

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

type CircuitBreakerMetrics struct {
	StateChanges     *prometheus.CounterVec
	RequestsTotal    *prometheus.CounterVec
	RequestDuration  *prometheus.HistogramVec
	FailureRate      *prometheus.GaugeVec
	SuccessRate      *prometheus.GaugeVec
	CurrentState     *prometheus.GaugeVec
	TimeInState      *prometheus.CounterVec
	HalfOpenAttempts *prometheus.CounterVec
	TripEvents       *prometheus.CounterVec
	RecoveryEvents   *prometheus.CounterVec
	Timeouts         *prometheus.CounterVec
}

var (
	circuitBreakerMetricsInstance *CircuitBreakerMetrics
	onceCircuitBreakerMetrics     sync.Once
)

func NewCircuitBreakerMetrics() *CircuitBreakerMetrics {
	onceCircuitBreakerMetrics.Do(func() {
		circuitBreakerMetricsInstance = &CircuitBreakerMetrics{
			StateChanges: promauto.NewCounterVec(prometheus.CounterOpts{
				Name: "circuitbreaker_state_changes_total",
				Help: "Total number of circuit breaker state changes",
			}, []string{"service", "from_state", "to_state"}),

			RequestsTotal: promauto.NewCounterVec(prometheus.CounterOpts{
				Name: "circuitbreaker_requests_total",
				Help: "Total number of requests processed by circuit breaker",
			}, []string{"service", "state", "result"}),

			RequestDuration: promauto.NewHistogramVec(prometheus.HistogramOpts{
				Name:    "circuitbreaker_request_duration_seconds",
				Help:    "Duration of requests processed by circuit breaker",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
			}, []string{"service", "state", "result"}),

			FailureRate: promauto.NewGaugeVec(prometheus.GaugeOpts{
				Name: "circuitbreaker_failure_rate",
				Help: "Current failure rate of the circuit breaker (0-1)",
			}, []string{"service"}),

			SuccessRate: promauto.NewGaugeVec(prometheus.GaugeOpts{
				Name: "circuitbreaker_success_rate",
				Help: "Current success rate of the circuit breaker (0-1)",
			}, []string{"service"}),

			CurrentState: promauto.NewGaugeVec(prometheus.GaugeOpts{
				Name: "circuitbreaker_state",
				Help: "Current state of the circuit breaker (0=closed, 1=open, 2=half-open)",
			}, []string{"service"}),

			TimeInState: promauto.NewCounterVec(prometheus.CounterOpts{
				Name: "circuitbreaker_time_in_state_seconds_total",
				Help: "Total time spent in each circuit breaker state",
			}, []string{"service", "state"}),

			HalfOpenAttempts: promauto.NewCounterVec(prometheus.CounterOpts{
				Name: "circuitbreaker_half_open_attempts_total",
				Help: "Total number of requests attempted in half-open state",
			}, []string{"service", "result"}),

			TripEvents: promauto.NewCounterVec(prometheus.CounterOpts{
				Name: "circuitbreaker_trip_events_total",
				Help: "Total number of circuit breaker trip events",
			}, []string{"service", "reason"}),

			RecoveryEvents: promauto.NewCounterVec(prometheus.CounterOpts{
				Name: "circuitbreaker_recovery_events_total",
				Help: "Total number of circuit breaker recovery events",
			}, []string{"service", "from_state"}),

			Timeouts: promauto.NewCounterVec(prometheus.CounterOpts{
				Name: "circuitbreaker_timeouts_total",
				Help: "Total number of circuit breaker timeouts",
			}, []string{"service", "state"}),
		}
	})
	return circuitBreakerMetricsInstance
}

func (m *CircuitBreakerMetrics) RecordStateChange(service, fromState, toState string) {
	m.StateChanges.WithLabelValues(service, fromState, toState).Inc()
}

func (m *CircuitBreakerMetrics) RecordRequest(service, state, result string, duration float64) {
	m.RequestsTotal.WithLabelValues(service, state, result).Inc()
	m.RequestDuration.WithLabelValues(service, state, result).Observe(duration)
}

func (m *CircuitBreakerMetrics) UpdateFailureRate(service string, rate float64) {
	m.FailureRate.WithLabelValues(service).Set(rate)
}

func (m *CircuitBreakerMetrics) UpdateSuccessRate(service string, rate float64) {
	m.SuccessRate.WithLabelValues(service).Set(rate)
}

func (m *CircuitBreakerMetrics) UpdateCurrentState(service string, state int) {
	m.CurrentState.WithLabelValues(service).Set(float64(state))
}

func (m *CircuitBreakerMetrics) RecordTimeInState(service, state string, duration float64) {
	m.TimeInState.WithLabelValues(service, state).Add(duration)
}

func (m *CircuitBreakerMetrics) RecordHalfOpenAttempt(service, result string) {
	m.HalfOpenAttempts.WithLabelValues(service, result).Inc()
}

func (m *CircuitBreakerMetrics) RecordTripEvent(service, reason string) {
	m.TripEvents.WithLabelValues(service, reason).Inc()
}

func (m *CircuitBreakerMetrics) RecordRecoveryEvent(service, fromState string) {
	m.RecoveryEvents.WithLabelValues(service, fromState).Inc()
}

func (m *CircuitBreakerMetrics) RecordTimeout(service, state string) {
	m.Timeouts.WithLabelValues(service, state).Inc()
}

