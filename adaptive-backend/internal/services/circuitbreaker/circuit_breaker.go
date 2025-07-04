package circuitbreaker

import (
	"adaptive-backend/internal/services/metrics"
	"sync"
	"time"
)

type State int

const (
	Closed State = iota
	Open
	HalfOpen
)

type Config struct {
	FailureThreshold int
	SuccessThreshold int
	Timeout          time.Duration
	ResetAfter       time.Duration
}

type CircuitBreaker struct {
	mu              sync.RWMutex
	state           State
	failureCount    int
	successCount    int
	lastFailureTime time.Time
	lastStateChange time.Time
	config          Config
	promMetrics     *metrics.CircuitBreakerMetrics
	localMetrics    LocalMetrics
	serviceName     string
}

type LocalMetrics struct {
	TotalRequests      int64
	SuccessfulRequests int64
	FailedRequests     int64
	CircuitOpens       int64
	CircuitCloses      int64
}

func New() *CircuitBreaker {
	config := Config{
		FailureThreshold: 5,
		SuccessThreshold: 3,
		Timeout:          30 * time.Second,
		ResetAfter:       2 * time.Minute,
	}
	return NewWithConfig(config)
}

func NewWithConfig(config Config) *CircuitBreaker {
	return &CircuitBreaker{
		state:           Closed,
		config:          config,
		promMetrics:     metrics.NewCircuitBreakerMetrics(),
		serviceName:     "ai_service",
		lastStateChange: time.Now(),
	}
}

func (cb *CircuitBreaker) CanExecute() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.localMetrics.TotalRequests++
	cb.promMetrics.RequestsTotal.WithLabelValues(cb.serviceName, cb.getStateString(), "attempted").Inc()

	switch cb.state {
	case Closed:
		return true
	case Open:
		if time.Since(cb.lastFailureTime) > cb.config.Timeout {
			cb.transitionToState(HalfOpen)
			cb.promMetrics.RecordHalfOpenAttempt(cb.serviceName, "transition")
			return true
		}
		cb.promMetrics.RequestsTotal.WithLabelValues(cb.serviceName, "open", "rejected").Inc()
		return false
	case HalfOpen:
		return true
	default:
		return false
	}
}

func (cb *CircuitBreaker) RecordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.localMetrics.SuccessfulRequests++
	cb.failureCount = 0

	cb.promMetrics.RequestsTotal.WithLabelValues(cb.serviceName, cb.getStateString(), "success").Inc()

	if cb.state == HalfOpen {
		cb.successCount++
		cb.promMetrics.RecordHalfOpenAttempt(cb.serviceName, "success")
		if cb.successCount >= cb.config.SuccessThreshold {
			cb.transitionToState(Closed)
			cb.localMetrics.CircuitCloses++
		}
	}
}

func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.localMetrics.FailedRequests++
	cb.failureCount++
	cb.lastFailureTime = time.Now()

	cb.promMetrics.RequestsTotal.WithLabelValues(cb.serviceName, cb.getStateString(), "failure").Inc()

	if cb.state == Closed && cb.failureCount >= cb.config.FailureThreshold {
		cb.transitionToState(Open)
		cb.localMetrics.CircuitOpens++
		cb.promMetrics.RecordTripEvent(cb.serviceName, "failure_threshold")
	} else if cb.state == HalfOpen {
		cb.transitionToState(Open)
		cb.localMetrics.CircuitOpens++
		cb.promMetrics.RecordTripEvent(cb.serviceName, "half_open_failure")
		cb.promMetrics.RecordHalfOpenAttempt(cb.serviceName, "failure")
	}
}

func (cb *CircuitBreaker) GetState() State {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

func (cb *CircuitBreaker) GetMetrics() LocalMetrics {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.localMetrics
}

func (cb *CircuitBreaker) GetSuccessRate() float64 {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	if cb.localMetrics.TotalRequests == 0 {
		return 0.0
	}
	rate := float64(cb.localMetrics.SuccessfulRequests) / float64(cb.localMetrics.TotalRequests)

	cb.promMetrics.UpdateSuccessRate(cb.serviceName, rate)
	cb.promMetrics.UpdateFailureRate(cb.serviceName, 1.0-rate)

	return rate * 100.0
}

func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	oldState := cb.getStateString()
	cb.state = Closed
	cb.failureCount = 0
	cb.successCount = 0
	cb.localMetrics = LocalMetrics{}

	cb.promMetrics.RecordRecoveryEvent(cb.serviceName, oldState)
	cb.promMetrics.UpdateCurrentState(cb.serviceName, int(cb.state))
}

func (cb *CircuitBreaker) RecordRequestDuration(duration time.Duration, success bool) {
	state := cb.getStateString()
	result := "failure"
	if success {
		result = "success"
	}

	cb.promMetrics.RecordRequest(cb.serviceName, state, result, duration.Seconds())
}

func (cb *CircuitBreaker) getStateString() string {
	switch cb.state {
	case Closed:
		return "closed"
	case Open:
		return "open"
	case HalfOpen:
		return "half_open"
	default:
		return "unknown"
	}
}

func (cb *CircuitBreaker) transitionToState(newState State) {
	if cb.state == newState {
		return
	}

	oldState := cb.getStateString()

	timeInState := time.Since(cb.lastStateChange).Seconds()
	cb.promMetrics.RecordTimeInState(cb.serviceName, oldState, timeInState)
	cb.promMetrics.RecordStateChange(cb.serviceName, oldState, cb.stateToString(newState))

	cb.state = newState
	cb.lastStateChange = time.Now()

	cb.promMetrics.UpdateCurrentState(cb.serviceName, int(cb.state))
}

func (cb *CircuitBreaker) stateToString(state State) string {
	switch state {
	case Closed:
		return "closed"
	case Open:
		return "open"
	case HalfOpen:
		return "half_open"
	default:
		return "unknown"
	}
}
