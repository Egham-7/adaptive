package circuitbreaker

import (
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
		serviceName:     "ai_service",
		lastStateChange: time.Now(),
	}
}

func (cb *CircuitBreaker) CanExecute() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.localMetrics.TotalRequests++

	switch cb.state {
	case Closed:
		return true
	case Open:
		if time.Since(cb.lastFailureTime) > cb.config.Timeout {
			cb.transitionToState(HalfOpen)
			return true
		}
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

	if cb.state == HalfOpen {
		cb.successCount++
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

	if cb.state == Closed && cb.failureCount >= cb.config.FailureThreshold {
		cb.transitionToState(Open)
		cb.localMetrics.CircuitOpens++
	} else if cb.state == HalfOpen {
		cb.transitionToState(Open)
		cb.localMetrics.CircuitOpens++
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

	return rate * 100.0
}

func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.state = Closed
	cb.failureCount = 0
	cb.successCount = 0
	cb.localMetrics = LocalMetrics{}
}

func (cb *CircuitBreaker) RecordRequestDuration(duration time.Duration, success bool) {
	// Duration recording functionality removed with metrics
}

func (cb *CircuitBreaker) transitionToState(newState State) {
	if cb.state == newState {
		return
	}

	cb.state = newState
	cb.lastStateChange = time.Now()
}
