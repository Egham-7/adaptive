package circuitbreaker

import (
	"context"
	"fmt"
	"strconv"
	"time"

	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/redis/go-redis/v9"
)

type State int

const (
	Closed State = iota
	Open
	HalfOpen
)

func (s State) String() string {
	switch s {
	case Closed:
		return "Closed"
	case Open:
		return "Open"
	case HalfOpen:
		return "HalfOpen"
	default:
		return fmt.Sprintf("Unknown(%d)", int(s))
	}
}

type Config struct {
	FailureThreshold int
	SuccessThreshold int
	Timeout          time.Duration
	ResetAfter       time.Duration
}

const (
	circuitBreakerKeyPrefix = "circuit_breaker:"
	stateKey                = "state"
	failureCountKey         = "failure_count"
	successCountKey         = "success_count"
	lastFailureTimeKey      = "last_failure_time"
	lastStateChangeKey      = "last_state_change"
	defaultTimeout          = 1 * time.Second
	maxRetries              = 3
)

type CircuitBreaker struct {
	redisClient *redis.Client
	serviceName string
	config      Config
	keyPrefix   string
}

type keyBuilder struct {
	prefix string
}

func (kb *keyBuilder) state() string        { return kb.prefix + stateKey }
func (kb *keyBuilder) failureCount() string { return kb.prefix + failureCountKey }
func (kb *keyBuilder) successCount() string { return kb.prefix + successCountKey }
func (kb *keyBuilder) lastFailure() string  { return kb.prefix + lastFailureTimeKey }
func (kb *keyBuilder) lastChange() string   { return kb.prefix + lastStateChangeKey }

type LocalMetrics struct {
	TotalRequests      int64
	SuccessfulRequests int64
	FailedRequests     int64
	CircuitOpens       int64
	CircuitCloses      int64
}

func New(redisClient *redis.Client) *CircuitBreaker {
	config := Config{
		FailureThreshold: 5,
		SuccessThreshold: 3,
		Timeout:          30 * time.Second,
		ResetAfter:       2 * time.Minute,
	}
	return NewWithConfig(redisClient, "default", config)
}

func NewForProvider(redisClient *redis.Client, providerName string) *CircuitBreaker {
	config := Config{
		FailureThreshold: 5,
		SuccessThreshold: 3,
		Timeout:          30 * time.Second,
		ResetAfter:       2 * time.Minute,
	}
	return NewWithConfig(redisClient, providerName, config)
}

func NewWithConfig(redisClient *redis.Client, serviceName string, config Config) *CircuitBreaker {
	keyPrefix := circuitBreakerKeyPrefix + serviceName + ":"

	// Verify Redis connection health
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := redisClient.Ping(ctx).Err(); err != nil {
		fiberlog.Errorf("Redis connection failed for circuit breaker %s: %v", serviceName, err)
	}

	cb := &CircuitBreaker{
		redisClient: redisClient,
		serviceName: serviceName,
		config:      config,
		keyPrefix:   keyPrefix,
	}

	cb.initializeState()
	return cb
}

func (cb *CircuitBreaker) initializeState() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	exists, err := cb.redisClient.Exists(ctx, cb.keyPrefix+stateKey).Result()
	if err != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to check state existence: %v", err)
		return
	}

	if exists == 0 {
		pipe := cb.redisClient.Pipeline()
		pipe.Set(ctx, cb.keyPrefix+stateKey, int(Closed), 0)
		pipe.Set(ctx, cb.keyPrefix+failureCountKey, 0, 0)
		pipe.Set(ctx, cb.keyPrefix+successCountKey, 0, 0)
		pipe.Set(ctx, cb.keyPrefix+lastStateChangeKey, time.Now().Unix(), 0)

		_, err := pipe.Exec(ctx)
		if err != nil {
			fiberlog.Errorf("CircuitBreaker: Failed to initialize state: %v", err)
		} else {
			fiberlog.Debugf("CircuitBreaker: Initialized state for service %s", cb.serviceName)
		}
	}
}

func (cb *CircuitBreaker) CanExecute() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	state, err := cb.getState(ctx)
	if err != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to get state, allowing execution: %v", err)
		return true
	}

	switch state {
	case Closed:
		return true
	case Open:
		lastFailureTime, err := cb.redisClient.Get(ctx, cb.keyPrefix+lastFailureTimeKey).Int64()
		if err != nil {
			fiberlog.Errorf("CircuitBreaker: Failed to get last failure time: %v", err)
			return false
		}

		if time.Since(time.Unix(lastFailureTime, 0)) > cb.config.Timeout {
			if cb.transitionToState(HalfOpen) {
				return true
			}
		}
		return false
	case HalfOpen:
		return true
	default:
		return false
	}
}

func (cb *CircuitBreaker) RecordSuccess() {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	kb := &keyBuilder{prefix: cb.keyPrefix}

	// Use Redis transactions for atomic operations with retries
	for attempt := range maxRetries {
		err := cb.redisClient.Watch(ctx, func(tx *redis.Tx) error {
			// Get current state within transaction
			state, err := cb.getState(ctx)
			if err != nil {
				return fmt.Errorf("failed to get current state: %w", err)
			}

			// Start pipeline
			pipe := tx.TxPipeline()

			// Always reset failure count on success
			pipe.Set(ctx, kb.failureCount(), 0, 0)

			if state == HalfOpen {
				// Increment success count
				pipe.Incr(ctx, kb.successCount())

				// Get current success count to check threshold
				successCountCmd := pipe.Get(ctx, kb.successCount())

				// Execute pipeline to get success count
				_, err := pipe.Exec(ctx)
				if err != nil {
					return err
				}

				// Check if we should transition to Closed
				successCount, err := successCountCmd.Int64()
				if err == nil && successCount >= int64(cb.config.SuccessThreshold) {
					// Start new pipeline for state transition
					pipe2 := tx.TxPipeline()
					pipe2.Set(ctx, kb.state(), int(Closed), 0)
					pipe2.Set(ctx, kb.successCount(), 0, 0)
					pipe2.Set(ctx, kb.lastChange(), time.Now().Unix(), 0)
					_, err = pipe2.Exec(ctx)
					if err != nil {
						return err
					}
					fiberlog.Debugf("CircuitBreaker: %s transitioned to Closed state", cb.serviceName)
				} else {
					fiberlog.Debugf("CircuitBreaker: %s recorded success in HalfOpen state", cb.serviceName)
				}
			} else {
				// Just execute the failure count reset
				_, err = pipe.Exec(ctx)
				if err != nil {
					return err
				}
				fiberlog.Debugf("CircuitBreaker: %s recorded success", cb.serviceName)
			}

			return nil
		}, kb.state(), kb.successCount())

		if err == nil {
			return // Success
		}

		if err != redis.TxFailedErr {
			fiberlog.Errorf("CircuitBreaker: Failed to record success: %v", err)
			return
		}

		// Retry on transaction failure with backoff
		if attempt < maxRetries-1 {
			time.Sleep(time.Duration(attempt+1) * 10 * time.Millisecond)
		}
	}

	fiberlog.Errorf("CircuitBreaker: Failed to record success after %d attempts", maxRetries)
}

func (cb *CircuitBreaker) RecordFailure() {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	kb := &keyBuilder{prefix: cb.keyPrefix}

	// Use Redis transactions for atomic operations with retries
	for attempt := range maxRetries {
		err := cb.redisClient.Watch(ctx, func(tx *redis.Tx) error {
			// Get current state within transaction
			state, err := cb.getState(ctx)
			if err != nil {
				return fmt.Errorf("failed to get current state: %w", err)
			}

			// Start pipeline
			pipe := tx.TxPipeline()

			// Increment failure count and set last failure time
			pipe.Incr(ctx, kb.failureCount())
			pipe.Set(ctx, kb.lastFailure(), time.Now().Unix(), 0)

			// Get the incremented failure count
			failureCountCmd := pipe.Get(ctx, kb.failureCount())

			// Execute pipeline to get failure count
			_, err = pipe.Exec(ctx)
			if err != nil {
				return err
			}

			// Check if we should transition to Open
			failureCount, err := failureCountCmd.Int64()
			if err != nil {
				return fmt.Errorf("failed to get failure count: %w", err)
			}

			shouldTransitionToOpen := (state == Closed && failureCount >= int64(cb.config.FailureThreshold)) ||
				state == HalfOpen

			if shouldTransitionToOpen {
				// Start new pipeline for state transition
				pipe2 := tx.TxPipeline()
				pipe2.Set(ctx, kb.state(), int(Open), 0)
				pipe2.Set(ctx, kb.lastChange(), time.Now().Unix(), 0)
				_, err = pipe2.Exec(ctx)
				if err != nil {
					return err
				}
				fiberlog.Debugf("CircuitBreaker: %s transitioned to Open state", cb.serviceName)
			} else {
				fiberlog.Debugf("CircuitBreaker: %s recorded failure", cb.serviceName)
			}

			return nil
		}, kb.state(), kb.failureCount())

		if err == nil {
			return // Success
		}

		if err != redis.TxFailedErr {
			fiberlog.Errorf("CircuitBreaker: Failed to record failure: %v", err)
			return
		}

		// Retry on transaction failure with backoff
		if attempt < maxRetries-1 {
			time.Sleep(time.Duration(attempt+1) * 10 * time.Millisecond)
		}
	}

	fiberlog.Errorf("CircuitBreaker: Failed to record failure after %d attempts", maxRetries)
}

func (cb *CircuitBreaker) GetState() State {
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	state, err := cb.getState(ctx)
	if err != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to get state, returning Closed: %v", err)
		return Closed
	}
	return state
}

func (cb *CircuitBreaker) Reset() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	pipe := cb.redisClient.Pipeline()
	pipe.Set(ctx, cb.keyPrefix+stateKey, int(Closed), 0)
	pipe.Set(ctx, cb.keyPrefix+failureCountKey, 0, 0)
	pipe.Set(ctx, cb.keyPrefix+successCountKey, 0, 0)
	pipe.Set(ctx, cb.keyPrefix+lastStateChangeKey, time.Now().Unix(), 0)

	_, err := pipe.Exec(ctx)
	if err != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to reset state: %v", err)
	} else {
		fiberlog.Infof("CircuitBreaker: Reset circuit breaker for service %s", cb.serviceName)
	}
}

func (cb *CircuitBreaker) getState(ctx context.Context) (State, error) {
	kb := &keyBuilder{prefix: cb.keyPrefix}
	stateStr, err := cb.redisClient.Get(ctx, kb.state()).Result()
	if err != nil {
		return Closed, fmt.Errorf("failed to get circuit breaker state: %w", err)
	}

	stateInt, err := strconv.Atoi(stateStr)
	if err != nil {
		return Closed, fmt.Errorf("invalid state value '%s': %w", stateStr, err)
	}

	return State(stateInt), nil
}

func (cb *CircuitBreaker) transitionToState(newState State) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	kb := &keyBuilder{prefix: cb.keyPrefix}

	// Use optimistic locking with retries
	for attempt := range maxRetries {
		err := cb.redisClient.Watch(ctx, func(tx *redis.Tx) error {
			currentState, err := cb.getState(ctx)
			if err != nil {
				return err
			}

			if currentState == newState {
				return nil // Already in desired state
			}

			pipe := tx.TxPipeline()
			pipe.Set(ctx, kb.state(), int(newState), 0)
			pipe.Set(ctx, kb.lastChange(), time.Now().Unix(), 0)

			if newState != HalfOpen {
				pipe.Set(ctx, kb.successCount(), 0, 0)
			}

			_, err = pipe.Exec(ctx)
			return err
		}, kb.state())

		if err == nil {
			fiberlog.Debugf("CircuitBreaker: %s transitioned to %s", cb.serviceName, newState)
			return true
		}

		if err != redis.TxFailedErr {
			fiberlog.Errorf("CircuitBreaker: %s state transition failed: %v", cb.serviceName, err)
			return false
		}

		// Retry on transaction failure
		time.Sleep(time.Duration(attempt+1) * 10 * time.Millisecond)
	}

	fiberlog.Errorf("CircuitBreaker: %s state transition failed after %d attempts", cb.serviceName, maxRetries)
	return false
}
