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
	defaultLockTTL          = 5 * time.Second
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
func (kb *keyBuilder) lock() string         { return kb.prefix + "lock" }

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
	lockKey := kb.lock()

	// Attempt to acquire lock with retries
	locked, lockErr := cb.acquireLockWithRetry(ctx, lockKey)
	if lockErr != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to acquire lock for success recording after retries: %v", lockErr)
		// Don't return - attempt to record success anyway for resilience
		// In high-concurrency scenarios, it's better to have potential duplicate updates than lost metrics
		cb.recordSuccessWithoutLock(ctx)
		return
	}
	if !locked {
		fiberlog.Warnf("CircuitBreaker: Could not acquire lock for success recording, attempting without lock")
		cb.recordSuccessWithoutLock(ctx)
		return
	}

	defer func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), defaultTimeout)
		defer cleanupCancel()
		if err := cb.redisClient.Del(cleanupCtx, lockKey).Err(); err != nil {
			fiberlog.Errorf("CircuitBreaker: Failed to release lock %s: %v", lockKey, err)
		}
	}()

	// Record success with lock protection
	if err := cb.recordSuccessWithLock(ctx, kb); err != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to record success: %v", err)
	}
}

func (cb *CircuitBreaker) RecordFailure() {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	kb := &keyBuilder{prefix: cb.keyPrefix}
	lockKey := kb.lock()

	// Attempt to acquire lock with retries
	locked, lockErr := cb.acquireLockWithRetry(ctx, lockKey)
	if lockErr != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to acquire lock for failure recording after retries: %v", lockErr)
		// Don't return - attempt to record failure anyway for resilience
		// In high-concurrency scenarios, it's better to have potential duplicate updates than lost metrics
		cb.recordFailureWithoutLock(ctx)
		return
	}
	if !locked {
		fiberlog.Warnf("CircuitBreaker: Could not acquire lock for failure recording, attempting without lock")
		cb.recordFailureWithoutLock(ctx)
		return
	}

	defer func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), defaultTimeout)
		defer cleanupCancel()
		if err := cb.redisClient.Del(cleanupCtx, lockKey).Err(); err != nil {
			fiberlog.Errorf("CircuitBreaker: Failed to release lock %s: %v", lockKey, err)
		}
	}()

	// Record failure with lock protection
	if err := cb.recordFailureWithLock(ctx, kb); err != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to record failure: %v", err)
	}
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

// acquireLockWithRetry attempts to acquire a lock with exponential backoff retry
func (cb *CircuitBreaker) acquireLockWithRetry(ctx context.Context, lockKey string) (bool, error) {
	const maxLockRetries = 3
	baseDelay := 10 * time.Millisecond

	for attempt := range maxLockRetries {
		locked, err := cb.redisClient.SetNX(ctx, lockKey, "1", defaultLockTTL).Result()
		if err != nil {
			// Redis error - return error instead of silent failure
			return false, fmt.Errorf("lock acquisition failed on attempt %d: %w", attempt+1, err)
		}

		if locked {
			return true, nil
		}

		// Exponential backoff before retry
		if attempt < maxLockRetries-1 {
			delay := time.Duration(1<<attempt) * baseDelay
			select {
			case <-ctx.Done():
				return false, ctx.Err()
			case <-time.After(delay):
				// Continue to next attempt
			}
		}
	}

	return false, nil // Could not acquire lock after all retries
}

// recordSuccessWithoutLock records success without lock protection (fallback)
func (cb *CircuitBreaker) recordSuccessWithoutLock(ctx context.Context) {
	kb := &keyBuilder{prefix: cb.keyPrefix}

	// Use atomic operations where possible to minimize inconsistency
	if err := cb.redisClient.Set(ctx, kb.failureCount(), 0, 0).Err(); err != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to reset failure count: %v", err)
	}

	state, err := cb.getState(ctx)
	if err != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to get state in fallback success recording: %v", err)
		return
	}

	if state == HalfOpen {
		// Use atomic increment even without lock
		successCount, err := cb.redisClient.Incr(ctx, kb.successCount()).Result()
		if err != nil {
			fiberlog.Errorf("CircuitBreaker: Failed to increment success count in fallback: %v", err)
			return
		}

		if successCount >= int64(cb.config.SuccessThreshold) {
			// State transition uses its own locking mechanism
			if cb.transitionToState(Closed) {
				fiberlog.Debugf("CircuitBreaker: Successfully transitioned to Closed state in fallback")
			}
		}
	}
}

// recordSuccessWithLock records success with lock protection
func (cb *CircuitBreaker) recordSuccessWithLock(ctx context.Context, kb *keyBuilder) error {
	// Reset failure count
	if err := cb.redisClient.Set(ctx, kb.failureCount(), 0, 0).Err(); err != nil {
		return fmt.Errorf("failed to reset failure count: %w", err)
	}

	state, err := cb.getState(ctx)
	if err != nil {
		return fmt.Errorf("failed to get current state: %w", err)
	}

	if state == HalfOpen {
		successCount, err := cb.redisClient.Incr(ctx, kb.successCount()).Result()
		if err != nil {
			return fmt.Errorf("failed to increment success count: %w", err)
		}

		if successCount >= int64(cb.config.SuccessThreshold) {
			if cb.transitionToState(Closed) {
				fiberlog.Debugf("CircuitBreaker: Successfully transitioned to Closed state")
			}
		}
	}

	return nil
}

// recordFailureWithoutLock records failure without lock protection (fallback)
func (cb *CircuitBreaker) recordFailureWithoutLock(ctx context.Context) {
	kb := &keyBuilder{prefix: cb.keyPrefix}

	// Use atomic operations where possible
	failureCount, err := cb.redisClient.Incr(ctx, kb.failureCount()).Result()
	if err != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to increment failure count in fallback: %v", err)
		return
	}

	// Record last failure time
	if err := cb.redisClient.Set(ctx, kb.lastFailure(), time.Now().Unix(), 0).Err(); err != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to set last failure time in fallback: %v", err)
	}

	state, err := cb.getState(ctx)
	if err != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to get state in fallback failure recording: %v", err)
		return
	}

	if (state == Closed && failureCount >= int64(cb.config.FailureThreshold)) || state == HalfOpen {
		// State transition uses its own locking mechanism
		if cb.transitionToState(Open) {
			fiberlog.Debugf("CircuitBreaker: Successfully transitioned to Open state in fallback")
		}
	}
}

// recordFailureWithLock records failure with lock protection
func (cb *CircuitBreaker) recordFailureWithLock(ctx context.Context, kb *keyBuilder) error {
	failureCount, err := cb.redisClient.Incr(ctx, kb.failureCount()).Result()
	if err != nil {
		return fmt.Errorf("failed to increment failure count: %w", err)
	}

	// Record last failure time
	if err := cb.redisClient.Set(ctx, kb.lastFailure(), time.Now().Unix(), 0).Err(); err != nil {
		return fmt.Errorf("failed to set last failure time: %w", err)
	}

	state, err := cb.getState(ctx)
	if err != nil {
		return fmt.Errorf("failed to get current state: %w", err)
	}

	if (state == Closed && failureCount >= int64(cb.config.FailureThreshold)) || state == HalfOpen {
		if cb.transitionToState(Open) {
			fiberlog.Debugf("CircuitBreaker: Successfully transitioned to Open state")
		}
	}

	return nil
}
