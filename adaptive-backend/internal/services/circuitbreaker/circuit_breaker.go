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

// Lua scripts for atomic circuit breaker operations
const (
	// recordSuccessScript atomically records success and handles state transitions
	// KEYS[1]: state key
	// KEYS[2]: failure_count key
	// KEYS[3]: success_count key
	// KEYS[4]: last_state_change key
	// ARGV[1]: success threshold (int)
	// ARGV[2]: current timestamp (unix seconds)
	recordSuccessScript = `
		local state = tonumber(redis.call('GET', KEYS[1]) or '0')
		redis.call('SET', KEYS[2], 0)  -- Reset failure count

		if state == 2 then  -- HalfOpen state
			local count = redis.call('INCR', KEYS[3])
			if count >= tonumber(ARGV[1]) then
				redis.call('SET', KEYS[1], 0)  -- Transition to Closed
				redis.call('SET', KEYS[3], 0)  -- Reset success count
				redis.call('SET', KEYS[4], ARGV[2])  -- Update last state change
				return 2  -- Transitioned to Closed
			end
			return 1  -- Success recorded in HalfOpen
		end
		return 0  -- Success recorded in other state
	`

	// recordFailureScript atomically records failure and handles state transitions
	// KEYS[1]: state key
	// KEYS[2]: failure_count key
	// KEYS[3]: last_failure_time key
	// KEYS[4]: last_state_change key
	// ARGV[1]: failure threshold (int)
	// ARGV[2]: current timestamp (unix seconds)
	recordFailureScript = `
		local state = tonumber(redis.call('GET', KEYS[1]) or '0')
		local failureCount = redis.call('INCR', KEYS[2])
		redis.call('SET', KEYS[3], ARGV[2])  -- Set last failure time

		local shouldTransitionToOpen = (state == 0 and failureCount >= tonumber(ARGV[1])) or state == 2

		if shouldTransitionToOpen then
			redis.call('SET', KEYS[1], 1)  -- Transition to Open
			redis.call('SET', KEYS[4], ARGV[2])  -- Update last state change
			return 1  -- Transitioned to Open
		end
		return 0  -- Failure recorded, no transition
	`
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

	// Use Lua script for atomic operation (eliminates retry loops and reduces round trips)
	keys := []string{
		kb.state(),
		kb.failureCount(),
		kb.successCount(),
		kb.lastChange(),
	}
	args := []any{
		cb.config.SuccessThreshold,
		time.Now().Unix(),
	}

	result, err := cb.redisClient.Eval(ctx, recordSuccessScript, keys, args...).Int()
	if err != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to record success: %v", err)
		return
	}

	// Log based on result
	switch result {
	case 2:
		fiberlog.Debugf("CircuitBreaker: %s transitioned to Closed state", cb.serviceName)
	case 1:
		fiberlog.Debugf("CircuitBreaker: %s recorded success in HalfOpen state", cb.serviceName)
	default:
		fiberlog.Debugf("CircuitBreaker: %s recorded success", cb.serviceName)
	}
}

func (cb *CircuitBreaker) RecordFailure() {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	kb := &keyBuilder{prefix: cb.keyPrefix}

	// Use Lua script for atomic operation (eliminates retry loops and reduces round trips)
	keys := []string{
		kb.state(),
		kb.failureCount(),
		kb.lastFailure(),
		kb.lastChange(),
	}
	args := []any{
		cb.config.FailureThreshold,
		time.Now().Unix(),
	}

	result, err := cb.redisClient.Eval(ctx, recordFailureScript, keys, args...).Int()
	if err != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to record failure: %v", err)
		return
	}

	// Log based on result
	if result == 1 {
		fiberlog.Debugf("CircuitBreaker: %s transitioned to Open state", cb.serviceName)
	} else {
		fiberlog.Debugf("CircuitBreaker: %s recorded failure", cb.serviceName)
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
