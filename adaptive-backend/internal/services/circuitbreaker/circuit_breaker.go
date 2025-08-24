package circuitbreaker

import (
	"context"
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
	lockTTL                 = 5 * time.Second
)

type CircuitBreaker struct {
	redisClient *redis.Client
	serviceName string
	config      Config
	keyPrefix   string
}

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
	return NewWithConfig(redisClient, config)
}

func NewWithConfig(redisClient *redis.Client, config Config) *CircuitBreaker {
	serviceName := "ai_service"
	keyPrefix := circuitBreakerKeyPrefix + serviceName + ":"

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
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	lockKey := cb.keyPrefix + "lock"
	locked, err := cb.redisClient.SetNX(ctx, lockKey, "1", lockTTL).Result()
	if err != nil || !locked {
		fiberlog.Debugf("CircuitBreaker: Could not acquire lock for success recording")
		return
	}
	defer cb.redisClient.Del(ctx, lockKey)

	cb.redisClient.Set(ctx, cb.keyPrefix+failureCountKey, 0, 0)

	state, err := cb.getState(ctx)
	if err != nil {
		return
	}

	if state == HalfOpen {
		successCount, err := cb.redisClient.Incr(ctx, cb.keyPrefix+successCountKey).Result()
		if err != nil {
			fiberlog.Errorf("CircuitBreaker: Failed to increment success count: %v", err)
			return
		}

		if successCount >= int64(cb.config.SuccessThreshold) {
			cb.transitionToState(Closed)
		}
	}
}

func (cb *CircuitBreaker) RecordFailure() {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	lockKey := cb.keyPrefix + "lock"
	locked, err := cb.redisClient.SetNX(ctx, lockKey, "1", lockTTL).Result()
	if err != nil || !locked {
		fiberlog.Debugf("CircuitBreaker: Could not acquire lock for failure recording")
		return
	}
	defer cb.redisClient.Del(ctx, lockKey)

	failureCount, err := cb.redisClient.Incr(ctx, cb.keyPrefix+failureCountKey).Result()
	if err != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to increment failure count: %v", err)
		return
	}

	cb.redisClient.Set(ctx, cb.keyPrefix+lastFailureTimeKey, time.Now().Unix(), 0)

	state, err := cb.getState(ctx)
	if err != nil {
		return
	}

	if (state == Closed && failureCount >= int64(cb.config.FailureThreshold)) || state == HalfOpen {
		cb.transitionToState(Open)
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

func (cb *CircuitBreaker) RecordRequestDuration(duration time.Duration, success bool) {
	// Duration recording functionality can be added here if needed
}

func (cb *CircuitBreaker) getState(ctx context.Context) (State, error) {
	stateStr, err := cb.redisClient.Get(ctx, cb.keyPrefix+stateKey).Result()
	if err != nil {
		return Closed, err
	}

	stateInt, err := strconv.Atoi(stateStr)
	if err != nil {
		return Closed, err
	}

	return State(stateInt), nil
}

func (cb *CircuitBreaker) transitionToState(newState State) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	currentState, err := cb.getState(ctx)
	if err != nil {
		return false
	}

	if currentState == newState {
		return true
	}

	err = cb.redisClient.Watch(ctx, func(tx *redis.Tx) error {
		pipe := tx.Pipeline()
		pipe.Set(ctx, cb.keyPrefix+stateKey, int(newState), 0)
		pipe.Set(ctx, cb.keyPrefix+lastStateChangeKey, time.Now().Unix(), 0)

		if newState != HalfOpen {
			pipe.Set(ctx, cb.keyPrefix+successCountKey, 0, 0)
		}

		_, err := pipe.Exec(ctx)
		return err
	}, cb.keyPrefix+stateKey)

	if err != nil {
		fiberlog.Errorf("CircuitBreaker: Failed to transition to state %v: %v", newState, err)
		return false
	}

	fiberlog.Debugf("CircuitBreaker: Transitioned from %v to %v for service %s", currentState, newState, cb.serviceName)
	return true
}
