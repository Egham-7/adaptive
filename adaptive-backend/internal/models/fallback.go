package models

import (
	"time"

	"github.com/gofiber/fiber/v2"
)

// Cache tier constants
const (
	CacheTierSemanticExact   = "semantic_exact"
	CacheTierSemanticSimilar = "semantic_similar"
	CacheTierPromptResponse  = "prompt_response"
)

// FallbackMode defines the strategy for handling provider failures
type FallbackMode string

const (
	FallbackModeSequential FallbackMode = "sequential"
	FallbackModeRace       FallbackMode = "race"
)

// CircuitBreakerConfig holds circuit breaker configuration
type CircuitBreakerConfig struct {
	FailureThreshold int `json:"failure_threshold,omitempty" yaml:"failure_threshold,omitempty"` // Number of failures before opening circuit
	SuccessThreshold int `json:"success_threshold,omitempty" yaml:"success_threshold,omitempty"` // Number of successes to close circuit
	TimeoutMs        int `json:"timeout_ms,omitempty" yaml:"timeout_ms,omitempty"`               // Timeout for circuit breaker in milliseconds
	ResetAfterMs     int `json:"reset_after_ms,omitempty" yaml:"reset_after_ms,omitempty"`       // Time to wait before trying to close circuit
}

// FallbackConfig holds the fallback configuration
// Fallback is enabled when Mode is non-empty, disabled when Mode is empty
type FallbackConfig struct {
	Mode           FallbackMode          `json:"mode,omitempty" yaml:"mode,omitempty"`                       // Fallback mode (sequential/race). Empty = disabled, non-empty = enabled
	TimeoutMs      int                   `json:"timeout_ms,omitempty" yaml:"timeout_ms,omitempty"`           // Timeout in milliseconds
	MaxRetries     int                   `json:"max_retries,omitempty" yaml:"max_retries,omitempty"`         // Maximum number of retries
	CircuitBreaker *CircuitBreakerConfig `json:"circuit_breaker,omitempty" yaml:"circuit_breaker,omitempty"` // Circuit breaker configuration
}

// ExecutionFunc is the function signature for executing a completion with a specific provider
type ExecutionFunc func(c *fiber.Ctx, provider Alternative, requestID string) error

// FallbackResult represents the result of a provider execution attempt
type FallbackResult struct {
	Success  bool
	Provider Alternative
	Error    error
	Duration time.Duration
}
