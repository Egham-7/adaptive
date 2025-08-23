package models

import "strings"

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

// ParseFallbackMode converts a string to FallbackMode enum
func ParseFallbackMode(mode string) FallbackMode {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "sequential":
		return FallbackModeSequential
	case "race":
		return FallbackModeRace
	default:
		return FallbackModeRace // Default to race
	}
}

// CircuitBreakerConfig holds circuit breaker configuration
type CircuitBreakerConfig struct {
	FailureThreshold int `json:"failure_threshold,omitempty" yaml:"failure_threshold,omitempty"` // Number of failures before opening circuit
	SuccessThreshold int `json:"success_threshold,omitempty" yaml:"success_threshold,omitempty"` // Number of successes to close circuit
	TimeoutMs        int `json:"timeout_ms,omitempty" yaml:"timeout_ms,omitempty"`               // Timeout for circuit breaker in milliseconds
	ResetAfterMs     int `json:"reset_after_ms,omitempty" yaml:"reset_after_ms,omitempty"`       // Time to wait before trying to close circuit
}

// FallbackConfig holds the fallback configuration with enabled toggle
type FallbackConfig struct {
	Enabled        bool                  `json:"enabled,omitempty" yaml:"enabled,omitempty"`                 // Whether fallback is enabled (default: true)
	Mode           FallbackMode          `json:"mode,omitempty" yaml:"mode,omitempty"`                       // Fallback mode (sequential/race)
	TimeoutMs      int                   `json:"timeout_ms,omitempty" yaml:"timeout_ms,omitempty"`           // Timeout in milliseconds
	MaxRetries     int                   `json:"max_retries,omitempty" yaml:"max_retries,omitempty"`         // Maximum number of retries
	CircuitBreaker *CircuitBreakerConfig `json:"circuit_breaker,omitempty" yaml:"circuit_breaker,omitempty"` // Circuit breaker configuration
}
