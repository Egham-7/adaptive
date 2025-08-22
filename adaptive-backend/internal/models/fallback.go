package models

import (
	"time"

	"github.com/gofiber/fiber/v2"
)

// ExecutionFunc is the function signature for executing a completion with a specific provider
type ExecutionFunc func(c *fiber.Ctx, provider Alternative, requestID string) error

// FallbackResult represents the result of a provider execution attempt
type FallbackResult struct {
	Success  bool
	Provider Alternative
	Error    error
	Duration time.Duration
}
