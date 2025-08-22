package request

import (
	"crypto/rand"
	"encoding/hex"

	"github.com/gofiber/fiber/v2"
)

// BaseService provides common request handling utilities that can be embedded and specialized
type BaseService struct{}

// NewBaseService creates a new base request service
func NewBaseService() *BaseService {
	return &BaseService{}
}

// GetRequestID extracts or generates a request ID from the context
func (s *BaseService) GetRequestID(c *fiber.Ctx) string {
	// Try to get from header first
	if reqID := c.Get("X-Request-ID"); reqID != "" {
		return reqID
	}

	// Try to get from Fiber's locals (might be set by middleware)
	if reqID := c.Locals("request_id"); reqID != nil {
		if str, ok := reqID.(string); ok && str != "" {
			return str
		}
	}

	// Generate a new one if not found
	return s.GenerateRequestID()
}

// GenerateRequestID creates a new random request ID
func (s *BaseService) GenerateRequestID() string {
	bytes := make([]byte, 8)
	if _, err := rand.Read(bytes); err != nil {
		// Fallback to a simple counter-based approach if crypto/rand fails
		return "req_unknown"
	}
	return "req_" + hex.EncodeToString(bytes)
}

// SetRequestID sets the request ID in the context locals
func (s *BaseService) SetRequestID(c *fiber.Ctx, requestID string) {
	c.Locals("request_id", requestID)
}
