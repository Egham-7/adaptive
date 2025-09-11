package request

import (
	"crypto/rand"
	"encoding/hex"
	"strings"

	"github.com/gofiber/fiber/v2"
)

const (
	// requestIDLocalKey is the shared key for storing request ID in fiber locals
	requestIDLocalKey = "request_id"
	// maxRequestIDLength is the maximum allowed length for request IDs
	maxRequestIDLength = 256
)

// BaseService provides common request handling utilities that can be embedded and specialized
type BaseService struct{}

// NewBaseService creates a new base request service
func NewBaseService() *BaseService {
	return &BaseService{}
}

// sanitizeRequestID sanitizes and caps the length of a request ID
func (s *BaseService) sanitizeRequestID(reqID string) string {
	// Trim whitespace and limit length
	sanitized := strings.TrimSpace(reqID)
	if len(sanitized) > maxRequestIDLength {
		sanitized = sanitized[:maxRequestIDLength]
	}
	return sanitized
}

// GetRequestID extracts or generates a request ID from the context
func (s *BaseService) GetRequestID(c *fiber.Ctx) string {
	// Check if we already have a cached value in locals
	if cachedID := c.Locals(requestIDLocalKey); cachedID != nil {
		if str, ok := cachedID.(string); ok && str != "" {
			return str
		}
	}

	var requestID string

	// Try to get from header first
	if headerID := c.Get("X-Request-ID"); headerID != "" {
		requestID = s.sanitizeRequestID(headerID)
	}

	// If no valid header ID, try to get from other locals (might be set by middleware)
	if requestID == "" {
		if reqID := c.Locals("request_id"); reqID != nil {
			if str, ok := reqID.(string); ok && str != "" {
				requestID = s.sanitizeRequestID(str)
			}
		}
	}

	// Generate a new one if not found or empty after sanitization
	if requestID == "" {
		requestID = s.GenerateRequestID()
	}

	// Store the final request ID in locals for caching
	c.Locals(requestIDLocalKey, requestID)

	return requestID
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
