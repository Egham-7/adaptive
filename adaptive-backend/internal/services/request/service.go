package request

import (
	"crypto/rand"
	"encoding/hex"
	"strings"

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

// GetAPIKey extracts the API key from the request
func (s *BaseService) GetAPIKey(c *fiber.Ctx) string {
	// Check X-Stainless-API-Key header (commonly used by OpenAI clients)
	if apiKey := c.Get("X-Stainless-API-Key"); apiKey != "" {
		return apiKey
	}

	// Check Authorization header with Bearer prefix
	if authHeader := c.Get("Authorization"); authHeader != "" {
		if after, ok := strings.CutPrefix(authHeader, "Bearer "); ok {
			return after
		}
	}

	// Check direct API-Key header
	if apiKey := c.Get("API-Key"); apiKey != "" {
		return apiKey
	}

	return ""
}

// GetUserID extracts user ID from the request (could be from API key, JWT, etc.)
func (s *BaseService) GetUserID(c *fiber.Ctx) string {
	// For now, use API key as user ID (this could be enhanced later)
	return s.GetAPIKey(c)
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
