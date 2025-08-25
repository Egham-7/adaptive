package messages

import (
	"fmt"

	"adaptive-backend/internal/models"

	"github.com/gofiber/fiber/v2"
)

// RequestService handles Anthropic Messages request parsing and validation
type RequestService struct{}

// NewRequestService creates a new RequestService
func NewRequestService() *RequestService {
	return &RequestService{}
}

// ParseRequest parses and validates an Anthropic Messages API request
func (rs *RequestService) ParseRequest(c *fiber.Ctx) (*models.AnthropicMessageRequest, error) {
	var req models.AnthropicMessageRequest
	if err := c.BodyParser(&req); err != nil {
		return nil, fmt.Errorf("invalid JSON in request body: %w", err)
	}

	return &req, nil
}

// GetRequestID extracts or generates a request ID for tracking
func (rs *RequestService) GetRequestID(c *fiber.Ctx) string {
	requestID := c.Get("X-Request-ID")
	if requestID == "" {
		requestID = fmt.Sprintf("anthro_%d", c.Context().Time().UnixNano())
	}
	return requestID
}
