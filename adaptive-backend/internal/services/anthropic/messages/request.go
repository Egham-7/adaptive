package messages

import (
	"errors"
	"fmt"
	"strings"

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

	if err := rs.validateRequest(&req); err != nil {
		return nil, fmt.Errorf("invalid request: %w", err)
	}

	return &req, nil
}

// validateRequest performs basic validation on the Anthropic request
func (rs *RequestService) validateRequest(req *models.AnthropicMessageRequest) error {
	if len(req.Messages) == 0 {
		return errors.New("messages array cannot be empty")
	}

	for i, msg := range req.Messages {
		if msg.Role == "" {
			return fmt.Errorf("message %d: role is required", i)
		}
		if msg.Role != "user" && msg.Role != "assistant" && msg.Role != "system" {
			return fmt.Errorf("message %d: invalid role '%s', must be 'user', 'assistant', or 'system'", i, msg.Role)
		}
		if len(msg.Content) > 0 {
			// Content validation - at least one content block should exist
			continue
		}
		return fmt.Errorf("message %d: content cannot be empty", i)
	}

	if req.MaxTokens <= 0 {
		return errors.New("max_tokens must be greater than 0")
	}

	return nil
}

// IsStreamingRequest determines if the request should be handled as streaming
func (rs *RequestService) IsStreamingRequest(c *fiber.Ctx) bool {
	// Check headers for event-stream
	accept := c.Get("Accept")
	contentType := c.Get("Content-Type")

	return strings.Contains(accept, "text/event-stream") ||
		strings.Contains(contentType, "text/event-stream")
}

// GetRequestID extracts or generates a request ID for tracking
func (rs *RequestService) GetRequestID(c *fiber.Ctx) string {
	requestID := c.Get("X-Request-ID")
	if requestID == "" {
		requestID = fmt.Sprintf("anthro_%d", c.Context().Time().UnixNano())
	}
	return requestID
}
