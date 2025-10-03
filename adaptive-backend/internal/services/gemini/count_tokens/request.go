package count_tokens

import (
	"fmt"

	"github.com/gofiber/fiber/v2"
	"google.golang.org/genai"
)

// CountTokensRequest represents the request body for count tokens
type CountTokensRequest struct {
	Contents []*genai.Content `json:"contents,omitzero"`
}

// RequestService handles Gemini count tokens request parsing and validation
type RequestService struct{}

// NewRequestService creates a new RequestService
func NewRequestService() *RequestService {
	return &RequestService{}
}

// GetRequestID extracts or generates a request ID for tracking
func (rs *RequestService) GetRequestID(c *fiber.Ctx) string {
	requestID := c.Get("X-Request-ID")
	if requestID == "" {
		requestID = fmt.Sprintf("gemini_count_%d", c.Context().Time().UnixNano())
	}
	return requestID
}

// ParseRequest parses the HTTP request into a CountTokensRequest
func (rs *RequestService) ParseRequest(c *fiber.Ctx) (*CountTokensRequest, error) {
	var req CountTokensRequest

	if err := c.BodyParser(&req); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}

	return &req, nil
}
