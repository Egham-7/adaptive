package generate

import (
	"fmt"

	"adaptive-backend/internal/models"

	"github.com/gofiber/fiber/v2"
)

// RequestService handles Gemini request parsing and validation
type RequestService struct{}

// NewRequestService creates a new RequestService
func NewRequestService() *RequestService {
	return &RequestService{}
}

// GetRequestID extracts or generates a request ID for tracking
func (rs *RequestService) GetRequestID(c *fiber.Ctx) string {
	requestID := c.Get("X-Request-ID")
	if requestID == "" {
		requestID = fmt.Sprintf("gemini_%d", c.Context().Time().UnixNano())
	}
	return requestID
}

// ParseRequest parses the HTTP request into a GeminiGenerateRequest
func (rs *RequestService) ParseRequest(c *fiber.Ctx) (*models.GeminiGenerateRequest, error) {
	var req models.GeminiGenerateRequest

	if err := c.BodyParser(&req); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}

	return &req, nil
}
