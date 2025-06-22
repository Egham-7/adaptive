package completions

import (
	"encoding/json"
	"time"

	"adaptive-backend/internal/models"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// RequestService handles HTTP request parsing and validation for chat completions
type RequestService struct{}

// NewRequestService creates a new request service
func NewRequestService() *RequestService {
	return &RequestService{}
}

// ParseChatCompletionRequest parses and validates a chat completion request
func (s *RequestService) ParseChatCompletionRequest(c *fiber.Ctx) (*models.ChatCompletionRequest, error) {
	requestID := s.GetRequestID(c)
	body := c.Body()
	fiberlog.Infof("[%s] Raw request body: %s", requestID, string(body))

	var req models.ChatCompletionRequest
	if err := c.BodyParser(&req); err != nil {
		fiberlog.Errorf("[%s] Failed to parse request body: %v", requestID, err)
		return nil, err
	}

	reqJSON, _ := json.Marshal(req)
	fiberlog.Infof("[%s] Parsed request: %s", requestID, string(reqJSON))
	return &req, nil
}

// GetRequestID extracts or generates a request ID from the context
func (s *RequestService) GetRequestID(c *fiber.Ctx) string {
	return c.Get("X-Request-ID", time.Now().String())
}

// GetAPIKey extracts the API key from the request headers
func (s *RequestService) GetAPIKey(c *fiber.Ctx) string {
	apiKey := string(c.Request().Header.Peek("X-Stainless-API-Key"))
	if apiKey == "" {
		return "anonymous"
	}
	return apiKey
}

// ExtractPrompt extracts the prompt from the last user message
func (s *RequestService) ExtractPrompt(req *models.ChatCompletionRequest) string {
	if len(req.Messages) == 0 {
		return ""
	}
	return req.Messages[len(req.Messages)-1].OfUser.Content.OfString.Value
}
