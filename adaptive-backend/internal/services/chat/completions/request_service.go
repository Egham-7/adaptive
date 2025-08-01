package completions

import (
	"encoding/json"
	"time"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/utils"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

const (
	headerRequestID = "X-Request-ID"
	headerAPIKey    = "X-Stainless-API-Key" // #nosec G101
)

// RequestService handles HTTP request parsing and validation for chat completions.
type RequestService struct{}

// NewRequestService creates a new request service.
func NewRequestService() *RequestService {
	return &RequestService{}
}

// ParseChatCompletionRequest parses and validates a chat completion request.
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

// GetRequestID extracts or generates a request ID from the context.
func (s *RequestService) GetRequestID(c *fiber.Ctx) string {
	return c.Get(headerRequestID, time.Now().String())
}

// GetAPIKey extracts the API key from the request headers.
func (s *RequestService) GetAPIKey(c *fiber.Ctx) string {
	apiKey := string(c.Request().Header.Peek(headerAPIKey))
	if apiKey == "" {
		return "anonymous"
	}
	return apiKey
}

// ExtractPrompt extracts the prompt from the last user message.
func (s *RequestService) ExtractPrompt(req *models.ChatCompletionRequest) string {
	prompt, err := utils.FindLastUserMessage(req.Messages)
	if err != nil {
		return ""
	}
	return prompt
}
