package completions

import (
	"fmt"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/request"
	"adaptive-backend/internal/utils"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// redactSensitiveInfo creates a concise summary of the request without sensitive data
func redactSensitiveInfo(req *models.ChatCompletionRequest) string {
	messageCount := len(req.Messages)
	modelName := string(req.Model)
	if modelName == "" {
		modelName = "unspecified"
	}

	var streamStr string
	if req.Stream {
		streamStr = ", streaming=true"
	}

	return fmt.Sprintf("model=%s, messages=%d%s", modelName, messageCount, streamStr)
}

// RequestService handles request parsing and validation for chat completions
// It embeds the base request service and specializes it for completions
type RequestService struct {
	*request.BaseService
}

// NewRequestService creates a new request service for completions
func NewRequestService() *RequestService {
	return &RequestService{
		BaseService: request.NewBaseService(),
	}
}

// ParseChatCompletionRequest parses and validates the chat completion request body
func (rs *RequestService) ParseChatCompletionRequest(c *fiber.Ctx) (*models.ChatCompletionRequest, error) {
	requestID := rs.GetRequestID(c)

	var req models.ChatCompletionRequest
	if err := c.BodyParser(&req); err != nil {
		fiberlog.Errorf("[%s] Failed to parse request body: %v", requestID, err)
		return nil, fiber.NewError(fiber.StatusBadRequest, fmt.Sprintf("Invalid request body: %v", err))
	}

	fiberlog.Debugf("[%s] Parsed request: %s", requestID, redactSensitiveInfo(&req))
	return &req, nil
}

// ValidateChatCompletionRequest validates the parsed chat completion request
func (rs *RequestService) ValidateChatCompletionRequest(req *models.ChatCompletionRequest) error {
	if len(req.Messages) == 0 {
		return &ValidationError{Field: "messages", Message: "Messages cannot be empty"}
	}

	// Add more validation as needed
	return nil
}

// ExtractPrompt extracts the prompt from the last user message
func (rs *RequestService) ExtractPrompt(req *models.ChatCompletionRequest) string {
	prompt, err := utils.FindLastUserMessage(req.Messages)
	if err != nil {
		return ""
	}
	return prompt
}

// ValidationError represents a request validation error
type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	return e.Message
}
