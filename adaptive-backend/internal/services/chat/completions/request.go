package completions

import (
	"encoding/json"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/request"
	"adaptive-backend/internal/utils"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

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