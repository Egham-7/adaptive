package completions

import (
	"fmt"

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

// ParseChatCompletionRequest parses the chat completion request body
func (rs *RequestService) ParseChatCompletionRequest(c *fiber.Ctx) (*models.ChatCompletionRequest, error) {
	requestID := rs.GetRequestID(c)

	var req models.ChatCompletionRequest
	if err := c.BodyParser(&req); err != nil {
		fiberlog.Errorf("[%s] Failed to parse request body: %v", requestID, err)
		return nil, fiber.NewError(fiber.StatusBadRequest, fmt.Sprintf("Invalid request body: %v", err))
	}

	return &req, nil
}

// ExtractPrompt extracts the prompt from the last user message
func (rs *RequestService) ExtractPrompt(req *models.ChatCompletionRequest) string {
	prompt, err := utils.FindLastUserMessage(req.Messages)
	if err != nil {
		return ""
	}
	return prompt
}
