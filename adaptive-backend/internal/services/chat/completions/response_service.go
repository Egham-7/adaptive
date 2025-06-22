package completions

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"adaptive-backend/internal/services/stream_readers/stream"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// ResponseService handles different types of HTTP responses for chat completions
type ResponseService struct{}

// NewResponseService creates a new response service
func NewResponseService() *ResponseService {
	return &ResponseService{}
}

// HandleStreamResponse manages streaming responses
func (s *ResponseService) HandleStreamResponse(
	c *fiber.Ctx,
	provider provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	requestID string,
) error {
	fiberlog.Infof("[%s] Initiating streaming response", requestID)

	resp, err := provider.Chat().
		Completions().
		StreamCompletion(req.ToOpenAIParams())
	if err != nil {
		fiberlog.Errorf("[%s] Failed to stream completion: %v", requestID, err)
		return s.HandleError(c, fiber.StatusInternalServerError,
			"Failed to stream completion: "+err.Error(), requestID)
	}

	s.setStreamHeaders(c)
	return stream.HandleStream(c, resp, requestID)
}

// HandleRegularResponse manages non-streaming responses
func (s *ResponseService) HandleRegularResponse(
	c *fiber.Ctx,
	provider provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	requestID string,
) error {
	fiberlog.Infof("[%s] Initiating regular response", requestID)

	resp, err := provider.Chat().
		Completions().
		CreateCompletion(req.ToOpenAIParams())
	if err != nil {
		fiberlog.Errorf("[%s] Failed to generate completion: %v", requestID, err)
		return s.HandleError(c, fiber.StatusInternalServerError,
			"Failed to generate completion: "+err.Error(), requestID)
	}

	fiberlog.Infof("[%s] Successfully generated completion", requestID)
	return c.JSON(resp)
}

// HandleError sends a standardized error response
func (s *ResponseService) HandleError(c *fiber.Ctx, statusCode int, message string, requestID string) error {
	fiberlog.Errorf("[%s] Error response: %d - %s", requestID, statusCode, message)
	return c.Status(statusCode).JSON(fiber.Map{
		"error": message,
	})
}

// HandleBadRequest handles 400 bad request errors
func (s *ResponseService) HandleBadRequest(c *fiber.Ctx, message string, requestID string) error {
	return s.HandleError(c, fiber.StatusBadRequest, message, requestID)
}

// HandleInternalError handles 500 internal server errors
func (s *ResponseService) HandleInternalError(c *fiber.Ctx, message string, requestID string) error {
	return s.HandleError(c, fiber.StatusInternalServerError, message, requestID)
}

// setStreamHeaders sets the necessary headers for streaming responses
func (s *ResponseService) setStreamHeaders(c *fiber.Ctx) {
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Transfer-Encoding", "chunked")
	c.Set("Access-Control-Allow-Origin", "*")
	c.Set("Access-Control-Allow-Headers", "Cache-Control")
}
