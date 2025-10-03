package count_tokens

import (
	"fmt"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"google.golang.org/genai"
)

// ResponseService handles Gemini count tokens response formatting
type ResponseService struct{}

// NewResponseService creates a new ResponseService
func NewResponseService() *ResponseService {
	return &ResponseService{}
}

// SendNonStreamingResponse sends a non-streaming count tokens response
func (rs *ResponseService) SendNonStreamingResponse(
	c *fiber.Ctx,
	response *genai.CountTokensResponse,
	requestID string,
) error {
	fiberlog.Debugf("[%s] Sending count tokens response - tokens: %d", requestID, response.TotalTokens)

	// Create response in Gemini format
	responseBody := map[string]interface{}{
		"totalTokens": response.TotalTokens,
	}

	c.Set("Content-Type", "application/json")
	if err := c.JSON(responseBody); err != nil {
		return fmt.Errorf("failed to send count tokens response: %w", err)
	}

	fiberlog.Infof("[%s] Count tokens response sent successfully", requestID)
	return nil
}
