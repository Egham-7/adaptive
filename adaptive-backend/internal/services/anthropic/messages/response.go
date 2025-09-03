package messages

import (
	"fmt"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/format_adapter"
	"adaptive-backend/internal/services/stream_readers/stream"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// ResponseService handles Anthropic Messages response processing and formatting
type ResponseService struct{}

// NewResponseService creates a new ResponseService
func NewResponseService() *ResponseService {
	return &ResponseService{}
}

// HandleNonStreamingResponse processes a non-streaming Anthropic response
func (rs *ResponseService) HandleNonStreamingResponse(
	c *fiber.Ctx,
	message *anthropic.Message,
	requestID string,
) error {
	fiberlog.Debugf("[%s] Converting Anthropic response to Adaptive format", requestID)
	// Convert response using format adapter
	adaptiveResponse, err := format_adapter.AnthropicToAdaptive.ConvertResponse(message, "anthropic")
	if err != nil {
		fiberlog.Errorf("[%s] Failed to convert Anthropic response: %v", requestID, err)
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": fiber.Map{
				"type":    "internal_server_error",
				"message": fmt.Sprintf("Response conversion error: %v", err),
			},
		})
	}

	fiberlog.Infof("[%s] Response converted successfully, sending to client", requestID)
	return c.JSON(adaptiveResponse)
}

// HandleStreamingResponse processes a streaming Anthropic response using the optimized stream handler
func (rs *ResponseService) HandleStreamingResponse(
	c *fiber.Ctx,
	anthropicStream *ssestream.Stream[anthropic.MessageStreamEventUnion],
	requestID string,
	provider string,
) error {
	fiberlog.Infof("[%s] Starting Anthropic streaming response handling", requestID)

	// Use the optimized stream handler that properly handles native Anthropic streams
	return stream.HandleAnthropicNativeStream(c, anthropicStream, requestID, provider)
}

// HandleError handles error responses for Anthropic Messages API
func (rs *ResponseService) HandleError(c *fiber.Ctx, err error, requestID string) error {
	fiberlog.Errorf("[%s] anthropic messages error: %v", requestID, err)

	// Use the same error sanitization as the main app
	sanitized := models.SanitizeError(err)
	statusCode := sanitized.GetStatusCode()

	// Create response with sanitized error
	response := fiber.Map{
		"error": fiber.Map{
			"type":    sanitized.Type,
			"message": sanitized.Message,
			"code":    statusCode,
		},
	}

	// Add retry info for retryable errors
	if sanitized.Retryable {
		response["error"].(fiber.Map)["retryable"] = true
		if sanitized.Type == models.ErrorTypeRateLimit {
			response["error"].(fiber.Map)["retry_after"] = "60s"
		}
	}

	// Add error code if available
	if sanitized.Code != "" {
		response["error"].(fiber.Map)["error_code"] = sanitized.Code
	}

	return c.Status(statusCode).JSON(response)
}

// HandleBadRequest handles validation and request parsing errors
func (rs *ResponseService) HandleBadRequest(c *fiber.Ctx, message, requestID string) error {
	fiberlog.Warnf("[%s] bad request: %s", requestID, message)
	return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
		"error": fiber.Map{
			"type":    "invalid_request_error",
			"message": message,
		},
	})
}

// HandleProviderNotConfigured handles cases where the provider is not available
func (rs *ResponseService) HandleProviderNotConfigured(c *fiber.Ctx, provider, requestID string) error {
	message := fmt.Sprintf("Provider '%s' is not configured for messages endpoint", provider)
	fiberlog.Warnf("[%s] %s", requestID, message)
	return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
		"error": fiber.Map{
			"type":    "invalid_request_error",
			"message": message,
		},
	})
}
