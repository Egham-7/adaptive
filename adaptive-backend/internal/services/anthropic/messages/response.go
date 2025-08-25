package messages

import (
	"bufio"
	"encoding/json"
	"fmt"

	"adaptive-backend/internal/services/format_adapter"

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

// HandleStreamingResponse processes a streaming Anthropic response
func (rs *ResponseService) HandleStreamingResponse(
	c *fiber.Ctx,
	stream *ssestream.Stream[anthropic.MessageStreamEventUnion],
	requestID string,
) error {
	ctx := c.Context()

	// Set SSE headers
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Access-Control-Allow-Origin", "*")

	// Use the stream handler for Anthropic streaming
	c.Context().SetBodyStreamWriter(func(w *bufio.Writer) {
		defer func() {
			if err := stream.Close(); err != nil {
				fiberlog.Errorf("[%s] failed to close anthropic stream: %v", requestID, err)
			}
		}()

		for stream.Next() {
			// Check for context cancellation
			select {
			case <-ctx.Done():
				fiberlog.Infof("[%s] stream cancelled by client", requestID)
				return
			default:
			}

			event := stream.Current()

			// Convert using format adapter
			adaptiveChunk, err := format_adapter.AnthropicToAdaptive.ConvertStreamingChunk(&event, "anthropic")
			if err != nil {
				fiberlog.Errorf("[%s] failed to convert streaming chunk: %v", requestID, err)
				continue
			}

			// Write as SSE
			chunkJSON, err := json.Marshal(adaptiveChunk)
			if err != nil {
				fiberlog.Errorf("[%s] failed to marshal chunk: %v", requestID, err)
				continue
			}

			if _, err := fmt.Fprintf(w, "data: %s\n\n", chunkJSON); err != nil {
				fiberlog.Errorf("[%s] failed to write chunk: %v", requestID, err)
				break
			}

			if err := w.Flush(); err != nil {
				fiberlog.Errorf("[%s] failed to flush chunk: %v", requestID, err)
				break
			}
		}

		// Check for stream errors
		if err := stream.Err(); err != nil {
			fiberlog.Errorf("[%s] anthropic stream error: %v", requestID, err)
		}

		// Write completion message
		if _, err := fmt.Fprintf(w, "data: [DONE]\n\n"); err != nil {
			fiberlog.Errorf("[%s] failed to write done message: %v", requestID, err)
		}
		if err := w.Flush(); err != nil {
			fiberlog.Errorf("[%s] failed to flush buffer: %v", requestID, err)
		}
	})

	return nil
}

// HandleError handles error responses for Anthropic Messages API
func (rs *ResponseService) HandleError(c *fiber.Ctx, err error, requestID string) error {
	fiberlog.Errorf("[%s] anthropic messages error: %v", requestID, err)

	// Handle different error types
	statusCode := fiber.StatusInternalServerError
	errorType := "api_error"
	errorMessage := err.Error()

	// You could add more specific error handling here based on Anthropic SDK error types
	// For example, checking for rate limits, authentication errors, etc.

	return c.Status(statusCode).JSON(fiber.Map{
		"error": fiber.Map{
			"type":    errorType,
			"message": errorMessage,
		},
	})
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
