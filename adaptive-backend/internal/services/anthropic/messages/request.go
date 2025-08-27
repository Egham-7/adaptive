package messages

import (
	"encoding/json"
	"fmt"
	"strings"

	"adaptive-backend/internal/models"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// RequestService handles Anthropic Messages request parsing and validation
type RequestService struct{}

// NewRequestService creates a new RequestService
func NewRequestService() *RequestService {
	return &RequestService{}
}

// ParseRequest parses and validates an Anthropic Messages API request
func (rs *RequestService) ParseRequest(c *fiber.Ctx) (*models.AnthropicMessageRequest, error) {
	requestID := rs.GetRequestID(c)

	// Parse with string content conversion
	req, err := rs.parseWithStringContentSupport(c.Body())
	if err != nil {
		fiberlog.Errorf("[%s] Failed to parse Anthropic Messages request body: %v", requestID, err)
		return nil, fmt.Errorf("invalid JSON in request body: %w", err)
	}

	// Validate request structure (fail fast)
	if err := rs.validateRequest(req, requestID); err != nil {
		fiberlog.Errorf("[%s] Request validation failed: %v", requestID, err)
		return nil, fmt.Errorf("invalid request: %w", err)
	}

	// Log sanitized request summary to avoid PII leakage
	summary := rs.redactSensitiveInfo(req)
	fiberlog.Debugf("[%s] Anthropic Messages request summary: %s", requestID, summary)

	return req, nil
}

// parseWithStringContentSupport handles parsing requests that may have string content
func (rs *RequestService) parseWithStringContentSupport(data []byte) (*models.AnthropicMessageRequest, error) {
	// Parse into flexible structure first
	var flexReq map[string]any
	if err := json.Unmarshal(data, &flexReq); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	// Convert string content to text blocks if needed
	if messages, ok := flexReq["messages"].([]any); ok {
		for _, msg := range messages {
			if msgMap, ok := msg.(map[string]any); ok {
				if content, exists := msgMap["content"]; exists {
					if contentStr, isString := content.(string); isString {
						// Convert string to text block format
						msgMap["content"] = []map[string]any{
							{
								"type": "text",
								"text": contentStr,
							},
						}
					}
				}
			}
		}
	}

	// Convert back to JSON and parse with the Anthropic SDK types
	modifiedJSON, err := json.Marshal(flexReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal modified request: %w", err)
	}

	var req models.AnthropicMessageRequest
	if err := json.Unmarshal(modifiedJSON, &req); err != nil {
		return nil, fmt.Errorf("failed to unmarshal into AnthropicMessageRequest: %w", err)
	}

	return &req, nil
}

// GetRequestID extracts or generates a request ID for tracking
func (rs *RequestService) GetRequestID(c *fiber.Ctx) string {
	requestID := c.Get("X-Request-ID")
	if requestID == "" {
		requestID = fmt.Sprintf("anthro_%d", c.Context().Time().UnixNano())
	}
	return requestID
}

// validateRequest performs basic input validation on the parsed request
func (rs *RequestService) validateRequest(req *models.AnthropicMessageRequest, requestID string) error {
	if req == nil {
		return fmt.Errorf("request is nil")
	}

	// Validate messages (must be non-nil and non-empty)
	if req.Messages == nil {
		return fmt.Errorf("messages cannot be nil")
	}
	if len(req.Messages) == 0 {
		return fmt.Errorf("messages cannot be empty")
	}

	// Validate each message
	for i, msg := range req.Messages {
		// Validate role (must be valid)
		if msg.Role != "user" && msg.Role != "assistant" {
			return fmt.Errorf("message %d has invalid role '%s', must be 'user' or 'assistant'", i, msg.Role)
		}

		// Validate content (must be non-empty)
		if len(msg.Content) == 0 {
			return fmt.Errorf("message %d has empty content", i)
		}

		// Check that at least one content block has non-empty text
		hasNonEmptyContent := false
		for _, block := range msg.Content {
			if textBlock := block.OfText; textBlock != nil {
				if strings.TrimSpace(textBlock.Text) != "" {
					hasNonEmptyContent = true
					break
				}
			}
		}
		if !hasNonEmptyContent {
			return fmt.Errorf("message %d has no non-empty text content", i)
		}
	}

	return nil
}

// redactSensitiveInfo creates a sanitized summary of the request for logging
func (rs *RequestService) redactSensitiveInfo(req *models.AnthropicMessageRequest) string {
	if req == nil {
		return "nil request"
	}

	summary := fmt.Sprintf("model=%s, max_tokens=%d, temp=%.2f, messages=%d", 
		req.Model, req.MaxTokens, req.Temperature.Value, len(req.Messages))

	// Add message content length summary without actual content
	var messageSummary []string
	for i, msg := range req.Messages {
		contentLength := 0
		if len(msg.Content) > 0 {
			// Count total characters across all content blocks
			for _, block := range msg.Content {
				if textBlock := block.OfText; textBlock != nil {
					contentLength += len(textBlock.Text)
				}
			}
		}
		messageSummary = append(messageSummary, fmt.Sprintf("%d:%s(%dchars)", i, msg.Role, contentLength))
	}

	if len(messageSummary) > 0 {
		summary += fmt.Sprintf(", msg_details=[%s]", strings.Join(messageSummary, ","))
	}

	// Add other relevant fields without sensitive content
	if len(req.System) > 0 {
		totalSystemLen := 0
		for _, block := range req.System {
			totalSystemLen += len(block.Text)
		}
		summary += fmt.Sprintf(", system_len=%d", totalSystemLen)
	}
	if len(req.Tools) > 0 {
		summary += fmt.Sprintf(", tools=%d", len(req.Tools))
	}

	return summary
}
