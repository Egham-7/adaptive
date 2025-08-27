package messages

import (
	"encoding/json"
	"fmt"

	"adaptive-backend/internal/models"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// redactSensitiveInfo creates a concise summary of the Anthropic message request without sensitive data
func redactSensitiveInfo(req *models.AnthropicMessageRequest) string {
	messageCount := len(req.Messages)
	modelName := string(req.Model)
	if modelName == "" {
		modelName = "unspecified"
	}

	var streamStr string
	if req.Stream != nil && *req.Stream {
		streamStr = ", streaming=true"
	}

	var systemStr string
	if len(req.System) > 0 {
		systemStr = fmt.Sprintf(", system_blocks=%d", len(req.System))
	}

	var toolsStr string
	if len(req.Tools) > 0 {
		toolsStr = fmt.Sprintf(", tools=%d", len(req.Tools))
	}

	var maxTokensStr string
	if req.MaxTokens > 0 {
		maxTokensStr = fmt.Sprintf(", max_tokens=%d", req.MaxTokens)
	}

	return fmt.Sprintf("model=%s, messages=%d%s%s%s%s", modelName, messageCount, streamStr, systemStr, toolsStr, maxTokensStr)
}

// RequestService handles Anthropic Messages request parsing and validation
type RequestService struct{}

// NewRequestService creates a new RequestService
func NewRequestService() *RequestService {
	return &RequestService{}
}

// ParseRequest parses and validates an Anthropic Messages API request
func (rs *RequestService) ParseRequest(c *fiber.Ctx) (*models.AnthropicMessageRequest, error) {
	requestID := rs.GetRequestID(c)

	// Log the raw request body
	rawBody := string(c.Body())
	fiberlog.Debugf("[%s] Raw Anthropic Messages request body: %s", requestID, rawBody)

	// Parse with string content conversion
	req, err := rs.parseWithStringContentSupport(c.Body())
	if err != nil {
		fiberlog.Errorf("[%s] Failed to parse Anthropic Messages request body: %v", requestID, err)
		return nil, fmt.Errorf("invalid JSON in request body: %w", err)
	}

	// Log the parsed request
	fiberlog.Debugf("[%s] Parsed Anthropic Messages request: %+v", requestID, req)

	return req, nil
}

// parseWithStringContentSupport handles parsing requests that may have string content
func (rs *RequestService) parseWithStringContentSupport(data []byte) (*models.AnthropicMessageRequest, error) {
	// Parse into flexible structure first
	var flexReq map[string]interface{}
	if err := json.Unmarshal(data, &flexReq); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	// Convert string content to text blocks if needed
	if messages, ok := flexReq["messages"].([]interface{}); ok {
		for _, msg := range messages {
			if msgMap, ok := msg.(map[string]interface{}); ok {
				if content, exists := msgMap["content"]; exists {
					if contentStr, isString := content.(string); isString {
						// Convert string to text block format
						msgMap["content"] = []map[string]interface{}{
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
