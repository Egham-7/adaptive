package format_adapter

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicparam "github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/openai/openai-go"
)

// OpenAIToAnthropicConverter handles conversion from OpenAI format to Anthropic format
type OpenAIToAnthropicConverter struct{}

// NewOpenAIToAnthropicConverter creates a new OpenAI to Anthropic converter
func NewOpenAIToAnthropicConverter() *OpenAIToAnthropicConverter {
	return &OpenAIToAnthropicConverter{}
}

// ConvertRequest converts OpenAI ChatCompletionNewParams to Anthropic MessageNewParams
func (c *OpenAIToAnthropicConverter) ConvertRequest(req *openai.ChatCompletionNewParams) (*anthropic.MessageNewParams, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Convert messages from OpenAI to Anthropic format
	anthropicMessages, systemMessage, err := c.convertMessagesFromOpenAI(req.Messages)
	if err != nil {
		return nil, fmt.Errorf("failed to convert messages: %w", err)
	}

	// Create base Anthropic request
	anthropicReq := &anthropic.MessageNewParams{
		Model:    anthropic.Model(req.Model),
		Messages: anthropicMessages,
	}

	// Set system message if present (Anthropic uses separate system field)
	if systemMessage != "" {
		anthropicReq.System = []anthropic.TextBlockParam{
			{
				Text: systemMessage,
			},
		}
	}

	// Convert parameters
	if err := c.convertParameters(req, anthropicReq); err != nil {
		return nil, fmt.Errorf("failed to convert parameters: %w", err)
	}

	// Convert tools if present
	if len(req.Tools) > 0 {
		anthropicTools, err := c.convertTools(req.Tools)
		if err != nil {
			return nil, fmt.Errorf("failed to convert tools: %w", err)
		}
		// Convert to ToolUnionParam slice
		var toolUnions []anthropic.ToolUnionParam
		for i := range anthropicTools {
			toolUnions = append(toolUnions, anthropic.ToolUnionParam{
				OfTool: &anthropicTools[i],
			})
		}
		anthropicReq.Tools = toolUnions
	}

	// Convert stop sequences if present
	if req.Stop.OfString.Value != "" || len(req.Stop.OfStringArray) > 0 {
		stopSequences, err := c.convertStopSequences(req.Stop)
		if err != nil {
			return nil, fmt.Errorf("failed to convert stop sequences: %w", err)
		}
		if len(stopSequences) > 0 {
			anthropicReq.StopSequences = stopSequences
		}
	}

	return anthropicReq, nil
}

// convertMessagesFromOpenAI converts OpenAI messages to Anthropic format
func (c *OpenAIToAnthropicConverter) convertMessagesFromOpenAI(messages []openai.ChatCompletionMessageParamUnion) ([]anthropic.MessageParam, string, error) {
	var anthropicMessages []anthropic.MessageParam
	var systemMessage string

	for _, msg := range messages {
		// OpenAI SDK uses union structs with Of* fields, not interface switches
		if msg.OfUser != nil {
			// Convert user message
			content, err := c.convertUserContent(msg.OfUser.Content)
			if err != nil {
				return nil, "", fmt.Errorf("failed to convert user message: %w", err)
			}

			anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
				Role:    anthropic.MessageParamRoleUser,
				Content: content,
			})

		} else if msg.OfAssistant != nil {
			// Convert assistant message
			content, err := c.convertAssistantContent(msg.OfAssistant.Content, msg.OfAssistant.ToolCalls)
			if err != nil {
				return nil, "", fmt.Errorf("failed to convert assistant message: %w", err)
			}

			anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
				Role:    anthropic.MessageParamRoleAssistant,
				Content: content,
			})

		} else if msg.OfSystem != nil {
			// Anthropic uses separate system field - collect all system messages
			systemContent := c.extractSystemContent(msg.OfSystem.Content)
			if systemMessage != "" {
				systemMessage += "\n\n" + systemContent
			} else {
				systemMessage = systemContent
			}

		} else if msg.OfTool != nil {
			// Convert tool messages to user messages with tool result content
			anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
				Role: anthropic.MessageParamRoleUser,
				Content: []anthropic.ContentBlockParamUnion{
					{
						OfToolResult: &anthropic.ToolResultBlockParam{
							Type:      "tool_result",
							ToolUseID: msg.OfTool.ToolCallID,
							Content:   []anthropic.ToolResultBlockParamContentUnion{{OfText: &anthropic.TextBlockParam{Type: "text", Text: msg.OfTool.Content.OfString.Value}}},
						},
					},
				},
			})
		} else if msg.OfFunction != nil {
			// Convert deprecated function messages to tool result format
			anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
				Role: anthropic.MessageParamRoleUser,
				Content: []anthropic.ContentBlockParamUnion{
					{
						OfToolResult: &anthropic.ToolResultBlockParam{
							Type:      "tool_result",
							ToolUseID: msg.OfFunction.Name, // Use function name as tool ID
							Content:   []anthropic.ToolResultBlockParamContentUnion{{OfText: &anthropic.TextBlockParam{Type: "text", Text: msg.OfFunction.Content.Value}}},
						},
					},
				},
			})
		}
	}

	return anthropicMessages, systemMessage, nil
}

// convertUserContent converts OpenAI user content to Anthropic format
func (c *OpenAIToAnthropicConverter) convertUserContent(content openai.ChatCompletionUserMessageParamContentUnion) ([]anthropic.ContentBlockParamUnion, error) {
	// OpenAI SDK uses union structs with Of* fields
	if content.OfString.Valid() {
		// Simple text content
		return []anthropic.ContentBlockParamUnion{
			{
				OfText: &anthropic.TextBlockParam{
					Type: "text",
					Text: content.OfString.Value,
				},
			},
		}, nil
	} else if len(content.OfArrayOfContentParts) > 0 {
		// Multi-part content (text + images, etc.)
		var parts []anthropic.ContentBlockParamUnion

		for _, part := range content.OfArrayOfContentParts {
			if part.OfText != nil {
				parts = append(parts, anthropic.ContentBlockParamUnion{
					OfText: &anthropic.TextBlockParam{
						Type: "text",
						Text: part.OfText.Text,
					},
				})
			} else if part.OfImageURL != nil {
				imageBlock, err := c.convertImageContent(*part.OfImageURL)
				if err != nil {
					return nil, fmt.Errorf("failed to convert image: %w", err)
				}
				parts = append(parts, *imageBlock)
			}
		}

		return parts, nil

	} else {
		// Fallback for unsupported content types
		return []anthropic.ContentBlockParamUnion{
			{
				OfText: &anthropic.TextBlockParam{
					Type: "text",
					Text: "Unsupported content type converted to text",
				},
			},
		}, nil
	}
}

// convertAssistantContent converts OpenAI assistant content to Anthropic format
func (c *OpenAIToAnthropicConverter) convertAssistantContent(content openai.ChatCompletionAssistantMessageParamContentUnion, toolCalls []openai.ChatCompletionMessageToolCallParam) ([]anthropic.ContentBlockParamUnion, error) {
	var contentParts []anthropic.ContentBlockParamUnion

	// Add text content if present (based on Context7 docs, assistant content can be string or null)
	if content.OfString.Valid() {
		contentParts = append(contentParts, anthropic.ContentBlockParamUnion{
			OfText: &anthropic.TextBlockParam{
				Type: "text",
				Text: content.OfString.Value,
			},
		})
	}

	// Convert tool calls to Anthropic tool_use blocks
	for _, toolCall := range toolCalls {
		// Based on Context7 docs, tool calls are ChatCompletionMessageToolCallParam
		if toolCall.Function.Arguments != "" {
			// Parse function arguments
			var args map[string]any
			if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
				// If JSON parsing fails, use the raw string in a simple map
				args = map[string]any{"arguments": toolCall.Function.Arguments}
			}

			contentParts = append(contentParts, anthropic.ContentBlockParamUnion{
				OfToolUse: &anthropic.ToolUseBlockParam{
					Type:  "tool_use",
					ID:    toolCall.ID,
					Name:  toolCall.Function.Name,
					Input: args,
				},
			})
		}
	}

	// If no content at all, add empty text block
	if len(contentParts) == 0 {
		contentParts = append(contentParts, anthropic.ContentBlockParamUnion{
			OfText: &anthropic.TextBlockParam{
				Type: "text",
				Text: "",
			},
		})
	}

	return contentParts, nil
}

// convertImageContent converts OpenAI image content to Anthropic format
func (c *OpenAIToAnthropicConverter) convertImageContent(img openai.ChatCompletionContentPartImageParam) (*anthropic.ContentBlockParamUnion, error) {
	if img.ImageURL.URL == "" {
		return nil, fmt.Errorf("image URL is required")
	}

	url := img.ImageURL.URL

	// Handle base64 data URLs (data:image/jpeg;base64,...)
	if strings.HasPrefix(url, "data:") {
		parts := strings.Split(url, ",")
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid data URL format")
		}

		// Extract media type from data URL header
		headerParts := strings.Split(parts[0], ":")
		if len(headerParts) < 2 {
			return nil, fmt.Errorf("invalid data URL header")
		}

		mediaTypeParts := strings.Split(headerParts[1], ";")
		mediaType := mediaTypeParts[0]

		return &anthropic.ContentBlockParamUnion{
			OfImage: &anthropic.ImageBlockParam{
				Type: "image",
				Source: anthropic.ImageBlockParamSourceUnion{
					OfBase64: &anthropic.Base64ImageSourceParam{
						Type:      "base64",
						MediaType: anthropic.Base64ImageSourceMediaType(mediaType),
						Data:      parts[1], // base64 data without data URL prefix
					},
				},
			},
		}, nil
	}

	// Handle regular URL references
	return &anthropic.ContentBlockParamUnion{
		OfImage: &anthropic.ImageBlockParam{
			Type: "image",
			Source: anthropic.ImageBlockParamSourceUnion{
				OfURL: &anthropic.URLImageSourceParam{
					Type: "url",
					URL:  url,
				},
			},
		},
	}, nil
}

// convertParameters converts OpenAI parameters to Anthropic equivalents
func (c *OpenAIToAnthropicConverter) convertParameters(openaiReq *openai.ChatCompletionNewParams, anthropicReq *anthropic.MessageNewParams) error {
	// Temperature (both use 0-1 range)
	if openaiReq.Temperature.Valid() {
		anthropicReq.Temperature = anthropicparam.Opt[float64]{Value: openaiReq.Temperature.Value}
	}

	// Max tokens (Anthropic requires this field)
	if openaiReq.MaxTokens.Valid() {
		anthropicReq.MaxTokens = openaiReq.MaxTokens.Value
	} else if openaiReq.MaxCompletionTokens.Valid() {
		anthropicReq.MaxTokens = openaiReq.MaxCompletionTokens.Value
	} else {
		// Anthropic requires max_tokens, set reasonable default
		anthropicReq.MaxTokens = 4096
	}

	// Top-p (both use 0-1 range)
	if openaiReq.TopP.Valid() {
		anthropicReq.TopP = anthropicparam.Opt[float64]{Value: openaiReq.TopP.Value}
	}

	// Note: Some OpenAI parameters don't have Anthropic equivalents:
	// - frequency_penalty, presence_penalty, logit_bias, n, etc.
	// These are silently ignored during conversion

	return nil
}

// convertTools converts OpenAI tools to Anthropic format
func (c *OpenAIToAnthropicConverter) convertTools(tools []openai.ChatCompletionToolParam) ([]anthropic.ToolParam, error) {
	var anthropicTools []anthropic.ToolParam

	for _, tool := range tools {
		// Only function tools are supported in this conversion
		anthropicTool := anthropic.ToolParam{
			Name:        tool.Function.Name,
			Description: anthropicparam.Opt[string]{Value: tool.Function.Description.Value},
		}

		// Convert function parameters (OpenAI JSON Schema) to Anthropic input_schema
		if tool.Function.Parameters != nil {
			// FunctionParameters is already map[string]any, so cast directly
			m := map[string]any(tool.Function.Parameters)
			// Extract properties (fallback to the whole map if no explicit "properties")
			var props map[string]any
			if p, ok := m["properties"].(map[string]any); ok {
				props = p
			} else {
				props = m
			}
			// Extract required
			var reqFields []string
			if r, ok := m["required"].([]any); ok {
				reqFields = make([]string, 0, len(r))
				for _, v := range r {
					if s, ok := v.(string); ok {
						reqFields = append(reqFields, s)
					}
				}
			}
			anthropicTool.InputSchema = anthropic.ToolInputSchemaParam{
				Type:       "object",
				Properties: props,
				Required:   reqFields,
			}
		}

		anthropicTools = append(anthropicTools, anthropicTool)
	}

	return anthropicTools, nil
}

// convertStopSequences converts OpenAI stop sequences to Anthropic format
func (c *OpenAIToAnthropicConverter) convertStopSequences(stop openai.ChatCompletionNewParamsStopUnion) ([]string, error) {
	// OpenAI SDK uses union structs with Of* fields
	if stop.OfString.Value != "" {
		return []string{stop.OfString.Value}, nil
	} else if len(stop.OfStringArray) > 0 {
		return stop.OfStringArray, nil
	}
	return nil, fmt.Errorf("no stop sequences found in union")
}

// extractSystemContent extracts string content from OpenAI system message
func (c *OpenAIToAnthropicConverter) extractSystemContent(content openai.ChatCompletionSystemMessageParamContentUnion) string {
	// OpenAI SDK uses union structs with Of* fields
	if content.OfString.Valid() {
		return content.OfString.Value
	} else if len(content.OfArrayOfContentParts) > 0 {
		var text strings.Builder
		for _, part := range content.OfArrayOfContentParts {
			if part.Text != "" {
				if text.Len() > 0 {
					text.WriteString("\n")
				}
				text.WriteString(part.Text)
			}
		}
		return text.String()
	}
	return ""
}

// convertFinishReason converts OpenAI finish reason to Anthropic stop reason
func (c *OpenAIToAnthropicConverter) convertFinishReason(finishReason string) anthropic.StopReason {
	switch finishReason {
	case "stop":
		return anthropic.StopReasonEndTurn
	case "length":
		return anthropic.StopReasonMaxTokens
	case "tool_calls":
		return anthropic.StopReasonToolUse
	case "content_filter":
		return anthropic.StopReasonRefusal
	default:
		return anthropic.StopReasonEndTurn
	}
}

// convertUsage converts OpenAI usage to Anthropic usage
func (c *OpenAIToAnthropicConverter) convertUsage(usage openai.CompletionUsage) anthropic.Usage {
	return anthropic.Usage{
		InputTokens:              usage.PromptTokens,
		OutputTokens:             usage.CompletionTokens,
		CacheCreationInputTokens: 0,
		CacheReadInputTokens:     0,
		ServerToolUse: anthropic.ServerToolUsage{
			WebSearchRequests: 0,
		},
		ServiceTier: anthropic.UsageServiceTierStandard,
	}
}

// convertResponseContent converts OpenAI message content to Anthropic content blocks
func (c *OpenAIToAnthropicConverter) convertResponseContent(content string, toolCalls []openai.ChatCompletionMessageToolCall) ([]anthropic.ContentBlockUnion, error) {
	var contentBlocks []anthropic.ContentBlockUnion

	// Add text content if present
	if content != "" {
		contentBlocks = append(contentBlocks, anthropic.ContentBlockUnion{
			Type: "text",
			Text: content,
		})
	}

	// Add tool calls if present
	for _, toolCall := range toolCalls {
		// Parse function arguments as raw JSON
		var input json.RawMessage
		if toolCall.Function.Arguments != "" {
			input = json.RawMessage(toolCall.Function.Arguments)
		} else {
			input = json.RawMessage("{}")
		}

		contentBlocks = append(contentBlocks, anthropic.ContentBlockUnion{
			Type:  "tool_use",
			ID:    toolCall.ID,
			Name:  toolCall.Function.Name,
			Input: input,
		})
	}

	return contentBlocks, nil
}

// ConvertResponse converts OpenAI ChatCompletion to Anthropic Message format
func (c *OpenAIToAnthropicConverter) ConvertResponse(resp *openai.ChatCompletion, provider string) (*anthropic.Message, error) {
	if resp == nil {
		return nil, fmt.Errorf("chat completion cannot be nil")
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in OpenAI response")
	}

	// Use the first choice (OpenAI can return multiple, Anthropic returns one)
	choice := resp.Choices[0]

	// Use helper function to convert response content
	contentBlocks, err := c.convertResponseContent(choice.Message.Content, choice.Message.ToolCalls)
	if err != nil {
		return nil, fmt.Errorf("failed to convert response content: %w", err)
	}

	// Use helper to convert stop reason
	stopReason := c.convertFinishReason(choice.FinishReason)

	// Use helper to convert model name
	modelName := anthropicparam.Opt[string]{Value: resp.Model}

	// Use helper to convert usage
	usage := c.convertUsage(resp.Usage)

	// Create Anthropic Message
	message := &anthropic.Message{
		ID:           resp.ID,
		Content:      contentBlocks,
		Model:        anthropic.Model(modelName.Value),
		Role:         "assistant",
		StopReason:   stopReason,
		StopSequence: "",
		Type:         "message",
		Usage:        usage,
	}

	return message, nil
}

// ConvertStreamingChunk converts OpenAI streaming chunk to Anthropic streaming event format
func (c *OpenAIToAnthropicConverter) ConvertStreamingChunk(chunk *openai.ChatCompletionChunk, provider string) (any, error) {
	if chunk == nil {
		return nil, fmt.Errorf("chat completion chunk cannot be nil")
	}

	if len(chunk.Choices) == 0 {
		return nil, fmt.Errorf("no choices in OpenAI streaming chunk")
	}

	choice := chunk.Choices[0]

	// Handle different types of streaming events based on OpenAI chunk content

	// Check if this is the start of streaming (when delta starts)
	if choice.Delta.Role != "" && choice.Delta.Content == "" && len(choice.Delta.ToolCalls) == 0 {
		// This is a message start event
		return &anthropic.MessageStartEvent{
			Type: "message_start",
			Message: anthropic.Message{
				ID:      chunk.ID,
				Content: []anthropic.ContentBlockUnion{},
				Model:   anthropic.Model(chunk.Model),
				Role:    "assistant",
				Type:    "message",
			},
		}, nil
	}

	// Handle content delta (text content)
	if choice.Delta.Content != "" {
		return &anthropic.ContentBlockDeltaEvent{
			Type:  "content_block_delta",
			Index: choice.Index,
			Delta: anthropic.RawContentBlockDeltaUnion{
				Type: "text_delta",
				Text: choice.Delta.Content,
			},
		}, nil
	}

	// Handle tool call delta
	if len(choice.Delta.ToolCalls) > 0 {
		toolCall := choice.Delta.ToolCalls[0]

		// Check if this is a new tool use block starting
		if toolCall.ID != "" && toolCall.Function.Name != "" {
			return &anthropic.ContentBlockStartEvent{
				Type:  "content_block_start",
				Index: choice.Index,
				ContentBlock: anthropic.ContentBlockStartEventContentBlockUnion{
					Type:  "tool_use",
					ID:    toolCall.ID,
					Name:  toolCall.Function.Name,
					Input: json.RawMessage("{}"), // Initially empty
				},
			}, nil
		}

		// Handle function arguments delta
		if toolCall.Function.Arguments != "" {
			return &anthropic.ContentBlockDeltaEvent{
				Type:  "content_block_delta",
				Index: choice.Index,
				Delta: anthropic.RawContentBlockDeltaUnion{
					Type:        "input_json_delta",
					PartialJSON: toolCall.Function.Arguments,
				},
			}, nil
		}
	}

	// Handle finish reason (end of streaming)
	if choice.FinishReason != "" {
		return &anthropic.MessageStopEvent{
			Type: "message_stop",
		}, nil
	}

	// Default case - return a simple delta event if we have any content
	return &anthropic.ContentBlockDeltaEvent{
		Type:  "content_block_delta",
		Index: choice.Index,
		Delta: anthropic.RawContentBlockDeltaUnion{
			Type: "text_delta",
			Text: "",
		},
	}, nil
}
