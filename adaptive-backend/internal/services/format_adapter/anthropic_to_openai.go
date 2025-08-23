package format_adapter

import (
	"adaptive-backend/internal/models"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
)

// AnthropicToOpenAIConverter handles conversion from Anthropic format to OpenAI format
type AnthropicToOpenAIConverter struct{}

// NewAnthropicToOpenAIConverter creates a new Anthropic to OpenAI converter
func NewAnthropicToOpenAIConverter() *AnthropicToOpenAIConverter {
	return &AnthropicToOpenAIConverter{}
}

// ConvertRequest converts Anthropic MessageNewParams to OpenAI ChatCompletionRequest
func (c *AnthropicToOpenAIConverter) ConvertRequest(req *models.AnthropicMessageRequest) (*models.ChatCompletionRequest, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Convert messages from Anthropic to OpenAI format
	openaiMessages, err := c.convertMessages(req.Messages, req.System)
	if err != nil {
		return nil, fmt.Errorf("failed to convert messages: %w", err)
	}

	// Create base OpenAI request
	openaiReq := &models.ChatCompletionRequest{
		Messages: openaiMessages,
		// Copy our custom fields
		ProtocolManagerConfig: req.ProtocolManagerConfig,
		SemanticCache:         req.SemanticCache,
		PromptCache:           req.PromptCache,
		Fallback:              req.Fallback,
		ProviderConfigs:       req.ProviderConfigs,
	}

	// Convert parameters
	if err := c.convertParameters(req, openaiReq); err != nil {
		return nil, fmt.Errorf("failed to convert parameters: %w", err)
	}

	// Convert tools if present
	if len(req.Tools) > 0 {
		openaiTools, err := c.convertTools(req.Tools)
		if err != nil {
			return nil, fmt.Errorf("failed to convert tools: %w", err)
		}
		openaiReq.Tools = openaiTools
	}

	// Convert stop sequences if present
	if len(req.StopSequences) > 0 {
		stopSequences := c.convertStopSequences(req.StopSequences)
		openaiReq.Stop = stopSequences
	}

	return openaiReq, nil
}

// convertMessages converts Anthropic messages to OpenAI format
func (c *AnthropicToOpenAIConverter) convertMessages(messages []anthropic.MessageParam, system []anthropic.TextBlockParam) ([]openai.ChatCompletionMessageParamUnion, error) {
	var openaiMessages []openai.ChatCompletionMessageParamUnion

	// Add system message first if present
	if len(system) > 0 {
		systemContent := c.extractSystemContent(system)
		if systemContent != "" {
			openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
				OfSystem: &openai.ChatCompletionSystemMessageParam{
					Role: "system",
					Content: openai.ChatCompletionSystemMessageParamContentUnion{
						OfString: param.Opt[string]{Value: systemContent},
					},
				},
			})
		}
	}

	for _, msg := range messages {
		switch msg.Role {
		case anthropic.MessageParamRoleUser:
			// Convert user message
			content, err := c.convertUserContent(msg.Content)
			if err != nil {
				return nil, fmt.Errorf("failed to convert user message: %w", err)
			}

			openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Role:    "user",
					Content: content,
				},
			})

		case anthropic.MessageParamRoleAssistant:
			// Convert assistant message
			content, toolCalls, err := c.convertAssistantContent(msg.Content)
			if err != nil {
				return nil, fmt.Errorf("failed to convert assistant message: %w", err)
			}

			assistantMsg := &openai.ChatCompletionAssistantMessageParam{
				Role: "assistant",
			}

			if content != "" {
				assistantMsg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: param.Opt[string]{Value: content},
				}
			}

			if len(toolCalls) > 0 {
				assistantMsg.ToolCalls = toolCalls
			}

			openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
				OfAssistant: assistantMsg,
			})
		}
	}

	return openaiMessages, nil
}

// convertUserContent converts Anthropic user content to OpenAI format
func (c *AnthropicToOpenAIConverter) convertUserContent(content []anthropic.ContentBlockParamUnion) (openai.ChatCompletionUserMessageParamContentUnion, error) {
	if len(content) == 0 {
		return openai.ChatCompletionUserMessageParamContentUnion{
			OfString: param.Opt[string]{Value: ""},
		}, nil
	}

	// If only one text block, return as string
	if len(content) == 1 && content[0].OfText != nil {
		return openai.ChatCompletionUserMessageParamContentUnion{
			OfString: param.Opt[string]{Value: content[0].OfText.Text},
		}, nil
	}

	// Multi-part content
	var parts []openai.ChatCompletionContentPartUnionParam

	for _, block := range content {
		switch {
		case block.OfText != nil:
			parts = append(parts, openai.ChatCompletionContentPartUnionParam{
				OfText: &openai.ChatCompletionContentPartTextParam{
					Type: "text",
					Text: block.OfText.Text,
				},
			})
		case block.OfImage != nil:
			imagePart, err := c.convertImageContent(*block.OfImage)
			if err != nil {
				return openai.ChatCompletionUserMessageParamContentUnion{}, fmt.Errorf("failed to convert image: %w", err)
			}
			parts = append(parts, *imagePart)
		case block.OfToolResult != nil:
			// Convert tool result to tool message
			toolContent := c.extractToolResultContent(block.OfToolResult.Content)
			parts = append(parts, openai.ChatCompletionContentPartUnionParam{
				OfText: &openai.ChatCompletionContentPartTextParam{
					Type: "text",
					Text: fmt.Sprintf("Tool result (ID: %s): %s", block.OfToolResult.ToolUseID, toolContent),
				},
			})
		}
	}

	return openai.ChatCompletionUserMessageParamContentUnion{
		OfArrayOfContentParts: parts,
	}, nil
}

// convertAssistantContent converts Anthropic assistant content to OpenAI format
func (c *AnthropicToOpenAIConverter) convertAssistantContent(content []anthropic.ContentBlockParamUnion) (string, []openai.ChatCompletionMessageToolCallParam, error) {
	var textContent strings.Builder
	var toolCalls []openai.ChatCompletionMessageToolCallParam

	for _, block := range content {
		switch {
		case block.OfText != nil:
			if textContent.Len() > 0 {
				textContent.WriteString("\n")
			}
			textContent.WriteString(block.OfText.Text)
		case block.OfToolUse != nil:
			// Convert tool use to OpenAI tool call
			inputBytes, err := json.Marshal(block.OfToolUse.Input)
			if err != nil {
				return "", nil, fmt.Errorf("failed to marshal tool input: %w", err)
			}

			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallParam{
				ID:   block.OfToolUse.ID,
				Type: "function",
				Function: openai.ChatCompletionMessageToolCallFunctionParam{
					Name:      block.OfToolUse.Name,
					Arguments: string(inputBytes),
				},
			})
		}
	}

	return textContent.String(), toolCalls, nil
}

// convertImageContent converts Anthropic image content to OpenAI format
func (c *AnthropicToOpenAIConverter) convertImageContent(img anthropic.ImageBlockParam) (*openai.ChatCompletionContentPartUnionParam, error) {
	var url string

	if img.Source.OfBase64 != nil {
		// Convert base64 to data URL format
		mediaType := string(img.Source.OfBase64.MediaType)
		data := img.Source.OfBase64.Data
		url = fmt.Sprintf("data:%s;base64,%s", mediaType, data)
	} else if img.Source.OfURL != nil {
		url = img.Source.OfURL.URL
	} else {
		return nil, fmt.Errorf("unsupported image source type")
	}

	return &openai.ChatCompletionContentPartUnionParam{
		OfImageURL: &openai.ChatCompletionContentPartImageParam{
			Type: "image_url",
			ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
				URL: url,
			},
		},
	}, nil
}

// convertParameters converts Anthropic parameters to OpenAI equivalents
func (c *AnthropicToOpenAIConverter) convertParameters(anthropicReq *models.AnthropicMessageRequest, openaiReq *models.ChatCompletionRequest) error {
	// Temperature (both use 0-1 range)
	if anthropicReq.Temperature.Valid() {
		openaiReq.Temperature = param.Opt[float64]{Value: anthropicReq.Temperature.Value}
	}

	// Max tokens
	if anthropicReq.MaxTokens > 0 {
		openaiReq.MaxTokens = param.Opt[int64]{Value: int64(anthropicReq.MaxTokens)}
	}

	// Top-p (both use 0-1 range)
	if anthropicReq.TopP.Valid() {
		openaiReq.TopP = param.Opt[float64]{Value: anthropicReq.TopP.Value}
	}

	// Note: Some Anthropic parameters don't have OpenAI equivalents
	// These are silently ignored during conversion

	return nil
}

// convertTools converts Anthropic tools to OpenAI format
func (c *AnthropicToOpenAIConverter) convertTools(tools []anthropic.ToolUnionParam) ([]openai.ChatCompletionToolParam, error) {
	var openaiTools []openai.ChatCompletionToolParam

	for _, tool := range tools {
		if tool.OfTool != nil {
			openaiTool := openai.ChatCompletionToolParam{
				Type: "function",
				Function: openai.FunctionDefinitionParam{
					Name: tool.OfTool.Name,
				},
			}

			if tool.OfTool.Description.Valid() {
				openaiTool.Function.Description = param.Opt[string]{Value: tool.OfTool.Description.Value}
			}

			// Convert input schema to function parameters
			if tool.OfTool.InputSchema.Properties != nil {
				// Type assertion needed since Properties is any
				if properties, ok := tool.OfTool.InputSchema.Properties.(map[string]any); ok {
					openaiTool.Function.Parameters = properties
				}
			}

			openaiTools = append(openaiTools, openaiTool)
		}
	}

	return openaiTools, nil
}

// convertStopSequences converts Anthropic stop sequences to OpenAI format
func (c *AnthropicToOpenAIConverter) convertStopSequences(stopSequences []string) openai.ChatCompletionNewParamsStopUnion {
	if len(stopSequences) == 1 {
		return openai.ChatCompletionNewParamsStopUnion{
			OfString: param.Opt[string]{Value: stopSequences[0]},
		}
	}
	return openai.ChatCompletionNewParamsStopUnion{
		OfStringArray: stopSequences,
	}
}

// extractSystemContent extracts string content from Anthropic system message
func (c *AnthropicToOpenAIConverter) extractSystemContent(system []anthropic.TextBlockParam) string {
	var text strings.Builder
	for _, block := range system {
		if text.Len() > 0 {
			text.WriteString("\n\n")
		}
		text.WriteString(block.Text)
	}
	return text.String()
}

// extractToolResultContent extracts content from Anthropic tool result
func (c *AnthropicToOpenAIConverter) extractToolResultContent(content []anthropic.ToolResultBlockParamContentUnion) string {
	var text strings.Builder
	for _, block := range content {
		switch {
		case block.OfText != nil:
			if text.Len() > 0 {
				text.WriteString("\n")
			}
			text.WriteString(block.OfText.Text)
		case block.OfImage != nil:
			if text.Len() > 0 {
				text.WriteString("\n")
			}
			text.WriteString("[Image content]")
		}
	}
	return text.String()
}

// convertStopReason converts Anthropic stop reason to OpenAI finish reason
func (c *AnthropicToOpenAIConverter) convertStopReason(stopReason anthropic.StopReason) string {
	switch stopReason {
	case anthropic.StopReasonEndTurn:
		return "stop"
	case anthropic.StopReasonMaxTokens:
		return "length"
	case anthropic.StopReasonToolUse:
		return "tool_calls"
	case anthropic.StopReasonRefusal:
		return "content_filter"
	default:
		return "stop"
	}
}

// convertUsage converts Anthropic usage to OpenAI usage
func (c *AnthropicToOpenAIConverter) convertUsage(usage anthropic.Usage) models.AdaptiveUsage {
	return models.AdaptiveUsage{
		PromptTokens:     usage.InputTokens,
		CompletionTokens: usage.OutputTokens,
		TotalTokens:      usage.InputTokens + usage.OutputTokens,
	}
}

// convertResponseContent converts Anthropic content blocks to OpenAI message content
func (c *AnthropicToOpenAIConverter) convertResponseContent(content []anthropic.ContentBlockUnion) (string, []openai.ChatCompletionMessageToolCall, error) {
	var textContent strings.Builder
	var toolCalls []openai.ChatCompletionMessageToolCall

	for _, block := range content {
		switch block.Type {
		case "text":
			if textContent.Len() > 0 {
				textContent.WriteString("\n")
			}
			textContent.WriteString(block.Text)
		case "tool_use":
			// Convert tool use to OpenAI tool call
			inputBytes, err := json.Marshal(block.Input)
			if err != nil {
				return "", nil, fmt.Errorf("failed to marshal tool input: %w", err)
			}

			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCall{
				ID:   block.ID,
				Type: "function",
				Function: openai.ChatCompletionMessageToolCallFunction{
					Name:      block.Name,
					Arguments: string(inputBytes),
				},
			})
		}
	}

	return textContent.String(), toolCalls, nil
}

// ConvertResponse converts Anthropic Message to OpenAI ChatCompletion format
func (c *AnthropicToOpenAIConverter) ConvertResponse(resp *anthropic.Message, provider string) (*models.ChatCompletion, error) {
	if resp == nil {
		return nil, fmt.Errorf("anthropic message cannot be nil")
	}

	// Use helper function to convert response content
	content, toolCalls, err := c.convertResponseContent(resp.Content)
	if err != nil {
		return nil, fmt.Errorf("failed to convert response content: %w", err)
	}

	// Use helper to convert stop reason
	finishReason := c.convertStopReason(resp.StopReason)

	// Use helper to convert usage
	usage := c.convertUsage(resp.Usage)

	// Create OpenAI ChatCompletion
	chatCompletion := &models.ChatCompletion{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: 0, // Would need timestamp conversion
		Model:   string(resp.Model),
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:      "assistant",
					Content:   content,
					ToolCalls: toolCalls,
				},
				FinishReason: finishReason,
			},
		},
		Usage: usage,
	}

	return chatCompletion, nil
}

// ConvertStreamingChunk converts Anthropic streaming event to OpenAI streaming chunk format
func (c *AnthropicToOpenAIConverter) ConvertStreamingChunk(chunk any, provider string) (*models.ChatCompletionChunk, error) {
	switch event := chunk.(type) {
	case *anthropic.MessageStartEvent:
		// Convert to OpenAI chat completion chunk
		return &models.ChatCompletionChunk{
			ID:      event.Message.ID,
			Object:  "chat.completion.chunk",
			Created: 0,
			Model:   string(event.Message.Model),
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Index: 0,
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Role: "assistant",
					},
					FinishReason: "",
				},
			},
		}, nil

	case *anthropic.ContentBlockStartEvent:
		if event.ContentBlock.Type == "tool_use" {
			// Start of tool call
			return &models.ChatCompletionChunk{
				ID:      "", // Would need to track from message start
				Object:  "chat.completion.chunk",
				Created: 0,
				Model:   "", // Would need to track from message start
				Choices: []openai.ChatCompletionChunkChoice{
					{
						Index: event.Index,
						Delta: openai.ChatCompletionChunkChoiceDelta{
							ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{
								{
									Index: 0,
									ID:    event.ContentBlock.ID,
									Type:  "function",
									Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
										Name:      event.ContentBlock.Name,
										Arguments: "",
									},
								},
							},
						},
						FinishReason: "",
					},
				},
			}, nil
		}
		return nil, nil // Skip non-tool content block starts

	case *anthropic.ContentBlockDeltaEvent:
		switch event.Delta.Type {
		case "text_delta":
			// Text content delta
			return &models.ChatCompletionChunk{
				ID:      "", // Would need to track from message start
				Object:  "chat.completion.chunk",
				Created: 0,
				Model:   "", // Would need to track from message start
				Choices: []openai.ChatCompletionChunkChoice{
					{
						Index: event.Index,
						Delta: openai.ChatCompletionChunkChoiceDelta{
							Content: event.Delta.Text,
						},
						FinishReason: "",
					},
				},
			}, nil
		case "input_json_delta":
			// Tool function arguments delta
			return &models.ChatCompletionChunk{
				ID:      "", // Would need to track from message start
				Object:  "chat.completion.chunk",
				Created: 0,
				Model:   "", // Would need to track from message start
				Choices: []openai.ChatCompletionChunkChoice{
					{
						Index: event.Index,
						Delta: openai.ChatCompletionChunkChoiceDelta{
							ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{
								{
									Index: 0,
									Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
										Arguments: event.Delta.PartialJSON,
									},
								},
							},
						},
						FinishReason: "",
					},
				},
			}, nil
		}

	case *anthropic.MessageStopEvent:
		// End of streaming
		return &models.ChatCompletionChunk{
			ID:      "", // Would need to track from message start
			Object:  "chat.completion.chunk",
			Created: 0,
			Model:   "", // Would need to track from message start
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Index:        0,
					Delta:        openai.ChatCompletionChunkChoiceDelta{},
					FinishReason: "stop",
				},
			},
		}, nil
	}

	return nil, fmt.Errorf("unsupported Anthropic streaming event type: %T", chunk)
}
