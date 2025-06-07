package anthropic

import (
	"context"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicssestream "github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

// AnthropicCompletions implements the Completions interface for Anthropic
type AnthropicCompletions struct {
	client *anthropic.Client
}

// CreateCompletion implements Completions interface
func (c *AnthropicCompletions) CreateCompletion(req *openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Convert OpenAI messages to Anthropic format
	messages := make([]anthropic.MessageParam, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = convertOpenAIToAnthropicMessage(msg)
	}

	// Determine which model to use
	model := determineAnthropicModel(string(req.Model))

	// Create Anthropic request
	anthropicReq := anthropic.MessageNewParams{
		Model:    model,
		Messages: messages,
	}

	if req.MaxTokens.Value > 0 {
		anthropicReq.MaxTokens = req.MaxTokens.Value
	}
	// Set optional parameters if provided
	if req.Temperature.Value > 0 {
		anthropicReq.Temperature = anthropic.Float(req.Temperature.Value)
	}

	if req.TopP.Value > 0 {
		anthropicReq.TopP = anthropic.Float(req.TopP.Value)
	}

	// Call Anthropic API
	resp, err := c.client.Messages.New(context.Background(), anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic chat completion failed: %w", err)
	}

	// Convert Anthropic response to OpenAI format
	return convertAnthropicToOpenAIResponse(resp), nil
}

// StreamCompletion implements Completions interface
func (c *AnthropicCompletions) StreamCompletion(req *openai.ChatCompletionNewParams) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Convert OpenAI messages to Anthropic format
	messages := make([]anthropic.MessageParam, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = convertOpenAIToAnthropicMessage(msg)
	}

	// Determine which model to use
	model := determineAnthropicModel(string(req.Model))

	// Create Anthropic request
	anthropicReq := anthropic.MessageNewParams{
		Model:     model,
		Messages:  messages,
		MaxTokens: req.MaxTokens.Value,
	}

	// Set optional parameters if provided
	if req.Temperature.Value > 0 {
		anthropicReq.Temperature = anthropic.Float(req.Temperature.Value)
	}

	if req.TopP.Value > 0 {
		anthropicReq.TopP = anthropic.Float(req.TopP.Value)
	}

	// Create streaming request
	stream := c.client.Messages.NewStreaming(context.Background(), anthropicReq)

	// Convert Anthropic stream to OpenAI format
	return convertAnthropicStreamToOpenAI(stream)
}

// convertOpenAIToAnthropicMessage converts OpenAI message to Anthropic format
func convertOpenAIToAnthropicMessage(msg openai.ChatCompletionMessageParamUnion) anthropic.MessageParam {
	var role anthropic.MessageParamRole
	var content string

	// Handle different message types using the union accessors
	if userMsg := msg.OfUser; userMsg != nil {
		role = anthropic.MessageParamRoleUser
		// Handle content union type
		contentUnion := userMsg.Content
		if contentUnion.OfString.Valid() {
			content = contentUnion.OfString.Value
		} else if len(contentUnion.OfArrayOfContentParts) > 0 {
			// For now, just take first text part
			for _, part := range contentUnion.OfArrayOfContentParts {
				if textPart := part.OfText; textPart != nil {
					content = textPart.Text
					break
				}
			}
		}
	} else if assistantMsg := msg.OfAssistant; assistantMsg != nil {
		role = anthropic.MessageParamRoleAssistant
		// Handle content union type
		contentUnion := assistantMsg.Content
		if contentUnion.OfString.Valid() {
			content = contentUnion.OfString.Value
		} else if len(contentUnion.OfArrayOfContentParts) > 0 {
			// For now, just take first text part
			for _, part := range contentUnion.OfArrayOfContentParts {
				if textPart := part.OfText; textPart != nil {
					content = textPart.Text
					break
				}
			}
		}
	} else if systemMsg := msg.OfSystem; systemMsg != nil {
		role = anthropic.MessageParamRoleUser // Anthropic doesn't have system role in messages
		contentUnion := systemMsg.Content
		if contentUnion.OfString.Valid() {
			content = contentUnion.OfString.Value
		} else if len(contentUnion.OfArrayOfContentParts) > 0 {
			// For now, just take first text part
			for _, part := range contentUnion.OfArrayOfContentParts {
				content = part.Text
				break
			}
		}
	} else if developerMsg := msg.OfDeveloper; developerMsg != nil {
		role = anthropic.MessageParamRoleUser // Treat developer as user
		contentUnion := developerMsg.Content
		if contentUnion.OfString.Valid() {
			content = contentUnion.OfString.Value
		} else if len(contentUnion.OfArrayOfContentParts) > 0 {
			// For now, just take first text part
			for _, part := range contentUnion.OfArrayOfContentParts {
				content = part.Text
				break
			}
		}
	} else {
		// Default case
		role = anthropic.MessageParamRoleUser
		content = "Unknown message type"
	}

	// Create the content block
	contentBlock := anthropic.NewTextBlock(content)

	// Create the message
	return anthropic.MessageParam{
		Role:    role,
		Content: []anthropic.ContentBlockParamUnion{contentBlock},
	}
}

// convertAnthropicToOpenAIResponse converts Anthropic response to OpenAI format
func convertAnthropicToOpenAIResponse(resp *anthropic.Message) *openai.ChatCompletion {
	// Extract content from Anthropic response
	var content string
	if len(resp.Content) > 0 {
		// Use the AsText() method to get TextBlock
		if textBlock := resp.Content[0].AsText(); textBlock.Type == "text" {
			content = textBlock.Text
		}
	}

	return &openai.ChatCompletion{
		ID:     resp.ID,
		Object: "chat.completion",
		Model:  string(resp.Model),
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: content,
				},
				FinishReason: "stop",
			},
		},
		Usage: openai.CompletionUsage{
			PromptTokens:     int64(resp.Usage.InputTokens),
			CompletionTokens: int64(resp.Usage.OutputTokens),
			TotalTokens:      int64(resp.Usage.InputTokens + resp.Usage.OutputTokens),
		},
	}
}

// determineAnthropicModel selects the appropriate model based on request
func determineAnthropicModel(requestedModel string) anthropic.Model {
	if requestedModel == "" {
		return anthropic.ModelClaude3_5SonnetLatest // Default model
	}

	// Map of supported models
	supportedModels := map[anthropic.Model]bool{
		anthropic.ModelClaude3_5HaikuLatest: true,
		anthropic.ModelClaudeOpus4_0:        true,
		anthropic.ModelClaudeSonnet4_0:      true,
	}

	// Try to match requested model with supported models
	for model := range supportedModels {
		if string(model) == requestedModel {
			return model
		}
	}

	// If no match found, return default
	return anthropic.ModelClaude3_5SonnetLatest
}

// convertAnthropicStreamToOpenAI converts Anthropic stream to OpenAI format
func convertAnthropicStreamToOpenAI(stream *anthropicssestream.Stream[anthropic.MessageStreamEventUnion]) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	adapter := NewAnthropicStreamAdapter(stream)
	return adapter.ConvertToOpenAIStream()
}
