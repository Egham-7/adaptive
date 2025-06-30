package anthropic

import (
	"context"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicssestream "github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

// AnthropicCompletions implements the Completions interface for Anthropic
type AnthropicCompletions struct {
	client *anthropic.Client
}

// NewAnthropicCompletions creates a new Anthropic completions client
func NewAnthropicCompletions(client *anthropic.Client) *AnthropicCompletions {
	return &AnthropicCompletions{
		client: client,
	}
}

// CreateCompletion implements Completions interface
func (c *AnthropicCompletions) CreateCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("messages array cannot be empty")
	}

	// Convert OpenAI messages to Anthropic format
	messages, systemPrompt, err := c.convertOpenAIToAnthropicMessages(req.Messages)
	if err != nil {
		return nil, fmt.Errorf("failed to convert messages: %w", err)
	}

	// Determine which model to use
	model := c.determineAnthropicModel(string(req.Model))

	// Create Anthropic request
	anthropicReq := anthropic.MessageNewParams{
		Model:     model,
		Messages:  messages,
		MaxTokens: c.getMaxTokens(req),
	}

	// Add system prompt if present
	if systemPrompt != "" {
		anthropicReq.System = []anthropic.TextBlockParam{
			{
				Type: "text",
				Text: systemPrompt,
			},
		}
	}

	// Set optional parameters
	c.setOptionalParameters(&anthropicReq, req)

	// Call Anthropic API
	resp, err := c.client.Messages.New(ctx, anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic chat completion failed: %w", err)
	}

	// Convert Anthropic response to OpenAI format
	return c.convertAnthropicToOpenAIResponse(resp), nil
}

// StreamCompletion implements Completions interface
func (c *AnthropicCompletions) StreamCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("messages array cannot be empty")
	}

	// Convert OpenAI messages to Anthropic format
	messages, systemPrompt, err := c.convertOpenAIToAnthropicMessages(req.Messages)
	if err != nil {
		return nil, fmt.Errorf("failed to convert messages: %w", err)
	}

	// Determine which model to use
	model := c.determineAnthropicModel(string(req.Model))

	// Create Anthropic request
	anthropicReq := anthropic.MessageNewParams{
		Model:     model,
		Messages:  messages,
		MaxTokens: c.getMaxTokens(req),
	}

	// Add system prompt if present
	if systemPrompt != "" {
		anthropicReq.System = []anthropic.TextBlockParam{
			{
				Type: "text",
				Text: systemPrompt,
			},
		}
	}

	// Set optional parameters
	c.setOptionalParameters(&anthropicReq, req)

	// Create streaming request
	stream := c.client.Messages.NewStreaming(ctx, anthropicReq)

	// Convert Anthropic stream to OpenAI format
	return c.convertAnthropicStreamToOpenAI(stream)
}

// convertOpenAIToAnthropicMessages converts OpenAI messages to Anthropic format
// Returns messages array and system prompt separately
func (c *AnthropicCompletions) convertOpenAIToAnthropicMessages(msgs []openai.ChatCompletionMessageParamUnion) ([]anthropic.MessageParam, string, error) {
	var messages []anthropic.MessageParam
	var systemPrompts []string

	for i, msg := range msgs {
		switch {
		case msg.OfUser != nil:
			anthropicMsg, err := c.convertUserMessage(msg.OfUser)
			if err != nil {
				return nil, "", fmt.Errorf("failed to convert user message at index %d: %w", i, err)
			}
			messages = append(messages, anthropicMsg)

		case msg.OfAssistant != nil:
			anthropicMsg, err := c.convertAssistantMessage(msg.OfAssistant)
			if err != nil {
				return nil, "", fmt.Errorf("failed to convert assistant message at index %d: %w", i, err)
			}
			messages = append(messages, anthropicMsg)

		case msg.OfSystem != nil:
			// Collect system messages into system prompt
			content := c.extractContentFromSystemMessage(msg.OfSystem)
			if content != "" {
				systemPrompts = append(systemPrompts, content)
			}

		case msg.OfDeveloper != nil:
			// Treat developer messages as system messages
			content := c.extractContentFromDeveloperMessage(msg.OfDeveloper)
			if content != "" {
				systemPrompts = append(systemPrompts, content)
			}

		case msg.OfTool != nil:
			// For now, convert tool messages to user messages with context
			anthropicMsg := c.convertToolMessage(msg.OfTool)
			messages = append(messages, anthropicMsg)

		case msg.OfFunction != nil:
			// Skip deprecated function messages
			continue

		default:
			return nil, "", fmt.Errorf("unknown message type at index %d", i)
		}
	}

	// Combine all system prompts
	systemPrompt := strings.Join(systemPrompts, "\n\n")

	return messages, systemPrompt, nil
}

// convertUserMessage converts OpenAI user message to Anthropic format
func (c *AnthropicCompletions) convertUserMessage(msg *openai.ChatCompletionUserMessageParam) (anthropic.MessageParam, error) {
	content, err := c.extractUserMessageContent(msg)
	if err != nil {
		return anthropic.MessageParam{}, err
	}

	return anthropic.NewUserMessage(anthropic.NewTextBlock(content)), nil
}

// convertAssistantMessage converts OpenAI assistant message to Anthropic format
func (c *AnthropicCompletions) convertAssistantMessage(msg *openai.ChatCompletionAssistantMessageParam) (anthropic.MessageParam, error) {
	content := c.extractAssistantMessageContent(msg)
	return anthropic.NewAssistantMessage(anthropic.NewTextBlock(content)), nil
}

// convertToolMessage converts OpenAI tool message to Anthropic format
func (c *AnthropicCompletions) convertToolMessage(msg *openai.ChatCompletionToolMessageParam) anthropic.MessageParam {
	content := c.extractToolMessageContent(msg)
	contextualContent := fmt.Sprintf("Tool response (ID: %s): %s", msg.ToolCallID, content)
	return anthropic.NewUserMessage(anthropic.NewTextBlock(contextualContent))
}


// extractUserMessageContent extracts content from OpenAI user message
func (c *AnthropicCompletions) extractUserMessageContent(msg *openai.ChatCompletionUserMessageParam) (string, error) {
	contentUnion := msg.Content

	// Handle string content
	if contentUnion.OfString.Valid() {
		return contentUnion.OfString.Value, nil
	}

	// Handle array of content parts
	if len(contentUnion.OfArrayOfContentParts) > 0 {
		var textParts []string

		for _, part := range contentUnion.OfArrayOfContentParts {
			switch {
			case part.OfText != nil:
				textParts = append(textParts, part.OfText.Text)
			case part.OfImageURL != nil:
				// For now, just indicate that an image was present
				textParts = append(textParts, "[Image content not supported in Anthropic conversion]")
			case part.OfInputAudio != nil:
				// For now, just indicate that audio was present
				textParts = append(textParts, "[Audio content not supported in Anthropic conversion]")
			case part.OfFile != nil:
				// For now, just indicate that a file was present
				textParts = append(textParts, "[File content not supported in Anthropic conversion]")
			}
		}

		if len(textParts) > 0 {
			return strings.Join(textParts, "\n"), nil
		}
	}

	return "", fmt.Errorf("no valid content found in user message")
}

// extractAssistantMessageContent extracts content from OpenAI assistant message
func (c *AnthropicCompletions) extractAssistantMessageContent(msg *openai.ChatCompletionAssistantMessageParam) string {
	contentUnion := msg.Content

	// Handle string content
	if contentUnion.OfString.Valid() {
		return contentUnion.OfString.Value
	}

	// Handle array of content parts
	if len(contentUnion.OfArrayOfContentParts) > 0 {
		var textParts []string

		for _, part := range contentUnion.OfArrayOfContentParts {
			switch {
			case part.OfText != nil:
				textParts = append(textParts, part.OfText.Text)
			case part.OfRefusal != nil:
				textParts = append(textParts, fmt.Sprintf("[Refusal: %s]", part.OfRefusal.Refusal))
			}
		}

		if len(textParts) > 0 {
			return strings.Join(textParts, "\n")
		}
	}

	// Handle refusal
	if msg.Refusal.Valid() && msg.Refusal.Value != "" {
		return fmt.Sprintf("[Refusal: %s]", msg.Refusal.Value)
	}

	// Handle tool calls (simplified representation)
	if len(msg.ToolCalls) > 0 {
		var toolCallTexts []string
		for _, toolCall := range msg.ToolCalls {
			toolCallTexts = append(toolCallTexts, fmt.Sprintf("Called function %s with arguments: %s",
				toolCall.Function.Name, toolCall.Function.Arguments))
		}
		return strings.Join(toolCallTexts, "\n")
	}

	return "No content available"
}

// extractContentFromSystemMessage extracts content from OpenAI system message
func (c *AnthropicCompletions) extractContentFromSystemMessage(msg *openai.ChatCompletionSystemMessageParam) string {
	contentUnion := msg.Content

	// Handle string content
	if contentUnion.OfString.Valid() {
		return contentUnion.OfString.Value
	}

	// Handle array of content parts
	if len(contentUnion.OfArrayOfContentParts) > 0 {
		var textParts []string
		for _, part := range contentUnion.OfArrayOfContentParts {
			textParts = append(textParts, part.Text)
		}
		return strings.Join(textParts, "\n")
	}

	return ""
}

// extractContentFromDeveloperMessage extracts content from OpenAI developer message
func (c *AnthropicCompletions) extractContentFromDeveloperMessage(msg *openai.ChatCompletionDeveloperMessageParam) string {
	contentUnion := msg.Content

	// Handle string content
	if contentUnion.OfString.Valid() {
		return contentUnion.OfString.Value
	}

	// Handle array of content parts
	if len(contentUnion.OfArrayOfContentParts) > 0 {
		var textParts []string
		for _, part := range contentUnion.OfArrayOfContentParts {
			textParts = append(textParts, part.Text)
		}
		return strings.Join(textParts, "\n")
	}

	return ""
}

// extractToolMessageContent extracts content from OpenAI tool message
func (c *AnthropicCompletions) extractToolMessageContent(msg *openai.ChatCompletionToolMessageParam) string {
	contentUnion := msg.Content

	// Handle string content
	if contentUnion.OfString.Valid() {
		return contentUnion.OfString.Value
	}

	// Handle array of content parts
	if len(contentUnion.OfArrayOfContentParts) > 0 {
		var textParts []string
		for _, part := range contentUnion.OfArrayOfContentParts {
			textParts = append(textParts, part.Text)
		}
		return strings.Join(textParts, "\n")
	}

	return ""
}

// getMaxTokens safely extracts max tokens from request
func (c *AnthropicCompletions) getMaxTokens(req *openai.ChatCompletionNewParams) int64 {
	// Prefer MaxCompletionTokens over deprecated MaxTokens
	if req.MaxCompletionTokens.Valid() && req.MaxCompletionTokens.Value > 0 {
		return req.MaxCompletionTokens.Value
	}

	if req.MaxTokens.Valid() && req.MaxTokens.Value > 0 {
		return req.MaxTokens.Value
	}

	// Default max tokens for Anthropic
	return 4096
}

// setOptionalParameters sets optional parameters on Anthropic request
func (c *AnthropicCompletions) setOptionalParameters(anthropicReq *anthropic.MessageNewParams, req *openai.ChatCompletionNewParams) {
	if req.Temperature.Valid() && req.Temperature.Value > 0 {
		anthropicReq.Temperature = anthropic.Float(req.Temperature.Value)
	}

	if req.TopP.Valid() && req.TopP.Value > 0 {
		anthropicReq.TopP = anthropic.Float(req.TopP.Value)
	}

	// Handle stop sequences
	if req.Stop.OfString.Valid() {
		anthropicReq.StopSequences = []string{req.Stop.OfString.Value}
	} else if len(req.Stop.OfStringArray) > 0 {
		anthropicReq.StopSequences = req.Stop.OfStringArray
	}
}

// convertAnthropicToOpenAIResponse converts Anthropic response to OpenAI format
func (c *AnthropicCompletions) convertAnthropicToOpenAIResponse(resp *anthropic.Message) *openai.ChatCompletion {
	// Extract content from Anthropic response
	var content string
	if len(resp.Content) > 0 {
		var textParts []string
		for _, contentBlock := range resp.Content {
			if textBlock := contentBlock.AsText(); textBlock.Type == "text" {
				textParts = append(textParts, textBlock.Text)
			}
		}
		content = strings.Join(textParts, "\n")
	}

	// Determine finish reason
	finishReason := c.mapAnthropicStopReason(resp.StopReason)

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
				FinishReason: finishReason,
			},
		},
		Usage: openai.CompletionUsage{
			PromptTokens:     int64(resp.Usage.InputTokens),
			CompletionTokens: int64(resp.Usage.OutputTokens),
			TotalTokens:      int64(resp.Usage.InputTokens + resp.Usage.OutputTokens),
		},
		SystemFingerprint: "", // Anthropic doesn't provide this
	}
}

// mapAnthropicStopReason maps Anthropic stop reason to OpenAI format
func (c *AnthropicCompletions) mapAnthropicStopReason(reason anthropic.StopReason) string {
	switch reason {
	case anthropic.StopReasonEndTurn:
		return "stop"
	case anthropic.StopReasonMaxTokens:
		return "length"
	case anthropic.StopReasonStopSequence:
		return "stop"
	case anthropic.StopReasonToolUse:
		return "tool_calls"
	default:
		return "stop"
	}
}

// determineAnthropicModel assumes the requested model is always a valid Anthropic model
func (c *AnthropicCompletions) determineAnthropicModel(requestedModel string) anthropic.Model {
	if requestedModel == "" {
		return anthropic.ModelClaude3_5SonnetLatest // Default model
	}

	// Assume the requested model is always a valid Anthropic model
	return anthropic.Model(requestedModel)
}

// convertAnthropicStreamToOpenAI converts Anthropic stream to OpenAI format
func (c *AnthropicCompletions) convertAnthropicStreamToOpenAI(stream *anthropicssestream.Stream[anthropic.MessageStreamEventUnion]) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	adapter := NewAnthropicStreamAdapter(stream)
	return adapter.ConvertToOpenAIStream()
}
