package anthropic

import (
	"context"
	"encoding/json"
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

	// Convert tools if present
	if len(req.Tools) > 0 {
		anthropicReq.Tools = c.convertTools(req.Tools)
	}

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

	// Convert tools if present
	if len(req.Tools) > 0 {
		anthropicReq.Tools = c.convertTools(req.Tools)
	}

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
			// Convert tool messages to tool result blocks
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

// convertUserMessage converts OpenAI user message to Anthropic format with full content support
func (c *AnthropicCompletions) convertUserMessage(msg *openai.ChatCompletionUserMessageParam) (anthropic.MessageParam, error) {
	contentUnion := msg.Content

	// Handle simple string content
	if contentUnion.OfString.Valid() {
		return anthropic.NewUserMessage(anthropic.NewTextBlock(contentUnion.OfString.Value)), nil
	}

	// Handle array of content parts
	if len(contentUnion.OfArrayOfContentParts) > 0 {
		var blocks []anthropic.ContentBlockParamUnion

		for _, part := range contentUnion.OfArrayOfContentParts {
			switch {
			case part.OfText != nil:
				blocks = append(blocks, anthropic.NewTextBlock(part.OfText.Text))

			case part.OfImageURL != nil:
				imageBlock, err := c.convertImageContent(part.OfImageURL)
				if err != nil {
					// Fallback to text description if conversion fails
					blocks = append(blocks, anthropic.NewTextBlock("[Image content - conversion failed: "+err.Error()+"]"))
				} else {
					blocks = append(blocks, imageBlock)
				}

			case part.OfInputAudio != nil:
				// Anthropic doesn't support audio yet, but be explicit about it
				blocks = append(blocks, anthropic.NewTextBlock("[Audio content - not yet supported by Anthropic API]"))

			case part.OfFile != nil:
				fileBlock, err := c.convertFileContent(part.OfFile)
				if err != nil {
					// Fallback to text description if conversion fails
					blocks = append(blocks, anthropic.NewTextBlock("[File content - conversion failed: "+err.Error()+"]"))
				} else {
					blocks = append(blocks, fileBlock)
				}
			}
		}

		if len(blocks) > 0 {
			return anthropic.MessageParam{
				Role:    anthropic.MessageParamRoleUser,
				Content: blocks,
			}, nil
		}
	}

	return anthropic.MessageParam{}, fmt.Errorf("no valid content found in user message")
}

// convertAssistantMessage converts OpenAI assistant message to Anthropic format with full tool call support
func (c *AnthropicCompletions) convertAssistantMessage(msg *openai.ChatCompletionAssistantMessageParam) (anthropic.MessageParam, error) {
	var blocks []anthropic.ContentBlockParamUnion

	// Handle text content
	content := c.extractAssistantMessageContent(msg)
	if content != "" {
		blocks = append(blocks, anthropic.NewTextBlock(content))
	}

	// Handle tool calls
	if len(msg.ToolCalls) > 0 {
		toolBlocks := c.convertToolCalls(msg.ToolCalls)
		blocks = append(blocks, toolBlocks...)
	}

	if len(blocks) > 0 {
		return anthropic.MessageParam{
			Role:    anthropic.MessageParamRoleAssistant,
			Content: blocks,
		}, nil
	}

	// Fallback to empty text block
	return anthropic.NewAssistantMessage(anthropic.NewTextBlock("")), nil
}

// convertToolMessage converts OpenAI tool message to Anthropic tool result
func (c *AnthropicCompletions) convertToolMessage(msg *openai.ChatCompletionToolMessageParam) anthropic.MessageParam {
	content := c.extractToolMessageContent(msg)

	// Create tool result block
	toolResultBlock := anthropic.NewToolResultBlock(
		msg.ToolCallID,
		content,
		false, // isError
	)

	return anthropic.NewUserMessage(toolResultBlock)
}

// convertImageContent converts OpenAI image content to Anthropic format
func (c *AnthropicCompletions) convertImageContent(imageURL *openai.ChatCompletionContentPartImageParam) (anthropic.ContentBlockParamUnion, error) {
	if imageURL.ImageURL.URL == "" {
		return anthropic.ContentBlockParamUnion{}, fmt.Errorf("image URL is empty")
	}

	// If it's a base64 data URL
	if strings.HasPrefix(imageURL.ImageURL.URL, "data:") {
		return c.convertBase64Image(imageURL.ImageURL.URL)
	}

	// If it's a regular URL
	return anthropic.NewImageBlock(anthropic.URLImageSourceParam{
		URL:  imageURL.ImageURL.URL,
		Type: "url",
	}), nil
}

// convertBase64Image converts base64 data URL to Anthropic format
func (c *AnthropicCompletions) convertBase64Image(dataURL string) (anthropic.ContentBlockParamUnion, error) {
	// Parse data:image/jpeg;base64,<data>
	parts := strings.Split(dataURL, ",")
	if len(parts) != 2 {
		return anthropic.ContentBlockParamUnion{}, fmt.Errorf("invalid data URL format")
	}

	header := parts[0]
	data := parts[1]

	// Extract media type
	var mediaType string
	if strings.Contains(header, "image/jpeg") {
		mediaType = "image/jpeg"
	} else if strings.Contains(header, "image/png") {
		mediaType = "image/png"
	} else if strings.Contains(header, "image/gif") {
		mediaType = "image/gif"
	} else if strings.Contains(header, "image/webp") {
		mediaType = "image/webp"
	} else {
		return anthropic.ContentBlockParamUnion{}, fmt.Errorf("unsupported image type")
	}

	return anthropic.NewImageBlockBase64(mediaType, data), nil
}

// convertFileContent converts OpenAI file content to Anthropic format
func (c *AnthropicCompletions) convertFileContent(file *openai.ChatCompletionContentPartFileParam) (anthropic.ContentBlockParamUnion, error) {
	// Handle file based on available data
	if file.File.FileData.Valid() {
		// Base64 file data
		return c.convertBase64File(file.File.FileData.Value, file.File.Filename.Value)
	}

	if file.File.FileID.Valid() {
		// File ID - would need to fetch from OpenAI API, simplified for now
		return anthropic.NewDocumentBlock(anthropic.PlainTextSourceParam{
			Data:      fmt.Sprintf("File ID: %s (content would need to be fetched)", file.File.FileID.Value),
			MediaType: "text/plain",
			Type:      "text",
		}), nil
	}

	return anthropic.ContentBlockParamUnion{}, fmt.Errorf("no valid file data found")
}

// convertBase64File converts base64 file data to Anthropic format
func (c *AnthropicCompletions) convertBase64File(fileData, filename string) (anthropic.ContentBlockParamUnion, error) {
	// Determine file type from filename or assume text
	if strings.HasSuffix(strings.ToLower(filename), ".pdf") {
		return anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{
			Data:      fileData,
			MediaType: "application/pdf",
			Type:      "base64",
		}), nil
	}

	// Default to text file
	return anthropic.NewDocumentBlock(anthropic.PlainTextSourceParam{
		Data:      fileData, // Note: This assumes text content, may need decoding
		MediaType: "text/plain",
		Type:      "text",
	}), nil
}

// convertToolCalls converts OpenAI tool calls to Anthropic format
func (c *AnthropicCompletions) convertToolCalls(toolCalls []openai.ChatCompletionMessageToolCallParam) []anthropic.ContentBlockParamUnion {
	var blocks []anthropic.ContentBlockParamUnion

	for _, toolCall := range toolCalls {
		block := anthropic.NewToolUseBlock(
			toolCall.ID,
			json.RawMessage(toolCall.Function.Arguments),
			toolCall.Function.Name,
		)
		blocks = append(blocks, block)
	}

	return blocks
}

// convertTools converts OpenAI tools to Anthropic format
func (c *AnthropicCompletions) convertTools(tools []openai.ChatCompletionToolParam) []anthropic.ToolUnionParam {
	var anthropicTools []anthropic.ToolUnionParam

	for _, tool := range tools {
		// Convert OpenAI function to Anthropic tool
		anthropicTool := anthropic.ToolUnionParamOfTool(
			anthropic.ToolInputSchemaParam{
				Type:       "object",
				Properties: tool.Function.Parameters,
			},
			tool.Function.Name,
		)

		if tool.Function.Description.Valid() {
			if anthropicTool.OfTool != nil {
				anthropicTool.OfTool.Description = anthropic.String(tool.Function.Description.Value)
			}
		}

		anthropicTools = append(anthropicTools, anthropicTool)
	}

	return anthropicTools
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

	// Don't include tool calls in text content since they're handled separately
	return ""
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

	// Handle tool choice
	if !c.isToolChoiceEmpty(req.ToolChoice) {
		anthropicReq.ToolChoice = c.convertToolChoice(req.ToolChoice)
	}
}

// isToolChoiceEmpty checks if tool choice is empty/default
func (c *AnthropicCompletions) isToolChoiceEmpty(toolChoice openai.ChatCompletionToolChoiceOptionUnionParam) bool {
	return !toolChoice.OfAuto.Valid() && toolChoice.OfChatCompletionNamedToolChoice == nil
}

// convertToolChoice converts OpenAI tool choice to Anthropic format
func (c *AnthropicCompletions) convertToolChoice(toolChoice openai.ChatCompletionToolChoiceOptionUnionParam) anthropic.ToolChoiceUnionParam {
	if toolChoice.OfAuto.Valid() {
		switch toolChoice.OfAuto.Value {
		case "auto":
			return anthropic.ToolChoiceUnionParam{
				OfAuto: &anthropic.ToolChoiceAutoParam{Type: "auto"},
			}
		case "none":
			return anthropic.ToolChoiceUnionParam{
				OfNone: &anthropic.ToolChoiceNoneParam{Type: "none"},
			}
		case "required":
			return anthropic.ToolChoiceUnionParam{
				OfAny: &anthropic.ToolChoiceAnyParam{Type: "any"},
			}
		}
	}

	if toolChoice.OfChatCompletionNamedToolChoice != nil {
		return anthropic.ToolChoiceParamOfTool(toolChoice.OfChatCompletionNamedToolChoice.Function.Name)
	}

	// Default to auto
	return anthropic.ToolChoiceUnionParam{
		OfAuto: &anthropic.ToolChoiceAutoParam{Type: "auto"},
	}
}

// convertAnthropicToOpenAIResponse converts Anthropic response to OpenAI format
func (c *AnthropicCompletions) convertAnthropicToOpenAIResponse(resp *anthropic.Message) *openai.ChatCompletion {
	// Extract content from Anthropic response
	var content string
	var toolCalls []openai.ChatCompletionMessageToolCall

	for _, contentBlock := range resp.Content {
		switch block := contentBlock.AsAny().(type) {
		case anthropic.TextBlock:
			if content != "" {
				content += "\n"
			}
			content += block.Text
		case anthropic.ToolUseBlock:
			toolCall := openai.ChatCompletionMessageToolCall{
				ID:   block.ID,
				Type: "function",
				Function: openai.ChatCompletionMessageToolCallFunction{
					Name:      block.Name,
					Arguments: string(block.Input),
				},
			}
			toolCalls = append(toolCalls, toolCall)
		}
	}

	// Determine finish reason
	finishReason := c.mapAnthropicStopReason(resp.StopReason)

	// Create the message
	message := openai.ChatCompletionMessage{
		Role: "assistant",
	}

	if content != "" {
		message.Content = content
	}

	if len(toolCalls) > 0 {
		message.ToolCalls = toolCalls
	}

	return &openai.ChatCompletion{
		ID:     resp.ID,
		Object: "chat.completion",
		Model:  string(resp.Model),
		Choices: []openai.ChatCompletionChoice{
			{
				Index:        0,
				Message:      message,
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
