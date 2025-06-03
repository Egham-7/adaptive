package gemini

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/packages/ssestream"
	"google.golang.org/genai"
)

// GeminiCompletions implements the Completions interface
type GeminiCompletions struct {
	chat *GeminiChat
}

// CreateCompletion calls Gemini's non-streaming API.
func (c *GeminiCompletions) CreateCompletion(
	req *openai.ChatCompletionNewParams,
) (*openai.ChatCompletion, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}
	ctx := context.Background()

	// Convert OpenAI messages to Gemini format
	parts := make([]*genai.Part, len(req.Messages))
	for i, msg := range req.Messages {
		parts[i] = &genai.Part{Text: extractMessageContent(msg)}
	}
	content := &genai.Content{Parts: parts}

	opts := convertGeminiOptions(req)

	resp, err := c.chat.service.client.Models.GenerateContent(
		ctx,
		string(req.Model),
		[]*genai.Content{content},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("gemini chat completion failed: %w", err)
	}

	return convertFromGeminiResponse(resp), nil
}

// StreamCompletion calls Gemini's streaming API.
func (c *GeminiCompletions) StreamCompletion(
	req *openai.ChatCompletionNewParams,
) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}
	ctx := context.Background()

	// Convert OpenAI messages to Gemini format
	parts := make([]*genai.Part, len(req.Messages))
	for i, msg := range req.Messages {
		parts[i] = &genai.Part{Text: extractMessageContent(msg)}
	}
	content := &genai.Content{Parts: parts}

	opts := convertGeminiOptions(req)

	stream := c.chat.service.client.Models.GenerateContentStream(
		ctx,
		string(req.Model),
		[]*genai.Content{content},
		opts,
	)

	// Create stream adapter and convert to OpenAI format
	adapter := NewGeminiStreamAdapter(stream)
	return adapter.ConvertToOpenAIStream()
}

// extractMessageContent extracts string content from OpenAI message union
func extractMessageContent(msg openai.ChatCompletionMessageParamUnion) string {
	// Handle different message types using the union pattern
	if msg.OfUser != nil {
		return extractUserContent(msg.OfUser.Content)
	}
	if msg.OfAssistant != nil {
		return extractAssistantContent(msg.OfAssistant.Content)
	}
	if msg.OfSystem != nil {
		return extractSystemContent(msg.OfSystem.Content)
	}
	if msg.OfDeveloper != nil {
		return extractDeveloperContent(msg.OfDeveloper.Content)
	}
	return ""
}

// extractUserContent extracts string content from user message content union
func extractUserContent(content openai.ChatCompletionUserMessageParamContentUnion) string {
	if !param.IsOmitted(content.OfString) {
		return content.OfString.Value
	}
	// For array content, concatenate text parts
	if !param.IsOmitted(content.OfArrayOfContentParts) {
		var result string
		for _, part := range content.OfArrayOfContentParts {
			if part.OfText != nil {
				result += part.OfText.Text
			}
		}
		return result
	}
	return ""
}

// extractAssistantContent extracts string content from assistant message content union
func extractAssistantContent(content openai.ChatCompletionAssistantMessageParamContentUnion) string {
	if !param.IsOmitted(content.OfString) {
		return content.OfString.Value
	}
	// For array content, concatenate text parts
	if !param.IsOmitted(content.OfArrayOfContentParts) {
		var result string
		for _, part := range content.OfArrayOfContentParts {
			if part.OfText != nil {
				result += part.OfText.Text
			}
		}
		return result
	}
	return ""
}

// extractSystemContent extracts string content from system message content union
func extractSystemContent(content openai.ChatCompletionSystemMessageParamContentUnion) string {
	if !param.IsOmitted(content.OfString) {
		return content.OfString.Value
	}
	// For array content, concatenate text parts
	if !param.IsOmitted(content.OfArrayOfContentParts) {
		var result string
		for _, part := range content.OfArrayOfContentParts {
			result += part.Text
		}
		return result
	}
	return ""
}

// extractDeveloperContent extracts string content from developer message content union
func extractDeveloperContent(content openai.ChatCompletionDeveloperMessageParamContentUnion) string {
	if !param.IsOmitted(content.OfString) {
		return content.OfString.Value
	}
	// For array content, concatenate text parts
	if !param.IsOmitted(content.OfArrayOfContentParts) {
		var result string
		for _, part := range content.OfArrayOfContentParts {
			result += part.Text
		}
		return result
	}
	return ""
}

// convertGeminiOptions maps your internal request into Gemini's GenerateContentConfig.
func convertGeminiOptions(
	req *openai.ChatCompletionNewParams,
) *genai.GenerateContentConfig {
	cfg := &genai.GenerateContentConfig{}

	// Set temperature and topP if provided
	if !param.IsOmitted(req.Temperature) {
		temp := float32(req.Temperature.Value)
		cfg.Temperature = &temp
	}
	if !param.IsOmitted(req.TopP) {
		topP := float32(req.TopP.Value)
		cfg.TopP = &topP
	}

	// Map OpenAI-style union for response_format → Gemini fields
	u := req.ResponseFormat
	if u.OfText != nil && !param.IsOmitted(u.OfText) {
		cfg.ResponseMIMEType = "text/plain"
	} else if u.OfJSONObject != nil && !param.IsOmitted(u.OfJSONObject) {
		cfg.ResponseMIMEType = "application/json"
	} else if u.OfJSONSchema != nil && !param.IsOmitted(u.OfJSONSchema) {
		// Set JSON output
		cfg.ResponseMIMEType = "application/json"

		// Now convert the OpenAI JSON-Schema param into a genai.Schema
		// by marshaling then unmarshaling it:
		schemaParam := u.OfJSONSchema.JSONSchema
		raw, err := json.Marshal(schemaParam)
		if err != nil {
			fmt.Printf("Error marshaling JSON schema: %v\n", err)
		} else {
			var gs genai.Schema
			if err2 := json.Unmarshal(raw, &gs); err2 != nil {
				fmt.Printf("Error unmarshaling JSON schema into genai.Schema: %v\n", err2)
			} else {
				cfg.ResponseSchema = &gs
			}
		}
	}

	return cfg
}

// convertFromGeminiResponse converts Gemini response to OpenAI format
func convertFromGeminiResponse(resp *genai.GenerateContentResponse) *openai.ChatCompletion {
	var content string
	finishReason := "stop"

	if len(resp.Candidates) > 0 {
		candidate := resp.Candidates[0]
		if len(candidate.Content.Parts) > 0 {
			content = candidate.Content.Parts[0].Text
		}

		// Map finish reason
		switch candidate.FinishReason {
		case genai.FinishReasonStop:
			finishReason = "stop"
		case genai.FinishReasonMaxTokens:
			finishReason = "length"
		case genai.FinishReasonSafety:
			finishReason = "content_filter"
		default:
			finishReason = "stop"
		}
	}

	// Extract usage information if available
	var usage openai.CompletionUsage
	if resp.UsageMetadata != nil {
		usage = openai.CompletionUsage{
			CompletionTokens: int64(resp.UsageMetadata.CandidatesTokenCount),
			PromptTokens:     int64(resp.UsageMetadata.PromptTokenCount),
			TotalTokens:      int64(resp.UsageMetadata.TotalTokenCount),
		}
	}

	return &openai.ChatCompletion{
		ID:     fmt.Sprintf("chatcmpl-gemini-%d", resp.UsageMetadata.TotalTokenCount),
		Object: "chat.completion",
		Model:  "gemini-pro",
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
		Usage: usage,
	}
}
