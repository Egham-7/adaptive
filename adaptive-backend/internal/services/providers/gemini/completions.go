package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

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
	ctx context.Context,
	req *openai.ChatCompletionNewParams,
) (*openai.ChatCompletion, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Convert OpenAI messages to Gemini format
	contents := make([]*genai.Content, 0, len(req.Messages))
	for _, msg := range req.Messages {
		parts, role := convertMessageToGeminiParts(msg)
		if len(parts) > 0 {
			contents = append(contents, &genai.Content{
				Parts: parts,
				Role:  role,
			})
		}
	}

	opts, err := convertGeminiOptions(req)
	if err != nil {
		return nil, fmt.Errorf("failed to convert gemini options: %w", err)
	}

	resp, err := c.chat.service.client.Models.GenerateContent(
		ctx,
		string(req.Model),
		contents,
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("gemini chat completion failed: %w", err)
	}

	return convertFromGeminiResponse(resp), nil
}

// StreamCompletion calls Gemini's streaming API.
func (c *GeminiCompletions) StreamCompletion(
	ctx context.Context,
	req *openai.ChatCompletionNewParams,
) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Convert OpenAI messages to Gemini format
	contents := make([]*genai.Content, 0, len(req.Messages))
	for _, msg := range req.Messages {
		parts, role := convertMessageToGeminiParts(msg)
		if len(parts) > 0 {
			contents = append(contents, &genai.Content{
				Parts: parts,
				Role:  role,
			})
		}
	}

	opts, err := convertGeminiOptions(req)
	if err != nil {
		return nil, fmt.Errorf("failed to convert gemini options: %w", err)
	}

	stream := c.chat.service.client.Models.GenerateContentStream(
		ctx,
		string(req.Model),
		contents,
		opts,
	)

	// Create stream adapter and convert to OpenAI format
	adapter := NewGeminiStreamAdapter(stream)
	return adapter.ConvertToOpenAIStream()
}

// convertGeminiOptions maps your internal request into Gemini's GenerateContentConfig.
func convertGeminiOptions(
	req *openai.ChatCompletionNewParams,
) (*genai.GenerateContentConfig, error) {
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
			return nil, fmt.Errorf("failed to marshal JSON schema: %w", err)
		}

		var gs genai.Schema
		if err := json.Unmarshal(raw, &gs); err != nil {
			return nil, fmt.Errorf("failed to unmarshal JSON schema into genai.Schema: %w", err)
		}

		cfg.ResponseSchema = &gs
	}

	return cfg, nil
}

// convertMessageToGeminiParts converts OpenAI message to Gemini parts and role
func convertMessageToGeminiParts(msg openai.ChatCompletionMessageParamUnion) ([]*genai.Part, string) {
	switch {
	case msg.OfUser != nil:
		return convertUserContent(msg.OfUser.Content), "user"
	case msg.OfAssistant != nil:
		return convertAssistantContent(msg.OfAssistant), "model"
	case msg.OfSystem != nil:
		return convertSystemContent(msg.OfSystem.Content), "user"
	case msg.OfDeveloper != nil:
		return convertDeveloperContent(msg.OfDeveloper.Content), "user"
	case msg.OfTool != nil:
		return convertToolContent(msg.OfTool), "user"
	default:
		return nil, ""
	}
}

// convertUserContent handles OpenAI user message content with all media types
func convertUserContent(content openai.ChatCompletionUserMessageParamContentUnion) []*genai.Part {
	if !param.IsOmitted(content.OfString) {
		return []*genai.Part{genai.NewPartFromText(content.OfString.Value)}
	}

	if param.IsOmitted(content.OfArrayOfContentParts) {
		return []*genai.Part{genai.NewPartFromText("")}
	}

	var parts []*genai.Part
	for _, part := range content.OfArrayOfContentParts {
		switch {
		case part.OfText != nil:
			parts = append(parts, genai.NewPartFromText(part.OfText.Text))
		case part.OfImageURL != nil:
			parts = append(parts, genai.NewPartFromURI(part.OfImageURL.ImageURL.URL, "image/*"))
		case part.OfInputAudio != nil:
			parts = append(parts, genai.NewPartFromBytes(
				[]byte(part.OfInputAudio.InputAudio.Data),
				string(part.OfInputAudio.InputAudio.Format),
			))
		case part.OfFile != nil:
			parts = append(parts, convertFilePart(part.OfFile.File))
		}
	}
	return parts
}

// convertAssistantContent handles OpenAI assistant message with content, refusal, and tool calls
func convertAssistantContent(msg *openai.ChatCompletionAssistantMessageParam) []*genai.Part {
	var parts []*genai.Part

	// Add content parts
	if !param.IsOmitted(msg.Content.OfString) {
		parts = append(parts, genai.NewPartFromText(msg.Content.OfString.Value))
	} else if !param.IsOmitted(msg.Content.OfArrayOfContentParts) {
		for _, part := range msg.Content.OfArrayOfContentParts {
			if part.OfText != nil {
				parts = append(parts, genai.NewPartFromText(part.OfText.Text))
			} else if part.OfRefusal != nil {
				parts = append(parts, genai.NewPartFromText(fmt.Sprintf("[Refusal: %s]", part.OfRefusal.Refusal)))
			}
		}
	}

	// Add refusal if present
	if !param.IsOmitted(msg.Refusal) && msg.Refusal.Value != "" {
		parts = append(parts, genai.NewPartFromText(fmt.Sprintf("[Refusal: %s]", msg.Refusal.Value)))
	}

	// Add tool calls as function calls
	for _, toolCall := range msg.ToolCalls {
		args := make(map[string]any)
		if toolCall.Function.Arguments != "" {
			err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args)

			if err != nil {
				log.Printf("Failed to unmarshal tool call arguments for function %s: %v", toolCall.Function.Name, err)
				continue
			}
		}
		parts = append(parts, genai.NewPartFromFunctionCall(toolCall.Function.Name, args))
	}

	if len(parts) == 0 {
		parts = []*genai.Part{genai.NewPartFromText("")}
	}
	return parts
}

// convertSystemContent handles system message content
func convertSystemContent(content openai.ChatCompletionSystemMessageParamContentUnion) []*genai.Part {
	if !param.IsOmitted(content.OfString) {
		return []*genai.Part{genai.NewPartFromText(content.OfString.Value)}
	}

	if param.IsOmitted(content.OfArrayOfContentParts) {
		return []*genai.Part{genai.NewPartFromText("")}
	}

	var parts []*genai.Part
	for _, part := range content.OfArrayOfContentParts {
		parts = append(parts, genai.NewPartFromText(part.Text))
	}
	return parts
}

// convertDeveloperContent handles developer message content
func convertDeveloperContent(content openai.ChatCompletionDeveloperMessageParamContentUnion) []*genai.Part {
	if !param.IsOmitted(content.OfString) {
		return []*genai.Part{genai.NewPartFromText(content.OfString.Value)}
	}

	if param.IsOmitted(content.OfArrayOfContentParts) {
		return []*genai.Part{genai.NewPartFromText("")}
	}

	var parts []*genai.Part
	for _, part := range content.OfArrayOfContentParts {
		parts = append(parts, genai.NewPartFromText(part.Text))
	}
	return parts
}

// convertToolContent handles OpenAI tool message as function response
func convertToolContent(msg *openai.ChatCompletionToolMessageParam) []*genai.Part {
	var content string
	if !param.IsOmitted(msg.Content.OfString) {
		content = msg.Content.OfString.Value
	} else if !param.IsOmitted(msg.Content.OfArrayOfContentParts) {
		var textParts []string
		for _, part := range msg.Content.OfArrayOfContentParts {
			textParts = append(textParts, part.Text)
		}
		content = strings.Join(textParts, "\n")
	}

	response := map[string]any{"result": content}
	return []*genai.Part{genai.NewPartFromFunctionResponse(msg.ToolCallID, response)}
}

// convertFilePart handles OpenAI file content
func convertFilePart(file openai.ChatCompletionContentPartFileFileParam) *genai.Part {
	if !param.IsOmitted(file.FileData) {
		return genai.NewPartFromBytes([]byte(file.FileData.Value), "application/octet-stream")
	}

	fileName := "unknown"
	if !param.IsOmitted(file.Filename) {
		fileName = file.Filename.Value
	}
	fileID := ""
	if !param.IsOmitted(file.FileID) {
		fileID = file.FileID.Value
	}
	return genai.NewPartFromText(fmt.Sprintf("[File: %s (ID: %s)]", fileName, fileID))
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
