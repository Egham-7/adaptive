package providers

import (
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"context"
	"fmt"
	"os"

	"github.com/cohesion-org/deepseek-go"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/packages/ssestream"
)

// DeepSeekService handles DeepSeek API interactions
type DeepSeekService struct {
	client *deepseek.Client
}

// NewDeepSeekService creates a new DeepSeek service
func NewDeepSeekService() (*DeepSeekService, error) {
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("DEEPSEEK_API_KEY environment variable not set")
	}

	client := deepseek.NewClient(apiKey)
	return &DeepSeekService{client: client}, nil
}

// Chat returns the chat interface
func (s *DeepSeekService) Chat() provider_interfaces.Chat {
	return &DeepSeekChat{service: s}
}

// DeepSeekChat implements the Chat interface
type DeepSeekChat struct {
	service *DeepSeekService
}

// Completions returns the completions interface
func (c *DeepSeekChat) Completions() provider_interfaces.Completions {
	return &DeepSeekCompletions{chat: c}
}

// DeepSeekCompletions implements the Completions interface
type DeepSeekCompletions struct {
	chat *DeepSeekChat
}

// CreateCompletion processes a chat completion request with DeepSeek
func (c *DeepSeekCompletions) CreateCompletion(req *openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Convert OpenAI messages to DeepSeek format
	messages := make([]deepseek.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = convertToDeepSeekMessage(msg)
	}

	// Determine which model to use
	model := determineDeepSeekModel(string(req.Model))

	// Create DeepSeek request
	deepseekReq := &deepseek.ChatCompletionRequest{
		Model:            model,
		Messages:         messages,
		Temperature:      float32(req.Temperature.Value),
		PresencePenalty:  float32(req.PresencePenalty.Value),
		FrequencyPenalty: float32(req.FrequencyPenalty.Value),
		MaxTokens:        int(req.MaxTokens.Value),
		TopP:             float32(req.TopP.Value),
		ResponseFormat:   convertDeepSeekResponseFormat(&req.ResponseFormat),
	}

	// Call DeepSeek API
	resp, err := c.chat.service.client.CreateChatCompletion(context.Background(), deepseekReq)
	if err != nil {
		return nil, fmt.Errorf("deepseek chat completion failed: %w", err)
	}

	// Convert DeepSeek response to OpenAI format
	return convertFromDeepSeekResponse(resp), nil
}

// StreamCompletion processes a streaming chat completion request with DeepSeek
func (c *DeepSeekCompletions) StreamCompletion(req *openai.ChatCompletionNewParams) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Convert OpenAI messages to DeepSeek format
	messages := make([]deepseek.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = convertToDeepSeekMessage(msg)
	}

	// Determine which model to use
	model := determineDeepSeekModel(string(req.Model))

	// Create DeepSeek streaming request
	deepseekReq := &deepseek.StreamChatCompletionRequest{
		Model:            model,
		Messages:         messages,
		Stream:           true,
		Temperature:      float32(req.Temperature.Value),
		PresencePenalty:  float32(req.PresencePenalty.Value),
		FrequencyPenalty: float32(req.FrequencyPenalty.Value),
		MaxTokens:        int(req.MaxTokens.Value),
		TopP:             float32(req.TopP.Value),
		ResponseFormat:   convertDeepSeekResponseFormat(&req.ResponseFormat),
	}

	// Call DeepSeek streaming API
	stream, err := c.chat.service.client.CreateChatCompletionStream(context.Background(), deepseekReq)
	if err != nil {
		return nil, fmt.Errorf("deepseek stream chat completion failed: %w", err)
	}
	// Create stream adapter and convert to OpenAI format
	adapter := NewDeepSeekStreamAdapter(stream)
	return adapter.ConvertToOpenAIStream()
}

// convertToDeepSeekMessage converts OpenAI message to DeepSeek format
func convertToDeepSeekMessage(msg openai.ChatCompletionMessageParamUnion) deepseek.ChatCompletionMessage {
	// Handle different message types using the union pattern
	if msg.OfUser != nil {
		content := extractUserContent(msg.OfUser.Content)
		return deepseek.ChatCompletionMessage{
			Role:    deepseek.ChatMessageRoleUser,
			Content: content,
		}
	}
	if msg.OfAssistant != nil {
		content := extractAssistantContent(msg.OfAssistant.Content)
		return deepseek.ChatCompletionMessage{
			Role:    deepseek.ChatMessageRoleAssistant,
			Content: content,
		}
	}
	if msg.OfSystem != nil {
		content := extractSystemContent(msg.OfSystem.Content)
		return deepseek.ChatCompletionMessage{
			Role:    deepseek.ChatMessageRoleSystem,
			Content: content,
		}
	}
	if msg.OfDeveloper != nil {
		content := extractDeveloperContent(msg.OfDeveloper.Content)
		return deepseek.ChatCompletionMessage{
			Role:    deepseek.ChatMessageRoleSystem, // Map developer to system
			Content: content,
		}
	}

	// Default fallback
	return deepseek.ChatCompletionMessage{
		Role:    deepseek.ChatMessageRoleUser,
		Content: "",
	}
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

// convertFromDeepSeekResponse converts DeepSeek response to OpenAI format
func convertFromDeepSeekResponse(resp *deepseek.ChatCompletionResponse) *openai.ChatCompletion {
	choices := make([]openai.ChatCompletionChoice, len(resp.Choices))
	for i, choice := range resp.Choices {
		choices[i] = openai.ChatCompletionChoice{
			Index: int64(choice.Index),
			Message: openai.ChatCompletionMessage{
				Role:    "assistant", // DeepSeek responses are always from assistant
				Content: choice.Message.Content,
			},
			FinishReason: choice.FinishReason,
		}
	}

	return &openai.ChatCompletion{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: resp.Created,
		Model:   resp.Model,
		Choices: choices,
		Usage: openai.CompletionUsage{
			CompletionTokens: int64(resp.Usage.CompletionTokens),
			PromptTokens:     int64(resp.Usage.PromptTokens),
			TotalTokens:      int64(resp.Usage.TotalTokens),
		},
	}
}

func convertDeepSeekResponseFormat(
	u *openai.ChatCompletionNewParamsResponseFormatUnion,
) *deepseek.ResponseFormat {
	if u == nil {
		return nil
	}
	// text?
	if u.OfText != nil && !param.IsOmitted(u.OfText) {
		return &deepseek.ResponseFormat{Type: "text"}
	}
	// json_object?
	if u.OfJSONObject != nil && !param.IsOmitted(u.OfJSONObject) {
		return &deepseek.ResponseFormat{Type: "json_object"}
	}
	// fallback to text if nothing matched
	return &deepseek.ResponseFormat{Type: "text"}
}

// determineDeepSeekModel selects the appropriate model based on request
func determineDeepSeekModel(requestedModel string) string {
	if requestedModel == "" {
		return deepseek.DeepSeekChat // Default model
	}

	// Map of supported models
	supportedModels := map[string]bool{
		deepseek.DeepSeekChat:     true,
		deepseek.DeepSeekReasoner: true,
	}

	// If the requested model is supported, use it
	if _, ok := supportedModels[requestedModel]; ok {
		return requestedModel
	}

	// For unknown models, fall back to default
	return deepseek.DeepSeekChat
}
