package providers

import (
	"adaptive-backend/internal/models"
	"context"
	"fmt"
	"os"

	"github.com/cohesion-org/deepseek-go"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
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

// StreamChatCompletion processes a streaming chat completion request with DeepSeek

func (s *DeepSeekService) StreamChatCompletion(req *models.ProviderChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Convert our messages to DeepSeek format
	messages := make([]deepseek.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = convertToDeepSeekMessage(msg)
	}

	// Determine which model to use
	model := determineDeepSeekModel(req.Model)

	// Create DeepSeek request
	deepseekReq := &deepseek.StreamChatCompletionRequest{
		Model:            model,
		Messages:         messages,
		Temperature:      req.Temperature,
		PresencePenalty:  req.PresencePenalty,
		FrequencyPenalty: req.FrequencyPenalty,
		MaxTokens:        req.MaxTokens,
		TopP:             req.TopP,
		Stream:           req.Stream,
		ResponseFormat:   convertDeepSeekResponseFormat(req.ResponseFormat),
	}

	stream, err := s.client.CreateChatCompletionStream(context.Background(), deepseekReq)
	if err != nil {
		return &models.ChatCompletionResponse{
			Provider: "deepseek",
			Error:    err.Error(),
		}, fmt.Errorf("deepseek stream chat completion failed: %w", err)
	}

	return &models.ChatCompletionResponse{
		Provider: "deepseek",
		Response: stream,
	}, nil
}

// CreateChatCompletion processes a chat completion request with DeepSeek
func (s *DeepSeekService) CreateChatCompletion(req *models.ProviderChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Convert our messages to DeepSeek format
	messages := make([]deepseek.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = convertToDeepSeekMessage(msg)
	}

	// Determine which model to use
	model := determineDeepSeekModel(req.Model)

	// Create DeepSeek request
	deepseekReq := &deepseek.ChatCompletionRequest{
		Model:            model,
		Messages:         messages,
		Temperature:      req.Temperature,
		PresencePenalty:  req.PresencePenalty,
		FrequencyPenalty: req.FrequencyPenalty,
		MaxTokens:        req.MaxTokens,
		TopP:             req.TopP,
		ResponseFormat:   convertDeepSeekResponseFormat(req.ResponseFormat),
	}

	// Call DeepSeek API
	resp, err := s.client.CreateChatCompletion(context.Background(), deepseekReq)
	if err != nil {
		return &models.ChatCompletionResponse{
			Provider: "deepseek",
			Error:    err.Error(),
		}, fmt.Errorf("deepseek chat completion failed: %w", err)
	}

	// Return the response
	return &models.ChatCompletionResponse{
		Provider: "deepseek",
		Response: resp,
	}, nil
}

// convertToDeepSeekMessage converts our message model to DeepSeek's format
func convertToDeepSeekMessage(msg models.Message) deepseek.ChatCompletionMessage {
	// Convert the role
	role := convertDeepSeekRole(msg.Role)

	// Create the message
	deepseekMsg := deepseek.ChatCompletionMessage{
		Role:    role,
		Content: msg.Content,
	}

	// Add tool call ID if present
	if msg.ToolCallID != "" {
		deepseekMsg.ToolCallID = msg.ToolCallID
	}

	return deepseekMsg
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

// convertDeepSeekRole maps string roles to DeepSeek role types
func convertDeepSeekRole(role string) string {
	switch role {
	case "user":
		return deepseek.ChatMessageRoleUser
	case "assistant":
		return deepseek.ChatMessageRoleAssistant
	case "system":
		return deepseek.ChatMessageRoleSystem
	default:
		return deepseek.ChatMessageRoleUser // Default to user if unknown
	}
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
