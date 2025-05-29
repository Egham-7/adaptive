package providers

import (
	"adaptive-backend/internal/models"
	"context"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// AnthropicService handles Anthropic API interactions
type AnthropicService struct {
	client *anthropic.Client
}

// NewAnthropicService creates a new Anthropic service
func NewAnthropicService() (*AnthropicService, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY environment variable not set")
	}

	client := anthropic.NewClient(
		option.WithAPIKey(apiKey),
	)

	return &AnthropicService{client: &client}, nil
}

// StreamChatCompletion processes a streaming chat completion request with Anthropic
func (s *AnthropicService) StreamChatCompletion(req *models.ProviderChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Convert our messages to Anthropic format
	messages := make([]anthropic.MessageParam, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = convertToAnthropicMessage(msg)
	}

	// Determine which model to use
	model := determineAnthropicModel(req.Model)

	// Create Anthropic request
	anthropicReq := anthropic.MessageNewParams{
		Model:     model,
		Messages:  messages,
		MaxTokens: int64(req.MaxTokens),
	}

	// Set optional parameters if provided
	if req.Temperature > 0 {
		anthropicReq.Temperature = anthropic.Float(float64(req.Temperature))
	}

	if req.TopP > 0 {
		anthropicReq.TopP = anthropic.Float(float64(req.TopP))
	}

	// Create streaming request
	stream := s.client.Messages.NewStreaming(context.Background(), anthropicReq)

	return &models.ChatCompletionResponse{
		Provider: "anthropic",
		Response: stream,
	}, nil
}

// CreateChatCompletion processes a chat completion request with Anthropic
func (s *AnthropicService) CreateChatCompletion(req *models.ProviderChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Convert our messages to Anthropic format
	messages := make([]anthropic.MessageParam, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = convertToAnthropicMessage(msg)
	}

	// Determine which model to use
	model := determineAnthropicModel(req.Model)

	// Create Anthropic request
	anthropicReq := anthropic.MessageNewParams{
		Model:     model,
		Messages:  messages,
		MaxTokens: int64(req.MaxTokens),
	}

	// Set optional parameters if provided
	if req.Temperature > 0 {
		anthropicReq.Temperature = anthropic.Float(float64(req.Temperature))
	}

	if req.TopP > 0 {
		anthropicReq.TopP = anthropic.Float(float64(req.TopP))
	}

	// Call Anthropic API
	resp, err := s.client.Messages.New(context.Background(), anthropicReq)
	if err != nil {
		return &models.ChatCompletionResponse{
			Provider: "anthropic",
			Error:    err.Error(),
		}, fmt.Errorf("anthropic chat completion failed: %w", err)
	}

	// Return the response
	return &models.ChatCompletionResponse{
		Provider: "anthropic",
		Response: resp,
	}, nil
}

// convertToAnthropicMessage converts our message model to Anthropic's format
func convertToAnthropicMessage(msg models.Message) anthropic.MessageParam {
	// Convert the role
	role := convertAnthropicRole(msg.Role)

	// Create the content block
	contentBlock := anthropic.NewTextBlock(msg.Content)

	// Create the message
	return anthropic.MessageParam{
		Role:    role,
		Content: []anthropic.ContentBlockParamUnion{contentBlock},
	}
}

// convertAnthropicRole maps string roles to Anthropic role types
func convertAnthropicRole(role string) anthropic.MessageParamRole {
	switch role {
	case "user":
		return anthropic.MessageParamRoleUser
	case "assistant":
		return anthropic.MessageParamRoleAssistant
	case "system":
		// Anthropic handles system messages differently, but we'll map it here
		return "system"
	default:
		return anthropic.MessageParamRoleUser // Default to user if unknown
	}
}

// determineAnthropicModel selects the appropriate model based on request
func determineAnthropicModel(requestedModel string) anthropic.Model {
	if requestedModel == "" {
		return anthropic.ModelClaude3_7SonnetLatest // Default model
	}

	// Map of supported models
	supportedModels := map[anthropic.Model]bool{
		anthropic.ModelClaude3_7SonnetLatest: true,
		anthropic.ModelClaude3OpusLatest:     true,
		anthropic.ModelClaude3_5HaikuLatest:  true,
		anthropic.ModelClaude3_5SonnetLatest: true,
	}

	rm := anthropic.Model(requestedModel)
	if _, ok := supportedModels[rm]; ok {
		return rm
	}

	// For unknown models, fall back to default
	return anthropic.ModelClaude3_7SonnetLatest
}
