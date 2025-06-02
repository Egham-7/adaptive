package providers

import (
	"adaptive-backend/internal/models"
	"context"
	"fmt"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

// OpenAIService handles OpenAI API interactions
type OpenAIService struct {
	client *openai.Client
}

// NewOpenAIService creates a new OpenAI service using the official SDK
func NewOpenAIService() *OpenAIService {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("Warning: OPENAI_API_KEY environment variable not set")
	}
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)
	return &OpenAIService{client: &client}
}

// StreamChatCompletion streams a chat completion using OpenAI
func (s *OpenAIService) StreamChatCompletion(req *models.ProviderChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}
	messages, err := convertToOpenAIMessageParams(req.Messages)
	if err != nil {
		return nil, fmt.Errorf("convert messages: %w", err)
	}
	model := determineOpenAIModel(req.Model)
	params := openai.ChatCompletionNewParams{
		Model:            model,
		Messages:         messages,
		Temperature:      openai.Float(float64(req.Temperature)),
		TopP:             openai.Float(float64(req.TopP)),
		MaxTokens:        openai.Int(int64(req.MaxTokens)),
		PresencePenalty:  openai.Float(float64(req.PresencePenalty)),
		FrequencyPenalty: openai.Float(float64(req.FrequencyPenalty)),
		ResponseFormat:   *req.ResponseFormat,
	}

	stream := s.client.Chat.Completions.NewStreaming(context.Background(), params)

	return &models.ChatCompletionResponse{
		Provider: "openai",
		Response: stream, // You must handle stream.Close() where you consume it!
	}, nil
}

// CreateChatCompletion processes a chat completion request with OpenAI
func (s *OpenAIService) CreateChatCompletion(req *models.ProviderChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}
	messages, err := convertToOpenAIMessageParams(req.Messages)
	if err != nil {
		return nil, fmt.Errorf("convert messages: %w", err)
	}
	model := determineOpenAIModel(req.Model)
	params := openai.ChatCompletionNewParams{
		Model:            model,
		Messages:         messages,
		Temperature:      openai.Float(float64(req.Temperature)),
		TopP:             openai.Float(float64(req.TopP)),
		MaxTokens:        openai.Int(int64(req.MaxTokens)),
		PresencePenalty:  openai.Float(float64(req.PresencePenalty)),
		FrequencyPenalty: openai.Float(float64(req.FrequencyPenalty)),
		ResponseFormat:   *req.ResponseFormat,
	}
	resp, err := s.client.Chat.Completions.New(context.Background(), params)
	if err != nil {
		return &models.ChatCompletionResponse{
			Provider: "openai",
			Error:    err.Error(),
		}, fmt.Errorf("openai chat completion failed: %w", err)
	}
	return &models.ChatCompletionResponse{
		Provider: "openai",
		Response: resp,
	}, nil
}

// -------- Conversion helpers --------

// convertToOpenAIMessageParams converts your messages to the new SDK's union types.
// This is a simple implementation: supports text system/user/assistant messages.
// Expand as needed for tools/function calls/multimodal.
func convertToOpenAIMessageParams(msgs []models.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	out := make([]openai.ChatCompletionMessageParamUnion, len(msgs))
	for i, m := range msgs {
		switch m.Role {
		case "user":
			out[i] = openai.UserMessage(m.Content)
		case "assistant":
			out[i] = openai.AssistantMessage(m.Content)
		case "system":
			out[i] = openai.SystemMessage(m.Content)
		default:
			return nil, fmt.Errorf("unknown message role: %s", m.Role)
		}
		// If you use tool/function/multimodal, add handling here.
	}
	return out, nil
}

// Use OpenAI's official model enums; fallback to GPT-4o if unknown.
func determineOpenAIModel(requestedModel string) string {
	switch requestedModel {
	case shared.ChatModelGPT4o, shared.ChatModelGPT4_1, shared.ChatModelGPT4_1Mini, shared.ChatModelO3, shared.ChatModelO4Mini, shared.ChatModelGPT4_1Nano:
		return requestedModel
	default:
		return shared.ChatModelGPT4o // default fallback
	}
}
