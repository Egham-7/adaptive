package providers

import (
	"adaptive-backend/internal/models"
	"context"
	"fmt"
	"os"

	"google.golang.org/genai"
)

type GeminiService struct {
	client *genai.Client
}

// NewGeminiService initializes the GeminiService with the provided API key.
func NewGeminiService() (*GeminiService, error) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("GOOGLE_API_KEY environment variable not set")
	}

	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create genai client: %w", err)
	}

	return &GeminiService{client: client}, nil
}

// CreateChatCompletion sends a chat completion request to the Gemini API.
func (s *GeminiService) CreateChatCompletion(req *models.ProviderChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	ctx := context.Background()
	parts := make([]*genai.Part, len(req.Messages))
	for i, msg := range req.Messages {
		parts[i] = &genai.Part{Text: msg.Content}
	}

	content := &genai.Content{Parts: parts}
	resp, err := s.client.Models.GenerateContent(ctx, req.Model, []*genai.Content{content}, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to generate content: %w", err)
	}

	text := resp.Text()

	return &models.ChatCompletionResponse{
		Provider: "gemini",
		Response: text,
	}, nil
}

// StreamChatCompletion streams a chat completion response from the Gemini API.
func (s *GeminiService) StreamChatCompletion(req *models.ProviderChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	ctx := context.Background()
	parts := make([]*genai.Part, len(req.Messages))
	for i, msg := range req.Messages {
		parts[i] = &genai.Part{Text: msg.Content}
	}

	content := &genai.Content{Parts: parts}
	stream := s.client.Models.GenerateContentStream(ctx, req.Model, []*genai.Content{content}, nil)

	return &models.ChatCompletionResponse{
		Provider: "gemini",
		Response: stream,
	}, nil
}
