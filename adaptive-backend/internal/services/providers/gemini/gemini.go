package gemini

import (
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"context"
	"fmt"
	"os"

	"google.golang.org/genai"
)

// GeminiService handles Gemini API interactions.
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

// Chat returns the chat interface
func (s *GeminiService) Chat() provider_interfaces.Chat {
	return &GeminiChat{service: s}
}
