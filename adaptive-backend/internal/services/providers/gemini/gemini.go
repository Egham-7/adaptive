package gemini

import (
	"adaptive-backend/internal/services/providers/gemini/chat"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// GeminiService handles Gemini API interactions using OpenAI Go SDK
type GeminiService struct {
	client *openai.Client
	chat   *chat.GeminiChat
}

// NewGeminiService creates a new Gemini service using OpenAI Go SDK with Google base URL
func NewGeminiService() (*GeminiService, error) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("GOOGLE_API_KEY environment variable not set")
	}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL("https://generativelanguage.googleapis.com/v1beta/openai/"),
	)

	chatService := chat.NewGeminiChat(&client)

	return &GeminiService{
		client: &client,
		chat:   chatService,
	}, nil
}

// Chat implements LLMProvider interface
func (s *GeminiService) Chat() provider_interfaces.Chat {
	return s.chat
}

func (s *GeminiService) GetProviderName() string {
	return "gemini"
}
