package openai

import (
	"adaptive-backend/internal/services/providers/openai/chat"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// OpenAIService handles OpenAI API interactions
type OpenAIService struct {
	client *openai.Client
	chat   *chat.OpenAIChat
}

// NewOpenAIService creates a new OpenAI service using the official SDK
func NewOpenAIService() (*OpenAIService, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY environment variable not set")
	}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	chatService := chat.NewOpenAIChat(&client)

	return &OpenAIService{
		client: &client,
		chat:   chatService,
	}, nil
}

func (s *OpenAIService) Chat() provider_interfaces.Chat {
	return s.chat
}

func (s *OpenAIService) GetProviderName() string {
	return "openai"
}
