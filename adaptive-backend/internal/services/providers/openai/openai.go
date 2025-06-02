package openai

import (
	"fmt"
	"os"

	"adaptive-backend/internal/services/providers/openai/chat"
	"adaptive-backend/internal/services/providers/provider_interfaces"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// OpenAIService handles OpenAI API interactions
type OpenAIService struct {
	client *openai.Client
	chat   *chat.OpenAIChat
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

	chatService := chat.NewOpenAIChat(&client)

	return &OpenAIService{
		client: &client,
		chat:   chatService,
	}
}

func (s *OpenAIService) Chat() provider_interfaces.Chat {
	return s.chat
}
