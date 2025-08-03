package anthropic

import (
	"adaptive-backend/internal/services/providers/anthropic/chat"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// AnthropicService handles Anthropic API interactions using OpenAI Go SDK
type AnthropicService struct {
	client *openai.Client
	chat   *chat.AnthropicChat
}

// NewAnthropicService creates a new Anthropic service using OpenAI Go SDK with Anthropic base URL
func NewAnthropicService() (*AnthropicService, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY environment variable not set")
	}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL("https://api.anthropic.com/v1/"),
	)

	chatService := chat.NewAnthropicChat(&client)

	return &AnthropicService{
		client: &client,
		chat:   chatService,
	}, nil
}

// Chat implements LLMProvider interface
func (s *AnthropicService) Chat() provider_interfaces.Chat {
	return s.chat
}

func (s *AnthropicService) GetProviderName() string {
	return "anthropic"
}
