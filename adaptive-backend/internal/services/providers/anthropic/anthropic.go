package anthropic

import (
	"adaptive-backend/internal/services/providers/provider_interfaces"
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

// Chat implements LLMProvider interface
func (s *AnthropicService) Chat() provider_interfaces.Chat {
	return &AnthropicChat{
		completions: &AnthropicCompletions{client: s.client},
	}
}

func (s *AnthropicService) GetProviderName() string {
	return "anthropic"
}
