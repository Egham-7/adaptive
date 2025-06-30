package deepseek

import (
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// DeepSeekService handles DeepSeek API interactions
type DeepSeekService struct {
	client *openai.Client
}

// NewDeepSeekService creates a new DeepSeek service
func NewDeepSeekService() (*DeepSeekService, error) {
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("DEEPSEEK_API_KEY environment variable not set")
	}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL("https://api.deepseek.com"),
	)
	return &DeepSeekService{client: &client}, nil
}

// Chat returns the chat interface
func (s *DeepSeekService) Chat() provider_interfaces.Chat {
	return &DeepSeekChat{service: s}
}

func (s *DeepSeekService) GetProviderName() string {
	return "deepseek"
}
