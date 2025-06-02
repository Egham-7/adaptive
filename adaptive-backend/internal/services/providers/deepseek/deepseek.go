package deepseek

import (
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"os"

	"github.com/cohesion-org/deepseek-go"
)

// DeepSeekService handles DeepSeek API interactions
type DeepSeekService struct {
	client *deepseek.Client
}

// NewDeepSeekService creates a new DeepSeek service
func NewDeepSeekService() (*DeepSeekService, error) {
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("DEEPSEEK_API_KEY environment variable not set")
	}

	client := deepseek.NewClient(apiKey)
	return &DeepSeekService{client: client}, nil
}

// Chat returns the chat interface
func (s *DeepSeekService) Chat() provider_interfaces.Chat {
	return &DeepSeekChat{service: s}
}
