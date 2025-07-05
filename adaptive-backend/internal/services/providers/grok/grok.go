package grok

import (
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// GrokService handles Grok API interactions
type GrokService struct {
	client *openai.Client
}

// NewGrokService creates a new Grok service
func NewGrokService() (*GrokService, error) {
	apiKey := os.Getenv("GROK_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("GROK_API_KEY environment variable not set")
	}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL("https://api.x.ai/v1"),
	)
	return &GrokService{client: &client}, nil
}

// Chat returns the chat interface
func (s *GrokService) Chat() provider_interfaces.Chat {
	return &GrokChat{service: s}
}

func (s *GrokService) GetProviderName() string {
	return "grok"
}
