package adaptive

import (
	"adaptive-backend/internal/services/providers/adaptive/chat"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// AdaptiveService handles Adaptive API interactions using OpenAI-compatible SDK
type AdaptiveService struct {
	client *openai.Client
	chat   *chat.AdaptiveChat
}

// NewAdaptiveService creates a new Adaptive service using the OpenAI SDK
func NewAdaptiveService() (*AdaptiveService, error) {
	apiKey := os.Getenv("ADAPTIVE_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("ADAPTIVE_API_KEY environment variable not set")
	}

	baseURL := os.Getenv("ADAPTIVE_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.adaptive.ai/v1" // Default Adaptive API base URL
	}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	)

	chatService := chat.NewAdaptiveChat(&client)

	return &AdaptiveService{
		client: &client,
		chat:   chatService,
	}, nil
}

func (s *AdaptiveService) Chat() provider_interfaces.Chat {
	return s.chat
}

func (s *AdaptiveService) GetProviderName() string {
	return "adaptive"
}
