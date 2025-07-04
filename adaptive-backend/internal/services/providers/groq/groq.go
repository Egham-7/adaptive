package groq

import (
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// GroqService handles Groq API interactions
type GroqService struct {
	client *openai.Client
}

// NewGroqService creates a new Groq service
func NewGroqService() (*GroqService, error) {
	apiKey := os.Getenv("GROQ_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("GROQ_API_KEY environment variable not set")
	}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL("https://api.groq.com/openai/v1"),
	)
	return &GroqService{client: &client}, nil
}

// Chat returns the chat interface
func (s *GroqService) Chat() provider_interfaces.Chat {
	return &GroqChat{service: s}
}

func (s *GroqService) GetProviderName() string {
	return "groq"
}
