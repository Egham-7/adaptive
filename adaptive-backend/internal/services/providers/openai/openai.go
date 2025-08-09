package openai

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers/openai/chat"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"net/http"
	"os"
	"time"

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

// NewCustomOpenAIService creates a custom OpenAI-compatible service with base URL override
func NewCustomOpenAIService(baseURL string, customConfig *models.ProviderConfig) (*OpenAIService, error) {
	if baseURL == "" {
		return nil, fmt.Errorf("base URL is required for custom provider")
	}

	// Build client options
	opts := []option.RequestOption{
		option.WithBaseURL(baseURL),
	}

	// Configure client options from custom config
	if customConfig != nil {
		// Configure API key if specified
		if customConfig.APIKey != nil {
			opts = append(opts, option.WithAPIKey(*customConfig.APIKey))
		}

		// Configure timeout if specified
		if customConfig.TimeoutMs != nil {
			timeout := time.Duration(*customConfig.TimeoutMs) * time.Millisecond
			opts = append(opts, option.WithHTTPClient(&http.Client{Timeout: timeout}))
		}

		// Add custom headers if specified
		if customConfig.Headers != nil {
			for key, value := range customConfig.Headers {
				opts = append(opts, option.WithHeader(key, value))
			}
		}
	}

	// Create client with custom configuration
	client := openai.NewClient(opts...)

	chatService := chat.NewOpenAIChat(&client)

	return &OpenAIService{
		client: &client,
		chat:   chatService,
	}, nil
}
