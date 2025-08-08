package anthropic

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers/anthropic/chat"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"net/http"
	"os"
	"time"

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

// NewCustomAnthropicService creates a new Anthropic service with custom configuration
func NewCustomAnthropicService(customConfig *models.ProviderConfig) (*AnthropicService, error) {
	if customConfig == nil {
		return nil, fmt.Errorf("custom config is required")
	}

	// Build client options
	opts := []option.RequestOption{}

	// Use custom base URL or default
	baseURL := "https://api.anthropic.com/v1/"
	if customConfig.BaseURL != nil {
		baseURL = *customConfig.BaseURL
	}
	opts = append(opts, option.WithBaseURL(baseURL))

	// Configure API key
	if customConfig.APIKey != nil {
		opts = append(opts, option.WithAPIKey(*customConfig.APIKey))
	} else {
		// Fall back to environment variable
		apiKey := os.Getenv("ANTHROPIC_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("ANTHROPIC_API_KEY environment variable not set and no API key in config")
		}
		opts = append(opts, option.WithAPIKey(apiKey))
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

	client := openai.NewClient(opts...)
	chatService := chat.NewAnthropicChat(&client)

	return &AnthropicService{
		client: &client,
		chat:   chatService,
	}, nil
}
