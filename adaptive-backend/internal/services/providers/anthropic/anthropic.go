package anthropic

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers/anthropic/chat"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"net/http"
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
func NewAnthropicService(cfg *config.Config) (*AnthropicService, error) {
	apiKey := cfg.GetProviderAPIKey("anthropic")
	if apiKey == "" {
		return nil, fmt.Errorf("anthropic API key not set in configuration")
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
	if customConfig.BaseURL != "" {
		baseURL = customConfig.BaseURL
	}
	opts = append(opts, option.WithBaseURL(baseURL))

	// Configure API key
	if customConfig.APIKey != "" {
		opts = append(opts, option.WithAPIKey(customConfig.APIKey))
	} else {
		// Fall back to configuration
		cfgInstance, err := config.New()
		if err != nil {
			return nil, fmt.Errorf("failed to load configuration: %w", err)
		}
		apiKey := cfgInstance.GetProviderAPIKey("anthropic")
		if apiKey == "" {
			return nil, fmt.Errorf("anthropic API key not set in configuration and no API key in custom config")
		}
		opts = append(opts, option.WithAPIKey(apiKey))
	}

	// Configure timeout if specified
	if customConfig.TimeoutMs > 0 {
		timeout := time.Duration(customConfig.TimeoutMs) * time.Millisecond
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

// NewAnthropicServiceWithConfig creates a new Anthropic service with custom configuration
func NewAnthropicServiceWithConfig(providerConfig models.ProviderConfig) (*AnthropicService, error) {
	if providerConfig.APIKey == "" {
		return nil, fmt.Errorf("anthropic API key not set in configuration")
	}

	// Build client options
	opts := []option.RequestOption{
		option.WithAPIKey(providerConfig.APIKey),
		option.WithBaseURL("https://api.anthropic.com/v1/"),
	}

	// Set custom base URL if provided
	if providerConfig.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(providerConfig.BaseURL))
	}

	// Set custom headers if provided
	if providerConfig.Headers != nil {
		for key, value := range providerConfig.Headers {
			opts = append(opts, option.WithHeader(key, value))
		}
	}

	// Set timeout if provided
	if providerConfig.TimeoutMs > 0 {
		timeout := time.Duration(providerConfig.TimeoutMs) * time.Millisecond
		opts = append(opts, option.WithHTTPClient(&http.Client{Timeout: timeout}))
	}

	client := openai.NewClient(opts...)
	chatService := chat.NewAnthropicChat(&client)

	return &AnthropicService{
		client: &client,
		chat:   chatService,
	}, nil
}
