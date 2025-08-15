package openai

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers/openai/chat"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"net/http"
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
func NewOpenAIService(cfg *config.Config, providerName string) (*OpenAIService, error) {
	providerConfig, exists := cfg.GetProviderConfig(providerName)
	if !exists {
		return nil, fmt.Errorf("provider '%s' not found in configuration", providerName)
	}

	if providerConfig.APIKey == "" {
		return nil, fmt.Errorf("%s API key not set in configuration", providerName)
	}

	opts := []option.RequestOption{
		option.WithAPIKey(providerConfig.APIKey),
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
		httpClient := &http.Client{Timeout: timeout}
		opts = append(opts, option.WithHTTPClient(httpClient))
	}

	client := openai.NewClient(opts...)
	chatService := chat.NewOpenAIChat(&client)

	return &OpenAIService{
		client: &client,
		chat:   chatService,
	}, nil
}

// NewOpenAIServiceWithOptions creates a new OpenAI service with custom options
func NewOpenAIServiceWithOptions(opts []option.RequestOption) (*OpenAIService, error) {
	client := openai.NewClient(opts...)
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
		if customConfig.APIKey != "" {
			opts = append(opts, option.WithAPIKey(customConfig.APIKey))
		}

		// Configure timeout if specified
		if customConfig.TimeoutMs != 0 {
			timeout := time.Duration(customConfig.TimeoutMs) * time.Millisecond
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
