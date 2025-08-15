package gemini

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers/gemini/chat"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"net/http"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// GeminiService handles Gemini API interactions using OpenAI Go SDK
type GeminiService struct {
	client *openai.Client
	chat   *chat.GeminiChat
}

// NewGeminiService creates a new Gemini service using OpenAI Go SDK with Gemini base URL
func NewGeminiService(cfg *config.Config) (*GeminiService, error) {
	apiKey := cfg.GetProviderAPIKey("gemini")
	if apiKey == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY not configured")
	}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL("https://generativelanguage.googleapis.com/v1beta/openai/"),
	)

	chatService := chat.NewGeminiChat(&client)

	return &GeminiService{
		client: &client,
		chat:   chatService,
	}, nil
}

// Chat implements LLMProvider interface
func (s *GeminiService) Chat() provider_interfaces.Chat {
	return s.chat
}

func (s *GeminiService) GetProviderName() string {
	return "gemini"
}

// NewCustomGeminiService creates a new Gemini service with custom configuration
func NewCustomGeminiService(cfg *config.Config, customConfig *models.ProviderConfig) (*GeminiService, error) {
	if customConfig == nil {
		return nil, fmt.Errorf("custom config is required")
	}

	// Build client options
	opts := []option.RequestOption{}

	// Use custom base URL or default
	baseURL := "https://generativelanguage.googleapis.com/v1beta/openai/"
	if customConfig.BaseURL != "" {
		baseURL = customConfig.BaseURL
	}
	opts = append(opts, option.WithBaseURL(baseURL))

	// Configure API key
	if customConfig.APIKey != "" {
		opts = append(opts, option.WithAPIKey(customConfig.APIKey))
	} else {
		// Fall back to config
		apiKey := cfg.GetProviderAPIKey("gemini")
		if apiKey == "" {
			return nil, fmt.Errorf("GEMINI_API_KEY not configured and no API key in config")
		}
		opts = append(opts, option.WithAPIKey(apiKey))
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

	client := openai.NewClient(opts...)
	chatService := chat.NewGeminiChat(&client)

	return &GeminiService{
		client: &client,
		chat:   chatService,
	}, nil
}

// NewGeminiServiceWithConfig creates a new Gemini service with custom configuration
func NewGeminiServiceWithConfig(providerConfig models.ProviderConfig) (*GeminiService, error) {
	if providerConfig.APIKey == "" {
		return nil, fmt.Errorf("gemini API key not set in configuration")
	}

	// Build client options
	opts := []option.RequestOption{
		option.WithAPIKey(providerConfig.APIKey),
		option.WithBaseURL("https://generativelanguage.googleapis.com/v1beta/"),
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
	chatService := chat.NewGeminiChat(&client)

	return &GeminiService{
		client: &client,
		chat:   chatService,
	}, nil
}
