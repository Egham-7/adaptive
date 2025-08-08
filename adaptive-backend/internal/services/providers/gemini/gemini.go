package gemini

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers/gemini/chat"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// GeminiService handles Gemini API interactions using OpenAI Go SDK
type GeminiService struct {
	client *openai.Client
	chat   *chat.GeminiChat
}

// NewGeminiService creates a new Gemini service using OpenAI Go SDK with Google base URL
func NewGeminiService() (*GeminiService, error) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("GOOGLE_API_KEY environment variable not set")
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
func NewCustomGeminiService(customConfig *models.ProviderConfig) (*GeminiService, error) {
	if customConfig == nil {
		return nil, fmt.Errorf("custom config is required")
	}

	// Build client options
	opts := []option.RequestOption{}

	// Use custom base URL or default
	baseURL := "https://generativelanguage.googleapis.com/v1beta/openai/"
	if customConfig.BaseURL != nil {
		baseURL = *customConfig.BaseURL
	}
	opts = append(opts, option.WithBaseURL(baseURL))

	// Configure API key
	if customConfig.APIKey != nil {
		opts = append(opts, option.WithAPIKey(*customConfig.APIKey))
	} else {
		// Fall back to environment variable
		apiKey := os.Getenv("GOOGLE_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("GOOGLE_API_KEY environment variable not set and no API key in config")
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
	chatService := chat.NewGeminiChat(&client)

	return &GeminiService{
		client: &client,
		chat:   chatService,
	}, nil
}
