package grok

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"net/http"
	"os"
	"time"

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

// NewCustomGrokService creates a new Grok service with custom configuration
func NewCustomGrokService(customConfig *models.ProviderConfig) (*GrokService, error) {
	if customConfig == nil {
		return nil, fmt.Errorf("custom config is required")
	}

	// Build client options
	opts := []option.RequestOption{}

	// Use custom base URL or default
	baseURL := "https://api.x.ai/v1"
	if customConfig.BaseURL != nil {
		baseURL = *customConfig.BaseURL
	}
	opts = append(opts, option.WithBaseURL(baseURL))

	// Configure API key
	if customConfig.APIKey != nil {
		opts = append(opts, option.WithAPIKey(*customConfig.APIKey))
	} else {
		// Fall back to environment variable
		apiKey := os.Getenv("GROK_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("GROK_API_KEY environment variable not set and no API key in config")
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
	return &GrokService{client: &client}, nil
}
