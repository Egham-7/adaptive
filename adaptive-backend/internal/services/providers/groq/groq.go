package groq

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

// NewCustomGroqService creates a new Groq service with custom configuration
func NewCustomGroqService(customConfig *models.ProviderConfig) (*GroqService, error) {
	if customConfig == nil {
		return nil, fmt.Errorf("custom config is required")
	}

	// Build client options
	opts := []option.RequestOption{}

	// Use custom base URL or default
	baseURL := "https://api.groq.com/openai/v1"
	if customConfig.BaseURL != nil {
		baseURL = *customConfig.BaseURL
	}
	opts = append(opts, option.WithBaseURL(baseURL))

	// Configure API key
	if customConfig.APIKey != nil {
		opts = append(opts, option.WithAPIKey(*customConfig.APIKey))
	} else {
		// Fall back to environment variable
		apiKey := os.Getenv("GROQ_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("GROQ_API_KEY environment variable not set and no API key in config")
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
	return &GroqService{client: &client}, nil
}
