package huggingface

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers/huggingface/chat"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// HuggingFaceService handles HuggingFace API interactions using OpenAI SDK
type HuggingFaceService struct {
	client *openai.Client
	chat   *chat.HuggingFaceChat
}

// NewHuggingFaceService creates a new HuggingFace service using OpenAI SDK with HF base URL
func NewHuggingFaceService() (*HuggingFaceService, error) {
	apiKey := os.Getenv("HF_TOKEN")
	if apiKey == "" {
		return nil, fmt.Errorf("HF_TOKEN environment variable not set")
	}

	baseURL := "https://router.huggingface.co/v1"

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	)

	chatService := chat.NewHuggingFaceChat(&client)

	return &HuggingFaceService{
		client: &client,
		chat:   chatService,
	}, nil
}

func (s *HuggingFaceService) Chat() provider_interfaces.Chat {
	return s.chat
}

func (s *HuggingFaceService) GetProviderName() string {
	return "huggingface"
}

// NewCustomHuggingFaceService creates a new HuggingFace service with custom configuration
func NewCustomHuggingFaceService(customConfig *models.ProviderConfig) (*HuggingFaceService, error) {
	if customConfig == nil {
		return nil, fmt.Errorf("custom config is required")
	}

	// Build client options
	opts := []option.RequestOption{}

	// Use custom base URL or default
	baseURL := "https://router.huggingface.co/v1"
	if customConfig.BaseURL != nil {
		baseURL = *customConfig.BaseURL
	}
	opts = append(opts, option.WithBaseURL(baseURL))

	// Configure API key
	if customConfig.APIKey != nil {
		opts = append(opts, option.WithAPIKey(*customConfig.APIKey))
	} else {
		// Fall back to environment variable
		apiKey := os.Getenv("HF_TOKEN")
		if apiKey == "" {
			return nil, fmt.Errorf("HF_TOKEN environment variable not set and no API key in config")
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
	chatService := chat.NewHuggingFaceChat(&client)

	return &HuggingFaceService{
		client: &client,
		chat:   chatService,
	}, nil
}
