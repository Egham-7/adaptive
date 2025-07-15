package huggingface

import (
	"adaptive-backend/internal/services/providers/huggingface/chat"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"os"

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
