package services

import (
	"adaptive-backend/internal/models"
	"errors"
	"strings"
)

type LLMProvider interface {
	CreateChatCompletion(req *models.ProviderChatCompletionRequest) (*models.ChatCompletionResponse, error)
}

func NewLLMProvider(providerName string) (LLMProvider, error) {
	switch strings.ToLower(providerName) {
	case "openai":
		return NewOpenAIService(), nil
	case "groq":
		service, err := NewGroqService()
		if err != nil {
			return nil, err
		}
		return service, nil
	default:
		return nil, errors.New("unsupported provider: " + providerName)
	}
}
