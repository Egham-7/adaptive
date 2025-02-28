package services

import (
	"adaptive-backend/internal/models"
	"errors"
)

type LLMProvider interface {
	CreateChatCompletion(req *models.ChatCompletionRequest) (*models.ChatCompletionResponse, error)
}

func NewLLMProvider(providerName string) (LLMProvider, error) {
	switch providerName {
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
