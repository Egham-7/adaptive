package providers

import (
	"adaptive-backend/internal/services/providers/anthropic"
	"adaptive-backend/internal/services/providers/openai"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"errors"
	"strings"
)

func NewLLMProvider(providerName string) (provider_interfaces.LLMProvider, error) {
	switch strings.ToLower(providerName) {
	case "openai":
		service, err := openai.NewOpenAIService()
		if err != nil {
			return nil, err
		}
		return service, nil

	case "groq":
		service, err := NewGroqService()
		if err != nil {
			return nil, err
		}
		return service, nil
	case "deepseek":
		service, err := NewDeepSeekService()
		if err != nil {
			return nil, err
		}
		return service, nil
	case "anthropic":
		service, err := anthropic.NewAnthropicService()
		if err != nil {
			return nil, err
		}
		return service, nil
	case "gemini":
		service, err := NewGeminiService()
		if err != nil {
			return nil, err
		}
		return service, nil
	default:
		return nil, errors.New("unsupported provider: " + providerName)
	}
}
