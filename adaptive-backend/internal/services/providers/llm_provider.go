package providers

import (
	"errors"
	"strings"

	"adaptive-backend/internal/services/providers/openai"
	"adaptive-backend/internal/services/providers/provider_interfaces"
)

func NewLLMProvider(providerName string) (provider_interfaces.LLMProvider, error) {
	switch strings.ToLower(providerName) {
	case "openai":
		return openai.NewOpenAIService(), nil
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
		service, err := NewAnthropicService()
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
