package providers

import (
	"adaptive-backend/internal/services/providers/anthropic"
	"adaptive-backend/internal/services/providers/deepseek"
	"adaptive-backend/internal/services/providers/gemini"
	"adaptive-backend/internal/services/providers/grok"
	"adaptive-backend/internal/services/providers/groq"
	"adaptive-backend/internal/services/providers/huggingface"
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

	case "deepseek":
		service, err := deepseek.NewDeepSeekService()
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
		service, err := gemini.NewGeminiService()
		if err != nil {
			return nil, err
		}
		return service, nil

	case "huggingface":
		service, err := huggingface.NewHuggingFaceService()
		if err != nil {
			return nil, err
		}
		return service, nil

	case "groq":
		service, err := groq.NewGroqService()
		if err != nil {
			return nil, err
		}
		return service, nil

	case "grok":
		service, err := grok.NewGrokService()
		if err != nil {
			return nil, err
		}
		return service, nil

	default:
		return nil, errors.New("unsupported provider: " + providerName)
	}
}
