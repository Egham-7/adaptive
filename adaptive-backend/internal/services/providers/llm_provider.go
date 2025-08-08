package providers

import (
	"adaptive-backend/internal/models"
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

func NewLLMProvider(providerName string, providerConfigs map[string]*models.ProviderConfig) (provider_interfaces.LLMProvider, error) {
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
		// Handle custom provider using OpenAI SDK with base URL override
		if providerConfigs == nil {
			return nil, errors.New("custom provider '" + providerName + "' requires provider_configs")
		}
		customConfig, exists := providerConfigs[providerName]
		if !exists || customConfig == nil {
			return nil, errors.New("custom provider '" + providerName + "' configuration not found")
		}
		if customConfig.BaseURL == nil {
			return nil, errors.New("custom provider '" + providerName + "' requires base URL")
		}
		service, err := openai.NewCustomOpenAIService(*customConfig.BaseURL, customConfig)
		if err != nil {
			return nil, err
		}
		return service, nil
	}
}
