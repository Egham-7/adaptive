package providers

import (
	"adaptive-backend/internal/services/minions"
	"adaptive-backend/internal/services/providers/anthropic"
	"adaptive-backend/internal/services/providers/deepseek"
	"adaptive-backend/internal/services/providers/gemini"
	"adaptive-backend/internal/services/providers/openai"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"errors"
	"strings"
)

func NewLLMProvider(providerName string, taskType *string, minionRegistry *minions.MinionRegistry) (provider_interfaces.LLMProvider, error) {
	switch strings.ToLower(providerName) {
	case "openai":
		service, err := openai.NewOpenAIService(nil)
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

case "minion":

    if minionRegistry == nil {
        return nil, errors.New("minion registry must be provided for minion provider")
    }

    if taskType == nil {
        return nil, errors.New("task type must be provided for minion provider")
    }

    baseURL, found := minionRegistry.GetMinionURL(*taskType)

    if !found {
        return nil, errors.New("minion not found for task type: " + *taskType)
    }

    service, err := openai.NewOpenAIService(&baseURL)
    if err != nil {
        return nil, err
    }
    return service, nil

	default:
		return nil, errors.New("unsupported provider: " + providerName)
	}
}
