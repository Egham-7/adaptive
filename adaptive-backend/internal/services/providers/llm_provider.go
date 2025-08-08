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

func createOpenAI() (provider_interfaces.LLMProvider, error) { return openai.NewOpenAIService() }
func createAnthropic() (provider_interfaces.LLMProvider, error) {
	return anthropic.NewAnthropicService()
}
func createDeepSeek() (provider_interfaces.LLMProvider, error) { return deepseek.NewDeepSeekService() }
func createGemini() (provider_interfaces.LLMProvider, error)   { return gemini.NewGeminiService() }
func createHuggingFace() (provider_interfaces.LLMProvider, error) {
	return huggingface.NewHuggingFaceService()
}
func createGroq() (provider_interfaces.LLMProvider, error) { return groq.NewGroqService() }
func createGrok() (provider_interfaces.LLMProvider, error) { return grok.NewGrokService() }

func createCustomOpenAI(config *models.ProviderConfig) (provider_interfaces.LLMProvider, error) {
	if config.BaseURL == nil {
		return nil, errors.New("custom OpenAI provider requires base URL")
	}
	return openai.NewCustomOpenAIService(*config.BaseURL, config)
}
func createCustomAnthropic(config *models.ProviderConfig) (provider_interfaces.LLMProvider, error) {
	return anthropic.NewCustomAnthropicService(config)
}
func createCustomDeepSeek(config *models.ProviderConfig) (provider_interfaces.LLMProvider, error) {
	return deepseek.NewCustomDeepSeekService(config)
}
func createCustomGemini(config *models.ProviderConfig) (provider_interfaces.LLMProvider, error) {
	return gemini.NewCustomGeminiService(config)
}
func createCustomHuggingFace(config *models.ProviderConfig) (provider_interfaces.LLMProvider, error) {
	return huggingface.NewCustomHuggingFaceService(config)
}
func createCustomGroq(config *models.ProviderConfig) (provider_interfaces.LLMProvider, error) {
	return groq.NewCustomGroqService(config)
}
func createCustomGrok(config *models.ProviderConfig) (provider_interfaces.LLMProvider, error) {
	return grok.NewCustomGrokService(config)
}

type providerConstructor func() (provider_interfaces.LLMProvider, error)
type customProviderConstructor func(*models.ProviderConfig) (provider_interfaces.LLMProvider, error)

var defaultConstructors = map[string]providerConstructor{
	"openai":      createOpenAI,
	"anthropic":   createAnthropic,
	"deepseek":    createDeepSeek,
	"gemini":      createGemini,
	"huggingface": createHuggingFace,
	"groq":        createGroq,
	"grok":        createGrok,
}

var customConstructors = map[string]customProviderConstructor{
	"openai":      createCustomOpenAI,
	"anthropic":   createCustomAnthropic,
	"deepseek":    createCustomDeepSeek,
	"gemini":      createCustomGemini,
	"huggingface": createCustomHuggingFace,
	"groq":        createCustomGroq,
	"grok":        createCustomGrok,
}

func NewLLMProvider(providerName string, providerConfigs map[string]*models.ProviderConfig) (provider_interfaces.LLMProvider, error) {
	normalizedName := strings.ToLower(providerName)

	// Check if custom config exists for this provider
	if providerConfigs != nil {
		if customConfig, exists := providerConfigs[providerName]; exists && customConfig != nil {
			// Use provider-specific custom constructor if available
			if customConstructor, exists := customConstructors[normalizedName]; exists {
				return customConstructor(customConfig)
			}

			// For unknown providers, use OpenAI SDK with custom base URL
			if customConfig.BaseURL == nil {
				return nil, errors.New("custom provider '" + providerName + "' requires base URL")
			}
			return openai.NewCustomOpenAIService(*customConfig.BaseURL, customConfig)
		}
	}

	// Use default provider constructor if available
	if defaultConstructor, exists := defaultConstructors[normalizedName]; exists {
		return defaultConstructor()
	}

	// Unknown provider
	return nil, errors.New("unknown provider '" + providerName + "'")
}
