package providers

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers/anthropic"
	"adaptive-backend/internal/services/providers/gemini"
	"adaptive-backend/internal/services/providers/openai"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"net/http"
	"time"

	"github.com/openai/openai-go/option"
)

// ProviderFactory creates LLM providers based on configuration
type ProviderFactory struct {
	cfg *config.Config
}

// NewProviderFactory creates a new provider factory
func NewProviderFactory(cfg *config.Config) *ProviderFactory {
	return &ProviderFactory{cfg: cfg}
}

// CreateProvider creates an LLM provider based on the provider name and configuration
func (f *ProviderFactory) CreateProvider(providerName string) (provider_interfaces.LLMProvider, error) {
	// First check if it's a configured provider
	providerConfig, exists := f.cfg.GetProviderConfig(providerName)
	if !exists || !providerConfig.Enabled {
		return nil, fmt.Errorf("provider '%s' is not configured or disabled", providerName)
	}

	// Create provider based on type
	switch providerConfig.Type {
	case "openai":
		return f.createOpenAIProvider(providerName, providerConfig)
	case "anthropic":
		return f.createAnthropicProvider(providerName, providerConfig)
	case "gemini":
		return f.createGeminiProvider(providerName, providerConfig)
	default:
		return nil, fmt.Errorf("unsupported provider type '%s' for provider '%s'", providerConfig.Type, providerName)
	}
}

// CreateProviderWithCustomConfig creates a provider with custom runtime configuration
func (f *ProviderFactory) CreateProviderWithCustomConfig(providerName string, customConfig *models.ProviderConfig) (provider_interfaces.LLMProvider, error) {
	// Check if the provider exists in config for base settings
	baseConfig, exists := f.cfg.GetProviderConfig(providerName)
	if !exists {
		// For unknown providers, try to infer from custom config or default to openai
		baseConfig = models.ProviderConfig{
			Type:    "openai", // Default type
			Enabled: true,
		}
	}

	// Override with custom config
	providerConfig := f.mergeCustomConfig(baseConfig, *customConfig)

	// Create provider based on type
	switch providerConfig.Type {
	case "openai":
		return f.createOpenAIProvider(providerName, providerConfig)
	case "anthropic":
		return f.createAnthropicProvider(providerName, providerConfig)
	case "gemini":
		return f.createGeminiProvider(providerName, providerConfig)
	default:
		return nil, fmt.Errorf("unsupported provider type '%s' for custom provider '%s'", providerConfig.Type, providerName)
	}
}

// createOpenAIProvider creates an OpenAI-compatible provider
func (f *ProviderFactory) createOpenAIProvider(providerName string, config models.ProviderConfig) (provider_interfaces.LLMProvider, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("API key not set for provider '%s'", providerName)
	}

	// Build client options
	opts := []option.RequestOption{
		option.WithAPIKey(config.APIKey),
	}

	// Set custom base URL if provided
	if config.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(config.BaseURL))
	}

	// Set custom headers if provided
	if config.Headers != nil {
		for key, value := range config.Headers {
			opts = append(opts, option.WithHeader(key, value))
		}
	}

	// Set timeout if provided
	if config.TimeoutMs > 0 {
		timeout := time.Duration(config.TimeoutMs) * time.Millisecond
		opts = append(opts, option.WithHTTPClient(&http.Client{Timeout: timeout}))
	}

	return openai.NewOpenAIServiceWithOptions(opts)
}

// createAnthropicProvider creates an Anthropic provider
func (f *ProviderFactory) createAnthropicProvider(providerName string, config models.ProviderConfig) (provider_interfaces.LLMProvider, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("API key not set for provider '%s'", providerName)
	}

	// For Anthropic, we can use custom config if provided
	if config.BaseURL != "" || config.Headers != nil || config.TimeoutMs > 0 {
		return anthropic.NewAnthropicServiceWithConfig(config)
	}

	return anthropic.NewAnthropicService(f.cfg)
}

// createGeminiProvider creates a Gemini provider
func (f *ProviderFactory) createGeminiProvider(providerName string, config models.ProviderConfig) (provider_interfaces.LLMProvider, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("API key not set for provider '%s'", providerName)
	}

	// For Gemini, we can use custom config if provided
	if config.BaseURL != "" || config.Headers != nil || config.TimeoutMs > 0 {
		return gemini.NewGeminiServiceWithConfig(config)
	}

	return gemini.NewGeminiService(f.cfg)
}

// mergeCustomConfig merges custom configuration with base configuration
func (f *ProviderFactory) mergeCustomConfig(base models.ProviderConfig, custom models.ProviderConfig) models.ProviderConfig {
	merged := base

	if custom.APIKey != "" {
		merged.APIKey = custom.APIKey
	}
	if custom.Type != "" {
		merged.Type = custom.Type
	}
	if custom.BaseURL != "" {
		merged.BaseURL = custom.BaseURL
	}
	if custom.TimeoutMs > 0 {
		merged.TimeoutMs = custom.TimeoutMs
	}
	if len(custom.Headers) > 0 {
		if merged.Headers == nil {
			merged.Headers = make(map[string]string)
		}
		for k, v := range custom.Headers {
			merged.Headers[k] = v
		}
	}

	return merged
}

// GetEnabledProviders returns a list of all enabled provider names
func (f *ProviderFactory) GetEnabledProviders() []string {
	var providers []string
	for name, config := range f.cfg.Providers {
		if config.Enabled {
			providers = append(providers, name)
		}
	}
	return providers
}
