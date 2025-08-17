package providers

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers/openai"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"strings"
)

// NewLLMProvider creates an OpenAI-compatible LLM provider with appropriate base URL
func NewLLMProvider(cfg *config.Config, providerName string, customConfigs map[string]*models.ProviderConfig) (provider_interfaces.LLMProvider, error) {
	// Basic input validation
	if cfg == nil {
		return nil, fmt.Errorf("configuration cannot be nil")
	}

	if strings.TrimSpace(providerName) == "" {
		return nil, fmt.Errorf("provider name cannot be empty")
	}

	// Merge YAML config with custom override config if provided
	var mergedConfig models.ProviderConfig
	var err error

	if customConfigs != nil {
		if customConfig, hasCustom := customConfigs[providerName]; hasCustom {
			// Merge YAML config with request override
			mergedConfig, err = cfg.MergeProviderConfig(providerName, customConfig)
			if err != nil {
				return nil, fmt.Errorf("failed to merge provider config for '%s': %w", providerName, err)
			}
			return openai.NewCustomOpenAIService(mergedConfig.BaseURL, &mergedConfig)
		}
	}

	// No custom config provided, use YAML config only
	mergedConfig, exists := cfg.GetProviderConfig(providerName)
	if !exists {
		return nil, fmt.Errorf("provider '%s' is not configured", providerName)
	}

	// Use standard config
	return openai.NewOpenAIService(cfg, providerName)
}
