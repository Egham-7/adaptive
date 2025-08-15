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

	// Check if provider is configured
	providerConfig, exists := cfg.GetProviderConfig(providerName)
	if !exists || !providerConfig.Enabled {
		return nil, fmt.Errorf("provider '%s' is not configured or disabled", providerName)
	}

	// Use custom config if provided
	if customConfigs != nil {
		if customConfig, hasCustom := customConfigs[providerName]; hasCustom {
			return openai.NewCustomOpenAIService("", customConfig)
		}
	}

	// Use standard config
	return openai.NewOpenAIService(cfg, providerName)
}
