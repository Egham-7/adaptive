package providers

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers/provider_interfaces"
)

// NewLLMProvider creates an LLM provider using the new factory system
func NewLLMProvider(cfg *config.Config, providerName string, customConfigs map[string]*models.ProviderConfig) (provider_interfaces.LLMProvider, error) {
	factory := NewProviderFactory(cfg)
	
	if customConfigs != nil {
		// Extract the specific provider's config from the map
		if providerConfig, exists := customConfigs[providerName]; exists {
			return factory.CreateProviderWithCustomConfig(providerName, providerConfig)
		}
	}
	
	return factory.CreateProvider(providerName)
}