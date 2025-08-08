package completions

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"context"
	"fmt"

	fiberlog "github.com/gofiber/fiber/v2/log"
)

// ProviderSelector handles provider selection with fallback logic.
type ProviderSelector struct{}

// NewProviderSelector creates a new provider selector.
func NewProviderSelector() *ProviderSelector {
	return &ProviderSelector{}
}

// SelectStandardProvider selects a standard provider with fallback logic.
func (ps *ProviderSelector) SelectStandardProvider(
	ctx context.Context,
	standardInfo *models.StandardLLMInfo,
	providerConfigs map[string]*models.ProviderConfig,
	requestID string,
) (provider_interfaces.LLMProvider, string, error) {
	return ps.selectProviderWithFallback(
		ctx,
		standardInfo.Provider,
		standardInfo.Model,
		standardInfo.Alternatives,
		providerConfigs,
		"standard",
		requestID,
	)
}

// SelectMinionProvider selects a minion provider with fallback logic.
func (ps *ProviderSelector) SelectMinionProvider(
	ctx context.Context,
	minionInfo *models.MinionInfo,
	providerConfigs map[string]*models.ProviderConfig,
	requestID string,
) (provider_interfaces.LLMProvider, string, error) {
	return ps.selectProviderWithFallback(
		ctx,
		minionInfo.Provider,
		minionInfo.Model,
		minionInfo.Alternatives,
		providerConfigs,
		"minion",
		requestID,
	)
}

// selectProviderWithFallback handles common provider selection logic with fallback
func (ps *ProviderSelector) selectProviderWithFallback(
	ctx context.Context,
	primaryProvider string,
	primaryModel string,
	alternatives []models.Alternative,
	providerConfigs map[string]*models.ProviderConfig,
	providerType string,
	requestID string,
) (provider_interfaces.LLMProvider, string, error) {
	// Try primary provider first
	prov, err := providers.NewLLMProvider(primaryProvider, providerConfigs)
	if err == nil {
		fiberlog.Infof("[%s] Using primary %s provider: %s (%s)", requestID, providerType, primaryProvider, primaryModel)
		return prov, primaryModel, nil
	}

	fiberlog.Warnf("[%s] Primary %s provider %s failed: %v", requestID, providerType, primaryProvider, err)

	// Primary failed, try alternatives
	if len(alternatives) == 0 {
		return nil, "", fmt.Errorf("primary %s provider failed and no alternatives: %v", providerType, err)
	}

	fallbackSvc := NewFallbackService()
	alternativesCopy := make([]models.Alternative, len(alternatives))
	copy(alternativesCopy, alternatives)
	result, fallbackErr := fallbackSvc.SelectAlternative(ctx, &alternativesCopy, providerConfigs, requestID)
	if fallbackErr != nil {
		return nil, "", fmt.Errorf("all %s providers failed: %v", providerType, fallbackErr)
	}

	fiberlog.Infof("[%s] Using %s alternative: %s (%s)", requestID, providerType, result.ProviderName, result.ModelName)
	return result.Provider, result.ModelName, nil
}
