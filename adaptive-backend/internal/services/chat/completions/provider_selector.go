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
	requestID string,
) (provider_interfaces.LLMProvider, string, error) {
	// Try primary provider first
	prov, err := providers.NewLLMProvider(standardInfo.Provider)
	if err == nil {
		fiberlog.Infof("[%s] Using primary standard provider: %s (%s)", requestID, standardInfo.Provider, standardInfo.Model)
		return prov, standardInfo.Model, nil
	}

	fiberlog.Warnf("[%s] Primary standard provider %s failed: %v", requestID, standardInfo.Provider, err)

	// Primary failed, try alternatives
	if len(standardInfo.Alternatives) == 0 {
		return nil, "", fmt.Errorf("primary standard provider failed and no alternatives: %v", err)
	}

	fallbackSvc := NewFallbackService()
	result, fallbackErr := fallbackSvc.SelectAlternative(ctx, standardInfo.Alternatives, requestID)
	if fallbackErr != nil {
		return nil, "", fmt.Errorf("all standard providers failed: %v", fallbackErr)
	}

	fiberlog.Infof("[%s] Using standard alternative: %s (%s)", requestID, result.ProviderName, result.ModelName)
	return result.Provider, result.ModelName, nil
}

// SelectMinionProvider selects a minion provider with fallback logic.
func (ps *ProviderSelector) SelectMinionProvider(
	ctx context.Context,
	minionInfo *models.MinionInfo,
	requestID string,
) (provider_interfaces.LLMProvider, string, error) {
	// Try primary provider first
	prov, err := providers.NewLLMProvider(minionInfo.Provider)
	if err == nil {
		fiberlog.Infof("[%s] Using primary minion provider: %s (%s)", requestID, minionInfo.Provider, minionInfo.Model)
		return prov, minionInfo.Model, nil
	}

	fiberlog.Warnf("[%s] Primary minion provider %s failed: %v", requestID, minionInfo.Provider, err)

	// Primary failed, try alternatives
	if len(minionInfo.Alternatives) == 0 {
		return nil, "", fmt.Errorf("primary minion provider failed and no alternatives: %v", err)
	}

	fallbackSvc := NewFallbackService()
	result, fallbackErr := fallbackSvc.SelectAlternative(ctx, minionInfo.Alternatives, requestID)
	if fallbackErr != nil {
		return nil, "", fmt.Errorf("all minion providers failed: %v", fallbackErr)
	}

	fiberlog.Infof("[%s] Using minion alternative: %s (%s)", requestID, result.ProviderName, result.ModelName)
	return result.Provider, result.ModelName, nil
}
