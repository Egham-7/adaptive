package utils

import (
	"fmt"
	"strings"
)

// ParseProviderModel parses a model specification in "provider:model" format.
// This function is strict - it requires exact format and will error if not correct.
// Examples:
//   - "openai:gpt-4" -> ("openai", "gpt-4", nil)
//   - "anthropic:claude-3-5-sonnet-20241022" -> ("anthropic", "claude-3-5-sonnet-20241022", nil)
//   - "gpt-4" -> error (no provider specified)
//   - "openai:" -> error (empty model)
//   - ":gpt-4" -> error (empty provider)
//   - "openai:gpt-4:extra" -> error (too many parts)
func ParseProviderModel(modelSpec string) (provider, model string, err error) {
	if modelSpec == "" {
		return "", "", fmt.Errorf("model specification cannot be empty")
	}

	// Must contain exactly one colon
	if !strings.Contains(modelSpec, ":") {
		return "", "", fmt.Errorf("model specification must be in 'provider:model' format, got '%s'", modelSpec)
	}

	// Split on colon - must have exactly 2 parts
	parts := strings.Split(modelSpec, ":")
	if len(parts) != 2 {
		return "", "", fmt.Errorf("model specification must be in 'provider:model' format with exactly one colon, got '%s'", modelSpec)
	}

	provider = strings.TrimSpace(parts[0])
	model = strings.TrimSpace(parts[1])

	// Both provider and model must be non-empty after trimming
	if provider == "" {
		return "", "", fmt.Errorf("provider cannot be empty in model specification '%s'", modelSpec)
	}

	if model == "" {
		return "", "", fmt.Errorf("model cannot be empty in model specification '%s'", modelSpec)
	}

	return provider, model, nil
}

// ParseProviderModelWithDefault parses a model specification, using defaultProvider if no provider is specified.
// This is less strict than ParseProviderModel - it allows model-only specifications.
// Examples:
//   - "openai:gpt-4" -> ("openai", "gpt-4", nil)
//   - "gpt-4" with defaultProvider="openai" -> ("openai", "gpt-4", nil)
//   - "anthropic:" -> error (empty model)
//   - ":gpt-4" -> error (empty provider)
func ParseProviderModelWithDefault(modelSpec, defaultProvider string) (provider, model string, err error) {
	if modelSpec == "" {
		return "", "", fmt.Errorf("model specification cannot be empty")
	}

	// If no colon, use default provider
	if !strings.Contains(modelSpec, ":") {
		if defaultProvider == "" {
			return "", "", fmt.Errorf("no provider specified in model '%s' and no default provider provided", modelSpec)
		}
		return defaultProvider, strings.TrimSpace(modelSpec), nil
	}

	// Use strict parsing for provider:model format
	return ParseProviderModel(modelSpec)
}
