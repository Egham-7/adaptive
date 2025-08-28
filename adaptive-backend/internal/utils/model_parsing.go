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
	// Trim whitespace from input
	trimmed := strings.TrimSpace(modelSpec)

	// Check for empty or whitespace-only specs
	if trimmed == "" {
		return "", "", fmt.Errorf("model specification cannot be empty or whitespace-only")
	}

	// Split on colon and ensure exactly 2 parts
	parts := strings.Split(trimmed, ":")
	if len(parts) != 2 {
		return "", "", fmt.Errorf("model specification must be in 'provider:model' format with exactly one colon, got '%s'", modelSpec)
	}

	// Trim whitespace from each part
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
	// Trim whitespace from input
	trimmed := strings.TrimSpace(modelSpec)

	// Check for empty or whitespace-only specs
	if trimmed == "" {
		return "", "", fmt.Errorf("model specification cannot be empty or whitespace-only")
	}

	// If no colon, use default provider
	if !strings.Contains(trimmed, ":") {
		if defaultProvider == "" {
			return "", "", fmt.Errorf("no provider specified in model '%s' and no default provider provided", modelSpec)
		}
		// Model name should be non-empty after trimming (already checked above)
		return defaultProvider, trimmed, nil
	}

	// Use strict parsing for provider:model format
	return ParseProviderModel(modelSpec)
}
