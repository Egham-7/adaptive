package models

// SelectModelRequest represents a provider-agnostic request for model selection
type SelectModelRequest struct {
	// Available models with their capabilities and constraints
	Models []ModelCapability `json:"models"`
	// The prompt text to analyze for optimal model selection
	Prompt string `json:"prompt"`
	// Optional user identifier for tracking and personalization
	User *string `json:"user,omitzero"`
	// Cost bias for model selection (0.0 = cheapest, 1.0 = best performance)
	CostBias *float32 `json:"cost_bias,omitzero"`
	// Model router cache configuration
	ModelRouterCache *CacheConfig `json:"model_router_cache,omitzero"`
}

// SelectModelResponse represents the response for model selection
type SelectModelResponse struct {
	// Selected provider
	Provider string `json:"provider"`
	// Selected model
	Model string `json:"model"`
	// Alternative provider/model combinations
	Alternatives []Alternative `json:"alternatives,omitzero"`
	// Additional metadata about the selection
	Metadata SelectionMetadata `json:"metadata"`
}

// SelectionMetadata provides additional information about the model selection
type SelectionMetadata struct {
	Reasoning   string  `json:"reasoning,omitzero"`
	CostPer1M   float64 `json:"cost_per_1m_tokens,omitzero"`
	Complexity  string  `json:"complexity,omitzero"`
	CacheSource string  `json:"cache_source,omitzero"`
}
