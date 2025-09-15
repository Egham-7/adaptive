package models

// PromptCacheConfig holds configuration for prompt caching
type CacheConfig struct {
	// YAML config fields (not overridable in requests)
	Enabled           bool    `json:"enabled,omitzero" yaml:"enabled"`
	SemanticThreshold float64 `json:"semantic_threshold,omitzero" yaml:"semantic_threshold"`
	OpenAIAPIKey      string  `json:"openai_api_key,omitzero" yaml:"openai_api_key"`
	EmbeddingModel    string  `json:"embedding_model,omitzero" yaml:"embedding_model"`
}
