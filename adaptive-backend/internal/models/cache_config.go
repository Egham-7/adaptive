package models

// PromptCacheConfig holds configuration for prompt caching
type CacheConfig struct {
	// YAML config fields (not overridable in requests)
	Enabled           bool    `json:"enabled,omitzero" yaml:"enabled"`
	DefaultTTLSeconds int     `json:"default_ttl_seconds,omitzero" yaml:"default_ttl_seconds"`
	RedisURL          string  `json:"redis_url,omitzero" yaml:"redis_url"`
	SemanticThreshold float64 `json:"semantic_threshold,omitzero" yaml:"semantic_threshold"`
	OpenAIAPIKey      string  `json:"openai_api_key,omitzero" yaml:"openai_api_key"`
	TTL               int     `json:"ttl,omitzero" yaml:"ttl,omitempty"` // TTL in seconds for request-level config
}

// ModelRouterConfig holds configuration for the model router
type ModelRouterConfig struct {
	// YAML config fields
	SemanticCache CacheConfig             `json:"semantic_cache" yaml:"semantic_cache"`
	Client        ModelRouterClientConfig `json:"client" yaml:"client"`
	// Request-level config fields
	Models              []ModelCapability `json:"models,omitzero"`
	CostBias            float32           `json:"cost_bias,omitzero"`
	ComplexityThreshold *float32          `json:"complexity_threshold,omitzero"`
}

// ModelRouterClientConfig holds client configuration for model router
type ModelRouterClientConfig struct {
	BaseURL        string               `json:"base_url,omitzero" yaml:"base_url"`
	TimeoutMs      int                  `json:"timeout_ms,omitzero" yaml:"timeout_ms"`
	CircuitBreaker CircuitBreakerConfig `json:"circuit_breaker" yaml:"circuit_breaker"`
}
