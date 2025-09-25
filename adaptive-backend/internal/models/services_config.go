package models

// ServicesConfig holds configuration for external services
type ServicesConfig struct {
	ModelRouter ModelRouterConfig `json:"model_router" yaml:"model_router"`
	Redis       RedisConfig       `json:"redis" yaml:"redis"`
}

// RedisConfig holds configuration for Redis
type RedisConfig struct {
	URL string `json:"url,omitzero" yaml:"url"`
}

// ModelRouterConfig holds configuration for the model router
type ModelRouterConfig struct {
	// YAML config fields
	SemanticCache CacheConfig             `json:"semantic_cache" yaml:"semantic_cache"`
	Client        ModelRouterClientConfig `json:"client" yaml:"client"`
	CostBias      float32                 `json:"cost_bias,omitzero" yaml:"cost_bias"`
	// Request-level config fields
	Models []ModelCapability `json:"models,omitzero"`
}

// ModelRouterClientConfig holds client configuration for model router
type ModelRouterClientConfig struct {
	BaseURL        string               `json:"base_url,omitzero" yaml:"base_url"`
	TimeoutMs      int                  `json:"timeout_ms,omitzero" yaml:"timeout_ms"`
	CircuitBreaker CircuitBreakerConfig `json:"circuit_breaker" yaml:"circuit_breaker"`
}
