package models

// SemanticCacheConfig holds configuration for semantic caching
type SemanticCacheConfig struct {
	Enabled      bool    `json:"enabled,omitempty" yaml:"enabled"`
	Threshold    float64 `json:"threshold,omitempty" yaml:"threshold"`
	RedisURL     string  `json:"redis_url,omitempty" yaml:"redis_url"`
	OpenAIAPIKey string  `json:"openai_api_key,omitempty" yaml:"openai_api_key"`
}

// PromptCacheConfig holds configuration for prompt caching
type PromptCacheConfig struct {
	Enabled           bool   `json:"enabled,omitempty" yaml:"enabled"`
	DefaultTTLSeconds int    `json:"default_ttl_seconds,omitempty" yaml:"default_ttl_seconds"`
	TTL               int    `json:"ttl,omitempty" yaml:"ttl,omitempty"` // TTL in seconds for request-level config
	RedisURL          string `json:"redis_url,omitempty" yaml:"redis_url"`
}

// ProtocolManagerConfig holds configuration for the protocol manager
type ProtocolManagerConfig struct {
	// YAML config fields
	SemanticCache SemanticCacheConfig         `json:"semantic_cache,omitempty" yaml:"semantic_cache"`
	Client        ProtocolManagerClientConfig `json:"client,omitempty" yaml:"client"`
	// Request-level config fields
	Models              []ModelCapability `json:"models,omitempty"`
	CostBias            float32           `json:"cost_bias,omitempty"`
	ComplexityThreshold *float32          `json:"complexity_threshold,omitempty"`
	TokenThreshold      *int              `json:"token_threshold,omitempty"`
}

// ProtocolManagerClientConfig holds client configuration for protocol manager
type ProtocolManagerClientConfig struct {
	BaseURL        string               `json:"base_url,omitempty" yaml:"base_url"`
	TimeoutMs      int                  `json:"timeout_ms,omitempty" yaml:"timeout_ms"`
	CircuitBreaker CircuitBreakerConfig `json:"circuit_breaker,omitempty" yaml:"circuit_breaker"`
}
