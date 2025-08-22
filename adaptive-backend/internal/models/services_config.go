package models

// ServicesConfig holds configuration for external services
type ServicesConfig struct {
	AdaptiveAI AdaptiveAIConfig `json:"adaptive_ai" yaml:"adaptive_ai"`
	Redis      RedisConfig      `json:"redis" yaml:"redis"`
}

// AdaptiveAIConfig holds configuration for the Adaptive AI service
type AdaptiveAIConfig struct {
	BaseURL string `json:"base_url,omitempty" yaml:"base_url"`
}

// RedisConfig holds configuration for Redis
type RedisConfig struct {
	URL string `json:"url,omitempty" yaml:"url"`
}
