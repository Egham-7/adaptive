package models

// ServicesConfig holds configuration for external services
type ServicesConfig struct {
	ModelRouter ModelRouterConfig `json:"model_router" yaml:"model_router"`
	Redis       RedisConfig       `json:"redis" yaml:"redis"`
}

// ModelRouterConfig holds configuration for the Model Router service
type ModelRouterConfig struct {
	BaseURL string `json:"base_url,omitzero" yaml:"base_url"`
}

// RedisConfig holds configuration for Redis
type RedisConfig struct {
	URL string `json:"url,omitzero" yaml:"url"`
}
