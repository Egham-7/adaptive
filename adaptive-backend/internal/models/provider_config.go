package models

// ProviderConfig holds configuration for LLM providers (unified for both YAML config and request overrides)
type ProviderConfig struct {
	APIKey         string            `yaml:"api_key" json:"api_key,omitzero"`
	BaseURL        string            `yaml:"base_url" json:"base_url,omitzero"`                 // Optional custom base URL
	AuthType       string            `yaml:"auth_type" json:"auth_type,omitzero"`               // "bearer", "api_key", "basic", "custom"
	AuthHeaderName string            `yaml:"auth_header_name" json:"auth_header_name,omitzero"` // Custom auth header name
	HealthEndpoint string            `yaml:"health_endpoint" json:"health_endpoint,omitzero"`   // Health check endpoint
	RateLimitRpm   *int              `yaml:"rate_limit_rpm" json:"rate_limit_rpm,omitzero"`     // Rate limit requests per minute
	TimeoutMs      int               `yaml:"timeout_ms" json:"timeout_ms,omitzero"`             // Optional timeout in milliseconds
	RetryConfig    map[string]any    `yaml:"retry_config" json:"retry_config,omitzero"`         // Retry configuration
	Headers        map[string]string `yaml:"headers" json:"headers,omitzero"`                   // Optional custom headers
	NativeFormat   string            `yaml:"native_format" json:"native_format,omitzero"`       // Native API format: "openai" or "anthropic"
}
