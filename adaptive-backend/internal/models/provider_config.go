package models

// ProviderConfig holds configuration for LLM providers (unified for both YAML config and request overrides)
type ProviderConfig struct {
	APIKey         string                 `yaml:"api_key" json:"api_key,omitempty"`
	BaseURL        string                 `yaml:"base_url" json:"base_url,omitempty"`                 // Optional custom base URL
	AuthType       string                 `yaml:"auth_type" json:"auth_type,omitempty"`               // "bearer", "api_key", "basic", "custom"
	AuthHeaderName string                 `yaml:"auth_header_name" json:"auth_header_name,omitempty"` // Custom auth header name
	HealthEndpoint string                 `yaml:"health_endpoint" json:"health_endpoint,omitempty"`   // Health check endpoint
	RateLimitRpm   *int                   `yaml:"rate_limit_rpm" json:"rate_limit_rpm,omitempty"`     // Rate limit requests per minute
	TimeoutMs      int                    `yaml:"timeout_ms" json:"timeout_ms,omitempty"`             // Optional timeout in milliseconds
	RetryConfig    map[string]interface{} `yaml:"retry_config" json:"retry_config,omitempty"`         // Retry configuration
	Headers        map[string]string      `yaml:"headers" json:"headers,omitempty"`                   // Optional custom headers
}
