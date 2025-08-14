package models

// ProviderConfig holds configuration for LLM providers (unified for both YAML config and request overrides)
type ProviderConfig struct {
	APIKey    string            `yaml:"api_key" json:"api_key,omitempty"`
	Enabled   bool              `yaml:"enabled" json:"enabled"`
	Type      string            `yaml:"type" json:"type,omitempty"`             // "openai", "anthropic", "gemini"
	BaseURL   string            `yaml:"base_url" json:"base_url,omitempty"`     // Optional custom base URL
	Headers   map[string]string `yaml:"headers" json:"headers,omitempty"`       // Optional custom headers
	TimeoutMs int               `yaml:"timeout_ms" json:"timeout_ms,omitempty"` // Optional timeout in milliseconds
}
