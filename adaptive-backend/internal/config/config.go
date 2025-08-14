package config

import (
	"adaptive-backend/internal/models"
	"fmt"
	"os"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"
)

// Config represents the complete application configuration
type Config struct {
	Server          ServerConfig                    `yaml:"server"`
	Providers       map[string]models.ProviderConfig `yaml:"providers"`
	Services        ServicesConfig                  `yaml:"services"`
	Fallback        FallbackConfig                  `yaml:"fallback"`
	PromptCache     PromptCacheConfig               `yaml:"prompt_cache"`
	ProtocolManager ProtocolManagerConfig           `yaml:"protocol_manager"`
}

// ServerConfig holds server-specific configuration
type ServerConfig struct {
	Addr           string `yaml:"addr"`
	AllowedOrigins string `yaml:"allowed_origins"`
	Environment    string `yaml:"environment"`
	LogLevel       string `yaml:"log_level"`
	JWTSecret      string `yaml:"jwt_secret"`
}


// ServicesConfig holds configuration for external services
type ServicesConfig struct {
	AdaptiveAI AdaptiveAIConfig `yaml:"adaptive_ai"`
	Redis      RedisConfig      `yaml:"redis"`
}

// AdaptiveAIConfig holds configuration for the Adaptive AI service
type AdaptiveAIConfig struct {
	BaseURL string `yaml:"base_url"`
}

// RedisConfig holds configuration for Redis
type RedisConfig struct {
	URL string `yaml:"url"`
}

// FallbackConfig holds configuration for fallback behavior
type FallbackConfig struct {
	Mode       string `yaml:"mode"`        // "race" or "sequential"
	TimeoutMs  int    `yaml:"timeout_ms"`
	MaxRetries int    `yaml:"max_retries"`
}

// SemanticCacheConfig holds configuration for semantic caching
type SemanticCacheConfig struct {
	Enabled      bool    `yaml:"enabled"`
	Threshold    float64 `yaml:"threshold"`
	RedisURL     string  `yaml:"redis_url"`
	OpenAIAPIKey string  `yaml:"openai_api_key"`
}

// PromptCacheConfig holds configuration for prompt caching
type PromptCacheConfig struct {
	Enabled           bool   `yaml:"enabled"`
	DefaultTTLSeconds int    `yaml:"default_ttl_seconds"`
	RedisURL          string `yaml:"redis_url"`
}

// ProtocolManagerConfig holds configuration for the protocol manager
type ProtocolManagerConfig struct {
	Cache  ProtocolManagerCacheConfig  `yaml:"cache"`
	Client ProtocolManagerClientConfig `yaml:"client"`
}

// ProtocolManagerCacheConfig holds cache configuration for protocol manager
type ProtocolManagerCacheConfig struct {
	Enabled       bool                `yaml:"enabled"`
	SemanticCache SemanticCacheConfig `yaml:"semantic_cache"`
}

// ProtocolManagerClientConfig holds client configuration for protocol manager
type ProtocolManagerClientConfig struct {
	BaseURL        string                           `yaml:"base_url"`
	TimeoutMs      int                              `yaml:"timeout_ms"`
	CircuitBreaker ProtocolManagerCircuitBreakerConfig `yaml:"circuit_breaker"`
}

// ProtocolManagerCircuitBreakerConfig holds circuit breaker configuration
type ProtocolManagerCircuitBreakerConfig struct {
	FailureThreshold int `yaml:"failure_threshold"`
	SuccessThreshold int `yaml:"success_threshold"`
	TimeoutMs        int `yaml:"timeout_ms"`
	ResetAfterMs     int `yaml:"reset_after_ms"`
}

// LoadFromFile loads configuration from a YAML file with environment variable substitution
func LoadFromFile(filepath string) (*Config, error) {
	// Read the file
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %w", filepath, err)
	}

	// Substitute environment variables
	content := substituteEnvVars(string(data))

	// Parse YAML
	var config Config
	if err := yaml.Unmarshal([]byte(content), &config); err != nil {
		return nil, fmt.Errorf("failed to parse YAML config: %w", err)
	}

	return &config, nil
}

// New creates a new Config instance by loading from the config file
func New() (*Config, error) {
	// Get config path from environment variable or use default
	configPath := os.Getenv("CONFIG_PATH")
	if configPath == "" {
		configPath = "config.yaml"
	}

	return LoadFromFile(configPath)
}

// substituteEnvVars replaces ${VAR_NAME} and ${VAR_NAME:-default} patterns with environment variables
func substituteEnvVars(content string) string {
	// Pattern matches ${VAR_NAME} or ${VAR_NAME:-default_value}
	re := regexp.MustCompile(`\$\{([^}:]+)(?::(-[^}]*))?\}`)
	
	return re.ReplaceAllStringFunc(content, func(match string) string {
		// Extract variable name and default value
		submatches := re.FindStringSubmatch(match)
		if len(submatches) < 2 {
			return match
		}
		
		varName := submatches[1]
		defaultValue := ""
		
		if len(submatches) > 2 && submatches[2] != "" {
			// Remove the leading '-' from default value
			defaultValue = strings.TrimPrefix(submatches[2], "-")
		}
		
		// Get environment variable value
		if value := os.Getenv(varName); value != "" {
			return value
		}
		
		return defaultValue
	})
}

// GetProviderAPIKey returns the API key for a specific provider
func (c *Config) GetProviderAPIKey(provider string) string {
	if providerConfig, exists := c.Providers[strings.ToLower(provider)]; exists {
		return providerConfig.APIKey
	}
	return ""
}

// GetEnabledProviders returns a map of enabled providers
func (c *Config) GetEnabledProviders() map[string]models.ProviderConfig {
	enabled := make(map[string]models.ProviderConfig)
	for name, config := range c.Providers {
		if config.Enabled {
			enabled[name] = config
		}
	}
	return enabled
}

// GetProviderConfig returns the configuration for a specific provider
func (c *Config) GetProviderConfig(provider string) (models.ProviderConfig, bool) {
	config, exists := c.Providers[strings.ToLower(provider)]
	return config, exists
}

// GetNormalizedLogLevel returns the log level in lowercase for consistent comparison
func (c *Config) GetNormalizedLogLevel() string {
	return strings.ToLower(c.Server.LogLevel)
}

// IsProduction returns true if the environment is production
func (c *Config) IsProduction() bool {
	return c.Server.Environment == "production"
}

// Validate checks if all required configuration values are set
func (c *Config) Validate() error {
	var missing []string

	if c.Server.Addr == "" {
		missing = append(missing, "server.addr")
	}
	if c.Server.AllowedOrigins == "" {
		missing = append(missing, "server.allowed_origins")
	}

	if len(missing) > 0 {
		return &ValidationError{MissingFields: missing}
	}

	return nil
}

// ValidationError represents configuration validation errors
type ValidationError struct {
	MissingFields []string
}

func (e *ValidationError) Error() string {
	return "missing required configuration fields: " + strings.Join(e.MissingFields, ", ")
}