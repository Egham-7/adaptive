package config

import (
	"adaptive-backend/internal/models"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"
)

const (
	defaultCostBiasFactor = 0.5
)

// Config represents the complete application configuration
type Config struct {
	Server          ServerConfig                     `yaml:"server"`
	Providers       map[string]models.ProviderConfig `yaml:"providers"`
	Services        ServicesConfig                   `yaml:"services"`
	Fallback        FallbackConfig                   `yaml:"fallback"`
	PromptCache     PromptCacheConfig                `yaml:"prompt_cache"`
	ProtocolManager ProtocolManagerConfig            `yaml:"protocol_manager"`
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
	Mode       string           `yaml:"mode"` // "race" or "sequential"
	TimeoutMs  int              `yaml:"timeout_ms"`
	MaxRetries int              `yaml:"max_retries"`
	WorkerPool WorkerPoolConfig `yaml:"worker_pool"`
}

// WorkerPoolConfig holds configuration for the worker pool
type WorkerPoolConfig struct {
	Workers   int `yaml:"workers"`    // Number of worker goroutines
	QueueSize int `yaml:"queue_size"` // Maximum number of queued tasks
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
	SemanticCache SemanticCacheConfig         `yaml:"semantic_cache"`
	Client        ProtocolManagerClientConfig `yaml:"client"`
}

// ProtocolManagerClientConfig holds client configuration for protocol manager
type ProtocolManagerClientConfig struct {
	BaseURL        string                              `yaml:"base_url"`
	TimeoutMs      int                                 `yaml:"timeout_ms"`
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
func LoadFromFile(configPath string) (*Config, error) {
	// Validate and clean the file path to prevent directory traversal
	cleanPath := filepath.Clean(configPath)

	// Ensure the path doesn't contain directory traversal attempts
	if strings.Contains(cleanPath, "..") {
		return nil, fmt.Errorf("invalid config path: path traversal not allowed")
	}

	// Restrict to certain file extensions for security
	ext := filepath.Ext(cleanPath)
	if ext != ".yaml" && ext != ".yml" {
		return nil, fmt.Errorf("invalid config file: only .yaml and .yml files are allowed")
	}

	// Read the file
	data, err := os.ReadFile(cleanPath) // #nosec G304 - path is validated above
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %w", cleanPath, err)
	}

	// Substitute environment variables
	content := substituteEnvVars(string(data))

	// Parse YAML
	var config Config
	if err := yaml.Unmarshal([]byte(content), &config); err != nil {
		return nil, fmt.Errorf("failed to parse YAML config: %w", err)
	}

	// Normalize provider map keys to lowercase for case-insensitive lookups
	if config.Providers != nil {
		normalizedProviders := make(map[string]models.ProviderConfig, len(config.Providers))
		for key, value := range config.Providers {
			normalizedProviders[strings.ToLower(key)] = value
		}
		config.Providers = normalizedProviders
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

// GetEnabledProviders returns a map of all configured providers (since all configured providers are considered enabled)
func (c *Config) GetEnabledProviders() map[string]models.ProviderConfig {
	// Since we removed the Enabled field, all configured providers are considered enabled
	return c.Providers
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

// MergeProviderConfig merges YAML provider config with request override config.
// The request override takes precedence over YAML config for non-empty values.
func (c *Config) MergeProviderConfig(providerName string, override *models.ProviderConfig) (models.ProviderConfig, error) {
	// Get base config from YAML
	baseConfig, exists := c.GetProviderConfig(providerName)
	if !exists {
		return models.ProviderConfig{}, fmt.Errorf("provider '%s' not found in YAML configuration", providerName)
	}

	// If no override provided, return base config
	if override == nil {
		return baseConfig, nil
	}

	// Create merged config starting with base
	merged := baseConfig

	// Override non-empty values from request
	if override.APIKey != "" {
		merged.APIKey = override.APIKey
	}
	if override.BaseURL != "" {
		merged.BaseURL = override.BaseURL
	}
	if override.AuthType != "" {
		merged.AuthType = override.AuthType
	}
	if override.AuthHeaderName != "" {
		merged.AuthHeaderName = override.AuthHeaderName
	}
	if override.HealthEndpoint != "" {
		merged.HealthEndpoint = override.HealthEndpoint
	}
	if override.RateLimitRpm != nil {
		merged.RateLimitRpm = override.RateLimitRpm
	}
	if override.TimeoutMs > 0 {
		merged.TimeoutMs = override.TimeoutMs
	}
	if override.RetryConfig != nil && len(override.RetryConfig) > 0 {
		// Merge retry config maps
		if merged.RetryConfig == nil {
			merged.RetryConfig = make(map[string]interface{})
		}
		for key, value := range override.RetryConfig {
			merged.RetryConfig[key] = value
		}
	}
	if override.Headers != nil && len(override.Headers) > 0 {
		// Merge headers maps
		if merged.Headers == nil {
			merged.Headers = make(map[string]string)
		}
		for key, value := range override.Headers {
			merged.Headers[key] = value
		}
	}

	return merged, nil
}

// MergeProviderConfigs merges YAML provider configs with a map of request override configs.
// Returns a map with all providers from YAML, with overrides applied where provided.
func (c *Config) MergeProviderConfigs(overrides map[string]*models.ProviderConfig) (map[string]models.ProviderConfig, error) {
	merged := make(map[string]models.ProviderConfig)

	// Start with all YAML providers
	for providerName, yamlConfig := range c.Providers {
		if overrides != nil {
			if override, hasOverride := overrides[providerName]; hasOverride {
				mergedConfig, err := c.MergeProviderConfig(providerName, override)
				if err != nil {
					return nil, fmt.Errorf("failed to merge config for provider '%s': %w", providerName, err)
				}
				merged[providerName] = mergedConfig
			} else {
				merged[providerName] = yamlConfig
			}
		} else {
			merged[providerName] = yamlConfig
		}
	}

	return merged, nil
}

// MergeProtocolManagerConfig merges YAML protocol manager config with request override.
// The request override takes precedence over YAML config for non-empty/non-nil values.
func (c *Config) MergeProtocolManagerConfig(override *models.ProtocolManagerConfig) *models.ProtocolManagerConfig {
	// Start with YAML defaults
	merged := &models.ProtocolManagerConfig{
		CostBias: float32(defaultCostBiasFactor), // Default value
	}

	// If no override provided, return defaults
	if override == nil {
		return merged
	}

	// Apply request overrides
	if override.Models != nil && len(override.Models) > 0 {
		merged.Models = override.Models
	}
	if override.CostBias > 0 {
		merged.CostBias = override.CostBias
	}
	if override.ComplexityThreshold != nil {
		merged.ComplexityThreshold = override.ComplexityThreshold
	}
	if override.TokenThreshold != nil {
		merged.TokenThreshold = override.TokenThreshold
	}

	return merged
}

// MergeFallbackConfig merges YAML fallback config with request override.
// The request override takes precedence over YAML config.
func (c *Config) MergeFallbackConfig(override *models.FallbackConfig) *models.FallbackConfig {
	// Start with YAML defaults
	merged := &models.FallbackConfig{
		Enabled: true,                          // Default enabled
		Mode:    models.FallbackModeParallel,   // Default parallel mode
	}

	// If no override provided, return defaults
	if override == nil {
		return merged
	}

	// Apply request overrides
	merged.Enabled = override.Enabled // Always use override value for boolean
	if override.Mode != "" {
		merged.Mode = override.Mode
	}

	return merged
}

// ValidationError represents configuration validation errors
type ValidationError struct {
	MissingFields []string
}

func (e *ValidationError) Error() string {
	return "missing required configuration fields: " + strings.Join(e.MissingFields, ", ")
}
