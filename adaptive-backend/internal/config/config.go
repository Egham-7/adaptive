package config

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"adaptive-backend/internal/models"

	"github.com/joho/godotenv"
	"gopkg.in/yaml.v3"
)

const (
	defaultCostBiasFactor = 0.5
)

// Config represents the complete application configuration
type Config struct {
	Server      models.ServerConfig      `yaml:"server"`
	Endpoints   models.EndpointsConfig   `yaml:"endpoints"`
	Services    models.ServicesConfig    `yaml:"services"`
	Fallback    models.FallbackConfig    `yaml:"fallback"`
	PromptCache models.CacheConfig       `yaml:"prompt_cache"`
	ModelRouter models.ModelRouterConfig `yaml:"protocol_manager"`
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
	if config.Endpoints.ChatCompletions.Providers != nil {
		normalizedProviders := make(map[string]models.ProviderConfig, len(config.Endpoints.ChatCompletions.Providers))
		for key, value := range config.Endpoints.ChatCompletions.Providers {
			normalizedProviders[strings.ToLower(key)] = value
		}
		config.Endpoints.ChatCompletions.Providers = normalizedProviders
	}

	// Normalize provider map keys to lowercase for Messages endpoint too
	if config.Endpoints.Messages.Providers != nil {
		normalizedProviders := make(map[string]models.ProviderConfig, len(config.Endpoints.Messages.Providers))
		for key, value := range config.Endpoints.Messages.Providers {
			normalizedProviders[strings.ToLower(key)] = value
		}
		config.Endpoints.Messages.Providers = normalizedProviders
	}

	return &config, nil
}

// loadEnvFiles loads environment variables from .env files in order of precedence
// Loads .env.local first (highest priority), then .env.development, then .env
func loadEnvFiles() {
	// Define env files in order of precedence (first has highest priority)
	envFiles := []string{".env.local", ".env.development", ".env"}

	for _, envFile := range envFiles {
		if _, err := os.Stat(envFile); err == nil {
			// File exists, try to load it
			if err := godotenv.Load(envFile); err == nil {
				fmt.Printf("Loaded environment variables from %s\n", envFile)
			}
		}
	}
}

// New creates a new Config instance by loading from the config file
func New() (*Config, error) {
	// Load environment variables from .env files
	loadEnvFiles()

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

// GetProviderAPIKey returns the API key for a specific provider from the specified endpoint
func (c *Config) GetProviderAPIKey(provider string, endpoint string) string {
	var providers map[string]models.ProviderConfig
	switch endpoint {
	case "chat_completions":
		providers = c.Endpoints.ChatCompletions.Providers
	case "messages":
		providers = c.Endpoints.Messages.Providers
	default:
		return ""
	}

	if providerConfig, exists := providers[strings.ToLower(provider)]; exists {
		return providerConfig.APIKey
	}
	return ""
}

// GetProviders returns a map of all configured providers from the specified endpoint.
// If a provider is present in the config, it is considered enabled.
func (c *Config) GetProviders(endpoint string) map[string]models.ProviderConfig {
	switch endpoint {
	case "chat_completions":
		return c.Endpoints.ChatCompletions.Providers
	case "messages":
		return c.Endpoints.Messages.Providers
	default:
		return nil
	}
}

// GetProviderConfig returns the configuration for a specific provider from the specified endpoint
func (c *Config) GetProviderConfig(provider string, endpoint string) (models.ProviderConfig, bool) {
	var providers map[string]models.ProviderConfig
	switch endpoint {
	case "chat_completions":
		providers = c.Endpoints.ChatCompletions.Providers
	case "messages":
		providers = c.Endpoints.Messages.Providers
	default:
		return models.ProviderConfig{}, false
	}

	config, exists := providers[strings.ToLower(provider)]
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

// cloneStringAnyMap creates a deep copy of a map[string]any
func cloneStringAnyMap(src map[string]any) map[string]any {
	if src == nil {
		return nil
	}
	dst := make(map[string]any, len(src))
	maps.Copy(dst, src)
	return dst
}

// cloneStringStringMap creates a deep copy of a map[string]string
func cloneStringStringMap(src map[string]string) map[string]string {
	if src == nil {
		return nil
	}
	dst := make(map[string]string, len(src))
	maps.Copy(dst, src)
	return dst
}

// MergeProviderConfig merges YAML provider config with request override config.
// The request override takes precedence over YAML config for non-empty values.
func (c *Config) MergeProviderConfig(providerName string, override *models.ProviderConfig, endpoint string) (models.ProviderConfig, error) {
	// Get base config from YAML
	baseConfig, exists := c.GetProviderConfig(providerName, endpoint)
	if !exists {
		return models.ProviderConfig{}, fmt.Errorf("provider '%s' not found in YAML configuration for endpoint '%s'", providerName, endpoint)
	}

	// If no override provided, return base config
	if override == nil {
		return baseConfig, nil
	}

	// Create merged config with proper deep copy of struct value and map fields
	merged := models.ProviderConfig{
		APIKey:         baseConfig.APIKey,
		BaseURL:        baseConfig.BaseURL,
		AuthType:       baseConfig.AuthType,
		AuthHeaderName: baseConfig.AuthHeaderName,
		HealthEndpoint: baseConfig.HealthEndpoint,
		RateLimitRpm:   baseConfig.RateLimitRpm,
		TimeoutMs:      baseConfig.TimeoutMs,
		RetryConfig:    cloneStringAnyMap(baseConfig.RetryConfig),
		Headers:        cloneStringStringMap(baseConfig.Headers),
		NativeFormat:   baseConfig.NativeFormat,
	}

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
	if len(override.RetryConfig) > 0 {
		// Merge retry config into cloned map
		if merged.RetryConfig == nil {
			merged.RetryConfig = make(map[string]any)
		}
		maps.Copy(merged.RetryConfig, override.RetryConfig)
	}
	if len(override.Headers) > 0 {
		// Merge headers into cloned map
		if merged.Headers == nil {
			merged.Headers = make(map[string]string)
		}
		maps.Copy(merged.Headers, override.Headers)
	}

	return merged, nil
}

// MergeProviderConfigs merges YAML provider configs with a map of request override configs.
// Returns a map with all providers from YAML, with overrides applied where provided.
func (c *Config) MergeProviderConfigs(overrides map[string]*models.ProviderConfig, endpoint string) (map[string]models.ProviderConfig, error) {
	merged := make(map[string]models.ProviderConfig)

	// Get the base providers for the specified endpoint
	baseProviders := c.GetProviders(endpoint)
	if baseProviders == nil {
		return nil, fmt.Errorf("unsupported endpoint: %s", endpoint)
	}

	// Start with all YAML providers for the specified endpoint
	for providerName, yamlConfig := range baseProviders {
		if overrides != nil {
			if override, hasOverride := overrides[providerName]; hasOverride {
				mergedConfig, err := c.MergeProviderConfig(providerName, override, endpoint)
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

// MergePromptCacheConfig merges YAML prompt cache config with request override.
// Only TTL can be overridden in requests for security reasons.
func (c *Config) MergePromptCacheConfig(override *models.CacheConfig) *models.CacheConfig {
	// Start with YAML config (these fields are not overridable)
	merged := &models.CacheConfig{
		Enabled:           c.PromptCache.Enabled,
		DefaultTTLSeconds: c.PromptCache.DefaultTTLSeconds,
		RedisURL:          c.PromptCache.RedisURL,
		SemanticThreshold: c.PromptCache.SemanticThreshold,
		OpenAIAPIKey:      c.PromptCache.OpenAIAPIKey,
	}

	// Apply request override if provided (only TTL is allowed)
	if override != nil {
		if override.TTL > 0 {
			merged.TTL = override.TTL
		}
	}

	return merged
}

// MergeModelRouterConfig merges YAML model router config with request override.
// The request override takes precedence over YAML config for non-empty/non-nil values.
func (c *Config) MergeModelRouterConfig(override *models.ModelRouterConfig) *models.ModelRouterConfig {
	// Start with YAML defaults
	merged := &models.ModelRouterConfig{
		CostBias:      float32(defaultCostBiasFactor), // Default value
		SemanticCache: c.ModelRouter.SemanticCache,    // Copy YAML semantic cache config
		Client:        c.ModelRouter.Client,           // Copy YAML client config
	}

	// If no override provided, return defaults with YAML config
	if override == nil {
		return merged
	}

	// Apply request overrides
	if len(override.Models) > 0 {
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

	// Merge semantic cache config - request override takes precedence
	if override.SemanticCache.Enabled != c.ModelRouter.SemanticCache.Enabled ||
		override.SemanticCache.SemanticThreshold != c.ModelRouter.SemanticCache.SemanticThreshold {
		merged.SemanticCache = override.SemanticCache
	}

	return merged
}

// MergeFallbackConfig merges YAML fallback config with request override.
// The request override takes precedence over YAML config.
// Fallback is disabled by default (empty mode), enabled when mode is set.
func (c *Config) MergeFallbackConfig(override *models.FallbackConfig) *models.FallbackConfig {
	// Start with defaults (disabled by default)
	merged := &models.FallbackConfig{
		Mode: "", // Empty mode = disabled by default
	}

	// If no override provided, return defaults (disabled)
	if override == nil {
		return merged
	}

	// Apply request overrides
	if override.Mode != "" {
		merged.Mode = override.Mode // Set mode to enable fallback
	}
	if override.TimeoutMs > 0 {
		merged.TimeoutMs = override.TimeoutMs
	}
	if override.MaxRetries > 0 {
		merged.MaxRetries = override.MaxRetries
	}
	if override.CircuitBreaker != nil {
		merged.CircuitBreaker = override.CircuitBreaker
	}

	return merged
}

// ResolveConfig creates a resolved config by merging YAML config with all request overrides.
// Returns a new Config struct with all merged values as single source of truth.
func (c *Config) ResolveConfig(req *models.ChatCompletionRequest) (*Config, error) {
	// Create a copy of the original config
	resolved := &Config{
		Server:   c.Server,
		Services: c.Services,
	}

	// Merge all configs with request overrides
	resolved.PromptCache = *c.MergePromptCacheConfig(req.PromptCache)
	resolved.ModelRouter = *c.MergeModelRouterConfig(req.ModelRouterConfig)
	resolved.Fallback = *c.MergeFallbackConfig(req.Fallback)

	providers, err := c.MergeProviderConfigs(req.ProviderConfigs, "chat_completions")
	if err != nil {
		return nil, err
	}
	resolved.Endpoints.ChatCompletions.Providers = providers

	return resolved, nil
}

// ResolveConfigFromAnthropicRequest creates a resolved config by merging YAML config with Anthropic request overrides.
// Returns a new Config struct with all merged values as single source of truth.
func (c *Config) ResolveConfigFromAnthropicRequest(req *models.AnthropicMessageRequest) (*Config, error) {
	// Create a copy of the original config
	resolved := &Config{
		Server:   c.Server,
		Services: c.Services,
	}

	// Merge all configs with request overrides
	resolved.PromptCache = *c.MergePromptCacheConfig(req.PromptCache)
	resolved.ModelRouter = *c.MergeModelRouterConfig(req.ModelRouterConfig)
	resolved.Fallback = *c.MergeFallbackConfig(req.Fallback)

	providers, err := c.MergeProviderConfigs(req.ProviderConfigs, "messages")
	if err != nil {
		return nil, err
	}
	resolved.Endpoints.Messages.Providers = providers

	return resolved, nil
}

// ValidationError represents configuration validation errors
type ValidationError struct {
	MissingFields []string
}

func (e *ValidationError) Error() string {
	return "missing required configuration fields: " + strings.Join(e.MissingFields, ", ")
}
