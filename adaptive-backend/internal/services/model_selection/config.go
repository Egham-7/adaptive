package model_selection

import (
	"adaptive-backend/internal/models"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"gopkg.in/yaml.v3"
)

type Config struct {
	ModelCapabilities map[string]models.ModelCapability  `yaml:"model_capabilities"`
	TaskModelMappings map[string]models.TaskModelMapping `yaml:"task_model_mappings"`
	TaskParameters    map[string]models.TaskParameters   `yaml:"task_parameters"`
	mu                sync.RWMutex
}

// ConfigLoader handles loading and caching of configuration
type ConfigLoader struct {
	configPath string
	config     *Config
	mu         sync.RWMutex
}

// NewConfigLoader creates a new configuration loader
func NewConfigLoader(configPath string) *ConfigLoader {
	loader := &ConfigLoader{
		configPath: configPath,
	}
	return loader
}

// LoadConfig loads the configuration from the YAML file
func (cl *ConfigLoader) LoadConfig() (*Config, error) {
	cl.mu.Lock()
	defer cl.mu.Unlock()

	// Return cached config if already loaded
	if cl.config != nil {
		return cl.config, nil
	}

	// Check if config file exists
	if _, err := os.Stat(cl.configPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("configuration file not found at %s", cl.configPath)
	}

	// Read the YAML file
	data, err := os.ReadFile(cl.configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %w", cl.configPath, err)
	}

	// Parse the YAML
	var rawConfig map[string]any
	if err := yaml.Unmarshal(data, &rawConfig); err != nil {
		return nil, fmt.Errorf("failed to parse YAML config: %w", err)
	}

	// Create the config struct
	config := &Config{
		ModelCapabilities: make(map[string]models.ModelCapability),
		TaskModelMappings: make(map[string]models.TaskModelMapping),
		TaskParameters:    make(map[string]models.TaskParameters),
	}

	// Parse model capabilities
	if capsRaw, ok := rawConfig["model_capabilities"].(map[string]any); ok {
		for name, capRaw := range capsRaw {
			if capMap, ok := capRaw.(map[string]any); ok {
				c := models.ModelCapability{}
				if desc, ok := capMap["description"].(string); ok {
					c.Description = desc
				}
				if prov, ok := capMap["provider"].(string); ok {
					c.Provider = models.ProviderType(prov)
				}
				if cost, ok := capMap["cost_per_1k_tokens"].(float64); ok {
					c.CostPer1kTokens = cost
				}
				if mt, ok := capMap["max_tokens"].(int); ok {
					c.MaxTokens = mt
				}
				if s, ok := capMap["supports_streaming"].(bool); ok {
					c.SupportsStreaming = s
				}
				if f, ok := capMap["supports_function_calling"].(bool); ok {
					c.SupportsFunctionCalling = f
				}
				if v, ok := capMap["supports_vision"].(bool); ok {
					c.SupportsVision = v
				}
				config.ModelCapabilities[name] = c
			}
		}
	}

	// Parse task model mappings
	if mapsRaw, ok := rawConfig["task_model_mappings"].(map[string]any); ok {
		for t, mRaw := range mapsRaw {
			if mMap, ok := mRaw.(map[string]any); ok {
				m := models.TaskModelMapping{}
				if easyRaw, ok := mMap["easy"].(map[string]any); ok {
					m.Easy = parseDifficultyConfig(easyRaw)
				}
				if medRaw, ok := mMap["medium"].(map[string]any); ok {
					m.Medium = parseDifficultyConfig(medRaw)
				}
				if hardRaw, ok := mMap["hard"].(map[string]any); ok {
					m.Hard = parseDifficultyConfig(hardRaw)
				}
				config.TaskModelMappings[t] = m
			}
		}
	}

	// Parse task parameters
	if paramsRaw, ok := rawConfig["task_parameters"].(map[string]any); ok {
		for t, pRaw := range paramsRaw {
			if pMap, ok := pRaw.(map[string]any); ok {
				p := models.TaskParameters{}
				if tmp, ok := pMap["temperature"].(float64); ok {
					p.Temperature = tmp
				}
				if tp, ok := pMap["top_p"].(float64); ok {
					p.TopP = tp
				}
				if pr, ok := pMap["presence_penalty"].(float64); ok {
					p.PresencePenalty = pr
				}
				if fr, ok := pMap["frequency_penalty"].(float64); ok {
					p.FrequencyPenalty = fr
				}
				if mt, ok := pMap["max_completion_tokens"].(int); ok {
					p.MaxCompletionTokens = mt
				}
				if n, ok := pMap["n"].(int); ok {
					p.N = n
				}
				config.TaskParameters[t] = p
			}
		}
	}

	cl.config = config
	return config, nil
}

// parseDifficultyConfig parses a difficulty configuration from raw map
func parseDifficultyConfig(raw map[string]any) models.DifficultyConfig {
	d := models.DifficultyConfig{}
	if model, ok := raw["model"].(string); ok {
		d.Model = model
	}
	if th, ok := raw["complexity_threshold"].(float64); ok {
		d.ComplexityThreshold = th
	}
	return d
}

// GetConfig returns the loaded configuration, loading it if necessary
func (cl *ConfigLoader) GetConfig() (*Config, error) {
	cl.mu.RLock()
	if cl.config != nil {
		defer cl.mu.RUnlock()
		return cl.config, nil
	}
	cl.mu.RUnlock()
	return cl.LoadConfig()
}

// ReloadConfig forces a reload of the configuration
func (cl *ConfigLoader) ReloadConfig() (*Config, error) {
	cl.mu.Lock()
	cl.config = nil
	cl.mu.Unlock()
	return cl.LoadConfig()
}

// GetModelCapability returns capability information for a specific model
func (c *Config) GetModelCapability(name string) (models.ModelCapability, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	cap, ok := c.ModelCapabilities[name]
	return cap, ok
}

// GetTaskModelMapping returns model mapping for a specific task type
func (c *Config) GetTaskModelMapping(
	t models.TaskType,
) (models.TaskModelMapping, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	m, ok := c.TaskModelMappings[string(t)]
	return m, ok
}

// GetTaskParameters returns parameters for a specific task type
func (c *Config) GetTaskParameters(
	t models.TaskType,
) (models.TaskParameters, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	p, ok := c.TaskParameters[string(t)]
	return p, ok
}

// GetConfigPath returns the path to the configuration file
func (cl *ConfigLoader) GetConfigPath() string {
	return cl.configPath
}

var (
	defaultConfigLoader *ConfigLoader
	configLoaderOnce    sync.Once
)

var (
	defaultConfigLoader *ConfigLoader
	configLoaderOnce    sync.Once
	configLoaderErr     error
)

// GetDefaultConfig returns the default configuration instance
func GetDefaultConfig() (*Config, error) {
	configLoaderOnce.Do(func() {
		const configFileName = "model_selection_config.yaml"
		cwd, err := os.Getwd()
		if err != nil {
			configLoaderErr = fmt.Errorf("failed to get working directory: %w", err)
			return
		}
		configPath := filepath.Join(cwd, "internal", "config", configFileName)

		defaultConfigLoader = NewConfigLoader(configPath)
	})
	if configLoaderErr != nil {
		return nil, configLoaderErr
	}
	return defaultConfigLoader.GetConfig()
}
