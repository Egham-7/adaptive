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

type ConfigLoader struct {
	configPath string
	config     *Config
	mu         sync.RWMutex
}

func NewConfigLoader(configPath string) *ConfigLoader {
	return &ConfigLoader{configPath: configPath}
}

// doLoad reads, parses, and returns a fresh *Config (no locking here)
func (cl *ConfigLoader) doLoad() (*Config, error) {
	if _, err := os.Stat(cl.configPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("configuration file not found at %s", cl.configPath)
	}
	data, err := os.ReadFile(cl.configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %w", cl.configPath, err)
	}
	var raw map[string]any
	if err := yaml.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to parse YAML config: %w", err)
	}

	cfg := &Config{
		ModelCapabilities: make(map[string]models.ModelCapability),
		TaskModelMappings: make(map[string]models.TaskModelMapping),
		TaskParameters:    make(map[string]models.TaskParameters),
	}

	if capsRaw, ok := raw["model_capabilities"].(map[string]any); ok {
		for name, capRaw := range capsRaw {
			if m, ok := capRaw.(map[string]any); ok {
				c := models.ModelCapability{}
				if desc, ok := m["description"].(string); ok {
					c.Description = desc
				}
				if prov, ok := m["provider"].(string); ok {
					c.Provider = models.ProviderType(prov)
				}
				if cost, ok := m["cost_per_1k_tokens"].(float64); ok {
					c.CostPer1kTokens = cost
				}
				if mt, ok := m["max_tokens"].(int); ok {
					c.MaxTokens = mt
				}
				if s, ok := m["supports_streaming"].(bool); ok {
					c.SupportsStreaming = s
				}
				if f, ok := m["supports_function_calling"].(bool); ok {
					c.SupportsFunctionCalling = f
				}
				if v, ok := m["supports_vision"].(bool); ok {
					c.SupportsVision = v
				}
				cfg.ModelCapabilities[name] = c
			}
		}
	}

	if mapsRaw, ok := raw["task_model_mappings"].(map[string]any); ok {
		for t, mRaw := range mapsRaw {
			if mMap, ok := mRaw.(map[string]any); ok {
				tm := models.TaskModelMapping{}
				if e, ok := mMap["easy"].(map[string]any); ok {
					tm.Easy = parseDifficultyConfig(e)
				}
				if m, ok := mMap["medium"].(map[string]any); ok {
					tm.Medium = parseDifficultyConfig(m)
				}
				if h, ok := mMap["hard"].(map[string]any); ok {
					tm.Hard = parseDifficultyConfig(h)
				}
				cfg.TaskModelMappings[t] = tm
			}
		}
	}

	if paramsRaw, ok := raw["task_parameters"].(map[string]any); ok {
		for t, pRaw := range paramsRaw {
			if pMap, ok := pRaw.(map[string]any); ok {
				tp := models.TaskParameters{}
				if v, ok := pMap["temperature"].(float64); ok {
					tp.Temperature = v
				}
				if v, ok := pMap["top_p"].(float64); ok {
					tp.TopP = v
				}
				if v, ok := pMap["presence_penalty"].(float64); ok {
					tp.PresencePenalty = v
				}
				if v, ok := pMap["frequency_penalty"].(float64); ok {
					tp.FrequencyPenalty = v
				}
				if v, ok := pMap["max_completion_tokens"].(int); ok {
					tp.MaxCompletionTokens = v
				}
				if v, ok := pMap["n"].(int); ok {
					tp.N = v
				}
				cfg.TaskParameters[t] = tp
			}
		}
	}

	return cfg, nil
}

// LoadConfig acquires write-lock, loads once, caches, and returns the config
func (cl *ConfigLoader) LoadConfig() (*Config, error) {
	cl.mu.Lock()
	defer cl.mu.Unlock()
	if cl.config != nil {
		return cl.config, nil
	}
	cfg, err := cl.doLoad()
	if err != nil {
		return nil, err
	}
	cl.config = cfg
	return cfg, nil
}

// GetConfig implements double-checked locking to avoid races
func (cl *ConfigLoader) GetConfig() (*Config, error) {
	// 1) fast path under read-lock
	cl.mu.RLock()
	if cl.config != nil {
		cl.mu.RUnlock()
		return cl.config, nil
	}
	cl.mu.RUnlock()

	// 2) upgrade to write-lock and check again
	cl.mu.Lock()
	defer cl.mu.Unlock()
	if cl.config != nil {
		return cl.config, nil
	}
	cfg, err := cl.doLoad()
	if err != nil {
		return nil, err
	}
	cl.config = cfg
	return cfg, nil
}

func parseDifficultyConfig(raw map[string]any) models.DifficultyConfig {
	d := models.DifficultyConfig{}
	if m, ok := raw["model"].(string); ok {
		d.Model = m
	}
	if t, ok := raw["complexity_threshold"].(float64); ok {
		d.ComplexityThreshold = t
	}
	return d
}

func (cl *ConfigLoader) GetConfigPath() string {
	return cl.configPath
}

var (
	defaultConfigLoader *ConfigLoader
	configLoaderOnce    sync.Once
	configLoaderErr     error
)

// GetDefaultConfig returns a singleton-loaded *Config
func GetDefaultConfig() (*Config, error) {
	configLoaderOnce.Do(func() {
		const configFileName = "model_selection_config.yaml"
		cwd, err := os.Getwd()
		if err != nil {
			configLoaderErr = fmt.Errorf("failed to get working directory: %w", err)
			return
		}
		path := filepath.Join(cwd, "internal", "config", configFileName)
		defaultConfigLoader = NewConfigLoader(path)
	})
	if configLoaderErr != nil {
		return nil, configLoaderErr
	}
	return defaultConfigLoader.GetConfig()
}

// Config methods

func (c *Config) GetModelCapability(name string) (models.ModelCapability, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	cap, ok := c.ModelCapabilities[name]
	return cap, ok
}

func (c *Config) GetTaskModelMapping(
	t models.TaskType,
) (models.TaskModelMapping, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	m, ok := c.TaskModelMappings[string(t)]
	return m, ok
}

func (c *Config) GetTaskParameters(
	t models.TaskType,
) (models.TaskParameters, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	p, ok := c.TaskParameters[string(t)]
	return p, ok
}
