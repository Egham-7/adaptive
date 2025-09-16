package models

// EndpointConfig holds endpoint-specific provider configurations
type EndpointConfig struct {
	Providers map[string]ProviderConfig `yaml:"providers"`
}

// EndpointsConfig holds all endpoint configurations
type EndpointsConfig struct {
	ChatCompletions EndpointConfig `yaml:"chat_completions"`
	Messages        EndpointConfig `yaml:"messages"`
	SelectModel     EndpointConfig `yaml:"select_model"`
}
