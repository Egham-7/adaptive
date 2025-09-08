package models

// ServerConfig holds server-specific configuration
type ServerConfig struct {
	Port           string `json:"port,omitzero" yaml:"port"`
	AllowedOrigins string `json:"allowed_origins,omitzero" yaml:"allowed_origins"`
	Environment    string `json:"environment,omitzero" yaml:"environment"`
	LogLevel       string `json:"log_level,omitzero" yaml:"log_level"`
	JWTSecret      string `json:"jwt_secret,omitzero" yaml:"jwt_secret"`
}
