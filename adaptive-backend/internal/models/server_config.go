package models

// ServerConfig holds server-specific configuration
type ServerConfig struct {
	Addr           string `json:"addr,omitempty" yaml:"addr"`
	AllowedOrigins string `json:"allowed_origins,omitempty" yaml:"allowed_origins"`
	Environment    string `json:"environment,omitempty" yaml:"environment"`
	LogLevel       string `json:"log_level,omitempty" yaml:"log_level"`
	JWTSecret      string `json:"jwt_secret,omitempty" yaml:"jwt_secret"`
}