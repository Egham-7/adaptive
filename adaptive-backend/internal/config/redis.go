package config

import (
	"context"
	"crypto/tls"
	"fmt"
	"os"
	"strconv"
	"time"

	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/redis/go-redis/v9"
)

const (
	defaultRedisAddr         = "localhost:6379"
	defaultRedisPassword     = ""
	defaultRedisDB           = 0
	defaultPoolSize          = 10
	defaultConnTimeout       = 5 * time.Second
	defaultReadTimeout       = 3 * time.Second
	defaultWriteTimeout      = 3 * time.Second
	defaultIdleTimeout       = 5 * time.Minute
	defaultMinIdleConns      = 2
	defaultMaxRetries        = 3
	defaultTLSEnabled        = false
	defaultTLSSkipVerify     = false
)

// RedisConfig holds Redis connection configuration
type RedisConfig struct {
	Addr         string
	Password     string
	DB           int
	PoolSize     int
	ConnTimeout  time.Duration
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
	IdleTimeout  time.Duration
	MinIdleConns int
	MaxRetries   int
	TLSEnabled   bool
	TLSSkipVerify bool
	TLSConfig    *tls.Config
}

// NewRedisConfig creates a new Redis configuration from environment variables
func NewRedisConfig() (*RedisConfig, error) {
	tlsEnabled := getBoolEnvOrDefault("REDIS_TLS_ENABLED", defaultTLSEnabled)
	tlsSkipVerify := getBoolEnvOrDefault("REDIS_TLS_SKIP_VERIFY", defaultTLSSkipVerify)
	
	config := &RedisConfig{
		Addr:          getEnvOrDefault("REDIS_ADDR", defaultRedisAddr),
		Password:      getEnvOrDefault("REDIS_PASSWORD", defaultRedisPassword),
		DB:            0, // Default DB
		PoolSize:      defaultPoolSize,
		ConnTimeout:   defaultConnTimeout,
		ReadTimeout:   defaultReadTimeout,
		WriteTimeout:  defaultWriteTimeout,
		IdleTimeout:   defaultIdleTimeout,
		MinIdleConns:  defaultMinIdleConns,
		MaxRetries:    defaultMaxRetries,
		TLSEnabled:    tlsEnabled,
		TLSSkipVerify: tlsSkipVerify,
	}

	// Configure TLS if enabled
	if tlsEnabled {
		config.TLSConfig = &tls.Config{
			InsecureSkipVerify: tlsSkipVerify,
		}
	}

	// Support REDIS_URL format (redis://password@host:port/db)
	if redisURL := os.Getenv("REDIS_URL"); redisURL != "" {
		// Parse Redis URL if provided
		opt, err := redis.ParseURL(redisURL)
		if err != nil {
			return nil, fmt.Errorf("failed to parse REDIS_URL: %w", err)
		}
		config.Addr = opt.Addr
		config.Password = opt.Password
		config.DB = opt.DB
	}

	return config, nil
}

// NewRedisClient creates a new Redis client with the given configuration
func NewRedisClient(config *RedisConfig) (*redis.Client, error) {
	options := &redis.Options{
		Addr:            config.Addr,
		Password:        config.Password,
		DB:              config.DB,
		PoolSize:        config.PoolSize,
		DialTimeout:     config.ConnTimeout,
		ReadTimeout:     config.ReadTimeout,
		WriteTimeout:    config.WriteTimeout,
		ConnMaxIdleTime: config.IdleTimeout,

		// Connection pool settings
		MinIdleConns: config.MinIdleConns,
		MaxRetries:   config.MaxRetries,

		// Health check
		OnConnect: func(ctx context.Context, cn *redis.Conn) error {
			fiberlog.Info("Redis connection established")
			return nil
		},
	}

	// Configure TLS if enabled
	if config.TLSEnabled && config.TLSConfig != nil {
		options.TLSConfig = config.TLSConfig
		fiberlog.Info("Redis TLS configuration enabled")
	}

	client := redis.NewClient(options)

	// Test the connection
	ctx, cancel := context.WithTimeout(context.Background(), config.ConnTimeout)
	defer cancel()

	pong, err := client.Ping(ctx).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	fiberlog.Infof("Redis connection successful: %s", pong)
	return client, nil
}

// NewRedisClientFromEnv creates a Redis client using environment variables
func NewRedisClientFromEnv() (*redis.Client, error) {
	config, err := NewRedisConfig()
	if err != nil {
		return nil, err
	}
	return NewRedisClient(config)
}

// getEnvOrDefault returns the environment variable value or a default value
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// getBoolEnvOrDefault returns the environment variable value as a boolean or a default value
func getBoolEnvOrDefault(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.ParseBool(value); err == nil {
			return parsed
		}
	}
	return defaultValue
}

// HealthCheck performs a health check on the Redis connection
func HealthCheck(client *redis.Client) error {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_, err := client.Ping(ctx).Result()
	if err != nil {
		return fmt.Errorf("redis health check failed: %w", err)
	}
	return nil
}

// GracefulShutdown closes the Redis client connection
func GracefulShutdown(client *redis.Client) error {
	if client != nil {
		return client.Close()
	}
	return nil
}
