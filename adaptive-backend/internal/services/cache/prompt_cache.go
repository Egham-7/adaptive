package cache

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/redis/go-redis/v9"
)

const (
	promptCacheKeyPrefix = "prompt_cache:"
	defaultTTL           = 3600 // 1 hour in seconds
)

// PromptCache provides Redis-based caching for prompt responses
type PromptCache struct {
	client *redis.Client
}

// NewPromptCache creates a new prompt cache instance using Redis
func NewPromptCache(cfg *config.Config) (*PromptCache, error) {
	fiberlog.Info("PromptCache: Initializing Redis-based prompt cache")

	// Get Redis connection configuration with fallback hierarchy
	redisURL := cfg.PromptCache.RedisURL
	if redisURL == "" {
		redisURL = cfg.Services.Redis.URL
	}
	if redisURL == "" {
		fiberlog.Error("PromptCache: Redis URL not set in configuration (checked prompt_cache.redis_url and services.redis.url)")
		return nil, fmt.Errorf("redis URL not set in configuration")
	}
	fiberlog.Debugf("PromptCache: Redis URL configured: %s", redisURL)

	// Parse Redis URL
	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		fiberlog.Errorf("PromptCache: Failed to parse Redis URL: %v", err)
		return nil, fmt.Errorf("failed to parse Redis URL: %w", err)
	}

	// Create Redis client
	client := redis.NewClient(opt)

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		fiberlog.Errorf("PromptCache: Failed to connect to Redis: %v", err)
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	fiberlog.Info("PromptCache: Redis connection established successfully")

	return &PromptCache{
		client: client,
	}, nil
}

// generateCacheKey creates a consistent cache key from the request
func (pc *PromptCache) generateCacheKey(req *models.ChatCompletionRequest) string {
	// Create a deterministic key by hashing the request content
	keyData := models.PromptCacheKey{
		Messages:    req.Messages,
		Model:       req.Model,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		TopP:        req.TopP,
		Tools:       req.Tools,
		ToolChoice:  req.ToolChoice,
	}

	jsonData, _ := json.Marshal(keyData)
	hash := sha256.Sum256(jsonData)
	return promptCacheKeyPrefix + hex.EncodeToString(hash[:])
}

// Get retrieves a cached response for the given request
func (pc *PromptCache) Get(req *models.ChatCompletionRequest, requestID string) (*models.ChatCompletion, bool) {
	if req.PromptCache == nil || !req.PromptCache.Enabled {
		fiberlog.Debugf("[%s] PromptCache: Cache disabled for request", requestID)
		return nil, false
	}

	key := pc.generateCacheKey(req)
	fiberlog.Debugf("[%s] PromptCache: Looking up key: %s", requestID, key)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	data, err := pc.client.Get(ctx, key).Result()
	if err == redis.Nil {
		fiberlog.Debugf("[%s] PromptCache: Cache miss", requestID)
		return nil, false
	} else if err != nil {
		fiberlog.Errorf("[%s] PromptCache: Error retrieving from cache: %v", requestID, err)
		return nil, false
	}

	var response models.ChatCompletion
	if err := json.Unmarshal([]byte(data), &response); err != nil {
		fiberlog.Errorf("[%s] PromptCache: Error unmarshaling cached response: %v", requestID, err)
		return nil, false
	}

	fiberlog.Infof("[%s] PromptCache: Cache hit", requestID)
	return &response, true
}

// Set stores a response in the cache with the configured TTL
func (pc *PromptCache) Set(req *models.ChatCompletionRequest, response *models.ChatCompletion, requestID string) error {
	if req.PromptCache == nil || !req.PromptCache.Enabled {
		fiberlog.Debugf("[%s] PromptCache: Cache disabled, skipping storage", requestID)
		return nil
	}

	key := pc.generateCacheKey(req)

	// Determine TTL
	ttl := time.Duration(defaultTTL) * time.Second
	if req.PromptCache.TTL > 0 {
		ttl = time.Duration(req.PromptCache.TTL) * time.Second
	}

	fiberlog.Debugf("[%s] PromptCache: Storing response with TTL: %v", requestID, ttl)

	data, err := json.Marshal(response)
	if err != nil {
		fiberlog.Errorf("[%s] PromptCache: Error marshaling response: %v", requestID, err)
		return fmt.Errorf("failed to marshal response: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	if err := pc.client.Set(ctx, key, data, ttl).Err(); err != nil {
		fiberlog.Errorf("[%s] PromptCache: Error storing in cache: %v", requestID, err)
		return fmt.Errorf("failed to store in cache: %w", err)
	}

	fiberlog.Debugf("[%s] PromptCache: Successfully cached response", requestID)
	return nil
}

// Delete removes a cached response
func (pc *PromptCache) Delete(req *models.ChatCompletionRequest) error {
	key := pc.generateCacheKey(req)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	return pc.client.Del(ctx, key).Err()
}

// Flush clears all prompt cache entries
func (pc *PromptCache) Flush() error {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	keys, err := pc.client.Keys(ctx, promptCacheKeyPrefix+"*").Result()
	if err != nil {
		return err
	}

	if len(keys) > 0 {
		return pc.client.Del(ctx, keys...).Err()
	}

	return nil
}

// Close closes the Redis connection
func (pc *PromptCache) Close() error {
	if pc.client != nil {
		return pc.client.Close()
	}
	return nil
}

// Stats returns cache statistics
func (pc *PromptCache) Stats() (*models.PromptCacheStats, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	info, err := pc.client.Info(ctx, "keyspace").Result()
	if err != nil {
		return nil, err
	}

	keys, err := pc.client.Keys(ctx, promptCacheKeyPrefix+"*").Result()
	if err != nil {
		return nil, err
	}

	return &models.PromptCacheStats{
		PromptCacheKeys: len(keys),
		RedisInfo:       info,
		Timestamp:       time.Now(),
	}, nil
}
