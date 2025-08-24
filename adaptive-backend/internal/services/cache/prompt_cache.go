package cache

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/utils"
	"fmt"

	"github.com/botirk38/semanticcache"
	"github.com/botirk38/semanticcache/backends"
	"github.com/botirk38/semanticcache/providers/openai"
	"github.com/botirk38/semanticcache/types"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/redis/go-redis/v9"
)

const (
	defaultSemanticThreshold = 0.9
)

// PromptCache provides semantic caching for prompt responses
type PromptCache struct {
	client            *redis.Client
	semanticCache     *semanticcache.SemanticCache[string, models.ChatCompletion]
	semanticThreshold float32
}

// NewPromptCache creates a new prompt cache instance with semantic caching support
func NewPromptCache(redisClient *redis.Client, config models.PromptCacheConfig) (*PromptCache, error) {
	fiberlog.Info("PromptCache: Initializing with semantic cache support")

	pc := &PromptCache{
		client: redisClient,
	}

	// Initialize semantic cache if enabled and properly configured
	if config.Enabled && config.OpenAIAPIKey != "" && config.RedisURL != "" {
		threshold := config.SemanticThreshold
		if threshold <= 0 || threshold > 1 {
			threshold = defaultSemanticThreshold
			fiberlog.Warnf("PromptCache: Invalid threshold value %.2f, using default %.2f", config.SemanticThreshold, defaultSemanticThreshold)
		}
		pc.semanticThreshold = float32(threshold)

		fiberlog.Debugf("PromptCache: Initializing semantic cache with threshold=%.2f", threshold)

		// Create Redis backend configuration
		backendConfig := types.BackendConfig{
			ConnectionString: config.RedisURL,
		}

		// Create backend factory and Redis backend
		factory := &backends.BackendFactory[string, models.ChatCompletion]{}
		backend, err := factory.NewBackend(types.BackendRedis, backendConfig)
		if err != nil {
			fiberlog.Errorf("PromptCache: Failed to create Redis backend: %v", err)
			return nil, fmt.Errorf("failed to create Redis backend: %w", err)
		}

		// Create OpenAI provider
		provider, err := openai.NewOpenAIProvider(openai.OpenAIConfig{
			APIKey: config.OpenAIAPIKey,
			Model:  "text-embedding-3-small",
		})
		if err != nil {
			fiberlog.Errorf("PromptCache: Failed to create OpenAI provider: %v", err)
			return nil, fmt.Errorf("failed to create OpenAI provider: %w", err)
		}

		// Create semantic cache
		semanticCache, err := semanticcache.NewSemanticCache(backend, provider, nil)
		if err != nil {
			fiberlog.Errorf("PromptCache: Failed to create semantic cache: %v", err)
			return nil, fmt.Errorf("failed to create semantic cache: %w", err)
		}

		pc.semanticCache = semanticCache
		fiberlog.Info("PromptCache: Semantic cache initialized successfully")
	} else {
		fiberlog.Info("PromptCache: Semantic cache disabled, using basic Redis cache")
	}

	return pc, nil
}

// Get retrieves a cached response for the given request using semantic similarity
func (pc *PromptCache) Get(req *models.ChatCompletionRequest, requestID string) (*models.ChatCompletion, string, bool) {
	if req.PromptCache == nil || !req.PromptCache.Enabled {
		fiberlog.Debugf("[%s] PromptCache: Cache disabled for request", requestID)
		return nil, "", false
	}

	if pc.semanticCache == nil {
		fiberlog.Debugf("[%s] PromptCache: Semantic cache not initialized", requestID)
		return nil, "", false
	}

	return pc.getFromCache(req, requestID)
}

// getFromCache retrieves from semantic cache with similarity matching
func (pc *PromptCache) getFromCache(req *models.ChatCompletionRequest, requestID string) (*models.ChatCompletion, string, bool) {
	// Extract prompt from messages for semantic search
	prompt, err := utils.FindLastUserMessage(req.Messages)
	if err != nil {
		fiberlog.Debugf("[%s] PromptCache: %v", requestID, err)
		return nil, "", false
	}

	fiberlog.Debugf("[%s] PromptCache: Starting semantic cache lookup", requestID)

	// Try exact match first
	fiberlog.Debugf("[%s] PromptCache: Trying exact key match", requestID)
	if hit, found := pc.semanticCache.Get(prompt); found {
		fiberlog.Infof("[%s] PromptCache: Exact cache hit", requestID)
		return &hit, "semantic_exact", true
	}

	// Try semantic similarity search
	fiberlog.Debugf("[%s] PromptCache: Trying semantic similarity search (threshold: %.2f)", requestID, pc.semanticThreshold)
	if hit, found, err := pc.semanticCache.Lookup(prompt, pc.semanticThreshold); err == nil && found {
		fiberlog.Infof("[%s] PromptCache: Semantic cache hit", requestID)
		return &hit, "semantic_similar", true
	} else if err != nil {
		fiberlog.Errorf("[%s] PromptCache: Error during semantic lookup: %v", requestID, err)
	}

	fiberlog.Debugf("[%s] PromptCache: Semantic cache miss", requestID)
	return nil, "", false
}

// Set stores a response in the cache with the configured TTL
func (pc *PromptCache) Set(req *models.ChatCompletionRequest, response *models.ChatCompletion, requestID string) error {
	if req.PromptCache == nil || !req.PromptCache.Enabled {
		fiberlog.Debugf("[%s] PromptCache: Cache disabled, skipping storage", requestID)
		return nil
	}

	if pc.semanticCache == nil {
		fiberlog.Debugf("[%s] PromptCache: Semantic cache not initialized, skipping storage", requestID)
		return nil
	}

	return pc.setInCache(req, response, requestID)
}

// setInCache stores in semantic cache
func (pc *PromptCache) setInCache(req *models.ChatCompletionRequest, response *models.ChatCompletion, requestID string) error {
	// Extract prompt from messages for semantic storage
	prompt, err := utils.FindLastUserMessage(req.Messages)
	if err != nil {
		fiberlog.Debugf("[%s] PromptCache: %v, skipping storage", requestID, err)
		return nil
	}

	fiberlog.Debugf("[%s] PromptCache: Storing response in semantic cache", requestID)
	err = pc.semanticCache.Set(prompt, prompt, *response)
	if err != nil {
		fiberlog.Errorf("[%s] PromptCache: Failed to store in semantic cache: %v", requestID, err)
		return fmt.Errorf("failed to store in semantic cache: %w", err)
	}

	fiberlog.Debugf("[%s] PromptCache: Successfully stored in semantic cache", requestID)
	return nil
}

// Flush clears all prompt cache entries
func (pc *PromptCache) Flush() error {
	if pc.semanticCache == nil {
		fiberlog.Debug("PromptCache: Semantic cache not initialized, nothing to flush")
		return nil
	}

	if err := pc.semanticCache.Flush(); err != nil {
		fiberlog.Errorf("PromptCache: Failed to flush semantic cache: %v", err)
		return fmt.Errorf("failed to flush semantic cache: %w", err)
	}

	return nil
}

// Close closes the Redis connection and semantic cache
func (pc *PromptCache) Close() error {
	if pc.semanticCache != nil {
		pc.semanticCache.Close()
	}
	if pc.client != nil {
		return pc.client.Close()
	}
	return nil
}
