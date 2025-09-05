package model_router

import (
	"fmt"

	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"

	"github.com/botirk38/semanticcache"
	"github.com/botirk38/semanticcache/backends"
	"github.com/botirk38/semanticcache/providers/openai"
	"github.com/botirk38/semanticcache/types"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

const (
	defaultSemanticThreshold = 0.9
)

// DefaultCacheConfig returns default cache configuration
func DefaultCacheConfig() models.CacheConfig {
	return models.CacheConfig{
		Enabled:           true,
		SemanticThreshold: defaultSemanticThreshold,
	}
}

// ModelRouterCache wraps the semanticcache library for protocol manager specific operations
type ModelRouterCache struct {
	cache             *semanticcache.SemanticCache[string, models.ModelSelectionResponse]
	semanticThreshold float32
}

// NewModelRouterCache creates a new protocol manager cache instance
func NewModelRouterCache(cfg *config.Config) (*ModelRouterCache, error) {
	fiberlog.Info("ModelRouterCache: Initializing cache")

	// Get semantic cache configuration
	semanticCacheConfig := cfg.ModelRouter.SemanticCache

	// Validate and set default threshold if invalid
	threshold := semanticCacheConfig.SemanticThreshold
	if threshold <= 0 || threshold > 1 {
		threshold = 0.9 // Default to sane value
		fiberlog.Warnf("ModelRouterCache: Invalid threshold value %.2f, using default 0.9", semanticCacheConfig.SemanticThreshold)
	}

	fiberlog.Debugf("ModelRouterCache: Configuration - enabled=%t, threshold=%.2f",
		semanticCacheConfig.Enabled, threshold)

	apiKey := semanticCacheConfig.OpenAIAPIKey
	if apiKey == "" {
		fiberlog.Error("ModelRouterCache: OpenAI API key not set in semantic cache configuration")
		return nil, fmt.Errorf("OpenAI API key not set in semantic cache configuration")
	}
	fiberlog.Debug("ModelRouterCache: OpenAI API key found")

	// Get Redis connection configuration
	redisURL := semanticCacheConfig.RedisURL
	if redisURL == "" {
		fiberlog.Error("ModelRouterCache: Redis URL not set in semantic cache configuration")
		return nil, fmt.Errorf("redis URL not set in semantic cache configuration")
	}
	fiberlog.Debug("ModelRouterCache: Redis URL configured")

	// Create Redis backend configuration
	config := types.BackendConfig{
		ConnectionString: redisURL,
	}

	// Create backend factory and Redis backend
	fiberlog.Debug("ModelRouterCache: Creating Redis backend")
	factory := &backends.BackendFactory[string, models.ModelSelectionResponse]{}
	backend, err := factory.NewBackend(types.BackendRedis, config)
	if err != nil {
		fiberlog.Errorf("ModelRouterCache: Failed to create Redis backend: %v", err)
		return nil, fmt.Errorf("failed to create Redis backend: %w", err)
	}
	fiberlog.Info("ModelRouterCache: Redis backend created successfully")

	// Create OpenAI provider
	fiberlog.Debug("ModelRouterCache: Creating OpenAI provider")
	provider, err := openai.NewOpenAIProvider(openai.OpenAIConfig{
		APIKey: apiKey,
		Model:  "text-embedding-3-small",
	})
	if err != nil {
		fiberlog.Errorf("ModelRouterCache: Failed to create OpenAI provider: %v", err)
		return nil, fmt.Errorf("failed to create OpenAI provider: %w", err)
	}
	fiberlog.Info("ModelRouterCache: OpenAI provider created successfully")

	// Create semantic cache
	fiberlog.Debug("ModelRouterCache: Creating semantic cache")
	cache, err := semanticcache.NewSemanticCache(backend, provider, nil)
	if err != nil {
		fiberlog.Errorf("ModelRouterCache: Failed to create semantic cache: %v", err)
		return nil, fmt.Errorf("failed to create semantic cache: %w", err)
	}
	fiberlog.Info("ModelRouterCache: Semantic cache created successfully")

	return &ModelRouterCache{
		cache:             cache,
		semanticThreshold: float32(threshold),
	}, nil
}

// Lookup searches for a cached protocol response using exact match first, then semantic similarity
func (pmc *ModelRouterCache) Lookup(prompt, requestID string) (*models.ModelSelectionResponse, string, bool) {
	fiberlog.Debugf("[%s] ModelRouterCache: Starting cache lookup", requestID)

	// 1) First try exact key matching
	fiberlog.Debugf("[%s] ModelRouterCache: Trying exact key match", requestID)
	if hit, found := pmc.cache.Get(prompt); found {
		fiberlog.Infof("[%s] ModelRouterCache: Exact cache hit", requestID)
		return &hit, "semantic_exact", true
	}
	fiberlog.Debugf("[%s] ModelRouterCache: No exact match found", requestID)

	// 2) If no exact match, try semantic similarity search
	fiberlog.Debugf("[%s] ModelRouterCache: Trying semantic similarity search (threshold: %.2f)", requestID, pmc.semanticThreshold)
	if hit, found, err := pmc.cache.Lookup(prompt, pmc.semanticThreshold); err == nil && found {
		fiberlog.Infof("[%s] ModelRouterCache: Semantic cache hit", requestID)
		return &hit, "semantic_similar", true
	} else if err != nil {
		fiberlog.Errorf("[%s] ModelRouterCache: Error during semantic lookup: %v", requestID, err)
	} else {
		fiberlog.Debugf("[%s] ModelRouterCache: No semantic match found", requestID)
	}

	fiberlog.Debugf("[%s] ModelRouterCache: Cache miss", requestID)
	return nil, "", false
}

// Store saves a protocol response to the cache
func (pmc *ModelRouterCache) Store(prompt string, resp models.ModelSelectionResponse) error {
	fiberlog.Debugf("ModelRouterCache: Storing model response (model: %s/%s)", resp.Provider, resp.Model)
	err := pmc.cache.Set(prompt, prompt, resp)
	if err != nil {
		fiberlog.Errorf("ModelRouterCache: Failed to store in cache: %v", err)
	} else {
		fiberlog.Debugf("ModelRouterCache: Successfully stored protocol response")
	}
	return err
}

// Close properly closes the cache and releases resources
func (pmc *ModelRouterCache) Close() {
	if pmc.cache != nil {
		pmc.cache.Close()
	}
}

// Len returns the number of items in the cache
func (pmc *ModelRouterCache) Len() int {
	return pmc.cache.Len()
}

// Flush clears all entries from the cache
func (pmc *ModelRouterCache) Flush() error {
	return pmc.cache.Flush()
}
