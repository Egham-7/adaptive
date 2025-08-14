package protocol_manager

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"fmt"

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

// ProtocolManagerCache wraps the semanticcache library for protocol manager specific operations
type ProtocolManagerCache struct {
	cache             *semanticcache.SemanticCache[string, models.ProtocolResponse]
	semanticThreshold float32
}

// NewProtocolManagerCache creates a new protocol manager cache instance
func NewProtocolManagerCache(cfg *config.Config) (*ProtocolManagerCache, error) {
	fiberlog.Info("ProtocolManagerCache: Initializing cache")

	// Get semantic cache configuration
	semanticCacheConfig := cfg.ProtocolManager.SemanticCache

	fiberlog.Debugf("ProtocolManagerCache: Configuration - enabled=%t, threshold=%.2f",
		semanticCacheConfig.Enabled, semanticCacheConfig.Threshold)

	apiKey := semanticCacheConfig.OpenAIAPIKey
	if apiKey == "" {
		fiberlog.Error("ProtocolManagerCache: OpenAI API key not set in semantic cache configuration")
		return nil, fmt.Errorf("OpenAI API key not set in semantic cache configuration")
	}
	fiberlog.Debug("ProtocolManagerCache: OpenAI API key found")

	// Get Redis connection configuration
	redisURL := semanticCacheConfig.RedisURL
	if redisURL == "" {
		fiberlog.Error("ProtocolManagerCache: Redis URL not set in semantic cache configuration")
		return nil, fmt.Errorf("redis URL not set in semantic cache configuration")
	}
	fiberlog.Debug("ProtocolManagerCache: Redis URL configured")

	// Create Redis backend configuration
	config := types.BackendConfig{
		ConnectionString: redisURL,
	}

	// Create backend factory and Redis backend
	fiberlog.Debug("ProtocolManagerCache: Creating Redis backend")
	factory := &backends.BackendFactory[string, models.ProtocolResponse]{}
	backend, err := factory.NewBackend(types.BackendRedis, config)
	if err != nil {
		fiberlog.Errorf("ProtocolManagerCache: Failed to create Redis backend: %v", err)
		return nil, fmt.Errorf("failed to create Redis backend: %w", err)
	}
	fiberlog.Info("ProtocolManagerCache: Redis backend created successfully")

	// Create OpenAI provider
	fiberlog.Debug("ProtocolManagerCache: Creating OpenAI provider")
	provider, err := openai.NewOpenAIProvider(openai.OpenAIConfig{
		APIKey: apiKey,
		Model:  "text-embedding-3-small",
	})
	if err != nil {
		fiberlog.Errorf("ProtocolManagerCache: Failed to create OpenAI provider: %v", err)
		return nil, fmt.Errorf("failed to create OpenAI provider: %w", err)
	}
	fiberlog.Info("ProtocolManagerCache: OpenAI provider created successfully")

	// Create semantic cache
	fiberlog.Debug("ProtocolManagerCache: Creating semantic cache")
	cache, err := semanticcache.NewSemanticCache(backend, provider, nil)
	if err != nil {
		fiberlog.Errorf("ProtocolManagerCache: Failed to create semantic cache: %v", err)
		return nil, fmt.Errorf("failed to create semantic cache: %w", err)
	}
	fiberlog.Info("ProtocolManagerCache: Semantic cache created successfully")

	return &ProtocolManagerCache{
		cache:             cache,
		semanticThreshold: float32(semanticCacheConfig.Threshold),
	}, nil
}

// Lookup searches for a cached protocol response using exact match first, then semantic similarity
func (pmc *ProtocolManagerCache) Lookup(prompt, requestID string) (*models.ProtocolResponse, string, bool) {
	fiberlog.Debugf("[%s] ProtocolManagerCache: Starting cache lookup", requestID)

	// 1) First try exact key matching
	fiberlog.Debugf("[%s] ProtocolManagerCache: Trying exact key match", requestID)
	if hit, found := pmc.cache.Get(prompt); found {
		fiberlog.Infof("[%s] ProtocolManagerCache: Exact cache hit", requestID)
		return &hit, "semantic_exact", true
	}
	fiberlog.Debugf("[%s] ProtocolManagerCache: No exact match found", requestID)

	// 2) If no exact match, try semantic similarity search
	fiberlog.Debugf("[%s] ProtocolManagerCache: Trying semantic similarity search (threshold: %.2f)", requestID, pmc.semanticThreshold)
	if hit, found, err := pmc.cache.Lookup(prompt, pmc.semanticThreshold); err == nil && found {
		fiberlog.Infof("[%s] ProtocolManagerCache: Semantic cache hit", requestID)
		return &hit, "semantic_similar", true
	} else if err != nil {
		fiberlog.Errorf("[%s] ProtocolManagerCache: Error during semantic lookup: %v", requestID, err)
	} else {
		fiberlog.Debugf("[%s] ProtocolManagerCache: No semantic match found", requestID)
	}

	fiberlog.Debugf("[%s] ProtocolManagerCache: Cache miss", requestID)
	return nil, "", false
}

// Store saves a protocol response to the cache
func (pmc *ProtocolManagerCache) Store(prompt string, resp models.ProtocolResponse) error {
	fiberlog.Debugf("ProtocolManagerCache: Storing protocol response (protocol: %s)", resp.Protocol)
	err := pmc.cache.Set(prompt, prompt, resp)
	if err != nil {
		fiberlog.Errorf("ProtocolManagerCache: Failed to store in cache: %v", err)
	} else {
		fiberlog.Debugf("ProtocolManagerCache: Successfully stored protocol response")
	}
	return err
}

// Close properly closes the cache and releases resources
func (pmc *ProtocolManagerCache) Close() {
	if pmc.cache != nil {
		pmc.cache.Close()
	}
}

// Len returns the number of items in the cache
func (pmc *ProtocolManagerCache) Len() int {
	return pmc.cache.Len()
}

// Flush clears all entries from the cache
func (pmc *ProtocolManagerCache) Flush() error {
	return pmc.cache.Flush()
}
