package cache

import (
	"context"
	"fmt"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/utils"

	"github.com/botirk38/semanticcache"
	"github.com/botirk38/semanticcache/options"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/redis/go-redis/v9"
)

// GeminiPromptCache provides semantic caching for Gemini generate responses
type GeminiPromptCache struct {
	client            *redis.Client
	semanticCache     *semanticcache.SemanticCache[string, models.GeminiGenerateContentResponse]
	semanticThreshold float32
}

// NewGeminiPromptCache creates a new Gemini prompt cache instance with semantic caching support
func NewGeminiPromptCache(redisClient *redis.Client, config models.CacheConfig, redisURL string) (*GeminiPromptCache, error) {
	fiberlog.Info("GeminiPromptCache: Initializing with semantic cache support")

	pc := &GeminiPromptCache{
		client: redisClient,
	}

	// Initialize semantic cache if enabled and properly configured
	if config.Enabled && config.OpenAIAPIKey != "" {
		threshold := config.SemanticThreshold
		if threshold <= 0 || threshold > 1 {
			threshold = defaultSemanticThreshold
			fiberlog.Warnf("GeminiPromptCache: Invalid threshold value %.2f, using default %.2f", config.SemanticThreshold, defaultSemanticThreshold)
		}
		pc.semanticThreshold = float32(threshold)

		fiberlog.Debugf("GeminiPromptCache: Initializing semantic cache with threshold=%.2f", threshold)

		// Create semantic cache with new interface
		embedModel := config.EmbeddingModel
		if embedModel == "" {
			embedModel = "text-embedding-3-small"
		}
		semanticCache, err := semanticcache.New(
			options.WithOpenAIProvider[string, models.GeminiGenerateContentResponse](config.OpenAIAPIKey, embedModel),
			options.WithRedisBackend[string, models.GeminiGenerateContentResponse](redisURL, 2), // Use database 2 for Gemini
		)
		if err != nil {
			fiberlog.Errorf("GeminiPromptCache: Failed to create semantic cache: %v", err)
			return nil, fmt.Errorf("failed to create Gemini semantic cache: %w", err)
		}

		pc.semanticCache = semanticCache
		fiberlog.Info("GeminiPromptCache: Semantic cache initialized successfully")
	} else {
		fiberlog.Info("GeminiPromptCache: Semantic cache disabled, using basic Redis cache")
	}

	return pc, nil
}

// Get retrieves a cached Gemini response for the given request using semantic similarity
func (pc *GeminiPromptCache) Get(ctx context.Context, req *models.GeminiGenerateRequest, requestID string) (*models.GeminiGenerateContentResponse, string, bool) {
	if req.PromptCache == nil || !req.PromptCache.Enabled {
		fiberlog.Debugf("[%s] GeminiPromptCache: Cache disabled for request", requestID)
		return nil, "", false
	}

	if pc.semanticCache == nil {
		fiberlog.Debugf("[%s] GeminiPromptCache: Semantic cache not initialized", requestID)
		return nil, "", false
	}

	// Use threshold override if provided in request, otherwise use cache default
	if req.PromptCache.SemanticThreshold > 0 {
		fiberlog.Debugf("[%s] GeminiPromptCache: Using threshold override: %.2f", requestID, req.PromptCache.SemanticThreshold)
		return pc.getFromCacheWithThreshold(ctx, req, requestID, float32(req.PromptCache.SemanticThreshold))
	}
	return pc.getFromCache(ctx, req, requestID)
}

// getFromCache retrieves from semantic cache with similarity matching using default threshold
func (pc *GeminiPromptCache) getFromCache(ctx context.Context, req *models.GeminiGenerateRequest, requestID string) (*models.GeminiGenerateContentResponse, string, bool) {
	return pc.getFromCacheWithThreshold(ctx, req, requestID, pc.semanticThreshold)
}

// getFromCacheWithThreshold retrieves from semantic cache with similarity matching using custom threshold
func (pc *GeminiPromptCache) getFromCacheWithThreshold(ctx context.Context, req *models.GeminiGenerateRequest, requestID string, threshold float32) (*models.GeminiGenerateContentResponse, string, bool) {
	// Extract prompt from contents for semantic search
	prompt, err := utils.ExtractPromptFromGeminiContents(req.Contents)
	if err != nil {
		fiberlog.Debugf("[%s] GeminiPromptCache: %v", requestID, err)
		return nil, "", false
	}

	fiberlog.Debugf("[%s] GeminiPromptCache: Starting semantic cache lookup", requestID)

	// Try exact match first
	fiberlog.Debugf("[%s] GeminiPromptCache: Trying exact key match", requestID)
	if hit, found, err := pc.semanticCache.Get(ctx, prompt); found && err == nil {
		fiberlog.Infof("[%s] GeminiPromptCache: Exact cache hit", requestID)
		return &hit, "semantic_exact", true
	} else if err != nil {
		fiberlog.Errorf("[%s] GeminiPromptCache: Error during exact lookup: %v", requestID, err)
	}

	// Try semantic similarity search with provided threshold
	fiberlog.Debugf("[%s] GeminiPromptCache: Trying semantic similarity search (threshold: %.2f)", requestID, threshold)
	if match, err := pc.semanticCache.Lookup(ctx, prompt, threshold); err == nil && match != nil {
		fiberlog.Infof("[%s] GeminiPromptCache: Semantic cache hit", requestID)
		return &match.Value, "semantic_similar", true
	} else if err != nil {
		fiberlog.Errorf("[%s] GeminiPromptCache: Error during semantic lookup: %v", requestID, err)
	}

	fiberlog.Debugf("[%s] GeminiPromptCache: Semantic cache miss", requestID)
	return nil, "", false
}

// Set stores a response in the cache for future retrieval
func (pc *GeminiPromptCache) Set(ctx context.Context, req *models.GeminiGenerateRequest, response *models.GeminiGenerateContentResponse, requestID string) error {
	if req.PromptCache == nil || !req.PromptCache.Enabled {
		fiberlog.Debugf("[%s] GeminiPromptCache: Cache disabled, not storing", requestID)
		return nil
	}

	if pc.semanticCache == nil {
		fiberlog.Debugf("[%s] GeminiPromptCache: Semantic cache not initialized, not storing", requestID)
		return nil
	}

	return pc.setInCache(ctx, req, response, requestID)
}

// setInCache stores in semantic cache
func (pc *GeminiPromptCache) setInCache(ctx context.Context, req *models.GeminiGenerateRequest, response *models.GeminiGenerateContentResponse, requestID string) error {
	// Extract prompt from contents for semantic storage
	prompt, err := utils.ExtractPromptFromGeminiContents(req.Contents)
	if err != nil {
		fiberlog.Debugf("[%s] GeminiPromptCache: %v, skipping storage", requestID, err)
		return nil
	}

	fiberlog.Debugf("[%s] GeminiPromptCache: Storing response in semantic cache", requestID)
	err = pc.semanticCache.Set(ctx, prompt, prompt, *response)
	if err != nil {
		fiberlog.Errorf("[%s] GeminiPromptCache: Failed to store in semantic cache: %v", requestID, err)
		return fmt.Errorf("failed to store in semantic cache: %w", err)
	}

	fiberlog.Debugf("[%s] GeminiPromptCache: Successfully stored in semantic cache", requestID)
	return nil
}

// SetAsync stores a response asynchronously in the cache
func (pc *GeminiPromptCache) SetAsync(ctx context.Context, req *models.GeminiGenerateRequest, response *models.GeminiGenerateContentResponse, requestID string) <-chan error {
	errCh := make(chan error, 1)

	if req.PromptCache == nil || !req.PromptCache.Enabled {
		errCh <- nil
		close(errCh)
		return errCh
	}

	if pc.semanticCache == nil {
		errCh <- nil
		close(errCh)
		return errCh
	}

	prompt, err := utils.ExtractPromptFromGeminiContents(req.Contents)
	if err != nil {
		errCh <- nil
		close(errCh)
		return errCh
	}

	fiberlog.Debugf("[%s] GeminiPromptCache: Storing response in semantic cache (async)", requestID)
	return pc.semanticCache.SetAsync(ctx, prompt, prompt, *response)
}

// GetAsync retrieves a cached response asynchronously using semantic similarity
func (pc *GeminiPromptCache) GetAsync(ctx context.Context, req *models.GeminiGenerateRequest, requestID string) <-chan semanticcache.GetResult[models.GeminiGenerateContentResponse] {
	resultCh := make(chan semanticcache.GetResult[models.GeminiGenerateContentResponse], 1)

	if req.PromptCache == nil || !req.PromptCache.Enabled {
		resultCh <- semanticcache.GetResult[models.GeminiGenerateContentResponse]{Found: false}
		close(resultCh)
		return resultCh
	}

	if pc.semanticCache == nil {
		resultCh <- semanticcache.GetResult[models.GeminiGenerateContentResponse]{Found: false}
		close(resultCh)
		return resultCh
	}

	prompt, err := utils.ExtractPromptFromGeminiContents(req.Contents)
	if err != nil {
		resultCh <- semanticcache.GetResult[models.GeminiGenerateContentResponse]{Error: err}
		close(resultCh)
		return resultCh
	}

	fiberlog.Debugf("[%s] GeminiPromptCache: Getting from semantic cache (async)", requestID)
	return pc.semanticCache.GetAsync(ctx, prompt)
}

// LookupAsync performs semantic similarity lookup asynchronously
func (pc *GeminiPromptCache) LookupAsync(ctx context.Context, req *models.GeminiGenerateRequest, requestID string) <-chan semanticcache.LookupResult[models.GeminiGenerateContentResponse] {
	resultCh := make(chan semanticcache.LookupResult[models.GeminiGenerateContentResponse], 1)

	if req.PromptCache == nil || !req.PromptCache.Enabled {
		resultCh <- semanticcache.LookupResult[models.GeminiGenerateContentResponse]{Match: nil}
		close(resultCh)
		return resultCh
	}

	if pc.semanticCache == nil {
		resultCh <- semanticcache.LookupResult[models.GeminiGenerateContentResponse]{Match: nil}
		close(resultCh)
		return resultCh
	}

	prompt, err := utils.ExtractPromptFromGeminiContents(req.Contents)
	if err != nil {
		resultCh <- semanticcache.LookupResult[models.GeminiGenerateContentResponse]{Error: err}
		close(resultCh)
		return resultCh
	}

	threshold := pc.semanticThreshold
	if req.PromptCache.SemanticThreshold > 0 {
		threshold = float32(req.PromptCache.SemanticThreshold)
	}

	fiberlog.Debugf("[%s] GeminiPromptCache: Semantic lookup (async, threshold: %.2f)", requestID, threshold)
	return pc.semanticCache.LookupAsync(ctx, prompt, threshold)
}

// Flush clears all Gemini prompt cache entries
func (pc *GeminiPromptCache) Flush(ctx context.Context) error {
	if pc.semanticCache == nil {
		fiberlog.Debug("GeminiPromptCache: Semantic cache not initialized, nothing to flush")
		return nil
	}

	if err := pc.semanticCache.Flush(ctx); err != nil {
		fiberlog.Errorf("GeminiPromptCache: Failed to flush semantic cache: %v", err)
		return fmt.Errorf("failed to flush Gemini semantic cache: %w", err)
	}

	return nil
}

// Close closes the Redis connection and semantic cache
func (pc *GeminiPromptCache) Close() error {
	if pc.semanticCache != nil {
		if err := pc.semanticCache.Close(); err != nil {
			// Return first error encountered
			if pc.client != nil {
				_ = pc.client.Close() // Still attempt to close client
			}
			return err
		}
	}
	if pc.client != nil {
		return pc.client.Close()
	}
	return nil
}
