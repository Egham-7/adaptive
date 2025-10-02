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

// AnthropicPromptCache provides semantic caching for Anthropic message responses
type AnthropicPromptCache struct {
	client            *redis.Client
	semanticCache     *semanticcache.SemanticCache[string, models.AnthropicMessage]
	semanticThreshold float32
}

// NewAnthropicPromptCache creates a new Anthropic prompt cache instance with semantic caching support
func NewAnthropicPromptCache(redisClient *redis.Client, config models.CacheConfig, redisURL string) (*AnthropicPromptCache, error) {
	fiberlog.Info("AnthropicPromptCache: Initializing with semantic cache support")

	pc := &AnthropicPromptCache{
		client: redisClient,
	}

	// Initialize semantic cache if enabled and properly configured
	if config.Enabled && config.OpenAIAPIKey != "" {
		threshold := config.SemanticThreshold
		if threshold <= 0 || threshold > 1 {
			threshold = defaultSemanticThreshold
			fiberlog.Warnf("AnthropicPromptCache: Invalid threshold value %.2f, using default %.2f", config.SemanticThreshold, defaultSemanticThreshold)
		}
		pc.semanticThreshold = float32(threshold)

		fiberlog.Debugf("AnthropicPromptCache: Initializing semantic cache with threshold=%.2f", threshold)

		// Create semantic cache with new interface
		embedModel := config.EmbeddingModel
		if embedModel == "" {
			embedModel = "text-embedding-3-small"
		}
		semanticCache, err := semanticcache.New(
			options.WithOpenAIProvider[string, models.AnthropicMessage](config.OpenAIAPIKey, embedModel),
			options.WithRedisBackend[string, models.AnthropicMessage](redisURL, 1), // Use database 1 for Anthropic
		)
		if err != nil {
			fiberlog.Errorf("AnthropicPromptCache: Failed to create semantic cache: %v", err)
			return nil, fmt.Errorf("failed to create Anthropic semantic cache: %w", err)
		}

		pc.semanticCache = semanticCache
		fiberlog.Info("AnthropicPromptCache: Semantic cache initialized successfully")
	} else {
		fiberlog.Info("AnthropicPromptCache: Semantic cache disabled, using basic Redis cache")
	}

	return pc, nil
}

// Get retrieves a cached Anthropic message response for the given request using semantic similarity
func (pc *AnthropicPromptCache) Get(ctx context.Context, req *models.AnthropicMessageRequest, requestID string) (*models.AnthropicMessage, string, bool) {
	if req.PromptCache == nil || !req.PromptCache.Enabled {
		fiberlog.Debugf("[%s] AnthropicPromptCache: Cache disabled for request", requestID)
		return nil, "", false
	}

	if pc.semanticCache == nil {
		fiberlog.Debugf("[%s] AnthropicPromptCache: Semantic cache not initialized", requestID)
		return nil, "", false
	}

	// Use threshold override if provided in request, otherwise use cache default
	threshold := pc.semanticThreshold
	if req.PromptCache.SemanticThreshold > 0 {
		threshold = float32(req.PromptCache.SemanticThreshold)
		fiberlog.Debugf("[%s] AnthropicPromptCache: Using threshold override: %.2f", requestID, req.PromptCache.SemanticThreshold)
	}

	// Extract prompt from messages for semantic search
	prompt, err := utils.ExtractPromptFromAnthropicMessages(req.Messages)
	if err != nil {
		fiberlog.Debugf("[%s] AnthropicPromptCache: %v", requestID, err)
		return nil, "", false
	}

	fiberlog.Debugf("[%s] AnthropicPromptCache: Starting semantic cache lookup", requestID)

	// Try exact match first
	fiberlog.Debugf("[%s] AnthropicPromptCache: Trying exact key match", requestID)
	if hit, found, err := pc.semanticCache.Get(ctx, prompt); found && err == nil {
		fiberlog.Infof("[%s] AnthropicPromptCache: Exact cache hit", requestID)
		return &hit, "semantic_exact", true
	} else if err != nil {
		fiberlog.Errorf("[%s] AnthropicPromptCache: Error during exact lookup: %v", requestID, err)
	}

	// Try semantic similarity search with provided threshold
	fiberlog.Debugf("[%s] AnthropicPromptCache: Trying semantic similarity search (threshold: %.2f)", requestID, threshold)
	if match, err := pc.semanticCache.Lookup(ctx, prompt, threshold); err == nil && match != nil {
		fiberlog.Infof("[%s] AnthropicPromptCache: Semantic cache hit", requestID)
		return &match.Value, "semantic_similar", true
	} else if err != nil {
		fiberlog.Errorf("[%s] AnthropicPromptCache: Error during semantic lookup: %v", requestID, err)
	}

	fiberlog.Debugf("[%s] AnthropicPromptCache: Semantic cache miss", requestID)
	return nil, "", false
}

// Set stores a response in the cache for future retrieval
func (pc *AnthropicPromptCache) Set(ctx context.Context, req *models.AnthropicMessageRequest, response *models.AnthropicMessage, requestID string) error {
	if req.PromptCache == nil || !req.PromptCache.Enabled {
		fiberlog.Debugf("[%s] AnthropicPromptCache: Cache disabled, not storing", requestID)
		return nil
	}

	if pc.semanticCache == nil {
		fiberlog.Debugf("[%s] AnthropicPromptCache: Semantic cache not initialized, not storing", requestID)
		return nil
	}

	// Extract prompt from messages for semantic storage
	prompt, err := utils.ExtractPromptFromAnthropicMessages(req.Messages)
	if err != nil {
		fiberlog.Debugf("[%s] AnthropicPromptCache: %v, skipping storage", requestID, err)
		return nil
	}

	fiberlog.Debugf("[%s] AnthropicPromptCache: Storing response in semantic cache", requestID)
	err = pc.semanticCache.Set(ctx, prompt, prompt, *response)
	if err != nil {
		fiberlog.Errorf("[%s] AnthropicPromptCache: Failed to store in semantic cache: %v", requestID, err)
		return fmt.Errorf("failed to store in semantic cache: %w", err)
	}

	fiberlog.Debugf("[%s] AnthropicPromptCache: Successfully stored in semantic cache", requestID)
	return nil
}

// SetAsync stores a response asynchronously in the cache
func (pc *AnthropicPromptCache) SetAsync(ctx context.Context, req *models.AnthropicMessageRequest, response *models.AnthropicMessage, requestID string) <-chan error {
	if req.PromptCache == nil || !req.PromptCache.Enabled || pc.semanticCache == nil {
		errCh := make(chan error, 1)
		errCh <- nil
		close(errCh)
		return errCh
	}

	prompt, err := utils.ExtractPromptFromAnthropicMessages(req.Messages)
	if err != nil {
		errCh := make(chan error, 1)
		errCh <- nil
		close(errCh)
		return errCh
	}

	fiberlog.Debugf("[%s] AnthropicPromptCache: Storing response in semantic cache (async)", requestID)
	return pc.semanticCache.SetAsync(ctx, prompt, prompt, *response)
}

// GetAsync retrieves a cached response asynchronously using semantic similarity
func (pc *AnthropicPromptCache) GetAsync(ctx context.Context, req *models.AnthropicMessageRequest, requestID string) <-chan semanticcache.GetResult[models.AnthropicMessage] {
	if req.PromptCache == nil || !req.PromptCache.Enabled || pc.semanticCache == nil {
		resultCh := make(chan semanticcache.GetResult[models.AnthropicMessage], 1)
		resultCh <- semanticcache.GetResult[models.AnthropicMessage]{Found: false}
		close(resultCh)
		return resultCh
	}

	prompt, err := utils.ExtractPromptFromAnthropicMessages(req.Messages)
	if err != nil {
		resultCh := make(chan semanticcache.GetResult[models.AnthropicMessage], 1)
		resultCh <- semanticcache.GetResult[models.AnthropicMessage]{Error: err}
		close(resultCh)
		return resultCh
	}

	fiberlog.Debugf("[%s] AnthropicPromptCache: Getting from semantic cache (async)", requestID)
	return pc.semanticCache.GetAsync(ctx, prompt)
}

// LookupAsync performs semantic similarity lookup asynchronously
func (pc *AnthropicPromptCache) LookupAsync(ctx context.Context, req *models.AnthropicMessageRequest, requestID string) <-chan semanticcache.LookupResult[models.AnthropicMessage] {
	if req.PromptCache == nil || !req.PromptCache.Enabled || pc.semanticCache == nil {
		resultCh := make(chan semanticcache.LookupResult[models.AnthropicMessage], 1)
		resultCh <- semanticcache.LookupResult[models.AnthropicMessage]{Match: nil}
		close(resultCh)
		return resultCh
	}

	prompt, err := utils.ExtractPromptFromAnthropicMessages(req.Messages)
	if err != nil {
		resultCh := make(chan semanticcache.LookupResult[models.AnthropicMessage], 1)
		resultCh <- semanticcache.LookupResult[models.AnthropicMessage]{Error: err}
		close(resultCh)
		return resultCh
	}

	threshold := pc.semanticThreshold
	if req.PromptCache.SemanticThreshold > 0 {
		threshold = float32(req.PromptCache.SemanticThreshold)
	}

	fiberlog.Debugf("[%s] AnthropicPromptCache: Semantic lookup (async, threshold: %.2f)", requestID, threshold)
	return pc.semanticCache.LookupAsync(ctx, prompt, threshold)
}

// Flush clears all Anthropic prompt cache entries
func (pc *AnthropicPromptCache) Flush(ctx context.Context) error {
	if pc.semanticCache == nil {
		fiberlog.Debug("AnthropicPromptCache: Semantic cache not initialized, nothing to flush")
		return nil
	}

	if err := pc.semanticCache.Flush(ctx); err != nil {
		fiberlog.Errorf("AnthropicPromptCache: Failed to flush semantic cache: %v", err)
		return fmt.Errorf("failed to flush Anthropic semantic cache: %w", err)
	}

	return nil
}

// Close closes the Redis connection and semantic cache
func (pc *AnthropicPromptCache) Close() error {
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
