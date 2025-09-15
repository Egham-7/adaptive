package model_router

import (
	"context"
	"fmt"

	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"

	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/redis/go-redis/v9"
)

// ModelRouter coordinates protocol selection and caching for model selection.
type ModelRouter struct {
	cache  *ModelRouterCache
	client *ModelRouterClient
	cfg    *config.Config
}

// NewModelRouter creates a new ModelRouter with cache configuration.
func NewModelRouter(cfg *config.Config, redisClient *redis.Client) (*ModelRouter, error) {
	var cache *ModelRouterCache
	var err error

	// Get semantic cache configuration from config
	semanticCacheConfig := cfg.Services.ModelRouter.SemanticCache

	fiberlog.Infof("ModelRouter: Initializing with semantic_cache enabled=%t, threshold=%.2f",
		semanticCacheConfig.Enabled, semanticCacheConfig.SemanticThreshold)

	// Create cache only if enabled
	if semanticCacheConfig.Enabled {
		cache, err = NewModelRouterCache(cfg)
		if err != nil {
			fiberlog.Errorf("ModelRouter: Failed to create cache: %v", err)
			return nil, fmt.Errorf("failed to create protocol manager cache: %w", err)
		}
		fiberlog.Info("ModelRouter: Cache initialized successfully")
	} else {
		fiberlog.Warn("ModelRouter: Cache is disabled")
	}

	client := NewModelRouterClient(cfg, redisClient)
	fiberlog.Info("ModelRouter: Client initialized successfully")

	return &ModelRouter{
		cache:  cache,
		client: client,
		cfg:    cfg,
	}, nil
}

// SelectModelWithCache checks the semantic cache, then calls the Python service for model selection if needed.
// It returns the model selection response, the source (cache or service), and any error encountered.
// If cacheConfigOverride is provided, it will temporarily override the cache behavior for this request.
func (pm *ModelRouter) SelectModelWithCache(
	ctx context.Context,
	prompt string,
	userID, requestID string,
	modelRouterConfig *models.ModelRouterConfig,
	cbs map[string]*circuitbreaker.CircuitBreaker,
	tools any,
	toolCall any,
) (*models.ModelSelectionResponse, string, error) {
	fiberlog.Debugf("[%s] Starting model selection for user: %s", requestID, userID)

	cacheConfigOverride := modelRouterConfig.SemanticCache
	fiberlog.Debugf("[%s] Using merged model router config - cost bias: %.2f", requestID, modelRouterConfig.CostBias)

	fiberlog.Debugf("[%s] Extracted prompt for caching (length: %d chars)", requestID, len(prompt))

	// 1) Check if cache should be used (either default cache or override config)
	useCache := cacheConfigOverride.Enabled
	fiberlog.Debugf("[%s] Cache config override provided: enabled=%t, threshold=%.2f",
		requestID, cacheConfigOverride.Enabled, cacheConfigOverride.SemanticThreshold)

	if useCache && pm.cache != nil {
		fiberlog.Debugf("[%s] Checking cache for existing model response", requestID)
		// Use threshold override if provided, otherwise use cache default
		var hit *models.ModelSelectionResponse
		var source string
		var found bool
		if cacheConfigOverride.SemanticThreshold > 0 {
			fiberlog.Debugf("[%s] Using threshold override: %.2f", requestID, cacheConfigOverride.SemanticThreshold)
			hit, source, found = pm.cache.LookupWithThreshold(ctx, prompt, requestID, float32(cacheConfigOverride.SemanticThreshold))
		} else {
			hit, source, found = pm.cache.Lookup(ctx, prompt, requestID)
		}
		if found {
			fiberlog.Infof("[%s] Cache hit (%s) - returning cached model: %s/%s", requestID, source, hit.Provider, hit.Model)
			return hit, source, nil
		}
		fiberlog.Debugf("[%s] Cache miss - proceeding to model selection service", requestID)
	} else {
		if !cacheConfigOverride.Enabled {
			fiberlog.Debugf("[%s] Cache disabled by request override", requestID)
		} else {
			fiberlog.Debugf("[%s] Cache disabled - proceeding directly to model selection service", requestID)
		}
	}

	// 2) Call Python service for model selection
	fiberlog.Debugf("[%s] Calling model selection service", requestID)

	// Filter out providers with open circuit breakers if circuit breakers are available
	if cbs != nil && modelRouterConfig != nil {
		pm.filterUnavailableProviders(modelRouterConfig, cbs, requestID)
	}

	req := models.ModelSelectionRequest{
		Prompt:   prompt,
		ToolCall: toolCall,
		Tools:    tools,
		UserID:   userID,
		Models:   modelRouterConfig.Models,
		CostBias: &modelRouterConfig.CostBias,
	}
	resp := pm.client.SelectModel(ctx, req)

	fiberlog.Infof("[%s] Model selected: %s/%s",
		requestID, resp.Provider, resp.Model)

	return &resp, "", nil
}

// StoreSuccessfulModel stores a model response in the semantic cache after successful completion
func (pm *ModelRouter) StoreSuccessfulModel(
	ctx context.Context,
	prompt string,
	resp models.ModelSelectionResponse,
	requestID string,
	modelRouterConfig *models.ModelRouterConfig,
) error {
	// Check if cache should be used - default to using cache if available
	useCache := pm.cache != nil

	// Override with explicit config if provided
	if modelRouterConfig != nil {
		useCache = modelRouterConfig.SemanticCache.Enabled
	}

	if !useCache || pm.cache == nil {
		fiberlog.Debugf("[%s] Semantic cache disabled - skipping storage", requestID)
		return nil
	}

	fiberlog.Debugf("[%s] Storing successful model response in semantic cache", requestID)
	if err := pm.cache.Store(ctx, prompt, resp); err != nil {
		fiberlog.Errorf("[%s] Failed to store model response in semantic cache: %v", requestID, err)
		return err
	}

	fiberlog.Debugf("[%s] Successfully stored model response in semantic cache", requestID)
	return nil
}

// ValidateContext ensures dependencies are set.
func (pm *ModelRouter) ValidateContext() error {
	fiberlog.Debug("ModelRouter: Validating context and dependencies")

	if pm.client == nil {
		fiberlog.Error("ModelRouter: Protocol manager client is missing")
		return fmt.Errorf("protocol manager client is required")
	}

	if pm.cache != nil {
		fiberlog.Debug("ModelRouter: Cache is enabled and available")
	} else {
		fiberlog.Debug("ModelRouter: Cache is disabled")
	}

	fiberlog.Debug("ModelRouter: Context validation successful")
	return nil
}

// filterUnavailableProviders removes providers with open circuit breakers from the model list
func (pm *ModelRouter) filterUnavailableProviders(
	config *models.ModelRouterConfig,
	cbs map[string]*circuitbreaker.CircuitBreaker,
	requestID string,
) {
	if config == nil || config.Models == nil {
		return
	}

	originalCount := len(config.Models)
	availableModels := make([]models.ModelCapability, 0, len(config.Models))

	for _, model := range config.Models {
		providerName := model.Provider
		if cb, exists := cbs[providerName]; exists && !cb.CanExecute() {
			fiberlog.Warnf("[%s] Filtering out provider %s due to open circuit breaker", requestID, providerName)
			continue
		}
		availableModels = append(availableModels, model)
	}

	config.Models = availableModels
	if len(availableModels) < originalCount {
		fiberlog.Infof("[%s] Filtered providers: %d -> %d available models", requestID, originalCount, len(availableModels))
	}
}

// Close properly closes the protocol manager cache during shutdown
func (pm *ModelRouter) Close() error {
	fiberlog.Info("ModelRouter: Shutting down")

	if pm.cache != nil {
		fiberlog.Info("ModelRouter: Closing cache connection")
		if err := pm.cache.Close(); err != nil {
			fiberlog.Errorf("ModelRouter: Failed to close cache: %v", err)
			return err
		}
		fiberlog.Info("ModelRouter: Cache closed successfully")
	} else {
		fiberlog.Debug("ModelRouter: No cache to close (cache disabled)")
	}

	fiberlog.Info("ModelRouter: Shutdown completed")
	return nil
}
