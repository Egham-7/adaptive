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
	semanticCacheConfig := cfg.Services.ModelRouter.SemanticCache

	fiberlog.Infof("ModelRouter: Initializing with semantic_cache enabled=%t, threshold=%.2f",
		semanticCacheConfig.Enabled, semanticCacheConfig.SemanticThreshold)

	// Create cache only if enabled
	cache, err := createCacheIfEnabled(cfg, semanticCacheConfig)
	if err != nil {
		return nil, err
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
	fiberlog.Infof("[%s] ═══ Model Selection Started ═══", requestID)
	fiberlog.Infof("[%s] User: %s | Prompt length: %d chars | Cost bias: %.2f",
		requestID, userID, len(prompt), modelRouterConfig.CostBias)

	cacheConfigOverride := modelRouterConfig.SemanticCache

	// 1) Check if cache should be used (either default cache or override config)
	useCache := cacheConfigOverride.Enabled
	if useCache && pm.cache != nil {
		fiberlog.Infof("[%s] 🔍 Cache enabled - checking semantic cache (threshold: %.2f)",
			requestID, cacheConfigOverride.SemanticThreshold)

		cacheResult := pm.lookupCache(ctx, prompt, requestID, cacheConfigOverride, cbs)
		if cacheResult.Hit {
			fiberlog.Infof("[%s] ✅ CACHE HIT (%s) - serving from cache: %s/%s",
				requestID, cacheResult.Source, cacheResult.Response.Provider, cacheResult.Response.Model)
			fiberlog.Infof("[%s] ═══ Model Selection Complete (Cache) ═══", requestID)
			return cacheResult.Response, cacheResult.Source, nil
		}
		fiberlog.Infof("[%s] ❌ Cache miss - proceeding to AI service", requestID)
	} else {
		if !cacheConfigOverride.Enabled {
			fiberlog.Infof("[%s] ⚠️  Cache disabled by request config", requestID)
		} else {
			fiberlog.Infof("[%s] ⚠️  Cache not initialized - bypassing", requestID)
		}
	}

	// 2) Call Python service for model selection
	fiberlog.Infof("[%s] 🤖 Calling AI model selection service", requestID)

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

	fiberlog.Infof("[%s] ✅ AI service selected: %s/%s",
		requestID, resp.Provider, resp.Model)
	fiberlog.Infof("[%s] ═══ Model Selection Complete (AI Service) ═══", requestID)

	return &resp, "", nil
}

// StoreSuccessfulModel stores a model response in the semantic cache (fire-and-forget)
func (pm *ModelRouter) StoreSuccessfulModel(
	ctx context.Context,
	prompt string,
	resp models.ModelSelectionResponse,
	requestID string,
	modelRouterConfig *models.ModelRouterConfig,
) error {
	if pm.cache != nil && (modelRouterConfig == nil || modelRouterConfig.SemanticCache.Enabled) {
		fiberlog.Infof("[%s] 💾 Storing successful response in cache: %s/%s",
			requestID, resp.Provider, resp.Model)
		pm.cache.StoreAsync(ctx, prompt, resp, requestID)
	} else {
		fiberlog.Debugf("[%s] ⏭️  Skipping cache storage (cache disabled or unavailable)", requestID)
	}
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
	filteredProviders := []string{}

	for _, model := range config.Models {
		providerName := model.Provider
		if cb, exists := cbs[providerName]; exists && !cb.CanExecute() {
			fiberlog.Warnf("[%s] 🚫 Filtering out provider %s/%s (circuit breaker open)",
				requestID, providerName, model.ModelName)
			filteredProviders = append(filteredProviders, providerName)
			continue
		}
		availableModels = append(availableModels, model)
	}

	config.Models = availableModels
	if len(availableModels) < originalCount {
		fiberlog.Warnf("[%s] ⚠️  Provider filtering: %d -> %d models (filtered: %v)",
			requestID, originalCount, len(availableModels), filteredProviders)
	} else {
		fiberlog.Debugf("[%s] All %d providers available", requestID, originalCount)
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

// createCacheIfEnabled creates a cache if semantic caching is enabled
func createCacheIfEnabled(cfg *config.Config, semanticCacheConfig models.CacheConfig) (*ModelRouterCache, error) {
	if !semanticCacheConfig.Enabled {
		fiberlog.Warn("ModelRouter: Cache is disabled")
		return nil, nil
	}

	cache, err := NewModelRouterCache(cfg)
	if err != nil {
		fiberlog.Errorf("ModelRouter: Failed to create cache: %v", err)
		return nil, fmt.Errorf("failed to create protocol manager cache: %w", err)
	}

	fiberlog.Info("ModelRouter: Cache initialized successfully")
	return cache, nil
}

// lookupCache performs cache lookup with circuit breaker validation (synchronous reads)
func (pm *ModelRouter) lookupCache(ctx context.Context, prompt, requestID string, cacheConfig models.CacheConfig, cbs map[string]*circuitbreaker.CircuitBreaker) models.CacheResult {
	threshold := pm.cache.semanticThreshold
	if cacheConfig.SemanticThreshold > 0 {
		threshold = float32(cacheConfig.SemanticThreshold)
		fiberlog.Infof("[%s] Using custom semantic threshold: %.2f (default: %.2f)",
			requestID, cacheConfig.SemanticThreshold, pm.cache.semanticThreshold)
	}

	fiberlog.Debugf("[%s] Performing cache lookup with threshold: %.2f", requestID, threshold)
	cachedResponse, source, found := pm.cache.Lookup(ctx, prompt, requestID, threshold)
	if !found {
		fiberlog.Debugf("[%s] No matching entry found in cache", requestID)
		return models.CacheResult{Hit: false}
	}

	fiberlog.Infof("[%s] Found cache entry from %s: %s/%s",
		requestID, source, cachedResponse.Provider, cachedResponse.Model)

	validResponse := pm.selectAvailableModel(cachedResponse, cbs, requestID)
	if validResponse == nil {
		fiberlog.Warnf("[%s] ⚠️  All cached models unavailable (circuit breakers open) - invalidating cache entry",
			requestID)
		pm.cache.DeleteAsync(ctx, prompt, cachedResponse.Provider, requestID)
		return models.CacheResult{Hit: false}
	}

	if validResponse.Provider != cachedResponse.Provider || validResponse.Model != cachedResponse.Model {
		fiberlog.Infof("[%s] Using alternative from cache: %s/%s (original unavailable)",
			requestID, validResponse.Provider, validResponse.Model)
	}

	return models.CacheResult{
		Response: validResponse,
		Source:   source,
		Hit:      true,
	}
}

// selectAvailableModel finds the first available model from cached response considering circuit breakers
func (pm *ModelRouter) selectAvailableModel(cachedResponse *models.ModelSelectionResponse, cbs map[string]*circuitbreaker.CircuitBreaker, requestID string) *models.ModelSelectionResponse {
	if cachedResponse == nil {
		return nil
	}

	// Build a list of all potential models (primary + alternatives)
	candidates := []models.Alternative{
		{Provider: cachedResponse.Provider, Model: cachedResponse.Model},
	}
	candidates = append(candidates, cachedResponse.Alternatives...)

	// Find first available model
	availableIdx := pm.findFirstAvailableModel(candidates, cbs, requestID)
	if availableIdx == -1 {
		return nil
	}

	selected := candidates[availableIdx]
	alternatives := pm.buildAlternativesList(candidates, availableIdx)

	return &models.ModelSelectionResponse{
		Provider:     selected.Provider,
		Model:        selected.Model,
		Alternatives: alternatives,
	}
}

// findFirstAvailableModel returns the index of the first available model, or -1 if none are available
func (pm *ModelRouter) findFirstAvailableModel(candidates []models.Alternative, cbs map[string]*circuitbreaker.CircuitBreaker, requestID string) int {
	for i, candidate := range candidates {
		if pm.isModelAvailable(candidate.Provider, cbs) {
			if i > 0 {
				fiberlog.Infof("[%s] 🔄 Using alternative %s/%s (primary unavailable)",
					requestID, candidate.Provider, candidate.Model)
			} else {
				fiberlog.Debugf("[%s] Primary model %s/%s available",
					requestID, candidate.Provider, candidate.Model)
			}
			return i
		}
		fiberlog.Debugf("[%s] 🚫 Model %s/%s unavailable (circuit breaker)",
			requestID, candidate.Provider, candidate.Model)
	}
	return -1
}

// isModelAvailable checks if a model provider is available via circuit breaker
func (pm *ModelRouter) isModelAvailable(provider string, cbs map[string]*circuitbreaker.CircuitBreaker) bool {
	if cbs == nil {
		return true
	}
	cb, exists := cbs[provider]
	return !exists || cb.CanExecute()
}

// buildAlternativesList creates alternatives list excluding the selected model
func (pm *ModelRouter) buildAlternativesList(candidates []models.Alternative, selectedIdx int) []models.Alternative {
	alternatives := make([]models.Alternative, 0, len(candidates)-1)
	for i, candidate := range candidates {
		if i != selectedIdx {
			alternatives = append(alternatives, candidate)
		}
	}
	return alternatives
}
