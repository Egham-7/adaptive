package protocol_manager

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/utils"
	"fmt"

	fiberlog "github.com/gofiber/fiber/v2/log"
)


// ProtocolManager coordinates protocol selection and caching for model selection.
type ProtocolManager struct {
	cache  *ProtocolManagerCache
	client *ProtocolManagerClient
	cfg    *config.Config
}

// NewProtocolManager creates a new ProtocolManager with cache configuration.
func NewProtocolManager(cfg *config.Config) (*ProtocolManager, error) {
	var cache *ProtocolManagerCache
	var err error

	// Get semantic cache configuration from config
	semanticCacheConfig := cfg.ProtocolManager.SemanticCache

	fiberlog.Infof("ProtocolManager: Initializing with semantic_cache enabled=%t, threshold=%.2f",
		semanticCacheConfig.Enabled, semanticCacheConfig.Threshold)

	// Create cache only if enabled
	if semanticCacheConfig.Enabled {
		cache, err = NewProtocolManagerCache(cfg)
		if err != nil {
			fiberlog.Errorf("ProtocolManager: Failed to create cache: %v", err)
			return nil, fmt.Errorf("failed to create protocol manager cache: %w", err)
		}
		fiberlog.Info("ProtocolManager: Cache initialized successfully")
	} else {
		fiberlog.Warn("ProtocolManager: Cache is disabled")
	}

	client := NewProtocolManagerClient(cfg)
	fiberlog.Info("ProtocolManager: Client initialized successfully")

	return &ProtocolManager{
		cache:  cache,
		client: client,
		cfg:    cfg,
	}, nil
}

// SelectProtocolWithCache checks the semantic cache, then calls the Python service for protocol selection if needed.
// It returns the protocol response, the source (cache or service), and any error encountered.
// If cacheConfigOverride is provided, it will temporarily override the cache behavior for this request.
func (pm *ProtocolManager) SelectProtocolWithCache(
	req models.ModelSelectionRequest,
	userID, requestID string,
	cbs map[string]*circuitbreaker.CircuitBreaker,
	cacheConfigOverride *models.CacheConfig,
) (*models.ProtocolResponse, string, error) {
	fiberlog.Debugf("[%s] Starting protocol selection for user: %s", requestID, userID)

	// Merge YAML protocol manager config with request override
	req.ProtocolManagerConfig = pm.cfg.MergeProtocolManagerConfig(req.ProtocolManagerConfig)
	fiberlog.Debugf("[%s] Using merged protocol manager config - cost bias: %.2f", requestID, req.ProtocolManagerConfig.CostBias)

	// Extract prompt from last message for cache key
	prompt, err := utils.ExtractLastMessage(req.ChatCompletionRequest.Messages)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to extract last message: %v", requestID, err)
		return nil, "", fmt.Errorf("failed to extract last message: %w", err)
	}

	fiberlog.Debugf("[%s] Extracted prompt for caching (length: %d chars)", requestID, len(prompt))

	// 1) Check if cache should be used (either default cache or override config)
	useCache := pm.cache != nil
	if cacheConfigOverride != nil {
		useCache = cacheConfigOverride.Enabled
		fiberlog.Debugf("[%s] Cache config override provided: enabled=%t, threshold=%.2f",
			requestID, cacheConfigOverride.Enabled, cacheConfigOverride.SemanticThreshold)
	}

	if useCache && pm.cache != nil {
		fiberlog.Debugf("[%s] Checking cache for existing protocol response", requestID)
		if hit, source, found := pm.cache.Lookup(prompt, requestID); found {
			fiberlog.Infof("[%s] Cache hit (%s) - returning cached protocol: %s", requestID, source, hit.Protocol)
			return hit, source, nil
		}
		fiberlog.Debugf("[%s] Cache miss - proceeding to protocol selection service", requestID)
	} else {
		if cacheConfigOverride != nil && !cacheConfigOverride.Enabled {
			fiberlog.Debugf("[%s] Cache disabled by request override", requestID)
		} else {
			fiberlog.Debugf("[%s] Cache disabled - proceeding directly to protocol selection service", requestID)
		}
	}

	// 2) Call Python service for protocol selection
	fiberlog.Debugf("[%s] Calling protocol selection service", requestID)
	resp := pm.client.SelectProtocol(req)

	fiberlog.Infof("[%s] Protocol selected: %s (provider: %s, model: %s)",
		requestID, resp.Protocol,
		getProviderFromResponse(resp), getModelFromResponse(resp))

	// 3) Do NOT store in cache yet - wait for successful completion
	// The cache will be populated after successful API call

	return &resp, "", nil
}

// StoreSuccessfulProtocol stores a protocol response in the semantic cache after successful completion
func (pm *ProtocolManager) StoreSuccessfulProtocol(
	req models.ModelSelectionRequest,
	resp models.ProtocolResponse,
	requestID string,
	cacheConfigOverride *models.CacheConfig,
) error {
	// Check if cache should be used
	useCache := pm.cache != nil
	if cacheConfigOverride != nil {
		useCache = cacheConfigOverride.Enabled
	}

	if !useCache || pm.cache == nil {
		fiberlog.Debugf("[%s] Semantic cache disabled - skipping storage", requestID)
		return nil
	}

	// Extract prompt from last message for cache key
	prompt, err := utils.ExtractLastMessage(req.ChatCompletionRequest.Messages)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to extract last message for semantic cache: %v", requestID, err)
		return fmt.Errorf("failed to extract last message: %w", err)
	}

	fiberlog.Debugf("[%s] Storing successful protocol response in semantic cache", requestID)
	if err := pm.cache.Store(prompt, resp); err != nil {
		fiberlog.Errorf("[%s] Failed to store protocol response in semantic cache: %v", requestID, err)
		return err
	}

	fiberlog.Debugf("[%s] Successfully stored protocol response in semantic cache", requestID)
	return nil
}

// Helper functions to extract provider and model from response for logging
func getProviderFromResponse(resp models.ProtocolResponse) string {
	if resp.Standard != nil {
		return resp.Standard.Provider
	}
	if resp.Minion != nil {
		return resp.Minion.Provider
	}
	return "unknown"
}

func getModelFromResponse(resp models.ProtocolResponse) string {
	if resp.Standard != nil {
		return resp.Standard.Model
	}
	if resp.Minion != nil {
		return resp.Minion.Model
	}
	return "unknown"
}

// GetClientMetrics returns circuit breaker metrics for the Python service client.
func (pm *ProtocolManager) GetClientMetrics() circuitbreaker.LocalMetrics {
	return pm.client.GetCircuitBreakerMetrics()
}

// ValidateContext ensures dependencies are set.
func (pm *ProtocolManager) ValidateContext() error {
	fiberlog.Debug("ProtocolManager: Validating context and dependencies")

	if pm.client == nil {
		fiberlog.Error("ProtocolManager: Protocol manager client is missing")
		return fmt.Errorf("protocol manager client is required")
	}

	if pm.cache != nil {
		fiberlog.Debug("ProtocolManager: Cache is enabled and available")
	} else {
		fiberlog.Debug("ProtocolManager: Cache is disabled")
	}

	fiberlog.Debug("ProtocolManager: Context validation successful")
	return nil
}

// Close properly closes the protocol manager cache during shutdown
func (pm *ProtocolManager) Close() error {
	fiberlog.Info("ProtocolManager: Shutting down")

	if pm.cache != nil {
		fiberlog.Info("ProtocolManager: Closing cache connection")
		pm.cache.Close()
		fiberlog.Info("ProtocolManager: Cache closed successfully")
	} else {
		fiberlog.Debug("ProtocolManager: No cache to close (cache disabled)")
	}

	fiberlog.Info("ProtocolManager: Shutdown completed")
	return nil
}
