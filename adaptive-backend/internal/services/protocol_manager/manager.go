package protocol_manager

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/utils"
	"fmt"

	fiberlog "github.com/gofiber/fiber/v2/log"
)

const (
	defaultCostBiasFactor = 0.5
)

// ProtocolManager coordinates protocol selection and caching for model selection.
type ProtocolManager struct {
	cache  *ProtocolManagerCache
	client *ProtocolManagerClient
}

// NewProtocolManager creates a new ProtocolManager with cache configuration.
func NewProtocolManager(cacheConfig *CacheConfig) (*ProtocolManager, error) {
	var cache *ProtocolManagerCache
	var err error

	// Use default config if nil
	if cacheConfig == nil {
		defaultConfig := DefaultCacheConfig()
		cacheConfig = &defaultConfig
		fiberlog.Info("ProtocolManager: Using default cache configuration")
	}

	fiberlog.Infof("ProtocolManager: Initializing with cache enabled=%t, threshold=%.2f",
		cacheConfig.Enabled, cacheConfig.SemanticThreshold)

	// Create cache only if enabled
	if cacheConfig.Enabled {
		cache, err = NewProtocolManagerCache(cacheConfig)
		if err != nil {
			fiberlog.Errorf("ProtocolManager: Failed to create cache: %v", err)
			return nil, fmt.Errorf("failed to create protocol manager cache: %w", err)
		}
		fiberlog.Info("ProtocolManager: Cache initialized successfully")
	} else {
		fiberlog.Warn("ProtocolManager: Cache is disabled")
	}

	client := NewProtocolManagerClient()
	fiberlog.Info("ProtocolManager: Client initialized successfully")

	return &ProtocolManager{
		cache:  cache,
		client: client,
	}, nil
}

// SelectProtocolWithCache checks the semantic cache, then calls the Python service for protocol selection if needed.
// It returns the protocol response, the source (cache or service), and any error encountered.
func (pm *ProtocolManager) SelectProtocolWithCache(
	req models.ModelSelectionRequest,
	userID, requestID string,
	cbs map[string]*circuitbreaker.CircuitBreaker,
) (*models.ProtocolResponse, string, error) {
	fiberlog.Debugf("[%s] Starting protocol selection for user: %s", requestID, userID)

	if req.CostBias == nil || *req.CostBias <= 0 {
		bias := float32(defaultCostBiasFactor)
		req.CostBias = &bias
		fiberlog.Debugf("[%s] Using default cost bias: %.2f", requestID, bias)
	} else {
		fiberlog.Debugf("[%s] Using provided cost bias: %.2f", requestID, *req.CostBias)
	}

	// Extract prompt from last message for cache key
	prompt, err := utils.ExtractLastMessage(req.Messages)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to extract last message: %v", requestID, err)
		return nil, "", fmt.Errorf("failed to extract last message: %w", err)
	}

	fiberlog.Debugf("[%s] Extracted prompt for caching (length: %d chars)", requestID, len(prompt))

	// 1) Check cache for existing protocol response (if cache is enabled)
	if pm.cache != nil {
		fiberlog.Debugf("[%s] Checking cache for existing protocol response", requestID)
		if hit, source, found := pm.cache.Lookup(prompt, requestID); found {
			fiberlog.Infof("[%s] Cache hit (%s) - returning cached protocol: %s", requestID, source, hit.Protocol)
			return hit, source, nil
		}
		fiberlog.Debugf("[%s] Cache miss - proceeding to protocol selection service", requestID)
	} else {
		fiberlog.Debugf("[%s] Cache disabled - proceeding directly to protocol selection service", requestID)
	}

	// 2) Call Python service for protocol selection
	fiberlog.Debugf("[%s] Calling protocol selection service", requestID)
	resp := pm.client.SelectProtocol(req)

	fiberlog.Infof("[%s] Protocol selected: %s (provider: %s, model: %s)",
		requestID, resp.Protocol,
		getProviderFromResponse(resp), getModelFromResponse(resp))

	// 3) Store in cache for future use (if cache is enabled)
	if pm.cache != nil {
		fiberlog.Debugf("[%s] Storing protocol response in cache", requestID)
		if err := pm.cache.Store(prompt, resp); err != nil {
			fiberlog.Errorf("[%s] Failed to store protocol response in cache: %v", requestID, err)
		} else {
			fiberlog.Debugf("[%s] Successfully stored protocol response in cache", requestID)
		}
	}

	return &resp, string(resp.Protocol), nil
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
