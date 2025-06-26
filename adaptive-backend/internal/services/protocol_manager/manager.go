package protocol_manager

import (
	"fmt"
	"log"
	"os"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"

	"github.com/botirk38/semanticcache"
)

const defaultCostBiasFactor = 0.15

// ProtocolManager coordinates protocol selection and caching for model orchestration.
type ProtocolManager struct {
	cache  *CacheManager
	client *ProtocolManagerClient
}

// NewProtocolManager creates a new ProtocolManager with semantic cache and client.
func NewProtocolManager(
	semanticThreshold float32,
	userCacheSize int,
) (*ProtocolManager, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY environment variable is not set")
	}
	embProv, err := semanticcache.NewOpenAIProvider(apiKey, "")
	if err != nil {
		return nil, fmt.Errorf("init embedding provider: %w", err)
	}

	cacheMgr, err := NewCacheManager(embProv)
	if err != nil {
		return nil, fmt.Errorf("init cache manager: %w", err)
	}

	client := NewProtocolManagerClient()

	return &ProtocolManager{
		cache:  cacheMgr,
		client: client,
	}, nil
}

// SelectProtocolWithCache checks the semantic cache, then calls the Python service for protocol selection if needed.
// It returns the orchestrator response, the source (cache or service), and any error encountered.
func (pm *ProtocolManager) SelectProtocolWithCache(
	req models.ModelSelectionRequest,
	userID, requestID string,
	cbs map[string]*circuitbreaker.CircuitBreaker,
) (*models.OrchestratorResponse, string, error) {
	if req.CostBias == nil || *req.CostBias <= 0 {
		bias := float32(defaultCostBiasFactor)
		req.CostBias = &bias
	}

	// 1) Check semantic cache first
	if hit, src, ok := pm.cache.Lookup(req.Prompt, userID); ok {
		log.Printf("[%s] cache hit (%s)", requestID, src)
		return &hit, src, nil
	}

	// 2) Call Python service for protocol selection
	resp, err := pm.client.SelectProtocol(req)
	if err != nil {
		log.Printf("[%s] protocol selection failed: %v", requestID, err)
		return nil, "error", fmt.Errorf("protocol selection failed: %w", err)
	}

	log.Printf("[%s] protocol selected: %s", requestID, resp.Protocol)

	// 3) Store in cache for future use
	pm.cache.Store(req.Prompt, userID, resp)

	return &resp, string(resp.Protocol), nil
}

// GetClientMetrics returns circuit breaker metrics for the Python service client.
func (pm *ProtocolManager) GetClientMetrics() circuitbreaker.LocalMetrics {
	return pm.client.GetCircuitBreakerMetrics()
}
