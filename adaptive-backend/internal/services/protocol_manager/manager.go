package protocol_manager

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/utils"
	"fmt"
	"os"

	"github.com/botirk38/semanticcache"
	"github.com/botirk38/semanticcache/backends"
	"github.com/botirk38/semanticcache/providers/openai"
	"github.com/botirk38/semanticcache/types"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

const (
	defaultCostBiasFactor = 0.5
	semanticThreshold     = 0.9
)

// ProtocolManager coordinates protocol selection and caching for model selection.
type ProtocolManager struct {
	cache  *semanticcache.SemanticCache[string, models.ProtocolResponse]
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

	// Get Redis connection configuration
	redisURL := os.Getenv("REDIS_URL")
	if redisURL == "" {
		return nil, fmt.Errorf("REDIS_URL environment variable is not set")
	}

	// Create Redis backend configuration
	config := types.BackendConfig{
		ConnectionString: redisURL,
	}

	// Create backend factory and Redis backend
	factory := &backends.BackendFactory[string, models.ProtocolResponse]{}
	backend, err := factory.NewBackend(types.BackendRedis, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create Redis backend: %w", err)
	}

	// Create OpenAI provider
	provider, err := openai.NewOpenAIProvider(openai.OpenAIConfig{
		APIKey: apiKey,
		Model:  "text-embedding-3-small",
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create OpenAI provider: %w", err)
	}

	// Create semantic cache
	cache, err := semanticcache.NewSemanticCache(backend, provider, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create semantic cache: %w", err)
	}

	client := NewProtocolManagerClient()

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
	if req.CostBias == nil || *req.CostBias <= 0 {
		bias := float32(defaultCostBiasFactor)
		req.CostBias = &bias
	}

	// Extract prompt from last message for cache key
	prompt, err := utils.ExtractLastMessage(req.Messages)
	if err != nil {
		return nil, "", fmt.Errorf("failed to extract last message: %w", err)
	}

	// 1) First try exact key matching
	if hit, found := pm.cache.Get(prompt); found {
		fiberlog.Infof("[%s] cache hit (exact)", requestID)
		return &hit, "exact", nil
	}

	// 2) If no exact match, try semantic similarity search
	if hit, found, err := pm.cache.Lookup(prompt, semanticThreshold); err == nil && found {
		fiberlog.Infof("[%s] cache hit (semantic)", requestID)
		return &hit, "semantic", nil
	}

	// 3) Call Python service for protocol selection
	resp := pm.client.SelectProtocol(req)

	fiberlog.Infof("[%s] protocol selected: %s", requestID, resp.Protocol)

	// 4) Store in cache for future use
	if err := pm.cache.Set(prompt, prompt, resp); err != nil {
		fiberlog.Errorf("failed to store in cache: %v", err)
	}

	return &resp, string(resp.Protocol), nil
}

// GetClientMetrics returns circuit breaker metrics for the Python service client.
func (pm *ProtocolManager) GetClientMetrics() circuitbreaker.LocalMetrics {
	return pm.client.GetCircuitBreakerMetrics()
}

// ValidateContext ensures dependencies are set.
func (pm *ProtocolManager) ValidateContext() error {
	if pm.cache == nil {
		return fmt.Errorf("semantic cache is required")
	}
	if pm.client == nil {
		return fmt.Errorf("protocol manager client is required")
	}
	return nil
}

// Close properly closes the semantic cache during shutdown
func (pm *ProtocolManager) Close() error {
	if pm.cache != nil {
		pm.cache.Close()
	}
	return nil
}
