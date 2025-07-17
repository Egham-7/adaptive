package protocol_manager

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/utils"
	"fmt"
	"os"

	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const defaultCostBiasFactor = 0.15

// ProtocolManager coordinates protocol selection and caching for model selection.
type ProtocolManager struct {
	cache  *SemanticCache
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
	// Create OpenAI client
	openaiClient := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	// Create Redis client
	redisClient, err := config.NewRedisClientFromEnv()
	if err != nil {
		return nil, fmt.Errorf("init redis client: %w", err)
	}

	// Create Redis-backed semantic cache
	cache := NewSemanticCache(redisClient, &openaiClient)

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

	// 1) Check semantic cache first
	if hit, src, ok := pm.cache.Lookup(prompt, userID); ok {
		fiberlog.Infof("[%s] cache hit (%s)", requestID, src)
		return &hit, src, nil
	}

	// 2) Call Python service for protocol selection
	resp := pm.client.SelectProtocol(req)

	fiberlog.Infof("[%s] protocol selected: %s", requestID, resp.Protocol)

	// 3) Store in cache for future use
	pm.cache.Store(prompt, userID, resp)

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

// Close properly closes the Redis client during shutdown
func (pm *ProtocolManager) Close() error {
	if pm.cache != nil {
		return pm.cache.Close()
	}
	return nil
}
