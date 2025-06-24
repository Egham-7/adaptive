package model_selection

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"fmt"
	"log"
	"os"

	"github.com/botirk38/semanticcache"
)

const defaultCostBiasFactor = 0.15

type ModelSelector struct {
	cache    *CacheManager
	minions  *MinionRouter
	standard *StandardRouter
}

func NewModelSelector(
	semanticThreshold float32,
	userCacheSize int,
) (*ModelSelector, error) {
	cfg, err := GetDefaultConfig()
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	classifier := NewPromptClassifierClient()
	apiKey := os.Getenv("OPENAI_API_KEY")
	embProv, err := semanticcache.NewOpenAIProvider(apiKey, "")
	if err != nil {
		return nil, fmt.Errorf("init embedding provider: %w", err)
	}

	cacheMgr, err := NewCacheManager(embProv)
	if err != nil {
		return nil, err
	}

	minRouter := NewMinionRouter(cfg)
	stdRouter := NewStandardRouter(
		cfg,
		classifier,
		make(map[string]*circuitbreaker.CircuitBreaker),
		defaultCostBiasFactor,
	)

	return &ModelSelector{
		cache:    cacheMgr,
		minions:  minRouter,
		standard: stdRouter,
	}, nil
}

// SelectModelWithCache first tries early‐minion routing, then cache,
// then falls back to standard LLM routing.
func (m *ModelSelector) SelectModelWithCache(
	req models.ModelSelectionRequest,
	userID, requestID string,
	cbs map[string]*circuitbreaker.CircuitBreaker,
) (*models.OrchestratorResponse, string, error) {
	if req.CostBias <= 0 {
		req.CostBias = defaultCostBiasFactor
	}

	// 1) Semantic cache
	if hit, src, ok := m.cache.Lookup(req.Prompt, userID); ok {
		log.Printf("[%s] cache hit (%s)", requestID, src)
		return &hit, src, nil
	}

	// 2) Early minion?
	cl, err := m.standard.classifier.Classify(req)
	if err != nil {
		return nil, "error", err
	}
	cx := m.standard.extractComplexityScore(cl)

	if m.minions.ShouldRouteEarly(cx, req.Prompt, req.ProviderConstraint, cbs) {
		resp := m.minions.Route(cl, req.CostBias, req.ProviderConstraint, cbs)
		log.Printf("[%s] early‐routed to minion", requestID)
		return resp, "minion", nil
	}

	// 3) Standard LLM selection
	resp, err := m.standard.Route(req)
	if err != nil {
		return nil, "error", err
	}
	log.Printf("[%s] routed via %s", requestID, resp.Protocol)

	m.cache.Store(req.Prompt, userID, *resp)
	return resp, string(resp.Protocol), nil
}
