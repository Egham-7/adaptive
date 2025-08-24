package select_model

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/model_router"
	"fmt"

	fiberlog "github.com/gofiber/fiber/v2/log"
)

// Service handles model selection logic
type Service struct {
	modelRouter *model_router.ModelRouter
	cfg         *config.Config
}

// NewService creates a new select model service
func NewService(modelRouter *model_router.ModelRouter, cfg *config.Config) *Service {
	return &Service{
		modelRouter: modelRouter,
		cfg:         cfg,
	}
}

// SelectModel performs model selection based on the request
func (s *Service) SelectModel(
	req *models.SelectModelRequest,
	userID, requestID string,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
) (*models.SelectModelResponse, error) {
	fiberlog.Infof("[%s] Starting model selection for user: %s", requestID, userID)

	// Build model router config from select model request fields
	requestConfig := &models.ModelRouterConfig{
		Models: req.Models,
	}

	// Set cost bias if provided
	if req.CostBias != nil {
		requestConfig.CostBias = *req.CostBias
	}

	// Set semantic cache config if provided
	if req.ModelRouterCache != nil {
		requestConfig.SemanticCache = models.SemanticCacheConfig{
			Enabled:   req.ModelRouterCache.Enabled,
			Threshold: float64(req.ModelRouterCache.SemanticThreshold),
		}
	}

	// Merge with YAML config to get defaults for other fields
	mergedConfig := s.cfg.MergeModelRouterConfig(requestConfig)

	fiberlog.Debugf("[%s] Built protocol config from select model request - cost bias: %.2f", requestID, mergedConfig.CostBias)

	// Perform protocol selection directly with prompt
	resp, cacheSource, err := s.modelRouter.SelectProtocolWithCache(
		req.Prompt, userID, requestID, mergedConfig, circuitBreakers,
	)
	if err != nil {
		fiberlog.Errorf("[%s] Protocol selection error: %v", requestID, err)
		return nil, fmt.Errorf("protocol selection failed: %w", err)
	}

	// Extract provider and model based on protocol type
	var provider, model string
	var alternatives []models.Alternative

	switch resp.Protocol {
	case models.ProtocolStandardLLM:
		if resp.Standard != nil {
			provider = resp.Standard.Provider
			model = resp.Standard.Model
			alternatives = resp.Standard.Alternatives
		}
	case models.ProtocolMinion:
		if resp.Minion != nil {
			provider = resp.Minion.Provider
			model = resp.Minion.Model
			alternatives = resp.Minion.Alternatives
		}
	}

	// Build metadata about the selection
	metadata := models.SelectionMetadata{
		CacheSource: cacheSource,
	}

	// Add cost and complexity information if available from protocol manager config
	if mergedConfig != nil {
		for _, modelCap := range mergedConfig.Models {
			if modelCap.ModelName == model && modelCap.Provider == provider {
				metadata.CostPer1M = modelCap.CostPer1MInputTokens
				if modelCap.Complexity != nil {
					metadata.Complexity = *modelCap.Complexity
				}
				break
			}
		}
	}

	fiberlog.Infof("[%s] model selection completed - provider: %s, model: %s", requestID, provider, model)

	return &models.SelectModelResponse{
		Provider:     provider,
		Model:        model,
		Alternatives: alternatives,
		Metadata:     metadata,
	}, nil
}
