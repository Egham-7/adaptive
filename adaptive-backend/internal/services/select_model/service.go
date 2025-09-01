package select_model

import (
	"fmt"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/model_router"

	fiberlog "github.com/gofiber/fiber/v2/log"
)

// Service handles model selection logic
type Service struct {
	modelRouter *model_router.ModelRouter
}

// NewService creates a new select model service
func NewService(modelRouter *model_router.ModelRouter) *Service {
	return &Service{
		modelRouter: modelRouter,
	}
}

// SelectModel performs model selection based on the request
func (s *Service) SelectModel(
	req *models.SelectModelRequest,
	userID, requestID string,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
	mergedConfig *models.ModelRouterConfig,
) (*models.SelectModelResponse, error) {
	fiberlog.Infof("[%s] Starting model selection for user: %s", requestID, userID)

	fiberlog.Debugf("[%s] Built model config from select model request - cost bias: %.2f", requestID, mergedConfig.CostBias)

	// Perform model selection directly with prompt
	// For direct select-model API, we don't have message history so no tool calls
	resp, cacheSource, err := s.modelRouter.SelectModelWithCache(
		req.Prompt, userID, requestID, mergedConfig, circuitBreakers,
		nil, nil, // No tools/tool_call context in direct select-model API
	)
	if err != nil {
		fiberlog.Errorf("[%s] Model selection error: %v", requestID, err)
		return nil, fmt.Errorf("model selection failed: %w", err)
	}

	// Build metadata about the selection
	metadata := models.SelectionMetadata{
		CacheSource: cacheSource,
	}

	// Add cost and complexity information if available from model router config
	if mergedConfig != nil {
		for _, modelCap := range mergedConfig.Models {
			if modelCap.ModelName == resp.Model && modelCap.Provider == resp.Provider {
				metadata.CostPer1M = modelCap.CostPer1MInputTokens
				if modelCap.Complexity != nil {
					metadata.Complexity = *modelCap.Complexity
				}
				break
			}
		}
	}

	fiberlog.Infof("[%s] model selection completed - provider: %s, model: %s", requestID, resp.Provider, resp.Model)

	return &models.SelectModelResponse{
		Provider:     resp.Provider,
		Model:        resp.Model,
		Alternatives: resp.Alternatives,
		Metadata:     metadata,
	}, nil
}
