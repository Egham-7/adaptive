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

	if mergedConfig != nil {
		fiberlog.Debugf("[%s] Built model config from select model request - cost bias: %.2f", requestID, mergedConfig.CostBias)
	} else {
		fiberlog.Debugf("[%s] Built model config from select model request - using default config (mergedConfig is nil)", requestID)
	}

	// Perform model selection directly with prompt
	// For direct select-model API, we don't have message history so no tool calls
	resp, _, err := s.modelRouter.SelectModelWithCache(
		req.Prompt, userID, requestID, mergedConfig, circuitBreakers,
		nil, nil, // No tools/tool_call context in direct select-model API
	)
	if err != nil {
		fiberlog.Errorf("[%s] Model selection error: %v", requestID, err)
		return nil, fmt.Errorf("model selection failed: %w", err)
	}

	fiberlog.Infof("[%s] model selection completed - provider: %s, model: %s", requestID, resp.Provider, resp.Model)

	return &models.SelectModelResponse{
		Provider:     resp.Provider,
		Model:        resp.Model,
		Alternatives: resp.Alternatives,
	}, nil
}
