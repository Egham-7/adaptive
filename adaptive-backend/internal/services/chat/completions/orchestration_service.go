package completions

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/minions"
	"adaptive-backend/internal/services/model_selection"
	"context"
	"fmt"

	fiberlog "github.com/gofiber/fiber/v2/log"
)

// OrchestrationService coordinates model selection and provider setup.
type OrchestrationService struct {
	modelSelector  *model_selection.ModelSelector
	minionRegistry *minions.MinionRegistry
}

// NewOrchestrationService creates a new orchestration service.
func NewOrchestrationService(
	modelSelector *model_selection.ModelSelector,
	minionRegistry *minions.MinionRegistry,
) *OrchestrationService {
	return &OrchestrationService{
		modelSelector:  modelSelector,
		minionRegistry: minionRegistry,
	}
}

// SelectAndConfigureProvider runs model selection and returns the chosen
// LLM provider along with the orchestrator response.
func (s *OrchestrationService) SelectAndConfigureProvider(
	ctx context.Context,
	req *models.ChatCompletionRequest,
	userID, requestID string,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
) (
	resp *models.OrchestratorResponse,
	err error,
) {
	fiberlog.Infof("[%s] Starting orchestration for user: %s", requestID, userID)

	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("messages array must contain at least one element")
	}
	prompt := req.Messages[len(req.Messages)-1].OfUser.Content.OfString.Value
	if prompt == "" {
		return nil, fmt.Errorf("last message content cannot be empty")
	}

	selReq := models.ModelSelectionRequest{
		Prompt:             prompt,
		ProviderConstraint: req.ProviderConstraint,
		CostBias:           req.CostBias,
	}

	resp, _, err = s.modelSelector.SelectModelWithCache(
		selReq, userID, requestID, circuitBreakers,
	)
	if err != nil {
		fiberlog.Errorf("[%s] Model selection error: %v", requestID, err)
		return nil, fmt.Errorf("model selection failed: %w", err)
	}

	return resp, nil
}

// ValidateOrchestrationContext ensures dependencies are set.
func (s *OrchestrationService) ValidateOrchestrationContext() error {
	if s.modelSelector == nil {
		return fmt.Errorf("model selector is required")
	}
	if s.minionRegistry == nil {
		return fmt.Errorf("minion registry is required")
	}
	return nil
}
