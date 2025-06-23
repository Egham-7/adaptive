package completions

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/minions"
	"adaptive-backend/internal/services/model_selection"
	"adaptive-backend/internal/services/providers"
	"adaptive-backend/internal/services/providers/provider_interfaces"
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
	provider provider_interfaces.LLMProvider,
	resp *models.OrchestratorResponse,
	err error,
) {
	fiberlog.Infof("[%s] Starting orchestration for user: %s", requestID, userID)

	if len(req.Messages) == 0 {
		return nil, nil, fmt.Errorf("messages array must contain at least one element")
	}
	prompt := req.Messages[len(req.Messages)-1].OfUser.Content.OfString.Value
	if prompt == "" {
		return nil, nil, fmt.Errorf("last message content cannot be empty")
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
		return nil, nil, fmt.Errorf("model selection failed: %w", err)
	}

	// configure the provider
	switch resp.Protocol {
	case models.ProtocolStandardLLM:
		info := resp.Standard
		fiberlog.Infof("[%s] Selected StandardLLM: %s/%s",
			requestID, info.Provider, info.Model)
		provider, err = providers.NewLLMProvider(
			info.Provider, &info.Model, s.minionRegistry,
		)
		if err != nil {
			return nil, resp,
				fmt.Errorf("failed to create LLM provider %s: %w", info.Provider, err)
		}

	case models.ProtocolMinion:
		info := resp.Minion
		fiberlog.Infof("[%s] Selected Minion task: %s",
			requestID, info.TaskType)
		provider, err = providers.NewLLMProvider(
			"minion", &info.TaskType, s.minionRegistry,
		)
		if err != nil {
			return nil, resp,
				fmt.Errorf("failed to create minion provider for task %s: %w",
					info.TaskType, err)
		}

	default:
		return nil, resp,
			fmt.Errorf("unknown orchestrator protocol: %s", resp.Protocol)
	}

	return provider, resp, nil
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
