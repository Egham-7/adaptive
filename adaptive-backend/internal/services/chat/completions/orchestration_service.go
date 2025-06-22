package completions

import (
	"context"
	"fmt"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/minions"
	"adaptive-backend/internal/services/model_selection"
	"adaptive-backend/internal/services/providers"
	"adaptive-backend/internal/services/providers/provider_interfaces"

	fiberlog "github.com/gofiber/fiber/v2/log"
)

// OrchestrationService coordinates the flow between different services for chat completions
type OrchestrationService struct {
	modelSelector  *model_selection.ModelSelector
	minionRegistry *minions.MinionRegistry
	raceService    *RaceService
}

// NewOrchestrationService creates a new orchestration service
func NewOrchestrationService(
	modelSelector *model_selection.ModelSelector,
	minionRegistry *minions.MinionRegistry,
) *OrchestrationService {
	raceService := NewRaceService(minionRegistry)
	return &OrchestrationService{
		modelSelector:  modelSelector,
		minionRegistry: minionRegistry,
		raceService:    raceService,
	}
}

// SelectAndConfigureProvider orchestrates the model selection and provider configuration
// Behavior:
// - If no alternatives are available, uses the primary provider directly
// - If alternatives are available, tries primary first, then races alternatives if primary fails
// This provides automatic failover with optimal performance (no unnecessary parallel requests)
func (s *OrchestrationService) SelectAndConfigureProvider(
	ctx context.Context,
	req *models.ChatCompletionRequest,
	userID, requestID string,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
) (*models.OrchestratorResult, error) {
	fiberlog.Infof("[%s] Starting orchestration for user: %s", requestID, userID)

	// Validate request has messages
	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("messages array must contain at least one element")
	}

	// Extract prompt from the last message
	prompt := req.Messages[len(req.Messages)-1].OfUser.Content.OfString.Value
	if prompt == "" {
		return nil, fmt.Errorf("last message content cannot be empty")
	}

	// Create model selection request
	selectReq := models.ModelSelectionRequest{
		Prompt: prompt,
	}

	// Call model selector to determine protocol and configuration
	orchestratorResponse, cacheType, err := s.modelSelector.SelectModelWithCache(selectReq, userID, requestID, circuitBreakers)
	if err != nil {
		fiberlog.Errorf("[%s] Model selection error: %v", requestID, err)
		return nil, fmt.Errorf("model selection failed: %w", err)
	}

	// Process the orchestrator response
	return s.handleOrchestratorResponse(orchestratorResponse, cacheType, requestID)
}

// handleOrchestratorResponse processes unified orchestrator responses
func (s *OrchestrationService) handleOrchestratorResponse(
	resp *models.OrchestratorResponse,
	cacheType string,
	requestID string,
) (*models.OrchestratorResult, error) {
	var provider provider_interfaces.LLMProvider
	var err error
	var providerName, protocolType, modelName, taskType string

	switch resp.Protocol {
	case models.ProtocolStandardLLM:
		fiberlog.Infof("[%s] Processing StandardLLM response: Model=%s, Provider=%s",
			requestID, resp.Model, resp.Provider)

		provider, err = providers.NewLLMProvider(resp.Provider, nil, s.minionRegistry)
		if err != nil {
			return nil, fmt.Errorf("failed to create LLM provider %s: %w", resp.Provider, err)
		}

		providerName = resp.Provider
		protocolType = "StandardLLM"
		modelName = resp.Model
		taskType = ""

	case models.ProtocolMinion:
		fiberlog.Infof("[%s] Processing Minion response: TaskType=%s",
			requestID, resp.TaskType)

		provider, err = providers.NewLLMProvider("minion", &resp.TaskType, s.minionRegistry)
		if err != nil {
			return nil, fmt.Errorf("failed to create minion provider for task %s: %w", resp.TaskType, err)
		}

		providerName = "minion"
		protocolType = "Minion"
		modelName = fmt.Sprintf("Minion-%s", resp.TaskType)
		taskType = resp.TaskType

	default:
		return nil, fmt.Errorf("unknown orchestrator protocol: %s", resp.Protocol)
	}

	if len(resp.Alternatives) > 0 {
		fiberlog.Infof("[%s] Primary provider ready, %d alternatives available for failover",
			requestID, len(resp.Alternatives))
	}

	return &models.OrchestratorResult{
		Provider:     provider,
		ProviderName: providerName,
		CacheType:    cacheType,
		ProtocolType: protocolType,
		ModelName:    modelName,
		Parameters:   resp.Parameters,
		TaskType:     taskType,
		Alternatives: resp.Alternatives,
	}, nil
}

func (s *OrchestrationService) ValidateOrchestrationContext() error {
	if s.modelSelector == nil {
		return fmt.Errorf("model selector is required")
	}
	if s.minionRegistry == nil {
		return fmt.Errorf("minion registry is required")
	}
	if s.raceService == nil {
		return fmt.Errorf("race service is required")
	}
	return nil
}
