package completions

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/minions"
	"adaptive-backend/internal/services/model_selection"
	"adaptive-backend/internal/services/providers"
	"fmt"

	fiberlog "github.com/gofiber/fiber/v2/log"
)

// OrchestrationService coordinates the flow between different services for chat completions
type OrchestrationService struct {
	modelSelector  *model_selection.ModelSelector
	minionRegistry *minions.MinionRegistry
}

// NewOrchestrationService creates a new orchestration service
func NewOrchestrationService(
	modelSelector *model_selection.ModelSelector,
	minionRegistry *minions.MinionRegistry,
) *OrchestrationService {
	return &OrchestrationService{
		modelSelector:  modelSelector,
		minionRegistry: minionRegistry,
	}
}

// SelectAndConfigureProvider orchestrates the model selection and provider configuration
func (s *OrchestrationService) SelectAndConfigureProvider(
	req *models.ChatCompletionRequest,
	userID, requestID string,
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
	orchestratorResponse, cacheType, err := s.modelSelector.SelectModelWithCache(selectReq, userID, requestID)
	if err != nil {
		fiberlog.Errorf("[%s] Model selection error: %v", requestID, err)
		return nil, fmt.Errorf("model selection failed: %w", err)
	}

	// Process the orchestrator response based on type
	switch resp := orchestratorResponse.(type) {
	case models.StandardLLMOrchestratorResponse:
		return s.handleStandardLLMResponse(resp, cacheType, requestID)
	case models.MinionOrchestratorResponse:
		return s.handleMinionResponse(resp, cacheType, requestID)
	default:
		return nil, fmt.Errorf("unknown orchestrator protocol received")
	}
}

// handleStandardLLMResponse processes standard LLM orchestrator responses
func (s *OrchestrationService) handleStandardLLMResponse(
	resp models.StandardLLMOrchestratorResponse,
	cacheType string,
	requestID string,
) (*models.OrchestratorResult, error) {
	fiberlog.Infof("[%s] Processing StandardLLM response: Model=%s, Provider=%s",
		requestID, resp.StandardLLMData.Model, resp.StandardLLMData.Provider)

	// Create provider
	provider, err := providers.NewLLMProvider(resp.StandardLLMData.Provider, nil, s.minionRegistry)
	if err != nil {
		return nil, fmt.Errorf("failed to create LLM provider %s: %w", resp.StandardLLMData.Provider, err)
	}

	return &models.OrchestratorResult{
		Provider:     provider,
		ProviderName: resp.StandardLLMData.Provider,
		CacheType:    cacheType,
		ProtocolType: "StandardLLM",
		ModelName:    resp.StandardLLMData.Model,
		Parameters:   resp.Parameters,
		TaskType:     "",
	}, nil
}

// handleMinionResponse processes minion orchestrator responses
func (s *OrchestrationService) handleMinionResponse(
	resp models.MinionOrchestratorResponse,
	cacheType string,
	requestID string,
) (*models.OrchestratorResult, error) {
	fiberlog.Infof("[%s] Processing Minion response: TaskType=%s",
		requestID, resp.MinionData.TaskType)

	// Create minion provider
	provider, err := providers.NewLLMProvider("minion", &resp.MinionData.TaskType, s.minionRegistry)
	if err != nil {
		return nil, fmt.Errorf("failed to create minion provider for task %s: %w", resp.MinionData.TaskType, err)
	}

	return &models.OrchestratorResult{
		Provider:     provider,
		ProviderName: "minion",
		CacheType:    cacheType,
		ProtocolType: "Minion",
		Parameters:   resp.MinionData.Parameters,
		ModelName:    fmt.Sprintf("Minion-%s", resp.MinionData.TaskType),
		TaskType:     resp.MinionData.TaskType,
	}, nil
}

func (s *OrchestrationService) ValidateOrchestrationContext() error {
	if s.modelSelector == nil {
		return fmt.Errorf("model selector is required")
	}
	if s.minionRegistry == nil {
		return fmt.Errorf("minion registry is required")
	}
	return nil
}
