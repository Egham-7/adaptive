package completions

import (
	"context"
	"fmt"

	"adaptive-backend/internal/models"
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

// OrchestratorResult holds the result of orchestration
type OrchestratorResult struct {
	Provider     provider_interfaces.LLMProvider
	ProviderName string // For logging and debugging
	CacheType    string
	ProtocolType string
	ModelName    string
	Parameters   models.OpenAIParameters
	TaskType     string               // For minions
	Alternatives []models.Alternative // For failover racing if primary fails
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
) (*OrchestratorResult, error) {
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
// Always returns primary provider, stores alternatives for potential failover
func (s *OrchestrationService) handleStandardLLMResponse(
	resp models.StandardLLMOrchestratorResponse,
	cacheType string,
	requestID string,
) (*OrchestratorResult, error) {
	fiberlog.Infof("[%s] Processing StandardLLM response: Model=%s, Provider=%s",
		requestID, resp.StandardLLMData.Model, resp.StandardLLMData.Provider)

	// Always create primary provider first
	provider, err := providers.NewLLMProvider(resp.StandardLLMData.Provider, nil, s.minionRegistry)
	if err != nil {
		return nil, fmt.Errorf("failed to create LLM provider %s: %w", resp.StandardLLMData.Provider, err)
	}

	if len(resp.Alternatives) > 0 {
		fiberlog.Infof("[%s] Primary provider ready, %d alternatives available for failover",
			requestID, len(resp.Alternatives))
	}

	return &OrchestratorResult{
		Provider:     provider,
		ProviderName: resp.StandardLLMData.Provider,
		CacheType:    cacheType,
		ProtocolType: "StandardLLM",
		ModelName:    resp.StandardLLMData.Model,
		Parameters:   resp.Parameters,
		TaskType:     "",
		Alternatives: resp.Alternatives,
	}, nil
}

// handleMinionResponse processes minion orchestrator responses
// Always returns primary provider, stores alternatives for potential failover
func (s *OrchestrationService) handleMinionResponse(
	resp models.MinionOrchestratorResponse,
	cacheType string,
	requestID string,
) (*OrchestratorResult, error) {
	fiberlog.Infof("[%s] Processing Minion response: TaskType=%s",
		requestID, resp.MinionData.TaskType)

	// Always create primary minion provider first
	provider, err := providers.NewLLMProvider("minion", &resp.MinionData.TaskType, s.minionRegistry)
	if err != nil {
		return nil, fmt.Errorf("failed to create minion provider for task %s: %w", resp.MinionData.TaskType, err)
	}

	if len(resp.Alternatives) > 0 {
		fiberlog.Infof("[%s] Primary minion ready, %d alternatives available for failover",
			requestID, len(resp.Alternatives))
	}

	return &OrchestratorResult{
		Provider:     provider,
		ProviderName: "minion",
		CacheType:    cacheType,
		ProtocolType: "Minion",
		Parameters:   resp.MinionData.Parameters,
		ModelName:    fmt.Sprintf("Minion-%s", resp.MinionData.TaskType),
		TaskType:     resp.MinionData.TaskType,
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
