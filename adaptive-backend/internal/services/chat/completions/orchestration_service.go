package completions

import (
	"context"
	"fmt"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/minions"
	"adaptive-backend/internal/services/protocol_manager"

	fiberlog "github.com/gofiber/fiber/v2/log"
)

// OrchestrationService coordinates protocol selection and provider setup.
type OrchestrationService struct {
	protocolManager *protocol_manager.ProtocolManager
	minionRegistry  *minions.MinionRegistry
}

// NewOrchestrationService creates a new orchestration service.
func NewOrchestrationService(
	protocolManager *protocol_manager.ProtocolManager,
	minionRegistry *minions.MinionRegistry,
) *OrchestrationService {
	return &OrchestrationService{
		protocolManager: protocolManager,
		minionRegistry:  minionRegistry,
	}
}

// SelectAndConfigureProvider runs protocol selection and returns the chosen
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

	resp, _, err = s.protocolManager.SelectProtocolWithCache(
		selReq, userID, requestID, circuitBreakers,
	)
	if err != nil {
		fiberlog.Errorf("[%s] Protocol selection error: %v", requestID, err)
		return nil, fmt.Errorf("protocol selection failed: %w", err)
	}

	return resp, nil
}

// ValidateOrchestrationContext ensures dependencies are set.
func (s *OrchestrationService) ValidateOrchestrationContext() error {
	if s.protocolManager == nil {
		return fmt.Errorf("protocol manager is required")
	}
	if s.minionRegistry == nil {
		return fmt.Errorf("minion registry is required")
	}
	return nil
}
