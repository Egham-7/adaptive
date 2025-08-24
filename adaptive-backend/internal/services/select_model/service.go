package select_model

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/protocol_manager"
	"fmt"

	fiberlog "github.com/gofiber/fiber/v2/log"
)

// Service handles model selection logic
type Service struct {
	protocolMgr *protocol_manager.ProtocolManager
}

// NewService creates a new select model service
func NewService(protocolMgr *protocol_manager.ProtocolManager) *Service {
	return &Service{
		protocolMgr: protocolMgr,
	}
}

// SelectModel performs model selection based on the request
func (s *Service) SelectModel(
	req *models.SelectModelRequest,
	userID, requestID string,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
) (*models.SelectModelResponse, error) {
	fiberlog.Infof("[%s] Starting model selection for user: %s", requestID, userID)

	// Ensure models are available in the protocol manager config
	protocolConfig := req.ProtocolManagerConfig
	if protocolConfig != nil {
		protocolConfig.Models = req.Models
	}

	// Perform protocol selection directly with prompt
	resp, cacheSource, err := s.protocolMgr.SelectProtocolWithCache(
		req.Prompt, userID, requestID, protocolConfig, circuitBreakers, nil,
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
	case models.ProtocolMinionsProtocol:
		// For minions protocol, prioritize standard
		if resp.Standard != nil {
			provider = resp.Standard.Provider
			model = resp.Standard.Model
			alternatives = resp.Standard.Alternatives
		} else if resp.Minion != nil {
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
	if protocolConfig != nil {
		for _, modelCap := range protocolConfig.Models {
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
