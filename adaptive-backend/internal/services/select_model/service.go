package select_model

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/format_adapter"
	"adaptive-backend/internal/services/protocol_manager"
	"fmt"

	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go"
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
) (*models.SelectModelResponse, error) {
	fiberlog.Infof("[%s] Starting model selection for user: %s", requestID, userID)

	// Create a minimal ChatCompletionRequest for protocol selection
	chatReq := &models.ChatCompletionRequest{
		Messages:              []openai.ChatCompletionMessageParamUnion{openai.UserMessage(req.Prompt)},
		ProtocolManagerConfig: req.ProtocolManagerConfig,
	}

	// Ensure models are available in the protocol manager config
	if chatReq.ProtocolManagerConfig != nil {
		chatReq.ProtocolManagerConfig.Models = req.Models
	}

	// Perform protocol selection
	resp, cacheSource, err := s.selectProtocol(chatReq, userID, requestID)
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
	if chatReq.ProtocolManagerConfig != nil {
		for _, modelCap := range chatReq.ProtocolManagerConfig.Models {
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

// selectProtocol runs protocol selection and returns the chosen protocol response and cache source
func (s *Service) selectProtocol(
	req *models.ChatCompletionRequest,
	userID, requestID string,
) (
	resp *models.ProtocolResponse,
	cacheSource string,
	err error,
) {
	fiberlog.Infof("[%s] Starting protocol selection for user: %s", requestID, userID)

	// Convert to OpenAI parameters using singleton adapter
	openAIParams, err := format_adapter.AdaptiveToOpenAI.ConvertRequest(req)
	if err != nil {
		return nil, "", fmt.Errorf("failed to convert request to OpenAI parameters: %w", err)
	}

	selReq := models.ModelSelectionRequest{
		ChatCompletionRequest: *openAIParams,
		ProtocolManagerConfig: req.ProtocolManagerConfig,
	}

	// Use empty circuit breakers map since we're not actually executing
	resp, cacheSource, err = s.protocolMgr.SelectProtocolWithCache(
		selReq, userID, requestID, make(map[string]*circuitbreaker.CircuitBreaker), req.SemanticCache,
	)
	if err != nil {
		fiberlog.Errorf("[%s] Protocol selection error: %v", requestID, err)
		return nil, "", fmt.Errorf("protocol selection failed: %w", err)
	}

	return resp, cacheSource, nil
}
