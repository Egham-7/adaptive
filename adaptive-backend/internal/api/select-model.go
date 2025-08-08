package api

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/chat/completions"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/protocol_manager"
	"fmt"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// SelectModelHandler handles model selection requests.
// It determines which model/provider would be selected for a given chat completion request
// without actually executing the completion.
type SelectModelHandler struct {
	reqSvc      *completions.RequestService
	protocolMgr *protocol_manager.ProtocolManager
}

// NewSelectModelHandler initializes the select model handler with required dependencies.
func NewSelectModelHandler() *SelectModelHandler {
	protocolMgr, err := protocol_manager.NewProtocolManager(nil)
	if err != nil {
		fiberlog.Fatalf("protocol manager initialization failed: %v", err)
	}
	return &SelectModelHandler{
		reqSvc:      completions.NewRequestService(),
		protocolMgr: protocolMgr,
	}
}

// SelectModel handles the model selection HTTP request.
// It processes a chat completion request and returns the selected model/provider
// without actually executing the completion.
func (h *SelectModelHandler) SelectModel(c *fiber.Ctx) error {
	reqID := h.reqSvc.GetRequestID(c)
	userID := h.reqSvc.GetAPIKey(c)
	fiberlog.Infof("[%s] starting model selection request", reqID)

	req, err := h.reqSvc.ParseChatCompletionRequest(c)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": fiber.Map{
				"message": fmt.Sprintf("Invalid request: %s", err.Error()),
				"type":    "invalid_request_error",
				"code":    "bad_request",
			},
		})
	}

	// Perform model selection using the same logic as completions
	resp, cacheSource, err := h.selectProtocol(req, userID, reqID)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": fiber.Map{
				"message": fmt.Sprintf("Model selection failed: %s", err.Error()),
				"type":    "internal_error",
				"code":    "model_selection_failed",
			},
		})
	}

	// Build metadata about the selection
	metadata := models.SelectionMetadata{
		CacheSource: cacheSource,
	}

	// Extract provider and model based on protocol type and update the request
	switch resp.Protocol {
	case models.ProtocolStandardLLM:
		if resp.Standard != nil {
			metadata.Provider = resp.Standard.Provider
			metadata.Model = resp.Standard.Model
			req.Model = resp.Standard.Parameters.Model
		}
	case models.ProtocolMinion:
		if resp.Minion != nil {
			metadata.Provider = resp.Minion.Provider
			metadata.Model = resp.Minion.Model
			req.Model = resp.Minion.Parameters.Model
		}
	case models.ProtocolMinionsProtocol:
		// For minions protocol, prioritize standard
		if resp.Standard != nil {
			metadata.Provider = resp.Standard.Provider
			metadata.Model = resp.Standard.Model
			req.Model = resp.Standard.Parameters.Model
		} else if resp.Minion != nil {
			metadata.Provider = resp.Minion.Provider
			metadata.Model = resp.Minion.Model
			req.Model = resp.Minion.Parameters.Model
		}
	}

	// Add cost and complexity information if available from protocol manager config
	if req.ProtocolManagerConfig != nil {
		for _, model := range req.ProtocolManagerConfig.Models {
			if model.ModelName == metadata.Model && model.Provider == metadata.Provider {
				metadata.CostPer1M = model.CostPer1MInputTokens
				if model.Complexity != nil {
					metadata.Complexity = *model.Complexity
				}
				break
			}
		}
	}

	// Apply optimal parameters from protocol manager
	paramSvc := completions.NewParameterService()
	params, err := paramSvc.GetParams(resp)
	if err != nil {
		fiberlog.Warnf("[%s] Failed to get parameters for optimization: %v", reqID, err)
	} else {
		if err := paramSvc.ApplyModelParameters(req, params, reqID); err != nil {
			fiberlog.Warnf("[%s] Failed to apply model parameters: %v", reqID, err)
		}
	}

	response := models.SelectModelResponse{
		Request:  req,
		Metadata: metadata,
	}

	fiberlog.Infof("[%s] model selection completed - provider: %s, model: %s", reqID, metadata.Provider, metadata.Model)

	return c.JSON(response)
}

// selectProtocol runs protocol selection and returns the chosen protocol response and cache source.
// This mirrors the same logic used in the completions handler.
func (h *SelectModelHandler) selectProtocol(
	req *models.ChatCompletionRequest,
	userID, requestID string,
) (
	resp *models.ProtocolResponse,
	cacheSource string,
	err error,
) {
	fiberlog.Infof("[%s] Starting protocol selection for user: %s", requestID, userID)

	openAIParams := req.ToOpenAIParams()
	if openAIParams == nil {
		return nil, "", fmt.Errorf("failed to convert request to OpenAI parameters")
	}

	selReq := models.ModelSelectionRequest{
		ChatCompletionRequest: *openAIParams,
		ProtocolManagerConfig: req.ProtocolManagerConfig,
	}

	// Use empty circuit breakers map since we're not actually executing
	resp, cacheSource, err = h.protocolMgr.SelectProtocolWithCache(
		selReq, userID, requestID, make(map[string]*circuitbreaker.CircuitBreaker), req.SemanticCache,
	)
	if err != nil {
		fiberlog.Errorf("[%s] Protocol selection error: %v", requestID, err)
		return nil, "", fmt.Errorf("protocol selection failed: %w", err)
	}

	return resp, cacheSource, nil
}
