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

// CompletionHandler handles chat completions end-to-end.
// It manages the lifecycle of chat completion requests, including provider selection,
// fallback handling, and response processing.
type CompletionHandler struct {
	reqSvc      *completions.RequestService
	respSvc     *completions.ResponseService
	paramSvc    *completions.ParameterService
	protocolMgr *protocol_manager.ProtocolManager
	fallbackSvc *completions.FallbackService
}

// NewCompletionHandler wires up dependencies and initializes the completion handler.
func NewCompletionHandler() *CompletionHandler {
	protocolMgr, err := protocol_manager.NewProtocolManager(nil)
	if err != nil {
		fiberlog.Fatalf("protocol manager initialization failed: %v", err)
	}
	return &CompletionHandler{
		reqSvc:      completions.NewRequestService(),
		respSvc:     completions.NewResponseService(protocolMgr),
		paramSvc:    completions.NewParameterService(),
		protocolMgr: protocolMgr,
		fallbackSvc: completions.NewFallbackService(),
	}
}

// ChatCompletion handles the chat completion HTTP request.
// It processes the request through provider selection, parameter configuration,
// and response handling with circuit breaking for reliability.
func (h *CompletionHandler) ChatCompletion(c *fiber.Ctx) error {
	reqID := h.reqSvc.GetRequestID(c)
	userID := h.reqSvc.GetAPIKey(c)
	fiberlog.Infof("[%s] starting chat completion request", reqID)

	req, err := h.reqSvc.ParseChatCompletionRequest(c)
	if err != nil {
		return h.respSvc.HandleBadRequest(c, err.Error(), reqID)
	}
	isStream := req.Stream

	// Configure fallback mode based on request
	h.configureFallbackMode(req, reqID)

	resp, cacheSource, err := h.selectProtocol(
		req, userID, reqID, make(map[string]*circuitbreaker.CircuitBreaker),
	)
	if err != nil {
		return h.respSvc.HandleInternalError(c, err.Error(), reqID)
	}

	params, err := h.paramSvc.GetParams(resp)
	if err != nil {
		return h.respSvc.HandleInternalError(c, err.Error(), reqID)
	}

	if err := h.paramSvc.ApplyModelParameters(req, params, reqID); err != nil {
		return h.respSvc.HandleInternalError(c, err.Error(), reqID)
	}

	return h.respSvc.HandleProtocol(c, resp.Protocol, req, resp, reqID, isStream, cacheSource)
}

// selectProtocol runs protocol selection and returns the chosen protocol response and cache source.
func (h *CompletionHandler) selectProtocol(
	req *models.ChatCompletionRequest,
	userID, requestID string,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
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

	resp, cacheSource, err = h.protocolMgr.SelectProtocolWithCache(
		selReq, userID, requestID, circuitBreakers, req.SemanticCache,
	)
	if err != nil {
		fiberlog.Errorf("[%s] Protocol selection error: %v", requestID, err)
		return nil, "", fmt.Errorf("protocol selection failed: %w", err)
	}

	return resp, cacheSource, nil
}

// configureFallbackMode sets the fallback mode based on the request configuration
func (h *CompletionHandler) configureFallbackMode(req *models.ChatCompletionRequest, reqID string) {
	// Default fallback behavior: enabled with parallel mode
	enabled := true
	mode := models.FallbackModeParallel

	// Override defaults if fallback config is provided
	if req.Fallback != nil {
		enabled = req.Fallback.Enabled
		if req.Fallback.Mode != "" {
			mode = req.Fallback.Mode
		}
	}

	// Check if fallback is disabled
	if !enabled {
		h.fallbackSvc.SetMode(completions.FallbackModeSequential)
		fiberlog.Debugf("[%s] Fallback is explicitly disabled, using single provider only", reqID)
		return
	}

	// Fallback is enabled, configure the mode
	switch mode {
	case models.FallbackModeSequential:
		h.fallbackSvc.SetMode(completions.FallbackModeSequential)
		fiberlog.Infof("[%s] Fallback enabled with mode: sequential", reqID)
	case models.FallbackModeParallel:
		h.fallbackSvc.SetMode(completions.FallbackModeRace)
		fiberlog.Infof("[%s] Fallback enabled with mode: parallel", reqID)
	default:
		// Unknown mode, default to parallel with warning
		h.fallbackSvc.SetMode(completions.FallbackModeRace)
		fiberlog.Warnf("[%s] Fallback enabled with unknown mode '%s', using default: parallel", reqID, mode)
	}
}
