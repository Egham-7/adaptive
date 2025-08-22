package api

import (
	"adaptive-backend/internal/config"
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
	cfg         *config.Config
	reqSvc      *completions.RequestService
	respSvc     *completions.ResponseService
	paramSvc    *completions.ParameterService
	protocolMgr *protocol_manager.ProtocolManager
}

// NewCompletionHandler wires up dependencies and initializes the completion handler.
func NewCompletionHandler(
	cfg *config.Config,
	reqSvc *completions.RequestService,
	respSvc *completions.ResponseService,
	paramSvc *completions.ParameterService,
	protocolMgr *protocol_manager.ProtocolManager,
) *CompletionHandler {
	return &CompletionHandler{
		cfg:         cfg,
		reqSvc:      reqSvc,
		respSvc:     respSvc,
		paramSvc:    paramSvc,
		protocolMgr: protocolMgr,
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

	// Fallback is now handled directly in completion service

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
		ProtocolManagerConfig: req.ProtocolManagerConfig, // Already merged in protocol manager
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

