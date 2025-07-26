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
		respSvc:     completions.NewResponseService(),
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

	resp, err := h.selectProtocol(
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

	return h.respSvc.HandleProtocol(c, resp.Protocol, req, resp, reqID, isStream)
}

// selectProtocol runs protocol selection and returns the chosen protocol response.
func (h *CompletionHandler) selectProtocol(
	req *models.ChatCompletionRequest,
	userID, requestID string,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
) (
	resp *models.ProtocolResponse,
	err error,
) {
	fiberlog.Infof("[%s] Starting protocol selection for user: %s", requestID, userID)

	var costBias *float32
	if req.CostBias != 0 {
		costBias = &req.CostBias
	}

	selReq := models.ModelSelectionRequest{
		Messages:           req.Messages,
		ProviderConstraint: req.ProviderConstraint,
		CostBias:           costBias,
	}

	resp, _, err = h.protocolMgr.SelectProtocolWithCache(
		selReq, userID, requestID, circuitBreakers,
	)
	if err != nil {
		fiberlog.Errorf("[%s] Protocol selection error: %v", requestID, err)
		return nil, fmt.Errorf("protocol selection failed: %w", err)
	}

	return resp, nil
}
