package api

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/chat/completions"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/protocol_manager"
	"fmt"
	"strings"

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
	fallbackSvc *completions.FallbackService
}

// NewCompletionHandler wires up dependencies and initializes the completion handler.
func NewCompletionHandler(
	cfg *config.Config,
	reqSvc *completions.RequestService,
	respSvc *completions.ResponseService,
	paramSvc *completions.ParameterService,
	protocolMgr *protocol_manager.ProtocolManager,
	fallbackSvc *completions.FallbackService,
) *CompletionHandler {
	return &CompletionHandler{
		cfg:         cfg,
		reqSvc:      reqSvc,
		respSvc:     respSvc,
		paramSvc:    paramSvc,
		protocolMgr: protocolMgr,
		fallbackSvc: fallbackSvc,
	}
}

// ChatCompletion handles the chat completion HTTP request.
// It processes the request through provider selection, parameter configuration,
// and response handling with circuit breaking for reliability.
func (h *CompletionHandler) ChatCompletion(c *fiber.Ctx) error {
	reqID := h.reqSvc.GetRequestID(c)

	// Parse request first to get user ID from the request body
	req, err := h.reqSvc.ParseChatCompletionRequest(c)
	if err != nil {
		return h.respSvc.HandleBadRequest(c, err.Error(), reqID)
	}

	// Extract user ID from request body (use "internal" if not provided)
	userID := "internal"
	if req.User.Value != "" {
		userID = req.User.Value
	}
	fiberlog.Infof("[%s] starting chat completion request", reqID)
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

	// Check if model is explicitly provided (non-empty) - if so, use manual override
	if req.Model != "" {
		fiberlog.Infof("[%s] Model explicitly provided (%s), using manual override instead of protocol manager", requestID, req.Model)
		return h.createManualProtocolResponse(req, requestID)
	}

	fiberlog.Debugf("[%s] No explicit model provided, proceeding with protocol manager selection", requestID)

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

// configureFallbackMode sets the fallback mode based on the request configuration
func (h *CompletionHandler) configureFallbackMode(req *models.ChatCompletionRequest, reqID string) {
	// Merge YAML fallback config with request override
	mergedFallback := h.cfg.MergeFallbackConfig(req.Fallback)
	enabled := mergedFallback.Enabled
	mode := mergedFallback.Mode

	fiberlog.Debugf("[%s] Using merged fallback config - enabled: %t, mode: %s", reqID, enabled, mode)

	// Check if fallback is disabled
	if !enabled {
		h.fallbackSvc.SetMode(models.FallbackModeSequential)
		fiberlog.Debugf("[%s] Fallback is explicitly disabled, using single provider only", reqID)
		return
	}

	// Fallback is enabled, configure the mode
	switch mode {
	case models.FallbackModeSequential:
		h.fallbackSvc.SetMode(models.FallbackModeSequential)
		fiberlog.Infof("[%s] Fallback enabled with mode: sequential", reqID)
	case models.FallbackModeRace:
		h.fallbackSvc.SetMode(models.FallbackModeRace)
		fiberlog.Infof("[%s] Fallback enabled with mode: race", reqID)
	default:
		// Unknown mode, default to parallel with warning
		h.fallbackSvc.SetMode(models.FallbackModeRace)
		fiberlog.Warnf("[%s] Fallback enabled with unknown mode '%s', using default: parallel", reqID, mode)
	}
}

// createManualProtocolResponse creates a manual protocol response when a model is explicitly provided
func (h *CompletionHandler) createManualProtocolResponse(
	req *models.ChatCompletionRequest,
	requestID string,
) (*models.ProtocolResponse, string, error) {
	modelSpec := string(req.Model)

	// Parse provider:model format
	provider, modelName, err := h.parseProviderModel(modelSpec)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to parse model specification '%s': %v", requestID, modelSpec, err)
		return nil, "", fmt.Errorf("invalid model format '%s', expected 'provider:model': %w", modelSpec, err)
	}

	fiberlog.Infof("[%s] Parsed model specification '%s' -> provider: %s, model: %s", requestID, modelSpec, provider, modelName)

	// Convert to OpenAI parameters
	openAIParams := req.ToOpenAIParams()
	if openAIParams == nil {
		return nil, "", fmt.Errorf("failed to convert request to OpenAI parameters")
	}

	// Create standard LLM info
	standardInfo := &models.StandardLLMInfo{
		Provider:     provider,
		Model:        modelName,
		Parameters:   *openAIParams,
		Alternatives: []models.Alternative{}, // No alternatives for manual override
	}

	// Create protocol response
	response := &models.ProtocolResponse{
		Protocol: models.ProtocolStandardLLM,
		Standard: standardInfo,
		Minion:   nil, // Manual override always uses standard protocol
	}

	return response, "manual_override", nil
}

// parseProviderModel parses a "provider:model" format string
func (h *CompletionHandler) parseProviderModel(modelSpec string) (provider, model string, err error) {
	parts := strings.SplitN(modelSpec, ":", 2)
	if len(parts) != 2 {
		return "", "", fmt.Errorf("model specification must be in 'provider:model' format")
	}

	provider = strings.TrimSpace(parts[0])
	model = strings.TrimSpace(parts[1])

	if provider == "" {
		return "", "", fmt.Errorf("provider cannot be empty")
	}

	if model == "" {
		return "", "", fmt.Errorf("model cannot be empty")
	}

	return provider, model, nil
}
