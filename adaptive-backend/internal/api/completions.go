package api

import (
	"context"
	"errors"
	"fmt"

	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/cache"
	"adaptive-backend/internal/services/chat/completions"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/format_adapter"
	"adaptive-backend/internal/services/model_router"
	"adaptive-backend/internal/stream/stream_simulator"
	"adaptive-backend/internal/utils"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go/v2/shared"
)

// Sentinel errors for proper HTTP status code mapping
var (
	ErrInvalidModelSpec = errors.New("invalid model specification")
)

// CompletionHandler handles chat completions end-to-end.
// It manages the lifecycle of chat completion requests, including provider selection,
// fallback handling, and response processing.
type CompletionHandler struct {
	cfg             *config.Config
	reqSvc          *completions.RequestService
	respSvc         *completions.ResponseService
	completionSvc   *completions.CompletionService
	modelRouter     *model_router.ModelRouter
	promptCache     *cache.OpenAIPromptCache
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker
}

// NewCompletionHandler wires up dependencies and initializes the completion handler.
func NewCompletionHandler(
	cfg *config.Config,
	reqSvc *completions.RequestService,
	respSvc *completions.ResponseService,
	completionSvc *completions.CompletionService,
	modelRouter *model_router.ModelRouter,
	promptCache *cache.OpenAIPromptCache,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
) *CompletionHandler {
	return &CompletionHandler{
		cfg:             cfg,
		reqSvc:          reqSvc,
		respSvc:         respSvc,
		completionSvc:   completionSvc,
		modelRouter:     modelRouter,
		promptCache:     promptCache,
		circuitBreakers: circuitBreakers,
	}
}

// ChatCompletion handles the chat completion HTTP request.
// It processes the request through provider selection, parameter configuration,
// and response handling with circuit breaking for reliability.
func (h *CompletionHandler) ChatCompletion(c *fiber.Ctx) error {
	reqID := h.reqSvc.GetRequestID(c)
	fiberlog.Infof("[%s] starting chat completion request", reqID)

	// Parse request first to get user ID from the request body
	req, err := h.reqSvc.ParseChatCompletionRequest(c)
	if err != nil {
		return h.respSvc.HandleBadRequest(c, err.Error(), reqID)
	}

	// Get userID from request
	userID := "anonymous"
	if req.User.Value != "" {
		userID = req.User.Value
	}
	isStream := req.Stream

	// Resolve config by merging YAML config with request overrides (single source of truth)
	resolvedConfig, err := h.cfg.ResolveConfig(req)
	if err != nil {
		return h.respSvc.HandleInternalError(c, fmt.Sprintf("failed to resolve config: %v", err), reqID)
	}

	// Check prompt cache first
	if cachedResponse, cacheSource, found := h.checkPromptCache(c.UserContext(), req, &resolvedConfig.PromptCache, reqID); found {
		fiberlog.Infof("[%s] prompt cache hit (%s) - returning cached response", reqID, cacheSource)
		if isStream {
			// Convert cached response to streaming format
			return stream_simulator.StreamOpenAICachedResponse(c, cachedResponse, reqID)
		}
		return c.JSON(cachedResponse)
	}

	resp, cacheSource, err := h.selectModel(
		c.UserContext(), req, userID, reqID, h.circuitBreakers, resolvedConfig,
	)
	if err != nil {
		// Check for invalid model specification error to return 400 instead of 500
		if errors.Is(err, ErrInvalidModelSpec) {
			return h.respSvc.HandleBadRequest(c, err.Error(), reqID)
		}
		return h.respSvc.HandleInternalError(c, err.Error(), reqID)
	}

	return h.completionSvc.HandleModel(c, req, resp, reqID, isStream, cacheSource, resolvedConfig)
}

// checkPromptCache checks if prompt cache is enabled and returns cached response if found
func (h *CompletionHandler) checkPromptCache(ctx context.Context, req *models.ChatCompletionRequest, promptCacheConfig *models.CacheConfig, requestID string) (*models.ChatCompletion, string, bool) {
	if !promptCacheConfig.Enabled {
		fiberlog.Debugf("[%s] prompt cache disabled", requestID)
		return nil, "", false
	}

	if h.promptCache == nil {
		fiberlog.Debugf("[%s] prompt cache service not available", requestID)
		return nil, "", false
	}

	return h.promptCache.Get(ctx, req, requestID)
}

// selectModel runs model selection and returns the chosen model response and cache source.
func (h *CompletionHandler) selectModel(
	ctx context.Context,
	req *models.ChatCompletionRequest,
	userID, requestID string,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
	resolvedConfig *config.Config,
) (
	resp *models.ModelSelectionResponse,
	cacheSource string,
	err error,
) {
	fiberlog.Infof("[%s] Starting model selection for user: %s", requestID, userID)

	// Check if model is explicitly provided (non-empty) - if so, use manual override
	if req.Model != "" {
		fiberlog.Infof("[%s] Model explicitly provided (%s), using manual override instead of model router", requestID, req.Model)
		return h.createManualModelResponse(req, requestID)
	}

	fiberlog.Debugf("[%s] No explicit model provided, proceeding with model router selection", requestID)

	// Check if the singleton adapter is available
	if format_adapter.AdaptiveToOpenAI == nil {
		return nil, "", fmt.Errorf("format_adapter.AdaptiveToOpenAI is not initialized")
	}

	// Convert to OpenAI parameters using singleton adapter
	openAIParams, err := format_adapter.AdaptiveToOpenAI.ConvertRequest(req)
	if err != nil {
		return nil, "", fmt.Errorf("failed to convert request to OpenAI parameters: %w", err)
	}

	// Extract prompt from messages
	prompt, err := utils.ExtractLastMessage(openAIParams.Messages)
	if err != nil {
		return nil, "", fmt.Errorf("failed to extract prompt: %w", err)
	}

	// Extract tool calls from the last message
	toolCall := utils.ExtractToolCallsFromLastMessage(openAIParams.Messages)

	resp, cacheSource, err = h.modelRouter.SelectModelWithCache(
		ctx,
		prompt, userID, requestID, &resolvedConfig.ModelRouter, circuitBreakers,
		req.Tools, toolCall,
	)
	if err != nil {
		fiberlog.Errorf("[%s] Model selection error: %v", requestID, err)
		return nil, "", fmt.Errorf("model selection failed: %w", err)
	}

	return resp, cacheSource, nil
}

// createManualModelResponse creates a manual model response when a model is explicitly provided
func (h *CompletionHandler) createManualModelResponse(
	req *models.ChatCompletionRequest,
	requestID string,
) (*models.ModelSelectionResponse, string, error) {
	modelSpec := string(req.Model)

	// Parse provider:model format
	provider, modelName, err := utils.ParseProviderModel(modelSpec)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to parse model specification '%s': %v", requestID, modelSpec, err)
		return nil, "", fmt.Errorf("%w: %s", ErrInvalidModelSpec, err.Error())
	}

	fiberlog.Infof("[%s] Parsed model specification '%s' -> provider: %s, model: %s", requestID, modelSpec, provider, modelName)

	// Check if the singleton adapter is available
	if format_adapter.AdaptiveToOpenAI == nil {
		return nil, "", fmt.Errorf("adaptive to OpenAI format adapter is not initialized")
	}

	// Convert to OpenAI parameters using singleton adapter
	openAIParams, err := format_adapter.AdaptiveToOpenAI.ConvertRequest(req)
	if err != nil {
		return nil, "", fmt.Errorf("failed to convert request to OpenAI parameters: %w", err)
	}

	// Ensure the OpenAI parameter's Model is set to the provider-stripped model name
	openAIParams.Model = shared.ChatModel(modelName)

	// Create simplified model selection response
	response := &models.ModelSelectionResponse{
		Provider:     provider,
		Model:        modelName,
		Alternatives: []models.Alternative{}, // No alternatives for manual override
	}

	return response, "manual_override", nil
}
