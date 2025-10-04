package gemini

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/gemini/count_tokens"
	"adaptive-backend/internal/services/model_router"
	"adaptive-backend/internal/utils"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"google.golang.org/genai"
)

// CountTokensHandler handles Gemini CountTokens API requests
type CountTokensHandler struct {
	cfg             *config.Config
	requestSvc      *count_tokens.RequestService
	countTokensSvc  *count_tokens.CountTokensService
	responseSvc     *count_tokens.ResponseService
	modelRouter     *model_router.ModelRouter
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker
}

// NewCountTokensHandler creates a new CountTokensHandler
func NewCountTokensHandler(
	cfg *config.Config,
	modelRouter *model_router.ModelRouter,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
) *CountTokensHandler {
	return &CountTokensHandler{
		cfg:             cfg,
		requestSvc:      count_tokens.NewRequestService(),
		countTokensSvc:  count_tokens.NewCountTokensService(),
		responseSvc:     count_tokens.NewResponseService(),
		modelRouter:     modelRouter,
		circuitBreakers: circuitBreakers,
	}
}

// CountTokens handles the Gemini CountTokens API HTTP request
func (h *CountTokensHandler) CountTokens(c *fiber.Ctx) error {
	requestID := h.requestSvc.GetRequestID(c)
	fiberlog.Infof("[%s] Starting Gemini CountTokens API request from %s", requestID, c.IP())

	// Parse request body
	req, err := h.requestSvc.ParseRequest(c)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to parse request: %v", requestID, err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": fiber.Map{
				"message": "Invalid request format",
				"type":    "invalid_request_error",
			},
		})
	}

	// Determine provider and model
	provider, model, err := h.resolveProviderAndModel(c, req, requestID)
	if err != nil {
		return err // Already formatted as fiber error
	}

	// Get provider configuration
	providerConfig, err := h.getProviderConfig(model, req.Contents, provider, requestID)
	if err != nil {
		return err // Already formatted as fiber error
	}

	// Validate circuit breaker
	if err := h.checkCircuitBreaker(provider, requestID); err != nil {
		return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{
			"error": fiber.Map{
				"message": "Service temporarily unavailable",
				"type":    "service_unavailable",
			},
		})
	}

	// Execute count tokens request
	response, err := h.countTokensSvc.HandleGeminiCountTokensProvider(c, req.Contents, model, providerConfig, requestID)
	if err != nil {
		fiberlog.Errorf("[%s] Count tokens request failed: %v", requestID, err)
		h.recordCircuitBreakerFailure(provider)
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": fiber.Map{
				"message": "Count tokens failed",
				"type":    "server_error",
			},
		})
	}

	// Record success
	h.recordCircuitBreakerSuccess(provider)

	// Send response
	if err := h.responseSvc.SendNonStreamingResponse(c, response, requestID); err != nil {
		fiberlog.Errorf("[%s] Failed to send response: %v", requestID, err)
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": fiber.Map{
				"message": "Failed to send response",
				"type":    "server_error",
			},
		})
	}

	return nil
}

// resolveProviderAndModel determines the provider and model to use
func (h *CountTokensHandler) resolveProviderAndModel(
	c *fiber.Ctx,
	req *count_tokens.CountTokensRequest,
	requestID string,
) (provider, model string, err error) {
	modelParam := c.Params("model")

	// If no model parameter, use model router
	if modelParam == "" {
		return h.selectModelViaRouter(c, req, requestID)
	}

	// Try parsing model parameter with strict format (provider:model)
	provider, model, parseErr := utils.ParseProviderModel(modelParam)
	if parseErr == nil {
		fiberlog.Debugf("[%s] Using parsed model: %s:%s", requestID, provider, model)
		return provider, model, nil
	}

	// If parsing fails, check if it's a special routing model or fallback to model router
	fiberlog.Debugf("[%s] Failed to parse model '%s': %v, using model router", requestID, modelParam, parseErr)
	return h.selectModelViaRouter(c, req, requestID)
}

// selectModelViaRouter uses the model router to select the optimal model
func (h *CountTokensHandler) selectModelViaRouter(
	c *fiber.Ctx,
	req *count_tokens.CountTokensRequest,
	requestID string,
) (provider, model string, err error) {
	fiberlog.Infof("[%s] Using model router for intelligent selection", requestID)

	// Extract prompt from contents
	prompt, err := utils.ExtractPromptFromGeminiContents(req.Contents)
	if err != nil {
		fiberlog.Warnf("[%s] Failed to extract prompt: %v", requestID, err)
		return "", "", c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": fiber.Map{
				"message": "Failed to extract prompt for routing",
				"type":    "invalid_request_error",
			},
		})
	}

	// Resolve configuration
	geminiReq := &models.GeminiGenerateRequest{Contents: req.Contents}
	resolvedConfig, err := h.cfg.ResolveConfigFromGeminiCountTokensRequest(geminiReq)
	if err != nil {
		fiberlog.Errorf("[%s] Config resolution failed: %v", requestID, err)
		return "", "", c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": fiber.Map{
				"message": "Configuration error",
				"type":    "server_error",
			},
		})
	}

	// Call model router
	toolCall := utils.ExtractToolCallsFromGeminiContents(req.Contents)
	routingDecision, _, err := h.modelRouter.SelectModelWithCache(
		c.UserContext(),
		prompt,
		"anonymous", // userID
		requestID,
		&resolvedConfig.Services.ModelRouter,
		h.circuitBreakers,
		nil, // tools
		toolCall,
	)
	if err != nil {
		fiberlog.Errorf("[%s] Model router failed: %v", requestID, err)
		return "", "", c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": fiber.Map{
				"message": "Model selection failed",
				"type":    "server_error",
			},
		})
	}

	provider = routingDecision.Provider
	model = routingDecision.Model
	fiberlog.Infof("[%s] Model router selected: %s:%s", requestID, provider, model)
	return provider, model, nil
}

// getProviderConfig retrieves and validates provider configuration
func (h *CountTokensHandler) getProviderConfig(
	model string,
	contents []*genai.Content,
	provider string,
	requestID string,
) (models.ProviderConfig, error) {
	geminiReq := &models.GeminiGenerateRequest{
		Model:    model,
		Contents: contents,
	}

	resolvedConfig, err := h.cfg.ResolveConfigFromGeminiCountTokensRequest(geminiReq)
	if err != nil {
		fiberlog.Errorf("[%s] Config resolution failed: %v", requestID, err)
		return models.ProviderConfig{}, err
	}

	providers := resolvedConfig.GetProviders("count_tokens")
	providerConfig, exists := providers[provider]
	if !exists {
		fiberlog.Errorf("[%s] Provider %s not configured", requestID, provider)
		return models.ProviderConfig{}, models.NewProviderError(provider, "not configured", nil)
	}

	return providerConfig, nil
}

// checkCircuitBreaker validates circuit breaker state
func (h *CountTokensHandler) checkCircuitBreaker(provider, requestID string) error {
	cb := h.circuitBreakers[provider]
	if cb != nil && !cb.CanExecute() {
		fiberlog.Warnf("[%s] Circuit breaker open for %s", requestID, provider)
		return models.NewProviderError(provider, "circuit breaker open", nil)
	}
	return nil
}

// recordCircuitBreakerSuccess records a successful request
func (h *CountTokensHandler) recordCircuitBreakerSuccess(provider string) {
	if cb := h.circuitBreakers[provider]; cb != nil {
		cb.RecordSuccess()
	}
}

// recordCircuitBreakerFailure records a failed request
func (h *CountTokensHandler) recordCircuitBreakerFailure(provider string) {
	if cb := h.circuitBreakers[provider]; cb != nil {
		cb.RecordFailure()
	}
}
