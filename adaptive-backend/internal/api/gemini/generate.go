package gemini

import (
	"context"
	"fmt"

	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/cache"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/fallback"
	"adaptive-backend/internal/services/gemini/generate"
	"adaptive-backend/internal/services/model_router"
	"adaptive-backend/internal/services/stream/stream_simulator"
	"adaptive-backend/internal/utils"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// GenerateHandler handles Gemini GenerateContent API requests using dedicated Gemini services
type GenerateHandler struct {
	cfg             *config.Config
	requestSvc      *generate.RequestService
	generateSvc     *generate.GenerateService
	responseSvc     *generate.ResponseService
	modelRouter     *model_router.ModelRouter
	promptCache     *cache.GeminiPromptCache
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker
	fallbackService *fallback.FallbackService
}

// NewGenerateHandler creates a new GenerateHandler with Gemini-specific services
func NewGenerateHandler(
	cfg *config.Config,
	modelRouter *model_router.ModelRouter,
	promptCache *cache.GeminiPromptCache,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
) *GenerateHandler {
	return &GenerateHandler{
		cfg:             cfg,
		requestSvc:      generate.NewRequestService(),
		generateSvc:     generate.NewGenerateService(),
		responseSvc:     generate.NewResponseService(modelRouter),
		modelRouter:     modelRouter,
		promptCache:     promptCache,
		circuitBreakers: circuitBreakers,
		fallbackService: fallback.NewFallbackService(cfg),
	}
}

// Generate handles the Gemini GenerateContent API HTTP request (non-streaming)
func (h *GenerateHandler) Generate(c *fiber.Ctx) error {
	requestID := h.requestSvc.GetRequestID(c)
	requestType := "GenerateContent"
	fiberlog.Infof("[%s] Starting Gemini %s API request from %s", requestID, requestType, c.IP())

	// Parse and validate request
	req, err := h.requestSvc.ParseRequest(c)
	if err != nil {
		fiberlog.Warnf("[%s] Request parsing failed: %v", requestID, err)
		return h.responseSvc.HandleError(c, models.NewValidationError("invalid request", err), requestID)
	}

	// Extract model from route parameter if present (for Gemini SDK compatibility)
	if routeModel := c.Params("model"); routeModel != "" {
		fiberlog.Debugf("[%s] Model found in route parameter: %s, overriding request model", requestID, routeModel)
		req.Model = routeModel
	}

	fiberlog.Debugf("[%s] Request parsed successfully - model: %s", requestID, req.Model)

	// Resolve configuration
	resolvedConfig, err := h.cfg.ResolveConfigFromGeminiRequest(req)
	if err != nil {
		fiberlog.Errorf("[%s] Config resolution failed: %v", requestID, err)
		return h.responseSvc.HandleError(c, fmt.Errorf("failed to resolve config: %w", err), requestID)
	}

	// Check prompt cache first
	if cachedResponse, cacheSource, found := h.checkPromptCache(c.UserContext(), req, &resolvedConfig.PromptCache, requestID); found {
		fiberlog.Infof("[%s] prompt cache hit (%s) - returning cached response", requestID, cacheSource)
		return c.JSON(cachedResponse)
	}

	// If a model is specified, try to directly route to the appropriate provider
	if req.Model != "" {
		fiberlog.Debugf("[%s] Model specified: %s, attempting direct routing", requestID, req.Model)

		// Parse provider and model from the model specification (expecting "provider:model" format)
		provider, parsedModel, err := utils.ParseProviderModel(req.Model)
		if err != nil {
			fiberlog.Debugf("[%s] Failed to parse model specification %s: %v, falling back to intelligent routing", requestID, req.Model, err)
			// Fall through to intelligent routing below instead of returning error
		} else {
			// Update the request with the parsed model name
			req.Model = parsedModel

			fiberlog.Infof("[%s] User-specified model %s routed to provider %s", requestID, req.Model, provider)

			// Get provider configuration
			providers := resolvedConfig.GetProviders("generate")
			providerConfig, exists := providers[provider]
			if !exists {
				return h.responseSvc.HandleError(c, models.NewValidationError(fmt.Sprintf("provider %s not configured", provider), nil), requestID)
			}

			// Direct execution - no fallback for user-specified models
			err = h.executeProviderRequest(c, req, provider, providerConfig, false, requestID, "")
			if err != nil {
				return h.responseSvc.HandleError(c, err, requestID)
			}

			// Store successful response in semantic cache for user-specified models
			modelResp := &models.ModelSelectionResponse{
				Provider: provider,
				Model:    parsedModel,
			}
			h.storeSuccessfulSemanticCache(c.UserContext(), req, modelResp, requestID)

			return nil
		}
	}

	// If no model is specified, use model router for selection WITH fallback
	fiberlog.Debugf("[%s] No model specified, using model router for selection with fallback", requestID)

	// Extract prompt for routing
	prompt, err := utils.ExtractPromptFromGeminiContents(req.Contents)
	if err != nil {
		fiberlog.Warnf("[%s] Failed to extract prompt for routing: %v", requestID, err)
		return h.responseSvc.HandleError(c, models.NewValidationError("failed to extract prompt for routing: "+err.Error(), err), requestID)
	}

	// Use model router to select best model WITH CIRCUIT BREAKERS
	userID := "anonymous"
	toolCall := utils.ExtractToolCallsFromGeminiContents(req.Contents)

	modelResp, cacheSource, err := h.modelRouter.SelectModelWithCache(
		c.UserContext(),
		prompt, userID, requestID, &resolvedConfig.Services.ModelRouter, h.circuitBreakers,
		req.Tools, toolCall,
	)
	if err != nil {
		fiberlog.Errorf("[%s] Model router selection failed: %v", requestID, err)
		return h.responseSvc.HandleError(c, err, requestID)
	}

	// Update request with selected model
	req.Model = modelResp.Model
	fiberlog.Infof("[%s] Model router selected - provider: %s, model: %s (with %d alternatives)",
		requestID, modelResp.Provider, modelResp.Model, len(modelResp.Alternatives))

	// Try primary provider first
	primary := models.Alternative{
		Provider: modelResp.Provider,
		Model:    modelResp.Model,
	}
	executeFunc := h.createExecuteFunc(req, false, cacheSource)

	fiberlog.Infof("[%s] Trying primary provider: %s/%s", requestID, primary.Provider, primary.Model)
	err = executeFunc(c, primary, requestID)

	if err == nil {
		// Primary succeeded
		fiberlog.Infof("[%s] ✅ Primary provider succeeded: %s/%s", requestID, primary.Provider, primary.Model)
		return nil
	}

	// Primary failed - check if we have alternatives
	if len(modelResp.Alternatives) == 0 {
		fiberlog.Errorf("[%s] ❌ Primary provider failed and no alternatives available: %v", requestID, err)
		return err
	}

	// Use fallback service with alternatives only
	fiberlog.Warnf("[%s] ⚠️  Primary provider failed: %v", requestID, err)
	fiberlog.Infof("[%s] Using fallback with %d alternatives", requestID, len(modelResp.Alternatives))

	fallbackConfig := h.fallbackService.GetFallbackConfig(req.Fallback)
	return h.fallbackService.Execute(c, modelResp.Alternatives, fallbackConfig, executeFunc, requestID, false)
}

// StreamGenerate handles the Gemini GenerateContent API HTTP request (streaming)
func (h *GenerateHandler) StreamGenerate(c *fiber.Ctx) error {
	requestID := h.requestSvc.GetRequestID(c)
	requestType := "StreamGenerateContent"
	fiberlog.Infof("[%s] Starting Gemini %s API request from %s", requestID, requestType, c.IP())

	// Parse and validate request
	req, err := h.requestSvc.ParseRequest(c)
	if err != nil {
		fiberlog.Warnf("[%s] Request parsing failed: %v", requestID, err)
		return h.responseSvc.HandleError(c, models.NewValidationError("invalid request", err), requestID)
	}

	// Extract model from route parameter if present (for Gemini SDK compatibility)
	if routeModel := c.Params("model"); routeModel != "" {
		fiberlog.Debugf("[%s] Model found in route parameter: %s, overriding request model", requestID, routeModel)
		req.Model = routeModel
	}

	fiberlog.Debugf("[%s] Request parsed successfully - model: %s", requestID, req.Model)

	// Resolve configuration
	resolvedConfig, err := h.cfg.ResolveConfigFromGeminiRequest(req)
	if err != nil {
		fiberlog.Errorf("[%s] Config resolution failed: %v", requestID, err)
		return h.responseSvc.HandleError(c, fmt.Errorf("failed to resolve config: %w", err), requestID)
	}

	// Check prompt cache first
	if cachedResponse, cacheSource, found := h.checkPromptCache(c.UserContext(), req, &resolvedConfig.PromptCache, requestID); found {
		fiberlog.Infof("[%s] prompt cache hit (%s) - streaming cached response", requestID, cacheSource)
		// Stream the cached response instead of returning as JSON
		return stream_simulator.StreamGeminiCachedResponse(c, cachedResponse, requestID)
	}

	// If a model is specified, try to directly route to the appropriate provider
	if req.Model != "" {
		fiberlog.Debugf("[%s] Model specified: %s, attempting direct routing", requestID, req.Model)

		// Parse provider and model from the model specification (expecting "provider:model" format)
		provider, parsedModel, err := utils.ParseProviderModel(req.Model)
		if err != nil {
			fiberlog.Debugf("[%s] Failed to parse model specification %s: %v, falling back to intelligent routing", requestID, req.Model, err)
			// Fall through to intelligent routing below instead of returning error
		} else {
			// Update the request with the parsed model name
			req.Model = parsedModel

			fiberlog.Infof("[%s] User-specified model %s routed to provider %s", requestID, req.Model, provider)

			// Get provider configuration
			providers := resolvedConfig.GetProviders("generate")
			providerConfig, exists := providers[provider]
			if !exists {
				return h.responseSvc.HandleError(c, models.NewValidationError(fmt.Sprintf("provider %s not configured", provider), nil), requestID)
			}

			// Direct execution - no fallback for user-specified models
			err = h.executeProviderRequest(c, req, provider, providerConfig, true, requestID, "")
			if err != nil {
				return h.responseSvc.HandleError(c, err, requestID)
			}

			// Store successful response in semantic cache for user-specified models
			modelResp := &models.ModelSelectionResponse{
				Provider: provider,
				Model:    parsedModel,
			}
			h.storeSuccessfulSemanticCache(c.UserContext(), req, modelResp, requestID)

			return nil
		}
	}

	// If no model is specified, use model router for selection WITH fallback
	fiberlog.Debugf("[%s] No model specified, using model router for selection with fallback", requestID)

	// Extract prompt for routing
	prompt, err := utils.ExtractPromptFromGeminiContents(req.Contents)
	if err != nil {
		fiberlog.Warnf("[%s] Failed to extract prompt for routing: %v", requestID, err)
		return h.responseSvc.HandleError(c, models.NewValidationError("failed to extract prompt for routing: "+err.Error(), err), requestID)
	}

	// Use model router to select best model WITH CIRCUIT BREAKERS
	userID := "anonymous"
	toolCall := utils.ExtractToolCallsFromGeminiContents(req.Contents)

	modelResp, cacheSource, err := h.modelRouter.SelectModelWithCache(
		c.UserContext(),
		prompt, userID, requestID, &resolvedConfig.Services.ModelRouter, h.circuitBreakers,
		req.Tools, toolCall,
	)
	if err != nil {
		fiberlog.Errorf("[%s] Model router selection failed: %v", requestID, err)
		return h.responseSvc.HandleError(c, err, requestID)
	}

	// Update request with selected model
	req.Model = modelResp.Model
	fiberlog.Infof("[%s] Model router selected - provider: %s, model: %s (with %d alternatives)",
		requestID, modelResp.Provider, modelResp.Model, len(modelResp.Alternatives))

	// Try primary provider first
	primary := models.Alternative{
		Provider: modelResp.Provider,
		Model:    modelResp.Model,
	}
	executeFunc := h.createExecuteFunc(req, true, cacheSource)

	fiberlog.Infof("[%s] Trying primary provider: %s/%s", requestID, primary.Provider, primary.Model)
	err = executeFunc(c, primary, requestID)

	if err == nil {
		// Primary succeeded
		fiberlog.Infof("[%s] ✅ Primary provider succeeded: %s/%s", requestID, primary.Provider, primary.Model)
		return nil
	}

	// Primary failed - check if we have alternatives
	if len(modelResp.Alternatives) == 0 {
		fiberlog.Errorf("[%s] ❌ Primary provider failed and no alternatives available: %v", requestID, err)
		return err
	}

	// Use fallback service with alternatives only
	fiberlog.Warnf("[%s] ⚠️  Primary provider failed: %v", requestID, err)
	fiberlog.Infof("[%s] Using fallback with %d alternatives", requestID, len(modelResp.Alternatives))

	fallbackConfig := h.fallbackService.GetFallbackConfig(req.Fallback)
	return h.fallbackService.Execute(c, modelResp.Alternatives, fallbackConfig, executeFunc, requestID, true)
}

// checkCircuitBreaker validates circuit breaker state for the provider
func (h *GenerateHandler) checkCircuitBreaker(provider, requestID string) error {
	cb := h.circuitBreakers[provider]
	if cb != nil && !cb.CanExecute() {
		fiberlog.Warnf("[%s] Circuit breaker is open for provider %s", requestID, provider)
		return models.NewProviderError(provider, "circuit breaker open", nil)
	}
	return nil
}

// executeWithCircuitBreaker executes the request and updates circuit breaker state
func (h *GenerateHandler) executeWithCircuitBreaker(
	c *fiber.Ctx,
	req *models.GeminiGenerateRequest,
	provider string,
	providerConfig models.ProviderConfig,
	isStreaming bool,
	requestID string,
	cacheSource string,
) error {
	cb := h.circuitBreakers[provider]

	// Execute the request with concrete types
	if isStreaming {
		return h.executeStreamingWithCircuitBreaker(c, req, provider, providerConfig, requestID, cb, cacheSource)
	}
	return h.executeNonStreamingWithCircuitBreaker(c, req, provider, providerConfig, requestID, cb, cacheSource)
}

// executeNonStreamingWithCircuitBreaker handles non-streaming requests
func (h *GenerateHandler) executeNonStreamingWithCircuitBreaker(
	c *fiber.Ctx,
	req *models.GeminiGenerateRequest,
	provider string,
	providerConfig models.ProviderConfig,
	requestID string,
	cb *circuitbreaker.CircuitBreaker,
	cacheSource string,
) error {
	// Execute the non-streaming request
	response, err := h.generateSvc.HandleGeminiNonStreamingProvider(c, req, providerConfig, requestID)
	if err != nil {
		// Record failure in circuit breaker
		if cb != nil {
			cb.RecordFailure()
			fiberlog.Warnf("[%s] 🔴 Circuit breaker recorded FAILURE for provider %s (non-streaming)", requestID, provider)
		}
		fiberlog.Errorf("[%s] Non-streaming provider request failed: %v", requestID, err)
		return err
	}

	// Handle the non-streaming response with proper cache source
	err = h.responseSvc.HandleNonStreamingResponse(c, response, requestID, provider, cacheSource)
	if err != nil {
		// Record failure in circuit breaker
		if cb != nil {
			cb.RecordFailure()
			fiberlog.Warnf("[%s] 🔴 Circuit breaker recorded FAILURE for provider %s (response handling)", requestID, provider)
		}
		fiberlog.Errorf("[%s] Non-streaming response handling failed: %v", requestID, err)
		return err
	}

	// Record success in circuit breaker
	if cb != nil {
		cb.RecordSuccess()
		fiberlog.Infof("[%s] 🟢 Circuit breaker recorded SUCCESS for provider %s (non-streaming)", requestID, provider)
	}

	fiberlog.Infof("[%s] Gemini GenerateContent request completed successfully", requestID)
	return nil
}

// executeStreamingWithCircuitBreaker handles streaming requests
func (h *GenerateHandler) executeStreamingWithCircuitBreaker(
	c *fiber.Ctx,
	req *models.GeminiGenerateRequest,
	provider string,
	providerConfig models.ProviderConfig,
	requestID string,
	cb *circuitbreaker.CircuitBreaker,
	cacheSource string,
) error {
	// Execute the streaming request
	streamIter, err := h.generateSvc.HandleGeminiStreamingProvider(c, req, providerConfig, requestID)
	if err != nil {
		// Record failure in circuit breaker
		if cb != nil {
			cb.RecordFailure()
			fiberlog.Warnf("[%s] 🔴 Circuit breaker recorded FAILURE for provider %s (streaming)", requestID, provider)
		}
		fiberlog.Errorf("[%s] Streaming provider request failed: %v", requestID, err)
		return h.responseSvc.HandleError(c, err, requestID)
	}

	// Handle the streaming response with proper cache source
	err = h.responseSvc.HandleStreamingResponse(c, streamIter, requestID, provider, cacheSource)
	if err != nil {
		// Record failure in circuit breaker
		if cb != nil {
			cb.RecordFailure()
			fiberlog.Warnf("[%s] 🔴 Circuit breaker recorded FAILURE for provider %s (streaming)", requestID, provider)
		}
		fiberlog.Errorf("[%s] Streaming response handling failed: %v", requestID, err)
		return h.responseSvc.HandleError(c, err, requestID)
	}

	// Record success in circuit breaker
	if cb != nil {
		cb.RecordSuccess()
		fiberlog.Infof("[%s] 🟢 Circuit breaker recorded SUCCESS for provider %s (streaming)", requestID, provider)
	}

	fiberlog.Infof("[%s] Gemini StreamGenerateContent request completed successfully", requestID)
	return nil
}

// checkPromptCache checks if prompt cache is enabled and returns cached response if found
func (h *GenerateHandler) checkPromptCache(ctx context.Context, req *models.GeminiGenerateRequest, promptCacheConfig *models.CacheConfig, requestID string) (*models.GeminiGenerateContentResponse, string, bool) {
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

// createExecuteFunc creates an execution function for the fallback service
func (h *GenerateHandler) createExecuteFunc(
	req *models.GeminiGenerateRequest,
	isStreaming bool,
	cacheSource string,
) models.ExecutionFunc {
	return func(c *fiber.Ctx, provider models.Alternative, reqID string) error {
		// Get provider configuration from resolved config
		resolvedConfig, err := h.cfg.ResolveConfigFromGeminiRequest(req)
		if err != nil {
			return fmt.Errorf("failed to resolve config: %w", err)
		}

		providers := resolvedConfig.GetProviders("generate")
		providerConfig, exists := providers[provider.Provider]
		if !exists {
			return fmt.Errorf("provider %s not configured", provider.Provider)
		}

		// Create a copy to avoid race conditions when mutating req.Model
		reqCopy := *req
		reqCopy.Model = provider.Model

		// Call the generate service and handle retryable errors
		err = h.executeProviderRequest(c, &reqCopy, provider.Provider, providerConfig, isStreaming, reqID, cacheSource)
		// Check if the error is a retryable provider error that should trigger fallback
		if err != nil {
			if appErr, ok := err.(*models.AppError); ok && appErr.Type == models.ErrorTypeProvider && appErr.Retryable {
				// Return the provider error to trigger fallback
				return err
			}
			// For non-retryable errors, return original AppError or create one with Retryable=false
			if appErr, ok := err.(*models.AppError); ok {
				// Return the original AppError to preserve the concrete type and Retryable=false signal
				return appErr
			}
			// For non-AppError types, create a non-retryable AppError
			return &models.AppError{
				Type:      models.ErrorTypeProvider,
				Message:   fmt.Sprintf("non-retryable error: %v", err),
				Retryable: false,
			}
		}

		// Store successful response in semantic cache
		modelResp := &models.ModelSelectionResponse{
			Provider: provider.Provider,
			Model:    provider.Model,
		}
		h.storeSuccessfulSemanticCache(c.UserContext(), &reqCopy, modelResp, reqID)

		return nil
	}
}

// executeProviderRequest handles the core provider request execution
func (h *GenerateHandler) executeProviderRequest(
	c *fiber.Ctx,
	req *models.GeminiGenerateRequest,
	provider string,
	providerConfig models.ProviderConfig,
	isStreaming bool,
	requestID string,
	cacheSource string,
) error {
	// Check circuit breaker state
	if err := h.checkCircuitBreaker(provider, requestID); err != nil {
		return err
	}

	// Execute request with circuit breaker tracking
	return h.executeWithCircuitBreaker(c, req, provider, providerConfig, isStreaming, requestID, cacheSource)
}

// storeSuccessfulSemanticCache stores successful responses in semantic cache
func (h *GenerateHandler) storeSuccessfulSemanticCache(ctx context.Context, req *models.GeminiGenerateRequest, modelResp *models.ModelSelectionResponse, requestID string) {
	h.responseSvc.StoreSuccessfulSemanticCache(ctx, req, modelResp, requestID)
}
