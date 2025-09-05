package api

import (
	"fmt"

	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/anthropic/messages"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/fallback"
	"adaptive-backend/internal/services/model_router"
	"adaptive-backend/internal/utils"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// MessagesHandler handles Anthropic Messages API requests using dedicated Anthropic services
type MessagesHandler struct {
	cfg             *config.Config
	requestSvc      *messages.RequestService
	messagesSvc     *messages.MessagesService
	responseSvc     *messages.ResponseService
	modelRouter     *model_router.ModelRouter
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker
	fallbackService *fallback.FallbackService
}

// NewMessagesHandler creates a new MessagesHandler with Anthropic-specific services
func NewMessagesHandler(
	cfg *config.Config,
	modelRouter *model_router.ModelRouter,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
) *MessagesHandler {
	return &MessagesHandler{
		cfg:             cfg,
		requestSvc:      messages.NewRequestService(),
		messagesSvc:     messages.NewMessagesService(),
		responseSvc:     messages.NewResponseService(),
		modelRouter:     modelRouter,
		circuitBreakers: circuitBreakers,
		fallbackService: fallback.NewFallbackService(cfg),
	}
}

// Messages handles the Anthropic Messages API HTTP request
func (h *MessagesHandler) Messages(c *fiber.Ctx) error {
	requestID := h.requestSvc.GetRequestID(c)
	fiberlog.Infof("[%s] Starting Anthropic Messages API request from %s", requestID, c.IP())

	// Parse Anthropic Messages request
	req, err := h.requestSvc.ParseRequest(c)
	if err != nil {
		fiberlog.Warnf("[%s] Request parsing failed: %v", requestID, err)
		return h.responseSvc.HandleBadRequest(c, err.Error(), requestID)
	}
	fiberlog.Debugf("[%s] Request parsed successfully - model: %s, messages: %d", requestID, req.Model, len(req.Messages))

	// Resolve config by merging YAML config with request overrides (single source of truth)
	resolvedConfig, err := h.cfg.ResolveConfigFromAnthropicRequest(req)
	if err != nil {
		fiberlog.Errorf("[%s] Config resolution failed: %v", requestID, err)
		return h.responseSvc.HandleError(c, fmt.Errorf("failed to resolve config: %w", err), requestID)
	}
	fiberlog.Debugf("[%s] Configuration resolved successfully", requestID)

	// Check if streaming is requested
	isStreaming := req.Stream != nil && *req.Stream
	fiberlog.Debugf("[%s] Request type: streaming=%t", requestID, isStreaming)

	// If a model is specified, directly route to the appropriate provider without model router
	if req.Model != "" {
		modelStr := string(req.Model)
		fiberlog.Debugf("[%s] Model specified: %s, directly routing to provider", requestID, modelStr)

		// Parse provider and model from the model specification (expecting "provider:model" format)
		provider, parsedModel, err := utils.ParseProviderModel(modelStr)
		if err != nil {
			fiberlog.Errorf("[%s] Failed to parse model specification %s: %v", requestID, modelStr, err)
			return h.responseSvc.HandleBadRequest(c, fmt.Sprintf("invalid model specification '%s': must be in 'provider:model' format", modelStr), requestID)
		}

		// Update the request with the parsed model name
		req.Model = anthropic.Model(parsedModel)

		fiberlog.Infof("[%s] User-specified model %s routed to provider %s", requestID, modelStr, provider)

		// Get provider configuration
		providers := resolvedConfig.GetProviders("messages")
		providerConfig, exists := providers[provider]
		if !exists {
			return h.responseSvc.HandleProviderNotConfigured(c, provider, requestID)
		}

		// Direct execution - no fallback for user-specified models
		return h.messagesSvc.HandleProviderRequest(c, req, provider, providerConfig, isStreaming, requestID, h.responseSvc)
	}

	// If no model is specified, use model router for selection WITH fallback
	if req.Model == "" {
		fiberlog.Debugf("[%s] No model specified, using model router for selection with fallback", requestID)

		// Extract prompt for routing
		prompt, err := utils.ExtractPromptFromAnthropicMessages(req.Messages)
		if err != nil {
			fiberlog.Warnf("[%s] Failed to extract prompt for routing: %v", requestID, err)
			return h.responseSvc.HandleBadRequest(c, "failed to extract prompt for routing: "+err.Error(), requestID)
		}

		// Use model router to select best model WITH CIRCUIT BREAKERS
		userID := "anonymous"
		toolCall := utils.ExtractToolCallsFromAnthropicMessages(req.Messages)

		modelResp, _, err := h.modelRouter.SelectModelWithCache(
			c.UserContext(),
			prompt, userID, requestID, &resolvedConfig.ModelRouter, h.circuitBreakers,
			req.Tools, toolCall,
		)
		if err != nil {
			fiberlog.Errorf("[%s] Model router selection failed: %v", requestID, err)
			return h.responseSvc.HandleError(c, err, requestID)
		}

		// Use fallback service with model router response (system-selected models get fallback)
		fallbackConfig := h.fallbackService.GetFallbackConfig(req.Fallback)

		// Create provider list with primary and alternatives from model router
		providers := []models.Alternative{{
			Provider: modelResp.Provider,
			Model:    modelResp.Model,
		}}
		providers = append(providers, modelResp.Alternatives...)

		// Update request with selected model
		req.Model = anthropic.Model(modelResp.Model)
		fiberlog.Infof("[%s] Model router selected - provider: %s, model: %s (with %d alternatives)",
			requestID, modelResp.Provider, modelResp.Model, len(modelResp.Alternatives))

		return h.fallbackService.Execute(c, providers, fallbackConfig, h.createExecuteFunc(req, isStreaming), requestID, isStreaming)
	}

	// If we reach here, something went wrong with the logic above
	return h.responseSvc.HandleError(c, fmt.Errorf("invalid request state - no model handling path matched"), requestID)
}

// createExecuteFunc creates an execution function for the fallback service
func (h *MessagesHandler) createExecuteFunc(
	req *models.AnthropicMessageRequest,
	isStreaming bool,
) models.ExecutionFunc {
	return func(c *fiber.Ctx, provider models.Alternative, reqID string) error {
		// Get provider configuration from resolved config
		resolvedConfig, err := h.cfg.ResolveConfigFromAnthropicRequest(req)
		if err != nil {
			return fmt.Errorf("failed to resolve config: %w", err)
		}

		providers := resolvedConfig.GetProviders("messages")
		providerConfig, exists := providers[provider.Provider]
		if !exists {
			return fmt.Errorf("provider %s not configured", provider.Provider)
		}

		// Create a copy to avoid race conditions when mutating req.Model
		reqCopy := *req
		reqCopy.Model = anthropic.Model(provider.Model)

		// Call the messages service and handle retryable errors
		err = h.messagesSvc.HandleProviderRequest(c, &reqCopy, provider.Provider, providerConfig, isStreaming, reqID, h.responseSvc)
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

		return nil
	}
}
