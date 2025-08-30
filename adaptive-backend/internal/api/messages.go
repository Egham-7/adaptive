package api

import (
	"fmt"

	"adaptive-backend/internal/config"
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
		fallbackService: fallback.NewFallbackService(),
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

	// Determine which provider to use
	provider := "anthropic" // Default to anthropic for messages endpoint

	// If a model is specified and contains a provider prefix, extract it
	if req.Model != "" {
		modelStr := string(req.Model)
		fiberlog.Debugf("[%s] Processing model specification: %s", requestID, modelStr)
		if parsedProvider, model, err := utils.ParseProviderModelWithDefault(modelStr, "anthropic"); err == nil {
			provider = parsedProvider // Use parsed provider
			// Update the model in the request to remove provider prefix
			req.Model = anthropic.Model(model)
			fiberlog.Infof("[%s] Model parsed - provider: %s, model: %s", requestID, provider, model)
		} else {
			// If parsing fails, treat as validation error
			fiberlog.Warnf("[%s] Model parsing failed: %v", requestID, err)
			return h.responseSvc.HandleBadRequest(c, "invalid model specification: "+err.Error(), requestID)
		}
	}

	// If no explicit model is provided, use model router for selection
	if req.Model == "" {
		fiberlog.Debugf("[%s] No model specified, using model router for selection", requestID)
		// Extract prompt for routing
		prompt, err := utils.ExtractPromptFromAnthropicMessages(req.Messages)
		if err != nil {
			fiberlog.Warnf("[%s] Failed to extract prompt for routing: %v", requestID, err)
			return h.responseSvc.HandleBadRequest(c, "failed to extract prompt for routing: "+err.Error(), requestID)
		}
		fiberlog.Debugf("[%s] Extracted prompt for routing (length: %d chars)", requestID, len(prompt))

		// Use model router to select best model WITH CIRCUIT BREAKERS
		userID := "anonymous" // Could extract from auth context if available

		protocolResp, _, err := h.modelRouter.SelectProtocolWithCache(
			prompt, userID, requestID, &resolvedConfig.ModelRouter, h.circuitBreakers,
		)
		if err != nil {
			fiberlog.Errorf("[%s] Model router selection failed: %v", requestID, err)
			return h.responseSvc.HandleError(c, err, requestID)
		}

		// Use the selected provider and model
		if protocolResp.Standard != nil {
			provider = protocolResp.Standard.Provider
			req.Model = anthropic.Model(protocolResp.Standard.Model)
			fiberlog.Infof("[%s] Model router selected (standard) - provider: %s, model: %s", requestID, provider, protocolResp.Standard.Model)
		} else if protocolResp.Minion != nil {
			provider = protocolResp.Minion.Provider
			req.Model = anthropic.Model(protocolResp.Minion.Model)
			fiberlog.Infof("[%s] Model router selected (minion) - provider: %s, model: %s", requestID, provider, protocolResp.Minion.Model)
		} else {
			fiberlog.Errorf("[%s] Model router returned invalid response - no standard or minion protocol found", requestID)
			return h.responseSvc.HandleError(c, fmt.Errorf("invalid protocol response from model router"), requestID)
		}
	}

	// Get provider configuration from resolved config
	providers := resolvedConfig.GetProviders("messages")
	providerConfig, exists := providers[provider]
	if !exists {
		return h.responseSvc.HandleProviderNotConfigured(c, provider, requestID)
	}
	fiberlog.Debugf("[%s] Provider configuration found for: %s", requestID, provider)

	// Delegate provider handling to messages service
	return h.messagesSvc.HandleProviderRequest(c, req, provider, providerConfig, isStreaming, requestID, h.responseSvc)
}
