package api

import (
	"fmt"

	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/anthropic/messages"
	"adaptive-backend/internal/services/model_router"
	"adaptive-backend/internal/utils"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// MessagesHandler handles Anthropic Messages API requests using dedicated Anthropic services
type MessagesHandler struct {
	cfg         *config.Config
	requestSvc  *messages.RequestService
	messagesSvc *messages.MessagesService
	responseSvc *messages.ResponseService
	modelRouter *model_router.ModelRouter
}

// NewMessagesHandler creates a new MessagesHandler with Anthropic-specific services
func NewMessagesHandler(
	cfg *config.Config,
	modelRouter *model_router.ModelRouter,
) *MessagesHandler {
	return &MessagesHandler{
		cfg:         cfg,
		requestSvc:  messages.NewRequestService(),
		messagesSvc: messages.NewMessagesService(),
		responseSvc: messages.NewResponseService(),
		modelRouter: modelRouter,
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

		// Use model router to select best model
		userID := "anonymous" // Could extract from auth context if available

		protocolResp, _, err := h.modelRouter.SelectProtocolWithCache(
			prompt, userID, requestID, &resolvedConfig.ModelRouter, nil,
		)
		if err != nil {
			fiberlog.Errorf("[%s] Model router selection failed: %v", requestID, err)
			return h.responseSvc.HandleError(c, err, requestID)
		}

		// Use the selected provider and model
		if protocolResp.Standard != nil {
			provider = protocolResp.Standard.Provider
			req.Model = anthropic.Model(protocolResp.Standard.Model)
			fiberlog.Infof("[%s] Model router selected - provider: %s, model: %s", requestID, provider, protocolResp.Standard.Model)
		}
	}

	// Get provider configuration from resolved config
	providers := resolvedConfig.GetProviders("messages")
	providerConfig, exists := providers[provider]
	if !exists {
		fiberlog.Warnf("[%s] Provider '%s' not configured for messages endpoint", requestID, provider)
		return h.responseSvc.HandleProviderNotConfigured(c, provider, requestID)
	}
	fiberlog.Debugf("[%s] Provider configuration found for: %s", requestID, provider)

	// For now, only support native Anthropic provider
	if provider != "anthropic" {
		fiberlog.Warnf("[%s] Unsupported provider requested: %s", requestID, provider)
		return h.responseSvc.HandleBadRequest(c,
			"Only 'anthropic' provider is currently supported for messages endpoint", requestID)
	}

	// Create Anthropic client
	fiberlog.Debugf("[%s] Creating Anthropic client", requestID)
	client := h.messagesSvc.CreateClient(providerConfig)

	fiberlog.Infof("[%s] Processing request - provider: %s, model: %s, streaming: %t",
		requestID, provider, req.Model, isStreaming)

	if isStreaming {
		return h.handleStreamingRequest(c, client, req, requestID)
	}

	return h.handleNonStreamingRequest(c, client, req, requestID)
}

// handleNonStreamingRequest processes non-streaming requests
func (h *MessagesHandler) handleNonStreamingRequest(
	c *fiber.Ctx,
	client *anthropic.Client,
	req *models.AnthropicMessageRequest,
	requestID string,
) error {
	fiberlog.Debugf("[%s] Sending non-streaming message to Anthropic", requestID)
	message, err := h.messagesSvc.SendMessage(c.Context(), client, req, requestID)
	if err != nil {
		fiberlog.Errorf("[%s] Non-streaming message failed: %v", requestID, err)
		return h.responseSvc.HandleError(c, err, requestID)
	}

	fiberlog.Infof("[%s] Non-streaming message completed successfully", requestID)
	return h.responseSvc.HandleNonStreamingResponse(c, message, requestID)
}

// handleStreamingRequest processes streaming requests
func (h *MessagesHandler) handleStreamingRequest(
	c *fiber.Ctx,
	client *anthropic.Client,
	req *models.AnthropicMessageRequest,
	requestID string,
) error {
	fiberlog.Debugf("[%s] Starting streaming message to Anthropic", requestID)
	stream, err := h.messagesSvc.SendStreamingMessage(c.Context(), client, req, requestID)
	if err != nil {
		fiberlog.Errorf("[%s] Streaming message initiation failed: %v", requestID, err)
		return h.responseSvc.HandleError(c, err, requestID)
	}

	fiberlog.Infof("[%s] Streaming message initiated successfully", requestID)
	return h.responseSvc.HandleStreamingResponse(c, stream, requestID)
}
