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
	fiberlog.Infof("[%s] starting anthropic messages request", requestID)

	// Parse Anthropic Messages request
	req, err := h.requestSvc.ParseRequest(c)
	if err != nil {
		return h.responseSvc.HandleBadRequest(c, err.Error(), requestID)
	}

	// Resolve config by merging YAML config with request overrides (single source of truth)
	resolvedConfig, err := h.cfg.ResolveConfigFromAnthropicRequest(req)
	if err != nil {
		return h.responseSvc.HandleError(c, fmt.Errorf("failed to resolve config: %w", err), requestID)
	}

	// Check if streaming is requested
	isStreaming := h.requestSvc.IsStreamingRequest(c)

	// Determine which provider to use
	provider := "anthropic" // Default to anthropic for messages endpoint

	// If a model is specified and contains a provider prefix, extract it
	if req.Model != "" {
		modelStr := string(req.Model)
		if parsedProvider, model, err := utils.ParseProviderModelWithDefault(modelStr, "anthropic"); err == nil {
			provider = parsedProvider // Use parsed provider
			// Update the model in the request to remove provider prefix
			req.Model = anthropic.Model(model)
		} else {
			// If parsing fails, treat as validation error
			return h.responseSvc.HandleBadRequest(c, "invalid model specification: "+err.Error(), requestID)
		}
	}

	// If no explicit model is provided, use model router for selection
	if req.Model == "" {
		// Extract prompt for routing
		prompt, err := utils.ExtractPromptFromAnthropicMessages(req.Messages)
		if err != nil {
			return h.responseSvc.HandleBadRequest(c, "failed to extract prompt for routing: "+err.Error(), requestID)
		}

		// Use model router to select best model
		userID := "anonymous" // Could extract from auth context if available

		protocolResp, _, err := h.modelRouter.SelectProtocolWithCache(
			prompt, userID, requestID, &resolvedConfig.ModelRouter, nil,
		)
		if err != nil {
			return h.responseSvc.HandleError(c, err, requestID)
		}

		// Use the selected provider and model
		if protocolResp.Standard != nil {
			provider = protocolResp.Standard.Provider
			req.Model = anthropic.Model(protocolResp.Standard.Model)
		}
	}

	// Get provider configuration from resolved config
	providers := resolvedConfig.GetProviders("messages")
	providerConfig, exists := providers[provider]
	if !exists {
		return h.responseSvc.HandleProviderNotConfigured(c, provider, requestID)
	}

	// For now, only support native Anthropic provider
	if provider != "anthropic" {
		return h.responseSvc.HandleBadRequest(c,
			"Only 'anthropic' provider is currently supported for messages endpoint", requestID)
	}

	// Create Anthropic client
	client := h.messagesSvc.CreateClient(providerConfig)

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
	message, err := h.messagesSvc.SendMessage(c.Context(), client, req, requestID)
	if err != nil {
		return h.responseSvc.HandleError(c, err, requestID)
	}

	return h.responseSvc.HandleNonStreamingResponse(c, message, requestID)
}

// handleStreamingRequest processes streaming requests
func (h *MessagesHandler) handleStreamingRequest(
	c *fiber.Ctx,
	client *anthropic.Client,
	req *models.AnthropicMessageRequest,
	requestID string,
) error {
	stream, err := h.messagesSvc.SendStreamingMessage(c.Context(), client, req, requestID)
	if err != nil {
		return h.responseSvc.HandleError(c, err, requestID)
	}

	return h.responseSvc.HandleStreamingResponse(c, stream, requestID)
}
