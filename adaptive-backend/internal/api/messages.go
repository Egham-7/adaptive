package api

import (
	"fmt"

	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/anthropic/messages"
	"adaptive-backend/internal/services/format_adapter"
	"adaptive-backend/internal/services/model_router"
	"adaptive-backend/internal/services/stream_adapters"
	"adaptive-backend/internal/services/stream_readers/stream"
	"adaptive-backend/internal/utils"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
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

	// Check provider's native format to determine if conversion is needed
	// If native_format is undefined, assume it's anthropic
	if providerConfig.NativeFormat == "anthropic" || providerConfig.NativeFormat == "" || provider == "anthropic" {
		// Native Anthropic format - use directly
		fiberlog.Debugf("[%s] Provider %s uses native Anthropic format", requestID, provider)
		client := h.messagesSvc.CreateClient(providerConfig)

		fiberlog.Infof("[%s] Processing native Anthropic request - model: %s, streaming: %t",
			requestID, req.Model, isStreaming)

		if isStreaming {
			return h.handleStreamingRequest(c, client, req, requestID)
		}
		return h.handleNonStreamingRequest(c, client, req, requestID)
	}

	// Provider uses different native format (likely OpenAI) - convert via format adapters
	fiberlog.Debugf("[%s] Provider %s native format: %s, converting to Anthropic format",
		requestID, provider, providerConfig.NativeFormat)
	return h.handleNonAnthropicProvider(c, req, provider, providerConfig, isStreaming, requestID)
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

// handleNonAnthropicProvider handles providers that use different native formats (e.g., OpenAI)
// by converting the Anthropic Messages request to the provider's format and back
func (h *MessagesHandler) handleNonAnthropicProvider(
	c *fiber.Ctx,
	req *models.AnthropicMessageRequest,
	provider string,
	providerConfig models.ProviderConfig,
	isStreaming bool,
	requestID string,
) error {
	// Convert Anthropic Messages request to OpenAI Chat Completions format
	openaiReq, err := format_adapter.AnthropicToOpenAI.ConvertRequest(req)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to convert Anthropic request to OpenAI format: %v", requestID, err)
		return h.responseSvc.HandleError(c, fmt.Errorf("failed to convert request format: %w", err), requestID)
	}

	// Create OpenAI client for the provider
	client, err := h.createOpenAIClient(provider, providerConfig, isStreaming)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to create OpenAI client for provider %s: %v", requestID, provider, err)
		return h.responseSvc.HandleError(c, fmt.Errorf("failed to create client: %w", err), requestID)
	}

	fiberlog.Infof("[%s] Processing converted request - provider: %s, model: %s, streaming: %t",
		requestID, provider, openaiReq.Model, isStreaming)

	if isStreaming {
		return h.handleConvertedStreamingRequest(c, client, openaiReq, provider, requestID)
	}

	return h.handleConvertedNonStreamingRequest(c, client, openaiReq, provider, requestID)
}

// createOpenAIClient creates an OpenAI client for non-Anthropic providers
func (h *MessagesHandler) createOpenAIClient(provider string, providerConfig models.ProviderConfig, isStream bool) (*openai.Client, error) {
	if providerConfig.APIKey == "" {
		return nil, fmt.Errorf("API key not configured for provider '%s'", provider)
	}

	opts := []option.RequestOption{
		option.WithAPIKey(providerConfig.APIKey),
	}

	if providerConfig.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(providerConfig.BaseURL))
	}

	if providerConfig.Headers != nil {
		for key, value := range providerConfig.Headers {
			opts = append(opts, option.WithHeader(key, value))
		}
	}

	client := openai.NewClient(opts...)
	return &client, nil
}

// handleConvertedNonStreamingRequest handles non-streaming requests for converted providers
func (h *MessagesHandler) handleConvertedNonStreamingRequest(
	c *fiber.Ctx,
	client *openai.Client,
	openaiReq *models.ChatCompletionRequest,
	provider string,
	requestID string,
) error {
	fiberlog.Debugf("[%s] Sending non-streaming request to %s", requestID, provider)

	// Convert to OpenAI parameters
	openAIParams, err := format_adapter.AdaptiveToOpenAI.ConvertRequest(openaiReq)
	if err != nil {
		return fmt.Errorf("failed to convert request to OpenAI parameters: %w", err)
	}

	// Make the OpenAI API call
	resp, err := client.Chat.Completions.New(c.Context(), *openAIParams)
	if err != nil {
		fiberlog.Errorf("[%s] Non-streaming request to %s failed: %v", requestID, provider, err)
		return h.responseSvc.HandleError(c, fmt.Errorf("%s API error: %w", provider, err), requestID)
	}

	// Convert OpenAI response back to Anthropic format
	anthropicResp, err := format_adapter.OpenAIToAnthropic.ConvertResponse(resp, provider)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to convert %s response to Anthropic format: %v", requestID, provider, err)
		return h.responseSvc.HandleError(c, fmt.Errorf("failed to convert response: %w", err), requestID)
	}

	fiberlog.Infof("[%s] Non-streaming request to %s completed successfully", requestID, provider)
	return c.JSON(anthropicResp)
}

// handleConvertedStreamingRequest handles streaming requests for converted providers
func (h *MessagesHandler) handleConvertedStreamingRequest(
	c *fiber.Ctx,
	client *openai.Client,
	openaiReq *models.ChatCompletionRequest,
	provider string,
	requestID string,
) error {
	fiberlog.Debugf("[%s] Starting streaming request to %s", requestID, provider)

	// Convert to OpenAI parameters with streaming enabled
	openAIParams, err := format_adapter.AdaptiveToOpenAI.ConvertRequest(openaiReq)
	if err != nil {
		return fmt.Errorf("failed to convert request to OpenAI parameters: %w", err)
	}

	// Create streaming request
	streamResp := client.Chat.Completions.NewStreaming(c.Context(), *openAIParams)

	// Create a converted stream reader that converts OpenAI SSE to Anthropic SSE format
	convertedAdapter := stream_adapters.NewOpenAIToAnthropicStreamAdapter(streamResp, provider, requestID)

	// Use the existing Anthropic stream handler which will use the AnthropicSSEReader
	return stream.HandleAnthropicStream(c, convertedAdapter, requestID, provider)
}
