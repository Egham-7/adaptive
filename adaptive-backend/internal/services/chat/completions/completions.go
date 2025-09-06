package completions

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/fallback"
	"adaptive-backend/internal/services/format_adapter"
	"adaptive-backend/internal/services/stream_adapters"
	"adaptive-backend/internal/services/stream_readers/stream"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicOption "github.com/anthropics/anthropic-sdk-go/option"
	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go/v2"
	openaiOption "github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/shared"
)

// CompletionService handles completion requests with fallback logic.
type CompletionService struct {
	fallbackService *fallback.FallbackService
	responseService *ResponseService
}

// NewCompletionService creates a new completion service.
func NewCompletionService(cfg *config.Config, responseService *ResponseService) *CompletionService {
	if responseService == nil {
		panic("NewCompletionService: responseService cannot be nil")
	}
	if cfg == nil {
		panic("NewCompletionService: cfg cannot be nil")
	}

	return &CompletionService{
		fallbackService: fallback.NewFallbackService(cfg),
		responseService: responseService,
	}
}

// createClient creates an OpenAI client for the given provider using resolved config
func (cs *CompletionService) createClient(providerName string, resolvedConfig *config.Config, isStream bool) (*openai.Client, error) {
	if resolvedConfig == nil {
		return nil, models.NewInternalError("resolved config is nil", nil)
	}

	// Use resolved config directly - no more merging needed
	providerConfig, exists := resolvedConfig.GetProviderConfig(providerName, "chat_completions")
	if !exists {
		return nil, models.NewProviderError(providerName, "provider is not configured", nil)
	}

	return cs.buildClient(providerConfig, providerName, isStream)
}

func (cs *CompletionService) buildClient(providerConfig models.ProviderConfig, providerName string, isStream bool) (*openai.Client, error) {
	if providerName == "" {
		return nil, models.NewValidationError("provider name cannot be empty", nil)
	}
	if providerConfig.APIKey == "" {
		return nil, models.NewProviderError(providerName, "API key not configured", nil)
	}

	opts := []openaiOption.RequestOption{
		openaiOption.WithAPIKey(providerConfig.APIKey),
	}

	if providerConfig.BaseURL != "" {
		opts = append(opts, openaiOption.WithBaseURL(providerConfig.BaseURL))
	}

	if providerConfig.Headers != nil {
		for key, value := range providerConfig.Headers {
			opts = append(opts, openaiOption.WithHeader(key, value))
		}
	}

	// Only apply HTTP client timeout for non-streaming requests
	// Streaming requests need to stay open for SSE connections
	if providerConfig.TimeoutMs > 0 && !isStream {
		timeout := time.Duration(providerConfig.TimeoutMs) * time.Millisecond
		httpClient := &http.Client{Timeout: timeout}
		opts = append(opts, openaiOption.WithHTTPClient(httpClient))
	}

	client := openai.NewClient(opts...)
	return &client, nil
}

// createAnthropicClient creates an Anthropic client with the given provider configuration
func (cs *CompletionService) createAnthropicClient(providerConfig models.ProviderConfig) *anthropic.Client {
	clientOpts := []anthropicOption.RequestOption{
		anthropicOption.WithAPIKey(providerConfig.APIKey),
	}

	// Set custom base URL if provided
	if providerConfig.BaseURL != "" {
		clientOpts = append(clientOpts, anthropicOption.WithBaseURL(providerConfig.BaseURL))
	}

	client := anthropic.NewClient(clientOpts...)
	return &client
}

// HandleCompletion handles completion requests with fallback.
func (cs *CompletionService) HandleCompletion(
	c *fiber.Ctx,
	req *models.ChatCompletionRequest,
	resp *models.ModelSelectionResponse,
	requestID string,
	isStream bool,
	cacheSource string,
	resolvedConfig *config.Config,
) error {
	if c == nil || req == nil || resp == nil || requestID == "" {
		return models.NewValidationError("invalid input parameters", nil)
	}

	// For streaming requests, use fallback with early failure detection
	if isStream {
		fallbackConfig := cs.fallbackService.GetFallbackConfig(req.Fallback)

		// Create provider list with primary and alternatives
		providers := []models.Alternative{{
			Provider: resp.Provider,
			Model:    resp.Model,
		}}

		providers = append(providers, resp.Alternatives...)

		return cs.fallbackService.Execute(c, providers, fallbackConfig, cs.createExecuteFunc(req, isStream, cacheSource, resolvedConfig), requestID, isStream)
	}

	// For non-streaming requests, use fallback logic
	fallbackConfig := cs.fallbackService.GetFallbackConfig(req.Fallback)

	// Create provider list with primary and alternatives
	providers := []models.Alternative{{
		Provider: resp.Provider,
		Model:    resp.Model,
	}}

	providers = append(providers, resp.Alternatives...)

	return cs.fallbackService.Execute(c, providers, fallbackConfig, cs.createExecuteFunc(req, isStream, cacheSource, resolvedConfig), requestID, isStream)
}

// createExecuteFunc creates an execution function for the fallback service
func (cs *CompletionService) createExecuteFunc(
	req *models.ChatCompletionRequest,
	isStream bool,
	cacheSource string,
	resolvedConfig *config.Config,
) models.ExecutionFunc {
	return func(c *fiber.Ctx, provider models.Alternative, reqID string) error {
		client, err := cs.createClient(provider.Provider, resolvedConfig, isStream)
		if err != nil {
			return fmt.Errorf("client creation failed: %w", err)
		}

		// Create a copy to avoid race conditions when mutating req.Model
		reqCopy := *req
		reqCopy.Model = shared.ChatModel(provider.Model)
		err = cs.executeCompletion(c, client, provider.Provider, &reqCopy, reqID, isStream, cacheSource, resolvedConfig)
		// Check if the error is a retryable provider error that should trigger fallback
		if err != nil {
			if appErr, ok := err.(*models.AppError); ok && appErr.Type == models.ErrorTypeProvider && appErr.Retryable {
				// Return the provider error to trigger fallback
				return err
			}
			// For non-retryable errors, wrap them to prevent fallback
			return fmt.Errorf("non-retryable error: %w", err)
		}

		return nil
	}
}

// executeCompletion handles the actual completion execution
func (cs *CompletionService) executeCompletion(
	c *fiber.Ctx,
	client *openai.Client,
	providerName string,
	req *models.ChatCompletionRequest,
	requestID string,
	isStream bool,
	cacheSource string,
	resolvedConfig *config.Config,
) error {
	// Get provider configuration to check native format
	providerConfig, exists := resolvedConfig.GetProviderConfig(providerName, "chat_completions")
	if !exists {
		return fmt.Errorf("provider '%s' is not configured", providerName)
	}

	// Check provider's native format to determine if conversion is needed
	// If native_format is undefined or "openai", use OpenAI format directly
	if providerConfig.NativeFormat == "openai" || providerConfig.NativeFormat == "" {
		// Native OpenAI format - use directly
		fiberlog.Debugf("[%s] Provider %s uses native OpenAI format", requestID, providerName)
		return cs.executeOpenAICompletion(c, client, providerName, req, requestID, isStream, cacheSource, resolvedConfig)
	}

	// Provider uses different native format - handle conversion
	fiberlog.Debugf("[%s] Provider %s native format: %s, requires format conversion",
		requestID, providerName, providerConfig.NativeFormat)

	if providerConfig.NativeFormat == "anthropic" {
		return cs.executeAnthropicCompletion(c, providerName, req, requestID, isStream, cacheSource, providerConfig)
	}

	return fmt.Errorf("native format '%s' not yet supported for chat completions endpoint", providerConfig.NativeFormat)
}

// executeAnthropicCompletion handles providers with native Anthropic format
func (cs *CompletionService) executeAnthropicCompletion(
	c *fiber.Ctx,
	providerName string,
	req *models.ChatCompletionRequest,
	requestID string,
	isStream bool,
	cacheSource string,
	providerConfig models.ProviderConfig,
) error {
	// Step 1: Convert Adaptive → OpenAI → Anthropic for the request
	// First convert to OpenAI format
	openAIParams, err := format_adapter.AdaptiveToOpenAI.ConvertRequest(req)
	if err != nil {
		return fmt.Errorf("failed to convert request to OpenAI parameters: %w", err)
	}

	// Then convert OpenAI → Anthropic
	anthropicReq, err := format_adapter.OpenAIToAnthropic.ConvertRequest(openAIParams)
	if err != nil {
		return fmt.Errorf("failed to convert OpenAI request to Anthropic format: %w", err)
	}

	// Create Anthropic client directly
	anthropicClient := cs.createAnthropicClient(providerConfig)

	if isStream {
		fiberlog.Infof("[%s] streaming Anthropic response from %s (will convert to OpenAI format)", requestID, providerName)

		// Convert to Anthropic params for streaming
		params := anthropic.MessageNewParams{
			MaxTokens:     anthropicReq.MaxTokens,
			Messages:      anthropicReq.Messages,
			Model:         anthropicReq.Model,
			Temperature:   anthropicReq.Temperature,
			TopK:          anthropicReq.TopK,
			TopP:          anthropicReq.TopP,
			Metadata:      anthropicReq.Metadata,
			ServiceTier:   anthropicReq.ServiceTier,
			StopSequences: anthropicReq.StopSequences,
			System:        anthropicReq.System,
			Thinking:      anthropicReq.Thinking,
			ToolChoice:    anthropicReq.ToolChoice,
			Tools:         anthropicReq.Tools,
		}

		// Make streaming request to Anthropic
		anthropicStream := anthropicClient.Messages.NewStreaming(c.Context(), params)

		// Convert Anthropic stream to io.Reader using stream adapter
		streamReader := stream_adapters.NewAnthropicToOpenAIStreamAdapter(anthropicStream, providerName, requestID)

		// Handle stream using the stream handler
		return stream.HandleAnthropicStream(c, streamReader, requestID, providerName)
	}

	fiberlog.Infof("[%s] generating Anthropic completion from %s (will convert to OpenAI format)", requestID, providerName)

	// Convert to Anthropic params
	ctx := c.UserContext()
	if providerConfig.TimeoutMs > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(providerConfig.TimeoutMs)*time.Millisecond)
		defer cancel()
	}

	params := anthropic.MessageNewParams{
		MaxTokens:     anthropicReq.MaxTokens,
		Messages:      anthropicReq.Messages,
		Model:         anthropicReq.Model,
		Temperature:   anthropicReq.Temperature,
		TopK:          anthropicReq.TopK,
		TopP:          anthropicReq.TopP,
		Metadata:      anthropicReq.Metadata,
		ServiceTier:   anthropicReq.ServiceTier,
		StopSequences: anthropicReq.StopSequences,
		System:        anthropicReq.System,
		Thinking:      anthropicReq.Thinking,
		ToolChoice:    anthropicReq.ToolChoice,
		Tools:         anthropicReq.Tools,
	}

	// Make non-streaming request to Anthropic
	anthropicResp, err := anthropicClient.Messages.New(ctx, params)
	if err != nil {
		return models.NewProviderError(providerName, "Anthropic completion request failed", err)
	}

	// Convert Anthropic response back to OpenAI format
	openaiResp, err := format_adapter.AnthropicToOpenAI.ConvertResponse(anthropicResp, providerName)
	if err != nil {
		return fmt.Errorf("failed to convert Anthropic response to OpenAI format: %w", err)
	}

	// Convert OpenAI response to Adaptive format with cache source
	adaptiveResp, err := format_adapter.OpenAIToAdaptive.ConvertResponse(openaiResp, providerName, cacheSource)
	if err != nil {
		return fmt.Errorf("failed to convert response to adaptive format: %w", err)
	}

	// Return JSON response
	return c.JSON(adaptiveResp)
}

// executeOpenAICompletion handles providers with native OpenAI format
func (cs *CompletionService) executeOpenAICompletion(
	c *fiber.Ctx,
	client *openai.Client,
	providerName string,
	req *models.ChatCompletionRequest,
	requestID string,
	isStream bool,
	cacheSource string,
	resolvedConfig *config.Config,
) error {
	if isStream {
		fiberlog.Infof("[%s] streaming response from %s", requestID, providerName)

		// Convert request using format adapter
		openAIParams, err := format_adapter.AdaptiveToOpenAI.ConvertRequest(req)
		if err != nil {
			return fmt.Errorf("failed to convert request to OpenAI parameters: %w", err)
		}

		streamResp := client.Chat.Completions.NewStreaming(c.UserContext(), *openAIParams)
		return stream.HandleOpenAIStream(c, streamResp, requestID, providerName, cacheSource)
	}

	fiberlog.Infof("[%s] generating completion from %s", requestID, providerName)

	// For non-streaming requests, apply context timeout based on provider config
	ctx := c.UserContext()
	if !isStream {
		if providerConfig, exists := resolvedConfig.GetProviderConfig(providerName, "chat_completions"); exists && providerConfig.TimeoutMs > 0 {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, time.Duration(providerConfig.TimeoutMs)*time.Millisecond)
			defer cancel()
		}
	}

	// Convert request using format adapter
	openAIParams, err := format_adapter.AdaptiveToOpenAI.ConvertRequest(req)
	if err != nil {
		return fmt.Errorf("failed to convert request to OpenAI parameters: %w", err)
	}

	resp, err := client.Chat.Completions.New(ctx, *openAIParams)
	if err != nil {
		return models.NewProviderError(providerName, "completion request failed", err)
	}

	// Convert response using format adapter with cache source
	adaptiveResp, err := format_adapter.OpenAIToAdaptive.ConvertResponse(resp, providerName, cacheSource)
	if err != nil {
		return fmt.Errorf("failed to convert response to adaptive format: %w", err)
	}

	return c.JSON(adaptiveResp)
}

// HandleModel handles completion requests with the selected model
func (cs *CompletionService) HandleModel(
	c *fiber.Ctx,
	req *models.ChatCompletionRequest,
	resp *models.ModelSelectionResponse,
	requestID string,
	isStream bool,
	cacheSource string,
	resolvedConfig *config.Config,
) error {
	if isStream {
		cs.responseService.SetStreamHeaders(c)
	}

	if err := cs.HandleCompletion(c, req, resp, requestID, isStream, cacheSource, resolvedConfig); err != nil {
		return cs.responseService.HandleError(c, fiber.StatusInternalServerError, err.Error(), requestID)
	}

	// Store successful response in semantic cache
	cs.responseService.StoreSuccessfulSemanticCache(c.UserContext(), req, resp, requestID)
	return nil
}
