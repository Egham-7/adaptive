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
	"adaptive-backend/internal/services/stream_readers/stream"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

// CompletionService handles completion requests with fallback logic.
type CompletionService struct {
	cfg             *config.Config
	fallbackService *fallback.FallbackService
	responseService *ResponseService
}

// NewCompletionService creates a new completion service.
func NewCompletionService(responseService *ResponseService) *CompletionService {

	if responseService == nil {
		panic("NewCompletionService: responseService cannot be nil")
	}

	return &CompletionService{
		fallbackService: fallback.NewFallbackService(),
		responseService: responseService,
	}
}

// createClient creates an OpenAI client for the given provider using resolved config
func (cs *CompletionService) createClient(providerName string, resolvedConfig *config.Config, isStream bool) (*openai.Client, error) {
	// Use resolved config directly - no more merging needed
	providerConfig, exists := resolvedConfig.GetProviderConfig(providerName, "chat_completions")
	if !exists {
		return nil, fmt.Errorf("provider '%s' is not configured", providerName)
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

	// Only apply HTTP client timeout for non-streaming requests
	// Streaming requests need to stay open for SSE connections
	if providerConfig.TimeoutMs > 0 && !isStream {
		timeout := time.Duration(providerConfig.TimeoutMs) * time.Millisecond
		httpClient := &http.Client{Timeout: timeout}
		opts = append(opts, option.WithHTTPClient(httpClient))
	}

	client := openai.NewClient(opts...)
	return &client, nil
}

// HandleStandardCompletion handles standard protocol completions with fallback.
func (cs *CompletionService) HandleStandardCompletion(
	c *fiber.Ctx,
	req *models.ChatCompletionRequest,
	standardInfo *models.StandardLLMInfo,
	requestID string,
	isStream bool,
	cacheSource string,
	resolvedConfig *config.Config,
) error {
	if c == nil || req == nil || standardInfo == nil || requestID == "" {
		return models.NewValidationError("invalid input parameters", nil)
	}

	// For streaming requests, do not attempt fallback as it can corrupt SSE streams
	if isStream {
		client, err := cs.createClient(standardInfo.Provider, resolvedConfig, isStream)
		if err != nil {
			return fmt.Errorf("client creation failed for streaming request: %w", err)
		}

		req.Model = shared.ChatModel(standardInfo.Model)
		return cs.executeCompletion(c, client, standardInfo.Provider, req, requestID, isStream, cacheSource, resolvedConfig)
	}

	// For non-streaming requests, use fallback logic
	fallbackConfig := cs.fallbackService.GetFallbackConfig(req.Fallback)

	// Create provider list with primary and alternatives
	providers := []models.Alternative{{
		Provider: standardInfo.Provider,
		Model:    standardInfo.Model,
	}}

	providers = append(providers, standardInfo.Alternatives...)

	return cs.fallbackService.Execute(c, providers, fallbackConfig, cs.createExecuteFunc(req, isStream, cacheSource, resolvedConfig), requestID, "standard", isStream)
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
		return cs.executeCompletion(c, client, provider.Provider, &reqCopy, reqID, isStream, cacheSource, resolvedConfig)
	}
}

// HandleMinionCompletion handles minion protocol completions with fallback.
func (cs *CompletionService) HandleMinionCompletion(
	c *fiber.Ctx,
	req *models.ChatCompletionRequest,
	minionInfo *models.MinionInfo,
	requestID string,
	isStream bool,
	cacheSource string,
	resolvedConfig *config.Config,
) error {
	if c == nil || req == nil || minionInfo == nil || requestID == "" {
		return models.NewValidationError("invalid input parameters", nil)
	}

	// For streaming requests, do not attempt fallback as it can corrupt SSE streams
	if isStream {
		client, err := cs.createClient(minionInfo.Provider, resolvedConfig, isStream)
		if err != nil {
			return fmt.Errorf("client creation failed for streaming request: %w", err)
		}

		req.Model = shared.ChatModel(minionInfo.Model)
		return cs.executeCompletion(c, client, minionInfo.Provider, req, requestID, isStream, cacheSource, resolvedConfig)
	}

	// For non-streaming requests, use fallback logic
	fallbackConfig := cs.fallbackService.GetFallbackConfig(req.Fallback)

	// Create provider list with primary and alternatives
	providers := []models.Alternative{{
		Provider: minionInfo.Provider,
		Model:    minionInfo.Model,
	}}

	providers = append(providers, minionInfo.Alternatives...)

	return cs.fallbackService.Execute(c, providers, fallbackConfig, cs.createExecuteFunc(req, isStream, cacheSource, resolvedConfig), requestID, "minion", isStream)
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
	if isStream {
		fiberlog.Infof("[%s] streaming response from %s", requestID, providerName)

		// Convert request using format adapter
		openAIParams, err := format_adapter.AdaptiveToOpenAI.ConvertRequest(req)
		if err != nil {
			return fmt.Errorf("failed to convert request to OpenAI parameters: %w", err)
		}

		streamResp := client.Chat.Completions.NewStreaming(c.Context(), *openAIParams)
		return stream.HandleOpenAIStream(c, streamResp, requestID, providerName, cacheSource)
	}

	fiberlog.Infof("[%s] generating completion from %s", requestID, providerName)

	// For non-streaming requests, apply context timeout based on provider config
	ctx := context.Background()
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

	// Convert response using format adapter
	adaptiveResp, err := format_adapter.OpenAIToAdaptive.ConvertResponse(resp, providerName)
	if err != nil {
		return fmt.Errorf("failed to convert response to adaptive format: %w", err)
	}
	models.SetCacheTier(&adaptiveResp.Usage, cacheSource)

	return c.JSON(adaptiveResp)
}

// HandleProtocol routes to the correct completion flow based on protocol
func (cs *CompletionService) HandleProtocol(
	c *fiber.Ctx,
	protocol models.ProtocolType,
	req *models.ChatCompletionRequest,
	resp *models.ProtocolResponse,
	requestID string,
	isStream bool,
	cacheSource string,
	resolvedConfig *config.Config,
) error {
	if isStream {
		cs.responseService.SetStreamHeaders(c)
	}

	const (
		errUnknownProtocol = "unknown protocol"
	)

	switch protocol {
	case models.ProtocolStandardLLM:
		if err := cs.HandleStandardCompletion(c, req, resp.Standard, requestID, isStream, cacheSource, resolvedConfig); err != nil {
			return cs.responseService.HandleError(c, fiber.StatusInternalServerError, err.Error(), requestID)
		}
		// Store successful response in semantic cache
		cs.responseService.StoreSuccessfulSemanticCache(req, resp, requestID)
		return nil

	case models.ProtocolMinion:
		if err := cs.HandleMinionCompletion(c, req, resp.Minion, requestID, isStream, cacheSource, resolvedConfig); err != nil {
			return cs.responseService.HandleError(c, fiber.StatusInternalServerError, err.Error(), requestID)
		}
		// Store successful response in semantic cache
		cs.responseService.StoreSuccessfulSemanticCache(req, resp, requestID)
		return nil

	default:
		return cs.responseService.HandleError(c, fiber.StatusInternalServerError,
			errUnknownProtocol+" "+string(protocol), requestID)
	}
}
