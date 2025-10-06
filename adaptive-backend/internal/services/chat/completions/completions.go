package completions

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/fallback"
	"adaptive-backend/internal/services/format_adapter"
	"adaptive-backend/internal/services/stream/handlers"
	"adaptive-backend/internal/utils/clientcache"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go/v2"
	openaiOption "github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/shared"
)

const (
	// Service type for chat completions configuration
	serviceTypeChatCompletions = "chat_completions"
)

// CompletionService handles completion requests with fallback logic.
type CompletionService struct {
	fallbackService *fallback.FallbackService
	responseService *ResponseService
	clientCache     *clientcache.Cache[*openai.Client]
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker
}

// NewCompletionService creates a new completion service.
func NewCompletionService(cfg *config.Config, responseService *ResponseService, circuitBreakers map[string]*circuitbreaker.CircuitBreaker) *CompletionService {
	if responseService == nil {
		panic("NewCompletionService: responseService cannot be nil")
	}
	if cfg == nil {
		panic("NewCompletionService: cfg cannot be nil")
	}

	return &CompletionService{
		fallbackService: fallback.NewFallbackService(cfg),
		responseService: responseService,
		clientCache:     clientcache.NewCache[*openai.Client](),
		circuitBreakers: circuitBreakers,
	}
}

// generateConfigHash creates a hash of the provider config to detect changes
func (cs *CompletionService) generateConfigHash(providerConfig models.ProviderConfig, isStream bool) (string, error) {
	// Create a simplified config struct for hashing (excluding sensitive data from logs)
	type configForHash struct {
		BaseURL    string
		TimeoutMs  int
		Headers    map[string]string
		IsStream   bool
		APIKeyHash string // Hash of API key instead of raw key
	}

	// Hash the API key separately
	apiKeyHash := sha256.Sum256([]byte(providerConfig.APIKey))

	hashConfig := configForHash{
		BaseURL:    providerConfig.BaseURL,
		TimeoutMs:  providerConfig.TimeoutMs,
		Headers:    providerConfig.Headers,
		IsStream:   isStream,
		APIKeyHash: fmt.Sprintf("%x", apiKeyHash[:8]), // Use first 8 bytes of hash
	}

	// Marshal to JSON for consistent hashing
	configJSON, err := json.Marshal(hashConfig)
	if err != nil {
		return "", err
	}

	// Generate SHA256 hash
	hash := sha256.Sum256(configJSON)
	return fmt.Sprintf("%x", hash[:16]), nil // Use first 16 bytes for cache key
}

// createClient creates or retrieves a cached OpenAI client for the given provider
func (cs *CompletionService) createClient(providerName string, resolvedConfig *config.Config, isStream bool) (*openai.Client, error) {
	if resolvedConfig == nil {
		return nil, models.NewInternalError("resolved config is nil", nil)
	}

	// Use resolved config directly - no more merging needed
	providerConfig, exists := resolvedConfig.GetProviderConfig(providerName, serviceTypeChatCompletions)
	if !exists {
		return nil, models.NewProviderError(providerName, "provider is not configured", nil)
	}

	// Generate cache key based on provider config hash
	configHash, err := cs.generateConfigHash(providerConfig, isStream)
	if err != nil {
		fiberlog.Warnf("Failed to generate config hash for %s: %v, creating new client without caching", providerName, err)
		return cs.buildClient(providerConfig, providerName, isStream)
	}

	cacheKey := fmt.Sprintf("%s:%s", providerName, configHash)

	// Use type-safe cache with singleflight to prevent duplicate client creation
	client, err := cs.clientCache.GetOrCreate(cacheKey, func() (*openai.Client, error) {
		fiberlog.Debugf("Creating new OpenAI client for %s (config hash: %s)", providerName, configHash[:8])
		return cs.buildClient(providerConfig, providerName, isStream)
	})
	if err != nil {
		return nil, err
	}

	fiberlog.Debugf("Using OpenAI client for %s (config hash: %s)", providerName, configHash[:8])
	return client, nil
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

// HandleCompletion handles completion requests with fallback for OpenAI-compatible providers.
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

	executeFunc := cs.createExecuteFunc(req, isStream, cacheSource, resolvedConfig)
	primary := models.Alternative{
		Provider: resp.Provider,
		Model:    resp.Model,
	}

	// Try primary provider first
	fiberlog.Infof("[%s] Trying primary provider: %s/%s", requestID, resp.Provider, resp.Model)
	err := executeFunc(c, primary, requestID)

	if err == nil {
		// Primary succeeded
		fiberlog.Infof("[%s] ✅ Primary provider succeeded: %s/%s", requestID, resp.Provider, resp.Model)
		return nil
	}

	// Primary failed - check if we have alternatives
	if len(resp.Alternatives) == 0 {
		fiberlog.Errorf("[%s] ❌ Primary provider failed and no alternatives available: %v", requestID, err)
		return err
	}

	// Use fallback service with alternatives only
	fiberlog.Warnf("[%s] ⚠️  Primary provider failed: %v", requestID, err)
	fiberlog.Infof("[%s] Using fallback with %d alternatives", requestID, len(resp.Alternatives))

	fallbackConfig := cs.fallbackService.GetFallbackConfig(req.Fallback)
	return cs.fallbackService.Execute(c, resp.Alternatives, fallbackConfig, executeFunc, requestID, isStream)
}

// createExecuteFunc creates an execution function for the fallback service
func (cs *CompletionService) createExecuteFunc(
	req *models.ChatCompletionRequest,
	isStream bool,
	cacheSource string,
	resolvedConfig *config.Config,
) models.ExecutionFunc {
	return func(c *fiber.Ctx, provider models.Alternative, reqID string) error {
		// Check circuit breaker before attempting execution
		if cb := cs.circuitBreakers[provider.Provider]; cb != nil {
			if !cb.CanExecute() {
				fiberlog.Warnf("[%s] Circuit breaker is OPEN for provider %s, skipping", reqID, provider.Provider)
				return models.NewCircuitBreakerError(provider.Provider)
			}
			fiberlog.Debugf("[%s] Circuit breaker check passed for provider %s", reqID, provider.Provider)
		}

		client, err := cs.createClient(provider.Provider, resolvedConfig, isStream)
		if err != nil {
			return fmt.Errorf("client creation failed for provider %s: %w", provider.Provider, err)
		}

		// Create a copy to avoid race conditions when mutating req.Model
		reqCopy := *req
		reqCopy.Model = shared.ChatModel(provider.Model)

		err = cs.executeOpenAICompletion(c, client, provider.Provider, &reqCopy, reqID, isStream, cacheSource, resolvedConfig)
		if err != nil {
			// Check if the error is a retryable provider error that should trigger fallback
			if appErr, ok := err.(*models.AppError); ok && appErr.Type == models.ErrorTypeProvider && appErr.Retryable {
				return err // Return as-is to trigger fallback
			}
			// For non-retryable errors, wrap them to prevent fallback
			return fmt.Errorf("non-retryable error from provider %s: %w", provider.Provider, err)
		}

		return nil
	}
}

// executeOpenAICompletion handles providers with OpenAI-compatible format
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
	// Convert request using format adapter
	openAIParams, err := format_adapter.AdaptiveToOpenAI.ConvertRequest(req)
	if err != nil {
		// Record failure in circuit breaker
		if cb := cs.circuitBreakers[providerName]; cb != nil {
			cb.RecordFailure()
		}
		return fmt.Errorf("failed to convert request to OpenAI parameters: %w", err)
	}

	if isStream {
		return cs.handleStreamingCompletion(c, client, providerName, openAIParams, requestID, cacheSource)
	}

	return cs.handleNonStreamingCompletion(c, client, providerName, openAIParams, requestID, cacheSource, resolvedConfig)
}

// handleStreamingCompletion handles streaming completions
func (cs *CompletionService) handleStreamingCompletion(
	c *fiber.Ctx,
	client *openai.Client,
	providerName string,
	openAIParams *openai.ChatCompletionNewParams,
	requestID string,
	cacheSource string,
) error {
	fiberlog.Infof("[%s] streaming response from %s", requestID, providerName)

	// Use context.Background() for streaming - c.UserContext() gets canceled too early
	// The stream handler will monitor fasthttpCtx for actual client disconnects
	streamResp := client.Chat.Completions.NewStreaming(context.Background(), *openAIParams)
	err := handlers.HandleOpenAI(c, streamResp, requestID, providerName, cacheSource)
	if err != nil {
		// Record failure in circuit breaker
		if cb := cs.circuitBreakers[providerName]; cb != nil {
			cb.RecordFailure()
		}
		return err
	}

	// Record success in circuit breaker
	if cb := cs.circuitBreakers[providerName]; cb != nil {
		cb.RecordSuccess()
	}

	return nil
}

// handleNonStreamingCompletion handles non-streaming completions
func (cs *CompletionService) handleNonStreamingCompletion(
	c *fiber.Ctx,
	client *openai.Client,
	providerName string,
	openAIParams *openai.ChatCompletionNewParams,
	requestID string,
	cacheSource string,
	resolvedConfig *config.Config,
) error {
	fiberlog.Infof("[%s] generating completion from %s", requestID, providerName)

	// Apply context timeout based on provider config
	ctx := c.UserContext()
	if providerConfig, exists := resolvedConfig.GetProviderConfig(providerName, serviceTypeChatCompletions); exists && providerConfig.TimeoutMs > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(providerConfig.TimeoutMs)*time.Millisecond)
		defer cancel()
	}

	resp, err := client.Chat.Completions.New(ctx, *openAIParams)
	if err != nil {
		// Record failure in circuit breaker
		if cb := cs.circuitBreakers[providerName]; cb != nil {
			cb.RecordFailure()
		}
		return models.NewProviderError(providerName, "completion request failed", err)
	}

	// Convert response using format adapter with cache source
	adaptiveResp, err := format_adapter.OpenAIToAdaptive.ConvertResponse(resp, providerName, cacheSource)
	if err != nil {
		// Record failure in circuit breaker
		if cb := cs.circuitBreakers[providerName]; cb != nil {
			cb.RecordFailure()
		}
		return fmt.Errorf("failed to convert response to adaptive format: %w", err)
	}

	// Record success in circuit breaker
	if cb := cs.circuitBreakers[providerName]; cb != nil {
		cb.RecordSuccess()
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
