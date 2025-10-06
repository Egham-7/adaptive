package messages

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"time"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/utils/clientcache"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// MessagesService handles Anthropic Messages API calls using the Anthropic SDK
type MessagesService struct {
	clientCache *clientcache.Cache[*anthropic.Client]
}

// NewMessagesService creates a new MessagesService
func NewMessagesService() *MessagesService {
	return &MessagesService{
		clientCache: clientcache.NewCache[*anthropic.Client](),
	}
}

// generateConfigHash creates a hash of the provider config to detect changes
func (ms *MessagesService) generateConfigHash(providerConfig models.ProviderConfig) (string, error) {
	type configForHash struct {
		BaseURL    string
		Headers    map[string]string
		APIKeyHash string
	}

	apiKeyHash := sha256.Sum256([]byte(providerConfig.APIKey))
	hashConfig := configForHash{
		BaseURL:    providerConfig.BaseURL,
		Headers:    providerConfig.Headers,
		APIKeyHash: fmt.Sprintf("%x", apiKeyHash[:8]),
	}

	configJSON, err := json.Marshal(hashConfig)
	if err != nil {
		return "", err
	}

	hash := sha256.Sum256(configJSON)
	return fmt.Sprintf("%x", hash[:16]), nil
}

// CreateClient creates or retrieves a cached Anthropic client
func (ms *MessagesService) CreateClient(providerConfig models.ProviderConfig) *anthropic.Client {
	// Generate cache key based on provider config hash
	configHash, err := ms.generateConfigHash(providerConfig)
	if err != nil {
		fiberlog.Warnf("Failed to generate config hash: %v, creating new client without caching", err)
		return ms.buildClient(providerConfig)
	}

	// Use type-safe cache with singleflight to prevent duplicate client creation
	client, err := ms.clientCache.GetOrCreate(configHash, func() (*anthropic.Client, error) {
		fiberlog.Debugf("Creating new Anthropic client (config hash: %s)", configHash[:8])
		return ms.buildClient(providerConfig), nil
	})
	if err != nil {
		// Should never happen since buildClient doesn't return error, but handle gracefully
		fiberlog.Warnf("Unexpected error from cache: %v, creating new client", err)
		return ms.buildClient(providerConfig)
	}

	fiberlog.Debugf("Using Anthropic client (config hash: %s)", configHash[:8])
	return client
}

// buildClient creates a new Anthropic client with the given configuration
func (ms *MessagesService) buildClient(providerConfig models.ProviderConfig) *anthropic.Client {
	clientOpts := []option.RequestOption{
		option.WithAPIKey(providerConfig.APIKey),
	}

	// Set custom base URL if provided
	if providerConfig.BaseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(providerConfig.BaseURL))
	}

	// Add custom headers if provided
	if providerConfig.Headers != nil {
		for key, value := range providerConfig.Headers {
			clientOpts = append(clientOpts, option.WithHeader(key, value))
		}
	}

	client := anthropic.NewClient(clientOpts...)
	return &client
}

// SendMessage sends a non-streaming message request to Anthropic
func (ms *MessagesService) SendMessage(
	ctx context.Context,
	client *anthropic.Client,
	req *models.AnthropicMessageRequest,
	requestID string,
) (*anthropic.Message, error) {
	fiberlog.Infof("[%s] Making non-streaming Anthropic API request - model: %s, max_tokens: %d",
		requestID, req.Model, req.MaxTokens)

	// Convert to Anthropic params directly
	params := anthropic.MessageNewParams{
		MaxTokens:     req.MaxTokens,
		Messages:      req.Messages,
		Model:         req.Model,
		Temperature:   req.Temperature,
		TopK:          req.TopK,
		TopP:          req.TopP,
		Metadata:      req.Metadata,
		ServiceTier:   req.ServiceTier,
		StopSequences: req.StopSequences,
		System:        req.System,
		Thinking:      req.Thinking,
		ToolChoice:    req.ToolChoice,
		Tools:         req.Tools,
	}

	startTime := time.Now()
	message, err := client.Messages.New(ctx, params)
	duration := time.Since(startTime)

	if err != nil {
		fiberlog.Errorf("[%s] Anthropic API request failed after %v: %v", requestID, duration, err)
		return nil, models.NewProviderError("anthropic", "message request failed", err)
	}

	fiberlog.Infof("[%s] Anthropic API request completed successfully in %v - usage: input:%d, output:%d",
		requestID, duration, message.Usage.InputTokens, message.Usage.OutputTokens)
	return message, nil
}

// SendStreamingMessage sends a streaming message request to Anthropic
func (ms *MessagesService) SendStreamingMessage(
	ctx context.Context,
	client *anthropic.Client,
	req *models.AnthropicMessageRequest,
	requestID string,
) (*ssestream.Stream[anthropic.MessageStreamEventUnion], error) {
	fiberlog.Infof("[%s] Making streaming Anthropic API request - model: %s, max_tokens: %d",
		requestID, req.Model, req.MaxTokens)

	// Convert to Anthropic params directly
	params := anthropic.MessageNewParams{
		MaxTokens:     req.MaxTokens,
		Messages:      req.Messages,
		Model:         req.Model,
		Temperature:   req.Temperature,
		TopK:          req.TopK,
		TopP:          req.TopP,
		Metadata:      req.Metadata,
		ServiceTier:   req.ServiceTier,
		StopSequences: req.StopSequences,
		System:        req.System,
		Thinking:      req.Thinking,
		ToolChoice:    req.ToolChoice,
		Tools:         req.Tools,
	}

	streamResp := client.Messages.NewStreaming(ctx, params)

	fiberlog.Debugf("[%s] Streaming request initiated successfully", requestID)
	return streamResp, nil
}

// handleAnthropicProvider handles requests using native Anthropic client
func (ms *MessagesService) HandleAnthropicProvider(
	c *fiber.Ctx,
	req *models.AnthropicMessageRequest,
	providerConfig models.ProviderConfig,
	isStreaming bool,
	requestID string,
	responseSvc *ResponseService,
	provider string,
	cacheSource string,
) error {
	fiberlog.Debugf("[%s] Using native Anthropic provider", requestID)
	client := ms.CreateClient(providerConfig)

	if isStreaming {
		// Use context.Background() for streaming - c.Context() gets canceled too early
		// The stream handler will monitor fasthttpCtx for actual client disconnects
		stream, err := ms.SendStreamingMessage(context.Background(), client, req, requestID)
		if err != nil {
			return responseSvc.HandleError(c, err, requestID)
		}
		return responseSvc.HandleStreamingResponse(c, stream, requestID, provider, cacheSource)
	}

	message, err := ms.SendMessage(c.Context(), client, req, requestID)
	if err != nil {
		return responseSvc.HandleError(c, err, requestID)
	}
	return responseSvc.HandleNonStreamingResponse(c, message, requestID, cacheSource)
}
