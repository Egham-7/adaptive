package messages

import (
	"context"
	"fmt"
	"time"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/format_adapter"
	"adaptive-backend/internal/services/stream/adapters"
	"adaptive-backend/internal/services/stream/handlers"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go/v2"
	openaiOption "github.com/openai/openai-go/v2/option"
)

// MessagesService handles Anthropic Messages API calls using the Anthropic SDK
type MessagesService struct{}

// NewMessagesService creates a new MessagesService
func NewMessagesService() *MessagesService {
	return &MessagesService{}
}

// CreateClient creates an Anthropic client with the given provider configuration
func (ms *MessagesService) CreateClient(providerConfig models.ProviderConfig) *anthropic.Client {
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

// HandleProviderRequest routes the request to the appropriate provider and handles response
func (ms *MessagesService) HandleProviderRequest(
	c *fiber.Ctx,
	req *models.AnthropicMessageRequest,
	provider string,
	providerConfig models.ProviderConfig,
	isStreaming bool,
	requestID string,
	responseSvc *ResponseService,
	cacheSource string,
) error {
	// Check provider's native format to determine if conversion is needed
	if providerConfig.NativeFormat == "anthropic" || providerConfig.NativeFormat == "" || provider == "anthropic" {
		// Native Anthropic format - use directly
		return ms.handleAnthropicProvider(c, req, providerConfig, isStreaming, requestID, responseSvc, provider, cacheSource)
	}

	// Provider uses different native format (likely OpenAI) - convert via format adapters
	return ms.handleNonAnthropicProvider(c, req, provider, providerConfig, isStreaming, requestID, cacheSource)
}

// handleAnthropicProvider handles requests using native Anthropic client
func (ms *MessagesService) handleAnthropicProvider(
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
		stream, err := ms.SendStreamingMessage(c.Context(), client, req, requestID)
		if err != nil {
			return responseSvc.HandleError(c, err, requestID)
		}
		return responseSvc.HandleStreamingResponse(c, stream, requestID, provider)
	}

	message, err := ms.SendMessage(c.Context(), client, req, requestID)
	if err != nil {
		return responseSvc.HandleError(c, err, requestID)
	}
	return responseSvc.HandleNonStreamingResponse(c, message, requestID, cacheSource)
}

// handleNonAnthropicProvider handles providers that use different native formats (e.g., OpenAI)
func (ms *MessagesService) handleNonAnthropicProvider(
	c *fiber.Ctx,
	req *models.AnthropicMessageRequest,
	provider string,
	providerConfig models.ProviderConfig,
	isStreaming bool,
	requestID string,
	cacheSource string,
) error {
	fiberlog.Infof("[%s] Converting Anthropic request for non-Anthropic provider: %s", requestID, provider)

	// Convert Anthropic Messages request to OpenAI Chat Completions format
	openaiReq, err := format_adapter.AnthropicToOpenAI.ConvertRequest(req)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to convert Anthropic request to OpenAI format: %v", requestID, err)
		return fmt.Errorf("failed to convert request format: %w", err)
	}

	// Create OpenAI client for the provider
	client, err := ms.createOpenAIClient(providerConfig)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to create OpenAI client for provider %s: %v", requestID, provider, err)
		return fmt.Errorf("failed to create client: %w", err)
	}

	if isStreaming {
		return ms.handleOpenAIStreamingRequest(c, client, openaiReq, provider, requestID, cacheSource)
	}
	return ms.handleOpenAINonStreamingRequest(c, client, openaiReq, provider, requestID, cacheSource)
}

// createOpenAIClient creates an OpenAI client for non-Anthropic providers
func (ms *MessagesService) createOpenAIClient(providerConfig models.ProviderConfig) (*openai.Client, error) {
	if providerConfig.APIKey == "" {
		return nil, fmt.Errorf("API key not configured for provider")
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

	client := openai.NewClient(opts...)
	return &client, nil
}

// handleOpenAINonStreamingRequest processes non-streaming requests for non-Anthropic providers
func (ms *MessagesService) handleOpenAINonStreamingRequest(
	c *fiber.Ctx,
	client *openai.Client,
	req *models.ChatCompletionRequest,
	provider string,
	requestID string,
	cacheSource string,
) error {
	fiberlog.Debugf("[%s] Sending non-streaming chat completion to %s", requestID, provider)

	response, err := client.Chat.Completions.New(c.Context(), openai.ChatCompletionNewParams{
		Messages:    req.Messages,
		Model:       req.Model,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
		Stop:        req.Stop,
	})
	if err != nil {
		fiberlog.Errorf("[%s] Chat completion failed for provider %s: %v", requestID, provider, err)
		return models.NewProviderError(provider, "completion request failed", err)
	}

	// Convert OpenAI response back to Anthropic format
	anthropicResp, err := format_adapter.OpenAIToAnthropic.ConvertResponse(response, provider, cacheSource)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to convert OpenAI response to Anthropic format: %v", requestID, err)
		return fmt.Errorf("failed to convert response format: %w", err)
	}

	fiberlog.Infof("[%s] Non-streaming completion completed successfully for provider %s", requestID, provider)
	return c.JSON(anthropicResp)
}

// handleOpenAIStreamingRequest processes streaming requests for non-Anthropic providers
func (ms *MessagesService) handleOpenAIStreamingRequest(
	c *fiber.Ctx,
	client *openai.Client,
	req *models.ChatCompletionRequest,
	provider string,
	requestID string,
	cacheSource string,
) error {
	fiberlog.Debugf("[%s] Starting streaming chat completion to %s", requestID, provider)

	// Create streaming request
	openaiStream := client.Chat.Completions.NewStreaming(c.Context(), openai.ChatCompletionNewParams{
		Messages:    req.Messages,
		Model:       req.Model,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
		Stop:        req.Stop,
	})

	// Convert OpenAI stream to Anthropic format using stream adapter
	streamReader := adapters.NewOpenAIToAnthropicStreamAdapter(openaiStream, provider, requestID)

	// Handle stream using the stream handler
	fiberlog.Infof("[%s] Streaming completion started for provider %s", requestID, provider)
	return handlers.HandleAnthropic(c, streamReader, requestID, provider)
}
