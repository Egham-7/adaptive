package messages

import (
	"context"
	"fmt"
	"time"

	"adaptive-backend/internal/models"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	fiberlog "github.com/gofiber/fiber/v2/log"
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
	// Set timeout if not already set
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, 5*time.Minute)
		defer cancel()
	}

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
		return nil, fmt.Errorf("anthropic API error: %w", err)
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
	// Set timeout if not already set
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, 10*time.Minute)
		defer cancel()
	}

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
