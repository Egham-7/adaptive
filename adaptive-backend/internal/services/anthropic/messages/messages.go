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

	fiberlog.Infof("[%s] making non-streaming anthropic request", requestID)

	// Convert to Anthropic params
	params := req.ToAnthropicParams()

	message, err := client.Messages.New(ctx, *params)
	if err != nil {
		fiberlog.Errorf("[%s] anthropic request failed: %v", requestID, err)
		return nil, fmt.Errorf("anthropic API error: %w", err)
	}

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

	fiberlog.Infof("[%s] making streaming anthropic request", requestID)

	// Convert to Anthropic params
	params := req.ToAnthropicParams()

	streamResp := client.Messages.NewStreaming(ctx, *params)
	return streamResp, nil
}
