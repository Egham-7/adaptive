package completions

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

type AdaptiveCompletions struct {
	client *openai.Client
}

// NewAdaptiveCompletions creates a new Adaptive completions service
func NewAdaptiveCompletions(client *openai.Client) *AdaptiveCompletions {
	return &AdaptiveCompletions{
		client: client,
	}
}

// CreateCompletion processes a chat completion request with Adaptive
func (c *AdaptiveCompletions) CreateCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}
	resp, err := c.client.Chat.Completions.New(ctx, *req)
	if err != nil {
		return nil, fmt.Errorf("adaptive chat completion failed: %w", err)
	}
	return resp, nil
}

// StreamCompletion streams a chat completion using Adaptive
func (c *AdaptiveCompletions) StreamCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	stream := c.client.Chat.Completions.NewStreaming(ctx, *req)

	return stream, nil
}
