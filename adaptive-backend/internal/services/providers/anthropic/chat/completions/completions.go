package completions

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

type AnthropicCompletions struct {
	client *openai.Client
}

// NewAnthropicCompletions creates a new Anthropic completions service using OpenAI Go SDK
func NewAnthropicCompletions(client *openai.Client) *AnthropicCompletions {
	return &AnthropicCompletions{
		client: client,
	}
}

// CreateCompletion processes a chat completion request with Anthropic using OpenAI-compatible API
func (c *AnthropicCompletions) CreateCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	resp, err := c.client.Chat.Completions.New(ctx, *req)
	if err != nil {
		return nil, fmt.Errorf("anthropic chat completion failed: %w", err)
	}

	return resp, nil
}

// StreamCompletion streams a chat completion using Anthropic's OpenAI-compatible API
func (c *AnthropicCompletions) StreamCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	stream := c.client.Chat.Completions.NewStreaming(ctx, *req)

	return stream, nil
}
