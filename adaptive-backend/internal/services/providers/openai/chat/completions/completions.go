package completions

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

type OpenAICompletions struct {
	client *openai.Client
}

// NewOpenAICompletions creates a new OpenAI completions service
func NewOpenAICompletions(client *openai.Client) *OpenAICompletions {
	return &OpenAICompletions{
		client: client,
	}
}

// CreateCompletion processes a chat completion request with OpenAI
func (c *OpenAICompletions) CreateCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}
	resp, err := c.client.Chat.Completions.New(ctx, *req)
	if err != nil {
		return nil, fmt.Errorf("openai chat completion failed: %w", err)
	}
	return resp, nil
}

// StreamCompletion streams a chat completion using OpenAI
func (c *OpenAICompletions) StreamCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	stream := c.client.Chat.Completions.NewStreaming(ctx, *req)

	return stream, nil
}
