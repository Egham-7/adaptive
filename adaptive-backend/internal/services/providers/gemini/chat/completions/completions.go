package completions

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

type GeminiCompletions struct {
	client *openai.Client
}

// NewGeminiCompletions creates a new Gemini completions service using OpenAI Go SDK
func NewGeminiCompletions(client *openai.Client) *GeminiCompletions {
	return &GeminiCompletions{
		client: client,
	}
}

// CreateCompletion processes a chat completion request with Gemini using OpenAI-compatible API
func (c *GeminiCompletions) CreateCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	resp, err := c.client.Chat.Completions.New(ctx, *req)
	if err != nil {
		return nil, fmt.Errorf("gemini chat completion failed: %w", err)
	}

	return resp, nil
}

// StreamCompletion streams a chat completion using Gemini's OpenAI-compatible API
func (c *GeminiCompletions) StreamCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	stream := c.client.Chat.Completions.NewStreaming(ctx, *req)

	return stream, nil
}