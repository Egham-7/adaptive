package completions

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

type HuggingFaceCompletions struct {
	client *openai.Client
}

// NewHuggingFaceCompletions creates a new HuggingFace completions service
func NewHuggingFaceCompletions(client *openai.Client) *HuggingFaceCompletions {
	if client == nil {
		panic("client cannot be nil")
	}
	return &HuggingFaceCompletions{
		client: client,
	}
}

// CreateCompletion processes a chat completion request with HuggingFace
func (c *HuggingFaceCompletions) CreateCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}
	resp, err := c.client.Chat.Completions.New(ctx, *req)
	if err != nil {
		return nil, fmt.Errorf("huggingface chat completion failed: %w", err)
	}
	return resp, nil
}

// StreamCompletion streams a chat completion using HuggingFace
func (c *HuggingFaceCompletions) StreamCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	stream := c.client.Chat.Completions.NewStreaming(ctx, *req)

	return stream, nil
}
