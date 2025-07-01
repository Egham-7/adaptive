package grok

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

// GrokCompletions implements the Completions interface
type GrokCompletions struct {
	chat *GrokChat
}

// CreateCompletion processes a chat completion request with Grok
func (c *GrokCompletions) CreateCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Map to Grok model if needed
	mappedModel := mapToGrokModel(string(req.Model))
	req.Model = openai.ChatModel(mappedModel)

	// Call Grok API using OpenAI client with custom base URL
	resp, err := c.chat.service.client.Chat.Completions.New(ctx, *req)
	if err != nil {
		return nil, fmt.Errorf("grok chat completion failed: %w", err)
	}

	return resp, nil
}

// StreamCompletion processes a streaming chat completion request with Grok
func (c *GrokCompletions) StreamCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Map to Grok model if needed
	mappedModel := mapToGrokModel(string(req.Model))
	req.Model = openai.ChatModel(mappedModel)

	// Call Grok streaming API using OpenAI client with custom base URL
	stream := c.chat.service.client.Chat.Completions.NewStreaming(ctx, *req)

	return stream, nil
}

// mapToGrokModel maps model names to Grok models
func mapToGrokModel(requestedModel string) string {
	if requestedModel == "" {
		return "grok-3" // Default model
	}

	// Map of supported models
	supportedModels := map[string]bool{
		"grok-beta":    true,
		"grok-3":       true,
		"grok-3-mini":  true,
	}

	// If the requested model is supported, use it
	if _, ok := supportedModels[requestedModel]; ok {
		return requestedModel
	}

	// For unknown models, fall back to default
	return "grok-3"
}