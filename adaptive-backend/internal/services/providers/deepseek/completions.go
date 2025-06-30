package deepseek

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

// DeepSeekCompletions implements the Completions interface
type DeepSeekCompletions struct {
	chat *DeepSeekChat
}

// CreateCompletion processes a chat completion request with DeepSeek
func (c *DeepSeekCompletions) CreateCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Map to DeepSeek model if needed
	mappedModel := mapToDeepSeekModel(string(req.Model))
	req.Model = openai.ChatModel(mappedModel)

	// Call DeepSeek API using OpenAI client with custom base URL
	resp, err := c.chat.service.client.Chat.Completions.New(ctx, *req)
	if err != nil {
		return nil, fmt.Errorf("deepseek chat completion failed: %w", err)
	}

	return resp, nil
}

// StreamCompletion processes a streaming chat completion request with DeepSeek
func (c *DeepSeekCompletions) StreamCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Map to DeepSeek model if needed
	mappedModel := mapToDeepSeekModel(string(req.Model))
	req.Model = openai.ChatModel(mappedModel)

	// Call DeepSeek streaming API using OpenAI client with custom base URL
	stream := c.chat.service.client.Chat.Completions.NewStreaming(ctx, *req)

	return stream, nil
}

// mapToDeepSeekModel maps model names to DeepSeek models
func mapToDeepSeekModel(requestedModel string) string {
	if requestedModel == "" {
		return "deepseek-chat" // Default model
	}

	// Map of supported models
	supportedModels := map[string]bool{
		"deepseek-chat":     true,
		"deepseek-reasoner": true,
	}

	// If the requested model is supported, use it
	if _, ok := supportedModels[requestedModel]; ok {
		return requestedModel
	}

	// For unknown models, fall back to default
	return "deepseek-chat"
}
