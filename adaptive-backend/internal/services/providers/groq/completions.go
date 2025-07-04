package groq

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

// GroqCompletions implements the Completions interface
type GroqCompletions struct {
	chat *GroqChat
}

// CreateCompletion processes a chat completion request with Groq
func (c *GroqCompletions) CreateCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Map to Groq model if needed
	mappedModel := mapToGroqModel(string(req.Model))
	req.Model = openai.ChatModel(mappedModel)

	// Call Groq API using OpenAI client with custom base URL
	resp, err := c.chat.service.client.Chat.Completions.New(ctx, *req)
	if err != nil {
		return nil, fmt.Errorf("groq chat completion failed: %w", err)
	}

	return resp, nil
}

// StreamCompletion processes a streaming chat completion request with Groq
func (c *GroqCompletions) StreamCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Map to Groq model if needed
	mappedModel := mapToGroqModel(string(req.Model))
	req.Model = openai.ChatModel(mappedModel)

	// Call Groq streaming API using OpenAI client with custom base URL
	stream := c.chat.service.client.Chat.Completions.NewStreaming(ctx, *req)

	return stream, nil
}

// mapToGroqModel maps model names to Groq models
func mapToGroqModel(requestedModel string) string {
	if requestedModel == "" {
		return "llama-3.3-70b-versatile" // Default model
	}

	// Map of supported production models
	supportedModels := map[string]bool{
		// Production Models
		"distil-whisper-large-v3-en":   true,
		"gemma2-9b-it":                 true,
		"llama-3.1-8b-instant":         true,
		"llama-3.3-70b-versatile":      true,
		"meta-llama/llama-guard-4-12b": true,
		"whisper-large-v3":             true,
		"whisper-large-v3-turbo":       true,

		// Preview Models
		"deepseek-r1-distill-llama-70b":                 true,
		"meta-llama/llama-4-maverick-17b-128e-instruct": true,
		"meta-llama/llama-4-scout-17b-16e-instruct":     true,
		"meta-llama/llama-prompt-guard-2-22m":           true,
		"meta-llama/llama-prompt-guard-2-86m":           true,
		"mistral-saba-24b":                              true,
		"playai-tts":                                    true,
		"playai-tts-arabic":                             true,
		"qwen-qwq-32b":                                  true,
		"qwen/qwen3-32b":                                true,

		// Preview Systems
		"compound-beta":      true,
		"compound-beta-mini": true,
	}

	// If the requested model is supported, use it
	if _, ok := supportedModels[requestedModel]; ok {
		return requestedModel
	}

	// For unknown models, fall back to default
	return "llama-3.3-70b-versatile"
}
