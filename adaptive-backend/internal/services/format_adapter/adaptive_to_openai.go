package format_adapter

import (
	"adaptive-backend/internal/models"
	"fmt"

	"github.com/openai/openai-go"
)

// AdaptiveToOpenAIConverter handles conversion from our adaptive types to standard OpenAI types
type AdaptiveToOpenAIConverter struct{}

// ConvertRequest converts our ChatCompletionRequest to standard OpenAI ChatCompletionNewParams
func (c *AdaptiveToOpenAIConverter) ConvertRequest(req *models.ChatCompletionRequest) (*openai.ChatCompletionNewParams, error) {
	if req == nil {
		return nil, fmt.Errorf("chat completion request cannot be nil")
	}

	// Create OpenAI params from our request (excluding our custom fields)
	return &openai.ChatCompletionNewParams{
		Messages:            req.Messages,
		Model:               req.Model,
		FrequencyPenalty:    req.FrequencyPenalty,
		Logprobs:            req.Logprobs,
		MaxCompletionTokens: req.MaxCompletionTokens,
		MaxTokens:           req.MaxTokens,
		N:                   req.N,
		PresencePenalty:     req.PresencePenalty,
		ResponseFormat:      req.ResponseFormat,
		Seed:                req.Seed,
		ServiceTier:         req.ServiceTier,
		Stop:                req.Stop,
		Store:               req.Store,
		StreamOptions:       req.StreamOptions,
		Temperature:         req.Temperature,
		ToolChoice:          req.ToolChoice,
		Tools:               req.Tools,
		TopLogprobs:         req.TopLogprobs,
		TopP:                req.TopP,
		User:                req.User,
		Audio:               req.Audio,
		LogitBias:           req.LogitBias,
		Metadata:            req.Metadata,
		Modalities:          req.Modalities,
		ReasoningEffort:     req.ReasoningEffort,
	}, nil
}

// ConvertResponse converts our ChatCompletion to standard OpenAI ChatCompletion format
func (c *AdaptiveToOpenAIConverter) ConvertResponse(resp *models.ChatCompletion) (*openai.ChatCompletion, error) {
	if resp == nil {
		return nil, fmt.Errorf("adaptive chat completion cannot be nil")
	}

	return &openai.ChatCompletion{
		ID:                resp.ID,
		Choices:           resp.Choices,
		Created:           resp.Created,
		Model:             resp.Model,
		Object:            "chat.completion",
		ServiceTier:       resp.ServiceTier,
		SystemFingerprint: resp.SystemFingerprint,
		Usage:             c.convertUsage(resp.Usage),
	}, nil
}

// ConvertStreamingChunk converts our ChatCompletionChunk to standard OpenAI ChatCompletionChunk
func (c *AdaptiveToOpenAIConverter) ConvertStreamingChunk(chunk *models.ChatCompletionChunk) (*openai.ChatCompletionChunk, error) {
	if chunk == nil {
		return nil, fmt.Errorf("adaptive chat completion chunk cannot be nil")
	}

	var usage openai.CompletionUsage
	if chunk.Usage != nil {
		usage = c.convertUsage(*chunk.Usage)
	}

	return &openai.ChatCompletionChunk{
		ID:                chunk.ID,
		Choices:           chunk.Choices,
		Created:           chunk.Created,
		Model:             chunk.Model,
		Object:            "chat.completion.chunk",
		ServiceTier:       chunk.ServiceTier,
		SystemFingerprint: chunk.SystemFingerprint,
		Usage:             usage,
	}, nil
}

// convertUsage converts AdaptiveUsage to OpenAI's Usage for compatibility
func (c *AdaptiveToOpenAIConverter) convertUsage(usage models.AdaptiveUsage) openai.CompletionUsage {
	return openai.CompletionUsage{
		CompletionTokens: usage.CompletionTokens,
		PromptTokens:     usage.PromptTokens,
		TotalTokens:      usage.TotalTokens,
	}
}
