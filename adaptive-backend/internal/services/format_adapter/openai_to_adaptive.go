package format_adapter

import (
	"fmt"

	"adaptive-backend/internal/models"

	"github.com/openai/openai-go/v2"
)

// OpenAIToAdaptiveConverter handles conversion from standard OpenAI types to our adaptive types
type OpenAIToAdaptiveConverter struct{}

// ConvertRequest converts standard OpenAI ChatCompletionNewParams to our ChatCompletionRequest
func (c *OpenAIToAdaptiveConverter) ConvertRequest(req *openai.ChatCompletionNewParams) (*models.ChatCompletionRequest, error) {
	if req == nil {
		return nil, fmt.Errorf("openai chat completion new params cannot be nil")
	}

	// Create our enhanced request with the standard params and custom fields
	return &models.ChatCompletionRequest{
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
		// Custom fields are left as nil/defaults - caller can set them as needed
		ModelRouterConfig: nil,
		PromptCache:       nil,
		Fallback:          nil,
		ProviderConfigs:   nil,
	}, nil
}

// ConvertResponse converts standard OpenAI ChatCompletion to our ChatCompletion
func (c *OpenAIToAdaptiveConverter) ConvertResponse(resp *openai.ChatCompletion, provider string) (*models.ChatCompletion, error) {
	if resp == nil {
		return nil, fmt.Errorf("openai chat completion cannot be nil")
	}

	return &models.ChatCompletion{
		ID:          resp.ID,
		Choices:     resp.Choices,
		Created:     resp.Created,
		Model:       resp.Model,
		Object:      string(resp.Object),
		ServiceTier: resp.ServiceTier,
		Usage:       c.convertUsage(resp.Usage),
		Provider:    provider,
	}, nil
}

// ConvertStreamingChunk converts standard OpenAI ChatCompletionChunk to our ChatCompletionChunk
func (c *OpenAIToAdaptiveConverter) ConvertStreamingChunk(chunk *openai.ChatCompletionChunk, provider string) (*models.ChatCompletionChunk, error) {
	if chunk == nil {
		return nil, fmt.Errorf("openai chat completion chunk cannot be nil")
	}

	var usage *models.AdaptiveUsage
	// Check if usage is provided (only in the last chunk typically)
	if chunk.Usage.CompletionTokens != 0 || chunk.Usage.PromptTokens != 0 || chunk.Usage.TotalTokens != 0 {
		converted := c.convertUsage(chunk.Usage)
		usage = &converted
	}

	return &models.ChatCompletionChunk{
		ID:          chunk.ID,
		Choices:     chunk.Choices,
		Created:     chunk.Created,
		Model:       chunk.Model,
		Object:      string(chunk.Object),
		ServiceTier: chunk.ServiceTier,
		Usage:       usage,
		Provider:    provider,
	}, nil
}

// convertUsage converts OpenAI's CompletionUsage to our AdaptiveUsage
func (c *OpenAIToAdaptiveConverter) convertUsage(usage openai.CompletionUsage) models.AdaptiveUsage {
	return models.AdaptiveUsage{
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		TotalTokens:      usage.TotalTokens,
		// Our custom fields start as empty - can be set by caller
		CacheTier: "",
	}
}
