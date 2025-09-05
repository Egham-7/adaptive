package format_adapter

import (
	"fmt"

	"adaptive-backend/internal/models"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/shared/constant"
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

	// Convert adaptive choices back to OpenAI types
	openAIChoices := make([]openai.ChatCompletionChoice, len(resp.Choices))
	for i, choice := range resp.Choices {
		openAIChoices[i] = openai.ChatCompletionChoice{
			FinishReason: choice.FinishReason,
			Index:        choice.Index,
			Logprobs:     choice.Logprobs,
			Message: openai.ChatCompletionMessage{
				Content:     choice.Message.Content,
				Refusal:     choice.Message.Refusal,
				Role:        constant.Assistant(choice.Message.Role),
				Annotations: choice.Message.Annotations,
				Audio:       choice.Message.Audio,
				ToolCalls:   choice.Message.ToolCalls,
			},
		}
	}

	return &openai.ChatCompletion{
		ID:          resp.ID,
		Choices:     openAIChoices,
		Created:     resp.Created,
		Model:       resp.Model,
		Object:      "chat.completion",
		ServiceTier: resp.ServiceTier,
		Usage:       c.convertUsage(resp.Usage),
	}, nil
}

// ConvertStreamingChunk converts our ChatCompletionChunk to standard OpenAI ChatCompletionChunk
func (c *AdaptiveToOpenAIConverter) ConvertStreamingChunk(chunk *models.ChatCompletionChunk) (*openai.ChatCompletionChunk, error) {
	if chunk == nil {
		return nil, fmt.Errorf("adaptive chat completion chunk cannot be nil")
	}

	// Convert adaptive choices back to OpenAI types
	openAIChoices := make([]openai.ChatCompletionChunkChoice, len(chunk.Choices))
	for i, choice := range chunk.Choices {
		openAIChoices[i] = openai.ChatCompletionChunkChoice{
			Delta: openai.ChatCompletionChunkChoiceDelta{
				Content:   choice.Delta.Content,
				Refusal:   choice.Delta.Refusal,
				Role:      choice.Delta.Role,
				ToolCalls: choice.Delta.ToolCalls,
			},
			FinishReason: choice.FinishReason,
			Index:        choice.Index,
			Logprobs:     choice.Logprobs,
		}
	}

	usage := c.convertUsage(chunk.Usage)

	return &openai.ChatCompletionChunk{
		ID:          chunk.ID,
		Choices:     openAIChoices,
		Created:     chunk.Created,
		Model:       chunk.Model,
		Object:      "chat.completion.chunk",
		ServiceTier: chunk.ServiceTier,
		Usage:       usage,
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
