package format_adapter

import (
	"fmt"

	"adaptive-backend/internal/models"

	"github.com/anthropics/anthropic-sdk-go"
)

// AnthropicToAdaptiveConverter handles conversion from standard Anthropic types to our adaptive types
type AnthropicToAdaptiveConverter struct{}

// ConvertRequest converts standard Anthropic MessageNewParams to our AnthropicMessageRequest
func (c *AnthropicToAdaptiveConverter) ConvertRequest(req *anthropic.MessageNewParams) (*models.AnthropicMessageRequest, error) {
	if req == nil {
		return nil, fmt.Errorf("anthropic message new params cannot be nil")
	}

	// Create our enhanced request with the standard params copied
	return &models.AnthropicMessageRequest{
		MaxTokens:     req.MaxTokens,
		Messages:      req.Messages,
		Model:         req.Model,
		Temperature:   req.Temperature,
		TopK:          req.TopK,
		TopP:          req.TopP,
		Metadata:      req.Metadata,
		ServiceTier:   req.ServiceTier,
		StopSequences: req.StopSequences,
		System:        req.System,
		Thinking:      req.Thinking,
		ToolChoice:    req.ToolChoice,
		Tools:         req.Tools,
		// Custom fields are left as nil/defaults - caller can set them as needed
		ModelRouterConfig:   nil,
		PromptResponseCache: nil,
		PromptCache:         nil,
		Fallback:            nil,
		ProviderConfigs:     nil,
	}, nil
}

// ConvertResponse converts standard Anthropic Message to our AdaptiveAnthropicMessage
func (c *AnthropicToAdaptiveConverter) ConvertResponse(resp *anthropic.Message, provider string) (*models.AnthropicMessage, error) {
	if resp == nil {
		return nil, fmt.Errorf("anthropic message cannot be nil")
	}

	return &models.AnthropicMessage{
		ID:           resp.ID,
		Content:      resp.Content,
		Model:        string(resp.Model),
		Role:         string(resp.Role),
		StopReason:   string(resp.StopReason),
		StopSequence: resp.StopSequence,
		Type:         string(resp.Type),
		Usage:        *c.convertUsage(resp.Usage),
		Provider:     provider,
	}, nil
}

// ConvertStreamingChunk converts standard Anthropic MessageStreamEventUnion to our AdaptiveAnthropicMessageChunk
func (c *AnthropicToAdaptiveConverter) ConvertStreamingChunk(chunk *anthropic.MessageStreamEventUnion, provider string) (*models.AnthropicMessageChunk, error) {
	if chunk == nil {
		return nil, fmt.Errorf("anthropic message stream event cannot be nil")
	}

	adaptive := &models.AnthropicMessageChunk{
		Type:     chunk.Type,
		Provider: provider,
	}

	// Use typed accessors for all event kinds instead of direct field access
	switch eventVariant := chunk.AsAny().(type) {
	case anthropic.MessageStartEvent:
		convertedMessage, err := c.ConvertResponse(&eventVariant.Message, provider)
		if err != nil {
			return nil, fmt.Errorf("failed to convert message in chunk: %w", err)
		}
		adaptive.Message = convertedMessage

	case anthropic.MessageDeltaEvent:
		// For MessageDeltaEvent, we need to create a MessageStreamEventUnionDelta from the MessageDeltaEventDelta
		// Since the union type has flattened fields, we can construct it by copying the relevant fields
		adaptive.Delta = &anthropic.MessageStreamEventUnionDelta{
			StopReason:   eventVariant.Delta.StopReason,
			StopSequence: eventVariant.Delta.StopSequence,
		}
		// Always include usage when it's present from the API
		adaptive.Usage = &models.AdaptiveAnthropicUsage{
			InputTokens:  eventVariant.Usage.InputTokens,
			OutputTokens: eventVariant.Usage.OutputTokens,
		}

	case anthropic.ContentBlockStartEvent:
		adaptive.ContentBlock = &eventVariant.ContentBlock
		adaptive.Index = &eventVariant.Index

	case anthropic.ContentBlockDeltaEvent:
		// For ContentBlockDeltaEvent, we need to create a MessageStreamEventUnionDelta from the RawContentBlockDeltaUnion
		// Based on the delta type, populate only the relevant fields
		deltaType := eventVariant.Delta.Type
		adaptive.Delta = &anthropic.MessageStreamEventUnionDelta{
			Type: deltaType,
		}

		// Populate only the relevant fields based on delta type
		switch deltaType {
		case "text_delta":
			adaptive.Delta.Text = eventVariant.Delta.Text
		case "input_json_delta":
			adaptive.Delta.PartialJSON = eventVariant.Delta.PartialJSON
			// Add other delta types as needed
		}

		adaptive.Index = &eventVariant.Index

	case anthropic.ContentBlockStopEvent:
		adaptive.Index = &eventVariant.Index

	default:
		// Handle unknown event types gracefully
		return nil, fmt.Errorf("unknown stream event type: %T", eventVariant)
	}

	return adaptive, nil
}

// convertUsage creates AdaptiveAnthropicUsage from Anthropic's Usage
func (c *AnthropicToAdaptiveConverter) convertUsage(usage anthropic.Usage) *models.AdaptiveAnthropicUsage {
	return &models.AdaptiveAnthropicUsage{
		CacheCreationInputTokens: usage.CacheCreationInputTokens,
		CacheReadInputTokens:     usage.CacheReadInputTokens,
		InputTokens:              usage.InputTokens,
		OutputTokens:             usage.OutputTokens,
		ServiceTier:              string(usage.ServiceTier),
	}
}
