package format_adapter

import (
	"fmt"

	"adaptive-backend/internal/models"

	"github.com/anthropics/anthropic-sdk-go"
)

// AdaptiveToAnthropicConverter handles conversion from our adaptive types to standard Anthropic types
type AdaptiveToAnthropicConverter struct{}

// ConvertRequest converts our AnthropicMessageRequest to standard Anthropic MessageNewParams
func (c *AdaptiveToAnthropicConverter) ConvertRequest(req *models.AnthropicMessageRequest) (*anthropic.MessageNewParams, error) {
	if req == nil {
		return nil, fmt.Errorf("anthropic message request cannot be nil")
	}

	// Convert our custom request to standard MessageNewParams
	params := &anthropic.MessageNewParams{
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
	}

	return params, nil
}

// ConvertResponse converts our AdaptiveAnthropicMessage to standard Anthropic Message format
func (c *AdaptiveToAnthropicConverter) ConvertResponse(resp *models.AnthropicMessage) (*anthropic.Message, error) {
	if resp == nil {
		return nil, fmt.Errorf("adaptive anthropic message cannot be nil")
	}

	return &anthropic.Message{
		ID:           resp.ID,
		Content:      resp.Content,
		Model:        anthropic.Model(resp.Model),
		Role:         "assistant",
		StopReason:   anthropic.StopReason(resp.StopReason),
		StopSequence: resp.StopSequence,
		Type:         "message",
		Usage:        c.convertUsage(&resp.Usage),
	}, nil
}

// ConvertStreamingChunk converts our AdaptiveAnthropicMessageChunk to standard Anthropic streaming event
func (c *AdaptiveToAnthropicConverter) ConvertStreamingChunk(chunk *models.AnthropicMessageChunk) (*anthropic.MessageStreamEventUnion, error) {
	if chunk == nil {
		return nil, fmt.Errorf("adaptive anthropic message chunk cannot be nil")
	}

	// Create base event union with ONLY the type - no extra fields
	event := &anthropic.MessageStreamEventUnion{
		Type: chunk.Type,
	}

	// Handle different event types - only include relevant fields per Anthropic spec
	switch chunk.Type {
	case "message_start":
		if chunk.Message != nil {
			convertedMessage, err := c.ConvertResponse(chunk.Message)
			if err != nil {
				return nil, fmt.Errorf("failed to convert message in chunk: %w", err)
			}
			event.Message = *convertedMessage
		}
	case "message_delta":
		if chunk.Delta != nil {
			event.Delta = anthropic.MessageStreamEventUnionDelta{
				StopReason:   anthropic.StopReason(chunk.Delta.StopReason),
				StopSequence: chunk.Delta.StopSequence,
			}
		}
		if chunk.Usage != nil {
			// Convert to MessageDeltaUsage type expected by message_delta events
			event.Usage = anthropic.MessageDeltaUsage{
				OutputTokens: chunk.Usage.OutputTokens,
			}
		}
	case "content_block_start":
		if chunk.ContentBlock != nil {
			event.ContentBlock = *chunk.ContentBlock
		}
		if chunk.Index != nil {
			event.Index = *chunk.Index
		}
	case "content_block_delta":
		if chunk.Delta != nil {
			event.Delta = anthropic.MessageStreamEventUnionDelta{
				Type:        chunk.Delta.Type,
				Text:        chunk.Delta.Text,
				PartialJSON: chunk.Delta.PartialJSON,
				Thinking:    chunk.Delta.Thinking,
				Signature:   chunk.Delta.Signature,
			}
		}
		if chunk.Index != nil {
			event.Index = *chunk.Index
		}
	case "content_block_stop":
		if chunk.Index != nil {
			event.Index = *chunk.Index
		}
	case "message_stop":
		// message_stop events only need type
		break
	}

	return event, nil
}

// convertUsage converts AdaptiveAnthropicUsage to Anthropic's Usage for compatibility
func (c *AdaptiveToAnthropicConverter) convertUsage(usage *models.AdaptiveAnthropicUsage) anthropic.Usage {
	return anthropic.Usage{
		CacheCreationInputTokens: usage.CacheCreationInputTokens,
		CacheReadInputTokens:     usage.CacheReadInputTokens,
		InputTokens:              usage.InputTokens,
		OutputTokens:             usage.OutputTokens,
	}
}

// SetCacheTier sets the cache tier on AdaptiveAnthropicUsage based on cache source type
func (c *AdaptiveToAnthropicConverter) SetCacheTier(usage *models.AdaptiveAnthropicUsage, cacheSource string) {
	switch cacheSource {
	case models.CacheTierSemanticExact:
		usage.CacheTier = models.CacheTierSemanticExact
	case models.CacheTierSemanticSimilar:
		usage.CacheTier = models.CacheTierSemanticSimilar
	case models.CacheTierPromptResponse:
		usage.CacheTier = models.CacheTierPromptResponse
	default:
		usage.CacheTier = ""
	}
}
