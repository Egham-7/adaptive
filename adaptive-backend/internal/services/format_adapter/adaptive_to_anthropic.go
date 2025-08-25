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

	// Return the embedded MessageNewParams (our request already extends the standard one)
	return &req.MessageNewParams, nil
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

	// Create base event union
	event := &anthropic.MessageStreamEventUnion{
		Type: chunk.Type,
	}

	// Handle different event types
	switch chunk.Type {
	case "message_start":
		if chunk.Message != nil {
			_, err := c.ConvertResponse(chunk.Message)
			if err != nil {
				return nil, fmt.Errorf("failed to convert message in chunk: %w", err)
			}
			// Note: Actual assignment would depend on Anthropic SDK union structure
		}
	case "message_delta":
		if chunk.Delta != nil {
			event.Delta = *chunk.Delta
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
			event.Delta = *chunk.Delta
		}
		if chunk.Index != nil {
			event.Index = *chunk.Index
		}
	case "content_block_stop":
		if chunk.Index != nil {
			event.Index = *chunk.Index
		}
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
	case "semantic_exact":
		usage.CacheTier = "semantic_exact"
	case "semantic_similar":
		usage.CacheTier = "semantic_similar"
	case "prompt_response":
		usage.CacheTier = "prompt_response"
	default:
		usage.CacheTier = ""
	}
}
