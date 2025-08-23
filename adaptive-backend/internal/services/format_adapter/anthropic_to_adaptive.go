package format_adapter

import (
	"adaptive-backend/internal/models"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
)

// AnthropicToAdaptiveConverter handles conversion from standard Anthropic types to our adaptive types
type AnthropicToAdaptiveConverter struct{}

// ConvertRequest converts standard Anthropic MessageNewParams to our AnthropicMessageRequest
func (c *AnthropicToAdaptiveConverter) ConvertRequest(req *anthropic.MessageNewParams) (*models.AnthropicMessageRequest, error) {
	if req == nil {
		return nil, fmt.Errorf("anthropic message new params cannot be nil")
	}

	// Create our enhanced request with the standard params embedded
	return &models.AnthropicMessageRequest{
		MessageNewParams: *req,
		// Custom fields are left as nil/defaults - caller can set them as needed
		ProtocolManagerConfig: nil,
		SemanticCache:         nil,
		PromptCache:           nil,
		Fallback:              nil,
		ProviderConfigs:       nil,
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

	// Handle different event types
	switch chunk.Type {
	case "message_start":
		startEvent := chunk.AsMessageStart()
		convertedMessage, err := c.ConvertResponse(&startEvent.Message, provider)
		if err != nil {
			return nil, fmt.Errorf("failed to convert message in chunk: %w", err)
		}
		adaptive.Message = convertedMessage
	case "message_delta":
		deltaEvent := chunk.AsMessageDelta()
		adaptive.Delta = &chunk.Delta
		if deltaEvent.Usage.OutputTokens != 0 || deltaEvent.Usage.InputTokens != 0 {
			adaptive.Usage = &models.AdaptiveAnthropicUsage{
				InputTokens:  deltaEvent.Usage.InputTokens,
				OutputTokens: deltaEvent.Usage.OutputTokens,
			}
		}
	case "content_block_start":
		adaptive.ContentBlock = &chunk.ContentBlock
		adaptive.Index = &chunk.Index
	case "content_block_delta":
		adaptive.Delta = &chunk.Delta
		adaptive.Index = &chunk.Index
	case "content_block_stop":
		adaptive.Index = &chunk.Index
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
