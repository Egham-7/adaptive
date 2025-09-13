package processors

import (
	"context"
	"encoding/json"
	"fmt"

	"adaptive-backend/internal/services/format_adapter"

	"github.com/anthropics/anthropic-sdk-go"
)

// AnthropicChunkProcessor handles Anthropic-specific format conversion
type AnthropicChunkProcessor struct {
	provider  string
	requestID string
}

// NewAnthropicChunkProcessor creates a new Anthropic chunk processor
func NewAnthropicChunkProcessor(provider, requestID string) *AnthropicChunkProcessor {
	return &AnthropicChunkProcessor{
		provider:  provider,
		requestID: requestID,
	}
}

// Process converts Anthropic chunk data using existing format adapter
func (p *AnthropicChunkProcessor) Process(ctx context.Context, data []byte) ([]byte, error) {
	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Skip empty content
	if len(data) == 0 {
		return []byte{}, nil
	}

	// Try to parse as Anthropic chunk format
	var anthropicChunk anthropic.MessageStreamEventUnion
	if err := json.Unmarshal(data, &anthropicChunk); err != nil {
		// If not JSON, treat as raw text and pass through
		return data, nil
	}

	// Use existing format adapter to convert to adaptive format
	if format_adapter.AnthropicToAdaptive == nil {
		return nil, fmt.Errorf("AnthropicToAdaptive converter is not initialized")
	}

	adaptiveChunk, err := format_adapter.AnthropicToAdaptive.ConvertStreamingChunk(&anthropicChunk, p.provider)
	if err != nil {
		return nil, fmt.Errorf("failed to convert Anthropic chunk: %w", err)
	}

	// Marshal to JSON
	chunkJSON, err := json.Marshal(adaptiveChunk)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal adaptive chunk: %w", err)
	}

	// Format as SSE event
	sseData := fmt.Sprintf("event: %s\ndata: %s\n\n", adaptiveChunk.Type, string(chunkJSON))
	return []byte(sseData), nil
}

// Provider returns the provider name
func (p *AnthropicChunkProcessor) Provider() string {
	return p.provider
}
