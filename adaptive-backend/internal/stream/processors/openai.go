package processors

import (
	"context"
	"encoding/json"
	"fmt"

	"adaptive-backend/internal/services/format_adapter"

	"github.com/openai/openai-go/v2"
)

// OpenAIChunkProcessor handles OpenAI-specific format conversion
type OpenAIChunkProcessor struct {
	provider    string
	cacheSource string
	requestID   string
}

// NewOpenAIChunkProcessor creates a new OpenAI chunk processor
func NewOpenAIChunkProcessor(provider, cacheSource, requestID string) *OpenAIChunkProcessor {
	return &OpenAIChunkProcessor{
		provider:    provider,
		cacheSource: cacheSource,
		requestID:   requestID,
	}
}

// Process converts OpenAI chunk data using existing format adapter
func (p *OpenAIChunkProcessor) Process(ctx context.Context, data []byte) ([]byte, error) {
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

	// Deserialize OpenAI chunk from data
	var openaiChunk openai.ChatCompletionChunk
	if err := json.Unmarshal(data, &openaiChunk); err != nil {
		return nil, fmt.Errorf("failed to unmarshal OpenAI chunk: %w", err)
	}

	// Use existing format adapter to convert to adaptive format
	adaptiveChunk, err := format_adapter.OpenAIToAdaptive.ConvertStreamingChunk(&openaiChunk, p.provider, p.cacheSource)
	if err != nil {
		return nil, fmt.Errorf("failed to convert OpenAI chunk: %w", err)
	}

	// Marshal to JSON
	chunkJSON, err := json.Marshal(adaptiveChunk)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal adaptive chunk: %w", err)
	}

	// Format as SSE event
	sseData := fmt.Sprintf("data: %s\n\n", string(chunkJSON))
	return []byte(sseData), nil
}

// Provider returns the provider name
func (p *OpenAIChunkProcessor) Provider() string {
	return p.provider
}
