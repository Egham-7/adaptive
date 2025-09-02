package sse

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/format_adapter"

	"github.com/anthropics/anthropic-sdk-go"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// AnthropicSSEReader handles Anthropic Server-Sent Events streaming
type AnthropicSSEReader struct {
	reader   *bufio.Reader
	reqID    string
	provider string
	ctx      context.Context
}

// NewAnthropicSSEReader creates a new Anthropic SSE reader
func NewAnthropicSSEReader(reader io.Reader, reqID, provider string, ctx context.Context) *AnthropicSSEReader {
	if ctx == nil {
		ctx = context.Background()
	}
	return &AnthropicSSEReader{
		reader:   bufio.NewReader(reader),
		reqID:    reqID,
		provider: provider,
		ctx:      ctx,
	}
}

// Read implements io.Reader interface for compatibility with stream processing
func (r *AnthropicSSEReader) Read(p []byte) (n int, err error) {
	// Check for context cancellation
	select {
	case <-r.ctx.Done():
		fiberlog.Infof("[%s] Context cancelled, stopping Anthropic stream", r.reqID)
		return 0, r.ctx.Err()
	default:
	}

	// Read data from the underlying reader
	return r.reader.Read(p)
}

// ReadChunk reads the next chunk from the Anthropic SSE stream
func (r *AnthropicSSEReader) ReadChunk() (*models.AnthropicMessageChunk, error) {
	for {
		line, readErr := r.reader.ReadString('\n')
		isEOF := readErr == io.EOF

		if readErr != nil && !isEOF {
			return nil, fmt.Errorf("failed to read from Anthropic SSE stream: %w", readErr)
		}

		line = strings.TrimSpace(line)

		// Skip empty lines
		if line == "" {
			if isEOF {
				return nil, io.EOF
			}
			continue
		}

		// Handle SSE format
		if data, found := strings.CutPrefix(line, "data: "); found {

			// Handle [DONE] marker
			if data == "[DONE]" {
				return &models.AnthropicMessageChunk{
					Type:     "message_stop",
					Provider: r.provider,
				}, nil
			}

			// Parse Anthropic streaming event
			var anthropicChunk anthropic.MessageStreamEventUnion
			if err := json.Unmarshal([]byte(data), &anthropicChunk); err != nil {
				fiberlog.Errorf("[%s] Failed to parse Anthropic streaming chunk: %v", r.reqID, err)
				if isEOF {
					return nil, io.EOF
				}
				continue
			}

			// Convert Anthropic chunk to our adaptive Anthropic format using format adapter
			adaptiveChunk, err := r.convertToAdaptiveAnthropicChunk(&anthropicChunk)
			if err != nil {
				fiberlog.Errorf("[%s] Failed to convert Anthropic chunk: %v", r.reqID, err)
				if isEOF {
					return nil, io.EOF
				}
				continue
			}

			return adaptiveChunk, nil
		}

		// Handle event lines (optional)
		if eventType, found := strings.CutPrefix(line, "event: "); found {
			// Log event type for debugging
			fiberlog.Debugf("[%s] Anthropic SSE event: %s", r.reqID, eventType)
			if isEOF {
				return nil, io.EOF
			}
			continue
		}

		// Skip other SSE metadata lines (id:, retry:, etc.)
		if strings.Contains(line, ":") {
			if isEOF {
				return nil, io.EOF
			}
			continue
		}

		// If we processed a partial line and reached this point, return EOF
		if isEOF {
			return nil, io.EOF
		}
	}
}

// convertToAdaptiveAnthropicChunk converts Anthropic MessageStreamEvent to our adaptive Anthropic format using format adapter
func (r *AnthropicSSEReader) convertToAdaptiveAnthropicChunk(anthropicChunk *anthropic.MessageStreamEventUnion) (*models.AnthropicMessageChunk, error) {
	// Check if the format adapter is available
	if format_adapter.AnthropicToAdaptive == nil {
		return nil, fmt.Errorf("format_adapter.AnthropicToAdaptive is not initialized")
	}

	// Convert using format adapter with provider information
	adaptiveChunk, err := format_adapter.AnthropicToAdaptive.ConvertStreamingChunk(anthropicChunk, r.provider)
	if err != nil {
		return nil, fmt.Errorf("failed to convert Anthropic chunk to adaptive format: %w", err)
	}

	return adaptiveChunk, nil
}

// Close closes the reader (if applicable)
func (r *AnthropicSSEReader) Close() error {
	// Anthropic SSE reader doesn't need explicit cleanup
	return nil
}
