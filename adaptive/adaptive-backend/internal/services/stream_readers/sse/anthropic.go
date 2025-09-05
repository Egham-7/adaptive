package sse

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/format_adapter"

	"github.com/anthropics/anthropic-sdk-go"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// AnthropicSSEReader handles Anthropic Server-Sent Events streaming
type AnthropicSSEReader struct {
	reader    *bufio.Reader
	reqID     string
	provider  string
	ctx       context.Context
	closer    io.Closer
	closeOnce sync.Once
}

// NewAnthropicSSEReader creates a new Anthropic SSE reader
func NewAnthropicSSEReader(reader io.Reader, reqID, provider string, ctx context.Context) *AnthropicSSEReader {
	if ctx == nil {
		ctx = context.Background()
	}

	var closer io.Closer
	if readCloser, ok := reader.(io.ReadCloser); ok {
		closer = readCloser
	}

	r := &AnthropicSSEReader{
		reader:   bufio.NewReader(reader),
		reqID:    reqID,
		provider: provider,
		ctx:      ctx,
		closer:   closer,
	}

	// Spawn goroutine to handle context cancellation
	if closer != nil {
		go func() {
			<-ctx.Done()
			fiberlog.Debugf("[%s] Context cancelled, closing Anthropic stream", reqID)
			if closeErr := closer.Close(); closeErr != nil {
				fiberlog.Debugf("[%s] Error closing stream on cancellation: %v", reqID, closeErr)
			}
		}()
	}

	return r
}

// Read implements io.Reader interface for compatibility with stream processing
func (r *AnthropicSSEReader) Read(p []byte) (n int, err error) {
	// Non-blocking context check for performance
	select {
	case <-r.ctx.Done():
		fiberlog.Infof("[%s] Context cancelled, stopping Anthropic stream", r.reqID)
		// Use the concurrency-safe Close method
		if closeErr := r.Close(); closeErr != nil {
			fiberlog.Debugf("[%s] Error closing reader on cancellation: %v", r.reqID, closeErr)
		}
		return 0, r.ctx.Err()
	default:
	}

	// Delegate to underlying reader (zero-copy when possible)
	return r.reader.Read(p)
}

// ReadChunk reads the next chunk from the Anthropic SSE stream with high performance
func (r *AnthropicSSEReader) ReadChunk() (*models.AnthropicMessageChunk, error) {
	var currentEventType string

	for {
		line, readErr := r.reader.ReadString('\n')
		isEOF := readErr == io.EOF

		// Handle read errors early (fail-fast principle)
		if readErr != nil && !isEOF {
			return nil, fmt.Errorf("failed to read from Anthropic SSE stream: %w", readErr)
		}

		// Optimize string trimming for common case (avoid allocation for empty lines)
		line = strings.TrimSpace(line)
		if line == "" {
			if isEOF {
				return nil, io.EOF
			}
			continue
		}

		// Parse SSE lines efficiently using string operations
		switch {
		case strings.HasPrefix(line, "event: "):
			// Extract event type efficiently
			currentEventType = line[7:] // Skip "event: "
			fiberlog.Debugf("[%s] Anthropic SSE event: %s", r.reqID, currentEventType)
			if isEOF {
				return nil, io.EOF
			}

		case strings.HasPrefix(line, "data: "):
			// Extract data efficiently
			data := line[6:] // Skip "data: "

			// Handle [DONE] marker (fast path for completion)
			if data == "[DONE]" {
				return &models.AnthropicMessageChunk{
					Type:     "message_stop",
					Provider: r.provider,
				}, nil
			}

			// Parse JSON chunk (reuse []byte to avoid string->[]byte conversion)
			dataBytes := []byte(data)
			var anthropicChunk anthropic.MessageStreamEventUnion
			if err := json.Unmarshal(dataBytes, &anthropicChunk); err != nil {
				fiberlog.Errorf("[%s] Failed to parse Anthropic streaming chunk: %v", r.reqID, err)
				if isEOF {
					return nil, io.EOF
				}
				continue
			}

			// Convert chunk with event type context
			adaptiveChunk, err := r.convertToAdaptiveAnthropicChunk(&anthropicChunk)
			if err != nil {
				fiberlog.Errorf("[%s] Failed to convert Anthropic chunk: %v", r.reqID, err)
				if isEOF {
					return nil, io.EOF
				}
				continue
			}

			// Reset event type after successful processing (avoid state leakage)
			return adaptiveChunk, nil

		default:
			// Skip other SSE metadata lines efficiently
			if strings.ContainsRune(line, ':') {
				if isEOF {
					return nil, io.EOF
				}
				continue
			}
		}

		// Handle EOF at end of processing
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
	var err error
	r.closeOnce.Do(func() {
		if r.closer != nil {
			err = r.closer.Close()
			r.closer = nil
		}
	})
	return err
}
