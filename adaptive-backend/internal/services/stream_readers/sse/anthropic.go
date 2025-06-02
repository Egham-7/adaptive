package sse

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"
	"time"

	"adaptive-backend/internal/services/stream_readers"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/openai/openai-go"
)

// AnthropicStreamReader implements a stream reader for Anthropic completions
type AnthropicStreamReader struct {
	stream_readers.BaseStreamReader
	stream ssestream.Stream[anthropic.MessageStreamEventUnion]
	done   bool
}

// NewAnthropicStreamReader creates a new stream reader for Anthropic completions
func NewAnthropicStreamReader(stream ssestream.Stream[anthropic.MessageStreamEventUnion], RequestID string) *AnthropicStreamReader {
	return &AnthropicStreamReader{
		BaseStreamReader: stream_readers.BaseStreamReader{
			Buffer:    []byte{},
			RequestID: RequestID,
		},
		stream: stream,
		done:   false,
	}
}

// Read implements io.Reader interface for Anthropic streams with enhanced error handling
func (r *AnthropicStreamReader) Read(p []byte) (n int, err error) {
	// If we already have data in Buffer, send that first
	if len(r.Buffer) > 0 {
		n = copy(p, r.Buffer)
		r.Buffer = r.Buffer[n:]
		return n, nil
	}

	// If we're done streaming, return EOF
	if r.done {
		return 0, io.EOF
	}

	// Get next event from Anthropic stream
	for {
		if !r.stream.Next() {
			// Check if there was an error
			if r.stream.Err() != nil {
				// Handle different error types appropriately
				errMsg := r.stream.Err().Error()
				if strings.Contains(errMsg, "EOF") || strings.Contains(errMsg, "stream closed") {
					// Normal end of stream
					r.Buffer = []byte("data: [DONE]\n\n")
					r.done = true
					return r.Read(p)
				}
				
				// Actual error occurred
				log.Printf("[%s] Error in Anthropic stream: %v", r.RequestID, r.stream.Err())
				safeErrMsg := strings.ReplaceAll(errMsg, "\"", "\\\"")
				safeErrMsg = strings.ReplaceAll(safeErrMsg, "\n", "\\n")
				r.Buffer = fmt.Appendf(nil, "data: {\"error\": \"%s\"}\n\n", safeErrMsg)
				r.done = true
				return r.Read(p)
			}

			// No error but no more events, send [DONE]
			r.Buffer = []byte("data: [DONE]\n\n")
			r.done = true
			return r.Read(p)
		}

		// Get the current event
		event := r.stream.Current()

		// Convert Anthropic event to OpenAI format
		openaiChunk := r.convertToOpenAIFormat(event)

		// Check if this is the last message (MessageStopEvent)
		if _, isStop := event.AsAny().(anthropic.MessageStopEvent); isStop {
			// This is the last event, we'll send [DONE] on the next Read
			r.done = true
		}

		// Only send response if we have content or finish reason
		if openaiChunk != nil {
			// Marshal the response to JSON
			jsonData, err := json.Marshal(openaiChunk)
			if err != nil {
				log.Printf("[%s] Error marshaling OpenAI-compatible response: %v", r.RequestID, err)
				r.Buffer = []byte("data: {\"error\": \"Failed to marshal response\"}\n\n")
				return r.Read(p)
			}

			// Format as SSE
			r.Buffer = fmt.Appendf(nil, "data: %s\n\n", jsonData)
			break
		}

		// If we didn't get content, continue to next event
		// This handles events like ContentBlockStartEvent that don't produce output
		if r.done {
			break
		}
	}

	// Recursively call Read to handle the newly filled Buffer
	return r.Read(p)
}

// convertToOpenAIFormat converts Anthropic events to OpenAI format with comprehensive event handling
func (r *AnthropicStreamReader) convertToOpenAIFormat(event anthropic.MessageStreamEventUnion) *openai.ChatCompletionChunk {
	baseChunk := &openai.ChatCompletionChunk{
		ID:      "chatcmpl-" + r.RequestID,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   "claude-3-5-sonnet",
	}

	switch val := event.AsAny().(type) {
	case anthropic.MessageStartEvent:
		// Start of message - send role
		baseChunk.Choices = []openai.ChatCompletionChunkChoice{
			{
				Index: 0,
				Delta: openai.ChatCompletionChunkChoiceDelta{
					Role: "assistant",
				},
			},
		}
		return baseChunk

	case anthropic.ContentBlockStartEvent:
		// Start of content block - no content to send yet
		return nil

	case anthropic.ContentBlockDeltaEvent:
		// Content delta - send text content
		switch delta := val.Delta.AsAny().(type) {
		case anthropic.TextDelta:
			baseChunk.Choices = []openai.ChatCompletionChunkChoice{
				{
					Index: 0,
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Content: delta.Text,
					},
				},
			}
			return baseChunk
		}

	case anthropic.ContentBlockStopEvent:
		// End of content block - no additional content
		return nil

	case anthropic.MessageDeltaEvent:
		// Message-level deltas (usage, stop reason, etc.)
		// Handle usage information if available
		return nil

	case anthropic.MessageStopEvent:
		// End of message - send finish reason
		baseChunk.Choices = []openai.ChatCompletionChunkChoice{
			{
				Index:        0,
				Delta:        openai.ChatCompletionChunkChoiceDelta{},
				FinishReason: "stop",
			},
		}
		return baseChunk

	default:
		// Handle unknown event types gracefully
		log.Printf("[%s] Unknown Anthropic event type: %T", r.RequestID, val)
		return nil
	}

	return nil
}

// Close implements io.Closer interface with proper resource cleanup
func (r *AnthropicStreamReader) Close() error {
	var err error
	r.CloseLock.Do(func() {
		// Mark as done to prevent further reads
		r.done = true
		
		// Clear buffer to free memory
		r.Buffer = nil
		
		// Anthropic streams don't have an explicit Close method,
		// but we ensure the stream is marked as finished
		log.Printf("[%s] Anthropic stream closed", r.RequestID)
	})
	return err
}
