package stream_readers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
)

// EnhancedAnthropicResponse extends the Anthropic event with additional fields
type EnhancedAnthropicResponse struct {
	anthropic.MessageStreamEventUnion
	Provider string `json:"provider"`
}

// AnthropicStreamReader implements a stream reader for Anthropic completions
type AnthropicStreamReader struct {
	BaseStreamReader
	stream ssestream.Stream[anthropic.MessageStreamEventUnion]
	done   bool
}

// NewAnthropicStreamReader creates a new stream reader for Anthropic completions
func NewAnthropicStreamReader(stream ssestream.Stream[anthropic.MessageStreamEventUnion], requestID string) *AnthropicStreamReader {
	return &AnthropicStreamReader{
		BaseStreamReader: BaseStreamReader{
			buffer:    []byte{},
			requestID: requestID,
		},
		stream: stream,
		done:   false,
	}
}

// Read implements io.Reader interface for Anthropic streams
func (r *AnthropicStreamReader) Read(p []byte) (n int, err error) {
	// If we already have data in buffer, send that first
	if len(r.buffer) > 0 {
		n = copy(p, r.buffer)
		r.buffer = r.buffer[n:]
		return n, nil
	}

	// If we're done streaming, return EOF
	if r.done {
		return 0, io.EOF
	}

	// Get next event from Anthropic stream
	if !r.stream.Next() {
		// Check if there was an error
		if r.stream.Err() != nil {
			// Format any error as SSE and mark as done
			log.Printf("[%s] Error in Anthropic stream: %v", r.requestID, r.stream.Err())
			safeErrMsg := strings.ReplaceAll(r.stream.Err().Error(), "\"", "\\\"")
			r.buffer = fmt.Appendf(nil, "data: {\"error\": \"%s\"}\n\n", safeErrMsg)
			r.done = true
			return r.Read(p) // Recursively call to handle the buffer
		}

		// No error but no more events, send [DONE]
		r.buffer = []byte("data: [DONE]\n\n")
		r.done = true
		return r.Read(p) // Recursively call to handle the buffer
	}

	// Get the current event
	event := r.stream.Current()

	// Create enhanced response with provider field
	enhanced := EnhancedAnthropicResponse{
		MessageStreamEventUnion: event,
		Provider:                "anthropic",
	}

	// Marshal the response to JSON
	jsonData, err := json.Marshal(enhanced)
	if err != nil {
		log.Printf("[%s] Error marshaling Anthropic response: %v", r.requestID, err)
		r.buffer = []byte("data: {\"error\": \"Failed to marshal response\"}\n\n")
		return r.Read(p)
	}

	// Format as SSE
	r.buffer = fmt.Appendf(nil, "data: %s\n\n", jsonData)

	// Check if this is the last message (MessageStopEvent)
	if _, isStop := event.AsAny().(anthropic.MessageStopEvent); isStop {
		// This is the last event, we'll send [DONE] on the next Read
		r.done = true
	}

	// Recursively call Read to handle the newly filled buffer
	return r.Read(p)
}

// Close implements io.Closer interface
func (r *AnthropicStreamReader) Close() error {
	var err error
	r.closeLock.Do(func() {
		// Anthropic streams don't have an explicit Close method,
		// but we can mark it as done to prevent further reads
		r.done = true
		log.Printf("[%s] Anthropic stream closed", r.requestID)
	})
	return err
}
