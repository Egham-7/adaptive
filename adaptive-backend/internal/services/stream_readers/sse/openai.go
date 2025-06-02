package sse

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"

	"adaptive-backend/internal/services/stream_readers"

	"github.com/openai/openai-go"
	ssestream "github.com/openai/openai-go/packages/ssestream"
)

// OpenAIStreamReader adapts OpenAI's streaming API to io.Reader
type OpenAIStreamReader struct {
	stream_readers.BaseStreamReader
	stream *ssestream.Stream[openai.ChatCompletionChunk]
	done   bool
}

// NewOpenAIStreamReader creates a new stream reader for OpenAI completions
func NewOpenAIStreamReader(stream *ssestream.Stream[openai.ChatCompletionChunk], RequestID string) *OpenAIStreamReader {
	return &OpenAIStreamReader{
		BaseStreamReader: stream_readers.BaseStreamReader{
			Buffer:    []byte{},
			RequestID: RequestID,
		},
		stream: stream,
		done:   false,
	}
}

// Read implements io.Reader interface for OpenAI streams with enhanced error handling
func (r *OpenAIStreamReader) Read(p []byte) (n int, err error) {
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

	// Get next chunk from OpenAI stream with enhanced error handling
	ok := r.stream.Next()
	if !ok {
		// Handle different error types appropriately
		if r.stream.Err() == nil || strings.Contains(r.stream.Err().Error(), "EOF") {
			// Normal end of stream
			r.Buffer = []byte("data: [DONE]\n\n")
			r.done = true
			return r.Read(p)
		}

		// Actual error occurred
		log.Printf("[%s] Error in OpenAI stream: %v", r.RequestID, r.stream.Err())
		safeErrMsg := strings.ReplaceAll(r.stream.Err().Error(), "\"", "\\\"")
		safeErrMsg = strings.ReplaceAll(safeErrMsg, "\n", "\\n")
		r.Buffer = fmt.Appendf(nil, "data: {\"error\": \"%s\"}\n\n", safeErrMsg)
		r.done = true
		return r.Read(p)
	}

	// Get the chunk directly from OpenAI (already in correct format)
	chunk := r.stream.Current()

	// Marshal the response to JSON
	jsonData, err := json.Marshal(chunk)
	if err != nil {
		log.Printf("[%s] Error marshaling OpenAI response: %v", r.RequestID, err)
		r.Buffer = []byte("data: {\"error\": \"Failed to marshal response\"}\n\n")
		return r.Read(p)
	}

	// Format as SSE
	r.Buffer = fmt.Appendf(nil, "data: %s\n\n", jsonData)

	// Only set done if ALL choices are finished with improved validation
	if len(chunk.Choices) > 0 {
		allDone := true
		for _, choice := range chunk.Choices {
			if choice.FinishReason == "" {
				allDone = false
				break
			}
		}
		if allDone {
			r.done = true
		}
	}

	// Recursively call Read to handle the newly filled Buffer
	return r.Read(p)
}

// Close implements io.Closer interface with proper resource cleanup
func (r *OpenAIStreamReader) Close() error {
	var err error
	r.CloseLock.Do(func() {
		// Mark as done to prevent further reads
		r.done = true

		// Clear buffer to free memory
		r.Buffer = nil

		// Close the underlying OpenAI stream
		if r.stream != nil {
			err = r.stream.Close()
		}

		log.Printf("[%s] OpenAI stream closed", r.RequestID)
	})
	return err
}
