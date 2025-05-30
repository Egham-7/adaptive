package sse

import (
	"adaptive-backend/internal/services/stream_readers"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"

	"github.com/conneroisu/groq-go"
)

// EnhancedGroqResponse extends the Groq response with additional fields
type EnhancedGroqResponse struct {
	*groq.ChatCompletionStreamResponse
	Provider string `json:"provider"`
}

// GroqStreamReader implements a stream reader for Groq completions
type GroqStreamReader struct {
	stream_readers.BaseStreamReader
	stream *groq.ChatCompletionStream
	done   bool
}

// NewGroqStreamReader creates a new stream reader for Groq completions
func NewGroqStreamReader(stream *groq.ChatCompletionStream, RequestID string) *GroqStreamReader {
	return &GroqStreamReader{
		BaseStreamReader: stream_readers.BaseStreamReader{
			Buffer:    []byte{},
			RequestID: RequestID,
		},
		stream: stream,
		done:   false,
	}
}

// Read implements io.Reader interface for Groq streams
func (r *GroqStreamReader) Read(p []byte) (n int, err error) {
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

	// Get next chunk from Groq stream
	response, err := r.stream.Recv()
	if err != nil {
		// Handle end of stream or errors
		if strings.Contains(err.Error(), "EOF") || strings.Contains(err.Error(), "stream closed") {
			// Send [DONE] and mark as complete
			r.Buffer = []byte("data: [DONE]\n\n")
			r.done = true
			return r.Read(p) // Recursively call to handle the Buffer
		}
		// Format any other error as SSE and mark as done
		log.Printf("[%s] Error in Groq stream: %v", r.RequestID, err)
		safeErrMsg := strings.ReplaceAll(err.Error(), "\"", "\\\"")
		r.Buffer = fmt.Appendf(nil, "data: {\"error\": \"%s\"}\n\n", safeErrMsg)
		r.done = true
		return r.Read(p) // Recursively call to handle the Buffer
	}

	enhanced := EnhancedGroqResponse{
		ChatCompletionStreamResponse: response,
		Provider:                     "groq",
	}

	// Marshal the response to JSON
	jsonData, err := json.Marshal(enhanced)
	if err != nil {
		log.Printf("[%s] Error marshaling Groq response: %v", r.RequestID, err)
		r.Buffer = []byte("data: {\"error\": \"Failed to marshal response\"}\n\n")
		return r.Read(p)
	}

	// Format as SSE
	r.Buffer = fmt.Appendf(nil, "data: %s\n\n", jsonData)

	// Check if this is the last message
	if len(response.Choices) > 0 && response.Choices[0].FinishReason != "" {
		// This is the last content chunk, we'll send [DONE] on the next Read
		r.done = true
	}

	// Recursively call Read to handle the newly filled Buffer
	return r.Read(p)
}

// Close implements io.Closer interface
func (r *GroqStreamReader) Close() error {
	var err error
	r.CloseLock.Do(func() {
		err = r.stream.Close()
		log.Printf("[%s] Groq stream closed", r.RequestID)
	})
	return err
}
