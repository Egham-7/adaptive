package stream_readers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"

	"github.com/conneroisu/groq-go"
)

type GroqStreamReader struct {
	BaseStreamReader
	stream *groq.ChatCompletionStream
	done   bool
}

// NewGroqStreamReader creates a new stream reader for Groq completions
func NewGroqStreamReader(stream *groq.ChatCompletionStream, requestID string) *GroqStreamReader {
	return &GroqStreamReader{
		BaseStreamReader: BaseStreamReader{
			buffer:    []byte{},
			requestID: requestID,
		},
		stream: stream,
		done:   false,
	}
}

// Read implements io.Reader interface for Groq streams
func (r *GroqStreamReader) Read(p []byte) (n int, err error) {
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

	// Get next chunk from Groq stream
	response, err := r.stream.Recv()
	if err != nil {
		// Handle end of stream or errors
		if strings.Contains(err.Error(), "EOF") || strings.Contains(err.Error(), "stream closed") {
			// Send [DONE] and mark as complete
			r.buffer = []byte("data: [DONE]\n\n")
			r.done = true
			return r.Read(p) // Recursively call to handle the buffer
		}

		// Format any other error as SSE and mark as done
		log.Printf("[%s] Error in Groq stream: %v", r.requestID, err)
		safeErrMsg := strings.ReplaceAll(err.Error(), "\"", "\\\"")
		r.buffer = fmt.Appendf(nil, "data: {\"error\": \"%s\"}\n\n", safeErrMsg)
		r.done = true
		return r.Read(p) // Recursively call to handle the buffer
	}

	type enhancedResponse struct {
		*groq.ChatCompletionStreamResponse
		Provider string `json:"provider"`
	}

	enhanced := enhancedResponse{
		ChatCompletionStreamResponse: response,
		Provider:                     "groq",
	}

	// Marshal the response to JSON
	jsonData, err := json.Marshal(enhanced)
	if err != nil {
		log.Printf("[%s] Error marshaling Groq response: %v", r.requestID, err)
		r.buffer = []byte("data: {\"error\": \"Failed to marshal response\"}\n\n")
		return r.Read(p)
	}

	// Format as SSE
	r.buffer = fmt.Appendf(nil, "data: %s\n\n", jsonData)

	// Check if this is the last message
	if len(response.Choices) > 0 && response.Choices[0].FinishReason != "" {
		// This is the last content chunk, we'll send [DONE] on the next Read
		r.done = true
	}

	// Recursively call Read to handle the newly filled buffer
	return r.Read(p)
}

// Close implements io.Closer interface
func (r *GroqStreamReader) Close() error {
	var err error
	r.closeLock.Do(func() {
		err = r.stream.Close()
		log.Printf("[%s] Groq stream closed", r.requestID)
	})
	return err
}
