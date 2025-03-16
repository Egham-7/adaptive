package stream_readers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// OpenAIStreamReader adapts OpenAI's streaming API to io.Reader
type OpenAIStreamReader struct {
	BaseStreamReader
	stream *openai.ChatCompletionStream
	done   bool
}

// NewOpenAIStreamReader creates a new stream reader for OpenAI completions
func NewOpenAIStreamReader(stream *openai.ChatCompletionStream, requestID string) *OpenAIStreamReader {
	return &OpenAIStreamReader{
		BaseStreamReader: BaseStreamReader{
			buffer:    []byte{},
			requestID: requestID,
		},
		stream: stream,
		done:   false,
	}
}

// Read implements io.Reader interface for OpenAI streams
func (r *OpenAIStreamReader) Read(p []byte) (n int, err error) {
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

	// Get next chunk from OpenAI stream
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
		log.Printf("[%s] Error in stream: %v", r.requestID, err)
		safeErrMsg := strings.ReplaceAll(err.Error(), "\"", "\\\"")
		r.buffer = fmt.Appendf(nil, "data: {\"error\": \"%s\"}\n\n", safeErrMsg)
		r.done = true
		return r.Read(p) // Recursively call to handle the buffer
	}

	type enhancedResponse struct {
		openai.ChatCompletionStreamResponse
		Provider string `json:"provider"`
	}

	enhanced := enhancedResponse{
		ChatCompletionStreamResponse: response,
		Provider:                     "openai",
	}

	// Marshal the response to JSON
	jsonData, err := json.Marshal(enhanced)
	if err != nil {
		log.Printf("[%s] Error marshaling response: %v", r.requestID, err)
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
func (r *OpenAIStreamReader) Close() error {
	var err error
	r.closeLock.Do(func() {
		err = r.stream.Close()
		log.Printf("[%s] OpenAI stream closed", r.requestID)
	})
	return err
}
