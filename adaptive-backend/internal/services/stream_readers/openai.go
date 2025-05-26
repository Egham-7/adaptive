package stream_readers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"

	"github.com/openai/openai-go"
	ssestream "github.com/openai/openai-go/packages/ssestream"
)

// EnhancedOpenAIResponse extends the OpenAI response with additional fields
type EnhancedOpenAIResponse struct {
	openai.ChatCompletionChunk
	Provider string `json:"provider"`
}

// OpenAIStreamReader adapts OpenAI's streaming API to io.Reader
type OpenAIStreamReader struct {
	BaseStreamReader
	stream *ssestream.Stream[openai.ChatCompletionChunk]
	done   bool
}

// NewOpenAIStreamReader creates a new stream reader for OpenAI completions
func NewOpenAIStreamReader(stream *ssestream.Stream[openai.ChatCompletionChunk], requestID string) *OpenAIStreamReader {
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
	ok := r.stream.Next()
	if !ok {
		// End of stream or error
		if r.stream.Err() == nil || strings.Contains(r.stream.Err().Error(), "EOF") {
			r.buffer = []byte("data: [DONE]\n\n")
			r.done = true
			return r.Read(p)
		}
		log.Printf("[%s] Error in stream: %v", r.requestID, r.stream.Err())
		safeErrMsg := strings.ReplaceAll(r.stream.Err().Error(), "\"", "\\\"")
		r.buffer = fmt.Appendf(nil, "data: {\"error\": \"%s\"}\n\n", safeErrMsg)
		r.done = true
		return r.Read(p)
	}

	// Compose enhanced response with provider
	chunk := r.stream.Current()
	enhanced := EnhancedOpenAIResponse{
		ChatCompletionChunk: chunk,
		Provider:            "openai",
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

	// Only set done if ALL choices are finished
	allDone := true
	for _, choice := range chunk.Choices {
		if choice.FinishReason == "" {
			allDone = false
			break
		}
	}
	r.done = allDone
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
