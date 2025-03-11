package services

import (
	"adaptive-backend/internal/models"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"
	"sync"

	deepseek "github.com/cohesion-org/deepseek-go"
	"github.com/conneroisu/groq-go"
	"github.com/sashabaranov/go-openai"
)

// StreamReader is the common interface for all LLM stream readers
type StreamReader interface {
	io.Reader
	io.Closer
}

// GetStreamReader creates the appropriate stream reader based on provider type
func GetStreamReader(resp *models.ChatCompletionResponse, provider string, requestID string) (StreamReader, error) {
	switch provider {
	case "openai":
		stream, ok := resp.Response.(*openai.ChatCompletionStream)
		if !ok {
			return nil, fmt.Errorf("invalid OpenAI stream type")
		}
		return NewOpenAIStreamReader(stream, requestID), nil
	case "groq":
		stream, ok := resp.Response.(*groq.ChatCompletionStream)
		if !ok {
			return nil, fmt.Errorf("invalid Groq stream type")
		}
		return NewGroqStreamReader(stream, requestID), nil
	case "deepseek":
		stream, ok := resp.Response.(deepseek.ChatCompletionStream)
		if !ok {
			return nil, fmt.Errorf("invalid DeepSeek stream type")
		}
		return NewDeepSeekStreamReader(stream, requestID), nil
	default:
		return nil, fmt.Errorf("unsupported provider for streaming: %s", provider)
	}
}

// BaseStreamReader provides common functionality for all stream readers
type BaseStreamReader struct {
	buffer    []byte
	requestID string
	closeLock sync.Once
}

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

	// Marshal the response to JSON
	jsonData, err := json.Marshal(response)
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

// GroqStreamReader adapts Groq's streaming API to io.Reader
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

	// Marshal the response to JSON
	jsonData, err := json.Marshal(response)
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

// DeepSeekStreamReader adapts DeepSeek's streaming API to io.Reader
type DeepSeekStreamReader struct {
	BaseStreamReader
	stream deepseek.ChatCompletionStream
	done   bool
}

// NewDeepSeekStreamReader creates a new stream reader for DeepSeek completions
func NewDeepSeekStreamReader(stream deepseek.ChatCompletionStream, requestID string) *DeepSeekStreamReader {
	return &DeepSeekStreamReader{
		BaseStreamReader: BaseStreamReader{
			buffer:    []byte{},
			requestID: requestID,
		},
		stream: stream,
		done:   false,
	}
}

// Read implements io.Reader interface for DeepSeek streams
func (r *DeepSeekStreamReader) Read(p []byte) (n int, err error) {
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

	// Get next chunk from DeepSeek stream
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
		log.Printf("[%s] Error in DeepSeek stream: %v", r.requestID, err)
		safeErrMsg := strings.ReplaceAll(err.Error(), "\"", "\\\"")
		r.buffer = fmt.Appendf(nil, "data: {\"error\": \"%s\"}\n\n", safeErrMsg)
		r.done = true
		return r.Read(p) // Recursively call to handle the buffer
	}

	// Marshal the response to JSON
	jsonData, err := json.Marshal(response)
	if err != nil {
		log.Printf("[%s] Error marshaling DeepSeek response: %v", r.requestID, err)
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
func (r *DeepSeekStreamReader) Close() error {
	var err error
	r.closeLock.Do(func() {
		err = r.stream.Close()
		log.Printf("[%s] DeepSeek stream closed", r.requestID)
	})
	return err
}
