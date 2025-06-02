package sse

import (
	"adaptive-backend/internal/services/stream_readers"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"

	"github.com/conneroisu/groq-go"
	"github.com/openai/openai-go"
)

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

// Read implements io.Reader interface for Groq streams with enhanced error handling
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

	// Get next chunk from Groq stream with retry logic
	for {
		response, err := r.stream.Recv()
		if err != nil {
			// Handle different error types appropriately
			if err == io.EOF || strings.Contains(err.Error(), "EOF") || strings.Contains(err.Error(), "stream closed") {
				// Normal end of stream
				r.Buffer = []byte("data: [DONE]\n\n")
				r.done = true
				return r.Read(p)
			}
			
			// Handle actual errors
			log.Printf("[%s] Error in Groq stream: %v", r.RequestID, err)
			safeErrMsg := strings.ReplaceAll(err.Error(), "\"", "\\\"")
			safeErrMsg = strings.ReplaceAll(safeErrMsg, "\n", "\\n")
			r.Buffer = fmt.Appendf(nil, "data: {\"error\": \"%s\"}\n\n", safeErrMsg)
			r.done = true
			return r.Read(p)
		}

		// Validate response structure
		if response == nil {
			log.Printf("[%s] Received nil response from Groq stream", r.RequestID)
			continue
		}

		// Convert Groq response to OpenAI format
		openaiChunk := r.convertToOpenAIFormat(response)
		if openaiChunk == nil {
			log.Printf("[%s] Failed to convert Groq response to OpenAI format", r.RequestID)
			continue
		}

		// Marshal the response to JSON
		jsonData, err := json.Marshal(openaiChunk)
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

		break
	}

	// Recursively call Read to handle the newly filled Buffer
	return r.Read(p)
}

// convertToOpenAIFormat converts Groq response to OpenAI format with type safety and validation
func (r *GroqStreamReader) convertToOpenAIFormat(groqResponse *groq.ChatCompletionStreamResponse) *openai.ChatCompletionChunk {
	// Validate input
	if groqResponse == nil {
		log.Printf("[%s] Received nil Groq response", r.RequestID)
		return nil
	}

	// Prepare base chunk with required fields
	chunk := &openai.ChatCompletionChunk{
		ID:      groqResponse.ID,
		Object:  "chat.completion.chunk",
		Created: groqResponse.Created,
		Model:   string(groqResponse.Model), // Safe conversion from ChatModel to string
	}

	// Add system fingerprint if available
	if groqResponse.SystemFingerprint != "" {
		chunk.SystemFingerprint = groqResponse.SystemFingerprint
	}

	// Process choices with validation
	choices := make([]openai.ChatCompletionChunkChoice, 0, len(groqResponse.Choices))
	
	for i, groqChoice := range groqResponse.Choices {
		choice := openai.ChatCompletionChunkChoice{
			Index: int64(i),
			Delta: openai.ChatCompletionChunkChoiceDelta{},
		}

		// Set content if available
		if groqChoice.Delta.Content != "" {
			choice.Delta.Content = groqChoice.Delta.Content
		}
		
		// Map finish reason with comprehensive handling
		if groqChoice.FinishReason != "" {
			choice.FinishReason = r.mapFinishReason(string(groqChoice.FinishReason))
		}
		
		// Map role with validation
		if groqChoice.Delta.Role != "" {
			choice.Delta.Role = r.mapRole(groqChoice.Delta.Role)
		}

		// Note: Tool calls handling would be implemented here if needed
		// Groq SDK may have different tool call structure
		
		choices = append(choices, choice)
	}

	chunk.Choices = choices
	return chunk
}

// mapFinishReason maps Groq finish reasons to OpenAI format
func (r *GroqStreamReader) mapFinishReason(reason string) string {
	switch reason {
	case "stop":
		return "stop"
	case "length":
		return "length"
	case "content_filter":
		return "content_filter"
	case "tool_calls":
		return "tool_calls"
	case "function_call":
		return "function_call"
	default:
		log.Printf("[%s] Unknown Groq finish reason: %s, defaulting to 'stop'", r.RequestID, reason)
		return "stop"
	}
}

// mapRole maps Groq roles to OpenAI format
func (r *GroqStreamReader) mapRole(role string) string {
	switch role {
	case "assistant", "user", "system", "tool", "function":
		return role
	default:
		log.Printf("[%s] Unknown Groq role: %s, defaulting to 'assistant'", r.RequestID, role)
		return "assistant"
	}
}

// Close implements io.Closer interface with proper resource cleanup
func (r *GroqStreamReader) Close() error {
	var err error
	r.CloseLock.Do(func() {
		// Mark as done to prevent further reads
		r.done = true
		
		// Clear buffer to free memory
		r.Buffer = nil
		
		// Close the underlying Groq stream
		if r.stream != nil {
			err = r.stream.Close()
		}
		
		log.Printf("[%s] Groq stream closed", r.RequestID)
	})
	return err
}