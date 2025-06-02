package sse

import (
	"adaptive-backend/internal/services/stream_readers"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"

	"github.com/cohesion-org/deepseek-go"
	"github.com/openai/openai-go"
)

// DeepSeekStreamReader implements a stream reader for DeepSeek completions
type DeepSeekStreamReader struct {
	stream_readers.BaseStreamReader
	stream deepseek.ChatCompletionStream
	done   bool
}

// NewDeepSeekStreamReader creates a new stream reader for DeepSeek completions
func NewDeepSeekStreamReader(stream deepseek.ChatCompletionStream, RequestID string) *DeepSeekStreamReader {
	return &DeepSeekStreamReader{
		BaseStreamReader: stream_readers.BaseStreamReader{
			Buffer:    []byte{},
			RequestID: RequestID,
		},
		stream: stream,
		done:   false,
	}
}

// Read implements io.Reader interface for DeepSeek streams with enhanced reasoning support
func (r *DeepSeekStreamReader) Read(p []byte) (n int, err error) {
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

	// Get next chunk from DeepSeek stream with enhanced error handling
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
			
			// Handle actual errors with better formatting
			log.Printf("[%s] Error in DeepSeek stream: %v", r.RequestID, err)
			safeErrMsg := strings.ReplaceAll(err.Error(), "\"", "\\\"")
			safeErrMsg = strings.ReplaceAll(safeErrMsg, "\n", "\\n")
			r.Buffer = fmt.Appendf(nil, "data: {\"error\": \"%s\"}\n\n", safeErrMsg)
			r.done = true
			return r.Read(p)
		}

		// Validate response structure
		if response == nil {
			log.Printf("[%s] Received nil response from DeepSeek stream", r.RequestID)
			continue
		}

		// Convert DeepSeek response to OpenAI format
		openaiChunk := r.convertToOpenAIFormat(response)
		if openaiChunk == nil {
			log.Printf("[%s] Failed to convert DeepSeek response to OpenAI format", r.RequestID)
			continue
		}

		// Marshal the response to JSON
		jsonData, err := json.Marshal(openaiChunk)
		if err != nil {
			log.Printf("[%s] Error marshaling DeepSeek response: %v", r.RequestID, err)
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

// convertToOpenAIFormat converts DeepSeek response to OpenAI format with reasoning support
func (r *DeepSeekStreamReader) convertToOpenAIFormat(deepseekResponse *deepseek.StreamChatCompletionResponse) *openai.ChatCompletionChunk {
	// Validate input
	if deepseekResponse == nil {
		log.Printf("[%s] Received nil DeepSeek response", r.RequestID)
		return nil
	}

	// Prepare base chunk with required fields
	chunk := &openai.ChatCompletionChunk{
		ID:      deepseekResponse.ID,
		Object:  "chat.completion.chunk",
		Created: deepseekResponse.Created,
		Model:   deepseekResponse.Model,
	}

	// Process choices with enhanced validation and reasoning support
	choices := make([]openai.ChatCompletionChunkChoice, 0, len(deepseekResponse.Choices))
	
	for i, deepseekChoice := range deepseekResponse.Choices {
		choice := openai.ChatCompletionChunkChoice{
			Index: int64(i),
			Delta: openai.ChatCompletionChunkChoiceDelta{},
		}

		// Handle content with reasoning support for R1 models
		if deepseekChoice.Delta.Content != "" {
			content := deepseekChoice.Delta.Content
			
			// For DeepSeek R1 models, handle reasoning content specially
			if r.isReasoningModel(deepseekResponse.Model) {
				content = r.processReasoningContent(content)
			}
			
			choice.Delta.Content = content
		}

		// Handle reasoning content if present (DeepSeek R1 specific)
		if deepseekChoice.Delta.ReasoningContent != "" {
			// For reasoning models, we might want to include thinking process
			// This is DeepSeek R1 specific functionality
			log.Printf("[%s] Reasoning content detected: %d chars", r.RequestID, len(deepseekChoice.Delta.ReasoningContent))
		}
		
		// Map finish reason with comprehensive handling
		if deepseekChoice.FinishReason != "" {
			choice.FinishReason = r.mapFinishReason(deepseekChoice.FinishReason)
		}
		
		// Map role with validation
		if deepseekChoice.Delta.Role != "" {
			choice.Delta.Role = r.mapRole(deepseekChoice.Delta.Role)
		}

		// Note: DeepSeek StreamDelta doesn't have ToolCalls field
		// Tool calls would be handled differently if supported
		
		choices = append(choices, choice)
	}

	chunk.Choices = choices
	return chunk
}

// isReasoningModel checks if the model is a reasoning model (like DeepSeek R1)
func (r *DeepSeekStreamReader) isReasoningModel(model string) bool {
	reasoningModels := []string{
		"deepseek-reasoner",
		"deepseek-r1",
		"deepseek-r1-distill-llama-70b",
		"deepseek-r1-distill-qwen-32b",
		"deepseek-r1-distill-qwen-14b",
		"deepseek-r1-distill-qwen-7b",
		"deepseek-r1-distill-qwen-1.5b",
	}
	
	for _, rm := range reasoningModels {
		if strings.Contains(strings.ToLower(model), rm) {
			return true
		}
	}
	return false
}

// processReasoningContent handles reasoning model content processing
func (r *DeepSeekStreamReader) processReasoningContent(content string) string {
	// For reasoning models, we might want to filter or process thinking tags
	// This could include removing <think></think> tags if desired
	// For now, we'll pass through all content to maintain transparency
	return content
}

// mapFinishReason maps DeepSeek finish reasons to OpenAI format
func (r *DeepSeekStreamReader) mapFinishReason(reason string) string {
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
		log.Printf("[%s] Unknown DeepSeek finish reason: %s, defaulting to 'stop'", r.RequestID, reason)
		return "stop"
	}
}

// mapRole maps DeepSeek roles to OpenAI format
func (r *DeepSeekStreamReader) mapRole(role string) string {
	switch role {
	case "assistant", "user", "system", "tool", "function":
		return role
	default:
		log.Printf("[%s] Unknown DeepSeek role: %s, defaulting to 'assistant'", r.RequestID, role)
		return "assistant"
	}
}

// Close implements io.Closer interface with proper resource cleanup
func (r *DeepSeekStreamReader) Close() error {
	var err error
	r.CloseLock.Do(func() {
		// Mark as done to prevent further reads
		r.done = true
		
		// Clear buffer to free memory
		r.Buffer = nil
		
		// Close the underlying DeepSeek stream
		if r.stream != nil {
			err = r.stream.Close()
		}
		
		log.Printf("[%s] DeepSeek stream closed", r.RequestID)
	})
	return err
}