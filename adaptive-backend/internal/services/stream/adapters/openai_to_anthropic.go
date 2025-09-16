package adapters

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"adaptive-backend/internal/services/format_adapter"

	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/ssestream"
)

// OpenAIToAnthropicStreamAdapter converts OpenAI streaming response to Anthropic SSE format
type OpenAIToAnthropicStreamAdapter struct {
	openaiStream *ssestream.Stream[openai.ChatCompletionChunk]
	provider     string
	requestID    string
	buffer       strings.Builder
	done         bool
}

// NewOpenAIToAnthropicStreamAdapter creates a stream adapter that converts OpenAI format to Anthropic SSE
func NewOpenAIToAnthropicStreamAdapter(
	openaiStream *ssestream.Stream[openai.ChatCompletionChunk],
	provider string,
	requestID string,
) io.Reader {
	return &OpenAIToAnthropicStreamAdapter{
		openaiStream: openaiStream,
		provider:     provider,
		requestID:    requestID,
	}
}

// Read implements io.Reader interface, converting OpenAI chunks to Anthropic SSE format
func (a *OpenAIToAnthropicStreamAdapter) Read(p []byte) (n int, err error) {
	// If we have buffered data, return it first
	if a.buffer.Len() > 0 {
		return a.readFromBuffer(p)
	}

	// If we're done, return EOF
	if a.done {
		return 0, io.EOF
	}

	// Loop to process chunks until we have valid data or reach end
	for {
		// Get the next chunk from OpenAI stream
		if !a.openaiStream.Next() {
			a.done = true
			// Check for errors
			if err := a.openaiStream.Err(); err != nil {
				return 0, err
			}
			// Write final SSE terminator
			a.buffer.WriteString("data: [DONE]\n\n")
			return a.readFromBuffer(p)
		}

		openaiChunk := a.openaiStream.Current()

		// Convert OpenAI chunk to Anthropic format
		anthropicEvent, err := format_adapter.OpenAIToAnthropic.ConvertStreamingChunk(&openaiChunk, a.provider)
		if err != nil {
			fiberlog.Errorf("[%s] Failed to convert OpenAI chunk to Anthropic format: %v", a.requestID, err)
			// Continue to next iteration to skip this bad chunk
			continue
		}

		// Marshal the Anthropic event to JSON
		anthropicJSON, err := json.Marshal(anthropicEvent)
		if err != nil {
			fiberlog.Errorf("[%s] Failed to marshal Anthropic event: %v", a.requestID, err)
			// Continue to next iteration to skip this bad chunk
			continue
		}

		// Format as SSE data and return
		a.buffer.WriteString(fmt.Sprintf("event: %s\ndata: %s\n\n", anthropicEvent.Type, string(anthropicJSON)))
		return a.readFromBuffer(p)
	}
}

// readFromBuffer reads data from the internal buffer
func (a *OpenAIToAnthropicStreamAdapter) readFromBuffer(p []byte) (int, error) {
	bufferData := a.buffer.String()
	if len(bufferData) == 0 {
		return 0, nil
	}

	// Copy what we can to the output buffer
	n := copy(p, bufferData)

	// Remove what we copied from the buffer
	if n >= len(bufferData) {
		a.buffer.Reset()
	} else {
		remaining := bufferData[n:]
		a.buffer.Reset()
		a.buffer.WriteString(remaining)
	}

	return n, nil
}
