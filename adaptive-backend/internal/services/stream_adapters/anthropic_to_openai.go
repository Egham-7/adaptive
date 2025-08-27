package stream_adapters

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"adaptive-backend/internal/services/format_adapter"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// AnthropicToOpenAIStreamAdapter converts Anthropic streaming response to OpenAI SSE format
type AnthropicToOpenAIStreamAdapter struct {
	anthropicStream *ssestream.Stream[anthropic.MessageStreamEventUnion]
	provider        string
	requestID       string
	buffer          strings.Builder
	done            bool
}

// NewAnthropicToOpenAIStreamAdapter creates a stream adapter that converts Anthropic format to OpenAI SSE
func NewAnthropicToOpenAIStreamAdapter(
	anthropicStream *ssestream.Stream[anthropic.MessageStreamEventUnion],
	provider string,
	requestID string,
) io.Reader {
	return &AnthropicToOpenAIStreamAdapter{
		anthropicStream: anthropicStream,
		provider:        provider,
		requestID:       requestID,
	}
}

// Read implements io.Reader interface, converting Anthropic events to OpenAI SSE format
func (a *AnthropicToOpenAIStreamAdapter) Read(p []byte) (n int, err error) {
	// If we have buffered data, return it first
	if a.buffer.Len() > 0 {
		return a.readFromBuffer(p)
	}

	// If we're done, return EOF
	if a.done {
		return 0, io.EOF
	}

	// Get the next event from Anthropic stream
	if !a.anthropicStream.Next() {
		a.done = true
		// Check for errors
		if err := a.anthropicStream.Err(); err != nil {
			return 0, err
		}
		// Write final SSE terminator
		a.buffer.WriteString("data: [DONE]\n\n")
		return a.readFromBuffer(p)
	}

	anthropicEvent := a.anthropicStream.Current()

	// Convert Anthropic event to OpenAI format
	openaiChunk, err := format_adapter.AnthropicToOpenAI.ConvertStreamingChunk(&anthropicEvent, a.provider)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to convert Anthropic event to OpenAI format: %v", a.requestID, err)
		// Continue to next chunk
		return a.Read(p)
	}

	// Skip if no chunk was produced (some events don't map to OpenAI chunks)
	if openaiChunk == nil {
		return a.Read(p)
	}

	// Marshal the OpenAI chunk to JSON
	openaiJSON, err := json.Marshal(openaiChunk)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to marshal OpenAI chunk: %v", a.requestID, err)
		// Continue to next chunk
		return a.Read(p)
	}

	// Format as SSE data
	a.buffer.WriteString(fmt.Sprintf("data: %s\n\n", string(openaiJSON)))

	return a.readFromBuffer(p)
}

// readFromBuffer reads data from the internal buffer
func (a *AnthropicToOpenAIStreamAdapter) readFromBuffer(p []byte) (int, error) {
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
