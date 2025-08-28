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
	messageID       string
	model           string
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
	for {
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
			continue // Return to top of loop to read from buffer
		}

		anthropicEvent := a.anthropicStream.Current()

		// Track message ID and model from MessageStartEvent
		concreteEvent := anthropicEvent.AsAny()
		switch event := concreteEvent.(type) {
		case anthropic.MessageStartEvent:
			a.messageID = event.Message.ID
			a.model = string(event.Message.Model)
		}

		// Convert Anthropic event to OpenAI format - convert value types to pointers as needed
		var eventToConvert any
		switch event := concreteEvent.(type) {
		case anthropic.MessageStartEvent:
			eventToConvert = &event
		case anthropic.ContentBlockStartEvent:
			eventToConvert = &event
		case anthropic.ContentBlockDeltaEvent:
			eventToConvert = &event
		case anthropic.MessageStopEvent:
			eventToConvert = &event
		default:
			eventToConvert = concreteEvent
		}

		openaiChunk, err := format_adapter.AnthropicToOpenAI.ConvertStreamingChunk(eventToConvert, a.provider)
		if err != nil {
			fiberlog.Errorf("[%s] Failed to convert Anthropic event to OpenAI format: %v", a.requestID, err)
			// Continue to next chunk instead of recursive call
			continue
		}

		// Skip if no chunk was produced (some events don't map to OpenAI chunks)
		if openaiChunk == nil {
			continue // Continue to next chunk instead of recursive call
		}

		// Set tracked ID and model if they're empty
		if openaiChunk.ID == "" {
			openaiChunk.ID = a.messageID
		}
		if openaiChunk.Model == "" {
			openaiChunk.Model = a.model
		}

		// Marshal the OpenAI chunk to JSON
		openaiJSON, err := json.Marshal(openaiChunk)
		if err != nil {
			fiberlog.Errorf("[%s] Failed to marshal OpenAI chunk: %v", a.requestID, err)
			// Continue to next chunk instead of recursive call
			continue
		}

		// Format as SSE data
		a.buffer.WriteString(fmt.Sprintf("data: %s\n\n", string(openaiJSON)))
		// Continue to top of loop to read from buffer
	}
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
