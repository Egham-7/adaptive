package adapters

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"adaptive-backend/internal/services/format_adapter"

	"github.com/anthropics/anthropic-sdk-go"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/ssestream"
)

const (
	// Anthropic streaming event types
	eventMessageStart      = "message_start"
	eventContentBlockStart = "content_block_start"
	eventContentBlockDelta = "content_block_delta"
	eventContentBlockStop  = "content_block_stop"
	eventMessageDelta      = "message_delta"
	eventMessageStop       = "message_stop"

	// SSE formatting
	sseEventFormat = "event: %s\ndata: %s\n\n"
	sseDoneMarker  = "data: [DONE]\n\n"

	// Content block constants
	textContentBlockIndex = 0
	textContentBlockType  = "text"
)

// streamState tracks the state of the Anthropic streaming conversion
type streamState struct {
	textContentStarted bool
	toolContentStarted bool
}

// OpenAIToAnthropicStreamAdapter converts OpenAI streaming response to Anthropic SSE format
type OpenAIToAnthropicStreamAdapter struct {
	openaiStream *ssestream.Stream[openai.ChatCompletionChunk]
	provider     string
	requestID    string
	buffer       strings.Builder
	done         bool
	state        streamState
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
	// Return buffered data if available
	if a.buffer.Len() > 0 {
		return a.readFromBuffer(p)
	}

	// Return EOF if stream is complete
	if a.done {
		return 0, io.EOF
	}

	// Process chunks until we have data to return
	for {
		if !a.openaiStream.Next() {
			return a.handleStreamEnd(p)
		}

		chunk := a.openaiStream.Current()
		if err := a.processChunk(&chunk); err != nil {
			fiberlog.Errorf("[%s] Failed to process OpenAI chunk: %v", a.requestID, err)
			continue
		}

		// Return data if buffer has content
		if a.buffer.Len() > 0 {
			return a.readFromBuffer(p)
		}
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

// handleStreamEnd processes the end of the OpenAI stream and returns final data
func (a *OpenAIToAnthropicStreamAdapter) handleStreamEnd(p []byte) (int, error) {
	a.done = true

	// Check for stream errors
	if err := a.openaiStream.Err(); err != nil {
		return 0, err
	}

	// Emit final events in correct order
	a.emitFinalEvents()
	return a.readFromBuffer(p)
}

// processChunk converts and buffers a single OpenAI chunk
func (a *OpenAIToAnthropicStreamAdapter) processChunk(chunk *openai.ChatCompletionChunk) error {
	event, err := format_adapter.OpenAIToAnthropic.ConvertStreamingChunk(chunk, a.provider)
	if err != nil {
		return fmt.Errorf("failed to convert chunk: %w", err)
	}

	// Skip empty events
	if event == nil {
		return nil
	}

	// Handle event sequencing based on type
	switch event.Type {
	case eventContentBlockDelta:
		return a.handleContentBlockDelta(event, chunk)
	case eventMessageDelta:
		return a.handleMessageDelta(event)
	case eventContentBlockStart:
		a.state.toolContentStarted = true
		return a.emitEvent(event)
	default:
		return a.emitEvent(event)
	}
}

// handleContentBlockDelta ensures content_block_start is emitted before content_block_delta
func (a *OpenAIToAnthropicStreamAdapter) handleContentBlockDelta(event *anthropic.MessageStreamEventUnion, chunk *openai.ChatCompletionChunk) error {
	// Emit content_block_start if this is the first text content
	if !a.state.textContentStarted && a.isTextContentDelta(chunk) {
		if err := a.emitTextContentBlockStart(); err != nil {
			return err
		}
		a.state.textContentStarted = true
	}

	return a.emitEvent(event)
}

// handleMessageDelta closes content blocks before emitting message_delta
func (a *OpenAIToAnthropicStreamAdapter) handleMessageDelta(event *anthropic.MessageStreamEventUnion) error {
	if a.state.textContentStarted {
		if err := a.emitContentBlockStop(); err != nil {
			return err
		}
		a.state.textContentStarted = false
	}

	return a.emitEvent(event)
}

// isTextContentDelta checks if chunk contains text content
func (a *OpenAIToAnthropicStreamAdapter) isTextContentDelta(chunk *openai.ChatCompletionChunk) bool {
	return len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != ""
}

// emitTextContentBlockStart emits the content_block_start event for text content
func (a *OpenAIToAnthropicStreamAdapter) emitTextContentBlockStart() error {
	event := map[string]any{
		"type":  eventContentBlockStart,
		"index": textContentBlockIndex,
		"content_block": map[string]any{
			"type": textContentBlockType,
			"text": "",
		},
	}
	return a.emitEventMap(eventContentBlockStart, event)
}

// emitContentBlockStop emits content_block_stop event
func (a *OpenAIToAnthropicStreamAdapter) emitContentBlockStop() error {
	event := map[string]any{
		"type":  eventContentBlockStop,
		"index": textContentBlockIndex,
	}
	return a.emitEventMap(eventContentBlockStop, event)
}

// emitFinalEvents emits the final sequence of events before stream end
func (a *OpenAIToAnthropicStreamAdapter) emitFinalEvents() {
	// Close any open content blocks
	if a.state.textContentStarted {
		_ = a.emitContentBlockStop() // Best effort - ignore errors at stream end
	}

	// Emit message_stop
	messageStopEvent := map[string]any{"type": eventMessageStop}
	_ = a.emitEventMap(eventMessageStop, messageStopEvent) // Best effort

	// Write final SSE terminator
	a.buffer.WriteString(sseDoneMarker)
}

// emitEvent marshals and writes an Anthropic event to the buffer
func (a *OpenAIToAnthropicStreamAdapter) emitEvent(event *anthropic.MessageStreamEventUnion) error {
	eventJSON, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("failed to marshal event: %w", err)
	}

	a.buffer.WriteString(fmt.Sprintf(sseEventFormat, event.Type, string(eventJSON)))
	return nil
}

// emitEventMap marshals and writes a map-based event to the buffer
func (a *OpenAIToAnthropicStreamAdapter) emitEventMap(eventType string, event map[string]any) error {
	eventJSON, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("failed to marshal %s event: %w", eventType, err)
	}

	a.buffer.WriteString(fmt.Sprintf(sseEventFormat, eventType, string(eventJSON)))
	return nil
}
