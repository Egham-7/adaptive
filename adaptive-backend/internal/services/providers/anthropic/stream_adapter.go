package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicssestream "github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

// AnthropicStreamAdapter implements a full streaming adapter for Anthropic to OpenAI conversion
type AnthropicStreamAdapter struct {
	anthropicStream *anthropicssestream.Stream[anthropic.MessageStreamEventUnion]
	openaiStream    *OpenAIStreamReader
	ctx             context.Context
	cancel          context.CancelFunc
	started         bool
	mu              sync.Mutex
	modelName       string
	modelMu         sync.RWMutex
	// usage tracking
	usageMu          sync.Mutex
	promptTokens     int
	completionTokens int
}

// OpenAIStreamReader implements the SSE stream interface for OpenAI chunks
type OpenAIStreamReader struct {
	buffer      *bytes.Buffer
	scanner     *bufio.Scanner
	chunkChan   chan openai.ChatCompletionChunk
	errorChan   chan error
	done        chan struct{}
	closed      bool
	mu          sync.RWMutex
	currentData []byte
	hasNext     bool
}

// NewAnthropicStreamAdapter creates a new Anthropic stream adapter
func NewAnthropicStreamAdapter(
	stream *anthropicssestream.Stream[anthropic.MessageStreamEventUnion],
) *AnthropicStreamAdapter {
	ctx, cancel := context.WithCancel(context.Background())

	buffer := &bytes.Buffer{}
	openaiReader := &OpenAIStreamReader{
		buffer:    buffer,
		scanner:   bufio.NewScanner(buffer),
		chunkChan: make(chan openai.ChatCompletionChunk, 100),
		errorChan: make(chan error, 1),
		done:      make(chan struct{}),
	}

	return &AnthropicStreamAdapter{
		anthropicStream:  stream,
		openaiStream:     openaiReader,
		ctx:              ctx,
		cancel:           cancel,
		modelName:        "claude-3-5-sonnet-20241022",
		promptTokens:     0,
		completionTokens: 0,
	}
}

// ConvertToOpenAIStream starts the conversion and returns the OpenAI stream
func (a *AnthropicStreamAdapter) ConvertToOpenAIStream() (
	*ssestream.Stream[openai.ChatCompletionChunk], error,
) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.started {
		return nil, fmt.Errorf("stream conversion already started")
	}
	a.started = true

	// Start the conversion goroutine
	go a.processAnthropicStream()

	// Create and return the OpenAI stream wrapper
	return ssestream.NewStream[openai.ChatCompletionChunk](a.openaiStream, nil), nil
}

// processAnthropicStream processes the Anthropic stream and converts events to OpenAI chunks
func (a *AnthropicStreamAdapter) processAnthropicStream() {
	for {
		select {
		case <-a.ctx.Done():
			a.closeStream()
			return
		default:
			if !a.anthropicStream.Next() {
				err := a.anthropicStream.Err()
				if err != nil && err != io.EOF {
					select {
					case a.openaiStream.errorChan <- fmt.Errorf("anthropic stream error: %w", err):
					case <-a.ctx.Done():
					}
				}
				// Send final chunk (with usage) and close
				a.sendFinalChunk()
				a.closeStream()
				return
			}

			event := a.anthropicStream.Current()

			// capture usage from message_delta events
			if md := event.AsMessageDelta(); md.Type == "message_delta" {
				a.usageMu.Lock()
				a.promptTokens = int(md.Usage.InputTokens)
				a.completionTokens = int(md.Usage.OutputTokens)
				a.usageMu.Unlock()
			}

			// Store model name from message_start event
			if messageStart := event.AsMessageStart(); messageStart.Type == "message_start" {
				a.modelMu.Lock()
				a.modelName = string(messageStart.Message.Model)
				a.modelMu.Unlock()
			}

			chunk := a.convertAnthropicEventToOpenAIChunk(event)
			if chunk != nil {
				a.writeChunkToBuffer(*chunk)
			}
		}
	}
}

// writeChunkToBuffer writes a chunk as SSE data to the buffer
func (a *AnthropicStreamAdapter) writeChunkToBuffer(chunk openai.ChatCompletionChunk) {
	a.openaiStream.mu.Lock()
	defer a.openaiStream.mu.Unlock()

	if a.openaiStream.closed {
		return
	}

	// Serialize chunk to JSON
	jsonData, err := json.Marshal(chunk)
	if err != nil {
		select {
		case a.openaiStream.errorChan <- fmt.Errorf("failed to marshal chunk: %w", err):
		case <-a.ctx.Done():
		}
		return
	}

	// Write as SSE format
	sseData := fmt.Sprintf("data: %s\n\n", string(jsonData))
	a.openaiStream.buffer.WriteString(sseData)

	// Signal that new data is available
	select {
	case a.openaiStream.chunkChan <- chunk:
	case <-a.ctx.Done():
	}
}

// convertAnthropicEventToOpenAIChunk converts Anthropic stream events to OpenAI chunks
func (a *AnthropicStreamAdapter) convertAnthropicEventToOpenAIChunk(
	event anthropic.MessageStreamEventUnion,
) *openai.ChatCompletionChunk {
	// Handle content block delta (actual text content)
	if contentBlockDelta := event.AsContentBlockDelta(); contentBlockDelta.Type == "content_block_delta" {
		if contentBlockDelta.Delta.Type == "text_delta" {
			a.modelMu.RLock()
			modelName := a.modelName
			a.modelMu.RUnlock()
			if modelName == "" {
				modelName = "claude-3-5-sonnet-20241022"
			}
			return &openai.ChatCompletionChunk{
				ID:      fmt.Sprintf("anthropic-stream-%d", contentBlockDelta.Index),
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   modelName,
				Choices: []openai.ChatCompletionChunkChoice{{
					Index: 0,
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Role:    "assistant",
						Content: contentBlockDelta.Delta.Text,
					},
				}},
			}
		}
	}

	// Handle message start (beginning of response)
	if messageStart := event.AsMessageStart(); messageStart.Type == "message_start" {
		return &openai.ChatCompletionChunk{
			ID:      messageStart.Message.ID,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   string(messageStart.Message.Model),
			Choices: []openai.ChatCompletionChunkChoice{{
				Index: 0,
				Delta: openai.ChatCompletionChunkChoiceDelta{
					Role: "assistant",
				},
			}},
		}
	}

	// Handle content block start
	if contentBlockStart := event.AsContentBlockStart(); contentBlockStart.Type == "content_block_start" {
		a.modelMu.RLock()
		modelName := a.modelName
		a.modelMu.RUnlock()
		if modelName == "" {
			modelName = "claude-3-5-sonnet-20241022"
		}
		return &openai.ChatCompletionChunk{
			ID:      fmt.Sprintf("anthropic-stream-%d", contentBlockStart.Index),
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   modelName,
			Choices: []openai.ChatCompletionChunkChoice{{
				Index: 0,
				Delta: openai.ChatCompletionChunkChoiceDelta{
					Role: "assistant",
				},
			}},
		}
	}

	return nil // Ignore other event types
}

// sendFinalChunk sends the final chunk to indicate stream completion, including usage
func (a *AnthropicStreamAdapter) sendFinalChunk() {
	a.modelMu.RLock()
	modelName := a.modelName
	a.modelMu.RUnlock()
	if modelName == "" {
		modelName = "claude-3-5-sonnet-20241022"
	}

	a.usageMu.Lock()
	prompt := a.promptTokens
	completion := a.completionTokens
	a.usageMu.Unlock()
	total := prompt + completion

	finalChunk := openai.ChatCompletionChunk{
		ID:      fmt.Sprintf("anthropic-stream-final-%d", time.Now().UnixNano()),
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   modelName,
		Choices: []openai.ChatCompletionChunkChoice{{
			Index:        0,
			Delta:        openai.ChatCompletionChunkChoiceDelta{},
			FinishReason: "stop",
		}},
		Usage: openai.CompletionUsage{
			PromptTokens:     int64(prompt),
			CompletionTokens: int64(completion),
			TotalTokens:      int64(total),
		},
	}

	a.writeChunkToBuffer(finalChunk)

	// Write final SSE termination
	a.openaiStream.mu.Lock()
	a.openaiStream.buffer.WriteString("data: [DONE]\n\n")
	a.openaiStream.mu.Unlock()
}

// closeStream closes all channels and marks the stream as closed
func (a *AnthropicStreamAdapter) closeStream() {
	a.openaiStream.mu.Lock()
	defer a.openaiStream.mu.Unlock()

	if !a.openaiStream.closed {
		a.openaiStream.closed = true
		close(a.openaiStream.chunkChan)
		close(a.openaiStream.errorChan)
		close(a.openaiStream.done)
	}
}

// Read implements io.Reader for OpenAIStreamReader
func (r *OpenAIStreamReader) Read(p []byte) (n int, err error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.closed {
		return 0, io.EOF
	}

	return r.buffer.Read(p)
}

// Next implements ssestream.Decoder interface
func (r *OpenAIStreamReader) Next() bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return false
	}

	// Check if we have data in the buffer
	if r.buffer.Len() > 0 {
		// Read the next line
		if r.scanner.Scan() {
			r.currentData = r.scanner.Bytes()
			r.hasNext = true
			return true
		}
	}

	// Wait for new data
	select {
	case _, ok := <-r.chunkChan:
		if !ok {
			return false
		}
		// Reset scanner to read from updated buffer
		r.scanner = bufio.NewScanner(r.buffer)
		if r.scanner.Scan() {
			r.currentData = r.scanner.Bytes()
			r.hasNext = true
			return true
		}
		return false
	case <-r.done:
		return false
	}
}

// Event implements ssestream.Decoder interface
func (r *OpenAIStreamReader) Event() ssestream.Event {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.hasNext {
		// Parse SSE event from current data
		line := string(r.currentData)
		if len(line) > 6 && line[:6] == "data: " {
			data := line[6:]
			if data == "[DONE]" {
				return ssestream.Event{
					Type: "done",
					Data: []byte("[DONE]"),
				}
			}
			return ssestream.Event{
				Type: "data",
				Data: []byte(data),
			}
		}
	}

	return ssestream.Event{
		Type: "data",
		Data: []byte("{}"),
	}
}

// Close implements ssestream.Decoder interface
func (r *OpenAIStreamReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.closed {
		r.closed = true
		close(r.done)
	}
	return nil
}

// Err implements ssestream.Decoder interface
func (r *OpenAIStreamReader) Err() error {
	select {
	case err := <-r.errorChan:
		return err
	default:
		return nil
	}
}

// Close closes the adapter and underlying streams
func (a *AnthropicStreamAdapter) Close() error {
	a.cancel()
	a.closeStream()
	return a.anthropicStream.Close()
}

// GetProviderName returns the provider name
func (a *AnthropicStreamAdapter) GetProviderName() string {
	return "anthropic"
}
