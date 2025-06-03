package anthropic

import (
	"context"
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
}

// OpenAIStreamReader implements the OpenAI stream interface
type OpenAIStreamReader struct {
	chunkChan chan openai.ChatCompletionChunk
	errorChan chan error
	done      chan struct{}
	closed    bool
	mu        sync.RWMutex
}

// NewAnthropicStreamAdapter creates a new Anthropic stream adapter
func NewAnthropicStreamAdapter(stream *anthropicssestream.Stream[anthropic.MessageStreamEventUnion]) *AnthropicStreamAdapter {
	ctx, cancel := context.WithCancel(context.Background())

	openaiReader := &OpenAIStreamReader{
		chunkChan: make(chan openai.ChatCompletionChunk, 10),
		errorChan: make(chan error, 1),
		done:      make(chan struct{}),
	}

	return &AnthropicStreamAdapter{
		anthropicStream: stream,
		openaiStream:    openaiReader,
		ctx:             ctx,
		cancel:          cancel,
	}
}

// ConvertToOpenAIStream starts the conversion and returns the OpenAI stream
func (a *AnthropicStreamAdapter) ConvertToOpenAIStream() (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.started {
		return nil, fmt.Errorf("stream conversion already started")
	}
	a.started = true

	// Start the conversion goroutine
	go a.processAnthropicStream()

	// Create and return the OpenAI stream wrapper
	return a.createOpenAIStream(), nil
}

// processAnthropicStream processes the Anthropic stream and converts events to OpenAI chunks
func (a *AnthropicStreamAdapter) processAnthropicStream() {
	defer close(a.openaiStream.chunkChan)
	defer close(a.openaiStream.errorChan)
	defer close(a.openaiStream.done)

	for {
		select {
		case <-a.ctx.Done():
			return
		default:
			if !a.anthropicStream.Next() {
				err := a.anthropicStream.Err()
				if err != nil {
					if err == io.EOF {
						// Send final chunk to indicate stream end
						a.sendFinalChunk()
						return
					}
					select {
					case a.openaiStream.errorChan <- fmt.Errorf("anthropic stream error: %w", err):
					case <-a.ctx.Done():
					}
				}
				return
			}

			event := a.anthropicStream.Current()

			// Convert and send chunk
			chunk := a.convertAnthropicEventToOpenAIChunk(event)
			if chunk != nil {
				select {
				case a.openaiStream.chunkChan <- *chunk:
				case <-a.ctx.Done():
					return
				}
			}
		}
	}
}

// convertAnthropicEventToOpenAIChunk converts Anthropic stream events to OpenAI chunks
func (a *AnthropicStreamAdapter) convertAnthropicEventToOpenAIChunk(event anthropic.MessageStreamEventUnion) *openai.ChatCompletionChunk {
	// Handle content block delta (actual text content)
	if contentBlockDelta := event.AsContentBlockDelta(); contentBlockDelta.Type == "content_block_delta" {
		if contentBlockDelta.Delta.Type == "text_delta" {
			return &openai.ChatCompletionChunk{
				ID:     fmt.Sprintf("anthropic-stream-%d", contentBlockDelta.Index),
				Object: "chat.completion.chunk",
				Model:  string(event.Message.Model),
				Choices: []openai.ChatCompletionChunkChoice{
					{
						Index: 0,
						Delta: openai.ChatCompletionChunkChoiceDelta{
							Role:    "assistant",
							Content: contentBlockDelta.Delta.Text,
						},
					},
				},
			}
		}
	}

	// Handle message start (beginning of response)
	if messageStart := event.AsMessageStart(); messageStart.Type == "message_start" {
		return &openai.ChatCompletionChunk{
			ID:     messageStart.Message.ID,
			Object: "chat.completion.chunk",
			Model:  string(messageStart.Message.Model),
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Index: 0,
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Role: "assistant",
					},
				},
			},
		}
	}

	// Handle content block start
	if contentBlockStart := event.AsContentBlockStart(); contentBlockStart.Type == "content_block_start" {
		return &openai.ChatCompletionChunk{
			ID:     fmt.Sprintf("anthropic-stream-%d", contentBlockStart.Index),
			Object: "chat.completion.chunk",
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Index: 0,
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Role: "assistant",
					},
				},
			},
		}
	}

	return nil // Ignore other event types
}

// sendFinalChunk sends the final chunk to indicate stream completion
func (a *AnthropicStreamAdapter) sendFinalChunk() {
	finalChunk := openai.ChatCompletionChunk{
		ID:     fmt.Sprintf("anthropic-stream-final-%d", time.Now().UnixNano()),
		Object: "chat.completion.chunk",
		Choices: []openai.ChatCompletionChunkChoice{
			{
				Index:        0,
				Delta:        openai.ChatCompletionChunkChoiceDelta{},
				FinishReason: "stop",
			},
		},
	}

	select {
	case a.openaiStream.chunkChan <- finalChunk:
	case <-a.ctx.Done():
	}
}

// createOpenAIStream creates a proper OpenAI stream from our reader
func (a *AnthropicStreamAdapter) createOpenAIStream() *ssestream.Stream[openai.ChatCompletionChunk] {
	return ssestream.NewStream[openai.ChatCompletionChunk](a.openaiStream, nil)
}

// Read implements io.Reader for OpenAIStreamReader
func (r *OpenAIStreamReader) Read(p []byte) (n int, err error) {
	return 0, fmt.Errorf("Read method not implemented for streaming")
}

// Next implements the Decoder interface for OpenAIStreamReader
func (r *OpenAIStreamReader) Next() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.closed {
		return false
	}

	select {
	case _, ok := <-r.chunkChan:
		return ok
	case <-r.done:
		return false
	}
}

// Event implements the Decoder interface for OpenAIStreamReader
func (r *OpenAIStreamReader) Event() ssestream.Event {
	return ssestream.Event{}
}

// Close closes the adapter and underlying streams
func (a *AnthropicStreamAdapter) Close() error {
	a.cancel()

	a.openaiStream.mu.Lock()
	defer a.openaiStream.mu.Unlock()

	if !a.openaiStream.closed {
		a.openaiStream.closed = true
	}

	return a.anthropicStream.Close()
}

// Close implements the Decoder interface for OpenAIStreamReader
func (r *OpenAIStreamReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.closed {
		r.closed = true
		close(r.done)
	}
	return nil
}

// Err implements the Decoder interface for OpenAIStreamReader
func (r *OpenAIStreamReader) Err() error {
	select {
	case err := <-r.errorChan:
		return err
	default:
		return nil
	}
}

// GetProviderName returns the provider name
func (a *AnthropicStreamAdapter) GetProviderName() string {
	return "anthropic"
}
