package anthropic

import (
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

// AnthropicStreamAdapter implements a streaming adapter for Anthropic to OpenAI conversion
// It implements both io.ReadCloser and ssestream.Decoder interfaces
type AnthropicStreamAdapter struct {
	anthropicStream *anthropicssestream.Stream[anthropic.MessageStreamEventUnion]
	ctx             context.Context
	cancel          context.CancelFunc
	started         bool
	mu              sync.Mutex
	modelName       string
	modelMu         sync.RWMutex
	messageID       string
	// usage tracking
	usageMu          sync.Mutex
	promptTokens     int
	completionTokens int
	// for ssestream.Decoder interface
	currentChunk *openai.ChatCompletionChunk
	err          error
	closed       bool
	finalSent    bool
}

// NewAnthropicStreamAdapter creates a new Anthropic stream adapter
func NewAnthropicStreamAdapter(
	stream *anthropicssestream.Stream[anthropic.MessageStreamEventUnion],
) *AnthropicStreamAdapter {
	ctx, cancel := context.WithCancel(context.Background())

	return &AnthropicStreamAdapter{
		anthropicStream: stream,
		ctx:             ctx,
		cancel:          cancel,
		modelName:       "claude-3-5-sonnet-20241022",
	}
}

// ConvertToOpenAIStream creates an OpenAI SSE stream from the Anthropic stream
func (a *AnthropicStreamAdapter) ConvertToOpenAIStream() (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.started {
		return nil, fmt.Errorf("stream conversion already started")
	}
	a.started = true

	// Return the SSE stream using this adapter as the decoder
	return ssestream.NewStream[openai.ChatCompletionChunk](a, nil), nil
}

// Next implements ssestream.Decoder interface
func (a *AnthropicStreamAdapter) Next() bool {
	if a.closed {
		return false
	}

	for {
		select {
		case <-a.ctx.Done():
			return false
		default:
			if !a.anthropicStream.Next() {
				err := a.anthropicStream.Err()
				if err != nil && err != io.EOF {
					a.err = err
				}
				// Send final usage chunk if needed
				if !a.finalSent {
					a.sendFinalChunk()
				}
				return false
			}

			event := a.anthropicStream.Current()

			// Update state
			a.updateState(event)

			// Convert event to chunk
			chunk := a.convertAnthropicEventToOpenAIChunk(event)
			if chunk != nil {
				a.currentChunk = chunk
				return true
			}
			// Continue loop if no chunk was generated
		}
	}
}

// Event implements ssestream.Decoder interface
func (a *AnthropicStreamAdapter) Event() ssestream.Event {
	if a.currentChunk == nil {
		return ssestream.Event{Type: "data", Data: []byte("{}")}
	}

	jsonData, err := json.Marshal(a.currentChunk)
	if err != nil {
		return ssestream.Event{Type: "data", Data: []byte("{}")}
	}

	return ssestream.Event{
		Type: "data",
		Data: jsonData,
	}
}

// Err implements ssestream.Decoder interface
func (a *AnthropicStreamAdapter) Err() error {
	return a.err
}

// Close implements both io.Closer and ssestream.Decoder interfaces
func (a *AnthropicStreamAdapter) Close() error {
	a.cancel()
	a.closed = true
	return a.anthropicStream.Close()
}

// updateState updates adapter state from Anthropic events
func (a *AnthropicStreamAdapter) updateState(event anthropic.MessageStreamEventUnion) {
	// Capture usage from message_delta events
	if md := event.AsMessageDelta(); md.Type == "message_delta" {
		a.usageMu.Lock()
		a.promptTokens = int(md.Usage.InputTokens)
		a.completionTokens = int(md.Usage.OutputTokens)
		a.usageMu.Unlock()
	}

	// Store model name and message ID from message_start event
	if messageStart := event.AsMessageStart(); messageStart.Type == "message_start" {
		a.modelMu.Lock()
		a.modelName = string(messageStart.Message.Model)
		a.messageID = messageStart.Message.ID
		a.modelMu.Unlock()
	}
}

// convertAnthropicEventToOpenAIChunk converts Anthropic stream events to OpenAI chunks
func (a *AnthropicStreamAdapter) convertAnthropicEventToOpenAIChunk(
	event anthropic.MessageStreamEventUnion,
) *openai.ChatCompletionChunk {
	a.modelMu.RLock()
	modelName := a.modelName
	messageID := a.messageID
	a.modelMu.RUnlock()

	if modelName == "" {
		modelName = "claude-3-5-sonnet-20241022"
	}
	if messageID == "" {
		messageID = fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	}

	// Handle content block delta (actual text content)
	if contentBlockDelta := event.AsContentBlockDelta(); contentBlockDelta.Type == "content_block_delta" {
		deltaEvent := contentBlockDelta.Delta.AsAny()
		if textDelta, ok := deltaEvent.(anthropic.TextDelta); ok {
			return &openai.ChatCompletionChunk{
				ID:      messageID,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   modelName,
				Choices: []openai.ChatCompletionChunkChoice{{
					Index: int64(contentBlockDelta.Index),
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Content: textDelta.Text,
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
		// Only send a chunk if it's a text block
		if contentBlock := contentBlockStart.ContentBlock.AsText(); contentBlock.Type == "text" {
			return &openai.ChatCompletionChunk{
				ID:      messageID,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   modelName,
				Choices: []openai.ChatCompletionChunkChoice{{
					Index: int64(contentBlockStart.Index),
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Role: "assistant",
					},
				}},
			}
		}
	}

	// Handle message stop - send a chunk with finish_reason
	if messageStop := event.AsMessageStop(); messageStop.Type == "message_stop" {
		return &openai.ChatCompletionChunk{
			ID:      messageID,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   modelName,
			Choices: []openai.ChatCompletionChunkChoice{{
				Index:        0,
				Delta:        openai.ChatCompletionChunkChoiceDelta{},
				FinishReason: "stop",
			}},
		}
	}

	return nil // Ignore other event types
}

// sendFinalChunk sends the final chunk to indicate stream completion, including usage
func (a *AnthropicStreamAdapter) sendFinalChunk() {
	if a.finalSent {
		return
	}

	a.modelMu.RLock()
	modelName := a.modelName
	messageID := a.messageID
	a.modelMu.RUnlock()

	if modelName == "" {
		modelName = "claude-3-5-sonnet-20241022"
	}
	if messageID == "" {
		messageID = fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	}

	a.usageMu.Lock()
	prompt := a.promptTokens
	completion := a.completionTokens
	a.usageMu.Unlock()
	total := prompt + completion

	// Send usage chunk if we have token counts
	if total > 0 {
		a.currentChunk = &openai.ChatCompletionChunk{
			ID:      messageID,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   modelName,
			Choices: []openai.ChatCompletionChunkChoice{{
				Index:        0,
				Delta:        openai.ChatCompletionChunkChoiceDelta{},
				FinishReason: "",
			}},
			Usage: openai.CompletionUsage{
				PromptTokens:     int64(prompt),
				CompletionTokens: int64(completion),
				TotalTokens:      int64(total),
			},
		}
	}

	a.finalSent = true
}

// GetProviderName returns the provider name
func (a *AnthropicStreamAdapter) GetProviderName() string {
	return "anthropic"
}
