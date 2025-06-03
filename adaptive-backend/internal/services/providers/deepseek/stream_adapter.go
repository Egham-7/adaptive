package deepseek

import (
	"context"
	"fmt"
	"io"
	"sync"

	openai_provider "adaptive-backend/internal/services/providers/openai"

	"github.com/cohesion-org/deepseek-go"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

// DeepSeekStreamAdapter implements a full streaming adapter for DeepSeek to OpenAI conversion
type DeepSeekStreamAdapter struct {
	deepseekStream deepseek.ChatCompletionStream
	decoder        *openai_provider.StreamDecoder
	ctx            context.Context
	cancel         context.CancelFunc
	started        bool
	mu             sync.Mutex
}

// NewDeepSeekStreamAdapter creates a new DeepSeek stream adapter
func NewDeepSeekStreamAdapter(stream deepseek.ChatCompletionStream) *DeepSeekStreamAdapter {
	ctx, cancel := context.WithCancel(context.Background())

	return &DeepSeekStreamAdapter{
		deepseekStream: stream,
		decoder:        openai_provider.NewStreamDecoder(),
		ctx:            ctx,
		cancel:         cancel,
	}
}

// ConvertToOpenAIStream starts the conversion and returns the OpenAI stream
func (d *DeepSeekStreamAdapter) ConvertToOpenAIStream() (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.started {
		return nil, fmt.Errorf("stream conversion already started")
	}
	d.started = true

	// Start the conversion goroutine
	go d.processDeepSeekStream()

	// Create and return the OpenAI stream wrapper
	return d.createOpenAIStream(), nil
}

// processDeepSeekStream processes the DeepSeek stream and converts responses to OpenAI chunks
func (d *DeepSeekStreamAdapter) processDeepSeekStream() {
	defer d.decoder.CloseSender()
	defer d.deepseekStream.Close()

	for {
		select {
		case <-d.ctx.Done():
			return
		default:
			response, err := d.deepseekStream.Recv()
			if err != nil {
				if err == io.EOF {
					// Send final chunk to indicate stream end
					d.sendFinalChunk()
					return
				}
				d.decoder.SendError(fmt.Errorf("deepseek stream error: %w", err))
				return
			}

			// Convert and send chunk
			chunk := d.convertDeepSeekResponseToOpenAIChunk(response)
			if chunk != nil {
				if !d.decoder.SendChunk(*chunk) {
					return
				}
			}
		}
	}
}

// convertDeepSeekResponseToOpenAIChunk converts DeepSeek stream response to OpenAI chunks
func (d *DeepSeekStreamAdapter) convertDeepSeekResponseToOpenAIChunk(response *deepseek.StreamChatCompletionResponse) *openai.ChatCompletionChunk {
	if response == nil {
		return nil
	}

	choices := make([]openai.ChatCompletionChunkChoice, len(response.Choices))
	for i, choice := range response.Choices {
		choices[i] = openai.ChatCompletionChunkChoice{
			Index: int64(choice.Index),
			Delta: openai.ChatCompletionChunkChoiceDelta{
				Role:    choice.Delta.Role,
				Content: choice.Delta.Content,
			},
			FinishReason: choice.FinishReason,
		}
	}

	var usage *openai.CompletionUsage
	if response.Usage != nil {
		usage = &openai.CompletionUsage{
			CompletionTokens: int64(response.Usage.CompletionTokens),
			PromptTokens:     int64(response.Usage.PromptTokens),
			TotalTokens:      int64(response.Usage.TotalTokens),
		}
	}

	return &openai.ChatCompletionChunk{
		ID:      response.ID,
		Object:  "chat.completion.chunk",
		Created: response.Created,
		Model:   response.Model,
		Choices: choices,
		Usage:   *usage,
	}
}

// sendFinalChunk sends the final chunk to indicate stream completion
func (d *DeepSeekStreamAdapter) sendFinalChunk() {
	finalChunk := openai.ChatCompletionChunk{
		ID:     "deepseek-stream",
		Object: "chat.completion.chunk",
		Choices: []openai.ChatCompletionChunkChoice{
			{
				Index:        0,
				Delta:        openai.ChatCompletionChunkChoiceDelta{},
				FinishReason: "stop",
			},
		},
	}

	d.decoder.SendChunk(finalChunk)
}

// createOpenAIStream creates a proper OpenAI stream from our decoder
func (d *DeepSeekStreamAdapter) createOpenAIStream() *ssestream.Stream[openai.ChatCompletionChunk] {
	return ssestream.NewStream[openai.ChatCompletionChunk](d.decoder, nil)
}

// Close closes the adapter and underlying streams
func (d *DeepSeekStreamAdapter) Close() error {
	d.cancel()

	if err := d.decoder.Close(); err != nil {
		return err
	}

	return d.deepseekStream.Close()
}

// GetProviderName returns the provider name
func (d *DeepSeekStreamAdapter) GetProviderName() string {
	return "deepseek"
}
