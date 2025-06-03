package gemini

import (
	"context"
	"fmt"
	"io"
	"iter"
	"sync"

	openai_provider "adaptive-backend/internal/services/providers/openai"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
	"google.golang.org/genai"
)

// GeminiStreamAdapter adapts Gemini streaming response to OpenAI format
type GeminiStreamAdapter struct {
	stream  iter.Seq2[*genai.GenerateContentResponse, error]
	decoder *openai_provider.StreamDecoder
	ctx     context.Context
	cancel  context.CancelFunc
	started bool
	mu      sync.Mutex
}

// NewGeminiStreamAdapter creates a new Gemini stream adapter
func NewGeminiStreamAdapter(stream iter.Seq2[*genai.GenerateContentResponse, error]) *GeminiStreamAdapter {
	ctx, cancel := context.WithCancel(context.Background())

	return &GeminiStreamAdapter{
		stream:  stream,
		decoder: openai_provider.NewStreamDecoder(),
		ctx:     ctx,
		cancel:  cancel,
	}
}

// ConvertToOpenAIStream starts the conversion and returns the OpenAI stream
func (a *GeminiStreamAdapter) ConvertToOpenAIStream() (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.started {
		return nil, fmt.Errorf("stream conversion already started")
	}
	a.started = true

	// Start the conversion goroutine
	go a.processGeminiStream()

	// Create and return the OpenAI stream wrapper
	return a.createOpenAIStream(), nil
}

// processGeminiStream processes the Gemini stream and converts responses to OpenAI chunks
func (a *GeminiStreamAdapter) processGeminiStream() {
	defer a.decoder.CloseSender()

	for resp, err := range a.stream {
		select {
		case <-a.ctx.Done():
			return
		default:
			if err != nil {
				if err == io.EOF {
					// Send final chunk to indicate stream end
					a.sendFinalChunk()
					return
				}
				a.decoder.SendError(fmt.Errorf("gemini stream error: %w", err))
				return
			}

			// Convert and send chunk
			chunk := a.convertGeminiResponseToOpenAIChunk(resp)
			if chunk != nil {
				if !a.decoder.SendChunk(*chunk) {
					return
				}
			}
		}
	}

	// Send final chunk when stream ends normally
	a.sendFinalChunk()
}

// convertGeminiResponseToOpenAIChunk converts a single Gemini response to OpenAI chunk
func (a *GeminiStreamAdapter) convertGeminiResponseToOpenAIChunk(resp *genai.GenerateContentResponse) *openai.ChatCompletionChunk {
	if resp == nil {
		return nil
	}

	var content string
	if len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
		content = resp.Candidates[0].Content.Parts[0].Text
	}

	return &openai.ChatCompletionChunk{
		ID:     resp.ResponseID,
		Object: "chat.completion.chunk",
		Model:  resp.ModelVersion,
		Choices: []openai.ChatCompletionChunkChoice{
			{
				Index: 0,
				Delta: openai.ChatCompletionChunkChoiceDelta{
					Role:    "assistant",
					Content: content,
				},
			},
		},
	}
}

// sendFinalChunk sends the final chunk to indicate stream completion
func (a *GeminiStreamAdapter) sendFinalChunk() {
	finalChunk := openai.ChatCompletionChunk{
		ID:     "gemini-stream",
		Object: "chat.completion.chunk",
		Choices: []openai.ChatCompletionChunkChoice{
			{
				Index:        0,
				Delta:        openai.ChatCompletionChunkChoiceDelta{},
				FinishReason: "stop",
			},
		},
	}

	a.decoder.SendChunk(finalChunk)
}

// createOpenAIStream creates a proper OpenAI stream from our decoder
func (a *GeminiStreamAdapter) createOpenAIStream() *ssestream.Stream[openai.ChatCompletionChunk] {
	return ssestream.NewStream[openai.ChatCompletionChunk](a.decoder, nil)
}

// Close closes the adapter and underlying streams
func (a *GeminiStreamAdapter) Close() error {
	a.cancel()
	return a.decoder.Close()
}

// GetProviderName returns the provider name
func (a *GeminiStreamAdapter) GetProviderName() string {
	return "gemini"
}
