package handlers

import (
	"io"
	"iter"

	"adaptive-backend/internal/services/stream/contracts"
	"adaptive-backend/internal/services/stream/processors"
	"adaptive-backend/internal/services/stream/readers"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/openai/openai-go/v2"
	openai_ssestream "github.com/openai/openai-go/v2/packages/ssestream"
	"google.golang.org/genai"
)

// StreamFactory creates properly layered streaming pipelines
type StreamFactory struct{}

// NewStreamFactory creates a new factory
func NewStreamFactory() *StreamFactory {
	return &StreamFactory{}
}

// CreateOpenAIPipeline creates a complete OpenAI streaming pipeline
func (f *StreamFactory) CreateOpenAIPipeline(
	stream *openai_ssestream.Stream[openai.ChatCompletionChunk],
	requestID, provider, cacheSource string,
) contracts.StreamHandler {
	reader := readers.NewOpenAIStreamReader(stream, requestID)
	processor := processors.NewOpenAIChunkProcessor(provider, cacheSource, requestID)
	return NewStreamOrchestrator(reader, processor, requestID)
}

// CreateAnthropicPipeline creates a complete Anthropic streaming pipeline
func (f *StreamFactory) CreateAnthropicPipeline(
	responseBody io.Reader,
	requestID, provider, cacheSource string,
) contracts.StreamHandler {
	reader := readers.NewAnthropicStreamReader(responseBody, requestID)
	processor := processors.NewAnthropicChunkProcessor(provider, cacheSource, requestID)
	return NewStreamOrchestrator(reader, processor, requestID)
}

// CreateAnthropicNativePipeline creates a complete Anthropic native streaming pipeline
func (f *StreamFactory) CreateAnthropicNativePipeline(
	stream *ssestream.Stream[anthropic.MessageStreamEventUnion],
	requestID, provider, cacheSource string,
) contracts.StreamHandler {
	reader := readers.NewAnthropicNativeStreamReader(stream, requestID)
	processor := processors.NewAnthropicChunkProcessor(provider, cacheSource, requestID)
	return NewStreamOrchestrator(reader, processor, requestID)
}

// CreateGeminiPipeline creates a complete Gemini streaming pipeline
func (f *StreamFactory) CreateGeminiPipeline(
	streamIter iter.Seq2[*genai.GenerateContentResponse, error],
	requestID, provider, cacheSource string,
) contracts.StreamHandler {
	reader := readers.NewGeminiStreamReader(streamIter, requestID)
	// Use Gemini processor to format as SSE events for SDK compatibility
	processor := processors.NewGeminiChunkProcessor(provider, cacheSource, requestID)
	return NewStreamOrchestrator(reader, processor, requestID)
}
