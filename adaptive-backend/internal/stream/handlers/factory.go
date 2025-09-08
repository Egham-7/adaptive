package handlers

import (
	"io"

	"adaptive-backend/internal/stream/contracts"
	"adaptive-backend/internal/stream/processors"
	"adaptive-backend/internal/stream/readers"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/openai/openai-go/v2"
	openai_ssestream "github.com/openai/openai-go/v2/packages/ssestream"
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
	requestID, provider string,
) contracts.StreamHandler {
	reader := readers.NewAnthropicStreamReader(responseBody, requestID)
	processor := processors.NewAnthropicChunkProcessor(provider, requestID)
	return NewStreamOrchestrator(reader, processor, requestID)
}

// CreateAnthropicNativePipeline creates a complete Anthropic native streaming pipeline
func (f *StreamFactory) CreateAnthropicNativePipeline(
	stream *ssestream.Stream[anthropic.MessageStreamEventUnion],
	requestID, provider string,
) contracts.StreamHandler {
	reader := readers.NewAnthropicNativeStreamReader(stream, requestID)
	processor := processors.NewAnthropicChunkProcessor(provider, requestID)
	return NewStreamOrchestrator(reader, processor, requestID)
}