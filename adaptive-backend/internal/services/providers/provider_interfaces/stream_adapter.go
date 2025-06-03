package provider_interfaces

import (
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

// StreamAdapter defines the interface that all provider stream adapters must implement
type StreamAdapter interface {
	// ConvertToOpenAIStream starts the conversion and returns the OpenAI stream
	ConvertToOpenAIStream() (*ssestream.Stream[openai.ChatCompletionChunk], error)

	// Close closes the adapter and underlying streams
	Close() error

	// GetProviderName returns the provider name
	GetProviderName() string
}
