package provider_interfaces

import (
	"context"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

type Completions interface {
	CreateCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*openai.ChatCompletion, error)
	StreamCompletion(ctx context.Context, req *openai.ChatCompletionNewParams) (*ssestream.Stream[openai.ChatCompletionChunk], error)
}

type Chat interface {
	Completions() Completions
}

type LLMProvider interface {
	Chat() Chat
}
