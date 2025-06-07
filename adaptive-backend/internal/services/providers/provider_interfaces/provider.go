package provider_interfaces

import (
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

type Completions interface {
	CreateCompletion(req *openai.ChatCompletionNewParams) (*openai.ChatCompletion, error)
	StreamCompletion(req *openai.ChatCompletionNewParams) (*ssestream.Stream[openai.ChatCompletionChunk], error)
}

type Chat interface {
	Completions() Completions
}

type LLMProvider interface {
	Chat() Chat
}
