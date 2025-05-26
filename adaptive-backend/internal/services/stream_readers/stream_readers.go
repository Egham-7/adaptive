package stream_readers

import (
	"adaptive-backend/internal/models"
	"fmt"
	"io"
	"sync"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicssestream "github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/cohesion-org/deepseek-go"
	"github.com/conneroisu/groq-go"
	"github.com/openai/openai-go"
	openaissestream "github.com/openai/openai-go/packages/ssestream"
)

// StreamReader is the common interface for all LLM stream readers
type StreamReader interface {
	io.Reader
	io.Closer
}

// BaseStreamReader provides common functionality for all stream readers
type BaseStreamReader struct {
	buffer    []byte
	requestID string
	closeLock sync.Once
}

// GetStreamReader creates the appropriate stream reader based on provider type
func GetStreamReader(resp *models.ChatCompletionResponse, provider string, requestID string) (StreamReader, error) {
	switch provider {
	case "openai":
		stream, ok := resp.Response.(*openaissestream.Stream[openai.ChatCompletionChunk])
		if !ok {
			return nil, fmt.Errorf("invalid OpenAI stream type")
		}
		return NewOpenAIStreamReader(stream, requestID), nil
	case "groq":
		stream, ok := resp.Response.(*groq.ChatCompletionStream)
		if !ok {
			return nil, fmt.Errorf("invalid Groq stream type")
		}
		return NewGroqStreamReader(stream, requestID), nil
	case "deepseek":
		stream, ok := resp.Response.(deepseek.ChatCompletionStream)
		if !ok {
			return nil, fmt.Errorf("invalid DeepSeek stream type")
		}
		return NewDeepSeekStreamReader(stream, requestID), nil

	case "anthropic":
		stream, ok := resp.Response.(anthropicssestream.Stream[anthropic.MessageStreamEventUnion])

		if !ok {
			return nil, fmt.Errorf("invalid Anthropic stream type")
		}
		return NewAnthropicStreamReader(stream, requestID), nil

	default:
		return nil, fmt.Errorf("unsupported provider for streaming: %s", provider)
	}
}
