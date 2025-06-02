package vercel

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/stream_readers"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicssestream "github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	deepseek "github.com/cohesion-org/deepseek-go"
	"github.com/conneroisu/groq-go"
	"github.com/openai/openai-go"
	openaissestream "github.com/openai/openai-go/packages/ssestream"
)

func GetDataStreamReader(resp *models.ChatCompletionResponse, provider string, requestID string) (stream_readers.StreamReader, error) {
	switch provider {
	case "openai":
		stream, ok := resp.Response.(*openaissestream.Stream[openai.ChatCompletionChunk])
		if !ok {
			return nil, fmt.Errorf("invalid OpenAI stream type")
		}
		return NewVercelDataStreamReaderFromAdapter(NewOpenAIAdapter(stream, requestID), requestID), nil

	case "groq":
		stream, ok := resp.Response.(*groq.ChatCompletionStream)
		if !ok {
			return nil, fmt.Errorf("invalid Groq stream type")
		}
		return NewVercelDataStreamReaderFromAdapter(NewGroqAdapter(stream, requestID), requestID), nil

	case "deepseek":
		stream, ok := resp.Response.(deepseek.ChatCompletionStream)
		if !ok {
			return nil, fmt.Errorf("invalid DeepSeek stream type")
		}
		return NewVercelDataStreamReaderFromAdapter(NewDeepSeekAdapter(stream, requestID), requestID), nil

	case "anthropic":
		stream, ok := resp.Response.(anthropicssestream.Stream[anthropic.MessageStreamEventUnion])
		if !ok {
			return nil, fmt.Errorf("invalid Anthropic stream type")
		}
		return NewVercelDataStreamReaderFromAdapter(NewAnthropicAdapter(stream, requestID), requestID), nil

	default:
		return nil, fmt.Errorf("unsupported provider for DataStream: %s", provider)
	}
}
