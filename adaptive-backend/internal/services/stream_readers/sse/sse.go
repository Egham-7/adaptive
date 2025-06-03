package sse

import (
	"adaptive-backend/internal/services/stream_readers"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

func GetSSEStreamReader(stream *ssestream.Stream[openai.ChatCompletionChunk], requestID string) (stream_readers.StreamReader, error) {
	return NewOpenAIStreamReader(stream, requestID), nil
}
