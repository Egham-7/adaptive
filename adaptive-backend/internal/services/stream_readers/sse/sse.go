package sse

import (
	"context"

	"adaptive-backend/internal/services/stream_readers"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

func GetSSEStreamReader(ctx context.Context, stream *ssestream.Stream[openai.ChatCompletionChunk], requestID string, provider string, cacheSource string) (stream_readers.StreamReader, error) {
	reader := NewOpenAIStreamReader(stream, requestID, provider, cacheSource)
	reader.SetContext(ctx)
	return reader, nil
}
