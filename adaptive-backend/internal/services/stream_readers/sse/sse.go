package sse

import (
	"adaptive-backend/internal/services/stream_readers"
	"context"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

func GetSSEStreamReader(ctx context.Context, stream *ssestream.Stream[openai.ChatCompletionChunk], requestID string, provider string) (stream_readers.StreamReader, error) {
	reader := NewOpenAIStreamReader(stream, requestID, provider)
	reader.SetContext(ctx)
	return reader, nil
}
