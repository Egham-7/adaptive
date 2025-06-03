package vercel

import (
	"adaptive-backend/internal/services/stream_readers"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

func GetDataStreamReader(stream *ssestream.Stream[openai.ChatCompletionChunk], provider string, requestID string) (stream_readers.StreamReader, error) {
	return NewVercelDataStreamReaderFromAdapter(NewOpenAIAdapter(stream, requestID), requestID), nil
}
