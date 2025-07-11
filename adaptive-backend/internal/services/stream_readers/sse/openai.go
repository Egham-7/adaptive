package sse

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/stream_readers"
	"encoding/json"
	"errors"
	"io"
	"strings"

	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go"
	ssestream "github.com/openai/openai-go/packages/ssestream"
)

const (
	sseDataPrefix  = "data: "
	sseLineSuffix  = "\n\n"
	sseDoneMessage = "data: [DONE]\n\n"
)


type OpenAIStreamReader struct {
	stream_readers.BaseStreamReader
	stream        *ssestream.Stream[openai.ChatCompletionChunk]
	done          bool
	buf           strings.Builder
	selectedModel string
	provider      string
}

func NewOpenAIStreamReader(
	stream *ssestream.Stream[openai.ChatCompletionChunk],
	requestID string,
	selectedModel string,
	provider string,
) *OpenAIStreamReader {
	r := &OpenAIStreamReader{
		BaseStreamReader: stream_readers.BaseStreamReader{
			Buffer:    make([]byte, 0, 1024),
			RequestID: requestID,
		},
		stream:        stream,
		selectedModel: selectedModel,
		provider:      provider,
	}
	r.buf.Grow(512)
	return r
}

func (r *OpenAIStreamReader) Read(p []byte) (n int, err error) {
	if len(r.Buffer) > 0 {
		n = copy(p, r.Buffer)
		r.Buffer = r.Buffer[n:]
		return n, nil
	}

	if r.done {
		return 0, io.EOF
	}

	// Check if we can advance to the next chunk
	hasNext := r.stream.Next()
	fiberlog.Debugf("[%s] Stream.Next() returned: %v", r.RequestID, hasNext)

	if !hasNext {
		// Check for stream errors
		if err := r.stream.Err(); err != nil && !errors.Is(err, io.EOF) {
			fiberlog.Errorf("[%s] Stream error detected: %v", r.RequestID, err)
			return r.handleError(err, p)
		}

		fiberlog.Infof("[%s] No more chunks available - sending [DONE]", r.RequestID)
		r.Buffer = []byte(sseDoneMessage)
		r.done = true
		return r.Read(p)
	}

	// Get and process the current chunk
	chunk := r.stream.Current()
	fiberlog.Debugf("[%s] Processing chunk: ID=%s, Choices=%d, Usage.TotalTokens=%d",
		r.RequestID, chunk.ID, len(chunk.Choices), chunk.Usage.TotalTokens)

	if err := r.processChunk(&chunk); err != nil {
		return r.handleError(err, p)
	}

	// Check if this is the final usage chunk
	if len(chunk.Choices) == 0 && chunk.Usage.TotalTokens > 0 {
		fiberlog.Infof("[%s] ✓ FINAL USAGE CHUNK DETECTED - TotalTokens=%d", r.RequestID, chunk.Usage.TotalTokens)
		r.done = true
	}

	return r.Read(p)
}

func (r *OpenAIStreamReader) handleError(err error, p []byte) (int, error) {
	fiberlog.Errorf("[%s] OpenAI stream error: %v", r.RequestID, err)

	r.buf.Reset()
	r.buf.WriteString(sseDataPrefix)
	r.buf.WriteString(`{"error": "`)
	r.buf.WriteString(strings.ReplaceAll(strings.ReplaceAll(err.Error(), "\"", "\\\""), "\n", "\\n"))
	r.buf.WriteString(`"}`)
	r.buf.WriteString(sseLineSuffix)

	r.Buffer = []byte(r.buf.String())
	r.done = true
	return r.Read(p)
}

// processChunk orchestrates the transformation and buffering of a chunk.
func (r *OpenAIStreamReader) processChunk(chunk *openai.ChatCompletionChunk) error {
	// Log original OpenAI chunk details
	fiberlog.Debugf("[%s] Original OpenAI chunk: ID=%s, Model=%s, Choices=%d",
		r.RequestID, chunk.ID, chunk.Model, len(chunk.Choices))

	// Log original choice details
	for i, choice := range chunk.Choices {
		fiberlog.Debugf("[%s] Original Choice[%d]: Role='%s', Content='%s', FinishReason='%s'",
			r.RequestID, i, choice.Delta.Role, choice.Delta.Content, choice.FinishReason)
	}

	// Convert OpenAI chunk to our adaptive chunk with cost savings
	adaptiveChunk := models.ConvertChunkToAdaptive(chunk, r.provider)

	jsonData, err := json.Marshal(adaptiveChunk)
	if err != nil {
		return err
	}

	// Log the actual JSON being sent
	fiberlog.Debugf("[%s] JSON output: %s", r.RequestID, string(jsonData))

	// Log chunk details
	fiberlog.Debugf("[%s] Outputting chunk: ID=%s, Model=%s, Choices=%d",
		r.RequestID, adaptiveChunk.ID, adaptiveChunk.Model, len(adaptiveChunk.Choices))

	// Log choice details if present
	for i, choice := range adaptiveChunk.Choices {
		fiberlog.Debugf("[%s] Choice[%d]: Role=%s, Content=%s, FinishReason=%s",
			r.RequestID, i, choice.Delta.Role, choice.Delta.Content, choice.FinishReason)
	}

	r.buf.Reset()
	r.buf.WriteString(sseDataPrefix)
	r.buf.Write(jsonData)
	r.buf.WriteString(sseLineSuffix)
	r.Buffer = []byte(r.buf.String())

	return nil
}

func (r *OpenAIStreamReader) Close() error {
	var err error
	r.CloseLock.Do(func() {
		r.done = true
		r.Buffer = nil
		if r.stream != nil {
			err = r.stream.Close()
		}
		fiberlog.Infof("[%s] OpenAI stream closed", r.RequestID)
	})
	return err
}
