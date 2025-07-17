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
	"github.com/valyala/bytebufferpool"
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
	selectedModel string
	provider      string
}

func NewOpenAIStreamReader(
	stream *ssestream.Stream[openai.ChatCompletionChunk],
	requestID string,
	selectedModel string,
	provider string,
) *OpenAIStreamReader {
	return &OpenAIStreamReader{
		BaseStreamReader: stream_readers.BaseStreamReader{
			Buffer:    make([]byte, 0, 2048), // Larger initial capacity
			RequestID: requestID,
		},
		stream:        stream,
		selectedModel: selectedModel,
		provider:      provider,
	}
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

	// Use bytebufferpool for error message building
	bb := bytebufferpool.Get()
	defer bytebufferpool.Put(bb)
	bb.Reset() // Clear any leftover data

	if _, writeErr := bb.WriteString(sseDataPrefix); writeErr != nil {
		fiberlog.Errorf("[%s] Failed to write SSE prefix: %v", r.RequestID, writeErr)
		return 0, writeErr
	}
	if _, writeErr := bb.WriteString(`{"error": "`); writeErr != nil {
		fiberlog.Errorf("[%s] Failed to write error prefix: %v", r.RequestID, writeErr)
		return 0, writeErr
	}
	if _, writeErr := bb.WriteString(strings.ReplaceAll(strings.ReplaceAll(err.Error(), "\"", "\\\""), "\n", "\\n")); writeErr != nil {
		fiberlog.Errorf("[%s] Failed to write error message: %v", r.RequestID, writeErr)
		return 0, writeErr
	}
	if _, writeErr := bb.WriteString(`"}`); writeErr != nil {
		fiberlog.Errorf("[%s] Failed to write error suffix: %v", r.RequestID, writeErr)
		return 0, writeErr
	}
	if _, writeErr := bb.WriteString(sseLineSuffix); writeErr != nil {
		fiberlog.Errorf("[%s] Failed to write SSE suffix: %v", r.RequestID, writeErr)
		return 0, writeErr
	}

	// Copy to our buffer efficiently
	if cap(r.Buffer) < bb.Len() {
		r.Buffer = make([]byte, bb.Len())
	}
	r.Buffer = r.Buffer[:bb.Len()]
	copy(r.Buffer, bb.B)

	r.done = true
	return r.Read(p)
}

// processChunk orchestrates the transformation and buffering of a chunk.
func (r *OpenAIStreamReader) processChunk(chunk *openai.ChatCompletionChunk) error {
	// Convert OpenAI chunk to our adaptive chunk with cost savings
	adaptiveChunk := models.ConvertChunkToAdaptive(chunk, r.provider)

	jsonData, err := json.Marshal(adaptiveChunk)
	if err != nil {
		return err
	}

	// Use bytebufferpool for efficient buffer management
	bb := bytebufferpool.Get()
	defer bytebufferpool.Put(bb)
	bb.Reset() // Clear any leftover data

	// Build SSE message efficiently
	if _, writeErr := bb.WriteString(sseDataPrefix); writeErr != nil {
		return writeErr
	}
	if _, writeErr := bb.Write(jsonData); writeErr != nil {
		return writeErr
	}
	if _, writeErr := bb.WriteString(sseLineSuffix); writeErr != nil {
		return writeErr
	}

	// Copy to our buffer efficiently
	if cap(r.Buffer) < bb.Len() {
		r.Buffer = make([]byte, bb.Len())
	}
	r.Buffer = r.Buffer[:bb.Len()]
	copy(r.Buffer, bb.B)

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
