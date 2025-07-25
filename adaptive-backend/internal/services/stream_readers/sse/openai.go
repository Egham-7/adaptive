package sse

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/stream_readers"
	"context"
	"encoding/json"
	"errors"
	"io"
	"sync"
	"time"

	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go"
	ssestream "github.com/openai/openai-go/packages/ssestream"
	"github.com/valyala/bytebufferpool"
)

const (
	sseDataPrefix     = "data: "
	sseLineSuffix     = "\n\n"
	sseDoneMessage    = "data: [DONE]\n\n"
	maxBufferSize     = 64 * 1024 // 64KB max buffer size
	initialBufferSize = 4096      // 4KB initial buffer - better for SSE chunks
	chunkBufferSize   = 8192      // 8KB chunk processing buffer
)

// StreamError represents a structured stream error
type StreamError struct {
	Type      string `json:"type"`
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
}

func (e *StreamError) Error() string {
	return e.Message
}

type OpenAIStreamReader struct {
	stream_readers.BaseStreamReader
	stream        *ssestream.Stream[openai.ChatCompletionChunk]
	done          bool
	doneMux       sync.RWMutex
	ctx           context.Context
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
			Buffer:    make([]byte, 0, initialBufferSize),
			RequestID: requestID,
		},
		stream:        stream,
		ctx:           context.Background(), // Default context, will be overridden
		selectedModel: selectedModel,
		provider:      provider,
	}
}

// SetContext allows setting the context for cancellation
func (r *OpenAIStreamReader) SetContext(ctx context.Context) {
	r.ctx = ctx
}

func (r *OpenAIStreamReader) Read(p []byte) (n int, err error) {
	// Check for context cancellation
	select {
	case <-r.ctx.Done():
		fiberlog.Infof("[%s] Context cancelled, stopping OpenAI stream", r.RequestID)
		return 0, r.ctx.Err()
	default:
	}

	if len(r.Buffer) > 0 {
		n = copy(p, r.Buffer)
		r.Buffer = r.Buffer[n:]
		return n, nil
	}

	r.doneMux.RLock()
	done := r.done
	r.doneMux.RUnlock()

	if done {
		return 0, io.EOF
	}

	// Check if we can advance to the next chunk
	hasNext := r.stream.Next()
	fiberlog.Debugf("[%s] Stream.Next() returned: %v", r.RequestID, hasNext)

	if !hasNext {
		// Check for stream errors
		if err := r.stream.Err(); err != nil {
			if errors.Is(err, io.EOF) {
				fiberlog.Infof("[%s] Stream completed normally", r.RequestID)
			} else {
				fiberlog.Errorf("[%s] Stream error detected: %v", r.RequestID, err)
				streamErr := &StreamError{
					Type:      "upstream_error",
					Message:   err.Error(),
					RequestID: r.RequestID,
				}
				return r.handleError(streamErr, p)
			}
		} else {
			fiberlog.Infof("[%s] No more chunks available - normal completion", r.RequestID)
		}

		r.Buffer = []byte(sseDoneMessage)
		r.doneMux.Lock()
		r.done = true
		r.doneMux.Unlock()
		return r.Read(p)
	}

	// Get and process the current chunk
	chunk := r.stream.Current()
	
	// Detailed chunk logging
	var choiceContent string
	if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
		content := chunk.Choices[0].Delta.Content
		if len(content) > 50 {
			choiceContent = content[:47] + "..."
		} else {
			choiceContent = content
		}
	}
	
	fiberlog.Debugf("[%s] Processing chunk: ID=%s, Choices=%d, Content='%s', FinishReason=%v, Usage.TotalTokens=%d",
		r.RequestID, chunk.ID, len(chunk.Choices), choiceContent, 
		func() string { if len(chunk.Choices) > 0 { return string(chunk.Choices[0].FinishReason) }; return "none" }(),
		chunk.Usage.TotalTokens)

	if err := r.processChunk(&chunk); err != nil {
		fiberlog.Errorf("[%s] Chunk processing failed for chunk ID=%s: %v", r.RequestID, chunk.ID, err)
		streamErr := &StreamError{
			Type:      "chunk_processing_error",
			Message:   err.Error(),
			RequestID: r.RequestID,
		}
		return r.handleError(streamErr, p)
	}

	return r.Read(p)
}

func (r *OpenAIStreamReader) handleError(err error, p []byte) (int, error) {
	fiberlog.Errorf("[%s] OpenAI stream error: %v", r.RequestID, err)

	// Create structured error response
	var errorResponse *StreamError
	if streamErr, ok := err.(*StreamError); ok {
		errorResponse = streamErr
	} else {
		errorResponse = &StreamError{
			Type:      "stream_error",
			Message:   err.Error(),
			RequestID: r.RequestID,
		}
	}

	errorJSON, jsonErr := json.Marshal(errorResponse)
	if jsonErr != nil {
		fiberlog.Errorf("[%s] Failed to marshal error JSON: %v", r.RequestID, jsonErr)
		return 0, jsonErr
	}

	// Use optimized buffer for error message building
	bb := bytebufferpool.Get()
	defer bytebufferpool.Put(bb)
	bb.Reset()
	// Pre-allocate expected error message size
	if cap(bb.B) < chunkBufferSize {
		bb.B = make([]byte, 0, chunkBufferSize)
	}

	if _, writeErr := bb.WriteString(sseDataPrefix); writeErr != nil {
		fiberlog.Errorf("[%s] Failed to write SSE prefix: %v", r.RequestID, writeErr)
		return 0, writeErr
	}
	if _, writeErr := bb.Write(errorJSON); writeErr != nil {
		fiberlog.Errorf("[%s] Failed to write error JSON: %v", r.RequestID, writeErr)
		return 0, writeErr
	}
	if _, writeErr := bb.WriteString(sseLineSuffix); writeErr != nil {
		fiberlog.Errorf("[%s] Failed to write SSE suffix: %v", r.RequestID, writeErr)
		return 0, writeErr
	}

	// Copy to our buffer efficiently with size limit
	if bb.Len() > maxBufferSize {
		fiberlog.Warnf("[%s] Error message too large (%d bytes), truncating to %d bytes", r.RequestID, bb.Len(), maxBufferSize)
		if cap(r.Buffer) < maxBufferSize {
			r.Buffer = make([]byte, maxBufferSize)
		}
		r.Buffer = r.Buffer[:maxBufferSize]
		copy(r.Buffer, bb.B[:maxBufferSize])
	} else {
		if cap(r.Buffer) < bb.Len() {
			r.Buffer = make([]byte, bb.Len())
		}
		r.Buffer = r.Buffer[:bb.Len()]
		copy(r.Buffer, bb.B)
	}

	return r.Read(p)
}

// processChunk orchestrates the transformation and buffering of a chunk.
func (r *OpenAIStreamReader) processChunk(chunk *openai.ChatCompletionChunk) error {
	startTime := time.Now()
	
	// Convert OpenAI chunk to our adaptive chunk with cost savings
	adaptiveChunk := models.ConvertChunkToAdaptive(chunk, r.provider)
	conversionTime := time.Since(startTime)

	// Use buffer pool for JSON marshaling
	jsonBB := bytebufferpool.Get()
	defer bytebufferpool.Put(jsonBB)
	jsonBB.Reset()
	
	// Marshal JSON directly to buffer pool
	marshalStart := time.Now()
	jsonData, err := json.Marshal(adaptiveChunk)
	if err != nil {
		fiberlog.Errorf("[%s] JSON marshaling failed for chunk: %v", r.RequestID, err)
		return err
	}
	jsonBB.Write(jsonData)
	marshalTime := time.Since(marshalStart)

	// Use bytebufferpool for efficient SSE message building  
	bb := bytebufferpool.Get()
	defer bytebufferpool.Put(bb)
	bb.Reset()
	// Pre-allocate expected chunk size to avoid reallocations
	if cap(bb.B) < chunkBufferSize {
		bb.B = make([]byte, 0, chunkBufferSize)
	}

	// Build SSE message efficiently
	if _, writeErr := bb.WriteString(sseDataPrefix); writeErr != nil {
		return writeErr
	}
	if _, writeErr := bb.Write(jsonBB.B); writeErr != nil {
		return writeErr
	}
	if _, writeErr := bb.WriteString(sseLineSuffix); writeErr != nil {
		return writeErr
	}

	// Copy to our buffer efficiently with size limit
	bufferStart := time.Now()
	if bb.Len() > maxBufferSize {
		fiberlog.Warnf("[%s] Chunk too large (%d bytes), truncating to %d bytes", r.RequestID, bb.Len(), maxBufferSize)
		if cap(r.Buffer) < maxBufferSize {
			r.Buffer = make([]byte, maxBufferSize)
		}
		r.Buffer = r.Buffer[:maxBufferSize]
		copy(r.Buffer, bb.B[:maxBufferSize])
		// Ensure we end with proper SSE format
		copy(r.Buffer[maxBufferSize-4:], []byte("\n\n"))
	} else {
		if cap(r.Buffer) < bb.Len() {
			r.Buffer = make([]byte, bb.Len())
		}
		r.Buffer = r.Buffer[:bb.Len()]
		copy(r.Buffer, bb.B)
	}
	bufferTime := time.Since(bufferStart)
	
	totalTime := time.Since(startTime)
	fiberlog.Debugf("[%s] Chunk processed: json_size=%d, sse_size=%d, conversion=%v, marshal=%v, buffer=%v, total=%v",
		r.RequestID, len(jsonData), bb.Len(), conversionTime, marshalTime, bufferTime, totalTime)

	return nil
}

func (r *OpenAIStreamReader) Close() error {
	var err error
	r.CloseLock.Do(func() {
		closeStart := time.Now()
		r.doneMux.Lock()
		r.done = true
		r.doneMux.Unlock()
		r.Buffer = nil
		if r.stream != nil {
			err = r.stream.Close()
			if err != nil {
				fiberlog.Errorf("[%s] Error closing OpenAI stream: %v", r.RequestID, err)
			}
		}
		closeDuration := time.Since(closeStart)
		fiberlog.Infof("[%s] OpenAI stream closed in %v", r.RequestID, closeDuration)
	})
	return err
}
