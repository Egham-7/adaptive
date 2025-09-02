package sse

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"sync"
	"time"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/format_adapter"
	"adaptive-backend/internal/services/stream_readers"

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

// OpenAIError represents the nested error structure in OpenAI responses
type OpenAIError struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

// StreamError represents an OpenAI-compatible stream error response
type StreamError struct {
	ErrorDetails OpenAIError `json:"error"`
}

func (se *StreamError) Error() string {
	return se.ErrorDetails.Message
}

type OpenAIStreamReader struct {
	stream_readers.BaseStreamReader
	stream      *ssestream.Stream[openai.ChatCompletionChunk]
	done        bool
	doneMux     sync.RWMutex
	ctx         context.Context
	provider    string
	cacheSource string
}

func NewOpenAIStreamReader(
	stream *ssestream.Stream[openai.ChatCompletionChunk],
	requestID string,
	provider string,
	cacheSource string,
) *OpenAIStreamReader {
	return &OpenAIStreamReader{
		BaseStreamReader: stream_readers.BaseStreamReader{
			Buffer:    make([]byte, 0, initialBufferSize),
			RequestID: requestID,
		},
		provider:    provider,
		cacheSource: cacheSource,
		stream:      stream,
		ctx:         context.Background(), // Default context, will be overridden
	}
}

// SetContext allows setting the context for cancellation
func (r *OpenAIStreamReader) SetContext(ctx context.Context) {
	r.ctx = ctx
}

// Read implements io.Reader interface for compatibility with stream processing
// Read implements io.Reader interface for compatibility with stream processing
func (r *OpenAIStreamReader) Read(p []byte) (n int, err error) {
	// Fast path: return buffered data first (common case optimization)
	if len(r.Buffer) > 0 {
		n = copy(p, r.Buffer)
		r.Buffer = r.Buffer[n:]
		return n, nil
	}

	// Check context cancellation (non-blocking select for performance)
	select {
	case <-r.ctx.Done():
		fiberlog.Infof("[%s] Context cancelled, stopping OpenAI stream", r.RequestID)
		if r.stream != nil {
			_ = r.stream.Close() // Best effort, ignore error
			r.stream = nil
		}
		return 0, r.ctx.Err()
	default:
	}

	// Single RLock for performance (avoid multiple lock/unlock cycles)
	r.doneMux.RLock()
	done := r.done
	r.doneMux.RUnlock()

	if done {
		return 0, io.EOF
	}

	// Try to advance to next chunk
	if !r.stream.Next() {
		// Handle stream termination conditions
		if streamErr := r.stream.Err(); streamErr != nil {
			// Explicit error - handle appropriately
			if errors.Is(streamErr, io.EOF) {
				fiberlog.Infof("[%s] Stream completed normally (EOF)", r.RequestID)
				r.setDoneAndBuffer()
				return copy(p, r.Buffer), nil
			}
			// Other errors - convert to SSE error and terminate
			return r.handleError(streamErr, p)
		}
		// No error, no chunks - temporary pause, return 0 to retry
		fiberlog.Debugf("[%s] No chunks available, will retry", r.RequestID)
		return 0, nil
	}

	// Get current chunk (no allocation)
	chunk := r.stream.Current()

	// Log chunk details for debugging
	r.logChunkDetails(&chunk)

	// Process chunk into buffer
	if err = r.processChunk(&chunk); err != nil {
		return r.handleError(err, p)
	}

	// Check for completion after successful processing
	if r.hasFinishReason(&chunk) {
		finishReason := string(chunk.Choices[0].FinishReason)
		fiberlog.Infof("[%s] Stream completed with finish_reason: %s", r.RequestID, finishReason)
		r.doneMux.Lock()
		r.done = true
		r.doneMux.Unlock()
	}

	// Return processed data from buffer
	n = copy(p, r.Buffer)
	r.Buffer = r.Buffer[n:]
	return n, nil
}

// setDoneAndBuffer atomically sets done flag and DONE message
func (r *OpenAIStreamReader) setDoneAndBuffer() {
	r.Buffer = []byte(sseDoneMessage)
	r.doneMux.Lock()
	r.done = true
	r.doneMux.Unlock()
}

// hasFinishReason checks if chunk has completion indicator
func (r *OpenAIStreamReader) hasFinishReason(chunk *openai.ChatCompletionChunk) bool {
	return len(chunk.Choices) > 0 && chunk.Choices[0].FinishReason != ""
}

// logChunkDetails logs chunk information (only called when debug logging enabled)
func (r *OpenAIStreamReader) logChunkDetails(chunk *openai.ChatCompletionChunk) {
	var content string
	if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
		c := chunk.Choices[0].Delta.Content
		if len(c) > 50 {
			content = c[:47] + "..."
		} else {
			content = c
		}
	}

	var finishReason string
	if len(chunk.Choices) > 0 {
		finishReason = string(chunk.Choices[0].FinishReason)
	} else {
		finishReason = "none"
	}

	fiberlog.Debugf("[%s] Processing chunk: ID=%s, Choices=%d, Content='%s', FinishReason=%s, Usage.TotalTokens=%d",
		r.RequestID, chunk.ID, len(chunk.Choices), content, finishReason, chunk.Usage.TotalTokens)
}

func (r *OpenAIStreamReader) handleError(err error, p []byte) (int, error) {
	fiberlog.Debugf("[%s] OpenAI stream error: %v", r.RequestID, err)

	// Create structured error response
	var errorResponse *StreamError
	if streamErr, ok := err.(*StreamError); ok {
		errorResponse = streamErr
	} else {
		errorResponse = &StreamError{
			ErrorDetails: OpenAIError{
				Message: err.Error(),
				Type:    "stream_error",
			},
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
		// Calculate correct truncation point to leave room for SSE suffix
		suffix := []byte(sseLineSuffix)
		truncateAt := maxBufferSize - len(suffix)
		copy(r.Buffer, bb.B[:truncateAt])
		// Ensure we end with proper SSE format
		copy(r.Buffer[truncateAt:], suffix)
	} else {
		if cap(r.Buffer) < bb.Len() {
			r.Buffer = make([]byte, bb.Len())
		}
		r.Buffer = r.Buffer[:bb.Len()]
		copy(r.Buffer, bb.B)
	}

	// Return the data directly instead of recursive call
	n := copy(p, r.Buffer)
	r.Buffer = r.Buffer[n:]
	return n, nil
}

// processChunk orchestrates the transformation and buffering of a chunk.
func (r *OpenAIStreamReader) processChunk(chunk *openai.ChatCompletionChunk) error {
	startTime := time.Now()

	// Convert OpenAI chunk to our adaptive chunk with cost savings
	adaptiveChunk, err := format_adapter.OpenAIToAdaptive.ConvertStreamingChunk(chunk, r.provider)
	if err != nil {
		fiberlog.Errorf("Failed to convert streaming chunk: %v", err)
		return err
	}

	// Set cache tier if usage is present and cache source is available
	if adaptiveChunk.Usage != nil && r.cacheSource != "" {
		models.SetCacheTier(adaptiveChunk.Usage, r.cacheSource)
	}
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
	if _, err := jsonBB.Write(jsonData); err != nil {
		fiberlog.Errorf("[%s] Failed to write JSON to buffer: %v", r.RequestID, err)
		return err
	}
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
		// Calculate correct truncation point to leave room for SSE suffix
		suffix := []byte("\n\n")
		truncateAt := maxBufferSize - len(suffix)
		copy(r.Buffer, bb.B[:truncateAt])
		// Ensure we end with proper SSE format
		copy(r.Buffer[truncateAt:], suffix)
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
