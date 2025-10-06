package readers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"sync"

	"adaptive-backend/internal/utils"

	"github.com/openai/openai-go/v2"
	ssestream "github.com/openai/openai-go/v2/packages/ssestream"
	"github.com/valyala/bytebufferpool"
)

// OpenAIStreamReader provides pure I/O reading from OpenAI streams
// This reader ONLY reads raw chunk data - no format conversion
type OpenAIStreamReader struct {
	stream     *ssestream.Stream[openai.ChatCompletionChunk]
	buffer     *bytebufferpool.ByteBuffer
	done       bool
	doneMux    sync.RWMutex
	requestID  string
	closeOnce  sync.Once
	firstChunk *openai.ChatCompletionChunk // Cached first chunk to replay
	firstRead  bool                        // Track if first chunk has been read
}

// NewOpenAIStreamReader creates a new OpenAI stream reader
// Validates stream by reading first chunk, returns error if stream is invalid
func NewOpenAIStreamReader(
	stream *ssestream.Stream[openai.ChatCompletionChunk],
	requestID string,
) (*OpenAIStreamReader, error) {
	// Validate stream by trying to get first chunk
	// This detects provider errors (429, 500, etc.) before starting HTTP stream
	if !stream.Next() {
		if err := stream.Err(); err != nil {
			return nil, err
		}
		return nil, errors.New("empty stream from provider")
	}

	// Get the first chunk - we'll replay it in Read()
	firstChunk := stream.Current()

	return &OpenAIStreamReader{
		stream:     stream,
		buffer:     utils.Get(), // Get buffer from pool
		requestID:  requestID,
		firstChunk: &firstChunk,
		firstRead:  false,
	}, nil
}

// Read implements io.Reader - pure I/O operation
func (r *OpenAIStreamReader) Read(p []byte) (n int, err error) {
	// Fast path: return buffered data first
	if len(r.buffer.B) > 0 {
		n = copy(p, r.buffer.B)
		r.buffer.B = r.buffer.B[n:]
		return n, nil
	}

	// Check if stream is done
	r.doneMux.RLock()
	done := r.done
	r.doneMux.RUnlock()

	if done {
		return 0, io.EOF
	}

	var chunk openai.ChatCompletionChunk

	// If we have a first chunk and haven't read it yet, use it
	if r.firstChunk != nil && !r.firstRead {
		chunk = *r.firstChunk
		r.firstRead = true
	} else {
		// Try to read next chunk from stream
		if !r.stream.Next() {
			// Handle stream termination
			if streamErr := r.stream.Err(); streamErr != nil {
				if errors.Is(streamErr, io.EOF) {
					r.setDone()
					return 0, io.EOF
				}
				// Treat context cancellation as normal termination (client disconnect)
				if errors.Is(streamErr, context.Canceled) || errors.Is(streamErr, context.DeadlineExceeded) {
					r.setDone()
					return 0, io.EOF
				}
				return 0, streamErr
			}
			// No chunks available, return 0 to indicate retry
			return 0, nil
		}

		// Get current chunk from stream
		chunk = r.stream.Current()
	}

	// Serialize complete chunk for processor layer
	chunkData, err := json.Marshal(&chunk)
	if err != nil {
		return 0, err
	}

	// Buffer the data
	r.buffer.B = append(r.buffer.B[:0], chunkData...)

	// Check for completion
	if r.hasFinishReason(&chunk) {
		r.setDone()
	}

	// Return data from buffer
	n = copy(p, r.buffer.B)
	r.buffer.B = r.buffer.B[n:]
	return n, nil
}

// Close implements io.Closer
func (r *OpenAIStreamReader) Close() error {
	var err error
	r.closeOnce.Do(func() {
		r.setDone()
		if r.stream != nil {
			err = r.stream.Close()
		}
		if r.buffer != nil {
			utils.Put(r.buffer) // Return buffer to pool
			r.buffer = nil
		}
	})
	return err
}

// setDone marks the stream as done (thread-safe)
func (r *OpenAIStreamReader) setDone() {
	r.doneMux.Lock()
	r.done = true
	r.doneMux.Unlock()
}

// hasFinishReason checks if chunk indicates completion
func (r *OpenAIStreamReader) hasFinishReason(chunk *openai.ChatCompletionChunk) bool {
	return len(chunk.Choices) > 0 && chunk.Choices[0].FinishReason != ""
}
