package openai

import (
	"encoding/json"
	"fmt"
	"sync"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

// StreamDecoder implements the OpenAI stream interface for any provider
type StreamDecoder struct {
	currentChunk *openai.ChatCompletionChunk // Current chunk after Next() returns true
	chunkChan    chan openai.ChatCompletionChunk
	errorChan    chan error
	done         chan struct{}
	closed       bool
	mu           sync.RWMutex
	lastError    error
}

// NewStreamDecoder creates a new stream decoder
func NewStreamDecoder() *StreamDecoder {
	return &StreamDecoder{
		chunkChan: make(chan openai.ChatCompletionChunk, 10),
		errorChan: make(chan error, 1),
		done:      make(chan struct{}),
	}
}

// SendChunk sends a chunk to the stream
func (r *StreamDecoder) SendChunk(chunk openai.ChatCompletionChunk) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.closed {
		return false
	}

	select {
	case r.chunkChan <- chunk:
		return true
	case <-r.done:
		return false
	}
}

// SendError sends an error to the stream
func (r *StreamDecoder) SendError(err error) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.closed {
		return false
	}
	r.lastError = err

	select {
	case r.errorChan <- err:
		return true
	case <-r.done:
		return false
	}
}

// CloseSender closes the sender channels (call from producer goroutine)
func (r *StreamDecoder) CloseSender() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.closed {
		close(r.chunkChan)
		close(r.errorChan)
	}
}

// Read implements io.Reader for StreamDecoder
func (r *StreamDecoder) Read(p []byte) (n int, err error) {
	return 0, fmt.Errorf("Read method not implemented for streaming")
}

// Next implements the Decoder interface for StreamDecoder
func (r *StreamDecoder) Next() bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return false
	}

	select {
	case chunk, ok := <-r.chunkChan:
		if ok {
			r.currentChunk = &chunk
		}
		return ok
	case <-r.done:
		return false
	}
}

// Event implements the Decoder interface for StreamDecoder
func (r *StreamDecoder) Event() ssestream.Event {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.currentChunk == nil {
		return ssestream.Event{}
	}

	// Marshal the current chunk to JSON for the event data
	data, err := json.Marshal(r.currentChunk)
	if err != nil {
		return ssestream.Event{}
	}

	return ssestream.Event{
		Data: data,
		Type: "completion",
	}
}

// Close implements the Decoder interface for StreamDecoder
func (r *StreamDecoder) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.closed {
		r.closed = true
		close(r.done)
	}
	return nil
}

// Err implements the Decoder interface for StreamDecoder
func (r *StreamDecoder) Err() error {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.lastError
}

// Chunk returns the current chunk after Next() returns true
func (r *StreamDecoder) Chunk() *openai.ChatCompletionChunk {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.currentChunk
}
