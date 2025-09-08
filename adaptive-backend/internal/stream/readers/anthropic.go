package readers

import (
	"encoding/json"
	"io"
	"sync"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
)

// AnthropicStreamReader provides pure I/O reading from Anthropic streams
type AnthropicStreamReader struct {
	reader    io.Reader
	closer    io.Closer
	requestID string
	closeOnce sync.Once
}

// NewAnthropicStreamReader creates a new Anthropic stream reader
func NewAnthropicStreamReader(reader io.Reader, requestID string) *AnthropicStreamReader {
	var closer io.Closer
	if readCloser, ok := reader.(io.ReadCloser); ok {
		closer = readCloser
	}

	return &AnthropicStreamReader{
		reader:    reader,
		closer:    closer,
		requestID: requestID,
	}
}

// Read implements io.Reader - delegates to underlying reader
func (r *AnthropicStreamReader) Read(p []byte) (n int, err error) {
	return r.reader.Read(p)
}

// Close implements io.Closer
func (r *AnthropicStreamReader) Close() error {
	var err error
	r.closeOnce.Do(func() {
		if r.closer != nil {
			err = r.closer.Close()
		}
	})
	return err
}

// AnthropicNativeStreamReader wraps native Anthropic SDK streams
type AnthropicNativeStreamReader struct {
	stream    *ssestream.Stream[anthropic.MessageStreamEventUnion]
	buffer    []byte
	requestID string
	closeOnce sync.Once
}

// NewAnthropicNativeStreamReader creates a new native Anthropic stream reader
func NewAnthropicNativeStreamReader(stream *ssestream.Stream[anthropic.MessageStreamEventUnion], requestID string) *AnthropicNativeStreamReader {
	return &AnthropicNativeStreamReader{
		stream:    stream,
		buffer:    make([]byte, 0, 4096),
		requestID: requestID,
	}
}

// Read implements io.Reader
func (r *AnthropicNativeStreamReader) Read(p []byte) (n int, err error) {
	// Return buffered data first
	if len(r.buffer) > 0 {
		n = copy(p, r.buffer)
		r.buffer = r.buffer[n:]
		return n, nil
	}

	// Try to get next event
	if !r.stream.Next() {
		if err := r.stream.Err(); err != nil {
			return 0, err
		}
		return 0, io.EOF
	}

	// Get current event and serialize
	event := r.stream.Current()
	eventData, err := json.Marshal(&event)
	if err != nil {
		return 0, err
	}

	// Buffer the data
	r.buffer = append(r.buffer[:0], eventData...)

	// Return data
	n = copy(p, r.buffer)
	r.buffer = r.buffer[n:]
	return n, nil
}

// Close implements io.Closer
func (r *AnthropicNativeStreamReader) Close() error {
	var err error
	r.closeOnce.Do(func() {
		if r.stream != nil {
			err = r.stream.Close()
		}
		r.buffer = nil
	})
	return err
}
