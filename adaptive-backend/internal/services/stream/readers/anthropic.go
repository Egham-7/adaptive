package readers

import (
	"encoding/json"
	"io"
	"sync"

	"adaptive-backend/internal/utils"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/valyala/bytebufferpool"
)

// AnthropicNativeStreamReader wraps native Anthropic SDK streams
type AnthropicNativeStreamReader struct {
	stream     *ssestream.Stream[anthropic.MessageStreamEventUnion]
	buffer     *bytebufferpool.ByteBuffer
	requestID  string
	closeOnce  sync.Once
	firstEvent *anthropic.MessageStreamEventUnion // Cached first event to replay
	firstRead  bool                               // Track if first event has been read
}

// NewAnthropicNativeStreamReader creates a new native Anthropic stream reader
// Validates stream by reading first event
func NewAnthropicNativeStreamReader(stream *ssestream.Stream[anthropic.MessageStreamEventUnion], requestID string) (*AnthropicNativeStreamReader, error) {
	// Validate stream by trying to get first event
	if !stream.Next() {
		if err := stream.Err(); err != nil {
			return nil, err
		}
		return nil, io.EOF
	}

	// Get the first event - we'll replay it in Read()
	firstEvent := stream.Current()

	return &AnthropicNativeStreamReader{
		stream:     stream,
		buffer:     utils.Get(), // Get buffer from pool
		requestID:  requestID,
		firstEvent: &firstEvent,
		firstRead:  false,
	}, nil
}

// Read implements io.Reader
func (r *AnthropicNativeStreamReader) Read(p []byte) (n int, err error) {
	// Return buffered data first
	if len(r.buffer.B) > 0 {
		n = copy(p, r.buffer.B)
		r.buffer.B = r.buffer.B[n:]
		return n, nil
	}

	var event anthropic.MessageStreamEventUnion

	// If we have a first event and haven't read it yet, use it
	if r.firstEvent != nil && !r.firstRead {
		event = *r.firstEvent
		r.firstRead = true
	} else {
		// Try to get next event
		if !r.stream.Next() {
			if err := r.stream.Err(); err != nil {
				return 0, err
			}
			return 0, io.EOF
		}

		// Get current event from stream
		event = r.stream.Current()
	}

	// Serialize event
	eventData, err := json.Marshal(&event)
	if err != nil {
		return 0, err
	}

	// Buffer the data
	r.buffer.B = append(r.buffer.B[:0], eventData...)

	// Return data
	n = copy(p, r.buffer.B)
	r.buffer.B = r.buffer.B[n:]
	return n, nil
}

// Close implements io.Closer
func (r *AnthropicNativeStreamReader) Close() error {
	var err error
	r.closeOnce.Do(func() {
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
