package readers

import (
	"encoding/json"
	"io"
	"iter"
	"sync"

	"google.golang.org/genai"
)

// GeminiStreamReader provides pure I/O reading from Gemini streams
// This reader ONLY reads raw chunk data - no format conversion
type GeminiStreamReader struct {
	iterator  iter.Seq2[*genai.GenerateContentResponse, error]
	buffer    []byte
	done      bool
	doneMux   sync.RWMutex
	requestID string
	closeOnce sync.Once
	next      func() (*genai.GenerateContentResponse, error, bool)
	stop      func()
}

// NewGeminiStreamReader creates a new Gemini stream reader
func NewGeminiStreamReader(
	streamIter iter.Seq2[*genai.GenerateContentResponse, error],
	requestID string,
) *GeminiStreamReader {
	reader := &GeminiStreamReader{
		iterator:  streamIter,
		buffer:    make([]byte, 0, 4096), // 4KB initial buffer
		requestID: requestID,
	}

	// Set up iterator function
	reader.setupIterator()

	return reader
}

// setupIterator sets up the iterator function and stop function
func (r *GeminiStreamReader) setupIterator() {
	stopCh := make(chan struct{})
	r.stop = func() {
		close(stopCh)
	}

	r.next = func() (*genai.GenerateContentResponse, error, bool) {
		for resp, err := range r.iterator {
			select {
			case <-stopCh:
				return nil, io.EOF, false
			default:
				if err != nil {
					return nil, err, false
				}
				if resp != nil {
					return resp, nil, true
				}
			}
		}
		return nil, io.EOF, false
	}
}

// Read implements io.Reader - pure I/O operation
func (r *GeminiStreamReader) Read(p []byte) (n int, err error) {
	// Fast path: return buffered data first
	if len(r.buffer) > 0 {
		n = copy(p, r.buffer)
		r.buffer = r.buffer[n:]
		return n, nil
	}

	// Check if stream is done
	r.doneMux.RLock()
	done := r.done
	r.doneMux.RUnlock()

	if done {
		return 0, io.EOF
	}

	// Read next chunk from Gemini stream
	chunk, err, hasNext := r.next()
	if !hasNext || err != nil {
		r.doneMux.Lock()
		r.done = true
		r.doneMux.Unlock()

		if err != nil && err != io.EOF {
			return 0, err
		}
		return 0, io.EOF
	}

	// Marshal chunk to JSON for SSE format (matching Gemini's native format)
	chunkData, err := json.Marshal(chunk)
	if err != nil {
		return 0, err
	}

	// Format as SSE data (matching Gemini API format: "data: {...}\n\n")
	sseData := append([]byte("data: "), chunkData...)
	sseData = append(sseData, '\n', '\n')

	// Copy to output buffer
	n = copy(p, sseData)
	if n < len(sseData) {
		// Buffer remaining data for next read
		r.buffer = append(r.buffer, sseData[n:]...)
	}

	return n, nil
}

// Close implements io.Closer
func (r *GeminiStreamReader) Close() error {
	var closeErr error
	r.closeOnce.Do(func() {
		r.doneMux.Lock()
		r.done = true
		r.doneMux.Unlock()

		if r.stop != nil {
			r.stop()
		}
	})
	return closeErr
}
