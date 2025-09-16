package writers

import (
	"bufio"

	"adaptive-backend/internal/services/stream/contracts"

	"github.com/valyala/fasthttp"
)

// HTTPStreamWriter handles HTTP streaming output with connection management
type HTTPStreamWriter struct {
	writer     *bufio.Writer
	connState  contracts.ConnectionState
	requestID  string
	totalBytes int64
}

// NewHTTPStreamWriter creates a new HTTP stream writer
func NewHTTPStreamWriter(writer *bufio.Writer, connState contracts.ConnectionState, requestID string) *HTTPStreamWriter {
	return &HTTPStreamWriter{
		writer:    writer,
		connState: connState,
		requestID: requestID,
	}
}

// Write writes data to the HTTP stream
func (w *HTTPStreamWriter) Write(data []byte) error {
	if len(data) == 0 {
		return nil
	}

	// Check connection state
	if !w.connState.IsConnected() {
		return contracts.NewClientDisconnectError(w.requestID)
	}

	// Write data
	n, err := w.writer.Write(data)
	if n > 0 {
		// Account for actual bytes written, even on partial write or error
		w.totalBytes += int64(n)
	}

	if err != nil {
		if contracts.IsConnectionClosed(err) {
			return contracts.NewClientDisconnectError(w.requestID)
		}
		return contracts.NewInternalError(w.requestID, "write failed", err)
	}

	return nil
}

// Flush flushes buffered data
func (w *HTTPStreamWriter) Flush() error {
	// Check connection state before flushing
	if !w.connState.IsConnected() {
		return contracts.NewClientDisconnectError(w.requestID)
	}

	if err := w.writer.Flush(); err != nil {
		if contracts.IsConnectionClosed(err) {
			return contracts.NewClientDisconnectError(w.requestID)
		}
		return contracts.NewInternalError(w.requestID, "flush failed", err)
	}

	return nil
}

// Close closes the writer
func (w *HTTPStreamWriter) Close() error {
	// Write completion message
	if w.connState.IsConnected() {
		n, writeErr := w.writer.WriteString("data: [DONE]\n\n")
		if writeErr == nil {
			// Add written bytes to total if write succeeded
			w.totalBytes += int64(n)

			// Flush and capture any error
			flushErr := w.writer.Flush()
			return flushErr
		}
		return writeErr
	}
	return nil
}

// TotalBytes returns total bytes written
func (w *HTTPStreamWriter) TotalBytes() int64 {
	return w.totalBytes
}

// FastHTTPConnectionState wraps FastHTTP context for connection state
type FastHTTPConnectionState struct {
	ctx *fasthttp.RequestCtx
}

// NewFastHTTPConnectionState creates connection state from FastHTTP context
func NewFastHTTPConnectionState(ctx *fasthttp.RequestCtx) *FastHTTPConnectionState {
	return &FastHTTPConnectionState{ctx: ctx}
}

// IsConnected checks if client is still connected
func (c *FastHTTPConnectionState) IsConnected() bool {
	if c.ctx == nil {
		return false
	}
	select {
	case <-c.ctx.Done():
		return false
	default:
		return true
	}
}

// Done returns channel that closes when client disconnects
func (c *FastHTTPConnectionState) Done() <-chan struct{} {
	if c.ctx == nil {
		// Return closed channel
		done := make(chan struct{})
		close(done)
		return done
	}
	return c.ctx.Done()
}
