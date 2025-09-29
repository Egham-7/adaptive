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
	sendDone   bool
}

// NewHTTPStreamWriter creates a new HTTP stream writer
func NewHTTPStreamWriter(writer *bufio.Writer, connState contracts.ConnectionState, requestID string, sendDone bool) *HTTPStreamWriter {
	return &HTTPStreamWriter{
		writer:    writer,
		connState: connState,
		requestID: requestID,
		sendDone:  sendDone,
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
	// Write completion message only if sendDone is true
	if w.connState.IsConnected() {
		if w.sendDone {
			n, writeErr := w.writer.WriteString("data: [DONE]\n\n")
			// Always add written bytes to total, even on partial writes
			w.totalBytes += int64(n)

			if writeErr != nil {
				if contracts.IsConnectionClosed(writeErr) {
					return contracts.NewClientDisconnectError(w.requestID)
				}
				return contracts.NewInternalError(w.requestID, "write failed", writeErr)
			}

			// Flush and capture any error
			if flushErr := w.writer.Flush(); flushErr != nil {
				if contracts.IsConnectionClosed(flushErr) {
					return contracts.NewClientDisconnectError(w.requestID)
				}
				return contracts.NewInternalError(w.requestID, "flush failed", flushErr)
			}
		} else {
			// Just flush without sending [DONE] message
			if flushErr := w.writer.Flush(); flushErr != nil {
				if contracts.IsConnectionClosed(flushErr) {
					return contracts.NewClientDisconnectError(w.requestID)
				}
				return contracts.NewInternalError(w.requestID, "flush failed", flushErr)
			}
		}
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
