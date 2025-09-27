package writers

import (
	"bufio"

	"adaptive-backend/internal/services/stream/contracts"
)

// JSONStreamWriter handles pure JSON streaming output without SSE formatting
// Used by providers that expect raw JSON streaming instead of Server-Sent Events
type JSONStreamWriter struct {
	writer     *bufio.Writer
	connState  contracts.ConnectionState
	requestID  string
	totalBytes int64
}

// NewJSONStreamWriter creates a new JSON stream writer
func NewJSONStreamWriter(writer *bufio.Writer, connState contracts.ConnectionState, requestID string) *JSONStreamWriter {
	return &JSONStreamWriter{
		writer:    writer,
		connState: connState,
		requestID: requestID,
	}
}

// Write writes JSON data directly to the stream without SSE formatting
func (w *JSONStreamWriter) Write(data []byte) error {
	if len(data) == 0 {
		return nil
	}

	// Check connection state
	if !w.connState.IsConnected() {
		return contracts.NewClientDisconnectError(w.requestID)
	}

	// Write JSON data directly
	return w.writeBytes(data)
}

// writeBytes is a helper method to write bytes and track total bytes
func (w *JSONStreamWriter) writeBytes(data []byte) error {
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
func (w *JSONStreamWriter) Flush() error {
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

// Close closes the writer without adding SSE termination
func (w *JSONStreamWriter) Close() error {
	// Just flush remaining data
	if w.connState.IsConnected() {
		return w.writer.Flush()
	}
	return nil
}

// TotalBytes returns total bytes written
func (w *JSONStreamWriter) TotalBytes() int64 {
	return w.totalBytes
}
