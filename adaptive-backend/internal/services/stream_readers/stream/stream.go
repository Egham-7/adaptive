package stream

import (
	"adaptive-backend/internal/services/cache"
	"adaptive-backend/internal/services/stream_readers/sse"
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
	"github.com/valyala/fasthttp"
)

var (
	bufferManager = cache.GetGlobalBufferManager()
)

// HandleStream manages the streaming response to the client with optimized performance
func HandleStream(c *fiber.Ctx, resp *ssestream.Stream[openai.ChatCompletionChunk], requestID string, selectedModel string, provider string) error {
	fiberlog.Infof("[%s] Starting stream handling", requestID)

	c.Context().SetBodyStreamWriter(fasthttp.StreamWriter(func(w *bufio.Writer) {
		startTime := time.Now()
		var totalBytes int64

		streamReader, err := sse.GetSSEStreamReader(resp, requestID, selectedModel, provider)
		if err != nil {
			sendErrorEvent(w, requestID, "Failed to create stream reader", err)
			return
		}
		defer closeStreamReader(streamReader, requestID)

		if err := pumpStreamData(w, streamReader, requestID, startTime, &totalBytes); err != nil {
			sendErrorEvent(w, requestID, "Stream error", err)
		}
	}))
	return nil
}

func pumpStreamData(w *bufio.Writer, streamReader io.Reader, requestID string, startTime time.Time, totalBytesPtr *int64) error {
	// Get buffer from pool - start with medium size for typical streaming
	buffer := bufferManager.GetBuffer(4096)
	defer bufferManager.PutBuffer(buffer)

	var totalBytes int64
	lastFlushTime := time.Now()

	for {
		// Set read deadline to prevent hanging
		if rc, ok := streamReader.(interface{ SetReadDeadline(time.Time) error }); ok {
			if err := rc.SetReadDeadline(time.Now().Add(30 * time.Second)); err != nil {
				fiberlog.Warnf("[%s] Warning: failed to set read deadline: %v", requestID, err)
			}
		}

		n, err := streamReader.Read(buffer)

		if n > 0 {
			totalBytes += int64(n)

			if writeErr := writeChunk(w, buffer[:n], requestID); writeErr != nil {
				return writeErr
			}

			// Flush periodically or when buffer gets large
			now := time.Now()
			flushInterval := now.Sub(lastFlushTime)
			if flushInterval >= 100*time.Millisecond || w.Buffered() >= 1024 {
				if flushErr := w.Flush(); flushErr != nil {
					return fmt.Errorf("[%s] flushing data: %w", requestID, flushErr)
				}

				lastFlushTime = now
			}
		}

		if err == io.EOF {
			// Final flush
			if flushErr := w.Flush(); flushErr != nil {
				fiberlog.Errorf("[%s] Error in final flush: %v", requestID, flushErr)
			}

			duration := time.Since(startTime)

			// Update the pointer for the caller
			if totalBytesPtr != nil {
				*totalBytesPtr = totalBytes
			}

			fiberlog.Infof("[%s] Stream completed: %d bytes in %v (%.2f KB/s)",
				requestID, totalBytes, duration, float64(totalBytes)/duration.Seconds()/1024)
			return nil
		}

		if err != nil {
			return fmt.Errorf("[%s] reading from stream: %w", requestID, err)
		}
	}
}

// writeChunk writes data efficiently with error handling
func writeChunk(w *bufio.Writer, data []byte, requestID string) error {
	written := 0
	for written < len(data) {
		n, err := w.Write(data[written:])
		if err != nil {
			return fmt.Errorf("[%s] writing chunk (offset %d): %w", requestID, written, err)
		}
		written += n
	}
	return nil
}

func closeStreamReader(streamReader io.ReadCloser, requestID string) {
	if err := streamReader.Close(); err != nil {
		fiberlog.Errorf("[%s] Error closing stream reader: %v", requestID, err)
	}
}

func sendErrorEvent(w *bufio.Writer, requestID, message string, err error) {
	if err == nil {
		fiberlog.Warnf("[%s] Warning: sendErrorEvent called with nil error", requestID)
		return
	}

	fiberlog.Errorf("[%s] %s: %v", requestID, message, err)

	// Use proper JSON marshaling for safe escaping
	errorResponse := map[string]string{
		"error":      err.Error(),
		"request_id": requestID,
	}

	errorJSON, jsonErr := json.Marshal(errorResponse)
	if jsonErr != nil {
		fiberlog.Errorf("[%s] Failed to marshal error JSON: %v", requestID, jsonErr)
		return
	}

	errorData := fmt.Sprintf("data: %s\n\n", string(errorJSON))

	if _, writeErr := w.WriteString(errorData); writeErr != nil {
		fiberlog.Errorf("[%s] Failed to write error event: %v", requestID, writeErr)
		return
	}

	if flushErr := w.Flush(); flushErr != nil {
		fiberlog.Errorf("[%s] Failed to flush error event: %v", requestID, flushErr)
	}
}
