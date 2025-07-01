package stream

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/cache"
	"adaptive-backend/internal/services/metrics"
	"adaptive-backend/internal/services/stream_readers"
	"adaptive-backend/internal/services/stream_readers/sse"
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"sync"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
	"github.com/valyala/fasthttp"
)

var (
	bufferManager     = cache.GetGlobalBufferManager()
	promStreamMetrics *metrics.StreamingMetrics
	metricsInitOnce   sync.Once
)

func initMetrics() {
	metricsInitOnce.Do(func() {
		promStreamMetrics = metrics.NewStreamingMetrics()
	})
}

// HandleStream manages the streaming response to the client with optimized performance
func HandleStream(c *fiber.Ctx, resp *ssestream.Stream[openai.ChatCompletionChunk], requestID string, selectedProvider string, selectedModel string, comparisonProvider models.ComparisonProvider) error {
	log.Printf("[%s] Starting stream handling", requestID)
	// Initialize metrics if needed
	initMetrics()

	// Extract provider and model from context or headers
	provider := c.Get("X-Provider", "unknown")
	model := c.Get("X-Model", "unknown")

	promStreamMetrics.RecordStreamStart(provider, model)

	c.Context().SetBodyStreamWriter(fasthttp.StreamWriter(func(w *bufio.Writer) {
		startTime := time.Now()
		var totalBytes int64
		var status string

		streamReader, err := selectStreamReader(resp, requestID, selectedProvider, selectedModel, comparisonProvider)
		if err != nil {
			status = "reader_error"
			promStreamMetrics.RecordError("reader_creation", provider)
			sendErrorEvent(w, requestID, "Failed to create stream reader", err)
			// Record completion metrics
			duration := time.Since(startTime)
			promStreamMetrics.RecordStreamEnd(provider, model, status, duration.Seconds(), totalBytes)
			return
		}
		defer closeStreamReader(streamReader, requestID)

		if err := pumpStreamData(w, streamReader, requestID, startTime, &totalBytes); err != nil {
			status = "stream_error"
			promStreamMetrics.RecordError("streaming", provider)
			sendErrorEvent(w, requestID, "Stream error", err)
		} else {
			status = "success"
		}

		// Record completion metrics
		duration := time.Since(startTime)
		promStreamMetrics.RecordStreamEnd(provider, model, status, duration.Seconds(), totalBytes)
	}))
	return nil
}

func selectStreamReader(resp *ssestream.Stream[openai.ChatCompletionChunk], requestID string, selectedProvider string, selectedModel string, comparisonProvider models.ComparisonProvider) (stream_readers.StreamReader, error) {
	return sse.GetSSEStreamReader(resp, requestID, selectedProvider, selectedModel, comparisonProvider)
}

func pumpStreamData(w *bufio.Writer, streamReader io.Reader, requestID string, startTime time.Time, totalBytesPtr *int64) error {
	// Get buffer from pool - start with medium size for typical streaming
	buffer := bufferManager.GetBuffer(4096)
	defer bufferManager.PutBuffer(buffer)

	promStreamMetrics.RecordBufferPoolHit("medium")

	var totalBytes int64
	lastFlushTime := time.Now()

	for {
		// Set read deadline to prevent hanging
		if rc, ok := streamReader.(interface{ SetReadDeadline(time.Time) error }); ok {
			if err := rc.SetReadDeadline(time.Now().Add(30 * time.Second)); err != nil {
				log.Printf("[%s] Warning: failed to set read deadline: %v", requestID, err)
			}
		}

		n, err := streamReader.Read(buffer)

		if n > 0 {
			totalBytes += int64(n)
			promStreamMetrics.RecordChunkSize(n)

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

				promStreamMetrics.RecordFlushInterval(flushInterval.Seconds())
				lastFlushTime = now
			}
		}

		if err == io.EOF {
			// Final flush
			if flushErr := w.Flush(); flushErr != nil {
				log.Printf("[%s] Error in final flush: %v", requestID, flushErr)
			}

			duration := time.Since(startTime)

			// Update the pointer for the caller
			if totalBytesPtr != nil {
				*totalBytesPtr = totalBytes
			}

			log.Printf("[%s] Stream completed: %d bytes in %v (%.2f KB/s)",
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
		log.Printf("[%s] Error closing stream reader: %v", requestID, err)
	}
}

func sendErrorEvent(w *bufio.Writer, requestID, message string, err error) {
	if err == nil {
		log.Printf("[%s] Warning: sendErrorEvent called with nil error", requestID)
		return
	}

	log.Printf("[%s] %s: %v", requestID, message, err)
	promStreamMetrics.RecordError("send_error", "unknown")

	// Use proper JSON marshaling for safe escaping
	errorResponse := map[string]string{
		"error":      err.Error(),
		"request_id": requestID,
	}

	errorJSON, jsonErr := json.Marshal(errorResponse)
	if jsonErr != nil {
		log.Printf("[%s] Failed to marshal error JSON: %v", requestID, jsonErr)
		return
	}

	errorData := fmt.Sprintf("data: %s\n\n", string(errorJSON))

	if _, writeErr := w.WriteString(errorData); writeErr != nil {
		log.Printf("[%s] Failed to write error event: %v", requestID, writeErr)
		return
	}

	if flushErr := w.Flush(); flushErr != nil {
		log.Printf("[%s] Failed to flush error event: %v", requestID, flushErr)
	}
}
