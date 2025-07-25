package stream

import (
	"adaptive-backend/internal/services/stream_readers/sse"
	"bufio"
	"bytes"
	"context"
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

// HandleStream manages the streaming response to the client with optimized performance
func HandleStream(c *fiber.Ctx, resp *ssestream.Stream[openai.ChatCompletionChunk], requestID string, selectedModel string, provider string) error {
	fiberlog.Infof("[%s] Starting stream handling", requestID)

	c.Context().SetBodyStreamWriter(fasthttp.StreamWriter(func(w *bufio.Writer) {
		startTime := time.Now()
		var totalBytes int64
		ctx := c.Context()

		streamReader, err := sse.GetSSEStreamReader(ctx, resp, requestID, selectedModel, provider)
		if err != nil {
			sendErrorEvent(w, requestID, "Failed to create stream reader", err)
			return
		}
		defer closeStreamReader(streamReader, requestID)

		if err := pumpStreamData(ctx, w, streamReader, requestID, startTime, &totalBytes); err != nil {
			if ctx.Err() != nil {
				fiberlog.Infof("[%s] Client disconnected during stream", requestID)
				return
			}
			sendErrorEvent(w, requestID, "Stream error", err)
		}
	}))
	return nil
}

func pumpStreamData(ctx context.Context, w *bufio.Writer, streamReader io.Reader, requestID string, startTime time.Time, totalBytesPtr *int64) error {
	// Use larger buffer for better throughput - SSE chunks are typically 1-4KB
	const bufferSize = 4096
	buffer := make([]byte, bufferSize) // Stack allocation for hot path

	var totalBytes int64
	var writeBuffer []byte      // Accumulate small writes before flushing
	const flushThreshold = 2048 // Flush when we accumulate 2KB

	// Performance tracking
	var readCount, flushCount int64
	var maxBufferSize int
	lastLogTime := startTime
	const logInterval = 5 * time.Second

	fiberlog.Debugf("[%s] Starting stream pump - buffer_size=%d flush_threshold=%d",
		requestID, bufferSize, flushThreshold)

	for {
		// Check for client disconnect
		if ctx != nil {
			select {
			case <-ctx.Done():
				fiberlog.Infof("[%s] Client disconnected, stopping stream", requestID)
				return ctx.Err()
			default:
			}
		}

		// Read timeout is handled by context cancellation and internal stream timeouts

		n, err := streamReader.Read(buffer)
		readCount++

		if n > 0 {
			totalBytes += int64(n)

			// Accumulate data for batch writing
			writeBuffer = append(writeBuffer, buffer[:n]...)

			// Track buffer utilization
			if len(writeBuffer) > maxBufferSize {
				maxBufferSize = len(writeBuffer)
			}

			// Flush when threshold reached or if it's a complete SSE message
			shouldFlush := len(writeBuffer) >= flushThreshold ||
				(len(writeBuffer) > 0 && bytes.HasSuffix(writeBuffer, []byte("\n\n")))

			if shouldFlush {
				flushCount++
				fiberlog.Debugf("[%s] Flushing batch: size=%d bytes, chunk_complete=%v",
					requestID, len(writeBuffer), bytes.HasSuffix(writeBuffer, []byte("\n\n")))

				if writeErr := writeChunk(w, writeBuffer, requestID); writeErr != nil {
					fiberlog.Errorf("[%s] Write error after %d reads, %d flushes: %v",
						requestID, readCount, flushCount, writeErr)
					return writeErr
				}
				writeBuffer = writeBuffer[:0] // Reset buffer
			}
		}

		// Periodic performance logging
		if time.Since(lastLogTime) >= logInterval {
			duration := time.Since(startTime)
			throughput := float64(totalBytes) / duration.Seconds() / 1024
			fiberlog.Infof("[%s] Stream progress: %d bytes, %d reads, %d flushes, %.2f KB/s, max_buffer=%d",
				requestID, totalBytes, readCount, flushCount, throughput, maxBufferSize)
			lastLogTime = time.Now()
		}

		if err == io.EOF {
			// Flush any remaining data
			if len(writeBuffer) > 0 {
				if writeErr := writeChunk(w, writeBuffer, requestID); writeErr != nil {
					fiberlog.Warnf("[%s] Warning: failed to flush final data: %v", requestID, writeErr)
				}
			}

			duration := time.Since(startTime)

			// Update the pointer for the caller
			if totalBytesPtr != nil {
				*totalBytesPtr = totalBytes
			}

			fiberlog.Infof("[%s] Stream completed: %d bytes in %v (%.2f KB/s), reads=%d, flushes=%d, max_buffer=%d",
				requestID, totalBytes, duration, float64(totalBytes)/duration.Seconds()/1024,
				readCount, flushCount, maxBufferSize)
			return nil
		}

		if err != nil {
			fiberlog.Errorf("[%s] Stream read error after %d successful reads (%d bytes): %v",
				requestID, readCount, totalBytes, err)
			return fmt.Errorf("[%s] reading from stream: %w", requestID, err)
		}
	}
}

// writeChunk writes data and flushes for optimal throughput
func writeChunk(w *bufio.Writer, data []byte, requestID string) error {
	if _, err := w.Write(data); err != nil {
		return fmt.Errorf("[%s] writing chunk: %w", requestID, err)
	}
	if err := w.Flush(); err != nil {
		return fmt.Errorf("[%s] flushing chunk: %w", requestID, err)
	}
	return nil
}

func closeStreamReader(streamReader io.ReadCloser, requestID string) {
	closeStart := time.Now()
	if err := streamReader.Close(); err != nil {
		fiberlog.Errorf("[%s] Error closing stream reader: %v", requestID, err)
	} else {
		fiberlog.Debugf("[%s] Stream reader closed in %v", requestID, time.Since(closeStart))
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
