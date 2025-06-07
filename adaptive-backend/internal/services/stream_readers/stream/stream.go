package stream

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"time"

	"adaptive-backend/internal/services/stream_readers"
	"adaptive-backend/internal/services/stream_readers/sse"

	"github.com/gofiber/fiber/v2"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
	"github.com/valyala/fasthttp"
)

// HandleStream manages the streaming response to the client
func HandleStream(c *fiber.Ctx, resp *ssestream.Stream[openai.ChatCompletionChunk], requestID string) error {
	c.Context().SetBodyStreamWriter(fasthttp.StreamWriter(func(w *bufio.Writer) {
		streamReader, err := selectStreamReader(resp, requestID)
		if err != nil {
			sendErrorEvent(w, requestID, "Failed to create stream reader", err)
			return
		}
		defer closeStreamReader(streamReader, requestID)

		if err := pumpStreamData(w, streamReader, requestID); err != nil {
			sendErrorEvent(w, requestID, "Stream error", err)
		}
	}))
	return nil
}

func selectStreamReader(resp *ssestream.Stream[openai.ChatCompletionChunk], requestID string) (stream_readers.StreamReader, error) {
	return sse.GetSSEStreamReader(resp, requestID)
}

// Helper functions
func closeStreamReader(streamReader io.ReadCloser, requestID string) {
	if err := streamReader.Close(); err != nil {
		log.Printf("[%s] Error closing stream reader: %v", requestID, err)
	}
	log.Printf("[%s] Stream completed", requestID)
}

func pumpStreamData(w *bufio.Writer, streamReader io.Reader, requestID string) error {
	buffer := make([]byte, 1024)
	startTime := time.Now()

	for {
		n, err := streamReader.Read(buffer)

		if n > 0 {
			if err := writeAndFlush(w, buffer[:n], requestID); err != nil {
				return err
			}
		}

		if err == io.EOF {
			log.Printf("[%s] Stream completed after %v", requestID, time.Since(startTime))
			return nil
		}

		if err != nil {
			return fmt.Errorf("reading from stream: %w", err)
		}
	}
}

func writeAndFlush(w *bufio.Writer, data []byte, requestID string) error {
	if _, err := w.Write(data); err != nil {
		return fmt.Errorf("[%s] writing to response: %w", requestID, err)
	}

	if err := w.Flush(); err != nil {
		return fmt.Errorf("[%s] flushing data: %w", requestID, err)
	}

	return nil
}

func sendErrorEvent(w *bufio.Writer, requestID, message string, err error) {
	if err == nil {
		log.Printf("[%s] Warning: sendErrorEvent called with nil error", requestID)
		return
	}

	log.Printf("[%s] %s: %v", requestID, message, err)

	if _, writeErr := fmt.Fprintf(w, "data: {\"error\": \"%s\"}\n\n", err.Error()); writeErr != nil {
		log.Printf("[%s] Failed to write error event: %v", requestID, writeErr)
		return
	}

	if flushErr := w.Flush(); flushErr != nil {
		log.Printf("[%s] Failed to flush error event: %v", requestID, flushErr)
		return
	}
}
