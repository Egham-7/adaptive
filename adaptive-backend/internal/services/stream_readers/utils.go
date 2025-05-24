package stream_readers

import (
	"adaptive-backend/internal/models"
	"bufio"
	"fmt"
	"io"
	"log"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/valyala/fasthttp"
	"strings" // Added for strings.TrimSpace and strings.HasPrefix
)

// handleStream manages the streaming response to the client
func HandleStream(c *fiber.Ctx, resp *models.ChatCompletionResponse, requestID string, streamOpt models.StreamOption) error {
	c.Context().SetBodyStreamWriter(fasthttp.StreamWriter(func(w *bufio.Writer) {
		streamReader, err := GetStreamReader(resp, resp.Provider, requestID)
		if err != nil {
			sendErrorEvent(w, requestID, "Failed to create stream reader", err)
			return
		}

		defer closeStreamReader(streamReader, requestID)

		if err := pumpStreamData(w, streamReader, requestID, streamOpt); err != nil {
			sendErrorEvent(w, requestID, "Stream error", err)
		}
	}))

	return nil
}

// Helper functions to make the main function cleaner
func closeStreamReader(streamReader io.ReadCloser, requestID string) {
	if err := streamReader.Close(); err != nil {
		log.Printf("[%s] Error closing stream reader: %v", requestID, err)
	}
	log.Printf("[%s] Stream completed", requestID)
}

func pumpStreamData(w *bufio.Writer, streamReader io.Reader, requestID string, streamOpt models.StreamOption) error {
	startTime := time.Now()

	if streamOpt == models.TEXT {
		scanner := bufio.NewScanner(streamReader)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "data: ") {
				payload := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
				if payload == "[DONE]" {
					continue // Skip [DONE] messages
				}
				if _, err := w.WriteString(payload + "\n"); err != nil {
					return fmt.Errorf("[%s] writing text data to response: %w", requestID, err)
				}
				if err := w.Flush(); err != nil {
					return fmt.Errorf("[%s] flushing text data: %w", requestID, err)
				}
			}
		}
		if err := scanner.Err(); err != nil {
			return fmt.Errorf("[%s] reading from stream with scanner: %w", requestID, err)
		}
		log.Printf("[%s] TEXT Stream completed after %v", requestID, time.Since(startTime))
		return nil
	} else { // Default to SSE or other modes
		buffer := make([]byte, 1024)
		for {
			n, err := streamReader.Read(buffer)

			if n > 0 {
				if err := writeAndFlush(w, buffer[:n], requestID); err != nil {
					return err
				}
			}

			if err == io.EOF {
				log.Printf("[%s] SSE Stream completed after %v", requestID, time.Since(startTime))
				return nil
			}

			if err != nil {
				return fmt.Errorf("[%s] reading from stream: %w", requestID, err)
			}
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
