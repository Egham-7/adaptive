package stream_simulator

import (
	"bufio"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"adaptive-backend/internal/models"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/valyala/fasthttp"
)

// StreamAnthropicCachedResponse converts a cached AnthropicMessage response to Server-Sent Events (SSE) format
func StreamAnthropicCachedResponse(c *fiber.Ctx, cachedResp *models.AnthropicMessage, requestID string) error {
	fiberlog.Infof("[%s] Streaming cached Anthropic response as SSE", requestID)

	// Get FastHTTP context
	fasthttpCtx := c.Context()

	// Set SSE headers for client compatibility
	fasthttpCtx.Response.Header.Set("Content-Type", "text/event-stream")
	fasthttpCtx.Response.Header.Set("Cache-Control", "no-cache")
	fasthttpCtx.Response.Header.Set("Connection", "keep-alive")

	fasthttpCtx.SetBodyStreamWriter(fasthttp.StreamWriter(func(w *bufio.Writer) {
		startTime := time.Now()

		// Create streaming chunks from the cached response
		chunks := convertToAnthropicStreamingChunks(cachedResp, requestID)

		fiberlog.Debugf("[%s] Generated %d streaming chunks from cached Anthropic response", requestID, len(chunks))

		// Send each chunk with proper SSE formatting
		for i, chunk := range chunks {
			// Check for client disconnect
			select {
			case <-fasthttpCtx.Done():
				fiberlog.Infof("[%s] Client disconnected during cached Anthropic stream", requestID)
				return
			default:
			}

			// Marshal chunk to JSON
			chunkJSON, err := json.Marshal(chunk)
			if err != nil {
				fiberlog.Errorf("[%s] Failed to marshal Anthropic streaming chunk %d: %v", requestID, i, err)
				sendAnthropicStreamErrorEvent(w, requestID, "Failed to marshal chunk", err)
				return
			}

			// Format as SSE event
			sseData := fmt.Sprintf("event: %s \ndata: %s\n\n", chunk.Type, string(chunkJSON))

			// Write chunk
			if _, err := w.WriteString(sseData); err != nil {
				fiberlog.Errorf("[%s] Failed to write Anthropic streaming chunk %d: %v", requestID, i, err)
				return
			}

			// Flush immediately for real-time streaming feel
			if err := w.Flush(); err != nil {
				fiberlog.Errorf("[%s] Failed to flush Anthropic streaming chunk %d: %v", requestID, i, err)
				return
			}

			// Add small delay to simulate streaming (optional)
			time.Sleep(15 * time.Millisecond) // Slightly slower than OpenAI to feel different
		}

		// Send final [DONE] event (Anthropic uses this too for compatibility)
		if _, err := w.WriteString("data: [DONE]\n\n"); err != nil {
			fiberlog.Errorf("[%s] Failed to write final [DONE] event: %v", requestID, err)
			return
		}

		if err := w.Flush(); err != nil {
			fiberlog.Errorf("[%s] Failed to flush final [DONE] event: %v", requestID, err)
			return
		}

		duration := time.Since(startTime)
		fiberlog.Infof("[%s] Cached Anthropic stream completed: %d chunks in %v", requestID, len(chunks), duration)
	}))

	return nil
}

// convertToAnthropicStreamingChunks converts an AnthropicMessage to streaming chunks
func convertToAnthropicStreamingChunks(message *models.AnthropicMessage, requestID string) []models.AnthropicMessageChunk {
	if len(message.Content) == 0 {
		fiberlog.Warnf("[%s] No content in cached Anthropic message", requestID)
		return []models.AnthropicMessageChunk{}
	}

	var chunks []models.AnthropicMessageChunk

	// Find the first text content block
	var textContent string
	for _, content := range message.Content {
		if content.Type == "text" && content.Text != "" {
			textContent = content.Text
			break
		}
	}

	if textContent == "" {
		fiberlog.Warnf("[%s] No text content found in cached Anthropic message", requestID)
		return []models.AnthropicMessageChunk{}
	}

	// 1. Send message_start event
	messageStartChunk := models.AnthropicMessageChunk{
		Type: "message_start",
		Message: &models.AnthropicMessage{
			ID:           message.ID,
			Content:      []anthropic.ContentBlockUnion{}, // Empty initially
			Model:        message.Model,
			Role:         message.Role,
			StopReason:   "",
			StopSequence: "",
			Type:         message.Type,
			Usage:        message.Usage,
			Provider:     message.Provider,
		},
		Provider: message.Provider,
	}
	chunks = append(chunks, messageStartChunk)

	// 2. Send content_block_start event
	var contentBlockIndex int64 = 0
	contentBlockStartChunk := models.AnthropicMessageChunk{
		Type:     "content_block_start",
		Index:    &contentBlockIndex,
		Provider: message.Provider,
	}
	chunks = append(chunks, contentBlockStartChunk)

	// 3. Split content into words and send content_block_delta events
	words := strings.Fields(textContent)
	if len(words) == 0 {
		words = []string{""}
	}

	wordsPerChunk := 2 // Fewer words per chunk for Anthropic to feel more responsive
	for i := 0; i < len(words); i += wordsPerChunk {
		end := min(i+wordsPerChunk, len(words))

		// Join words for this chunk
		chunkText := strings.Join(words[i:end], " ")
		if i > 0 {
			chunkText = " " + chunkText // Add space between chunks
		}

		contentDeltaChunk := models.AnthropicMessageChunk{
			Type:  "content_block_delta",
			Index: &contentBlockIndex,
			Delta: &anthropic.MessageStreamEventUnionDelta{
				Type: "text_delta",
				Text: chunkText,
			},
			Provider: message.Provider,
		}
		chunks = append(chunks, contentDeltaChunk)
	}

	// 4. Send content_block_stop event
	contentBlockStopChunk := models.AnthropicMessageChunk{
		Type:     "content_block_stop",
		Index:    &contentBlockIndex,
		Provider: message.Provider,
	}
	chunks = append(chunks, contentBlockStopChunk)

	// 5. Send message_delta event with stop reason and usage
	messageDeltaChunk := models.AnthropicMessageChunk{
		Type: "message_delta",
		Delta: &anthropic.MessageStreamEventUnionDelta{
			StopReason: anthropic.StopReason(message.StopReason),
		},
		Usage:    &message.Usage,
		Provider: message.Provider,
	}
	chunks = append(chunks, messageDeltaChunk)

	// 6. Send message_stop event
	messageStopChunk := models.AnthropicMessageChunk{
		Type:     "message_stop",
		Provider: message.Provider,
	}
	chunks = append(chunks, messageStopChunk)

	return chunks
}

// sendAnthropicStreamErrorEvent sends an error event in SSE format for Anthropic
func sendAnthropicStreamErrorEvent(w *bufio.Writer, requestID, message string, err error) {
	if err == nil {
		fiberlog.Warnf("[%s] Warning: sendAnthropicStreamErrorEvent called with nil error", requestID)
		return
	}

	fiberlog.Errorf("[%s] %s: %v", requestID, message, err)

	// Create error response in Anthropic format
	errorResponse := map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    "cached_stream_error",
			"message": err.Error(),
		},
	}

	errorJSON, jsonErr := json.Marshal(errorResponse)
	if jsonErr != nil {
		fiberlog.Errorf("[%s] Failed to marshal Anthropic stream error JSON: %v", requestID, jsonErr)
		return
	}

	errorData := fmt.Sprintf("data: %s\n\n", string(errorJSON))

	if _, writeErr := w.WriteString(errorData); writeErr != nil {
		fiberlog.Errorf("[%s] Failed to write Anthropic stream error event: %v", requestID, writeErr)
		return
	}

	if flushErr := w.Flush(); flushErr != nil {
		fiberlog.Errorf("[%s] Failed to flush Anthropic stream error event: %v", requestID, flushErr)
	}
}
