package cache

import (
	"bufio"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"adaptive-backend/internal/models"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go"
	"github.com/valyala/fasthttp"
)

// StreamCachedResponse converts a cached ChatCompletion response to Server-Sent Events (SSE) format
func StreamCachedResponse(c *fiber.Ctx, cachedResp *models.ChatCompletion, requestID string) error {
	fiberlog.Infof("[%s] Streaming cached response as SSE", requestID)

	// Get FastHTTP context
	fasthttpCtx := c.Context()

	// Set SSE headers for client compatibility
	fasthttpCtx.Response.Header.Set("Content-Type", "text/event-stream")
	fasthttpCtx.Response.Header.Set("Cache-Control", "no-cache")
	fasthttpCtx.Response.Header.Set("Connection", "keep-alive")

	fasthttpCtx.SetBodyStreamWriter(fasthttp.StreamWriter(func(w *bufio.Writer) {
		startTime := time.Now()

		// Create streaming chunks from the cached response
		chunks := convertToStreamingChunks(cachedResp, requestID)

		fiberlog.Debugf("[%s] Generated %d streaming chunks from cached response", requestID, len(chunks))

		// Send each chunk with proper SSE formatting
		for i, chunk := range chunks {
			// Check for client disconnect
			select {
			case <-fasthttpCtx.Done():
				fiberlog.Infof("[%s] Client disconnected during cached stream", requestID)
				return
			default:
			}

			// Marshal chunk to JSON
			chunkJSON, err := json.Marshal(chunk)
			if err != nil {
				fiberlog.Errorf("[%s] Failed to marshal streaming chunk %d: %v", requestID, i, err)
				sendStreamErrorEvent(w, requestID, "Failed to marshal chunk", err)
				return
			}

			// Format as SSE event
			sseData := fmt.Sprintf("data: %s\n\n", string(chunkJSON))

			// Write chunk
			if _, err := w.WriteString(sseData); err != nil {
				fiberlog.Errorf("[%s] Failed to write streaming chunk %d: %v", requestID, i, err)
				return
			}

			// Flush immediately for real-time streaming feel
			if err := w.Flush(); err != nil {
				fiberlog.Errorf("[%s] Failed to flush streaming chunk %d: %v", requestID, i, err)
				return
			}

			// Add small delay to simulate streaming (optional)
			time.Sleep(10 * time.Millisecond)
		}

		// Send final [DONE] event
		if _, err := w.WriteString("data: [DONE]\n\n"); err != nil {
			fiberlog.Errorf("[%s] Failed to write final [DONE] event: %v", requestID, err)
			return
		}

		if err := w.Flush(); err != nil {
			fiberlog.Errorf("[%s] Failed to flush final [DONE] event: %v", requestID, err)
			return
		}

		duration := time.Since(startTime)
		fiberlog.Infof("[%s] Cached stream completed: %d chunks in %v", requestID, len(chunks), duration)
	}))

	return nil
}

// convertToStreamingChunks converts a ChatCompletion to streaming chunks
func convertToStreamingChunks(completion *models.ChatCompletion, requestID string) []models.ChatCompletionChunk {
	if len(completion.Choices) == 0 {
		fiberlog.Warnf("[%s] No choices in cached completion", requestID)
		return []models.ChatCompletionChunk{}
	}

	var chunks []models.ChatCompletionChunk
	choice := completion.Choices[0]
	content := choice.Message.Content

	// Split content into words for word-by-word streaming
	words := strings.Fields(content)
	if len(words) == 0 {
		// Handle empty content case
		words = []string{""}
	}

	// Create initial chunk with role
	firstChunk := models.ChatCompletionChunk{
		ID:      completion.ID,
		Object:  "chat.completion.chunk",
		Created: completion.Created,
		Model:   completion.Model,
		Choices: []openai.ChatCompletionChunkChoice{
			{
				Index: choice.Index,
				Delta: openai.ChatCompletionChunkChoiceDelta{
					Role: string(choice.Message.Role),
				},
				Logprobs:     openai.ChatCompletionChunkChoiceLogprobs{},
				FinishReason: "",
			},
		},
		Usage:    &completion.Usage,
		Provider: completion.Provider,
	}
	chunks = append(chunks, firstChunk)

	// Create chunks for content, sending a few words at a time
	wordsPerChunk := 3 // Adjust as needed for streaming feel
	for i := 0; i < len(words); i += wordsPerChunk {
		end := min(i+wordsPerChunk, len(words))

		// Join words for this chunk
		chunkContent := strings.Join(words[i:end], " ")
		if i > 0 {
			chunkContent = " " + chunkContent // Add space between chunks
		}

		contentChunk := models.ChatCompletionChunk{
			ID:      completion.ID,
			Object:  "chat.completion.chunk",
			Created: completion.Created,
			Model:   completion.Model,
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Index: choice.Index,
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Content: chunkContent,
					},
					Logprobs:     openai.ChatCompletionChunkChoiceLogprobs{},
					FinishReason: "",
				},
			},
			Provider: completion.Provider,
		}
		chunks = append(chunks, contentChunk)
	}

	// Create final chunk with finish reason
	finalChunk := models.ChatCompletionChunk{
		ID:      completion.ID,
		Created: completion.Created,
		Model:   completion.Model,
		Object:  "chat.completion.chunk",
		Choices: []openai.ChatCompletionChunkChoice{
			{
				Index: choice.Index,
				Delta: openai.ChatCompletionChunkChoiceDelta{
					Content: "",
				},
				Logprobs:     openai.ChatCompletionChunkChoiceLogprobs{},
				FinishReason: choice.FinishReason,
			},
		},
		Usage:    &completion.Usage,
		Provider: completion.Provider,
	}
	chunks = append(chunks, finalChunk)

	return chunks
}

// sendStreamErrorEvent sends an error event in SSE format
func sendStreamErrorEvent(w *bufio.Writer, requestID, message string, err error) {
	if err == nil {
		fiberlog.Warnf("[%s] Warning: sendStreamErrorEvent called with nil error", requestID)
		return
	}

	fiberlog.Errorf("[%s] %s: %v", requestID, message, err)

	// Create error response in OpenAI format
	errorResponse := map[string]any{
		"error": map[string]any{
			"message":    err.Error(),
			"type":       "cached_stream_error",
			"request_id": requestID,
		},
	}

	errorJSON, jsonErr := json.Marshal(errorResponse)
	if jsonErr != nil {
		fiberlog.Errorf("[%s] Failed to marshal stream error JSON: %v", requestID, jsonErr)
		return
	}

	errorData := fmt.Sprintf("data: %s\n\n", string(errorJSON))

	if _, writeErr := w.WriteString(errorData); writeErr != nil {
		fiberlog.Errorf("[%s] Failed to write stream error event: %v", requestID, writeErr)
		return
	}

	if flushErr := w.Flush(); flushErr != nil {
		fiberlog.Errorf("[%s] Failed to flush stream error event: %v", requestID, flushErr)
	}
}
