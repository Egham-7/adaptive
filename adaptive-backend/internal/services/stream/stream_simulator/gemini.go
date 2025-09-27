package stream_simulator

import (
	"bufio"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"adaptive-backend/internal/models"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/valyala/fasthttp"
	"google.golang.org/genai"
)

// StreamGeminiCachedResponse converts a cached GeminiGenerateContentResponse to Server-Sent Events (SSE) format
func StreamGeminiCachedResponse(c *fiber.Ctx, cachedResp *models.GeminiGenerateContentResponse, requestID string) error {
	fiberlog.Infof("[%s] Streaming cached Gemini response as SSE", requestID)

	// Get FastHTTP context
	fasthttpCtx := c.Context()

	// Set SSE headers for client compatibility
	fasthttpCtx.Response.Header.Set("Content-Type", "text/event-stream")
	fasthttpCtx.Response.Header.Set("Cache-Control", "no-cache")
	fasthttpCtx.Response.Header.Set("Connection", "keep-alive")

	fasthttpCtx.SetBodyStreamWriter(fasthttp.StreamWriter(func(w *bufio.Writer) {
		startTime := time.Now()

		// Create streaming chunks from the cached response
		chunks := convertToGeminiStreamingChunks(cachedResp, requestID)

		fiberlog.Debugf("[%s] Generated %d streaming chunks from cached Gemini response", requestID, len(chunks))

		// Send each chunk with proper SSE formatting
		for i, chunk := range chunks {
			// Check for client disconnect
			select {
			case <-fasthttpCtx.Done():
				fiberlog.Infof("[%s] Client disconnected during cached Gemini stream", requestID)
				return
			default:
			}

			// Marshal chunk to JSON
			chunkJSON, err := json.Marshal(chunk)
			if err != nil {
				fiberlog.Errorf("[%s] Failed to marshal Gemini streaming chunk %d: %v", requestID, i, err)
				sendGeminiStreamErrorEvent(w, requestID, "Failed to marshal chunk", err)
				return
			}

			// Format as SSE event
			sseData := fmt.Sprintf("data: %s\n\n", string(chunkJSON))

			// Write chunk
			if _, err := w.WriteString(sseData); err != nil {
				fiberlog.Errorf("[%s] Failed to write Gemini streaming chunk %d: %v", requestID, i, err)
				return
			}

			// Flush immediately for real-time streaming feel
			if err := w.Flush(); err != nil {
				fiberlog.Errorf("[%s] Failed to flush Gemini streaming chunk %d: %v", requestID, i, err)
				return
			}

			// Add small delay to simulate streaming (slightly faster than Anthropic)
			time.Sleep(12 * time.Millisecond)
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
		fiberlog.Infof("[%s] Cached Gemini stream completed: %d chunks in %v", requestID, len(chunks), duration)
	}))

	return nil
}

// convertToGeminiStreamingChunks converts a GeminiGenerateContentResponse to streaming chunks
func convertToGeminiStreamingChunks(response *models.GeminiGenerateContentResponse, requestID string) []*models.GeminiGenerateContentResponse {
	if len(response.Candidates) == 0 {
		fiberlog.Warnf("[%s] No candidates in cached Gemini response", requestID)
		return []*models.GeminiGenerateContentResponse{}
	}

	var chunks []*models.GeminiGenerateContentResponse
	candidate := response.Candidates[0]

	// Find the first text content part
	var textContent string
	if candidate.Content != nil && len(candidate.Content.Parts) > 0 {
		for _, part := range candidate.Content.Parts {
			if part != nil && part.Text != "" {
				textContent = part.Text
				break
			}
		}
	}

	// Safely extract role from candidate content
	role := ""
	if candidate.Content != nil {
		role = candidate.Content.Role
	}

	if textContent == "" {
		fiberlog.Warnf("[%s] No text content found in cached Gemini response", requestID)
		// Return a single chunk with empty content but preserve metadata
		emptyChunk := &models.GeminiGenerateContentResponse{
			Candidates: []*genai.Candidate{
				{
					Content: &genai.Content{
						Parts: []*genai.Part{{Text: ""}},
						Role:  role,
					},
					FinishReason:     candidate.FinishReason,
					SafetyRatings:    candidate.SafetyRatings,
					CitationMetadata: candidate.CitationMetadata,
				},
			},
			ModelVersion:   response.ModelVersion,
			UsageMetadata:  response.UsageMetadata,
			PromptFeedback: response.PromptFeedback,
			Provider:       response.Provider,
		}
		return []*models.GeminiGenerateContentResponse{emptyChunk}
	}

	// Split content into words for word-by-word streaming
	words := strings.Fields(textContent)
	if len(words) == 0 {
		words = []string{""}
	}

	// Create streaming chunks with progressive content
	wordsPerChunk := 2 // Fewer words per chunk for more responsive feel
	var accumulatedContent strings.Builder

	for i := 0; i < len(words); i += wordsPerChunk {
		end := min(i+wordsPerChunk, len(words))

		// Add words to accumulated content
		chunkWords := words[i:end]
		if i > 0 {
			accumulatedContent.WriteString(" ")
		}
		accumulatedContent.WriteString(strings.Join(chunkWords, " "))

		// Create chunk with accumulated content
		chunk := &models.GeminiGenerateContentResponse{
			Candidates: []*genai.Candidate{
				{
					Content: &genai.Content{
						Parts: []*genai.Part{{Text: accumulatedContent.String()}},
						Role:  role,
					},
					// Don't set FinishReason until the last chunk
					SafetyRatings:    candidate.SafetyRatings,
					CitationMetadata: candidate.CitationMetadata,
				},
			},
			ModelVersion:   response.ModelVersion,
			PromptFeedback: response.PromptFeedback,
			Provider:       response.Provider,
			CreateTime:     response.CreateTime,
			ResponseID:     response.ResponseID,
		}

		chunks = append(chunks, chunk)
	}

	// Update the final chunk with finish reason and usage metadata
	if len(chunks) > 0 {
		finalChunk := chunks[len(chunks)-1]
		finalChunk.Candidates[0].FinishReason = candidate.FinishReason
		finalChunk.UsageMetadata = response.UsageMetadata
	}

	return chunks
}

// sendGeminiStreamErrorEvent sends an error event in SSE format for Gemini
func sendGeminiStreamErrorEvent(w *bufio.Writer, requestID, message string, err error) {
	if err == nil {
		fiberlog.Warnf("[%s] Warning: sendGeminiStreamErrorEvent called with nil error", requestID)
		return
	}

	fiberlog.Errorf("[%s] %s: %v", requestID, message, err)

	// Create error response in Gemini format
	errorResponse := map[string]any{
		"error": map[string]any{
			"code":    500,
			"message": err.Error(),
			"status":  "INTERNAL",
			"details": []any{},
		},
	}

	errorJSON, jsonErr := json.Marshal(errorResponse)
	if jsonErr != nil {
		fiberlog.Errorf("[%s] Failed to marshal Gemini stream error JSON: %v", requestID, jsonErr)
		return
	}

	errorData := fmt.Sprintf("data: %s\n\n", string(errorJSON))

	if _, writeErr := w.WriteString(errorData); writeErr != nil {
		fiberlog.Errorf("[%s] Failed to write Gemini stream error event: %v", requestID, writeErr)
		return
	}

	if flushErr := w.Flush(); flushErr != nil {
		fiberlog.Errorf("[%s] Failed to flush Gemini stream error event: %v", requestID, flushErr)
	}
}
