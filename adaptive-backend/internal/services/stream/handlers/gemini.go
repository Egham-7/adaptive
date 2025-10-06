package handlers

import (
	"bufio"
	"iter"

	"adaptive-backend/internal/services/stream/contracts"
	"adaptive-backend/internal/services/stream/writers"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/valyala/fasthttp"
	"google.golang.org/genai"
)

// HandleGemini manages Gemini streaming response using proper layered architecture
func HandleGemini(c *fiber.Ctx, streamIter iter.Seq2[*genai.GenerateContentResponse, error], requestID, provider, cacheSource string) error {
	fiberlog.Infof("[%s] Starting Gemini stream handling", requestID)

	fasthttpCtx := c.Context()
	// Use SSE format for Gemini SDK compatibility (matches responseLineRE regex)
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Access-Control-Allow-Origin", "*")

	// Channel to capture errors from the async stream handler
	errCh := make(chan error, 1)

	fasthttpCtx.SetBodyStreamWriter(fasthttp.StreamWriter(func(w *bufio.Writer) {
		// Create connection state tracker
		connState := writers.NewFastHTTPConnectionState(fasthttpCtx)

		// Create HTTP writer for SSE formatting (Gemini SDK expects SSE format without [DONE])
		sseWriter := writers.NewHTTPStreamWriter(w, connState, requestID, false)

		// Create streaming pipeline using factory
		factory := NewStreamFactory()
		handler := factory.CreateGeminiPipeline(streamIter, requestID, provider, cacheSource)

		// Handle the stream and capture error
		if err := handler.Handle(fasthttpCtx, sseWriter); err != nil {
			if !contracts.IsExpectedError(err) {
				fiberlog.Errorf("[%s] Stream error: %v", requestID, err)
				errCh <- err // Send error to channel for caller
			} else {
				fiberlog.Infof("[%s] Stream ended: %v", requestID, err)
				errCh <- nil // Expected error is treated as success
			}
		} else {
			errCh <- nil // No error
		}
	}))

	// Wait for the stream to complete and return any error
	return <-errCh
}
