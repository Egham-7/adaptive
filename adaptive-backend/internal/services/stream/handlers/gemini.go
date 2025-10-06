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

	// Create streaming pipeline - validates stream internally by reading first chunk
	// If validation fails (429, 500, etc.), error is returned BEFORE HTTP streaming starts
	factory := NewStreamFactory()
	handler, err := factory.CreateGeminiPipeline(streamIter, requestID, provider, cacheSource)
	if err != nil {
		fiberlog.Errorf("[%s] Stream validation failed: %v", requestID, err)
		return err
	}

	fiberlog.Infof("[%s] Stream validated successfully, starting HTTP stream", requestID)

	fasthttpCtx := c.Context()
	// Use SSE format for Gemini SDK compatibility (matches responseLineRE regex)
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Access-Control-Allow-Origin", "*")

	fasthttpCtx.SetBodyStreamWriter(fasthttp.StreamWriter(func(w *bufio.Writer) {
		// Create connection state tracker
		connState := writers.NewFastHTTPConnectionState(fasthttpCtx)

		// Create HTTP writer for SSE formatting (Gemini SDK expects SSE format without [DONE])
		sseWriter := writers.NewHTTPStreamWriter(w, connState, requestID, false)

		// Handle the stream
		if err := handler.Handle(fasthttpCtx, sseWriter); err != nil {
			if !contracts.IsExpectedError(err) {
				fiberlog.Errorf("[%s] Stream error: %v", requestID, err)
			} else {
				fiberlog.Infof("[%s] Stream ended: %v", requestID, err)
			}
		}
	}))

	return nil
}
