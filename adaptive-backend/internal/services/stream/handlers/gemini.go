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
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Access-Control-Allow-Origin", "*")

	fasthttpCtx.SetBodyStreamWriter(fasthttp.StreamWriter(func(w *bufio.Writer) {
		// Create connection state tracker
		connState := writers.NewFastHTTPConnectionState(fasthttpCtx)

		// Create HTTP writer
		httpWriter := writers.NewHTTPStreamWriter(w, connState, requestID)

		// Create streaming pipeline using factory
		factory := NewStreamFactory()
		handler := factory.CreateGeminiPipeline(streamIter, requestID, provider, cacheSource)

		// Handle the stream
		if err := handler.Handle(fasthttpCtx, httpWriter); err != nil {
			if !contracts.IsExpectedError(err) {
				fiberlog.Errorf("[%s] Stream error: %v", requestID, err)
			} else {
				fiberlog.Infof("[%s] Stream ended: %v", requestID, err)
			}
		}
	}))

	return nil
}
