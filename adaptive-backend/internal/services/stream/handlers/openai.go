package handlers

import (
	"bufio"

	"adaptive-backend/internal/services/stream/contracts"
	"adaptive-backend/internal/services/stream/writers"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go/v2"
	openai_ssestream "github.com/openai/openai-go/v2/packages/ssestream"
	"github.com/valyala/fasthttp"
)

// HandleOpenAI manages OpenAI streaming response using proper layered architecture
func HandleOpenAI(c *fiber.Ctx, resp *openai_ssestream.Stream[openai.ChatCompletionChunk], requestID, provider, cacheSource string) error {
	fiberlog.Infof("[%s] Starting OpenAI stream handling", requestID)

	fasthttpCtx := c.Context()
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Access-Control-Allow-Origin", "*")

	fasthttpCtx.SetBodyStreamWriter(fasthttp.StreamWriter(func(w *bufio.Writer) {
		// Create connection state tracker
		connState := writers.NewFastHTTPConnectionState(fasthttpCtx)

		// Create HTTP writer (with [DONE] message for OpenAI compatibility)
		httpWriter := writers.NewHTTPStreamWriter(w, connState, requestID, true)

		// Create streaming pipeline using factory
		factory := NewStreamFactory()
		handler := factory.CreateOpenAIPipeline(resp, requestID, provider, cacheSource)

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
