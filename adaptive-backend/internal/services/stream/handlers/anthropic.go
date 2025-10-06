package handlers

import (
	"bufio"

	"adaptive-backend/internal/services/stream/contracts"
	"adaptive-backend/internal/services/stream/writers"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/valyala/fasthttp"
)

// HandleAnthropicNative handles native Anthropic SDK streams using proper layered architecture
func HandleAnthropicNative(c *fiber.Ctx, stream *ssestream.Stream[anthropic.MessageStreamEventUnion], requestID, provider, cacheSource string) error {
	fiberlog.Infof("[%s] Starting native Anthropic stream handling", requestID)

	// Create streaming pipeline - validates stream internally by reading first event
	// If validation fails (429, 500, etc.), error is returned BEFORE HTTP streaming starts
	factory := NewStreamFactory()
	handler, err := factory.CreateAnthropicNativePipeline(stream, requestID, provider, cacheSource)
	if err != nil {
		fiberlog.Errorf("[%s] Stream validation failed: %v", requestID, err)
		return err
	}

	fiberlog.Infof("[%s] Stream validated successfully, starting HTTP stream", requestID)

	fasthttpCtx := c.Context()
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Access-Control-Allow-Origin", "*")

	fasthttpCtx.SetBodyStreamWriter(fasthttp.StreamWriter(func(w *bufio.Writer) {
		// Create connection state tracker
		connState := writers.NewFastHTTPConnectionState(fasthttpCtx)

		// Create HTTP writer (with [DONE] message for Anthropic compatibility)
		httpWriter := writers.NewHTTPStreamWriter(w, connState, requestID, true)

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
