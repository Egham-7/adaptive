package handlers

import (
	"bufio"
	"io"

	"adaptive-backend/internal/stream/contracts"
	"adaptive-backend/internal/stream/writers"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/valyala/fasthttp"
)

// HandleAnthropic manages Anthropic streaming response using proper layered architecture
func HandleAnthropic(c *fiber.Ctx, responseBody io.Reader, requestID, provider string) error {
	fiberlog.Infof("[%s] Starting Anthropic stream handling", requestID)

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
		handler := factory.CreateAnthropicPipeline(responseBody, requestID, provider)
		
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

// HandleAnthropicNative handles native Anthropic SDK streams using proper layered architecture
func HandleAnthropicNative(c *fiber.Ctx, stream *ssestream.Stream[anthropic.MessageStreamEventUnion], requestID, provider string) error {
	fiberlog.Infof("[%s] Starting native Anthropic stream handling", requestID)

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
		handler := factory.CreateAnthropicNativePipeline(stream, requestID, provider)
		
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