package vercel

import (
	"fmt"
	"io"
	"log"
	"sync"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
)

type AnthropicAdapter struct {
	stream    ssestream.Stream[anthropic.MessageStreamEventUnion]
	requestID string
	done      bool
	closeLock sync.Once
}

func NewAnthropicAdapter(stream ssestream.Stream[anthropic.MessageStreamEventUnion], requestID string) *AnthropicAdapter {
	return &AnthropicAdapter{
		stream:    stream,
		requestID: requestID,
	}
}

func (a *AnthropicAdapter) NextChunk() (*InternalProviderChunk, error) {
	if a.done {
		return nil, io.EOF
	}

	if !a.stream.Next() {
		if a.stream.Err() != nil {
			log.Printf("[%s] Anthropic stream error: %v", a.requestID, a.stream.Err())
			return &InternalProviderChunk{
				Error: fmt.Sprintf("Anthropic stream error: %s", a.stream.Err().Error()),
			}, nil
		}
		a.done = true
		return &InternalProviderChunk{
			FinishReason: "stop",
		}, io.EOF
	}

	event := a.stream.Current()
	switch val := event.AsAny().(type) {
	case anthropic.ContentBlockDeltaEvent:
		switch delta := val.Delta.AsAny().(type) {
		case anthropic.TextDelta:
			return &InternalProviderChunk{Text: delta.Text}, nil
		default:
			return nil, nil
		}
	case anthropic.MessageStopEvent:
		a.done = true
		return &InternalProviderChunk{FinishReason: "stop"}, io.EOF
	default:
		return nil, nil
	}
}

func (a *AnthropicAdapter) Close() error {
	a.closeLock.Do(func() {
		log.Printf("[%s] Anthropic adapter closed", a.requestID)
		a.done = true
	})
	return nil
}
