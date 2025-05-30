package vercel

import (
	"fmt"
	"io"
	"log"
	"sync"

	"github.com/conneroisu/groq-go"
)

// GroqAdapter adapts Groq's streaming responses to Vercel's DataStream format
type GroqAdapter struct {
	stream    *groq.ChatCompletionStream
	requestID string
	done      bool
	closeLock sync.Once
}

// NewGroqAdapter initializes a new GroqAdapter
func NewGroqAdapter(stream *groq.ChatCompletionStream, requestID string) *GroqAdapter {
	return &GroqAdapter{
		stream:    stream,
		requestID: requestID,
	}
}

// NextChunk processes the next chunk from the Groq stream
func (g *GroqAdapter) NextChunk() (*InternalProviderChunk, error) {
	if g.done {
		return nil, io.EOF
	}

	event, err := g.stream.Recv()
	if err != nil {
		if err == io.EOF {
			g.done = true
			return &InternalProviderChunk{
				FinishReason: "stop",
			}, io.EOF
		}
		log.Printf("[%s] Groq stream error: %v", g.requestID, err)
		return &InternalProviderChunk{
			Error: fmt.Sprintf("Groq stream error: %s", err.Error()),
		}, nil
	}

	if len(event.Choices) > 0 {
		delta := event.Choices[0].Delta
		return &InternalProviderChunk{
			Text: delta.Content,
		}, nil
	}

	return nil, nil
}

// Close terminates the Groq stream
func (g *GroqAdapter) Close() error {
	g.closeLock.Do(func() {
		log.Printf("[%s] Groq adapter closed", g.requestID)
		g.done = true
	})
	return nil
}
