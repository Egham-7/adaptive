package vercel

import (
	"fmt"
	"io"
	"log"
	"strings"
	"sync"

	"github.com/openai/openai-go"
	ssestream "github.com/openai/openai-go/packages/ssestream"
)

type OpenAIAdapter struct {
	stream    *ssestream.Stream[openai.ChatCompletionChunk]
	requestID string
	done      bool
	closeLock sync.Once
}

func NewOpenAIAdapter(stream *ssestream.Stream[openai.ChatCompletionChunk], requestID string) *OpenAIAdapter {
	return &OpenAIAdapter{
		stream:    stream,
		requestID: requestID,
	}
}

func (o *OpenAIAdapter) NextChunk() (*InternalProviderChunk, error) {
	if o.done {
		return nil, io.EOF
	}

	if !o.stream.Next() {
		if err := o.stream.Err(); err != nil && !strings.Contains(err.Error(), "EOF") {
			log.Printf("[%s] OpenAI stream error: %v", o.requestID, err)
			return &InternalProviderChunk{
				Error: fmt.Sprintf("OpenAI stream error: %s", err.Error()),
			}, nil
		}
		o.done = true
		return &InternalProviderChunk{
			FinishReason: "stop",
		}, io.EOF
	}

	chunk := o.stream.Current()

	if len(chunk.Choices) == 0 {
		return nil, nil
	}

	choice := chunk.Choices[0]
	internal := &InternalProviderChunk{
		Text: choice.Delta.Content,
	}

	// Handle stream finish
	if choice.FinishReason != "" {
		internal.FinishReason = choice.FinishReason
		o.done = true
	}

	return internal, nil
}

func (o *OpenAIAdapter) Close() error {
	o.closeLock.Do(func() {
		log.Printf("[%s] OpenAI adapter closed", o.requestID)
		o.done = true
		_ = o.stream.Close()
	})
	return nil
}
