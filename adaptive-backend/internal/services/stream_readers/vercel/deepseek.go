package vercel

import (
	"fmt"
	"io"
	"log"
	"strings"
	"sync"

	deepseek "github.com/cohesion-org/deepseek-go"
)

// DeepSeekAdapter adapts DeepSeek's streaming responses to Vercel's DataStream format
type DeepSeekAdapter struct {
	stream    deepseek.ChatCompletionStream
	requestID string
	done      bool
	closeLock sync.Once
}

// NewDeepSeekAdapter creates a new DeepSeek adapter
func NewDeepSeekAdapter(stream deepseek.ChatCompletionStream, requestID string) *DeepSeekAdapter {
	return &DeepSeekAdapter{
		stream:    stream,
		requestID: requestID,
	}
}

// NextChunk fetches the next available chunk and wraps it as InternalProviderChunk
func (d *DeepSeekAdapter) NextChunk() (*InternalProviderChunk, error) {
	if d.done {
		return nil, io.EOF
	}

	resp, err := d.stream.Recv()
	if err != nil {
		if strings.Contains(err.Error(), "EOF") || strings.Contains(err.Error(), "stream closed") {
			d.done = true
			return &InternalProviderChunk{FinishReason: "stop"}, io.EOF
		}
		log.Printf("[%s] DeepSeek adapter stream error: %v", d.requestID, err)
		return &InternalProviderChunk{Error: fmt.Sprintf("DeepSeek stream error: %s", err.Error())}, nil
	}

	if len(resp.Choices) == 0 {
		return nil, nil
	}

	choice := resp.Choices[0]

	if choice.FinishReason != "" {
		d.done = true
		return &InternalProviderChunk{FinishReason: choice.FinishReason}, io.EOF
	}

	return &InternalProviderChunk{Text: choice.Delta.Content}, nil
}

// Close terminates the stream
func (d *DeepSeekAdapter) Close() error {
	d.closeLock.Do(func() {
		log.Printf("[%s] DeepSeek adapter closed", d.requestID)
		d.done = true
		_ = d.stream.Close()
	})
	return nil
}
