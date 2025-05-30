package vercel

import (
	"adaptive-backend/internal/services/stream_readers"
	"fmt"
	"io"
	"log"
	"strings"
)

// InternalProviderChunk represents a simplified internal chunk structure
type InternalProviderChunk struct {
	Text         string `json:"text,omitempty"`
	ToolCalls    []any  `json:"tool_calls,omitempty"`
	Error        string `json:"error,omitempty"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// VercelDataStreamReader emits data formatted for Vercel's AI SDK (DataStream protocol)
type VercelDataStreamReader struct {
	stream_readers.BaseStreamReader
	adapter    ProviderAdapter
	done       bool
	sentFinish bool
}

// NewVercelDataStreamReaderFromAdapter creates a new reader from a provider adapter
func NewVercelDataStreamReaderFromAdapter(adapter ProviderAdapter, requestID string) *VercelDataStreamReader {
	return &VercelDataStreamReader{
		BaseStreamReader: stream_readers.BaseStreamReader{
			Buffer:    []byte{},
			RequestID: requestID,
		},
		adapter:    adapter,
		sentFinish: false,
		done:       false,
	}
}

// Read implements io.Reader interface for Vercel DataStream format
func (r *VercelDataStreamReader) Read(p []byte) (n int, err error) {
	if len(r.Buffer) > 0 {
		n = copy(p, r.Buffer)
		r.Buffer = r.Buffer[n:]
		return n, nil
	}

	if r.done {
		return 0, io.EOF
	}

	chunk, err := r.adapter.NextChunk()
	if err != nil && err != io.EOF {
		log.Printf("[%s] Adapter error: %v", r.RequestID, err)
		errMsg := fmt.Sprintf(`3:"Adapter error: %s"`+"\n\n", strings.ReplaceAll(err.Error(), `"`, `\"`))
		r.Buffer = []byte(errMsg)
		r.done = true
		return r.Read(p)
	}

	if chunk == nil {
		return r.Read(p)
	}

	if chunk.Error != "" {
		r.Buffer = []byte(fmt.Sprintf(`3:"%s"`+"\n\n", strings.ReplaceAll(chunk.Error, `"`, `\"`)))
		r.done = true
		return r.Read(p)
	}

	if chunk.Text != "" {
		escaped := strings.ReplaceAll(chunk.Text, `"`, `\"`)
		escaped = strings.ReplaceAll(escaped, "\n", `\n`)
		r.Buffer = []byte(fmt.Sprintf(`0:"%s"`+"\n\n", escaped))
	}

	if chunk.FinishReason != "" && !r.sentFinish {
		r.Buffer = append(r.Buffer,
			[]byte(fmt.Sprintf(`d:{"finishReason":"%s","usage":{"promptTokens":0,"completionTokens":0}}`+"\n\n", chunk.FinishReason))...,
		)
		r.sentFinish = true
		r.done = true
	}

	return r.Read(p)
}

// Close releases any underlying resources
func (r *VercelDataStreamReader) Close() error {
	var err error
	r.CloseLock.Do(func() {
		err = r.adapter.Close()
		log.Printf("[%s] VercelDataStreamReader closed", r.RequestID)
		r.done = true
	})
	return err
}
