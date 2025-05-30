package vercel

import (
	"adaptive-backend/internal/services/stream_readers"
	"encoding/json"
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

// VercelDataStreamReader adapts a generic stream to Vercel's AI SDK DataStream format.
type VercelDataStreamReader struct {
	stream_readers.BaseStreamReader
	underlyingStream   io.ReadCloser
	vercelFormatHeader string
	done               bool
	sentFinish         bool
	decoder            *json.Decoder
}

// NewVercelDataStreamReader creates a new stream reader for Vercel DataStream format.
func NewVercelDataStreamReader(stream io.ReadCloser, requestID string) *VercelDataStreamReader {
	return &VercelDataStreamReader{
		BaseStreamReader: stream_readers.BaseStreamReader{
			Buffer:    []byte{},
			RequestID: requestID,
		},
		underlyingStream:   stream,
		vercelFormatHeader: "x-vercel-ai-data-stream: v1",
		done:               false,
		sentFinish:         false,
		decoder:            json.NewDecoder(stream),
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

	var chunk InternalProviderChunk
	if err := r.decoder.Decode(&chunk); err != nil {
		if err == io.EOF {
			if !r.sentFinish {
				finishMsg := `d:{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0}}` + "\n\n"
				r.Buffer = []byte(finishMsg)
				r.sentFinish = true
				return r.Read(p)
			}
			return 0, io.EOF
		}
		log.Printf("[%s] Error decoding chunk from underlying stream: %v", r.RequestID, err)
		errMsg := fmt.Sprintf(`3:"Error reading from underlying stream: %s"`+"\n\n", strings.ReplaceAll(err.Error(), `"`, `\"`))
		r.Buffer = []byte(errMsg)
		r.done = true
		return r.Read(p)
	}

	if chunk.Error != "" {
		errMsg := fmt.Sprintf(`3:"%s"`+"\n\n", strings.ReplaceAll(chunk.Error, `"`, `\"`))
		r.Buffer = append(r.Buffer, []byte(errMsg)...)
	} else if chunk.Text != "" {
		escapedText := strings.ReplaceAll(chunk.Text, `"`, `\"`)
		escapedText = strings.ReplaceAll(escapedText, "\n", `\n`)
		textMsg := fmt.Sprintf(`0:"%s"`+"\n\n", escapedText)
		r.Buffer = append(r.Buffer, []byte(textMsg)...)
	}

	if chunk.FinishReason != "" && !r.sentFinish {
		finishMsg := fmt.Sprintf(`d:{"finishReason":"%s","usage":{"promptTokens":0,"completionTokens":0}}`+"\n\n", chunk.FinishReason)
		r.Buffer = append(r.Buffer, []byte(finishMsg)...)
		r.sentFinish = true
		r.done = true
	}

	return r.Read(p)
}

// Close closes the underlying stream safely
func (r *VercelDataStreamReader) Close() error {
	var err error
	r.CloseLock.Do(func() {
		if r.underlyingStream != nil {
			log.Printf("[%s] Closing underlying stream for VercelDataStreamReader", r.RequestID)
			err = r.underlyingStream.Close()
		}
		r.done = true
		log.Printf("[%s] VercelDataStreamReader closed, requestID: %s", r.RequestID, r.RequestID)
	})
	return err
}
