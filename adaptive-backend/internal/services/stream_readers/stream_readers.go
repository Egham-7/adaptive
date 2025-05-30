package stream_readers

import (
	"io"
	"sync"
)

// StreamReader is the common interface for all LLM stream readers
type StreamReader interface {
	io.Reader
	io.Closer
}

// BaseStreamReader provides common functionality for all stream readers
type BaseStreamReader struct {
	Buffer    []byte
	RequestID string
	CloseLock sync.Once
}
