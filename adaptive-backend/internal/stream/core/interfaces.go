package core

import (
	"context"
	"io"
)

// StreamReader provides pure I/O reading interface
type StreamReader interface {
	io.Reader
	io.Closer
}

// ChunkProcessor handles format conversion and business logic
type ChunkProcessor interface {
	Process(ctx context.Context, data []byte) ([]byte, error)
	Provider() string
}

// StreamWriter handles output with flush capabilities
type StreamWriter interface {
	Write([]byte) error
	Flush() error
	Close() error
}

// StreamHandler orchestrates the streaming pipeline
type StreamHandler interface {
	Handle(ctx context.Context, writer StreamWriter) error
}

// ConnectionState tracks client connection status
type ConnectionState interface {
	IsConnected() bool
	Done() <-chan struct{}
}
