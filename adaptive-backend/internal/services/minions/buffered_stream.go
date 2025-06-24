package minions

import (
	"io"
)

// BufferedStream implements an in-memory ssestream.Stream[T]-compatible stream.
type BufferedStream[T any] struct {
	chunks []T
	idx    int
	closed bool
}

func NewBufferedStream[T any](chunks []T) *BufferedStream[T] {
	return &BufferedStream[T]{chunks: chunks}
}

func (b *BufferedStream[T]) Next() bool {
	if b.closed || b.idx >= len(b.chunks) {
		return false
	}
	b.idx++
	return true
}

func (b *BufferedStream[T]) Current() T {
	if b.idx == 0 || b.idx > len(b.chunks) {
		var zero T
		return zero
	}
	return b.chunks[b.idx-1]
}

func (b *BufferedStream[T]) Event() struct{ Data T } {
	return struct{ Data T }{Data: b.Current()}
}

func (b *BufferedStream[T]) Err() error {
	if b.closed {
		return io.EOF
	}
	return nil
}

func (b *BufferedStream[T]) Close() error {
	b.closed = true
	return nil
}
