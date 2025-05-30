package vercel

import "io"

type ProviderAdapter interface {
	NextChunk() (*InternalProviderChunk, error)
	io.Closer
}
