package processors

import (
	"context"
)

// PassthroughProcessor passes data through without any modification
// Used when no format conversion is needed
type PassthroughProcessor struct {
	provider  string
	requestID string
}

// NewPassthroughProcessor creates a new passthrough processor
func NewPassthroughProcessor(provider, requestID string) *PassthroughProcessor {
	return &PassthroughProcessor{
		provider:  provider,
		requestID: requestID,
	}
}

// Process passes data through without any modification
func (p *PassthroughProcessor) Process(ctx context.Context, data []byte) ([]byte, error) {
	// Return data as-is without any processing
	return data, nil
}

// Provider returns the provider name
func (p *PassthroughProcessor) Provider() string {
	return p.provider
}
