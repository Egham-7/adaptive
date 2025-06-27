package minions

import (
	"encoding/json"
	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

// Memory chunk decoder for streaming
type MemoryChunkDecoder struct {
	chunks []*openai.ChatCompletionChunk
	idx    int
	evt    ssestream.Event
}

func newMemoryChunkDecoder(chunks []*openai.ChatCompletionChunk) *MemoryChunkDecoder {
	return &MemoryChunkDecoder{chunks: chunks}
}

func (d *MemoryChunkDecoder) Next() bool {
	if d.idx >= len(d.chunks) {
		return false
	}
	chunk := d.chunks[d.idx]
	d.idx++
	data, err := json.Marshal(chunk)
	if err != nil {
		return false
	}
	d.evt = ssestream.Event{Type: "", Data: data}
	return true
}

func (d *MemoryChunkDecoder) Event() ssestream.Event { return d.evt }
func (d *MemoryChunkDecoder) Close() error           { return nil }
func (d *MemoryChunkDecoder) Err() error             { return nil }
