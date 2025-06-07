package sse

import (
	"adaptive-backend/internal/services/stream_readers"
	"encoding/json"
	"errors"
	"io"
	"log"
	"strings"

	"github.com/openai/openai-go"
	ssestream "github.com/openai/openai-go/packages/ssestream"
)

const (
	sseDataPrefix  = "data: "
	sseLineSuffix  = "\n\n"
	sseDoneMessage = "data: [DONE]\n\n"
)

// TODO: Remove chunk cleaning logic once AI SDK integration is complete.
// The SDK should provide clean chunks without empty fields by default.

type OpenAIStreamReader struct {
	stream_readers.BaseStreamReader
	stream *ssestream.Stream[openai.ChatCompletionChunk]
	done   bool
	buf    strings.Builder
}

func NewOpenAIStreamReader(
	stream *ssestream.Stream[openai.ChatCompletionChunk],
	requestID string,
) *OpenAIStreamReader {
	r := &OpenAIStreamReader{
		BaseStreamReader: stream_readers.BaseStreamReader{
			Buffer:    make([]byte, 0, 1024),
			RequestID: requestID,
		},
		stream: stream,
	}
	r.buf.Grow(512)
	return r
}

func (r *OpenAIStreamReader) Read(p []byte) (n int, err error) {
	if len(r.Buffer) > 0 {
		n = copy(p, r.Buffer)
		r.Buffer = r.Buffer[n:]
		return n, nil
	}

	if r.done {
		return 0, io.EOF
	}

	if !r.stream.Next() {
		if err := r.stream.Err(); err == nil || errors.Is(err, io.EOF) {
			r.Buffer = []byte(sseDoneMessage)
			r.done = true
			return r.Read(p)
		}
		return r.handleError(r.stream.Err(), p)
	}

	chunk := r.stream.Current()
	if err := r.processChunk(&chunk); err != nil {
		return r.handleError(err, p)
	}

	return r.Read(p)
}

func (r *OpenAIStreamReader) handleError(err error, p []byte) (int, error) {
	log.Printf("[%s] OpenAI stream error: %v", r.RequestID, err)

	r.buf.Reset()
	r.buf.WriteString(sseDataPrefix)
	r.buf.WriteString(`{"error": "`)
	r.buf.WriteString(strings.ReplaceAll(strings.ReplaceAll(err.Error(), "\"", "\\\""), "\n", "\\n"))
	r.buf.WriteString(`"}`)
	r.buf.WriteString(sseLineSuffix)

	r.Buffer = []byte(r.buf.String())
	r.done = true
	return r.Read(p)
}

// processChunk orchestrates the transformation and buffering of a chunk.
func (r *OpenAIStreamReader) processChunk(chunk *openai.ChatCompletionChunk) error {
	outputMap := map[string]any{
		"id":      chunk.ID,
		"object":  chunk.Object,
		"created": chunk.Created,
		"model":   chunk.Model,
	}

	if usageMap := buildUsageMap(&chunk.Usage); usageMap != nil {
		outputMap["usage"] = usageMap
	}

	if choicesList := buildChoicesList(chunk.Choices); len(choicesList) > 0 {
		outputMap["choices"] = choicesList
	}

	jsonData, err := json.Marshal(outputMap)
	if err != nil {
		return err
	}

	r.buf.Reset()
	r.buf.WriteString(sseDataPrefix)
	r.buf.Write(jsonData)
	r.buf.WriteString(sseLineSuffix)
	r.Buffer = []byte(r.buf.String())

	r.updateStreamStatus(chunk)
	return nil
}

// buildUsageMap creates the usage map, converting tokens to int64.
// Returns nil if usage is nil or empty to prevent it from being marshaled.
func buildUsageMap(usage *openai.CompletionUsage) map[string]int64 {
	if usage == nil || usage.TotalTokens == 0 {
		return nil
	}
	return map[string]int64{
		"prompt_tokens":     int64(usage.PromptTokens),
		"completion_tokens": int64(usage.CompletionTokens),
		"total_tokens":      int64(usage.TotalTokens),
	}
}

// buildChoicesList iterates over choices and builds a list of choice maps.
func buildChoicesList(choices []openai.ChatCompletionChunkChoice) []map[string]any {
	if len(choices) == 0 {
		return nil
	}
	list := make([]map[string]any, 0, len(choices))
	for _, choice := range choices {
		list = append(list, buildChoiceMap(choice))
	}
	return list
}

// buildChoiceMap creates a map for a single choice, including its delta.
func buildChoiceMap(choice openai.ChatCompletionChunkChoice) map[string]any {
	choiceMap := map[string]any{
		"index": choice.Index,
		"delta": buildDeltaMap(choice.Delta),
	}
	if choice.FinishReason != "" {
		choiceMap["finish_reason"] = choice.FinishReason
	}
	return choiceMap
}

// buildDeltaMap creates a map for a delta, omitting empty fields.
func buildDeltaMap(delta openai.ChatCompletionChunkChoiceDelta) map[string]any {
	deltaMap := make(map[string]any)
	if delta.Content != "" {
		deltaMap["content"] = delta.Content
	}
	if delta.Role != "" {
		deltaMap["role"] = delta.Role
	}
	if len(delta.ToolCalls) > 0 {
		deltaMap["tool_calls"] = delta.ToolCalls
	}
	return deltaMap
}

// updateStreamStatus checks the chunk and updates the reader's done status.
func (r *OpenAIStreamReader) updateStreamStatus(chunk *openai.ChatCompletionChunk) {
	if len(chunk.Choices) > 0 {
		allFinished := true
		for i := range chunk.Choices {
			if chunk.Choices[i].FinishReason == "" {
				allFinished = false
				break
			}
		}
		r.done = allFinished
	} else if chunk.Usage.TotalTokens > 0 {
		// This handles the final usage-only chunk.
		r.done = true
	}
}

func (r *OpenAIStreamReader) Close() error {
	var err error
	r.CloseLock.Do(func() {
		r.done = true
		r.Buffer = nil
		if r.stream != nil {
			err = r.stream.Close()
		}
		log.Printf("[%s] OpenAI stream closed", r.RequestID)
	})
	return err
}
