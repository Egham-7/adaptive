package models

import (
	"time"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
	"github.com/openai/openai-go/v2/shared"
)

// PromptCacheKey represents the structure used to generate cache keys
type PromptCacheKey struct {
	Messages    []openai.ChatCompletionMessageParamUnion        `json:"messages"`
	Model       shared.ChatModel                                `json:"model"`
	Temperature param.Opt[float64]                              `json:"temperature"`
	MaxTokens   param.Opt[int64]                                `json:"max_tokens"`
	TopP        param.Opt[float64]                              `json:"top_p"`
	Tools       []openai.ChatCompletionToolMessageParam         `json:"tools,omitzero"`
	ToolChoice  openai.ChatCompletionToolChoiceOptionUnionParam `json:"tool_choice"`
}

// PromptCacheEntry represents a cached entry with metadata
type PromptCacheEntry struct {
	Response  *ChatCompletion `json:"response"`
	CreatedAt time.Time       `json:"created_at"`
	TTL       time.Duration   `json:"ttl"`
	Key       string          `json:"key"`
}
