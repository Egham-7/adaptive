package models

import (
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"
)

// PromptCacheKey represents the structure used to generate cache keys
type PromptCacheKey struct {
	Messages    []openai.ChatCompletionMessageParamUnion        `json:"messages"`
	Model       shared.ChatModel                                `json:"model"`
	Temperature param.Opt[float64]                              `json:"temperature"`
	MaxTokens   param.Opt[int64]                                `json:"max_tokens"`
	TopP        param.Opt[float64]                              `json:"top_p"`
	Tools       []openai.ChatCompletionToolParam                `json:"tools,omitempty"`
	ToolChoice  openai.ChatCompletionToolChoiceOptionUnionParam `json:"tool_choice"`
}

// PromptCacheStats represents cache statistics
type PromptCacheStats struct {
	PromptCacheKeys int            `json:"prompt_cache_keys"`
	RedisInfo       string         `json:"redis_info"`
	Timestamp       time.Time      `json:"timestamp"`
	TTLStats        map[string]any `json:"ttl_stats,omitempty"`
}

// PromptCacheEntry represents a cached entry with metadata
type PromptCacheEntry struct {
	Response  *ChatCompletion `json:"response"`
	CreatedAt time.Time       `json:"created_at"`
	TTL       time.Duration   `json:"ttl"`
	Key       string          `json:"key"`
}
