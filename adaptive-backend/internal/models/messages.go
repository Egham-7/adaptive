package models

import (
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
)

// AnthropicMessageRequest uses individual fields from anthropic.MessageNewParams with our custom fields
type AnthropicMessageRequest struct {
	// Core Anthropic Messages API fields (from anthropic.MessageNewParams)
	MaxTokens     int64                                 `json:"max_tokens,omitzero"`
	Messages      []anthropic.MessageParam              `json:"messages"`
	Model         anthropic.Model                       `json:"model"`
	Temperature   param.Opt[float64]                    `json:"temperature,omitzero"`
	TopK          param.Opt[int64]                      `json:"top_k,omitzero"`
	TopP          param.Opt[float64]                    `json:"top_p,omitzero"`
	Metadata      anthropic.MetadataParam               `json:"metadata,omitzero"`
	ServiceTier   anthropic.MessageNewParamsServiceTier `json:"service_tier,omitzero"`
	StopSequences []string                              `json:"stop_sequences,omitzero"`
	System        []anthropic.TextBlockParam            `json:"system,omitzero"`
	Stream        *bool                                 `json:"stream,omitzero"`
	Thinking      anthropic.ThinkingConfigParamUnion    `json:"thinking,omitzero"`
	ToolChoice    anthropic.ToolChoiceUnionParam        `json:"tool_choice,omitzero"`
	Tools         []anthropic.ToolUnionParam            `json:"tools,omitzero"`

	// Custom fields for our internal processing
	ModelRouterConfig   *ModelRouterConfig         `json:"model_router,omitzero"`
	PromptResponseCache *CacheConfig               `json:"prompt_response_cache,omitzero"` // Optional prompt response cache configuration
	PromptCache         *CacheConfig               `json:"prompt_cache,omitzero"`          // Optional prompt response cache configuration
	Fallback            *FallbackConfig            `json:"fallback,omitzero"`              // Fallback configuration with enabled toggle
	ProviderConfigs     map[string]*ProviderConfig `json:"provider_configs,omitzero"`      // Custom provider configurations by provider name
}

// AdaptiveAnthropicUsage extends Anthropic's Usage with cache tier information
type AdaptiveAnthropicUsage struct {
	CacheCreationInputTokens int64  `json:"cache_creation_input_tokens,omitzero"`
	CacheReadInputTokens     int64  `json:"cache_read_input_tokens,omitzero"`
	InputTokens              int64  `json:"input_tokens,omitzero"`
	OutputTokens             int64  `json:"output_tokens,omitzero"`
	ServiceTier              string `json:"service_tier,omitempty"`
	// Cache tier information for adaptive system
	CacheTier string `json:"cache_tier,omitempty"` // e.g., "semantic_exact", "semantic_similar", "prompt_response"
}

// AnthropicMessage extends Anthropic's Message with enhanced usage and provider info
type AnthropicMessage struct {
	ID           string                        `json:"id"`
	Content      []anthropic.ContentBlockUnion `json:"content,omitzero"`
	Model        string                        `json:"model"`
	Role         string                        `json:"role"`
	StopReason   string                        `json:"stop_reason,omitzero"`
	StopSequence string                        `json:"stop_sequence,omitzero"`
	Type         string                        `json:"type"`
	Usage        AdaptiveAnthropicUsage        `json:"usage,omitzero"`
	Provider     string                        `json:"provider,omitzero"`
}

// AnthropicMessageChunk matches Anthropic's streaming format exactly, with our provider extension
type AnthropicMessageChunk struct {
	Type string `json:"type"`

	// Fields for different event types - only populated based on event type
	Message      *AnthropicMessage                                  `json:"message,omitzero"`       // message_start only
	Delta        *anthropic.MessageStreamEventUnionDelta            `json:"delta,omitzero"`         // content_block_delta, message_delta
	Usage        *AdaptiveAnthropicUsage                            `json:"usage,omitzero"`         // message_delta only
	ContentBlock *anthropic.ContentBlockStartEventContentBlockUnion `json:"content_block,omitzero"` // content_block_start only
	Index        *int64                                             `json:"index,omitzero"`         // content_block_start, content_block_delta, content_block_stop

	// Adaptive-specific fields
	Provider string `json:"provider,omitzero"` // Keep this for internal tracking, but it will be omitted when empty
}
