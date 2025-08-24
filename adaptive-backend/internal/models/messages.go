package models

import "github.com/anthropics/anthropic-sdk-go"

// AnthropicMessageRequest extends anthropic.MessageNewParams with our custom fields
type AnthropicMessageRequest struct {
	anthropic.MessageNewParams

	// Custom fields for our internal processing
	ModelRouterConfig   *ModelRouterConfig         `json:"model_router,omitempty"`
	PromptResponseCache *CacheConfig               `json:"prompt_response_cache,omitempty"` // Optional prompt response cache configuration
	PromptCache         *PromptCacheConfig         `json:"prompt_cache,omitempty"`          // Optional prompt response cache configuration
	Fallback            *FallbackConfig            `json:"fallback,omitempty"`              // Fallback configuration with enabled toggle
	ProviderConfigs     map[string]*ProviderConfig `json:"provider_configs,omitempty"`      // Custom provider configurations by provider name
}

// ToAnthropicParams converts AnthropicMessageRequest to Anthropic's MessageNewParams
func (r *AnthropicMessageRequest) ToAnthropicParams() *anthropic.MessageNewParams {
	return &r.MessageNewParams
}

// AdaptiveAnthropicUsage extends Anthropic's Usage with cache tier information
type AdaptiveAnthropicUsage struct {
	CacheCreationInputTokens int64  `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64  `json:"cache_read_input_tokens"`
	InputTokens              int64  `json:"input_tokens"`
	OutputTokens             int64  `json:"output_tokens"`
	ServiceTier              string `json:"service_tier,omitempty"`
	// Cache tier information for adaptive system
	CacheTier string `json:"cache_tier,omitempty"` // e.g., "semantic_exact", "semantic_similar", "prompt_response"
}

// AnthropicMessage extends Anthropic's Message with enhanced usage and provider info
type AnthropicMessage struct {
	ID           string                        `json:"id"`
	Content      []anthropic.ContentBlockUnion `json:"content"`
	Model        string                        `json:"model"`
	Role         string                        `json:"role"`
	StopReason   string                        `json:"stop_reason"`
	StopSequence string                        `json:"stop_sequence"`
	Type         string                        `json:"type"`
	Usage        AdaptiveAnthropicUsage        `json:"usage"`
	Provider     string                        `json:"provider,omitempty"`
}

// AnthropicMessageChunk extends Anthropic's MessageStreamEventUnion with enhanced usage and provider info
type AnthropicMessageChunk struct {
	Type         string                                             `json:"type"`
	Message      *AnthropicMessage                                  `json:"message,omitempty"`
	Delta        *anthropic.MessageStreamEventUnionDelta            `json:"delta,omitempty"`
	Usage        *AdaptiveAnthropicUsage                            `json:"usage,omitempty"`
	ContentBlock *anthropic.ContentBlockStartEventContentBlockUnion `json:"content_block,omitempty"`
	Index        *int64                                             `json:"index,omitempty"`
	Provider     string                                             `json:"provider,omitempty"`
}
