package models

import (
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
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
	ServiceTier              string `json:"service_tier,omitzero"`
	// Cache tier information for adaptive system
	CacheTier string `json:"cache_tier,omitzero"` // e.g., "semantic_exact", "semantic_similar", "prompt_response"
}

type TextCitationUnion struct {
	CitedText     string `json:"cited_text,omitzero"`
	DocumentIndex int64  `json:"document_index,omitzero"`
	DocumentTitle string `json:"document_title,omitzero"`
	// This field is from variant [CitationCharLocation].
	EndCharIndex int64  `json:"end_char_index,omitzero"`
	FileID       string `json:"file_id,omitzero"`
	// This field is from variant [CitationCharLocation].
	StartCharIndex int64 `json:"start_char_index,omitzero"`
	// Any of "char_location", "page_location", "content_block_location",
	// "web_search_result_location", "search_result_location".
	Type string `json:"type"`
	// This field is from variant [CitationPageLocation].
	EndPageNumber int64 `json:"end_page_number,omitzero"`
	// This field is from variant [CitationPageLocation].
	StartPageNumber int64 `json:"start_page_number,omitzero"`
	EndBlockIndex   int64 `json:"end_block_index,omitzero"`
	StartBlockIndex int64 `json:"start_block_index,omitzero"`
	// This field is from variant [CitationsWebSearchResultLocation].
	EncryptedIndex string `json:"encrypted_index,omitzero"`
	Title          string `json:"title,omitzero"`
	// This field is from variant [CitationsWebSearchResultLocation].
	URL string `json:"url,omitzero"`
	// This field is from variant [CitationsSearchResultLocation].
	SearchResultIndex int64 `json:"search_result_index,omitzero"`
	// This field is from variant [CitationsSearchResultLocation].
	Source string `json:"source,omitzero"`
}

type WebSearchToolResultBlockContentUnion struct {
	// This field will be present if the value is a [[]WebSearchResultBlock] instead of
	// an object.
	OfWebSearchResultBlockArray []anthropic.WebSearchResultBlock `json:",inline,omitzero"`
	// This field is from variant [WebSearchToolResultError].
	ErrorCode anthropic.WebSearchToolResultErrorErrorCode `json:"error_code,omitzero"`
	// This field is from variant [WebSearchToolResultError].
	Type constant.WebSearchToolResultError `json:"type,omitzero"`
}

type ContentBlockUnion struct {
	// This field is from variant [TextBlock].
	Citations []TextCitationUnion `json:"citations,omitzero"`
	// This field is from variant [TextBlock].
	Text string `json:"text,omitzero"`
	// Any of "text", "thinking", "redacted_thinking", "tool_use", "server_tool_use",
	// "web_search_tool_result".
	Type string `json:"type"`
	// This field is from variant [ThinkingBlock].
	Signature string `json:"signature,omitzero"`
	// This field is from variant [ThinkingBlock].
	Thinking string `json:"thinking,omitzero"`
	// This field is from variant [RedactedThinkingBlock].
	Data string `json:"data,omitzero"`
	ID   string `json:"id,omitzero"`
	// necessary custom code modification
	Input any    `json:"input,omitzero"`
	Name  string `json:"name,omitzero"`
	// This field is from variant [WebSearchToolResultBlock].
	Content anthropic.WebSearchToolResultBlockContentUnion `json:"content,omitzero"`
	// This field is from variant [WebSearchToolResultBlock].
	ToolUseID string `json:"tool_use_id,omitzero"`
}

// AnthropicMessage extends Anthropic's Message with enhanced usage and provider info
type AnthropicMessage struct {
	ID           string                 `json:"id"`
	Content      []ContentBlockUnion    `json:"content,omitzero"`
	Model        string                 `json:"model"`
	Role         string                 `json:"role"`
	StopReason   string                 `json:"stop_reason,omitzero"`
	StopSequence string                 `json:"stop_sequence,omitzero"`
	Type         string                 `json:"type"`
	Usage        AdaptiveAnthropicUsage `json:"usage,omitzero"`
	Provider     string                 `json:"provider,omitzero"`
}

// Custom Anthropic streaming event types that produce clean, minimal JSON

// AdaptiveTextDelta represents a text content delta event
type AdaptiveTextDelta struct {
	Type string `json:"type"` // "text_delta"
	Text string `json:"text"`
}

// AdaptiveDelta represents delta information for both message and content block events
type AdaptiveDelta struct {
	// Message delta fields
	StopReason   string `json:"stop_reason,omitzero"`
	StopSequence string `json:"stop_sequence,omitzero"`

	// Content block delta fields
	Type        string `json:"type,omitzero"`
	Text        string `json:"text,omitzero"`
	PartialJSON string `json:"partial_json,omitzero"`
	Thinking    string `json:"thinking,omitzero"`
	Signature   string `json:"signature,omitzero"`
}

// AdaptiveMessageStartEvent represents the message_start event
type AdaptiveMessageStartEvent struct {
	Type    string           `json:"type"` // "message_start"
	Message AnthropicMessage `json:"message"`
}

// AdaptiveContentBlockDeltaEvent represents content_block_delta events
type AdaptiveContentBlockDeltaEvent struct {
	Type  string            `json:"type"` // "content_block_delta"
	Index int               `json:"index"`
	Delta AdaptiveTextDelta `json:"delta"`
}

// AdaptiveMessageDeltaEvent represents message_delta events (with stop reason and usage)
type AdaptiveMessageDeltaEvent struct {
	Type  string                  `json:"type"` // "message_delta"
	Delta AdaptiveDelta           `json:"delta"`
	Usage *AdaptiveAnthropicUsage `json:"usage,omitzero"`
}

// AnthropicMessageChunk matches Anthropic's streaming format exactly, with our provider extension
type AnthropicMessageChunk struct {
	Type string `json:"type"`

	// Fields for different event types - only populated based on event type
	Message      *AnthropicMessage                                  `json:"message,omitzero"`       // message_start only
	Delta        *AdaptiveDelta                                     `json:"delta,omitzero"`         // content_block_delta, message_delta
	Usage        *AdaptiveAnthropicUsage                            `json:"usage,omitzero"`         // message_delta only
	ContentBlock *anthropic.ContentBlockStartEventContentBlockUnion `json:"content_block,omitzero"` // content_block_start only
	Index        *int64                                             `json:"index,omitzero"`         // content_block_start, content_block_delta, content_block_stop

	// Adaptive-specific fields
	Provider string `json:"provider,omitzero"` // Keep this for internal tracking, but it will be omitted when empty
}

// Helper constructors for clean event creation

// NewMessageStartEvent creates a clean message_start event
func NewMessageStartEvent(message AnthropicMessage) AdaptiveMessageStartEvent {
	return AdaptiveMessageStartEvent{
		Type:    "message_start",
		Message: message,
	}
}

// NewContentBlockDeltaEvent creates a clean content_block_delta event for text
func NewContentBlockDeltaEvent(index int, text string) AdaptiveContentBlockDeltaEvent {
	return AdaptiveContentBlockDeltaEvent{
		Type:  "content_block_delta",
		Index: index,
		Delta: AdaptiveTextDelta{
			Type: "text_delta",
			Text: text,
		},
	}
}

// NewMessageDeltaEvent creates a clean message_delta event with stop reason and usage
func NewMessageDeltaEvent(stopReason string, usage *AdaptiveAnthropicUsage) AdaptiveMessageDeltaEvent {
	return AdaptiveMessageDeltaEvent{
		Type: "message_delta",
		Delta: AdaptiveDelta{
			StopReason: stopReason,
		},
		Usage: usage,
	}
}
