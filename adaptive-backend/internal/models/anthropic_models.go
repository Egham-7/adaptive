package models

import "github.com/anthropics/anthropic-sdk-go"

// AnthropicMessageRequest extends anthropic.MessageNewParams with our custom fields
type AnthropicMessageRequest struct {
	anthropic.MessageNewParams

	// Custom fields for our internal processing
	ProtocolManagerConfig *ProtocolManagerConfig     `json:"protocol_manager,omitempty"`
	SemanticCache         *CacheConfig               `json:"semantic_cache,omitempty"`   // Optional semantic cache configuration
	PromptCache           *PromptCacheConfig         `json:"prompt_cache,omitempty"`     // Optional prompt response cache configuration
	Fallback              *FallbackConfig            `json:"fallback,omitempty"`         // Fallback configuration with enabled toggle
	ProviderConfigs       map[string]*ProviderConfig `json:"provider_configs,omitempty"` // Custom provider configurations by provider name
}
