package format_adapter

// Package-level singleton adapter instances for efficient reuse
var (
	// OpenAI adapters
	AdaptiveToOpenAI *AdaptiveToOpenAIConverter
	OpenAIToAdaptive *OpenAIToAdaptiveConverter

	// Anthropic adapters
	AdaptiveToAnthropic *AdaptiveToAnthropicConverter
	AnthropicToAdaptive *AnthropicToAdaptiveConverter

	// OpenAI to Anthropic adapters
	OpenAIToAnthropic *OpenAIToAnthropicConverter
	AnthropicToOpenAI *AnthropicToOpenAIConverter
)

func init() {
	// Initialize all adapter singletons
	AdaptiveToOpenAI = &AdaptiveToOpenAIConverter{}
	OpenAIToAdaptive = &OpenAIToAdaptiveConverter{}
	AdaptiveToAnthropic = &AdaptiveToAnthropicConverter{}
	AnthropicToAdaptive = &AnthropicToAdaptiveConverter{}
	OpenAIToAnthropic = &OpenAIToAnthropicConverter{}
	AnthropicToOpenAI = &AnthropicToOpenAIConverter{}
}
