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

	// Gemini adapters
	AdaptiveToGemini *AdaptiveToGeminiConverter
	GeminiToAdaptive *GeminiToAdaptiveConverter
)

func init() {
	// Initialize all adapter singletons
	AdaptiveToOpenAI = &AdaptiveToOpenAIConverter{}
	OpenAIToAdaptive = &OpenAIToAdaptiveConverter{}
	AdaptiveToAnthropic = &AdaptiveToAnthropicConverter{}
	AnthropicToAdaptive = &AnthropicToAdaptiveConverter{}
	AdaptiveToGemini = &AdaptiveToGeminiConverter{}
	GeminiToAdaptive = &GeminiToAdaptiveConverter{}
}
