package models

import (
	"time"

	"google.golang.org/genai"
)

// GeminiGenerateRequest represents a request for Gemini's GenerateContent API
// Uses the actual genai SDK types with our custom extensions
type GeminiGenerateRequest struct {
	// Core Gemini API fields - use SDK types directly
	Model             string                       `json:"model,omitzero"`
	Contents          []*genai.Content             `json:"contents,omitzero"`
	Tools             []*genai.Tool                `json:"tools,omitzero"`
	ToolConfig        *genai.ToolConfig            `json:"tool_config,omitzero"`
	SafetySettings    []*genai.SafetySetting       `json:"safety_settings,omitzero"`
	SystemInstruction *genai.Content               `json:"system_instruction,omitzero"`
	GenerationConfig  *genai.GenerateContentConfig `json:"generation_config,omitzero"`

	// Custom fields for our internal processing
	ModelRouterConfig *ModelRouterConfig         `json:"model_router,omitzero"`
	PromptCache       *CacheConfig               `json:"prompt_cache,omitzero"`
	Fallback          *FallbackConfig            `json:"fallback,omitzero"`
	ProviderConfigs   map[string]*ProviderConfig `json:"provider_configs,omitzero"`
}

// AdaptiveGeminiUsage extends genai.UsageMetadata with cache tier information
type AdaptiveGeminiUsage struct {
	PromptTokenCount        int32  `json:"prompt_token_count"`
	CandidatesTokenCount    int32  `json:"candidates_token_count"`
	TotalTokenCount         int32  `json:"total_token_count"`
	CachedContentTokenCount int32  `json:"cached_content_token_count,omitzero"`
	CacheTier               string `json:"cache_tier,omitzero"`
}

type GeminiGenerateContentResponse struct {
	// Optional. Used to retain the full HTTP response.
	SDKHTTPResponse *genai.HTTPResponse `json:"sdkHttpResponse,omitzero"`
	// Response variations returned by the model.
	Candidates []*genai.Candidate `json:"candidates,omitzero"`
	// Timestamp when the request is made to the server.
	CreateTime time.Time `json:"createTime,omitzero"`
	// Output only. The model version used to generate the response.
	ModelVersion string `json:"modelVersion,omitzero"`
	// Output only. Content filter results for a prompt sent in the request. Note: Sent
	// only in the first stream chunk. Only happens when no candidates were generated due
	// to content violations.
	PromptFeedback *genai.GenerateContentResponsePromptFeedback `json:"promptFeedback,omitzero"`
	// Output only. response_id is used to identify each response. It is the encoding of
	// the event_id.
	ResponseID string `json:"responseId,omitzero"`
	// Usage metadata about the response(s).
	UsageMetadata *genai.GenerateContentResponseUsageMetadata `json:"usageMetadata,omitzero"`

	Provider string `json:"provider,omitzero"`
}
