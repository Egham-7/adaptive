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
	CacheTokensDetails []*genai.ModalityTokenCount `json:"cacheTokensDetails,omitempty"`
	// Output only. Number of tokens in the cached part in the input (the cached content).
	CachedContentTokenCount int32 `json:"cachedContentTokenCount,omitempty"`
	// Number of tokens in the response(s). This includes all the generated response candidates.
	CandidatesTokenCount int32 `json:"candidatesTokenCount,omitempty"`
	// Output only. List of modalities that were returned in the response.
	CandidatesTokensDetails []*genai.ModalityTokenCount `json:"candidatesTokensDetails,omitempty"`
	// Number of tokens in the prompt. When cached_content is set, this is still the total
	// effective prompt size meaning this includes the number of tokens in the cached content.
	PromptTokenCount int32 `json:"promptTokenCount,omitempty"`
	// Output only. List of modalities that were processed in the request input.
	PromptTokensDetails []*genai.ModalityTokenCount `json:"promptTokensDetails,omitempty"`
	// Output only. Number of tokens present in thoughts output.
	ThoughtsTokenCount int32 `json:"thoughtsTokenCount,omitempty"`
	// Output only. Number of tokens present in tool-use prompt(s).
	ToolUsePromptTokenCount int32 `json:"toolUsePromptTokenCount,omitempty"`
	// Output only. List of modalities that were processed for tool-use request inputs.
	ToolUsePromptTokensDetails []*genai.ModalityTokenCount `json:"toolUsePromptTokensDetails,omitempty"`
	// Total token count for prompt, response candidates, and tool-use prompts (if present).
	TotalTokenCount int32 `json:"totalTokenCount,omitempty"`
	// Output only. Traffic type. This shows whether a request consumes Pay-As-You-Go or
	// Provisioned Throughput quota.
	TrafficType genai.TrafficType `json:"trafficType,omitempty"`

	CacheTier string `json:"cacheTier,omitempty"`
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
	UsageMetadata *AdaptiveGeminiUsage `json:"usageMetadata,omitzero"`

	Provider string `json:"provider,omitzero"`
}
