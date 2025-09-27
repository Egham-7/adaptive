// Package models defines core types for model routing and selection.
package models

// TaskType represents different types of tasks.
type TaskType string

const (
	TaskOpenQA         TaskType = "Open QA"
	TaskClosedQA       TaskType = "Closed QA"
	TaskSummarization  TaskType = "Summarization"
	TaskTextGeneration TaskType = "Text Generation"
	TaskCodeGeneration TaskType = "Code Generation"
	TaskChatbot        TaskType = "Chatbot"
	TaskClassification TaskType = "Classification"
	TaskRewrite        TaskType = "Rewrite"
	TaskBrainstorming  TaskType = "Brainstorming"
	TaskExtraction     TaskType = "Extraction"
	TaskOther          TaskType = "Other"
)

// SelectModelRequest represents a provider-agnostic request for model selection
type SelectModelRequest struct {
	// Available models with their capabilities and constraints
	Models []ModelCapability `json:"models"`
	// The prompt text to analyze for optimal model selection
	Prompt string `json:"prompt"`
	// Optional user identifier for tracking and personalization
	User *string `json:"user,omitzero"`
	// Cost bias for model selection (0.0 = cheapest, 1.0 = best performance)
	CostBias *float32 `json:"cost_bias,omitzero"`
	// Model router cache configuration
	ModelRouterCache *CacheConfig `json:"model_router_cache,omitzero"`

	// Tool-related fields for function calling detection
	ToolCall any `json:"tool_call,omitzero"` // Current tool call being made
	Tools    any `json:"tools,omitzero"`     // Available tool definitions
}

// SelectModelResponse represents the response for model selection
type SelectModelResponse struct {
	// Selected provider
	Provider string `json:"provider"`
	// Selected model
	Model string `json:"model"`
	// Alternative provider/model combinations
	Alternatives []Alternative `json:"alternatives,omitzero"`
	CacheTier    string        `json:"cache_tier,omitzero"`
}

// ModelSelectionRequest represents a request for model selection.
// This matches the Python AI service expected format.
type ModelSelectionRequest struct {
	// The user prompt to analyze
	Prompt string `json:"prompt"`

	// Tool-related fields for function calling detection
	ToolCall any `json:"tool_call,omitzero"` // Current tool call being made
	Tools    any `json:"tools,omitzero"`     // Available tool definitions

	// Our custom parameters for model selection (flattened for Python service)
	UserID   string            `json:"user_id,omitzero"`
	Models   []ModelCapability `json:"models,omitzero"`
	CostBias *float32          `json:"cost_bias,omitzero"`
}

// Alternative represents a provider+model fallback candidate.
type Alternative struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
}

// ModelSelectionResponse represents the simplified response from model selection.
type ModelSelectionResponse struct {
	Provider     string        `json:"provider"`
	Model        string        `json:"model"`
	Alternatives []Alternative `json:"alternatives,omitzero"`
}

// IsValid validates that the ModelSelectionResponse has required fields
func (m *ModelSelectionResponse) IsValid() bool {
	return m != nil && m.Provider != "" && m.Model != ""
}

// CacheResult represents the result of a cache lookup operation
type CacheResult struct {
	Response *ModelSelectionResponse `json:"response,omitzero"`
	Source   string                  `json:"source,omitzero"`
	Hit      bool                    `json:"hit"`
}
