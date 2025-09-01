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

// ModelSelectionRequest represents a request for model selection.
// This matches the Python AI service expected format.
type ModelSelectionRequest struct {
	// The user prompt to analyze
	Prompt string `json:"prompt"`

	// Tool-related fields for function calling detection
	ToolCall any `json:"tool_call,omitzero"` // Current tool call being made
	Tools    any `json:"tools,omitzero"`     // Available tool definitions

	// Our custom parameters for model selection (flattened for Python service)
	UserID              string            `json:"user_id,omitzero"`
	Models              []ModelCapability `json:"models,omitzero"`
	CostBias            *float32          `json:"cost_bias,omitzero"`
	ComplexityThreshold *float32          `json:"complexity_threshold,omitzero"`
	TokenThreshold      *int              `json:"token_threshold,omitzero"`
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
