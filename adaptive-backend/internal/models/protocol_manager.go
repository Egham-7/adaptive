// Package models defines core types for protocol management and model selection.
package models

import (
	"github.com/openai/openai-go"
)

// ProviderType represents supported LLM providers - aligned with Python ProviderType enum.
type ProviderType string

const (
	ProviderOpenAI      ProviderType = "openai"
	ProviderAnthropic   ProviderType = "anthropic"
	ProviderGoogle      ProviderType = "google"
	ProviderGroq        ProviderType = "groq"
	ProviderDeepseek    ProviderType = "deepseek"
	ProviderMistral     ProviderType = "mistral"
	ProviderGrok        ProviderType = "grok"
	ProviderHuggingFace ProviderType = "huggingface"
)

// TaskType represents different types of tasks - aligned with Python TaskType enum.
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

// ProtocolType represents protocol response types - aligned with Python ProtocolType enum.
type ProtocolType string

const (
	ProtocolStandardLLM     ProtocolType = "standard_llm"
	ProtocolMinion          ProtocolType = "minion"
	ProtocolMinionsProtocol ProtocolType = "minions_protocol"
)

// ModelSelectionRequest represents an incoming selection request that matches the Python service structure.
type ModelSelectionRequest struct {
	// The OpenAI chat completion request
	ChatCompletionRequest openai.ChatCompletionNewParams `json:"chat_completion_request"`

	// Our custom parameters for model selection
	UserID             *string  `json:"user_id,omitempty"`
	ProviderConstraint []string `json:"provider_constraint,omitempty"`
	CostBias           *float32 `json:"cost_bias,omitempty"`
}

// OpenAIParameters aliases the ChatCompletion params type from OpenAI Go SDK.
type OpenAIParameters = openai.ChatCompletionNewParams

// Alternative represents a provider+model fallback candidate.
type Alternative struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
}

// StandardLLMInfo holds the chosen remote model details.
type StandardLLMInfo struct {
	Provider     string           `json:"provider"`
	Model        string           `json:"model"`
	Parameters   OpenAIParameters `json:"parameters"`
	Alternatives []Alternative    `json:"alternatives,omitempty"`
}

// MinionInfo holds the chosen HuggingFace model details.
type MinionInfo struct {
	Provider     string           `json:"provider"`
	Model        string           `json:"model"`
	Parameters   OpenAIParameters `json:"parameters"`
	Alternatives []Alternative    `json:"alternatives,omitempty"`
}

// ProtocolResponse is the union of standard and/or minion info.
type ProtocolResponse struct {
	Protocol ProtocolType     `json:"protocol"`
	Standard *StandardLLMInfo `json:"standard,omitempty"`
	Minion   *MinionInfo      `json:"minion,omitempty"`
}

// CacheConfig holds configuration for the protocol manager cache
type CacheConfig struct {
	Enabled           bool    `json:"enabled"`
	SemanticThreshold float32 `json:"semantic_threshold"`
}
