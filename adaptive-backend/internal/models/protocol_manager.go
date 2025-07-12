// Package models defines core types for protocol management and model selection.
package models

import (
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"time"

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

// ModelCapability represents the capabilities of a model.
type ModelCapability struct {
	Description             string       `json:"description"`
	Provider                ProviderType `json:"provider"`
	ModelName               string       `json:"model_name"`
	CostPer1MInputTokens    float64      `json:"cost_per_1m_input_tokens"`
	CostPer1MOutputTokens   float64      `json:"cost_per_1m_output_tokens"`
	MaxContextTokens        int          `json:"max_context_tokens"`
	MaxOutputTokens         *int         `json:"max_output_tokens,omitempty"`
	SupportsFunctionCalling bool         `json:"supports_function_calling"`
	LanguagesSupported      []string     `json:"languages_supported"`
	ModelSizeParams         *string      `json:"model_size_params,omitempty"`
	LatencyTier             *string      `json:"latency_tier,omitempty"`
}

// ModelSelectionRequest represents an incoming selection request.
type ModelSelectionRequest struct {
	Messages           []openai.ChatCompletionMessageParamUnion `json:"messages"`
	UserID             *string                                  `json:"user_id,omitempty"`
	ProviderConstraint []string                                 `json:"provider_constraint,omitempty"`
	CostBias           *float32                                 `json:"cost_bias,omitempty"`
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

// RaceResult represents a parallel provider race outcome.
type RaceResult struct {
	Provider     provider_interfaces.LLMProvider
	ProviderName string
	ModelName    string
	TaskType     string
	Duration     time.Duration
	Error        error
}
