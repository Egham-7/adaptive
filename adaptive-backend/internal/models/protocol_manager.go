// Package models defines core types for protocol management and model selection.
package models

import (
	"time"

	"adaptive-backend/internal/services/providers/provider_interfaces"

	"github.com/openai/openai-go"
)

// ProviderType represents supported LLM providers - aligned with Python ProviderType enum.
type ProviderType string

const (
	ProviderOpenAI    ProviderType = "openai"
	ProviderAnthropic ProviderType = "anthropic"
	ProviderGoogle    ProviderType = "google"
	ProviderGroq      ProviderType = "groq"
	ProviderDeepseek  ProviderType = "deepseek"
	ProviderMistral   ProviderType = "mistral"
	ProviderGrok      ProviderType = "grok"
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

// DifficultyLevel represents task difficulty levels - aligned with Python DifficultyLevel enum.
type DifficultyLevel string

const (
	DifficultyEasy   DifficultyLevel = "easy"
	DifficultyMedium DifficultyLevel = "medium"
	DifficultyHard   DifficultyLevel = "hard"
)

// ProtocolType represents orchestrator response types - aligned with Python ProtocolType enum.
type ProtocolType string

const (
	ProtocolStandardLLM     ProtocolType = "standard_llm"
	ProtocolMinion          ProtocolType = "minion"
	ProtocolMinionsProtocol ProtocolType = "minions_protocol"
)

// ClassificationResult represents the response from the prompt classifier.
type ClassificationResult struct {
	TaskType1             []string  `json:"task_type_1"`
	TaskType2             []string  `json:"task_type_2"`
	TaskTypeProb          []float64 `json:"task_type_prob"`
	CreativityScope       []float64 `json:"creativity_scope"`
	Reasoning             []float64 `json:"reasoning"`
	ContextualKnowledge   []float64 `json:"contextual_knowledge"`
	PromptComplexityScore []float64 `json:"prompt_complexity_score"`
	DomainKnowledge       []float64 `json:"domain_knowledge"`
	NumberOfFewShots      []float64 `json:"number_of_few_shots"`
	NoLabelReason         []float64 `json:"no_label_reason"`
	ConstraintCt          []float64 `json:"constraint_ct"`
}

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

// TaskModelEntry represents a model entry for a task.
type TaskModelEntry struct {
	Provider  ProviderType `json:"provider"`
	ModelName string       `json:"model_name"`
}

// TaskModelMapping maps a task to model entries.
type TaskModelMapping struct {
	ModelEntries []TaskModelEntry `json:"model_entries"`
}

// ModelSelectionConfig holds configuration for model selection.
type ModelSelectionConfig struct {
	ModelCapabilities map[string]ModelCapability    `json:"model_capabilities"`
	TaskModelMappings map[TaskType]TaskModelMapping `json:"task_model_mappings"`
}

// ModelSelectionRequest represents an incoming selection request.
type ModelSelectionRequest struct {
	Prompt             string   `json:"prompt"`
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

// MinionAlternative represents a minion task alternative.
type MinionAlternative struct {
	TaskType string `json:"task_type"`
}

// StandardLLMInfo holds the chosen remote model details.
type StandardLLMInfo struct {
	Provider     string           `json:"provider"`
	Model        string           `json:"model"`
	Confidence   float64          `json:"confidence"`
	Parameters   OpenAIParameters `json:"parameters"`
	Alternatives []Alternative    `json:"alternatives,omitempty"`
}

// MinionInfo holds the chosen minion task details.
type MinionInfo struct {
	TaskType     string              `json:"task_type"`
	Parameters   OpenAIParameters    `json:"parameters"`
	Alternatives []MinionAlternative `json:"alternatives,omitempty"`
}

// OrchestratorResponse is the union of standard and/or minion info.
type OrchestratorResponse struct {
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

// ValidTaskTypes returns all supported TaskType values.
func ValidTaskTypes() []TaskType {
	return []TaskType{
		TaskOpenQA,
		TaskClosedQA,
		TaskSummarization,
		TaskTextGeneration,
		TaskCodeGeneration,
		TaskChatbot,
		TaskClassification,
		TaskRewrite,
		TaskBrainstorming,
		TaskExtraction,
		TaskOther,
	}
}

// ValidProviders returns all supported ProviderType values.
func ValidProviders() []ProviderType {
	return []ProviderType{
		ProviderOpenAI,
		ProviderAnthropic,
		ProviderGoogle,
		ProviderGroq,
		ProviderDeepseek,
		ProviderMistral,
		ProviderGrok,
	}
}

// IsValidTaskType checks if a string matches a known TaskType.
func IsValidTaskType(taskType string) bool {
	for _, t := range ValidTaskTypes() {
		if string(t) == taskType {
			return true
		}
	}
	return false
}

// IsValidProvider checks if a string matches a known provider.
func IsValidProvider(provider string) bool {
	for _, p := range ValidProviders() {
		if string(p) == provider {
			return true
		}
	}
	return false
}
