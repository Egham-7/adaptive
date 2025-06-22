package models

import (
	"time"

	"adaptive-backend/internal/services/providers/provider_interfaces"

	"github.com/openai/openai-go"
)

// ProviderType represents supported LLM providers
type ProviderType string

type OpenAIParameters = openai.ChatCompletionNewParams

const (
	ProviderOpenAI    ProviderType = "OpenAI"
	ProviderAnthropic ProviderType = "Anthropic"
	ProviderGoogle    ProviderType = "Google"
	ProviderGroq      ProviderType = "GROQ"
	ProviderDeepseek  ProviderType = "DEEPSEEK"
)

// TaskType represents different types of tasks
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

// DifficultyLevel represents task difficulty levels
type DifficultyLevel string

const (
	DifficultyEasy   DifficultyLevel = "easy"
	DifficultyMedium DifficultyLevel = "medium"
	DifficultyHard   DifficultyLevel = "hard"
)

// ClassificationResult represents the response from the Python prompt classifier
type ClassificationResult struct {
	TaskType1             []string  `json:"task_type_1"`
	CreativityScope       []float64 `json:"creativity_scope"`
	Reasoning             []float64 `json:"reasoning"`
	ContextualKnowledge   []float64 `json:"contextual_knowledge"`
	PromptComplexityScore []float64 `json:"prompt_complexity_score"`
	DomainKnowledge       []float64 `json:"domain_knowledge"`
}

// ModelCapability represents the capabilities and configuration of a model
type ModelCapability struct {
	Description             string       `yaml:"description" json:"description"`
	Provider                ProviderType `yaml:"provider" json:"provider"`
	CostPer1kTokens         float64      `yaml:"cost_per_1k_tokens" json:"cost_per_1k_tokens"`
	MaxTokens               int          `yaml:"max_tokens" json:"max_tokens"`
	SupportsStreaming       bool         `yaml:"supports_streaming" json:"supports_streaming"`
	SupportsFunctionCalling bool         `yaml:"supports_function_calling" json:"supports_function_calling"`
	SupportsVision          bool         `yaml:"supports_vision" json:"supports_vision"`
}

// DifficultyConfig represents configuration for a specific difficulty level
type DifficultyConfig struct {
	Model               string  `yaml:"model" json:"model"`
	ComplexityThreshold float64 `yaml:"complexity_threshold" json:"complexity_threshold"`
}

// TaskModelMapping represents model mappings for different difficulty levels of a task
type TaskModelMapping struct {
	Easy   DifficultyConfig `yaml:"easy" json:"easy"`
	Medium DifficultyConfig `yaml:"medium" json:"medium"`
	Hard   DifficultyConfig `yaml:"hard" json:"hard"`
}

// TaskParameters represents parameters for different task types
type TaskParameters struct {
	Temperature         float64 `yaml:"temperature" json:"temperature"`
	TopP                float64 `yaml:"top_p" json:"top_p"`
	PresencePenalty     float64 `yaml:"presence_penalty" json:"presence_penalty"`
	FrequencyPenalty    float64 `yaml:"frequency_penalty" json:"frequency_penalty"`
	MaxCompletionTokens int     `yaml:"max_completion_tokens" json:"max_completion_tokens"`
	N                   int     `yaml:"n" json:"n"`
}

// ModelSelectionConfig represents the complete configuration for model selection
type ModelSelectionConfig struct {
	ModelCapabilities map[string]ModelCapability    `yaml:"model_capabilities" json:"model_capabilities"`
	TaskModelMappings map[TaskType]TaskModelMapping `yaml:"task_model_mappings" json:"task_model_mappings"`
	TaskParameters    map[TaskType]TaskParameters   `yaml:"task_parameters" json:"task_parameters"`
}

// ModelSelectionRequest represents a request for model selection
type ModelSelectionRequest struct {
	Prompt string `json:"prompt"`
	UserID string `json:"user_id,omitempty"` // For caching purposes
}

// PromptScores represents extracted scores from classification
type PromptScores struct {
	CreativityScope       []float64 `json:"creativity_scope"`
	Reasoning             []float64 `json:"reasoning"`
	ContextualKnowledge   []float64 `json:"contextual_knowledge"`
	PromptComplexityScore []float64 `json:"prompt_complexity_score"`
	DomainKnowledge       []float64 `json:"domain_knowledge"`
}

// Alternative represents an alternative model/provider combination for fallback
type Alternative struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
}

// UnifiedOrchestratorResponse consolidates all orchestrator response types
type OrchestratorResponse struct {
	Protocol     LiteralString    `json:"protocol"`
	Provider     string           `json:"provider,omitempty"`
	Model        string           `json:"model,omitempty"`
	TaskType     string           `json:"task_type,omitempty"`
	Confidence   float64          `json:"confidence,omitempty"`
	Parameters   OpenAIParameters `json:"parameters"`
	Alternatives []Alternative    `json:"alternatives,omitempty"`
}

// LiteralString is a string‐alias to use as a discriminant.
type LiteralString string

const (
	ProtocolStandardLLM     LiteralString = "standard_llm"
	ProtocolMinion          LiteralString = "minion"
	ProtocolMinionsProtocol LiteralString = "minions_protocol"
)

// ValidTaskTypes returns all valid task types
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

// ValidProviders returns all valid provider types
func ValidProviders() []ProviderType {
	return []ProviderType{
		ProviderOpenAI,
		ProviderAnthropic,
		ProviderGoogle,
		ProviderGroq,
		ProviderDeepseek,
	}
}

// IsValidTaskType checks if a task type is valid
func IsValidTaskType(taskType string) bool {
	for _, valid := range ValidTaskTypes() {
		if string(valid) == taskType {
			return true
		}
	}
	return false
}

// IsValidProvider checks if a provider is valid
func IsValidProvider(provider string) bool {
	for _, valid := range ValidProviders() {
		if string(valid) == provider {
			return true
		}
	}
	return false
}

// OrchestratorResult holds the result of orchestration
type OrchestratorResult struct {
	Provider     provider_interfaces.LLMProvider
	ProviderName string // For logging and debugging
	CacheType    string
	ProtocolType string
	ModelName    string
	Parameters   OpenAIParameters
	TaskType     string        // For minions
	Alternatives []Alternative // For failover racing if primary fails
}

// RaceResult represents the result of a parallel request race
type RaceResult struct {
	Provider     provider_interfaces.LLMProvider
	ProviderName string
	ModelName    string
	TaskType     string
	Duration     time.Duration
	Error        error
}
