package models

import (
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"time"

	"github.com/openai/openai-go"
)

// ProviderType represents supported LLM providers
type ProviderType string

// OpenAIParameters aliases the ChatCompletion params type
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

// ClassificationResult represents the response from the prompt classifier
type ClassificationResult struct {
	TaskType1             []string  `json:"task_type_1"`
	CreativityScope       []float64 `json:"creativity_scope"`
	Reasoning             []float64 `json:"reasoning"`
	ContextualKnowledge   []float64 `json:"contextual_knowledge"`
	PromptComplexityScore []float64 `json:"prompt_complexity_score"`
	DomainKnowledge       []float64 `json:"domain_knowledge"`
}

// ModelCapability represents the capabilities of a model
type ModelCapability struct {
	Description             string       `yaml:"description" json:"description"`
	Provider                ProviderType `yaml:"provider" json:"provider"`
	CostPer1kTokens         float64      `yaml:"cost_per_1k_tokens" json:"cost_per_1k_tokens"`
	MaxTokens               int          `yaml:"max_tokens" json:"max_tokens"`
	SupportsStreaming       bool         `yaml:"supports_streaming" json:"supports_streaming"`
	SupportsFunctionCalling bool         `yaml:"supports_function_calling" json:"supports_function_calling"`
	SupportsVision          bool         `yaml:"supports_vision" json:"supports_vision"`
}

// DifficultyConfig represents config for a difficulty level
type DifficultyConfig struct {
	Model               string  `yaml:"model" json:"model"`
	ComplexityThreshold float64 `yaml:"complexity_threshold" json:"complexity_threshold"`
}

// TaskModelMapping maps a task to models per difficulty
type TaskModelMapping struct {
	Easy   DifficultyConfig `yaml:"easy" json:"easy"`
	Medium DifficultyConfig `yaml:"medium" json:"medium"`
	Hard   DifficultyConfig `yaml:"hard" json:"hard"`
}

// TaskParameters represents LLM params for a task
type TaskParameters struct {
	Temperature         float64 `yaml:"temperature" json:"temperature"`
	TopP                float64 `yaml:"top_p" json:"top_p"`
	PresencePenalty     float64 `yaml:"presence_penalty" json:"presence_penalty"`
	FrequencyPenalty    float64 `yaml:"frequency_penalty" json:"frequency_penalty"`
	MaxCompletionTokens int     `yaml:"max_completion_tokens" json:"max_completion_tokens"`
	N                   int     `yaml:"n" json:"n"`
}

// ModelSelectionConfig holds global YAML config
type ModelSelectionConfig struct {
	ModelCapabilities map[string]ModelCapability    `yaml:"model_capabilities" json:"model_capabilities"`
	TaskModelMappings map[TaskType]TaskModelMapping `yaml:"task_model_mappings" json:"task_model_mappings"`
	TaskParameters    map[TaskType]TaskParameters   `yaml:"task_parameters" json:"task_parameters"`
}

// ModelSelectionRequest represents an incoming selection request
type ModelSelectionRequest struct {
	Prompt             string   `json:"prompt"`
	UserID             string   `json:"user_id,omitempty"`
	ProviderConstraint []string `json:"provider_constraint,omitempty"`
	CostBias           float32  `json:"cost_bias,omitempty"`
}

// PromptScores duplicates the scores for narrow use cases
type PromptScores struct {
	CreativityScope       []float64 `json:"creativity_scope"`
	Reasoning             []float64 `json:"reasoning"`
	ContextualKnowledge   []float64 `json:"contextual_knowledge"`
	PromptComplexityScore []float64 `json:"prompt_complexity_score"`
	DomainKnowledge       []float64 `json:"domain_knowledge"`
}

// Alternative is a provider+model fallback candidate
type Alternative struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
}

// ProtocolType discriminates the orchestrator response shape
type ProtocolType string

const (
	ProtocolStandardLLM     ProtocolType = "standard_llm"
	ProtocolMinion          ProtocolType = "minion"
	ProtocolMinionsProtocol ProtocolType = "minions_protocol"
)

// StandardLLMInfo holds the chosen remote model details
type StandardLLMInfo struct {
	Provider     string           `json:"provider"`
	Model        string           `json:"model"`
	Confidence   float64          `json:"confidence"`
	Parameters   OpenAIParameters `json:"parameters"`
	Alternatives []Alternative    `json:"alternatives,omitempty"`
}

// MinionInfo holds the chosen minion task details
type MinionInfo struct {
	TaskType     string           `json:"task_type"`
	Parameters   OpenAIParameters `json:"parameters"`
	Alternatives []Alternative    `json:"alternatives,omitempty"`
}

// OrchestratorResponse is the union of standard and/or minion info
type OrchestratorResponse struct {
	Protocol ProtocolType     `json:"protocol"`
	Standard *StandardLLMInfo `json:"standard,omitempty"`
	Minion   *MinionInfo      `json:"minion,omitempty"`
}

// OrchestratorResult holds internal routing results
type OrchestratorResult struct {
	Provider     provider_interfaces.LLMProvider
	ProviderName string
	CacheType    string
	ProtocolType ProtocolType
	ModelName    string
	Parameters   OpenAIParameters
	TaskType     string
	Alternatives []Alternative
}

// RaceResult represents a parallel provider race outcome
type RaceResult struct {
	Provider     provider_interfaces.LLMProvider
	ProviderName string
	ModelName    string
	TaskType     string
	Duration     time.Duration
	Error        error
}

// ValidTaskTypes returns all supported TaskType values
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

// ValidProviders returns all supported ProviderType values
func ValidProviders() []ProviderType {
	return []ProviderType{
		ProviderOpenAI,
		ProviderAnthropic,
		ProviderGoogle,
		ProviderGroq,
		ProviderDeepseek,
	}
}

// IsValidTaskType checks if a string matches a known TaskType
func IsValidTaskType(taskType string) bool {
	for _, t := range ValidTaskTypes() {
		if string(t) == taskType {
			return true
		}
	}
	return false
}

// IsValidProvider checks if a string matches a known provider
func IsValidProvider(provider string) bool {
	for _, p := range ValidProviders() {
		if string(p) == provider {
			return true
		}
	}
	return false
}
