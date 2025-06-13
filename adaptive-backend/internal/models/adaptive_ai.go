package models

import (
	"time"

	"github.com/openai/openai-go"
)

// TaskType represents the different types of tasks
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

// DifficultyLevel represents task difficulty
type DifficultyLevel string

const (
	DifficultyEasy   DifficultyLevel = "easy"
	DifficultyMedium DifficultyLevel = "medium"
	DifficultyHard   DifficultyLevel = "hard"
)

// ModelProvider represents different AI providers
type ModelProvider string

const (
	ProviderOpenAI    ModelProvider = "openai"
	ProviderAnthropic ModelProvider = "anthropic"
	ProviderGroq      ModelProvider = "groq"
	ProviderDeepSeek  ModelProvider = "deepseek"
	ProviderGemini    ModelProvider = "gemini"
)

// ModelConfig represents the HuggingFace model configuration
type ModelConfig struct {
	TargetSizes map[string]int       `json:"target_sizes"`
	TaskTypeMap map[string]string    `json:"task_type_map"`
	WeightsMap  map[string][]float32 `json:"weights_map"`
	DivisorMap  map[string]float32   `json:"divisor_map"`
}

// PromptClassifierConfig holds configuration for the prompt classifier
type PromptClassifierConfig struct {
	ModelID      string        `json:"model_id"`
	ModelPath    string        `json:"model_path"`
	Timeout      time.Duration `json:"timeout"`
	MaxRetries   int           `json:"max_retries"`
	MaxSeqLength int           `json:"max_seq_length"`
}

// ModelCapability defines model capabilities and costs
type ModelCapability struct {
	Description             string        `json:"description"`
	Provider                ModelProvider `json:"provider"`
	CostPer1KTokens         float64       `json:"cost_per_1k_tokens"`
	MaxTokens               int           `json:"max_tokens"`
	SupportsStreaming       bool          `json:"supports_streaming"`
	SupportsFunctionCalling bool          `json:"supports_function_calling"`
	SupportsVision          bool          `json:"supports_vision"`
}

// TaskDifficultyConfig represents model configuration for a difficulty level
type TaskDifficultyConfig struct {
	Model               string  `json:"model"`
	ComplexityThreshold float64 `json:"complexity_threshold"`
}

// TaskModelMapping maps difficulty levels to model configurations for a task
type TaskModelMapping struct {
	Easy   TaskDifficultyConfig `json:"easy"`
	Medium TaskDifficultyConfig `json:"medium"`
	Hard   TaskDifficultyConfig `json:"hard"`
}

// TaskParameters defines default parameters for different task types
type TaskParameters struct {
	Temperature         float64 `json:"temperature"`
	TopP                float64 `json:"top_p"`
	PresencePenalty     float64 `json:"presence_penalty"`
	FrequencyPenalty    float64 `json:"frequency_penalty"`
	MaxCompletionTokens int     `json:"max_completion_tokens"`
	N                   int     `json:"n"`
}

// AdaptiveAIConfig holds configuration for the adaptive AI service
type AdaptiveAIConfig struct {
	DefaultModel         string        `json:"default_model"`
	FallbackModel        string        `json:"fallback_model"`
	CostBias             float64       `json:"cost_bias"`
	MaxWorkers           int           `json:"max_workers"`
	RequestTimeout       time.Duration `json:"request_timeout"`
	EnableFallback       bool          `json:"enable_fallback"`
	CircuitBreakerConfig struct {
		FailureThreshold int           `json:"failure_threshold"`
		SuccessThreshold int           `json:"success_threshold"`
		Timeout          time.Duration `json:"timeout"`
		ResetAfter       time.Duration `json:"reset_after"`
	} `json:"circuit_breaker_config"`
}

// ModelSelectionRequest represents a request for model selection
type ModelSelectionRequest struct {
	Prompt    string  `json:"prompt"`
	Provider  string  `json:"provider,omitempty"`
	CostBias  float64 `json:"cost_bias,omitempty"`
	Context   string  `json:"context,omitempty"`
	TaskHint  string  `json:"task_hint,omitempty"`
	MaxTokens int     `json:"max_tokens,omitempty"`
	Stream    bool    `json:"stream,omitempty"`
}

// ModelSelectionResponse represents the response from model selection
type ModelSelectionResponse struct {
	SelectedModel    string                         `json:"selected_model"`
	Provider         string                         `json:"provider"`
	Parameters       openai.ChatCompletionNewParams `json:"parameters"`
	Confidence       float64                        `json:"confidence,omitempty"`
	Reasoning        string                         `json:"reasoning,omitempty"`
	Alternatives     []string                       `json:"alternatives,omitempty"`
	TaskType         string                         `json:"task_type,omitempty"`
	Difficulty       string                         `json:"difficulty,omitempty"`
	ComplexityScore  float32                        `json:"complexity_score,omitempty"`
	ProcessingTimeMs int64                          `json:"processing_time_ms,omitempty"`
}

// BatchModelSelectionRequest represents a batch request for model selection
type BatchModelSelectionRequest struct {
	Prompts  []string `json:"prompts"`
	Provider string   `json:"provider,omitempty"`
	CostBias float64  `json:"cost_bias,omitempty"`
}

// BatchModelSelectionResponse represents a batch response from model selection
type BatchModelSelectionResponse struct {
	Results           []*ModelSelectionResponse `json:"results"`
	TotalPrompts      int                       `json:"total_prompts"`
	SuccessfulResults int                       `json:"successful_results"`
	FailedResults     int                       `json:"failed_results"`
	ProcessingTimeMs  int64                     `json:"processing_time_ms"`
}

type PromptClassificationResult struct {
	ModelVersion string `json:"model_version"`
	// The primary predicted task type (e.g., "Open QA").
	TaskType1 string `json:"task_type_1"`
	// The secondary predicted task type, or "NA".
	TaskType2 string `json:"task_type_2"`
	// The probability of the primary task type.
	TaskTypeProbability float32 `json:"task_type_prob"`
	// The final calculated complexity score.
	PromptComplexityScore float32 `json:"prompt_complexity_score"`
	// Score for creativity scope (0-1).
	CreativityScope float32 `json:"creativity_scope"`
	// Score for reasoning (0-1).
	Reasoning float32 `json:"reasoning"`
	// Score for constraint handling (0-1).
	ConstraintHandling float32 `json:"constraint_ct"` // Note the name change to match Python
	// Score for contextual knowledge (0-1).
	ContextualKnowledge float32 `json:"contextual_knowledge"`
	// Score for few-shot learning requirement (0-1).
	FewShotLearning float32 `json:"number_of_few_shots"` // Note the name change
	// Score for domain knowledge (0-1).
	DomainKnowledge float32 `json:"domain_knowledge"`
	// Score for no-label reasoning (0-1).
	LabelReasoning float32 `json:"no_label_reason"` // Note the name change
	// Time in milliseconds for the classification.
	ProcessingTimeMs int64 `json:"processing_time_ms"`
}

// HealthStatus represents the health status of a service
type HealthStatus struct {
	Service          string                 `json:"service"`
	Status           string                 `json:"status"`
	Timestamp        time.Time              `json:"timestamp"`
	Details          map[string]interface{} `json:"details,omitempty"`
	LatencyMs        int64                  `json:"latency_ms,omitempty"`
	InferenceType    string                 `json:"inference_type,omitempty"`
	ModelPath        string                 `json:"model_path,omitempty"`
	ModelID          string                 `json:"model_id,omitempty"`
	Initialized      bool                   `json:"initialized"`
	ModelCount       int                    `json:"model_count,omitempty"`
	TaskTypes        int                    `json:"task_types,omitempty"`
	CostBias         float64                `json:"cost_bias,omitempty"`
	ClassifierHealth *HealthStatus          `json:"classifier_health,omitempty"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error     string    `json:"error"`
	Code      string    `json:"code,omitempty"`
	Timestamp time.Time `json:"timestamp"`
	RequestID string    `json:"request_id,omitempty"`
}
