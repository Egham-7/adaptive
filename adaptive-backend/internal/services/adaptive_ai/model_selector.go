package adaptive_ai

import (
	"adaptive-backend/internal/models"
	"context"
	"log"
	"math"
	"strings"
	"time"

	"github.com/openai/openai-go"
)

// ModelSelectorService provides intelligent model selection based on prompt classification
type ModelSelectorService struct {
	promptClassifier  *PromptClassifierService
	modelCapabilities map[string]models.ModelCapability
	taskModelMappings map[models.TaskType]models.TaskModelMapping
	taskParameters    map[models.TaskType]models.TaskParameters
	costBias          float64
	logger            *log.Logger
}

// Default model capabilities
var defaultModelCapabilities = map[string]models.ModelCapability{
	"gpt-4o": {
		Description:             "Most capable GPT-4 model, great for complex tasks",
		Provider:                models.ProviderOpenAI,
		CostPer1KTokens:         0.03,
		MaxTokens:               128000,
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          true,
	},
	"gpt-4": {
		Description:             "High intelligence model for complex reasoning",
		Provider:                models.ProviderOpenAI,
		CostPer1KTokens:         0.03,
		MaxTokens:               8192,
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          false,
	},
	"gpt-4-turbo": {
		Description:             "Faster GPT-4 with good performance",
		Provider:                models.ProviderOpenAI,
		CostPer1KTokens:         0.01,
		MaxTokens:               128000,
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          true,
	},
	"gpt-3.5-turbo": {
		Description:             "Fast and efficient for simpler tasks",
		Provider:                models.ProviderOpenAI,
		CostPer1KTokens:         0.002,
		MaxTokens:               16385,
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          false,
	},
	"claude-3-5-sonnet-20241022": {
		Description:             "Anthropic's most capable model",
		Provider:                models.ProviderAnthropic,
		CostPer1KTokens:         0.015,
		MaxTokens:               200000,
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          true,
	},
	"claude-3-haiku-20240307": {
		Description:             "Fast and cost-effective Anthropic model",
		Provider:                models.ProviderAnthropic,
		CostPer1KTokens:         0.0008,
		MaxTokens:               200000,
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          false,
	},
	"llama-3.2-3b-preview": {
		Description:             "Groq's fast inference model",
		Provider:                models.ProviderGroq,
		CostPer1KTokens:         0.0006,
		MaxTokens:               8192,
		SupportsStreaming:       true,
		SupportsFunctionCalling: false,
		SupportsVision:          false,
	},
	"deepseek-chat": {
		Description:             "DeepSeek's capable reasoning model",
		Provider:                models.ProviderDeepSeek,
		CostPer1KTokens:         0.0014,
		MaxTokens:               32768,
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          false,
	},
	"gemini-pro": {
		Description:             "Google's multimodal model",
		Provider:                models.ProviderGemini,
		CostPer1KTokens:         0.0015,
		MaxTokens:               32768,
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          true,
	},
}

// Default task model mappings
var defaultTaskModelMappings = map[models.TaskType]models.TaskModelMapping{
	models.TaskOpenQA: {
		Easy:   models.TaskDifficultyConfig{Model: "gpt-3.5-turbo", ComplexityThreshold: 0.3},
		Medium: models.TaskDifficultyConfig{Model: "gpt-4-turbo", ComplexityThreshold: 0.7},
		Hard:   models.TaskDifficultyConfig{Model: "gpt-4o", ComplexityThreshold: 1.0},
	},
	models.TaskClosedQA: {
		Easy:   models.TaskDifficultyConfig{Model: "gpt-3.5-turbo", ComplexityThreshold: 0.2},
		Medium: models.TaskDifficultyConfig{Model: "gpt-4", ComplexityThreshold: 0.6},
		Hard:   models.TaskDifficultyConfig{Model: "gpt-4o", ComplexityThreshold: 1.0},
	},
	models.TaskSummarization: {
		Easy:   models.TaskDifficultyConfig{Model: "gpt-3.5-turbo", ComplexityThreshold: 0.3},
		Medium: models.TaskDifficultyConfig{Model: "claude-3-haiku-20240307", ComplexityThreshold: 0.6},
		Hard:   models.TaskDifficultyConfig{Model: "claude-3-5-sonnet-20241022", ComplexityThreshold: 1.0},
	},
	models.TaskTextGeneration: {
		Easy:   models.TaskDifficultyConfig{Model: "gpt-3.5-turbo", ComplexityThreshold: 0.4},
		Medium: models.TaskDifficultyConfig{Model: "gpt-4-turbo", ComplexityThreshold: 0.7},
		Hard:   models.TaskDifficultyConfig{Model: "gpt-4o", ComplexityThreshold: 1.0},
	},
	models.TaskCodeGeneration: {
		Easy:   models.TaskDifficultyConfig{Model: "gpt-3.5-turbo", ComplexityThreshold: 0.3},
		Medium: models.TaskDifficultyConfig{Model: "gpt-4", ComplexityThreshold: 0.6},
		Hard:   models.TaskDifficultyConfig{Model: "gpt-4o", ComplexityThreshold: 1.0},
	},
	models.TaskChatbot: {
		Easy:   models.TaskDifficultyConfig{Model: "gpt-3.5-turbo", ComplexityThreshold: 0.4},
		Medium: models.TaskDifficultyConfig{Model: "gpt-4-turbo", ComplexityThreshold: 0.7},
		Hard:   models.TaskDifficultyConfig{Model: "claude-3-5-sonnet-20241022", ComplexityThreshold: 1.0},
	},
	models.TaskClassification: {
		Easy:   models.TaskDifficultyConfig{Model: "gpt-3.5-turbo", ComplexityThreshold: 0.2},
		Medium: models.TaskDifficultyConfig{Model: "gpt-4", ComplexityThreshold: 0.5},
		Hard:   models.TaskDifficultyConfig{Model: "gpt-4o", ComplexityThreshold: 1.0},
	},
	models.TaskRewrite: {
		Easy:   models.TaskDifficultyConfig{Model: "gpt-3.5-turbo", ComplexityThreshold: 0.3},
		Medium: models.TaskDifficultyConfig{Model: "gpt-4-turbo", ComplexityThreshold: 0.6},
		Hard:   models.TaskDifficultyConfig{Model: "claude-3-5-sonnet-20241022", ComplexityThreshold: 1.0},
	},
	models.TaskBrainstorming: {
		Easy:   models.TaskDifficultyConfig{Model: "gpt-4-turbo", ComplexityThreshold: 0.5},
		Medium: models.TaskDifficultyConfig{Model: "gpt-4o", ComplexityThreshold: 0.7},
		Hard:   models.TaskDifficultyConfig{Model: "claude-3-5-sonnet-20241022", ComplexityThreshold: 1.0},
	},
	models.TaskExtraction: {
		Easy:   models.TaskDifficultyConfig{Model: "gpt-3.5-turbo", ComplexityThreshold: 0.2},
		Medium: models.TaskDifficultyConfig{Model: "gpt-4", ComplexityThreshold: 0.5},
		Hard:   models.TaskDifficultyConfig{Model: "gpt-4o", ComplexityThreshold: 1.0},
	},
	models.TaskOther: {
		Easy:   models.TaskDifficultyConfig{Model: "gpt-3.5-turbo", ComplexityThreshold: 0.4},
		Medium: models.TaskDifficultyConfig{Model: "gpt-4-turbo", ComplexityThreshold: 0.7},
		Hard:   models.TaskDifficultyConfig{Model: "gpt-4o", ComplexityThreshold: 1.0},
	},
}

// Default task parameters
var defaultTaskParameters = map[models.TaskType]models.TaskParameters{
	models.TaskOpenQA: {
		Temperature:         0.7,
		TopP:                0.9,
		PresencePenalty:     0.0,
		FrequencyPenalty:    0.0,
		MaxCompletionTokens: 1000,
		N:                   1,
	},
	models.TaskClosedQA: {
		Temperature:         0.3,
		TopP:                0.8,
		PresencePenalty:     0.0,
		FrequencyPenalty:    0.1,
		MaxCompletionTokens: 500,
		N:                   1,
	},
	models.TaskSummarization: {
		Temperature:         0.5,
		TopP:                0.9,
		PresencePenalty:     0.0,
		FrequencyPenalty:    0.2,
		MaxCompletionTokens: 800,
		N:                   1,
	},
	models.TaskTextGeneration: {
		Temperature:         0.8,
		TopP:                0.95,
		PresencePenalty:     0.1,
		FrequencyPenalty:    0.1,
		MaxCompletionTokens: 1500,
		N:                   1,
	},
	models.TaskCodeGeneration: {
		Temperature:         0.2,
		TopP:                0.85,
		PresencePenalty:     0.0,
		FrequencyPenalty:    0.0,
		MaxCompletionTokens: 2000,
		N:                   1,
	},
	models.TaskChatbot: {
		Temperature:         0.7,
		TopP:                0.9,
		PresencePenalty:     0.2,
		FrequencyPenalty:    0.2,
		MaxCompletionTokens: 1000,
		N:                   1,
	},
	models.TaskClassification: {
		Temperature:         0.1,
		TopP:                0.8,
		PresencePenalty:     0.0,
		FrequencyPenalty:    0.0,
		MaxCompletionTokens: 200,
		N:                   1,
	},
	models.TaskRewrite: {
		Temperature:         0.6,
		TopP:                0.9,
		PresencePenalty:     0.1,
		FrequencyPenalty:    0.1,
		MaxCompletionTokens: 1200,
		N:                   1,
	},
	models.TaskBrainstorming: {
		Temperature:         0.9,
		TopP:                0.95,
		PresencePenalty:     0.3,
		FrequencyPenalty:    0.0,
		MaxCompletionTokens: 1500,
		N:                   2,
	},
	models.TaskExtraction: {
		Temperature:         0.2,
		TopP:                0.8,
		PresencePenalty:     0.0,
		FrequencyPenalty:    0.0,
		MaxCompletionTokens: 500,
		N:                   1,
	},
	models.TaskOther: {
		Temperature:         0.7,
		TopP:                0.9,
		PresencePenalty:     0.0,
		FrequencyPenalty:    0.0,
		MaxCompletionTokens: 1000,
		N:                   1,
	},
}

// NewModelSelectorService creates a new model selector service
func NewModelSelectorService(promptClassifier *PromptClassifierService, logger *log.Logger) *ModelSelectorService {
	if logger == nil {
		logger = log.Default()
	}

	return &ModelSelectorService{
		promptClassifier:  promptClassifier,
		modelCapabilities: defaultModelCapabilities,
		taskModelMappings: defaultTaskModelMappings,
		taskParameters:    defaultTaskParameters,
		costBias:          0.5, // Neutral by default
		logger:            logger,
	}
}

// SelectModel selects the best model for the given prompt
func (ms *ModelSelectorService) SelectModel(ctx context.Context, req models.SelectModelRequest) (*models.SelectModelResponse, error) {
	startTime := time.Now()

	// Classify the prompt
	classification, err := ms.promptClassifier.ClassifyPrompt(ctx, req.Prompt)
	if err != nil {
		ms.logger.Printf("Failed to classify prompt, using fallback: %v", err)
		return ms.getFallbackResponse(req.Provider), nil
	}

	// Parse task type
	taskType := ms.parseTaskType(classification.TaskType1)

	// Get task mapping
	taskMapping, exists := ms.taskModelMappings[taskType]
	if !exists {
		ms.logger.Printf("No mapping for task type %s, using fallback", taskType)
		return ms.getFallbackResponse(req.Provider), nil
	}

	// Use request cost bias if provided, otherwise use service default
	costBias := ms.costBias
	if req.CostBias > 0 {
		costBias = req.CostBias
	}

	// Select difficulty level based on complexity and cost bias
	difficulty := ms.selectDifficultyLevel(classification.PromptComplexityScore, taskMapping, costBias)

	// Get selected model configuration
	var selectedConfig models.TaskDifficultyConfig
	switch difficulty {
	case models.DifficultyEasy:
		selectedConfig = taskMapping.Easy
	case models.DifficultyMedium:
		selectedConfig = taskMapping.Medium
	case models.DifficultyHard:
		selectedConfig = taskMapping.Hard
	default:
		selectedConfig = taskMapping.Medium
	}

	// Apply provider filter if specified
	finalModel := ms.applyProviderFilter(selectedConfig.Model, req.Provider)

	// Get model capability info
	modelInfo, exists := ms.modelCapabilities[finalModel]
	if !exists {
		ms.logger.Printf("Model %s not found in capabilities, using fallback", finalModel)
		return ms.getFallbackResponse(req.Provider), nil
	}

	// Generate optimized parameters
	parameters := ms.generateParameters(taskType, classification, finalModel)

	ms.logger.Printf("Selected model %s (%s) for %s task (complexity: %.3f, difficulty: %s) in %dms",
		finalModel, modelInfo.Provider, taskType, classification.PromptComplexityScore, difficulty, time.Since(startTime).Milliseconds())

	return &models.SelectModelResponse{
		SelectedModel: finalModel,
		Provider:      string(modelInfo.Provider),
		Parameters:    parameters,
	}, nil
}

// parseTaskType converts string task type to TaskType enum
func (ms *ModelSelectorService) parseTaskType(taskTypeStr string) models.TaskType {
	taskTypeStr = strings.TrimSpace(taskTypeStr)

	switch taskTypeStr {
	case "Open QA":
		return models.TaskOpenQA
	case "Closed QA":
		return models.TaskClosedQA
	case "Summarization":
		return models.TaskSummarization
	case "Text Generation":
		return models.TaskTextGeneration
	case "Code Generation":
		return models.TaskCodeGeneration
	case "Chatbot":
		return models.TaskChatbot
	case "Classification":
		return models.TaskClassification
	case "Rewrite":
		return models.TaskRewrite
	case "Brainstorming":
		return models.TaskBrainstorming
	case "Extraction":
		return models.TaskExtraction
	default:
		return models.TaskOther
	}
}

// selectDifficultyLevel selects difficulty based on complexity score and cost bias
func (ms *ModelSelectorService) selectDifficultyLevel(complexityScore float32, taskMapping models.TaskModelMapping, costBias float64) models.DifficultyLevel {
	// Apply sigmoid-scaled cost bias adjustment
	adjustedScore := ms.applyCostBiasAdjustment(float64(complexityScore), costBias)

	// Select difficulty using thresholds
	if adjustedScore <= taskMapping.Easy.ComplexityThreshold {
		return models.DifficultyEasy
	} else if adjustedScore >= taskMapping.Hard.ComplexityThreshold {
		return models.DifficultyHard
	}
	return models.DifficultyMedium
}

// applyCostBiasAdjustment applies sigmoid-scaled cost bias adjustment
func (ms *ModelSelectorService) applyCostBiasAdjustment(complexityScore float64, costBias float64) float64 {
	// Neutral bias shortcut
	if math.Abs(costBias-0.5) < 0.01 {
		return complexityScore
	}

	// Calculate sigmoid-scaled adjustment
	sigmoid := func(x float64) float64 {
		return 1 / (1 + math.Exp(-x))
	}

	// Convert cost bias to adjustment range
	biasStrength := 2 * (costBias - 0.5)                    // [-1, 1] range
	normalizedStrength := 3 * biasStrength                  // More pronounced curve
	adjustment := (sigmoid(normalizedStrength) - 0.5) * 0.4 // Scale to [-0.2, 0.2]

	return complexityScore + adjustment
}

// applyProviderFilter filters model selection by provider if specified
func (ms *ModelSelectorService) applyProviderFilter(selectedModel, providerFilter string) string {
	if providerFilter == "" {
		return selectedModel
	}

	// Check if selected model matches provider
	if modelInfo, exists := ms.modelCapabilities[selectedModel]; exists {
		if string(modelInfo.Provider) == providerFilter {
			return selectedModel
		}
	}

	// Find alternative model from the same provider
	for modelName, modelInfo := range ms.modelCapabilities {
		if string(modelInfo.Provider) == providerFilter {
			return modelName
		}
	}

	// No model found for provider, return original
	ms.logger.Printf("No model found for provider %s, using original selection %s", providerFilter, selectedModel)
	return selectedModel
}

// generateParameters generates optimized parameters for the task and model
func (ms *ModelSelectorService) generateParameters(taskType models.TaskType, classification *models.PromptClassificationResult, modelName string) openai.ChatCompletionNewParams {
	// Get base parameters for task type
	baseParams, exists := ms.taskParameters[taskType]
	if !exists {
		baseParams = ms.taskParameters[models.TaskOther]
	}

	// Adjust parameters based on classification results
	temperature := ms.adjustTemperature(baseParams.Temperature, classification.CreativityScope)
	topP := ms.adjustTopP(baseParams.TopP, classification.CreativityScope)
	presencePenalty := ms.adjustPresencePenalty(baseParams.PresencePenalty, classification.DomainKnowledge)
	frequencyPenalty := ms.adjustFrequencyPenalty(baseParams.FrequencyPenalty, classification.Reasoning)
	maxTokens := ms.adjustMaxTokens(baseParams.MaxCompletionTokens, classification.ContextualKnowledge)

	return openai.ChatCompletionNewParams{
		Model:            modelName,
		Temperature:      openai.Float(temperature),
		TopP:             openai.Float(topP),
		PresencePenalty:  openai.Float(presencePenalty),
		FrequencyPenalty: openai.Float(frequencyPenalty),
		MaxTokens:        openai.Int(int64(maxTokens)),
		N:                openai.Int(int64(baseParams.N)),
	}
}

// Parameter adjustment functions
func (ms *ModelSelectorService) adjustTemperature(base float64, creativityScope float32) float64 {
	adjustment := (float64(creativityScope) - 0.5) * 0.5
	result := base + adjustment
	return math.Max(0.0, math.Min(2.0, result))
}

func (ms *ModelSelectorService) adjustTopP(base float64, creativityScope float32) float64 {
	adjustment := (float64(creativityScope) - 0.5) * 0.3
	result := base + adjustment
	return math.Max(0.0, math.Min(1.0, result))
}

func (ms *ModelSelectorService) adjustPresencePenalty(base float64, domainKnowledge float32) float64 {
	adjustment := (float64(domainKnowledge) - 0.5) * 0.4
	result := base + adjustment
	return math.Max(-2.0, math.Min(2.0, result))
}

func (ms *ModelSelectorService) adjustFrequencyPenalty(base float64, reasoning float32) float64 {
	adjustment := (float64(reasoning) - 0.5) * 0.4
	result := base + adjustment
	return math.Max(-2.0, math.Min(2.0, result))
}

func (ms *ModelSelectorService) adjustMaxTokens(base int, contextualKnowledge float32) int {
	adjustment := int((float64(contextualKnowledge) - 0.5) * 500)
	result := base + adjustment
	return int(math.Max(50, math.Min(4096, float64(result))))
}

// getFallbackResponse returns a fallback response when classification fails
func (ms *ModelSelectorService) getFallbackResponse(providerFilter string) *models.SelectModelResponse {
	model := "gpt-4o"
	provider := "openai"

	// Apply provider filter for fallback
	if providerFilter != "" {
		for modelName, modelInfo := range ms.modelCapabilities {
			if string(modelInfo.Provider) == providerFilter {
				model = modelName
				provider = providerFilter
				break
			}
		}
	}

	return &models.SelectModelResponse{
		SelectedModel: model,
		Provider:      provider,
		Parameters: openai.ChatCompletionNewParams{
			Model:            model,
			Temperature:      openai.Float(0.7),
			TopP:             openai.Float(0.9),
			PresencePenalty:  openai.Float(0.0),
			FrequencyPenalty: openai.Float(0.0),
			MaxTokens:        openai.Int(1000),
			N:                openai.Int(1),
		},
	}
}

// SetCostBias updates the cost bias setting
func (ms *ModelSelectorService) SetCostBias(costBias float64) {
	ms.costBias = math.Max(0.0, math.Min(1.0, costBias))
	ms.logger.Printf("Updated cost bias to %.2f", ms.costBias)
}

// GetHealth returns service health information
func (ms *ModelSelectorService) GetHealth(ctx context.Context) *models.HealthStatus {
	classifierHealth := ms.promptClassifier.GetHealth(ctx)

	health := &models.HealthStatus{
		Service:          "model_selector",
		Status:           "healthy",
		Timestamp:        time.Now().UTC(),
		ModelCount:       len(ms.modelCapabilities),
		TaskTypes:        len(ms.taskModelMappings),
		CostBias:         ms.costBias,
		ClassifierHealth: classifierHealth,
		InferenceType:    "local_model_selection",
	}

	// Overall health depends on classifier health
	if classifierHealth.Status != "healthy" {
		health.Status = "degraded"
		health.Details = map[string]any{
			"issue": "Prompt classifier is not healthy",
		}
	}

	return health
}
