package model_selection

import (
	"fmt"
	"log"
	"math"
	"os"
	"strings"

	"adaptive-backend/internal/models"

	"github.com/botirk38/semanticcache"
	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
)

const (
	defaultGlobalCacheSize    = 2000
	defaultUserCachePoolSize  = 200
	defaultUserCacheSize      = 150
	defaultSemanticThreshold  = 0.85
	minionComplexityThreshold = 0.2
	minionPromptLength        = 30
)

// ModelSelector returns either a standard‐LLM or minion orchestrator response.
type ModelSelector struct {
	config               *Config
	classifier           *PromptClassifierClient
	globalCache          *semanticcache.SemanticCache[string, models.StandardLLMOrchestratorResponse]
	userCachePool        *lru.Cache[string, *semanticcache.SemanticCache[string, models.StandardLLMOrchestratorResponse]]
	embeddingProvider    semanticcache.EmbeddingProvider
	semanticThreshold    float32
	defaultUserCacheSize int
	costBiasEnabled      bool
	costBiasFactor       float64
}

// NewModelSelector constructs the selector, loading config and caches.
func NewModelSelector() (*ModelSelector, error) {
	cfg, err := GetDefaultConfig()
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	classifier := NewPromptClassifierClient()
	apiKey := os.Getenv("OPENAI_API_KEY")
	embProv, err := semanticcache.NewOpenAIProvider(apiKey, "")
	if err != nil {
		return nil, fmt.Errorf("init embedding provider: %w", err)
	}
	globalCache, err := semanticcache.NewSemanticCache[string, models.StandardLLMOrchestratorResponse](
		defaultGlobalCacheSize, embProv, nil,
	)
	if err != nil {
		return nil, fmt.Errorf("global cache: %w", err)
	}
	userPool, err := lru.New[string, *semanticcache.SemanticCache[string, models.StandardLLMOrchestratorResponse]](
		defaultUserCachePoolSize,
	)
	if err != nil {
		return nil, fmt.Errorf("user cache pool: %w", err)
	}
	return &ModelSelector{
		config:               cfg,
		classifier:           classifier,
		globalCache:          globalCache,
		userCachePool:        userPool,
		embeddingProvider:    embProv,
		semanticThreshold:    defaultSemanticThreshold,
		defaultUserCacheSize: defaultUserCacheSize,
		costBiasEnabled:      true,
		costBiasFactor:       0.15,
	}, nil
}

// SelectModelWithCache returns either a StandardLLMOrchestratorResponse
// or MinionOrchestratorResponse, plus cache tier ("user","global","miss","minion").
func (ms *ModelSelector) SelectModelWithCache(
	req models.ModelSelectionRequest,
	userID, requestID string,
) (models.OrchestratorResponse, string, error) {
	// classify prompt
	cl, err := ms.classifier.Classify(req)
	if err != nil {
		return nil, "error", err
	}
	// branch to minion on low complexity or short prompt
	cx := ms.extractComplexityScore(cl)
	if cx < minionComplexityThreshold || len(req.Prompt) < minionPromptLength {
		tt := ms.validateTaskType(cl.TaskType1)
		resp := models.MinionOrchestratorResponse{
			Protocol:   models.ProtocolMinion,
			MinionData: models.MinionDetails{TaskType: string(tt)},
		}
		log.Printf("[%s] routed to minion: task=%s cx=%.2f len=%d",
			requestID, tt, cx, len(req.Prompt))
		return resp, "minion", nil
	}

	// standard‐LLM path: semantic cache
	key := req.Prompt
	uc := ms.getUserCache(userID)
	if uc != nil {
		if val, found, _ := uc.Lookup(key, ms.semanticThreshold); found {
			log.Printf("[%s] cache hit (user)", requestID)
			return val, "user", nil
		}
	}
	if val, found, _ := ms.globalCache.Lookup(key, ms.semanticThreshold); found {
		log.Printf("[%s] cache hit (global)", requestID)
		if uc != nil {
			_ = uc.Set(key, key, val)
		}
		return val, "global", nil
	}

	// cache miss → full standard selection
	stdResp, err := ms.selectStandard(cl)
	if err != nil {
		return nil, "miss", err
	}
	log.Printf("[%s] cache miss → standard_llm %s", requestID, stdResp.StandardLLMData.Model)

	_ = ms.globalCache.Set(key, key, *stdResp)
	if uc != nil {
		_ = uc.Set(key, key, *stdResp)
	}
	return *stdResp, "miss", nil
}

// selectStandard runs the pure model selection and wraps into StandardLLMOrchestratorResponse.
func (ms *ModelSelector) selectStandard(
	cl models.ClassificationResult,
) (*models.StandardLLMOrchestratorResponse, error) {
	tt := ms.validateTaskType(cl.TaskType1)
	cx := ms.extractComplexityScore(cl)
	mapping, ok := ms.config.GetTaskModelMapping(tt)
	var modelName string
	var threshold float64

	if !ok {
		// default fallback model
		modelName = defaultModelFor(cx)
		threshold = 0.5
	} else {
		adj := ms.adjustComplexityForCostBias(cx, mapping)
		d := ms.selectDifficultyLevel(adj, mapping)
		cfg := difficultyConfig(mapping, d)
		modelName = cfg.Model
		threshold = cfg.ComplexityThreshold
	}

	cap, ok := ms.config.GetModelCapability(modelName)
	if !ok {
		cap = ms.getDefaultCapability(modelName)
	}

	params := ms.generateParameters(tt)
	conf := ms.calculateConfidence(cx, threshold)

	return &models.StandardLLMOrchestratorResponse{
		Protocol:        models.ProtocolStandardLLM,
		StandardLLMData: models.StandardLLMDetails{Provider: string(cap.Provider), Model: modelName},
		Confidence:      conf,
		Parameters:      params,
	}, nil
}

func (ms *ModelSelector) getUserCache(
	userID string,
) *semanticcache.SemanticCache[string, models.StandardLLMOrchestratorResponse] {
	if cache, ok := ms.userCachePool.Get(userID); ok {
		return cache
	}
	newC, err := semanticcache.NewSemanticCache[string, models.StandardLLMOrchestratorResponse](
		ms.defaultUserCacheSize, ms.embeddingProvider, nil,
	)
	if err != nil {
		log.Printf("user cache init failed for %s: %v", userID, err)
		return nil
	}
	ms.userCachePool.Add(userID, newC)
	return newC
}

// --- helpers ---

func (ms *ModelSelector) validateTaskType(types []string) models.TaskType {
	if len(types) == 0 {
		return models.TaskOther
	}
	if models.IsValidTaskType(types[0]) {
		return models.TaskType(types[0])
	}
	return models.TaskOther
}

func (ms *ModelSelector) extractComplexityScore(
	cl models.ClassificationResult,
) float64 {
	if len(cl.PromptComplexityScore) > 0 {
		return cl.PromptComplexityScore[0]
	}
	return 0.5
}

func (ms *ModelSelector) adjustComplexityForCostBias(
	complexity float64,
	m models.TaskModelMapping,
) float64 {
	if !ms.costBiasEnabled {
		return complexity
	}
	eCost := ms.getModelCost(m.Easy.Model)
	hCost := ms.getModelCost(m.Hard.Model)
	spread := hCost - eCost
	if spread <= 0 {
		return complexity
	}
	bias := ms.sigmoid((complexity - 0.5) * ms.costBiasFactor)
	adj := complexity + bias*0.1
	if adj < 0 {
		return 0
	}
	if adj > 1 {
		return 1
	}
	return adj
}

func (ms *ModelSelector) sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (ms *ModelSelector) getModelCost(name string) float64 {
	if cap, ok := ms.config.GetModelCapability(name); ok {
		return cap.CostPer1kTokens
	}
	return 0.01
}

func (ms *ModelSelector) selectDifficultyLevel(
	complexity float64,
	m models.TaskModelMapping,
) models.DifficultyLevel {
	switch {
	case complexity <= m.Easy.ComplexityThreshold:
		return models.DifficultyEasy
	case complexity <= m.Medium.ComplexityThreshold:
		return models.DifficultyMedium
	default:
		return models.DifficultyHard
	}
}

func (ms *ModelSelector) applyProviderConstraint(
	provider string,
) (string, error) {
	var cands []string
	for name, cap := range ms.config.ModelCapabilities {
		if strings.EqualFold(string(cap.Provider), provider) {
			cands = append(cands, name)
		}
	}
	if len(cands) == 0 {
		return "", fmt.Errorf("no models for provider %s", provider)
	}
	return cands[0], nil
}

func (ms *ModelSelector) generateParameters(
	taskType models.TaskType,
) openai.ChatCompletionNewParams {
	tp, ok := ms.config.GetTaskParameters(taskType)
	if !ok {
		tp = ms.getDefaultTaskParameters()
	}
	return openai.ChatCompletionNewParams{
		Temperature:      param.Opt[float64]{Value: tp.Temperature},
		TopP:             param.Opt[float64]{Value: tp.TopP},
		PresencePenalty:  param.Opt[float64]{Value: tp.PresencePenalty},
		FrequencyPenalty: param.Opt[float64]{Value: tp.FrequencyPenalty},
		MaxTokens:        param.Opt[int64]{Value: int64(tp.MaxCompletionTokens)},
		N:                param.Opt[int64]{Value: int64(tp.N)},
	}
}

func (ms *ModelSelector) calculateConfidence(
	complexity, threshold float64,
) float64 {
	d := math.Abs(complexity - threshold)
	conf := 1.0 - d
	switch {
	case conf < 0.1:
		return 0.1
	case conf > 1:
		return 1.0
	default:
		return conf
	}
}

// defaultModelFor picks a simple fallback model by complexity.
func defaultModelFor(cx float64) string {
	switch {
	case cx <= 0.3:
		return "gpt-4.1-nano"
	case cx <= 0.6:
		return "gpt-4.1-mini"
	default:
		return "gpt-4o"
	}
}

// difficultyConfig returns the config for a given difficulty.
func difficultyConfig(
	m models.TaskModelMapping,
	d models.DifficultyLevel,
) models.DifficultyConfig {
	switch d {
	case models.DifficultyEasy:
		return m.Easy
	case models.DifficultyMedium:
		return m.Medium
	default:
		return m.Hard
	}
}

func (ms *ModelSelector) getDefaultTaskParameters() models.TaskParameters {
	return models.TaskParameters{
		Temperature:         0.7,
		TopP:                0.9,
		PresencePenalty:     0.0,
		FrequencyPenalty:    0.0,
		MaxCompletionTokens: 1000,
		N:                   1,
	}
}

// getDefaultCapability returns a placeholder capability.
func (ms *ModelSelector) getDefaultCapability(
	modelName string,
) models.ModelCapability {
	return models.ModelCapability{
		Description:             fmt.Sprintf("Default capability for %s", modelName),
		Provider:                models.ProviderOpenAI,
		CostPer1kTokens:         0.002,
		MaxTokens:               4096,
		SupportsStreaming:       true,
		SupportsFunctionCalling: false,
		SupportsVision:          false,
	}
}
