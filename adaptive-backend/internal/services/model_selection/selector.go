package model_selection

import (
	"adaptive-backend/internal/models"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/botirk38/semanticcache"
	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/pkoukk/tiktoken-go"
)

const (
	defaultGlobalCacheSize     = 2000
	defaultUserCachePoolSize   = 200
	defaultUserCacheSize       = 150
	defaultSemanticThreshold   = 0.85
	minionComplexityThreshold  = 0.3
	minionPromptTokenThreshold = 4
)

type ModelSelector struct {
	config               *Config
	classifier           *PromptClassifierClient
	globalCache          *semanticcache.SemanticCache[string, models.StandardLLMOrchestratorResponse]
	userCachePool        *lru.Cache[string, *semanticcache.SemanticCache[string, models.StandardLLMOrchestratorResponse]]
	embeddingProvider    semanticcache.EmbeddingProvider
	semanticThreshold    float32
	defaultUserCacheSize int
	costBiasFactor       float64
	tokenizer            *tiktoken.Tiktoken
}

func NewModelSelector() (*ModelSelector, error) {
	cfg, err := GetDefaultConfig()
	if err != nil {
		return nil, err
	}
	classifier := NewPromptClassifierClient()
	apiKey := os.Getenv("OPENAI_API_KEY")
	embProv, err := semanticcache.NewOpenAIProvider(apiKey, "")
	if err != nil {
		return nil, err
	}
	globalCache, err := semanticcache.NewSemanticCache[string, models.StandardLLMOrchestratorResponse](
		defaultGlobalCacheSize, embProv, nil,
	)
	if err != nil {
		return nil, err
	}
	userPool, err := lru.New[string, *semanticcache.SemanticCache[string, models.StandardLLMOrchestratorResponse]](
		defaultUserCachePoolSize,
	)
	if err != nil {
		return nil, err
	}
	tk, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		return nil, err
	}
	return &ModelSelector{
		config:               cfg,
		classifier:           classifier,
		globalCache:          globalCache,
		userCachePool:        userPool,
		embeddingProvider:    embProv,
		semanticThreshold:    defaultSemanticThreshold,
		defaultUserCacheSize: defaultUserCacheSize,
		costBiasFactor:       0.15,
		tokenizer:            tk,
	}, nil
}

func (ms *ModelSelector) SelectModelWithCache(
	req models.ModelSelectionRequest,
	userID, requestID string,
) (models.OrchestratorResponse, string, error) {
	cl, err := ms.classifier.Classify(req)
	if err != nil {
		return nil, "error", err
	}
	cx := ms.extractComplexityScore(cl)
	tokCount := len(ms.tokenizer.Encode(req.Prompt, nil, nil))
	if cx < minionComplexityThreshold || tokCount < minionPromptTokenThreshold {
		tt := ms.validateTaskType(cl.TaskType1)
		resp := models.MinionOrchestratorResponse{
			Protocol:   models.ProtocolMinion,
			MinionData: models.MinionDetails{TaskType: string(tt)},
		}
		log.Printf("[%s] minion: task=%s cx=%.2f tokens=%d", requestID, tt, cx, tokCount)
		return resp, "minion", nil
	}
	key := req.Prompt
	if uc := ms.getUserCache(userID); uc != nil {
		if val, found, err := uc.Lookup(key, ms.semanticThreshold); err != nil {
			log.Printf("[%s] user cache lookup error: %v", requestID, err)
		} else if found {
			log.Printf("[%s] cache hit user", requestID)
			return val, "user", nil
		}
	}
	if val, found, _ := ms.globalCache.Lookup(key, ms.semanticThreshold); found {
		log.Printf("[%s] cache hit global", requestID)
		if uc := ms.getUserCache(userID); uc != nil {
			_ = uc.Set(key, key, val)
		}
		return val, "global", nil
	}
	stdResp, err := ms.selectStandard(cl)
	if err != nil {
		return nil, "miss", err
	}
	log.Printf("[%s] cache miss → %s", requestID, stdResp.StandardLLMData.Model)
	_ = ms.globalCache.Set(key, key, *stdResp)
	if uc := ms.getUserCache(userID); uc != nil {
		_ = uc.Set(key, key, *stdResp)
	}
	return *stdResp, "miss", nil
}

func (ms *ModelSelector) selectStandard(
	cl models.ClassificationResult,
) (*models.StandardLLMOrchestratorResponse, error) {
	tt := ms.validateTaskType(cl.TaskType1)
	cx := ms.extractComplexityScore(cl)
	modelName, threshold := ms.chooseModel(tt, cx)
	params := ms.generateParameters(tt, cl, modelName)
	cap, ok := ms.config.GetModelCapability(modelName)
	if !ok {
		cap = ms.getDefaultCapability(modelName)
	}
	conf := ms.calculateConfidence(cx, threshold)
	alts := ms.getAlternatives(tt, modelName)
	return &models.StandardLLMOrchestratorResponse{
		Protocol: models.ProtocolStandardLLM,
		StandardLLMData: models.StandardLLMDetails{
			Provider: string(cap.Provider),
			Model:    modelName,
		},
		Confidence:   conf,
		Parameters:   params,
		Alternatives: alts,
	}, nil
}

func (ms *ModelSelector) chooseModel(tt models.TaskType, cx float64) (string, float64) {
	if mapping, ok := ms.config.GetTaskModelMapping(tt); ok {
		adj := ms.adjustComplexityForCostBias(cx)
		level := ms.selectDifficultyLevel(adj, mapping)
		cfg := difficultyConfig(mapping, level)
		return cfg.Model, cfg.ComplexityThreshold
	}
	return defaultModelFor(cx), 0.5
}

func (ms *ModelSelector) generateParameters(
	tt models.TaskType,
	cl models.ClassificationResult,
	modelName string,
) openai.ChatCompletionNewParams {
	tp, ok := ms.config.GetTaskParameters(tt)
	if !ok {
		tp = ms.getDefaultTaskParameters()
	}
	cs, rs, dk, ck := valOrDefault(cl.CreativityScope),
		valOrDefault(cl.Reasoning),
		valOrDefault(cl.DomainKnowledge),
		valOrDefault(cl.ContextualKnowledge)
	temp := tp.Temperature*(1-rs*0.3) + cs*0.1
	topP := tp.TopP*(1-dk*0.2) + ck*0.05
	return openai.ChatCompletionNewParams{
		Model:            modelName,
		Temperature:      param.Opt[float64]{Value: temp},
		TopP:             param.Opt[float64]{Value: topP},
		PresencePenalty:  param.Opt[float64]{Value: tp.PresencePenalty},
		FrequencyPenalty: param.Opt[float64]{Value: tp.FrequencyPenalty},
		MaxTokens:        param.Opt[int64]{Value: int64(tp.MaxCompletionTokens)},
		N:                param.Opt[int64]{Value: int64(tp.N)},
	}
}

func valOrDefault(xs []float64) float64 {
	if len(xs) > 0 {
		return xs[0]
	}
	return 0.5
}

func (ms *ModelSelector) validateTaskType(types []string) models.TaskType {
	if len(types) > 0 && models.IsValidTaskType(types[0]) {
		return models.TaskType(types[0])
	}
	return models.TaskOther
}

func (ms *ModelSelector) extractComplexityScore(
	cl models.ClassificationResult,
) float64 {
	return valOrDefault(cl.PromptComplexityScore)
}

func (ms *ModelSelector) calculateConfidence(
	complexity, threshold float64,
) float64 {
	d := math.Abs(complexity - threshold)
	conf := 1.0 - d
	switch {
	case conf < 0.1:
		return 0.1
	case conf > 1.0:
		return 1.0
	default:
		return conf
	}
}

func (ms *ModelSelector) adjustComplexityForCostBias(
	complexity float64,
) float64 {
	bias := ms.costBiasFactor
	if math.Abs(bias-0.5) < 1e-2 {
		return complexity
	}
	norm := 3 * 2 * (bias - 0.5)
	return clamp01(complexity + (sigmoid(norm)-0.5)*0.4)
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func clamp01(x float64) float64 {
	switch {
	case x < 0:
		return 0
	case x > 1:
		return 1
	default:
		return x
	}
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

func (ms *ModelSelector) getAlternatives(
	tt models.TaskType,
	selected string,
) []string {
	out := make([]string, 0, 3)
	if mapping, ok := ms.config.GetTaskModelMapping(tt); ok {
		for _, cfg := range []models.DifficultyConfig{mapping.Easy, mapping.Medium, mapping.Hard} {
			if cfg.Model != selected {
				out = append(out, cfg.Model)
			}
		}
	}
	if len(out) > 2 {
		return out[:2]
	}
	return out
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
		log.Printf("cache init failed for %s: %v", userID, err)
		return nil
	}
	ms.userCachePool.Add(userID, newC)
	return newC
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
