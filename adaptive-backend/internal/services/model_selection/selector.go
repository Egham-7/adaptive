package model_selection

import (
	"adaptive-backend/internal/models"
	"fmt"
	"log"
	"math"
	"os"
	"slices"
	"strings"

	"github.com/botirk38/semanticcache"
	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
)

const (
	defaultGlobalCacheSize     = 2000
	defaultUserCachePoolSize   = 200
	defaultUserCacheSize       = 150
	defaultSemanticThreshold   = 0.85
	minionComplexityThreshold  = 0.3  // same as Python
	minionPromptTokenThreshold = 4    // tokens, not chars
	largeWindowTokenThreshold  = 2048 // (optional minions_protocol)
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
		costBiasFactor:       0.15, // unused now that we read from cfg
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
	tok := countTokens(req.Prompt)
	if cx < minionComplexityThreshold || tok < minionPromptTokenThreshold {
		tt := ms.validateTaskType(cl.TaskType1)
		resp := models.MinionOrchestratorResponse{
			Protocol:   models.ProtocolMinion,
			MinionData: models.MinionDetails{TaskType: string(tt)},
		}
		log.Printf("[%s] routed to minion: task=%s cx=%.2f tokens=%d",
			requestID, tt, cx, tok)
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
	log.Printf("[%s] cache miss → standard_llm %s",
		requestID, stdResp.StandardLLMData.Model)

	_ = ms.globalCache.Set(key, key, *stdResp)
	if uc != nil {
		_ = uc.Set(key, key, *stdResp)
	}
	return *stdResp, "miss", nil
}

func countTokens(prompt string) int {
	// very simple whitespace tokenizer
	return len(strings.Fields(prompt))
}

// selectStandard runs the pure model selection and wraps into StandardLLMOrchestratorResponse.
func (ms *ModelSelector) selectStandard(
	cl models.ClassificationResult,
) (*models.StandardLLMOrchestratorResponse, error) {
	tt := ms.validateTaskType(cl.TaskType1)
	cx := ms.extractComplexityScore(cl)

	mapping, ok := ms.config.GetTaskModelMapping(tt)
	var (
		modelName string
		threshold float64
	)
	if !ok {
		// default fallback
		modelName = defaultModelFor(cx)
		threshold = 0.5
	} else {
		adj := ms.adjustComplexityForCostBias(cx)
		d := ms.selectDifficultyLevel(adj, mapping)
		cfg := difficultyConfig(mapping, d)
		modelName = cfg.Model
		threshold = cfg.ComplexityThreshold
	}

	cap, ok := ms.config.GetModelCapability(modelName)
	if !ok {
		cap = ms.getDefaultCapability(modelName)
	}

	params := ms.generateParameters(tt, cl)
	conf := ms.calculateConfidence(cx, threshold)
	alts := ms.getAlternatives(mapping, modelName)

	return &models.StandardLLMOrchestratorResponse{
		Protocol:        models.ProtocolStandardLLM,
		StandardLLMData: models.StandardLLMDetails{Provider: string(cap.Provider), Model: modelName},
		Confidence:      conf,
		Parameters:      params,
		Alternatives:    alts,
	}, nil
}

// --- helpers --------------------------------------------------------------

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

func extractOrDefault(xs []float64) float64 {
	if len(xs) > 0 {
		return xs[0]
	}
	return 0.5
}

func (ms *ModelSelector) generateParameters(
	tt models.TaskType,
	cl models.ClassificationResult,
) openai.ChatCompletionNewParams {
	tp, ok := ms.config.GetTaskParameters(tt)
	if !ok {
		tp = ms.getDefaultTaskParameters()
	}
	cs := extractOrDefault(cl.CreativityScope)
	rs := extractOrDefault(cl.Reasoning)
	dk := extractOrDefault(cl.DomainKnowledge)
	ck := extractOrDefault(cl.ContextualKnowledge)

	// simple heuristic tweaks:
	temp := tp.Temperature*(1-rs*0.3) + cs*0.1
	topP := tp.TopP*(1-dk*0.2) + ck*0.05

	return openai.ChatCompletionNewParams{
		Temperature:      param.Opt[float64]{Value: temp},
		TopP:             param.Opt[float64]{Value: topP},
		PresencePenalty:  param.Opt[float64]{Value: tp.PresencePenalty},
		FrequencyPenalty: param.Opt[float64]{Value: tp.FrequencyPenalty},
		MaxTokens:        param.Opt[int64]{Value: int64(tp.MaxCompletionTokens)},
		N:                param.Opt[int64]{Value: int64(tp.N)},
	}
}

func (ms *ModelSelector) adjustComplexityForCostBias(
	complexity float64,
) float64 {
	bias := ms.costBiasFactor
	if math.Abs(bias-0.5) < 1e-2 {
		return complexity
	}
	strength := 2 * (bias - 0.5) // -> [-1,1]
	norm := 3 * strength         // amplify
	adj := complexity + (sigmoid(norm)-0.5)*0.4
	return clamp01(adj)
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

func (ms *ModelSelector) getAlternatives(
	m models.TaskModelMapping,
	selected string,
) []string {
	out := []string{}
	for _, cfg := range []models.DifficultyConfig{m.Easy, m.Medium, m.Hard} {
		if cfg.Model != selected && !slices.Contains(out, cfg.Model) {
			out = append(out, cfg.Model)
		}
	}
	if len(out) > 2 {
		return out[:2]
	}
	return out
}

// getUserCache, getDefaultTaskParameters, getDefaultCapability unchanged.

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
