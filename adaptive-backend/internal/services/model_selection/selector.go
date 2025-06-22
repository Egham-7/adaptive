package model_selection

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/botirk38/semanticcache"
	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"golang.org/x/exp/slices"
)

const (
	defaultGlobalCacheSize    = 2000
	defaultUserCachePoolSize  = 200
	defaultUserCacheSize      = 150
	minionComplexityThreshold = 0.2
	minionPromptLength        = 30
	defaultCostBiasFactor     = 0.15
)

var defaultSemanticThreshold float32 = 0.85

type ModelSelector struct {
	config               *Config
	classifier           *PromptClassifierClient
	globalCache          *semanticcache.SemanticCache[string, models.OrchestratorResponse]
	userCachePool        *lru.Cache[string, *semanticcache.SemanticCache[string, models.OrchestratorResponse]]
	embeddingProvider    semanticcache.EmbeddingProvider
	semanticThreshold    float32
	defaultUserCacheSize int
}

func NewModelSelector(
	semanticThreshold float32,
	userCacheSize int,
) (*ModelSelector, error) {
	if semanticThreshold <= 0 {
		semanticThreshold = defaultSemanticThreshold
	}
	if userCacheSize <= 0 {
		userCacheSize = defaultUserCacheSize
	}
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

	globalCache, err := semanticcache.NewSemanticCache[string, models.OrchestratorResponse](
		defaultGlobalCacheSize, embProv, nil,
	)
	if err != nil {
		return nil, fmt.Errorf("global cache: %w", err)
	}

	userPool, err := lru.New[
		string,
		*semanticcache.SemanticCache[string, models.OrchestratorResponse],
	](defaultUserCachePoolSize)
	if err != nil {
		return nil, fmt.Errorf("user cache pool: %w", err)
	}

	return &ModelSelector{
		config:               cfg,
		classifier:           classifier,
		globalCache:          globalCache,
		userCachePool:        userPool,
		embeddingProvider:    embProv,
		semanticThreshold:    semanticThreshold,
		defaultUserCacheSize: userCacheSize,
	}, nil
}

func (ms *ModelSelector) SelectModelWithCache(
	req models.ModelSelectionRequest,
	userID, requestID string,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
) (*models.OrchestratorResponse, string, error) {
	if req.CostBias <= 0 {
		req.CostBias = defaultCostBiasFactor
	}

	cl, err := ms.classifier.Classify(req)
	if err != nil {
		return nil, "error", err
	}
	cx := ms.extractComplexityScore(cl)

	// Minion path
	if (cx < minionComplexityThreshold || len(req.Prompt) < minionPromptLength) &&
		ms.isProviderHealthy("minion", circuitBreakers) &&
		(len(req.ProviderConstraint) == 0 ||
			slices.Contains(req.ProviderConstraint, "minion")) {

		tt := ms.validateTaskType(cl.TaskType1)
		alts := ms.generateMinionAlternatives(cl, circuitBreakers, req.CostBias)
		alts = ms.filterByProviderConstraint(alts, req.ProviderConstraint)

		resp := &models.OrchestratorResponse{
			Protocol:     models.ProtocolMinion,
			TaskType:     string(tt),
			Parameters:   ms.generateParameters(tt),
			Alternatives: alts,
		}
		log.Printf("[%s] routed to minion: task=%s cx=%.2f len=%d",
			requestID, tt, cx, len(req.Prompt))
		return resp, "minion", nil
	}

	// Semantic cache lookup
	key := req.Prompt
	if uc := ms.getUserCache(userID); uc != nil {
		if val, found, _ := uc.Lookup(key, ms.semanticThreshold); found {
			log.Printf("[%s] cache hit (user)", requestID)
			return &val, "user", nil
		}
	}
	if val, found, _ := ms.globalCache.Lookup(key, ms.semanticThreshold); found {
		log.Printf("[%s] cache hit (global)", requestID)
		if uc := ms.getUserCache(userID); uc != nil {
			_ = uc.Set(key, key, val)
		}
		return &val, "global", nil
	}

	// Cache miss -> standard LLM
	stdResp, err := ms.selectStandard(cl, req.ProviderConstraint, circuitBreakers, req.CostBias)
	if err != nil {
		return nil, "miss", err
	}
	log.Printf("[%s] cache miss -> standard_llm %s", requestID, stdResp.Model)
	_ = ms.globalCache.Set(key, key, *stdResp)
	if uc := ms.getUserCache(userID); uc != nil {
		_ = uc.Set(key, key, *stdResp)
	}
	return stdResp, "miss", nil
}

func (ms *ModelSelector) selectStandard(
	cl models.ClassificationResult,
	providerConstraint []string,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
	costBiasFactor float32,
) (*models.OrchestratorResponse, error) {
	tt := ms.validateTaskType(cl.TaskType1)
	cx := ms.extractComplexityScore(cl)

	mapping, hasMapping := ms.config.GetTaskModelMapping(tt)
	var modelName string
	var threshold float64

	if !hasMapping {
		modelName = defaultModelFor(cx)
		threshold = 0.5
	} else {
		adj := ms.adjustComplexityForCostBias(cx, mapping, costBiasFactor)
		level := ms.selectDifficultyLevel(adj, mapping)
		cfg := difficultyConfig(mapping, level)
		modelName = cfg.Model
		threshold = cfg.ComplexityThreshold
	}

	cap, hasCap := ms.config.GetModelCapability(modelName)
	if !hasCap {
		cap = ms.getDefaultCapability(modelName)
	}
	provider := string(cap.Provider)

	if !ms.isProviderHealthy(provider, circuitBreakers) {
		log.Printf("Primary provider %s unhealthy, selecting alternative", provider)
		if hasMapping {
			for _, cfg := range []models.DifficultyConfig{
				mapping.Easy, mapping.Medium, mapping.Hard,
			} {
				if cfg.Model == modelName {
					continue
				}
				if altCap, ok := ms.config.GetModelCapability(cfg.Model); ok &&
					ms.isProviderHealthy(string(altCap.Provider), circuitBreakers) {

					modelName = cfg.Model
					cap = altCap
					provider = string(cap.Provider)
					threshold = cfg.ComplexityThreshold
					log.Printf("Switched to %s (%s)", modelName, provider)
					break
				}
			}
		}
	}

	params := ms.generateParameters(tt)
	confidence := ms.calculateConfidence(cx, threshold)

	alts := ms.generateStandardLLMAlternatives(cl, modelName, circuitBreakers)
	alts = ms.filterByProviderConstraint(alts, providerConstraint)

	if len(providerConstraint) > 0 &&
		!slices.Contains(providerConstraint, provider) &&
		len(alts) > 0 {

		provider = alts[0].Provider
		modelName = alts[0].Model
		alts = alts[1:]
	} else if len(providerConstraint) > 0 &&
		!slices.Contains(providerConstraint, provider) &&
		len(alts) == 0 {
		return nil, fmt.Errorf("no available providers match the specified constraints")
	}

	return &models.OrchestratorResponse{
		Protocol:     models.ProtocolStandardLLM,
		Provider:     provider,
		Model:        modelName,
		Confidence:   confidence,
		Parameters:   params,
		Alternatives: alts,
	}, nil
}

func (ms *ModelSelector) getUserCache(
	userID string,
) *semanticcache.SemanticCache[string, models.OrchestratorResponse] {
	if cache, ok := ms.userCachePool.Get(userID); ok {
		return cache
	}
	cache, err := semanticcache.NewSemanticCache[string, models.OrchestratorResponse](
		ms.defaultUserCacheSize, ms.embeddingProvider, nil,
	)
	if err != nil {
		log.Printf("user cache init failed for %s: %v", userID, err)
		return nil
	}
	ms.userCachePool.Add(userID, cache)
	return cache
}

// filterByProviderConstraint retains only those alternatives whose Provider
// appears in the non-empty constraints list.
func (ms *ModelSelector) filterByProviderConstraint(
	alts []models.Alternative,
	constraints []string,
) []models.Alternative {
	if len(constraints) == 0 {
		return alts
	}
	var filtered []models.Alternative
	for _, alt := range alts {
		if slices.Contains(constraints, alt.Provider) {
			filtered = append(filtered, alt)
		}
	}
	return filtered
}

func (ms *ModelSelector) validateTaskType(
	types []string,
) models.TaskType {
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
	costBiasFactor float32,
) float64 {
	eCost := ms.getModelCost(m.Easy.Model)
	hCost := ms.getModelCost(m.Hard.Model)
	if spread := hCost - eCost; spread <= 0 {
		return complexity
	}
	bias := ms.sigmoid((complexity-0.5)*float64(costBiasFactor)) * 0.1
	adj := complexity + bias
	switch {
	case adj < 0:
		return 0
	case adj > 1:
		return 1
	default:
		return adj
	}
}

func (ms *ModelSelector) sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
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
	conf := 1 - d
	switch {
	case conf < 0.1:
		return 0.1
	case conf > 1:
		return 1
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

func (ms *ModelSelector) getDefaultTaskParameters() models.TaskParameters {
	return models.TaskParameters{
		Temperature:         0.7,
		TopP:                0.9,
		PresencePenalty:     0,
		FrequencyPenalty:    0,
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

func (ms *ModelSelector) generateStandardLLMAlternatives(
	cl models.ClassificationResult,
	primaryModel string,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
) []models.Alternative {
	tt := ms.validateTaskType(cl.TaskType1)
	mapping, ok := ms.config.GetTaskModelMapping(tt)
	if !ok {
		return ms.filterHealthyAlternatives(
			ms.getDefaultAlternatives(primaryModel),
			circuitBreakers,
		)
	}

	var alts []models.Alternative
	for _, cfg := range []models.DifficultyConfig{mapping.Easy, mapping.Medium, mapping.Hard} {
		if cfg.Model != primaryModel {
			if cap, ok := ms.config.GetModelCapability(cfg.Model); ok {
				alts = append(alts, models.Alternative{
					Provider: string(cap.Provider),
					Model:    cfg.Model,
				})
			}
		}
	}
	alts = ms.filterHealthyAlternatives(alts, circuitBreakers)
	if len(alts) > 3 {
		alts = alts[:3]
	}
	return alts
}

func (ms *ModelSelector) generateMinionAlternatives(
	cl models.ClassificationResult,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
	costBiasFactor float32,
) []models.Alternative {
	cx := ms.extractComplexityScore(cl)
	tt := ms.validateTaskType(cl.TaskType1)

	taskAlts := ms.getAlternativeMinionTasks(tt)
	var alts []models.Alternative
	for _, t := range taskAlts {
		alts = append(alts, models.Alternative{
			Provider: "minion",
			Model:    string(t),
		})
	}

	if mapping, ok := ms.config.GetTaskModelMapping(tt); ok {
		adj := ms.adjustComplexityForCostBias(cx, mapping, costBiasFactor)
		level := ms.selectDifficultyLevel(adj, mapping)
		cfg := difficultyConfig(mapping, level)

		provider := "OpenAI"
		if cap, ok2 := ms.config.GetModelCapability(cfg.Model); ok2 {
			provider = string(cap.Provider)
		}
		alts = append(alts, models.Alternative{
			Provider: provider,
			Model:    cfg.Model,
		})
	} else {
		alts = append(alts, models.Alternative{
			Provider: "OpenAI",
			Model:    defaultModelFor(cx),
		})
	}

	alts = ms.filterHealthyAlternatives(alts, circuitBreakers)
	if len(alts) > 3 {
		alts = alts[:3]
	}
	return alts
}

func (ms *ModelSelector) getDefaultAlternatives(
	exclude string,
) []models.Alternative {
	defaults := []string{"gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"}
	var alts []models.Alternative
	for _, m := range defaults {
		if m != exclude {
			alts = append(alts, models.Alternative{
				Provider: "OpenAI",
				Model:    m,
			})
		}
	}
	return alts
}

func (ms *ModelSelector) isProviderHealthy(
	provider string,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
) bool {
	if circuitBreakers == nil {
		return true
	}
	if cb, ok := circuitBreakers[provider]; ok {
		return cb.GetState() != circuitbreaker.Open
	}
	return true
}

func (ms *ModelSelector) filterHealthyAlternatives(
	alts []models.Alternative,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
) []models.Alternative {
	if circuitBreakers == nil {
		return alts
	}
	var healthy []models.Alternative
	for _, alt := range alts {
		if ms.isProviderHealthy(alt.Provider, circuitBreakers) {
			healthy = append(healthy, alt)
		}
	}
	return healthy
}

func (ms *ModelSelector) getAlternativeMinionTasks(
	primary models.TaskType,
) []models.TaskType {
	switch primary {
	case models.TaskCodeGeneration:
		return []models.TaskType{
			models.TaskTextGeneration, models.TaskRewrite, models.TaskOther,
		}
	case models.TaskOpenQA:
		return []models.TaskType{
			models.TaskClosedQA, models.TaskChatbot, models.TaskOther,
		}
	case models.TaskClosedQA:
		return []models.TaskType{
			models.TaskOpenQA, models.TaskChatbot, models.TaskOther,
		}
	case models.TaskSummarization:
		return []models.TaskType{
			models.TaskExtraction, models.TaskRewrite, models.TaskOther,
		}
	case models.TaskTextGeneration:
		return []models.TaskType{
			models.TaskCodeGeneration, models.TaskBrainstorming, models.TaskOther,
		}
	case models.TaskClassification:
		return []models.TaskType{models.TaskExtraction, models.TaskOther}
	case models.TaskRewrite:
		return []models.TaskType{
			models.TaskTextGeneration, models.TaskSummarization, models.TaskOther,
		}
	case models.TaskBrainstorming:
		return []models.TaskType{
			models.TaskTextGeneration, models.TaskChatbot, models.TaskOther,
		}
	case models.TaskExtraction:
		return []models.TaskType{
			models.TaskClassification, models.TaskSummarization, models.TaskOther,
		}
	case models.TaskChatbot:
		return []models.TaskType{
			models.TaskOpenQA, models.TaskBrainstorming, models.TaskOther,
		}
	default:
		return []models.TaskType{
			models.TaskChatbot, models.TaskTextGeneration, models.TaskOther,
		}
	}
}
