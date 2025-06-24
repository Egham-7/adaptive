package model_selection

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"fmt"
	"log"
	"math"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"golang.org/x/exp/slices"
)

type StandardRouter struct {
	config          *Config
	classifier      *PromptClassifierClient
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker
	costBiasFactor  float32
}

func NewStandardRouter(
	cfg *Config,
	cls *PromptClassifierClient,
	cbs map[string]*circuitbreaker.CircuitBreaker,
	costBias float32,
) *StandardRouter {
	return &StandardRouter{
		config:          cfg,
		classifier:      cls,
		circuitBreakers: cbs,
		costBiasFactor:  costBias,
	}
}

// Route selects a remote LLM and returns a typed OrchestratorResponse.
func (s *StandardRouter) Route(
	req models.ModelSelectionRequest,
) (*models.OrchestratorResponse, error) {
	cl, err := s.classifier.Classify(req)
	if err != nil {
		return nil, err
	}
	cx := s.extractComplexityScore(cl)
	tt := s.validateTaskType(cl.TaskType1)

	mapping, hasMapping := s.config.GetTaskModelMapping(tt)
	var modelName string
	var threshold float64

	if !hasMapping {
		modelName = s.defaultModelFor(cx)
		threshold = 0.5
	} else {
		adj := s.adjustComplexityForCostBias(cx, mapping, s.costBiasFactor)
		level := s.selectDifficultyLevel(adj, mapping)
		cfg := s.difficultyConfig(mapping, level)
		modelName = cfg.Model
		threshold = cfg.ComplexityThreshold
	}

	cap, hasCap := s.config.GetModelCapability(modelName)
	if !hasCap {
		cap = s.getDefaultCapability(modelName)
	}
	provider := string(cap.Provider)

	if !s.isProviderHealthy(provider) {
		log.Printf("Primary provider %s unhealthy, selecting alternative", provider)
		if hasMapping {
			for _, cfg := range []models.DifficultyConfig{
				mapping.Easy, mapping.Medium, mapping.Hard,
			} {
				if cfg.Model == modelName {
					continue
				}
				if altCap, ok := s.config.GetModelCapability(cfg.Model); ok &&
					s.isProviderHealthy(string(altCap.Provider)) {
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

	params := s.generateParameters(tt)
	confidence := s.calculateConfidence(cx, threshold)

	alts := s.generateStandardLLMAlternatives(cl, modelName)
	alts = s.filterByProviderConstraint(alts, req.ProviderConstraint)

	if len(req.ProviderConstraint) > 0 &&
		!slices.Contains(req.ProviderConstraint, provider) &&
		len(alts) > 0 {
		provider = alts[0].Provider
		modelName = alts[0].Model
		alts = alts[1:]
	} else if len(req.ProviderConstraint) > 0 &&
		!slices.Contains(req.ProviderConstraint, provider) &&
		len(alts) == 0 {
		return nil, fmt.Errorf("no available providers match constraints")
	}

	standardInfo := &models.StandardLLMInfo{
		Provider:     provider,
		Model:        modelName,
		Confidence:   confidence,
		Parameters:   params,
		Alternatives: alts,
	}

	return &models.OrchestratorResponse{
		Protocol: models.ProtocolStandardLLM,
		Standard: standardInfo,
	}, nil
}

func (s *StandardRouter) validateTaskType(types []string) models.TaskType {
	if len(types) == 0 {
		return models.TaskOther
	}
	if models.IsValidTaskType(types[0]) {
		return models.TaskType(types[0])
	}
	return models.TaskOther
}

func (s *StandardRouter) extractComplexityScore(
	cl models.ClassificationResult,
) float64 {
	if len(cl.PromptComplexityScore) > 0 {
		return cl.PromptComplexityScore[0]
	}
	return 0.5
}

func (s *StandardRouter) adjustComplexityForCostBias(
	complexity float64,
	mapping models.TaskModelMapping,
	costBiasFactor float32,
) float64 {
	eCost := s.getModelCost(mapping.Easy.Model)
	hCost := s.getModelCost(mapping.Hard.Model)
	if spread := hCost - eCost; spread <= 0 {
		return complexity
	}
	bias := s.sigmoid((complexity-0.5)*float64(costBiasFactor)) * 0.1
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

func (s *StandardRouter) sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (s *StandardRouter) getModelCost(name string) float64 {
	if cap, ok := s.config.GetModelCapability(name); ok {
		return cap.CostPer1kTokens
	}
	return 0.01
}

func (s *StandardRouter) selectDifficultyLevel(
	complexity float64,
	mapping models.TaskModelMapping,
) models.DifficultyLevel {
	switch {
	case complexity <= mapping.Easy.ComplexityThreshold:
		return models.DifficultyEasy
	case complexity <= mapping.Medium.ComplexityThreshold:
		return models.DifficultyMedium
	default:
		return models.DifficultyHard
	}
}

func (s *StandardRouter) difficultyConfig(
	mapping models.TaskModelMapping,
	level models.DifficultyLevel,
) models.DifficultyConfig {
	switch level {
	case models.DifficultyEasy:
		return mapping.Easy
	case models.DifficultyMedium:
		return mapping.Medium
	default:
		return mapping.Hard
	}
}

func (s *StandardRouter) getDefaultCapability(
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

func (s *StandardRouter) generateParameters(
	tt models.TaskType,
) openai.ChatCompletionNewParams {
	tp, ok := s.config.GetTaskParameters(tt)
	if !ok {
		tp = s.getDefaultTaskParameters()
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

func (s *StandardRouter) getDefaultTaskParameters() models.TaskParameters {
	return models.TaskParameters{
		Temperature:         0.7,
		TopP:                0.9,
		PresencePenalty:     0,
		FrequencyPenalty:    0,
		MaxCompletionTokens: 1000,
		N:                   1,
	}
}

func (s *StandardRouter) calculateConfidence(
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

func (s *StandardRouter) generateStandardLLMAlternatives(
	cl models.ClassificationResult,
	primaryModel string,
) []models.Alternative {
	tt := s.validateTaskType(cl.TaskType1)
	mapping, ok := s.config.GetTaskModelMapping(tt)
	var alts []models.Alternative
	if !ok {
		alts = s.getDefaultAlternatives(primaryModel)
	} else {
		for _, cfg := range []models.DifficultyConfig{
			mapping.Easy, mapping.Medium, mapping.Hard,
		} {
			if cfg.Model == primaryModel {
				continue
			}
			if cap, ok := s.config.GetModelCapability(cfg.Model); ok {
				alts = append(alts, models.Alternative{
					Provider: string(cap.Provider),
					Model:    cfg.Model,
				})
			}
		}
	}
	alts = s.filterHealthyAlternatives(alts)
	if len(alts) > 3 {
		return alts[:3]
	}
	return alts
}

func (s *StandardRouter) getDefaultAlternatives(
	exclude string,
) []models.Alternative {
	defaults := []string{"gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"}
	var alts []models.Alternative
	for _, m := range defaults {
		if m != exclude {
			alts = append(alts, models.Alternative{
				Provider: string(models.ProviderOpenAI),
				Model:    m,
			})
		}
	}
	return alts
}

func (s *StandardRouter) filterByProviderConstraint(
	alts []models.Alternative,
	constraint []string,
) []models.Alternative {
	if len(constraint) == 0 {
		return alts
	}
	var out []models.Alternative
	for _, alt := range alts {
		if slices.Contains(constraint, alt.Provider) {
			out = append(out, alt)
		}
	}
	return out
}

func (s *StandardRouter) isProviderHealthy(provider string) bool {
	if s.circuitBreakers == nil {
		return true
	}
	if cb, ok := s.circuitBreakers[provider]; ok {
		return cb.GetState() != circuitbreaker.Open
	}
	return true
}

func (s *StandardRouter) filterHealthyAlternatives(
	alts []models.Alternative,
) []models.Alternative {
	if s.circuitBreakers == nil {
		return alts
	}
	var healthy []models.Alternative
	for _, alt := range alts {
		if s.isProviderHealthy(alt.Provider) {
			healthy = append(healthy, alt)
		}
	}
	return healthy
}

func (s *StandardRouter) defaultModelFor(cx float64) string {
	switch {
	case cx <= 0.3:
		return "gpt-4.1-nano"
	case cx <= 0.6:
		return "gpt-4.1-mini"
	default:
		return "gpt-4o"
	}
}
