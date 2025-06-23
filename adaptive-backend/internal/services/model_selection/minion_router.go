package model_selection

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"math"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"golang.org/x/exp/slices"
)

const (
	minionComplexityThreshold = 0.2
	minionPromptLength        = 30

	complexityThresholdNano = 0.3
	complexityThresholdMini = 0.6

	modelNameNano = "gpt-4.1-nano"
	modelNameMini = "gpt-4.1-mini"
	modelName4o   = "gpt-4o"
)

type MinionRouter struct {
	config *Config
}

func NewMinionRouter(cfg *Config) *MinionRouter {
	return &MinionRouter{config: cfg}
}

// ShouldRouteEarly returns true if the prompt is simple/small enough
// to route directly to a minion.
func (m *MinionRouter) ShouldRouteEarly(
	cx float64,
	prompt string,
	constraint []string,
	cbs map[string]*circuitbreaker.CircuitBreaker,
) bool {
	return (cx < minionComplexityThreshold || len(prompt) < minionPromptLength) &&
		m.isProviderHealthy("minion", cbs) &&
		(len(constraint) == 0 || slices.Contains(constraint, "minion"))
}

// Route builds the OrchestratorResponse for the minion protocol.
func (m *MinionRouter) Route(
	cl models.ClassificationResult,
	costBias float32,
	constraint []string,
	cbs map[string]*circuitbreaker.CircuitBreaker,
) *models.OrchestratorResponse {
	tt := m.validateTaskType(cl.TaskType1)
	alts := m.generateMinionAlternatives(cl, cbs, costBias)
	alts = m.filterByProviderConstraint(alts, constraint)

	return &models.OrchestratorResponse{
		Protocol: models.ProtocolMinion,
		Minion: &models.MinionInfo{
			TaskType:     string(tt),
			Parameters:   m.generateParameters(tt),
			Alternatives: alts,
		},
	}
}

func (m *MinionRouter) generateMinionAlternatives(
	cl models.ClassificationResult,
	cbs map[string]*circuitbreaker.CircuitBreaker,
	costBiasFactor float32,
) []models.Alternative {
	cx := m.extractComplexityScore(cl)
	tt := m.validateTaskType(cl.TaskType1)

	// 1) Task-specific minions
	var alts []models.Alternative
	for _, t := range m.getAlternativeMinionTasks(tt) {
		alts = append(alts, models.Alternative{
			Provider: "minion",
			Model:    string(t),
		})
	}

	// 2) Fallback to a standard LLM for this task
	if mapping, ok := m.config.GetTaskModelMapping(tt); ok {
		adj := m.adjustComplexityForCostBias(cx, mapping, costBiasFactor)
		level := m.selectDifficultyLevel(adj, mapping)
		cfg := m.difficultyConfig(mapping, level)

		provider := string(models.ProviderOpenAI)
		if cap, ok2 := m.config.GetModelCapability(cfg.Model); ok2 {
			provider = string(cap.Provider)
		}
		alts = append(alts, models.Alternative{
			Provider: provider,
			Model:    cfg.Model,
		})
	} else {
		alts = append(alts, models.Alternative{
			Provider: string(models.ProviderOpenAI),
			Model:    m.defaultModelFor(cx),
		})
	}

	// 3) Filter unhealthy providers & limit to 3 alternatives
	alts = m.filterHealthyAlternatives(alts, cbs)
	if len(alts) > 3 {
		alts = alts[:3]
	}
	return alts
}

func (m *MinionRouter) extractComplexityScore(
	cl models.ClassificationResult,
) float64 {
	if len(cl.PromptComplexityScore) > 0 {
		return cl.PromptComplexityScore[0]
	}
	return 0.5
}

func (m *MinionRouter) validateTaskType(types []string) models.TaskType {
	if len(types) == 0 {
		return models.TaskOther
	}
	if models.IsValidTaskType(types[0]) {
		return models.TaskType(types[0])
	}
	return models.TaskOther
}

func (m *MinionRouter) generateParameters(
	tt models.TaskType,
) openai.ChatCompletionNewParams {
	tp, ok := m.config.GetTaskParameters(tt)
	if !ok {
		tp = m.getDefaultTaskParameters()
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

func (m *MinionRouter) getDefaultTaskParameters() models.TaskParameters {
	return models.TaskParameters{
		Temperature:         0.7,
		TopP:                0.9,
		PresencePenalty:     0,
		FrequencyPenalty:    0,
		MaxCompletionTokens: 1000,
		N:                   1,
	}
}

func (m *MinionRouter) adjustComplexityForCostBias(
	complexity float64,
	mapping models.TaskModelMapping,
	costBiasFactor float32,
) float64 {
	eCost := m.getModelCost(mapping.Easy.Model)
	hCost := m.getModelCost(mapping.Hard.Model)
	if spread := hCost - eCost; spread <= 0 {
		return complexity
	}
	bias := m.sigmoid((complexity-0.5)*float64(costBiasFactor)) * 0.1
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

func (m *MinionRouter) sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (m *MinionRouter) getModelCost(name string) float64 {
	if cap, ok := m.config.GetModelCapability(name); ok {
		return cap.CostPer1kTokens
	}
	return 0.01
}

func (m *MinionRouter) selectDifficultyLevel(
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

func (m *MinionRouter) difficultyConfig(
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

func (m *MinionRouter) getAlternativeMinionTasks(
	primary models.TaskType,
) []models.TaskType {
	switch primary {
	case models.TaskCodeGeneration:
		return []models.TaskType{models.TaskTextGeneration, models.TaskRewrite, models.TaskOther}
	case models.TaskOpenQA:
		return []models.TaskType{models.TaskClosedQA, models.TaskChatbot, models.TaskOther}
	case models.TaskClosedQA:
		return []models.TaskType{models.TaskOpenQA, models.TaskChatbot, models.TaskOther}
	case models.TaskSummarization:
		return []models.TaskType{models.TaskExtraction, models.TaskRewrite, models.TaskOther}
	case models.TaskTextGeneration:
		return []models.TaskType{models.TaskCodeGeneration, models.TaskBrainstorming, models.TaskOther}
	case models.TaskClassification:
		return []models.TaskType{models.TaskExtraction, models.TaskOther}
	case models.TaskRewrite:
		return []models.TaskType{models.TaskTextGeneration, models.TaskSummarization, models.TaskOther}
	case models.TaskBrainstorming:
		return []models.TaskType{models.TaskTextGeneration, models.TaskChatbot, models.TaskOther}
	case models.TaskExtraction:
		return []models.TaskType{models.TaskClassification, models.TaskSummarization, models.TaskOther}
	case models.TaskChatbot:
		return []models.TaskType{models.TaskOpenQA, models.TaskBrainstorming, models.TaskOther}
	default:
		return []models.TaskType{models.TaskChatbot, models.TaskTextGeneration, models.TaskOther}
	}
}

func (m *MinionRouter) filterByProviderConstraint(
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

func (m *MinionRouter) isProviderHealthy(
	provider string,
	cbs map[string]*circuitbreaker.CircuitBreaker,
) bool {
	if cbs == nil {
		return true
	}
	if cb, ok := cbs[provider]; ok {
		return cb.GetState() != circuitbreaker.Open
	}
	return true
}

func (m *MinionRouter) filterHealthyAlternatives(
	alts []models.Alternative,
	cbs map[string]*circuitbreaker.CircuitBreaker,
) []models.Alternative {
	if cbs == nil {
		return alts
	}
	var healthy []models.Alternative
	for _, alt := range alts {
		if m.isProviderHealthy(alt.Provider, cbs) {
			healthy = append(healthy, alt)
		}
	}
	return healthy
}

func (m *MinionRouter) defaultModelFor(cx float64) string {
	switch {
	case cx <= complexityThresholdNano:
		return modelNameNano
	case cx <= complexityThresholdMini:
		return modelNameMini
	default:
		return modelName4o
	}
}
