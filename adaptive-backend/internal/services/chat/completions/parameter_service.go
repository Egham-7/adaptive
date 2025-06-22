package completions

import (
	"adaptive-backend/internal/models"

	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go/packages/param"
)

// ParameterService handles the application and validation of model parameters for chat completions
type ParameterService struct{}

// NewParameterService creates a new parameter service
func NewParameterService() *ParameterService {
	return &ParameterService{}
}

// ApplyModelParameters applies OpenAI parameters to a chat completion request
func (s *ParameterService) ApplyModelParameters(
	req *models.ChatCompletionRequest,
	params models.OpenAIParameters,
	modelName string,
	requestID string,
) error {
	fiberlog.Infof("[%s] Applying model parameters", requestID)

	// Apply parameters with logging
	req.MaxTokens = params.MaxTokens
	fiberlog.Debugf("[%s] Applied MaxTokens: %d", requestID, params.MaxTokens)

	req.Temperature = params.Temperature
	fiberlog.Debugf("[%s] Applied Temperature: %f", requestID, params.Temperature)

	req.TopP = params.TopP
	fiberlog.Debugf("[%s] Applied TopP: %f", requestID, params.TopP)
	req.PresencePenalty = params.PresencePenalty
	fiberlog.Debugf("[%s] Applied PresencePenalty: %f", requestID, params.PresencePenalty)

	req.FrequencyPenalty = params.FrequencyPenalty
	fiberlog.Debugf("[%s] Applied FrequencyPenalty: %f", requestID, params.FrequencyPenalty)

	req.N = params.N
	fiberlog.Debugf("[%s] Applied N: %d", requestID, params.N)

	req.Model = modelName

	fiberlog.Infof("[%s] Parameter application complete", requestID)
	return nil
}

// GetDefaultParameters returns sensible default parameters for chat completions
func (s *ParameterService) GetDefaultParameters() models.OpenAIParameters {
	temperature := 0.7
	topP := 1.0
	maxTokens := 1000
	n := 1
	presencePenalty := 0.0
	frequencyPenalty := 0.0

	return models.OpenAIParameters{
		Temperature:      param.Opt[float64]{Value: temperature},
		TopP:             param.Opt[float64]{Value: topP},
		MaxTokens:        param.Opt[int64]{Value: int64(maxTokens)},
		N:                param.Opt[int64]{Value: int64(n)},
		PresencePenalty:  param.Opt[float64]{Value: presencePenalty},
		FrequencyPenalty: param.Opt[float64]{Value: frequencyPenalty},
	}
}

// ExtractParametersFromRequest extracts parameters from a chat completion request
func (s *ParameterService) ExtractParametersFromRequest(req *models.ChatCompletionRequest) models.OpenAIParameters {
	return models.OpenAIParameters{
		Temperature:      req.Temperature,
		TopP:             req.TopP,
		MaxTokens:        req.MaxTokens,
		N:                req.N,
		PresencePenalty:  req.PresencePenalty,
		FrequencyPenalty: req.FrequencyPenalty,
	}
}
