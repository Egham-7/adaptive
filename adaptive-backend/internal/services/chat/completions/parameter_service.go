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
	requestID string,
) error {
	fiberlog.Infof("[%s] Applying model parameters", requestID)

	// Apply parameters with logging
	req.MaxTokens = params.MaxTokens
	fiberlog.Debugf("[%s] Applied MaxTokens: %v", requestID, params.MaxTokens)

	req.Temperature = params.Temperature
	fiberlog.Debugf("[%s] Applied Temperature: %v", requestID, params.Temperature)

	req.TopP = params.TopP
	fiberlog.Debugf("[%s] Applied TopP: %v", requestID, params.TopP)

	req.PresencePenalty = params.PresencePenalty
	fiberlog.Debugf("[%s] Applied PresencePenalty: %v", requestID, params.PresencePenalty)

	req.FrequencyPenalty = params.FrequencyPenalty
	fiberlog.Debugf("[%s] Applied FrequencyPenalty: %v", requestID, params.FrequencyPenalty)

	req.N = params.N
	fiberlog.Debugf("[%s] Applied N: %v", requestID, params.N)

	fiberlog.Infof("[%s] Parameter application complete", requestID)
	return nil
}

// GetDefaultParameters returns sensible default parameters for chat completions
func (s *ParameterService) GetDefaultParameters() models.OpenAIParameters {
	return models.OpenAIParameters{
		Temperature:      param.Opt[float64]{Value: 0.7},
		TopP:             param.Opt[float64]{Value: 1.0},
		MaxTokens:        param.Opt[int64]{Value: 1000},
		N:                param.Opt[int64]{Value: 1},
		PresencePenalty:  param.Opt[float64]{Value: 0.0},
		FrequencyPenalty: param.Opt[float64]{Value: 0.0},
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
