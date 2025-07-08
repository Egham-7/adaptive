package completions

import (
	"adaptive-backend/internal/models"
	"fmt"

	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go/packages/param"
)

const (
	defaultTemperature      = 0.7
	defaultTopP             = 1.0
	defaultMaxTokens        = 1000
	defaultN                = 1
	defaultPresencePenalty  = 0.0
	defaultFrequencyPenalty = 0.0
)

// ParameterService handles the application and validation of model parameters for chat completions.
type ParameterService struct{}

// NewParameterService creates a new parameter service.
func NewParameterService() *ParameterService {
	return &ParameterService{}
}

func (s *ParameterService) GetParams(resp *models.ProtocolResponse) (*models.OpenAIParameters, error) {
	// Pick parameters from the "standard" branch on MinionsProtocol
	switch resp.Protocol {
	case models.ProtocolStandardLLM:
		return &resp.Standard.Parameters, nil
	case models.ProtocolMinion:
		return &resp.Standard.Parameters, nil
	case models.ProtocolMinionsProtocol:
		return &resp.Standard.Parameters, nil
	default:
		return nil, fmt.Errorf("unsupported protocol: %s", resp.Protocol)

	}
}

// ApplyModelParameters applies OpenAI parameters to a chat completion request.
func (s *ParameterService) ApplyModelParameters(
	req *models.ChatCompletionRequest,
	params *models.OpenAIParameters,
	requestID string,
) error {
	fiberlog.Infof("[%s] Applying model parameters", requestID)

	// Apply parameters with logging and basic validation
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

// GetDefaultParameters returns sensible default parameters for chat completions.
func (s *ParameterService) GetDefaultParameters() models.OpenAIParameters {
	return models.OpenAIParameters{
		Temperature:      param.Opt[float64]{Value: defaultTemperature},
		TopP:             param.Opt[float64]{Value: defaultTopP},
		MaxTokens:        param.Opt[int64]{Value: defaultMaxTokens},
		N:                param.Opt[int64]{Value: defaultN},
		PresencePenalty:  param.Opt[float64]{Value: defaultPresencePenalty},
		FrequencyPenalty: param.Opt[float64]{Value: defaultFrequencyPenalty},
	}
}

// ExtractParametersFromRequest extracts parameters from a chat completion request.
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
