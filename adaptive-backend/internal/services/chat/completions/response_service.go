package completions

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/minions"
	"adaptive-backend/internal/services/providers"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"adaptive-backend/internal/services/stream_readers/stream"
	"fmt"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go/shared"
)

const (
	protocolStandard   = "standard"
	protocolMinion     = "minion"
	protocolMinions    = "minions_protocol"
	errUnknownProtocol = "unknown protocol"
)

// ResponseService handles HTTP responses for all protocols.
type ResponseService struct{}

// NewResponseService creates a new response service.
func NewResponseService() *ResponseService {
	return &ResponseService{}
}

// HandleProtocol routes to the correct response flow based on protocol.
// remoteProv is the standard-LLM provider, minionProv is used for minion or
// MinionS protocols. req is the original ChatCompletionRequest.
func (s *ResponseService) HandleProtocol(
	c *fiber.Ctx,
	protocol models.ProtocolType,
	req *models.ChatCompletionRequest,
	resp *models.ProtocolResponse,
	requestID string,
	isStream bool,
) error {
	switch protocol {
	case models.ProtocolStandardLLM:
		if resp.Standard != nil {
			req.Model = shared.ChatModel(resp.Standard.Model)
			fiberlog.Infof("[%s] Set standard model to: %s", requestID, resp.Standard.Model)
		}
		return s.handleStandard(c, req, resp.Standard, requestID, isStream)

	case models.ProtocolMinion:
		if resp.Minion != nil {
			req.Model = shared.ChatModel(resp.Minion.Model)
			fiberlog.Infof("[%s] Using minion model: %s with provider: %s", requestID, resp.Minion.Model, resp.Minion.Provider)
		}
		return s.handleMinion(c, req, resp.Minion, requestID, isStream)

	case models.ProtocolMinionsProtocol:
		return s.handleMinionsProtocol(c, req, resp, requestID, isStream)

	default:
		return s.HandleError(c, fiber.StatusInternalServerError,
			errUnknownProtocol+" "+string(protocol), requestID)
	}
}

// handleProtocolGeneric handles both streaming and regular flows for any protocol.
func (s *ResponseService) handleProtocolGeneric(
	c *fiber.Ctx,
	prov provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	requestID string,
	isStream bool,
	protocolName string,
) error {
	provider := prov.GetProviderName() // Get provider name once

	if isStream {
		fiberlog.Infof("[%s] streaming %s response", requestID, protocolName)
		s.setStreamHeaders(c) // Set headers early

		streamResp, err := prov.Chat().
			Completions().
			StreamCompletion(c.Context(), req.ToOpenAIParams())
		if err != nil {
			fiberlog.Errorf("[%s] %s stream failed: %v", requestID, protocolName, err)
			return fmt.Errorf("stream completion failed: %w", err)
		}

		return stream.HandleStream(c, streamResp, requestID, string(req.Model), provider)
	}

	fiberlog.Infof("[%s] generating %s completion", requestID, protocolName)
	regResp, err := prov.Chat().
		Completions().
		CreateCompletion(c.Context(), req.ToOpenAIParams())
	if err != nil {
		fiberlog.Errorf("[%s] %s create failed: %v", requestID, protocolName, err)
		return fmt.Errorf("create completion failed: %w", err)
	}

	adaptiveResp := models.ConvertToAdaptive(regResp, provider)
	return c.JSON(adaptiveResp)
}

// handleStandard handles both streaming and regular standard-LLM flows with fallback.
func (s *ResponseService) handleStandard(
	c *fiber.Ctx,
	req *models.ChatCompletionRequest,
	standardInfo *models.StandardLLMInfo,
	requestID string,
	isStream bool,
) error {
	if standardInfo == nil {
		return s.HandleError(c, fiber.StatusInternalServerError,
			"Standard info is nil", requestID)
	}

	// Try primary provider first
	prov, err := providers.NewLLMProvider(standardInfo.Provider)
	if err != nil {
		fiberlog.Warnf("[%s] Primary standard provider %s failed: %v", requestID, standardInfo.Provider, err)

		// Primary failed, try alternatives
		if len(standardInfo.Alternatives) > 0 {
			fiberlog.Infof("[%s] Trying %d standard alternatives", requestID, len(standardInfo.Alternatives))
			fallbackSvc := NewFallbackService()
			result, err := fallbackSvc.SelectAlternative(c.Context(), standardInfo.Alternatives, requestID)
			if err != nil {
				return s.HandleError(c, fiber.StatusInternalServerError,
					fmt.Sprintf("All standard providers failed: %v", err), requestID)
			}
			prov = result.Provider
			req.Model = shared.ChatModel(result.ModelName)
			fiberlog.Infof("[%s] Using standard alternative: %s (%s)", requestID, result.ProviderName, result.ModelName)
		} else {
			return s.HandleError(c, fiber.StatusInternalServerError,
				fmt.Sprintf("Primary standard provider failed and no alternatives: %v", err), requestID)
		}
	} else {
		req.Model = shared.ChatModel(standardInfo.Model)
		fiberlog.Infof("[%s] Using primary standard provider: %s (%s)", requestID, standardInfo.Provider, standardInfo.Model)
	}

	// Try the completion, fallback if it fails
	if err := s.handleProtocolGeneric(c, prov, req, requestID, isStream, protocolStandard); err != nil {
		fiberlog.Warnf("[%s] Standard completion failed: %v", requestID, err)

		// Completion failed, try alternatives if available
		if len(standardInfo.Alternatives) == 0 {
			return s.HandleError(c, fiber.StatusInternalServerError,
				fmt.Sprintf("Standard completion failed and no alternatives: %v", err), requestID)
		}

		fiberlog.Infof("[%s] Trying %d standard alternatives after completion failure", requestID, len(standardInfo.Alternatives))
		fallbackSvc := NewFallbackService()
		result, fallbackErr := fallbackSvc.SelectAlternative(c.Context(), standardInfo.Alternatives, requestID)
		if fallbackErr != nil {
			return s.HandleError(c, fiber.StatusInternalServerError,
				fmt.Sprintf("All standard providers failed: %v", fallbackErr), requestID)
		}

		req.Model = shared.ChatModel(result.ModelName)
		fiberlog.Infof("[%s] Using standard fallback after completion failure: %s (%s)", requestID, result.ProviderName, result.ModelName)

		// Try the fallback provider
		return s.handleProtocolGeneric(c, result.Provider, req, requestID, isStream, protocolStandard)
	}

	return nil
}

// handleMinion handles both streaming and regular minion flows with fallback.
func (s *ResponseService) handleMinion(
	c *fiber.Ctx,
	req *models.ChatCompletionRequest,
	minionInfo *models.MinionInfo,
	requestID string,
	isStream bool,
) error {
	if minionInfo == nil {
		return s.HandleError(c, fiber.StatusInternalServerError,
			"Minion info is nil", requestID)
	}

	// Try primary provider first
	prov, err := providers.NewLLMProvider(minionInfo.Provider)
	if err != nil {
		fiberlog.Warnf("[%s] Primary minion provider %s failed: %v", requestID, minionInfo.Provider, err)

		// Primary failed, try alternatives
		if len(minionInfo.Alternatives) > 0 {
			fiberlog.Infof("[%s] Trying %d minion alternatives", requestID, len(minionInfo.Alternatives))
			fallbackSvc := NewFallbackService()
			result, err := fallbackSvc.SelectAlternative(c.Context(), minionInfo.Alternatives, requestID)
			if err != nil {
				return s.HandleError(c, fiber.StatusInternalServerError,
					fmt.Sprintf("All minion providers failed: %v", err), requestID)
			}
			prov = result.Provider
			req.Model = shared.ChatModel(result.ModelName)
			fiberlog.Infof("[%s] Using minion alternative: %s (%s)", requestID, result.ProviderName, result.ModelName)
		} else {
			return s.HandleError(c, fiber.StatusInternalServerError,
				fmt.Sprintf("Primary minion provider failed and no alternatives: %v", err), requestID)
		}
	} else {
		req.Model = shared.ChatModel(minionInfo.Model)
		fiberlog.Infof("[%s] Using primary minion provider: %s (%s)", requestID, minionInfo.Provider, minionInfo.Model)
	}

	// Try the completion, fallback if it fails
	if err := s.handleProtocolGeneric(c, prov, req, requestID, isStream, protocolMinion); err != nil {
		fiberlog.Warnf("[%s] Minion completion failed: %v", requestID, err)

		// Completion failed, try alternatives if available
		if len(minionInfo.Alternatives) == 0 {
			return s.HandleError(c, fiber.StatusInternalServerError,
				fmt.Sprintf("Minion completion failed and no alternatives: %v", err), requestID)
		}

		fiberlog.Infof("[%s] Trying %d minion alternatives after completion failure", requestID, len(minionInfo.Alternatives))
		fallbackSvc := NewFallbackService()
		result, fallbackErr := fallbackSvc.SelectAlternative(c.Context(), minionInfo.Alternatives, requestID)
		if fallbackErr != nil {
			return s.HandleError(c, fiber.StatusInternalServerError,
				fmt.Sprintf("All minion providers failed: %v", fallbackErr), requestID)
		}

		req.Model = shared.ChatModel(result.ModelName)
		fiberlog.Infof("[%s] Using minion fallback after completion failure: %s (%s)", requestID, result.ProviderName, result.ModelName)

		// Try the fallback provider
		return s.handleProtocolGeneric(c, result.Provider, req, requestID, isStream, protocolMinion)
	}

	return nil
}

// handleMinionsProtocol handles the MinionS protocol, both stream and non-stream.
func (s *ResponseService) handleMinionsProtocol(
	c *fiber.Ctx,
	req *models.ChatCompletionRequest,
	resp *models.ProtocolResponse,
	requestID string,
	isStream bool,
) error {
	orchestrator := minions.NewMinionsOrchestrationService()

	// Both models are required for MinionsProtocol
	if resp.Standard == nil || resp.Minion == nil {
		return s.HandleError(c, fiber.StatusInternalServerError,
			"MinionsProtocol requires both standard and minion models", requestID)
	}

	// Try standard provider first
	remoteProv, err := providers.NewLLMProvider(resp.Standard.Provider)
	if err != nil {
		fiberlog.Warnf("[%s] Primary standard provider %s failed: %v", requestID, resp.Standard.Provider, err)

		// Primary failed, try alternatives
		if len(resp.Standard.Alternatives) > 0 {
			fallbackSvc := NewFallbackService()
			result, err := fallbackSvc.SelectAlternative(c.Context(), resp.Standard.Alternatives, requestID)
			if err != nil {
				return s.HandleError(c, fiber.StatusInternalServerError,
					fmt.Sprintf("All standard providers failed: %v", err), requestID)
			}
			remoteProv = result.Provider
			req.Model = shared.ChatModel(result.ModelName)
			fiberlog.Infof("[%s] Using standard alternative: %s (%s)", requestID, result.ProviderName, result.ModelName)
		} else {
			return s.HandleError(c, fiber.StatusInternalServerError,
				fmt.Sprintf("Primary standard provider failed and no alternatives: %v", err), requestID)
		}
	} else {
		req.Model = shared.ChatModel(resp.Standard.Model)
		fiberlog.Infof("[%s] Using primary standard provider: %s (%s)", requestID, resp.Standard.Provider, resp.Standard.Model)
	}

	// Try minion provider first
	minionProv, err := providers.NewLLMProvider(resp.Minion.Provider)
	minionModel := resp.Minion.Model
	if err != nil {
		fiberlog.Warnf("[%s] Primary minion provider %s failed: %v", requestID, resp.Minion.Provider, err)

		// Primary failed, try alternatives
		if len(resp.Minion.Alternatives) > 0 {
			fallbackSvc := NewFallbackService()
			result, err := fallbackSvc.SelectAlternative(c.Context(), resp.Minion.Alternatives, requestID)
			if err != nil {
				return s.HandleError(c, fiber.StatusInternalServerError,
					fmt.Sprintf("All minion providers failed: %v", err), requestID)
			}
			minionProv = result.Provider
			minionModel = result.ModelName
			fiberlog.Infof("[%s] Using minion alternative: %s (%s)", requestID, result.ProviderName, result.ModelName)
		} else {
			return s.HandleError(c, fiber.StatusInternalServerError,
				fmt.Sprintf("Primary minion provider failed and no alternatives: %v", err), requestID)
		}
	} else {
		fiberlog.Infof("[%s] Using primary minion provider: %s (%s)", requestID, resp.Minion.Provider, resp.Minion.Model)
	}

	if isStream {
		fiberlog.Infof("[%s] streaming MinionS response", requestID)
		s.setStreamHeaders(c)
	} else {
		fiberlog.Infof("[%s] generating MinionS completion", requestID)
	}

	return s.tryMinionSWithFallback(c, orchestrator, remoteProv, minionProv, req, minionModel, resp, requestID, isStream)
}

// HandleError sends a standardized error response.
func (s *ResponseService) HandleError(
	c *fiber.Ctx,
	statusCode int,
	message string,
	requestID string,
) error {
	fiberlog.Errorf("[%s] Error %d: %s", requestID, statusCode, message)
	return c.Status(statusCode).JSON(fiber.Map{
		"error": message,
	})
}

// HandleBadRequest handles 400 errors.
func (s *ResponseService) HandleBadRequest(
	c *fiber.Ctx,
	message, requestID string,
) error {
	return s.HandleError(c, fiber.StatusBadRequest, message, requestID)
}

// HandleInternalError handles 500 errors.
func (s *ResponseService) HandleInternalError(
	c *fiber.Ctx,
	message, requestID string,
) error {
	return s.HandleError(c, fiber.StatusInternalServerError, message, requestID)
}

// tryMinionSWithFallback attempts MinionS orchestration with fallback alternatives.
func (s *ResponseService) tryMinionSWithFallback(
	c *fiber.Ctx,
	orchestrator *minions.MinionsOrchestrationService,
	remoteProv, minionProv provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	minionModel string,
	resp *models.ProtocolResponse,
	requestID string,
	isStream bool,
) error {
	// Try initial orchestration
	if isStream {
		streamResp, err := orchestrator.OrchestrateMinionSStream(
			c.Context(), remoteProv, minionProv, req, minionModel,
		)
		if err == nil {
			provider := minionProv.GetProviderName()
			return stream.HandleStream(c, streamResp, requestID, string(req.Model), provider)
		}
		fiberlog.Warnf("[%s] MinionS stream failed: %v", requestID, err)
	} else {
		result, err := orchestrator.OrchestrateMinionS(
			c.Context(), remoteProv, minionProv, req, minionModel,
		)
		if err == nil {
			return c.JSON(result)
		}
		fiberlog.Warnf("[%s] MinionS create failed: %v", requestID, err)
	}

	// Try standard alternatives first
	if len(resp.Standard.Alternatives) > 0 {
		if altErr := s.tryMinionSStandardAlternatives(c, orchestrator, minionProv, req, minionModel, resp.Standard.Alternatives, requestID, isStream); altErr == nil {
			return nil
		}
	}

	// Try minion alternatives
	if len(resp.Minion.Alternatives) > 0 {
		if altErr := s.tryMinionSMinionAlternatives(c, orchestrator, remoteProv, req, resp.Minion.Alternatives, requestID, isStream); altErr == nil {
			return nil
		}
	}

	return s.HandleError(c, fiber.StatusInternalServerError,
		fmt.Sprintf("MinionS %s failed", map[bool]string{true: "streaming", false: "protocol"}[isStream]), requestID)
}

// tryMinionSStandardAlternatives tries standard alternatives for MinionS.
func (s *ResponseService) tryMinionSStandardAlternatives(
	c *fiber.Ctx,
	orchestrator *minions.MinionsOrchestrationService,
	minionProv provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	minionModel string,
	alternatives []models.Alternative,
	requestID string,
	isStream bool,
) error {
	fiberlog.Infof("[%s] Trying standard alternatives for MinionS restart", requestID)
	fallbackSvc := NewFallbackService()
	standardResult, err := fallbackSvc.SelectAlternative(c.Context(), alternatives, requestID)
	if err != nil {
		return err
	}

	fiberlog.Infof("[%s] Restarting MinionS with standard alternative: %s", requestID, standardResult.ProviderName)

	if isStream {
		streamResp, retryErr := orchestrator.OrchestrateMinionSStream(
			c.Context(), standardResult.Provider, minionProv, req, minionModel,
		)
		if retryErr != nil {
			fiberlog.Warnf("[%s] MinionS retry with standard alternative failed: %v", requestID, retryErr)
			return retryErr
		}
		provider := minionProv.GetProviderName()
		return stream.HandleStream(c, streamResp, requestID, string(req.Model), provider)
	}

	result, retryErr := orchestrator.OrchestrateMinionS(
		c.Context(), standardResult.Provider, minionProv, req, minionModel,
	)
	if retryErr != nil {
		fiberlog.Warnf("[%s] MinionS retry with standard alternative failed: %v", requestID, retryErr)
		return retryErr
	}
	return c.JSON(result)
}

// tryMinionSMinionAlternatives tries minion alternatives for MinionS.
func (s *ResponseService) tryMinionSMinionAlternatives(
	c *fiber.Ctx,
	orchestrator *minions.MinionsOrchestrationService,
	remoteProv provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	alternatives []models.Alternative,
	requestID string,
	isStream bool,
) error {
	fiberlog.Infof("[%s] Trying minion alternatives for MinionS restart", requestID)
	fallbackSvc := NewFallbackService()
	minionResult, err := fallbackSvc.SelectAlternative(c.Context(), alternatives, requestID)
	if err != nil {
		return err
	}

	fiberlog.Infof("[%s] Restarting MinionS with minion alternative: %s", requestID, minionResult.ProviderName)

	if isStream {
		streamResp, retryErr := orchestrator.OrchestrateMinionSStream(
			c.Context(), remoteProv, minionResult.Provider, req, minionResult.ModelName,
		)
		if retryErr != nil {
			fiberlog.Warnf("[%s] MinionS retry with minion alternative failed: %v", requestID, retryErr)
			return retryErr
		}
		provider := minionResult.Provider.GetProviderName()
		return stream.HandleStream(c, streamResp, requestID, string(req.Model), provider)
	}

	result, retryErr := orchestrator.OrchestrateMinionS(
		c.Context(), remoteProv, minionResult.Provider, req, minionResult.ModelName,
	)
	if retryErr != nil {
		fiberlog.Warnf("[%s] MinionS retry with minion alternative failed: %v", requestID, retryErr)
		return retryErr
	}
	return c.JSON(result)
}

// setStreamHeaders sets SSE headers.
func (s *ResponseService) setStreamHeaders(c *fiber.Ctx) {
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Transfer-Encoding", "chunked")
	c.Set("Access-Control-Allow-Origin", "*")
	c.Set("Access-Control-Allow-Headers", "Cache-Control")
}
