package completions

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/minions"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"adaptive-backend/internal/services/stream_readers/stream"
	"fmt"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go/shared"
)

// CompletionService handles completion requests with fallback logic.
type CompletionService struct {
	providerSelector *ProviderSelector
}

// NewCompletionService creates a new completion service.
func NewCompletionService() *CompletionService {
	return &CompletionService{
		providerSelector: NewProviderSelector(),
	}
}

// HandleStandardCompletion handles standard protocol completions with fallback.
func (cs *CompletionService) HandleStandardCompletion(
	c *fiber.Ctx,
	req *models.ChatCompletionRequest,
	standardInfo *models.StandardLLMInfo,
	requestID string,
	isStream bool,
) error {
	// Get provider with fallback
	prov, model, err := cs.providerSelector.SelectStandardProvider(c.Context(), standardInfo, requestID)
	if err != nil {
		return fmt.Errorf("standard provider selection failed: %w", err)
	}
	req.Model = shared.ChatModel(model)

	// Try the completion, fallback if it fails
	if err := cs.handleProtocolGeneric(c, prov, req, requestID, isStream, protocolStandard); err != nil {
		fiberlog.Warnf("[%s] Standard completion failed: %v", requestID, err)

		// Completion failed, try alternatives if available
		if len(standardInfo.Alternatives) == 0 {
			return fmt.Errorf("standard completion failed and no alternatives: %w", err)
		}

		fiberlog.Infof("[%s] Trying %d standard alternatives after completion failure", requestID, len(standardInfo.Alternatives))
		fallbackSvc := NewFallbackService()
		result, fallbackErr := fallbackSvc.SelectAlternative(c.Context(), standardInfo.Alternatives, requestID)
		if fallbackErr != nil {
			return fmt.Errorf("all standard providers failed: %w", fallbackErr)
		}

		req.Model = shared.ChatModel(result.ModelName)
		fiberlog.Infof("[%s] Using standard fallback after completion failure: %s (%s)", requestID, result.ProviderName, result.ModelName)

		// Try the fallback provider
		return cs.handleProtocolGeneric(c, result.Provider, req, requestID, isStream, protocolStandard)
	}

	return nil
}

// HandleMinionCompletion handles minion protocol completions with fallback.
func (cs *CompletionService) HandleMinionCompletion(
	c *fiber.Ctx,
	req *models.ChatCompletionRequest,
	minionInfo *models.MinionInfo,
	requestID string,
	isStream bool,
) error {
	// Get provider with fallback
	prov, model, err := cs.providerSelector.SelectMinionProvider(c.Context(), minionInfo, requestID)
	if err != nil {
		return fmt.Errorf("minion provider selection failed: %w", err)
	}
	req.Model = shared.ChatModel(model)

	// Try the completion, fallback if it fails
	if err := cs.handleProtocolGeneric(c, prov, req, requestID, isStream, protocolMinion); err != nil {
		fiberlog.Warnf("[%s] Minion completion failed: %v", requestID, err)

		// Completion failed, try alternatives if available
		if len(minionInfo.Alternatives) == 0 {
			return fmt.Errorf("minion completion failed and no alternatives: %w", err)
		}

		fiberlog.Infof("[%s] Trying %d minion alternatives after completion failure", requestID, len(minionInfo.Alternatives))
		fallbackSvc := NewFallbackService()
		result, fallbackErr := fallbackSvc.SelectAlternative(c.Context(), minionInfo.Alternatives, requestID)
		if fallbackErr != nil {
			return fmt.Errorf("all minion providers failed: %w", fallbackErr)
		}

		req.Model = shared.ChatModel(result.ModelName)
		fiberlog.Infof("[%s] Using minion fallback after completion failure: %s (%s)", requestID, result.ProviderName, result.ModelName)

		// Try the fallback provider
		return cs.handleProtocolGeneric(c, result.Provider, req, requestID, isStream, protocolMinion)
	}

	return nil
}

// HandleMinionsProtocolCompletion handles MinionS protocol with fallback.
func (cs *CompletionService) HandleMinionsProtocolCompletion(
	c *fiber.Ctx,
	req *models.ChatCompletionRequest,
	resp *models.ProtocolResponse,
	requestID string,
	isStream bool,
) error {
	orchestrator := minions.NewMinionsOrchestrationService()

	// Get standard provider with fallback
	remoteProv, standardModel, err := cs.providerSelector.SelectStandardProvider(c.Context(), resp.Standard, requestID)
	if err != nil {
		return fmt.Errorf("standard provider selection failed: %w", err)
	}
	req.Model = shared.ChatModel(standardModel)

	// Get minion provider with fallback
	minionProv, minionModel, err := cs.providerSelector.SelectMinionProvider(c.Context(), resp.Minion, requestID)
	if err != nil {
		return fmt.Errorf("minion provider selection failed: %w", err)
	}

	if isStream {
		fiberlog.Infof("[%s] streaming MinionS response", requestID)
	} else {
		fiberlog.Infof("[%s] generating MinionS completion", requestID)
	}

	return cs.tryMinionSWithFallback(c, orchestrator, remoteProv, minionProv, req, minionModel, resp, requestID, isStream)
}

// handleProtocolGeneric handles both streaming and regular flows for any protocol.
func (cs *CompletionService) handleProtocolGeneric(
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

// tryMinionSWithFallback attempts MinionS orchestration with fallback alternatives.
func (cs *CompletionService) tryMinionSWithFallback(
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
		if altErr := cs.tryMinionSStandardAlternatives(c, orchestrator, minionProv, req, minionModel, resp.Standard.Alternatives, requestID, isStream); altErr == nil {
			return nil
		}
	}

	// Try minion alternatives
	if len(resp.Minion.Alternatives) > 0 {
		if altErr := cs.tryMinionSMinionAlternatives(c, orchestrator, remoteProv, req, resp.Minion.Alternatives, requestID, isStream); altErr == nil {
			return nil
		}
	}

	return fmt.Errorf("MinionS %s failed", map[bool]string{true: "streaming", false: "protocol"}[isStream])
}

// tryMinionSStandardAlternatives tries standard alternatives for MinionS.
func (cs *CompletionService) tryMinionSStandardAlternatives(
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
func (cs *CompletionService) tryMinionSMinionAlternatives(
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
