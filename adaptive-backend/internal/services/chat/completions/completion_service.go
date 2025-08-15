package completions

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/cache"
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
	cfg              *config.Config
	providerSelector *ProviderSelector
	promptCache      *cache.PromptCache
	fallbackService  *FallbackService
}

// NewCompletionService creates a new completion service.
func NewCompletionService(cfg *config.Config, fallbackService *FallbackService) *CompletionService {
	// Initialize prompt cache, but don't fail if it can't be created
	promptCache, err := cache.NewPromptCache(cfg)
	if err != nil {
		fiberlog.Warnf("Failed to initialize prompt cache: %v", err)
		promptCache = nil
	}

	return &CompletionService{
		cfg:              cfg,
		providerSelector: NewProviderSelector(cfg, fallbackService),
		promptCache:      promptCache,
		fallbackService:  fallbackService,
	}
}

// HandleStandardCompletion handles standard protocol completions with fallback.
func (cs *CompletionService) HandleStandardCompletion(
	c *fiber.Ctx,
	req *models.ChatCompletionRequest,
	standardInfo *models.StandardLLMInfo,
	requestID string,
	isStream bool,
	cacheSource string,
) error {
	// Get provider with fallback
	prov, model, err := cs.providerSelector.SelectStandardProvider(c.Context(), standardInfo, req.ProviderConfigs, requestID)
	if err != nil {
		return fmt.Errorf("standard provider selection failed: %w", err)
	}

	return cs.handleCompletionWithFallback(c, req, prov, model, standardInfo.Alternatives, protocolStandard, requestID, isStream, cacheSource)
}

// HandleMinionCompletion handles minion protocol completions with fallback.
func (cs *CompletionService) HandleMinionCompletion(
	c *fiber.Ctx,
	req *models.ChatCompletionRequest,
	minionInfo *models.MinionInfo,
	requestID string,
	isStream bool,
	cacheSource string,
) error {
	// Get provider with fallback
	prov, model, err := cs.providerSelector.SelectMinionProvider(c.Context(), minionInfo, req.ProviderConfigs, requestID)
	if err != nil {
		return fmt.Errorf("minion provider selection failed: %w", err)
	}

	return cs.handleCompletionWithFallback(c, req, prov, model, minionInfo.Alternatives, protocolMinion, requestID, isStream, cacheSource)
}

// HandleMinionsProtocolCompletion handles MinionS protocol with fallback.
func (cs *CompletionService) HandleMinionsProtocolCompletion(
	c *fiber.Ctx,
	req *models.ChatCompletionRequest,
	resp *models.ProtocolResponse,
	requestID string,
	isStream bool,
	cacheSource string,
) error {
	orchestrator := minions.NewMinionsOrchestrationService()

	// Get standard provider with fallback
	remoteProv, standardModel, err := cs.providerSelector.SelectStandardProvider(c.Context(), resp.Standard, req.ProviderConfigs, requestID)
	if err != nil {
		return fmt.Errorf("standard provider selection failed: %w", err)
	}
	req.Model = shared.ChatModel(standardModel)

	// Get minion provider with fallback
	minionProv, minionModel, err := cs.providerSelector.SelectMinionProvider(c.Context(), resp.Minion, req.ProviderConfigs, requestID)
	if err != nil {
		return fmt.Errorf("minion provider selection failed: %w", err)
	}

	if isStream {
		fiberlog.Infof("[%s] streaming MinionS response", requestID)
	} else {
		fiberlog.Infof("[%s] generating MinionS completion", requestID)
	}

	return cs.tryMinionSWithFallback(c, orchestrator, remoteProv, minionProv, req, minionModel, resp, requestID, isStream, cacheSource)
}

// handleCompletionWithFallback handles completion requests with fallback logic (DRY)
func (cs *CompletionService) handleCompletionWithFallback(
	c *fiber.Ctx,
	req *models.ChatCompletionRequest,
	initialProvider provider_interfaces.LLMProvider,
	initialModel string,
	alternatives []models.Alternative,
	protocolName string,
	requestID string,
	isStream bool,
	cacheSource string,
) error {
	// Create mutable copy of alternatives for tracking attempts
	alternativesCopy := make([]models.Alternative, len(alternatives))
	copy(alternativesCopy, alternatives)

	// Set initial model
	req.Model = shared.ChatModel(initialModel)

	// Try the completion with initial provider, fallback if it fails
	if err := cs.handleProtocolGeneric(c, initialProvider, req, requestID, isStream, protocolName, cacheSource); err != nil {
		fiberlog.Warnf("[%s] %s completion failed: %v", requestID, protocolName, err)

		// Completion failed, try remaining alternatives
		if len(alternativesCopy) == 0 {
			return fmt.Errorf("%s completion failed and no alternatives remaining: %w", protocolName, err)
		}

		fiberlog.Infof("[%s] Trying %d remaining %s alternatives after completion failure", requestID, len(alternativesCopy), protocolName)
		result, fallbackErr := cs.fallbackService.SelectAlternative(c.Context(), &alternativesCopy, req.ProviderConfigs, requestID)
		if fallbackErr != nil {
			return fmt.Errorf("all %s providers failed: %w", protocolName, fallbackErr)
		}

		req.Model = shared.ChatModel(result.ModelName)
		fiberlog.Infof("[%s] Using %s fallback after completion failure: %s (%s)", requestID, protocolName, result.ProviderName, result.ModelName)

		// Try the fallback provider
		return cs.handleProtocolGeneric(c, result.Provider, req, requestID, isStream, protocolName, cacheSource)
	}

	return nil
}

// handleProtocolGeneric handles both streaming and regular flows for any protocol.
func (cs *CompletionService) handleProtocolGeneric(
	c *fiber.Ctx,
	prov provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	requestID string,
	isStream bool,
	protocolName string,
	cacheSource string,
) error {
	provider := prov.GetProviderName() // Get provider name once

	// Check prompt cache first for both streaming and non-streaming requests
	if cs.promptCache != nil {
		if cachedResp, found := cs.promptCache.Get(req, requestID); found {
			fiberlog.Infof("[%s] Prompt cache hit for %s", requestID, protocolName)
			// Set cache tier for cached response
			models.SetCacheTier(&cachedResp.Usage, "prompt_response")
			if isStream {
				// Convert cached response to streaming format
				return cache.StreamCachedResponse(c, cachedResp, requestID)
			}
			return c.JSON(cachedResp)
		}
	}

	if isStream {
		fiberlog.Infof("[%s] streaming %s response", requestID, protocolName)

		streamResp, err := prov.Chat().
			Completions().
			StreamCompletion(c.Context(), req.ToOpenAIParams())
		if err != nil {
			fiberlog.Errorf("[%s] %s stream failed: %v", requestID, protocolName, err)
			return fmt.Errorf("stream completion failed: %w", err)
		}

		return stream.HandleStream(c, streamResp, requestID, provider, cacheSource)
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
	// Set cache tier based on source
	models.SetCacheTier(&adaptiveResp.Usage, cacheSource)

	// Store successful response in prompt cache (only for non-streaming requests)
	if !isStream && cs.promptCache != nil {
		if err := cs.promptCache.Set(req, adaptiveResp, requestID); err != nil {
			fiberlog.Warnf("[%s] Failed to store response in prompt cache: %v", requestID, err)
		}
	}

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
	cacheSource string,
) error {
	// Try initial orchestration
	if isStream {
		streamResp, err := orchestrator.OrchestrateMinionSStream(
			c.Context(), remoteProv, minionProv, req, minionModel,
		)
		if err == nil {
			provider := minionProv.GetProviderName()
			return stream.HandleStream(c, streamResp, requestID, provider, cacheSource)
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
	standardAlternatives := make([]models.Alternative, len(resp.Standard.Alternatives))
	copy(standardAlternatives, resp.Standard.Alternatives)
	if len(standardAlternatives) > 0 {
		if altErr := cs.tryMinionSStandardAlternatives(c, orchestrator, minionProv, req, minionModel, &standardAlternatives, requestID, isStream, cacheSource); altErr == nil {
			return nil
		}
	}

	// Try minion alternatives
	minionAlternatives := make([]models.Alternative, len(resp.Minion.Alternatives))
	copy(minionAlternatives, resp.Minion.Alternatives)
	if len(minionAlternatives) > 0 {
		if altErr := cs.tryMinionSMinionAlternatives(c, orchestrator, remoteProv, req, &minionAlternatives, requestID, isStream, cacheSource); altErr == nil {
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
	alternatives *[]models.Alternative,
	requestID string,
	isStream bool,
	cacheSource string,
) error {
	fiberlog.Infof("[%s] Trying standard alternatives for MinionS restart", requestID)
	standardResult, err := cs.fallbackService.SelectAlternative(c.Context(), alternatives, req.ProviderConfigs, requestID)
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
		return stream.HandleStream(c, streamResp, requestID, provider, cacheSource)
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
	alternatives *[]models.Alternative,
	requestID string,
	isStream bool,
	cacheSource string,
) error {
	fiberlog.Infof("[%s] Trying minion alternatives for MinionS restart", requestID)
	minionResult, err := cs.fallbackService.SelectAlternative(c.Context(), alternatives, req.ProviderConfigs, requestID)
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
		return stream.HandleStream(c, streamResp, requestID, provider, cacheSource)
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
