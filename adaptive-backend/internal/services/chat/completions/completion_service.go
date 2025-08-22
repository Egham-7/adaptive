package completions

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/cache"
	"adaptive-backend/internal/services/stream_readers/stream"
	"fmt"
	"net/http"
	"time"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

// CompletionService handles completion requests with fallback logic.
type CompletionService struct {
	cfg         *config.Config
	promptCache *cache.PromptCache
}

// NewCompletionService creates a new completion service.
func NewCompletionService(cfg *config.Config) *CompletionService {
	// Initialize prompt cache, but don't fail if it can't be created
	promptCache, err := cache.NewPromptCache(cfg)
	if err != nil {
		fiberlog.Warnf("Failed to initialize prompt cache: %v", err)
		promptCache = nil
	}

	return &CompletionService{
		cfg:         cfg,
		promptCache: promptCache,
	}
}

// createClient creates an OpenAI client for the given provider
func (cs *CompletionService) createClient(providerName string, customConfigs map[string]*models.ProviderConfig) (*openai.Client, error) {
	// Merge YAML config with custom override config if provided
	if customConfigs != nil {
		if customConfig, hasCustom := customConfigs[providerName]; hasCustom {
			providerConfig, err := cs.cfg.MergeProviderConfig(providerName, customConfig)
			if err != nil {
				return nil, fmt.Errorf("failed to merge provider config for '%s': %w", providerName, err)
			}
			return cs.buildClient(providerConfig, providerName)
		}
	}

	// Use YAML config only
	providerConfig, exists := cs.cfg.GetProviderConfig(providerName)
	if !exists {
		return nil, fmt.Errorf("provider '%s' is not configured", providerName)
	}

	return cs.buildClient(providerConfig, providerName)
}

func (cs *CompletionService) buildClient(providerConfig models.ProviderConfig, providerName string) (*openai.Client, error) {
	if providerConfig.APIKey == "" {
		return nil, fmt.Errorf("%s API key not set in configuration", providerName)
	}

	opts := []option.RequestOption{
		option.WithAPIKey(providerConfig.APIKey),
	}

	if providerConfig.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(providerConfig.BaseURL))
	}

	if providerConfig.Headers != nil {
		for key, value := range providerConfig.Headers {
			opts = append(opts, option.WithHeader(key, value))
		}
	}

	if providerConfig.TimeoutMs > 0 {
		timeout := time.Duration(providerConfig.TimeoutMs) * time.Millisecond
		httpClient := &http.Client{Timeout: timeout}
		opts = append(opts, option.WithHTTPClient(httpClient))
	}

	client := openai.NewClient(opts...)
	return &client, nil
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
	// Try primary provider first
	client, err := cs.createClient(standardInfo.Provider, req.ProviderConfigs)
	if err == nil {
		fiberlog.Infof("[%s] Using primary standard provider: %s (%s)", requestID, standardInfo.Provider, standardInfo.Model)
		req.Model = shared.ChatModel(standardInfo.Model)
		if execErr := cs.executeCompletion(c, client, standardInfo.Provider, req, requestID, isStream, cacheSource); execErr == nil {
			return nil
		} else {
			fiberlog.Warnf("[%s] Primary standard provider %s (%s) execution failed: %v", requestID, standardInfo.Provider, standardInfo.Model, execErr)
		}
	} else {
		fiberlog.Warnf("[%s] Primary standard provider %s creation failed: %v", requestID, standardInfo.Provider, err)
	}

	// Try alternatives
	for _, alt := range standardInfo.Alternatives {
		client, err := cs.createClient(alt.Provider, req.ProviderConfigs)
		if err != nil {
			fiberlog.Warnf("[%s] Alternative standard provider %s creation failed: %v", requestID, alt.Provider, err)
			continue
		}

		fiberlog.Infof("[%s] Using alternative standard provider: %s (%s)", requestID, alt.Provider, alt.Model)
		req.Model = shared.ChatModel(alt.Model)
		if execErr := cs.executeCompletion(c, client, alt.Provider, req, requestID, isStream, cacheSource); execErr == nil {
			return nil
		} else {
			fiberlog.Warnf("[%s] Alternative standard provider %s (%s) execution failed: %v", requestID, alt.Provider, alt.Model, execErr)
		}
	}

	return fmt.Errorf("all standard providers failed (creation or execution)")
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
	// Try primary provider first
	client, err := cs.createClient(minionInfo.Provider, req.ProviderConfigs)
	if err == nil {
		fiberlog.Infof("[%s] Using primary minion provider: %s (%s)", requestID, minionInfo.Provider, minionInfo.Model)
		req.Model = shared.ChatModel(minionInfo.Model)
		if execErr := cs.executeCompletion(c, client, minionInfo.Provider, req, requestID, isStream, cacheSource); execErr == nil {
			return nil
		} else {
			fiberlog.Warnf("[%s] Primary minion provider %s (%s) execution failed: %v", requestID, minionInfo.Provider, minionInfo.Model, execErr)
		}
	} else {
		fiberlog.Warnf("[%s] Primary minion provider %s creation failed: %v", requestID, minionInfo.Provider, err)
	}

	// Try alternatives
	for _, alt := range minionInfo.Alternatives {
		client, err := cs.createClient(alt.Provider, req.ProviderConfigs)
		if err != nil {
			fiberlog.Warnf("[%s] Alternative minion provider %s creation failed: %v", requestID, alt.Provider, err)
			continue
		}

		fiberlog.Infof("[%s] Using alternative minion provider: %s (%s)", requestID, alt.Provider, alt.Model)
		req.Model = shared.ChatModel(alt.Model)
		if execErr := cs.executeCompletion(c, client, alt.Provider, req, requestID, isStream, cacheSource); execErr == nil {
			return nil
		} else {
			fiberlog.Warnf("[%s] Alternative minion provider %s (%s) execution failed: %v", requestID, alt.Provider, alt.Model, execErr)
		}
	}

	return fmt.Errorf("all minion providers failed (creation or execution)")
}

// executeCompletion handles the actual completion execution
func (cs *CompletionService) executeCompletion(
	c *fiber.Ctx,
	client *openai.Client,
	providerName string,
	req *models.ChatCompletionRequest,
	requestID string,
	isStream bool,
	cacheSource string,
) error {
	// Check prompt cache first
	if cs.promptCache != nil {
		if cachedResp, found := cs.promptCache.Get(req, requestID); found {
			fiberlog.Infof("[%s] Prompt cache hit", requestID)
			models.SetCacheTier(&cachedResp.Usage, "prompt_response")
			if isStream {
				return cache.StreamCachedResponse(c, cachedResp, requestID)
			}
			return c.JSON(cachedResp)
		}
	}

	if isStream {
		fiberlog.Infof("[%s] streaming response from %s", requestID, providerName)
		streamResp := client.Chat.Completions.NewStreaming(c.Context(), *req.ToOpenAIParams())
		return stream.HandleStream(c, streamResp, requestID, providerName, cacheSource)
	}

	fiberlog.Infof("[%s] generating completion from %s", requestID, providerName)
	resp, err := client.Chat.Completions.New(c.Context(), *req.ToOpenAIParams())
	if err != nil {
		return fmt.Errorf("completion failed: %w", err)
	}

	adaptiveResp := models.ConvertToAdaptive(resp, providerName)
	models.SetCacheTier(&adaptiveResp.Usage, cacheSource)

	// Store in prompt cache
	if cs.promptCache != nil {
		if err := cs.promptCache.Set(req, adaptiveResp, requestID); err != nil {
			fiberlog.Warnf("[%s] Failed to store response in prompt cache: %v", requestID, err)
		}
	}

	return c.JSON(adaptiveResp)
}
