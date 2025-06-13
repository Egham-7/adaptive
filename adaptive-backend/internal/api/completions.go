package api

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/adaptive_ai"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/metrics"
	"adaptive-backend/internal/services/providers"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"adaptive-backend/internal/services/stream_readers/stream"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/botirk38/semanticcache"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/log"
	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/openai/openai-go"
)

type ChatCompletionHandler struct {
	promptClassifier       *adaptive_ai.PromptClassifierService
	modelSelector          *adaptive_ai.ModelSelectorService
	circuitBreaker         *circuitbreaker.CircuitBreaker
	globalPromptModelCache *semanticcache.SemanticCache[string, models.SelectModelResponse]
	userPromptModelCache   *lru.Cache[string, *semanticcache.SemanticCache[string, models.SelectModelResponse]]
	embeddingProvider      semanticcache.EmbeddingProvider
	metrics                *metrics.ChatMetrics
	providerConstraint     string
}

// Shared metrics instance to avoid duplicate registration
var sharedMetrics = metrics.NewChatMetrics()

func NewChatCompletionHandler() *ChatCompletionHandler {
	return createChatCompletionHandler("")
}

func NewProviderChatCompletionHandler(provider string) *ChatCompletionHandler {
	return createChatCompletionHandler(provider)
}

func createChatCompletionHandler(providerConstraint string) *ChatCompletionHandler {
	// Create prompt classifier config
	classifierConfig := models.PromptClassifierConfig{
		ModelID:      "botirk/tiny-prompt-task-complexity-classifier",
		ModelPath:    "./models/prompt_classifier",
		Timeout:      30 * time.Second,
		MaxRetries:   3,
		MaxSeqLength: 512,
	}

	// Create prompt classifier service
	promptClassifier, err := adaptive_ai.NewPromptClassifierService(classifierConfig, nil)
	if err != nil {
		log.Fatalf("Failed to create prompt classifier service: %v", err)
	}

	// Create model selector service
	modelSelector := adaptive_ai.NewModelSelectorService(promptClassifier, nil)

	// Create embedding provider for semantic cache
	embeddingProvider, err := semanticcache.NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "")
	if err != nil {
		log.Fatalf("Failed to create embedding provider: %v", err)
	}

	// Create global semantic cache
	globalCache, err := semanticcache.NewSemanticCache[string, models.SelectModelResponse](
		2000, // cache size
		embeddingProvider,
		nil,
	)
	if err != nil {
		log.Fatalf("Failed to create global semantic cache: %v", err)
	}

	// Create user cache pool
	userCache, err := lru.New[string, *semanticcache.SemanticCache[string, models.SelectModelResponse]](200)
	if err != nil {
		log.Fatalf("Failed to create user LRU cache: %v", err)
	}

	// Create circuit breaker
	circuitBreaker := circuitbreaker.NewWithConfig(circuitbreaker.Config{
		FailureThreshold: 3,
		SuccessThreshold: 2,
		Timeout:          20 * time.Second,
		ResetAfter:       2 * time.Minute,
	})

	return &ChatCompletionHandler{
		promptClassifier:       promptClassifier,
		modelSelector:          modelSelector,
		circuitBreaker:         circuitBreaker,
		globalPromptModelCache: globalCache,
		userPromptModelCache:   userCache,
		embeddingProvider:      embeddingProvider,
		metrics:                sharedMetrics,
		providerConstraint:     providerConstraint,
	}
}

func (h *ChatCompletionHandler) ChatCompletion(c *fiber.Ctx) error {
	start := time.Now()
	requestID := h.getRequestID(c)
	apiKey := h.getAPIKey(c)

	req, err := h.parseRequest(c)
	if err != nil {
		h.recordError(start, "400", false)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body: " + err.Error(),
		})
	}
	isStream := req.Stream

	modelInfo, err := h.selectModel(req, apiKey, requestID)
	if err != nil {
		h.recordError(start, "500", isStream)
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}
	if h.providerConstraint != "" {
		log.Infof("[%s] Selected model: %s (%s)", requestID, modelInfo.SelectedModel, h.providerConstraint)
	} else {
		log.Infof("[%s] Selected model: %s", requestID, modelInfo.SelectedModel)
	}

	// Record model selection metric
	h.recordModelSelection(modelInfo.SelectedModel)

	h.applyModelParameters(req, modelInfo)
	log.Infof("[%s] Applied model parameters: %+v", requestID, modelInfo.Parameters)

	provider, err := providers.NewLLMProvider(modelInfo.Provider)
	if err != nil {
		h.recordError(start, "400", isStream)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	log.Infof("[%s] Using provider: %s", requestID, modelInfo.Provider)

	if isStream {
		log.Infof("[%s] Streaming enabled", requestID)
		return h.handleStreamResponse(c, provider, req, start, requestID)
	}
	log.Infof("[%s] Streaming disabled", requestID)
	return h.handleRegularResponse(c, provider, req, start)
}

func (h *ChatCompletionHandler) getRequestID(c *fiber.Ctx) string {
	return c.Get("X-Request-ID", time.Now().String())
}

func (h *ChatCompletionHandler) getAPIKey(c *fiber.Ctx) string {
	apiKey := string(c.Request().Header.Peek("X-Stainless-API-Key"))
	if apiKey == "" {
		return "anonymous"
	}
	return apiKey
}

func (h *ChatCompletionHandler) parseRequest(c *fiber.Ctx) (*models.ChatCompletionRequest, error) {
	requestID := h.getRequestID(c)

	// Get and log the raw body
	body := c.Body()
	log.Infof("[%s] Raw request body: %s", requestID, string(body))

	var req models.ChatCompletionRequest
	if err := c.BodyParser(&req); err != nil {
		log.Errorf("[%s] Failed to parse request body: %v", requestID, err)
		return nil, err
	}

	// Log the parsed request
	reqJSON, _ := json.Marshal(req)
	log.Infof("[%s] Parsed request: %s", requestID, string(reqJSON))

	return &req, nil
}

func (h *ChatCompletionHandler) selectModel(req *models.ChatCompletionRequest, apiKey, requestID string) (*models.SelectModelResponse, error) {
	if len(req.Messages) == 0 {
		return nil, errors.New("messages array must contain at least one element")
	}

	prompt := req.Messages[len(req.Messages)-1].OfUser.Content.OfString.Value

	selectReq := models.SelectModelRequest{
		Prompt:   prompt,
		Provider: h.providerConstraint,
	}

	modelInfo, cacheType, err := h.selectModelWithCache(selectReq, apiKey, requestID)
	if err != nil {
		log.Errorf("[%s] Model selection error: %v", requestID, err)
		return nil, err
	}

	// Record cache metrics
	if cacheType == "user" || cacheType == "global" {
		h.recordCacheHit(cacheType)
	}

	return modelInfo, err
}

func (h *ChatCompletionHandler) selectModelWithCache(req models.SelectModelRequest, userID string, requestID string) (*models.SelectModelResponse, string, error) {
	threshold := float32(0.85)
	userCache := h.getUserCache(userID)

	// Create cache key that includes provider constraint
	cacheKey := req.Prompt
	if req.Provider != "" {
		cacheKey = fmt.Sprintf("%s:provider:%s", req.Prompt, req.Provider)
	}

	// Check user-specific cache
	if userCache != nil {
		if val, found, err := userCache.Lookup(cacheKey, threshold); err == nil && found {
			log.Infof("[%s] Semantic cache hit (user)", requestID)
			return &val, "user", nil
		}
	}

	// Check global cache
	if val, found, err := h.globalPromptModelCache.Lookup(cacheKey, threshold); err == nil && found {
		log.Infof("[%s] Semantic cache hit (global)", requestID)

		if userCache != nil {
			_ = userCache.Set(cacheKey, cacheKey, val)
		}

		return &val, "global", nil
	}

	// Cache miss - use local model selection
	start := time.Now()

	if !h.circuitBreaker.CanExecute() {
		h.circuitBreaker.RecordRequestDuration(time.Since(start), false)
		log.Infof("[%s] Circuit breaker open, using fallback", requestID)
		return h.getFallbackModel(req.Provider), "fallback", nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	modelInfo, err := h.modelSelector.SelectModel(ctx, req)
	if err != nil {
		h.circuitBreaker.RecordFailure()
		h.circuitBreaker.RecordRequestDuration(time.Since(start), false)
		log.Errorf("[%s] Local model selection failed, using fallback: %v", requestID, err)
		return h.getFallbackModel(req.Provider), "fallback", nil
	}

	h.circuitBreaker.RecordSuccess()
	h.circuitBreaker.RecordRequestDuration(time.Since(start), true)

	log.Infof("[%s] Cache miss - selected model %s from %s", requestID, modelInfo.SelectedModel, modelInfo.Provider)

	// Cache the result
	if err := h.globalPromptModelCache.Set(cacheKey, cacheKey, *modelInfo); err != nil {
		log.Errorf("[%s] Failed to cache in global cache: %v", requestID, err)
	}

	if userCache != nil {
		if err := userCache.Set(cacheKey, cacheKey, *modelInfo); err != nil {
			log.Errorf("[%s] Failed to cache in user cache: %v", requestID, err)
		}
	}

	return modelInfo, "miss", nil
}

func (h *ChatCompletionHandler) getUserCache(userID string) *semanticcache.SemanticCache[string, models.SelectModelResponse] {
	if cache, ok := h.userPromptModelCache.Get(userID); ok {
		return cache
	}

	newCache, err := semanticcache.NewSemanticCache[string, models.SelectModelResponse](
		150, // user cache size
		h.embeddingProvider,
		nil,
	)
	if err != nil {
		log.Errorf("Warning: Failed to create user cache for %s: %v", userID, err)
		return nil
	}

	h.userPromptModelCache.Add(userID, newCache)
	return newCache
}

func (h *ChatCompletionHandler) getFallbackModel(provider string) *models.SelectModelResponse {
	var selectedModel string
	var fallbackProvider string

	if provider != "" {
		// Provider-specific fallbacks
		switch provider {
		case "openai":
			selectedModel = "gpt-4o"
			fallbackProvider = "openai"
		case "anthropic":
			selectedModel = "claude-3-5-sonnet-20241022"
			fallbackProvider = "anthropic"
		case "groq":
			selectedModel = "llama-3.2-3b-preview"
			fallbackProvider = "groq"
		case "deepseek":
			selectedModel = "deepseek-chat"
			fallbackProvider = "deepseek"
		case "gemini":
			selectedModel = "gemini-pro"
			fallbackProvider = "gemini"
		default:
			selectedModel = "gpt-4o"
			fallbackProvider = "openai"
		}
	} else {
		selectedModel = "gpt-4o"
		fallbackProvider = "openai"
	}

	return &models.SelectModelResponse{
		SelectedModel: selectedModel,
		Provider:      fallbackProvider,
		Parameters: openai.ChatCompletionNewParams{
			Model:            selectedModel,
			MaxTokens:        openai.Int(4096),
			Temperature:      openai.Float(0.7),
			TopP:             openai.Float(1.0),
			FrequencyPenalty: openai.Float(0.0),
			PresencePenalty:  openai.Float(0.0),
			N:                openai.Int(1),
		},
	}
}

func (h *ChatCompletionHandler) applyModelParameters(req *models.ChatCompletionRequest, modelInfo *models.SelectModelResponse) {
	req.Model = modelInfo.SelectedModel
	req.MaxTokens = modelInfo.Parameters.MaxCompletionTokens
	req.Temperature = modelInfo.Parameters.Temperature
	req.TopP = modelInfo.Parameters.TopP
	req.PresencePenalty = modelInfo.Parameters.PresencePenalty
	req.FrequencyPenalty = modelInfo.Parameters.FrequencyPenalty
	req.N = modelInfo.Parameters.N
	req.TopLogprobs = modelInfo.Parameters.TopLogprobs
	req.Logprobs = modelInfo.Parameters.Logprobs
	req.MaxCompletionTokens = modelInfo.Parameters.MaxCompletionTokens
	req.ReasoningEffort = modelInfo.Parameters.ReasoningEffort
}

func (h *ChatCompletionHandler) getMethodType(isStream bool) string {
	if isStream {
		return "stream"
	}
	return "completion"
}

func (h *ChatCompletionHandler) getProviderName() string {
	if h.providerConstraint != "" {
		return h.providerConstraint
	}
	return "unknown"
}

func (h *ChatCompletionHandler) recordError(start time.Time, statusCode string, isStream bool) {
	if h.metrics != nil {
		methodType := h.getMethodType(isStream)
		provider := h.getProviderName()
		h.metrics.RequestDuration.WithLabelValues(methodType, statusCode, provider).Observe(time.Since(start).Seconds())
	}
}

func (h *ChatCompletionHandler) recordSuccess(start time.Time, isStream bool) {
	if h.metrics != nil {
		methodType := h.getMethodType(isStream)
		provider := h.getProviderName()
		h.metrics.RequestDuration.WithLabelValues(methodType, "200", provider).Observe(time.Since(start).Seconds())
	}
}

func (h *ChatCompletionHandler) recordCacheHit(cacheType string) {
	if h.metrics != nil {
		provider := h.getProviderName()
		h.metrics.CacheHits.WithLabelValues(cacheType, provider).Inc()
	}
}

func (h *ChatCompletionHandler) recordModelSelection(model string) {
	if h.metrics != nil {
		provider := h.getProviderName()
		h.metrics.ModelSelections.WithLabelValues(model, provider).Inc()
	}
}

func (h *ChatCompletionHandler) handleStreamResponse(c *fiber.Ctx, provider provider_interfaces.LLMProvider, req *models.ChatCompletionRequest, start time.Time, requestID string) error {
	resp, err := provider.Chat().Completions().StreamCompletion(req.ToOpenAIParams())
	if err != nil {
		h.recordError(start, "500", true)
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to stream completion: " + err.Error(),
		})
	}

	h.setStreamHeaders(c)
	h.recordSuccess(start, true)

	log.Infof("[%s] Starting stream handling", requestID)

	return stream.HandleStream(c, resp, requestID)
}

func (h *ChatCompletionHandler) handleRegularResponse(c *fiber.Ctx, provider provider_interfaces.LLMProvider, req *models.ChatCompletionRequest, start time.Time) error {
	resp, err := provider.Chat().Completions().CreateCompletion(req.ToOpenAIParams())
	if err != nil {
		h.recordError(start, "500", false)
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to generate completion: " + err.Error(),
		})
	}

	h.recordSuccess(start, false)
	return c.JSON(resp)
}

func (h *ChatCompletionHandler) setStreamHeaders(c *fiber.Ctx) {
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Transfer-Encoding", "chunked")
}
