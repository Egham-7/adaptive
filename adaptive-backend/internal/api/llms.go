package api

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services"
	"adaptive-backend/internal/services/metrics"
	"adaptive-backend/internal/services/providers"
	"adaptive-backend/internal/services/stream_readers"
	"log"
	"os"
	"time"

	"github.com/botirk38/semanticcache"
	"github.com/gofiber/fiber/v2"
	lru "github.com/hashicorp/golang-lru/v2"
)

type ChatCompletionHandler struct {
	promptClassifierClient *services.PromptClassifierClient
	globalPromptModelCache *semanticcache.SemanticCache[string, string]
	userPromptModelCache   *lru.Cache[string, *semanticcache.SemanticCache[string, string]]
	embeddingProvider      semanticcache.EmbeddingProvider
	metrics                *metrics.ChatMetrics
}

func NewChatCompletionHandler() *ChatCompletionHandler {
	chatMetrics := metrics.NewChatMetrics()
	embeddingProvider, err := semanticcache.NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "")
	if err != nil {
		log.Fatalf("Failed to create embedding provider: %v", err)
	}

	globalCache, err := semanticcache.NewSemanticCache[string, string](1000, embeddingProvider, nil)
	if err != nil {
		log.Fatalf("Failed to create global semantic cache: %v", err)
	}

	userCache, err := lru.New[string, *semanticcache.SemanticCache[string, string]](100)
	if err != nil {
		log.Fatalf("Failed to create user LRU cache: %v", err)
	}

	return &ChatCompletionHandler{
		promptClassifierClient: services.NewPromptClassifierClient(),
		globalPromptModelCache: globalCache,
		userPromptModelCache:   userCache,
		embeddingProvider:      embeddingProvider,
		metrics:                chatMetrics,
	}
}

func (h *ChatCompletionHandler) getUserCache(userID string) *semanticcache.SemanticCache[string, string] {
	if cache, ok := h.userPromptModelCache.Get(userID); ok {
		return cache
	}
	newCache, err := semanticcache.NewSemanticCache[string, string](100, h.embeddingProvider, nil)
	if err != nil {
		log.Printf("[WARN] Failed to create user semantic cache for %s: %v", userID, err)
		return nil
	}
	h.userPromptModelCache.Add(userID, newCache)
	return newCache
}

func (h *ChatCompletionHandler) getModelFromCacheOrSelect(prompt string, userID string, requestID string) (string, string, error) {
	const threshold = 0.9
	userCache := h.getUserCache(userID)

	// User-level semantic cache
	if userCache != nil {
		if val, found, err := userCache.Lookup(prompt, threshold); err == nil && found {
			log.Printf("[%s] Semantic cache hit (user)", requestID)
			if h.metrics != nil {
				h.metrics.CacheHits.WithLabelValues("user").Inc()
				h.metrics.ModelSelections.WithLabelValues(val).Inc()
			}
			return val, "user", nil
		}
	}

	// Global semantic cache
	if val, found, err := h.globalPromptModelCache.Lookup(prompt, threshold); err == nil && found {
		log.Printf("[%s] Semantic cache hit (global)", requestID)
		if userCache != nil {
			_ = userCache.Set(prompt, prompt, val)
		}
		if h.metrics != nil {
			h.metrics.CacheHits.WithLabelValues("global").Inc()
			h.metrics.ModelSelections.WithLabelValues(val).Inc()
		}
		return val, "global", nil
	}

	// Cache miss, select model
	modelInfo, err := h.promptClassifierClient.SelectModel(prompt)
	if err != nil {
		return "", "miss", err
	}
	log.Printf("[%s] Semantic cache miss - selected model %s", requestID, modelInfo.SelectedModel)

	_ = h.globalPromptModelCache.Set(prompt, prompt, modelInfo.SelectedModel)
	if userCache != nil {
		_ = userCache.Set(prompt, prompt, modelInfo.SelectedModel)
	}
	if h.metrics != nil {
		h.metrics.ModelSelections.WithLabelValues(modelInfo.SelectedModel).Inc()
	}
	return modelInfo.SelectedModel, "miss", nil
}

func (h *ChatCompletionHandler) StreamChatCompletion(c *fiber.Ctx) error {
	start := time.Now()
	requestID := c.Get("X-Request-ID", time.Now().String())
	apiKey := string(c.Request().Header.Peek("X-API-Key"))
	if apiKey == "" {
		apiKey = "anonymous"
	}

	var req models.ChatCompletionRequest
	if err := c.BodyParser(&req); err != nil {
		if h.metrics != nil {
			h.metrics.RequestDuration.WithLabelValues("stream", "400").Observe(time.Since(start).Seconds())
		}
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body: " + err.Error(),
		})
	}

	prompt := req.Messages[len(req.Messages)-1].Content

	modelName, cacheType, err := h.getModelFromCacheOrSelect(prompt, apiKey, requestID)
	if h.metrics != nil {
		h.metrics.CacheLookups.WithLabelValues(cacheType).Inc()
	}
	if err != nil {
		if h.metrics != nil {
			h.metrics.RequestDuration.WithLabelValues("stream", "500").Observe(time.Since(start).Seconds())
		}
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}

	fullReq := models.ProviderChatCompletionRequest{
		Provider:         "openai",
		Model:            modelName,
		Messages:         req.Messages,
		Temperature:      0.7,
		N:                1,
		MaxTokens:        256,
		PresencePenalty:  0,
		FrequencyPenalty: 0,
		Stream:           true,
	}

	provider, err := providers.NewLLMProvider(fullReq.Provider)
	if err != nil {
		if h.metrics != nil {
			h.metrics.RequestDuration.WithLabelValues("stream", "400").Observe(time.Since(start).Seconds())
		}
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}

	resp, err := provider.StreamChatCompletion(&fullReq)
	if err != nil {
		if h.metrics != nil {
			h.metrics.RequestDuration.WithLabelValues("stream", "500").Observe(time.Since(start).Seconds())
		}
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to stream completion: " + err.Error(),
		})
	}

	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Transfer-Encoding", "chunked")

	if h.metrics != nil {
		h.metrics.RequestDuration.WithLabelValues("stream", "200").Observe(time.Since(start).Seconds())
	}
	return stream_readers.HandleStream(c, resp, requestID)
}

func (h *ChatCompletionHandler) ChatCompletion(c *fiber.Ctx) error {
	start := time.Now()
	requestID := c.Get("X-Request-ID", time.Now().String())
	apiKey := string(c.Request().Header.Peek("X-API-Key"))
	if apiKey == "" {
		apiKey = "anonymous"
	}

	var req models.ChatCompletionRequest
	if err := c.BodyParser(&req); err != nil {
		if h.metrics != nil {
			h.metrics.RequestDuration.WithLabelValues("completion", "400").Observe(time.Since(start).Seconds())
		}
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid request body: " + err.Error()})
	}

	prompt := req.Messages[len(req.Messages)-1].Content

	modelName, cacheType, err := h.getModelFromCacheOrSelect(prompt, apiKey, requestID)
	if h.metrics != nil {
		h.metrics.CacheLookups.WithLabelValues(cacheType).Inc()
	}
	if err != nil {
		if h.metrics != nil {
			h.metrics.RequestDuration.WithLabelValues("completion", "500").Observe(time.Since(start).Seconds())
		}
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}

	fullReq := models.ProviderChatCompletionRequest{
		Provider:         "openai",
		Model:            modelName,
		Messages:         req.Messages,
		Temperature:      0.7,
		N:                1,
		MaxTokens:        256,
		PresencePenalty:  0,
		FrequencyPenalty: 0,
	}

	provider, err := providers.NewLLMProvider(fullReq.Provider)
	if err != nil {
		if h.metrics != nil {
			h.metrics.RequestDuration.WithLabelValues("completion", "400").Observe(time.Since(start).Seconds())
		}
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}

	resp, err := provider.CreateChatCompletion(&fullReq)
	if err != nil {
		if h.metrics != nil {
			h.metrics.RequestDuration.WithLabelValues("completion", "500").Observe(time.Since(start).Seconds())
		}
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to generate completion: " + err.Error(),
		})
	}

	if h.metrics != nil {
		h.metrics.RequestDuration.WithLabelValues("completion", "200").Observe(time.Since(start).Seconds())
	}
	return c.JSON(resp)
}
