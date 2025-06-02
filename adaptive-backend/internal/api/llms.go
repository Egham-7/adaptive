package api

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services"
	"adaptive-backend/internal/services/metrics"
	"adaptive-backend/internal/services/providers"
	"adaptive-backend/internal/services/stream_readers/stream"
	"time"

	"github.com/gofiber/fiber/v2"
)

type ChatCompletionHandler struct {
	promptClassifierClient *services.PromptClassifierClient
	metrics                *metrics.ChatMetrics
}

func NewChatCompletionHandler() *ChatCompletionHandler {
	chatMetrics := metrics.NewChatMetrics()

	return &ChatCompletionHandler{
		promptClassifierClient: services.NewPromptClassifierClient(),
		metrics:                chatMetrics,
	}
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

	modelName, cacheType, err := h.promptClassifierClient.SelectModelWithCache(prompt, apiKey, requestID)
	if h.metrics != nil {
		h.metrics.CacheLookups.WithLabelValues(cacheType).Inc()
		if cacheType == "user" || cacheType == "global" {
			h.metrics.CacheHits.WithLabelValues(cacheType).Inc()
		}
		h.metrics.ModelSelections.WithLabelValues(modelName).Inc()
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
		ResponseFormat:   req.ResponseFormat,
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
	return stream.HandleStream(c, resp, requestID, req.RequestOptions)
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

	modelName, cacheType, err := h.promptClassifierClient.SelectModelWithCache(prompt, apiKey, requestID)
	if h.metrics != nil {
		h.metrics.CacheLookups.WithLabelValues(cacheType).Inc()
		if cacheType == "user" || cacheType == "global" {
			h.metrics.CacheHits.WithLabelValues(cacheType).Inc()
		}
		h.metrics.ModelSelections.WithLabelValues(modelName).Inc()
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

