package api

import (
	"time"

	"adaptive-backend/internal/services"
	"adaptive-backend/internal/services/metrics"
	"adaptive-backend/internal/services/providers"
	"adaptive-backend/internal/services/stream_readers/stream"

	"github.com/gofiber/fiber/v2"
	"github.com/openai/openai-go"
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

	var req openai.ChatCompletionNewParams
	if err := c.BodyParser(&req); err != nil {
		if h.metrics != nil {
			h.metrics.RequestDuration.WithLabelValues("stream", "400").Observe(time.Since(start).Seconds())
		}
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body: " + err.Error(),
		})
	}

	prompt := req.Messages[len(req.Messages)-1].OfUser.Content.OfString.Value

	modelInfo, cacheType, err := h.promptClassifierClient.SelectModelWithCache(prompt, apiKey, requestID)
	if h.metrics != nil {
		h.metrics.CacheLookups.WithLabelValues(cacheType).Inc()
		if cacheType == "user" || cacheType == "global" {
			h.metrics.CacheHits.WithLabelValues(cacheType).Inc()
		}
		h.metrics.ModelSelections.WithLabelValues(modelInfo.SelectedModel).Inc()
	}
	if err != nil {
		if h.metrics != nil {
			h.metrics.RequestDuration.WithLabelValues("stream", "500").Observe(time.Since(start).Seconds())
		}
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}

	req.Model = modelInfo.SelectedModel
	req.Temperature = openai.Float(modelInfo.Parameters.Temperature)
	req.N = openai.Int(modelInfo.Parameters.N)
	req.PresencePenalty = openai.Float(modelInfo.Parameters.PresencePenalty)
	req.MaxTokens = openai.Int(modelInfo.Parameters.MaxTokens)
	req.FrequencyPenalty = openai.Float(modelInfo.Parameters.FrequencyPenalty)
	req.TopP = openai.Float(modelInfo.Parameters.TopP)

	provider, err := providers.NewLLMProvider(modelInfo.Provider)
	if err != nil {
		if h.metrics != nil {
			h.metrics.RequestDuration.WithLabelValues("stream", "400").Observe(time.Since(start).Seconds())
		}
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}

	resp, err := provider.Chat().Completions().StreamCompletion(&req)
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
	return stream.HandleStream(c, resp, requestID)
}

func (h *ChatCompletionHandler) ChatCompletion(c *fiber.Ctx) error {
	start := time.Now()
	requestID := c.Get("X-Request-ID", time.Now().String())
	apiKey := string(c.Request().Header.Peek("X-API-Key"))
	if apiKey == "" {
		apiKey = "anonymous"
	}

	var req openai.ChatCompletionNewParams
	if err := c.BodyParser(&req); err != nil {
		if h.metrics != nil {
			h.metrics.RequestDuration.WithLabelValues("completion", "400").Observe(time.Since(start).Seconds())
		}
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid request body: " + err.Error()})
	}

	prompt := req.Messages[len(req.Messages)-1].OfUser.Content.OfString.Value

	modelInfo, cacheType, err := h.promptClassifierClient.SelectModelWithCache(prompt, apiKey, requestID)
	if h.metrics != nil {
		h.metrics.CacheLookups.WithLabelValues(cacheType).Inc()
		if cacheType == "user" || cacheType == "global" {
			h.metrics.CacheHits.WithLabelValues(cacheType).Inc()
		}
		h.metrics.ModelSelections.WithLabelValues(modelInfo.SelectedModel).Inc()
	}
	if err != nil {
		if h.metrics != nil {
			h.metrics.RequestDuration.WithLabelValues("completion", "500").Observe(time.Since(start).Seconds())
		}
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}

	provider, err := providers.NewLLMProvider(modelInfo.Provider)
	if err != nil {
		if h.metrics != nil {
			h.metrics.RequestDuration.WithLabelValues("completion", "400").Observe(time.Since(start).Seconds())
		}
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}
	req.Model = modelInfo.SelectedModel
	req.N = openai.Int(modelInfo.Parameters.N)
	req.Temperature = openai.Float(modelInfo.Parameters.Temperature)
	req.PresencePenalty = openai.Float(modelInfo.Parameters.PresencePenalty)
	req.MaxTokens = openai.Int(modelInfo.Parameters.MaxTokens)
	req.FrequencyPenalty = openai.Float(modelInfo.Parameters.FrequencyPenalty)
	req.TopP = openai.Float(modelInfo.Parameters.TopP)

	resp, err := provider.Chat().Completions().CreateCompletion(&req)
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
