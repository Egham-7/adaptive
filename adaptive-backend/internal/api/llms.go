package api

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services"
	"adaptive-backend/internal/services/metrics"
	"adaptive-backend/internal/services/providers"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"adaptive-backend/internal/services/stream_readers/stream"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/openai/openai-go"
)

type ChatCompletionHandler struct {
	promptClassifierClient *services.PromptClassifierClient
	metrics                *metrics.ChatMetrics
}

func NewChatCompletionHandler() *ChatCompletionHandler {
	return &ChatCompletionHandler{
		promptClassifierClient: services.NewPromptClassifierClient(),
		metrics:                metrics.NewChatMetrics(),
	}
}

func (h *ChatCompletionHandler) ChatCompletion(c *fiber.Ctx) error {
	start := time.Now()
	requestID := h.getRequestID(c)
	apiKey := h.getAPIKey(c)

	req, isStream, err := h.parseRequest(c)
	if err != nil {
		h.recordError(start, "400", isStream)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body: " + err.Error(),
		})
	}

	modelInfo, err := h.selectModel(req, apiKey, requestID)
	if err != nil {
		h.recordError(start, "500", isStream)
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	h.applyModelParameters(req, modelInfo)

	provider, err := providers.NewLLMProvider(modelInfo.Provider)
	if err != nil {
		h.recordError(start, "400", isStream)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	if isStream {
		return h.handleStreamResponse(c, provider, req, start, requestID)
	}

	return h.handleRegularResponse(c, provider, req, start)
}

func (h *ChatCompletionHandler) getRequestID(c *fiber.Ctx) string {
	return c.Get("X-Request-ID", time.Now().String())
}

func (h *ChatCompletionHandler) getAPIKey(c *fiber.Ctx) string {
	apiKey := string(c.Request().Header.Peek("X-API-Key"))
	if apiKey == "" {
		return "anonymous"
	}
	return apiKey
}

func (h *ChatCompletionHandler) parseRequest(c *fiber.Ctx) (*openai.ChatCompletionNewParams, bool, error) {
	var reqBody map[string]interface{}
	if err := c.BodyParser(&reqBody); err != nil {
		return nil, false, err
	}

	// Check if stream is requested
	isStream := false
	if streamVal, exists := reqBody["stream"]; exists {
		if streamBool, ok := streamVal.(bool); ok {
			isStream = streamBool
		}
	}

	// Parse the request body into ChatCompletionNewParams
	var req openai.ChatCompletionNewParams
	if err := c.BodyParser(&req); err != nil {
		return nil, isStream, err
	}

	return &req, isStream, nil
}

func (h *ChatCompletionHandler) selectModel(req *openai.ChatCompletionNewParams, apiKey, requestID string) (*models.SelectModelResponse, error) {
	prompt := req.Messages[len(req.Messages)-1].OfUser.Content.OfString.Value

	modelInfo, cacheType, err := h.promptClassifierClient.SelectModelWithCache(prompt, apiKey, requestID)

	if h.metrics != nil {
		h.metrics.CacheLookups.WithLabelValues(cacheType).Inc()
		if cacheType == "user" || cacheType == "global" {
			h.metrics.CacheHits.WithLabelValues(cacheType).Inc()
		}
		if err == nil {
			h.metrics.ModelSelections.WithLabelValues(modelInfo.SelectedModel).Inc()
		}
	}

	return modelInfo, err
}

func (h *ChatCompletionHandler) applyModelParameters(req *openai.ChatCompletionNewParams, modelInfo *models.SelectModelResponse) {
	req.Model = modelInfo.SelectedModel
	req.Temperature = modelInfo.Parameters.Temperature
	req.N = modelInfo.Parameters.N
	req.PresencePenalty = modelInfo.Parameters.PresencePenalty
	req.MaxTokens = modelInfo.Parameters.MaxTokens
	req.FrequencyPenalty = modelInfo.Parameters.FrequencyPenalty
	req.TopP = modelInfo.Parameters.TopP
}

func (h *ChatCompletionHandler) getMethodType(isStream bool) string {
	if isStream {
		return "stream"
	}
	return "completion"
}

func (h *ChatCompletionHandler) recordError(start time.Time, statusCode string, isStream bool) {
	if h.metrics != nil {
		methodType := h.getMethodType(isStream)
		h.metrics.RequestDuration.WithLabelValues(methodType, statusCode).Observe(time.Since(start).Seconds())
	}
}

func (h *ChatCompletionHandler) recordSuccess(start time.Time, isStream bool) {
	if h.metrics != nil {
		methodType := h.getMethodType(isStream)
		h.metrics.RequestDuration.WithLabelValues(methodType, "200").Observe(time.Since(start).Seconds())
	}
}

func (h *ChatCompletionHandler) handleStreamResponse(c *fiber.Ctx, provider provider_interfaces.LLMProvider, req *openai.ChatCompletionNewParams, start time.Time, requestID string) error {
	resp, err := provider.Chat().Completions().StreamCompletion(req)
	if err != nil {
		h.recordError(start, "500", true)
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to stream completion: " + err.Error(),
		})
	}

	h.setStreamHeaders(c)
	h.recordSuccess(start, true)

	return stream.HandleStream(c, resp, requestID)
}

func (h *ChatCompletionHandler) handleRegularResponse(c *fiber.Ctx, provider provider_interfaces.LLMProvider, req *openai.ChatCompletionNewParams, start time.Time) error {
	resp, err := provider.Chat().Completions().CreateCompletion(req)
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
