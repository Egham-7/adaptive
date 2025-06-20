package api

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services"
	"adaptive-backend/internal/services/metrics"
	"adaptive-backend/internal/services/minions"
	"adaptive-backend/internal/services/providers"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"adaptive-backend/internal/services/stream_readers/stream"
	"encoding/json"
	"errors"
	"os"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/log"
)

type ChatCompletionHandler struct {
	promptClassifierClient *services.PromptClassifierClient
	metrics                *metrics.ChatMetrics
	providerConstraint     string
	minionRegistry         *minions.MinionRegistry
}

// Shared metrics instance to avoid duplicate registration
var sharedMetrics = metrics.NewChatMetrics()

func NewChatCompletionHandler() *ChatCompletionHandler {
	minionRegistry := minions.NewMinionRegistry(11)

	minionRegistry.RegisterMinion("Open QA", os.Getenv("OPEN_QA_MINION_URL"))
	minionRegistry.RegisterMinion("Closed QA", os.Getenv("CLOSED_QA_MINION_URL"))
	minionRegistry.RegisterMinion("Summarization", os.Getenv("SUMMARIZATION_MINION_URL"))
	minionRegistry.RegisterMinion("Text Generation", os.Getenv("TEXT_GENERATION_MINION_URL"))
	minionRegistry.RegisterMinion("Classification", os.Getenv("CLASSIFICATION_MINION_URL"))
	minionRegistry.RegisterMinion("Code Generation", os.Getenv("CODE_GENERATION_MINION_URL"))
	minionRegistry.RegisterMinion("Chatbot", os.Getenv("CHATBOT_MINION_URL"))
	minionRegistry.RegisterMinion("Rewrite", os.Getenv("REWRITE_MINION_URL"))
	minionRegistry.RegisterMinion("Brainstorming", os.Getenv("BRAINSTORMING_MINION_URL"))
	minionRegistry.RegisterMinion("Extraction, or Summarization", os.Getenv("EXTRACTION_MINION_URL"))
	minionRegistry.RegisterMinion("Other", os.Getenv("OTHER_MINION_URL"))

	return &ChatCompletionHandler{
		promptClassifierClient: services.NewPromptClassifierClient(),
		metrics:                sharedMetrics,
		providerConstraint:     "", // No provider constraint for main endpoint
		minionRegistry:         minionRegistry,
	}
}

func NewProviderChatCompletionHandler(provider string) *ChatCompletionHandler {
	return &ChatCompletionHandler{
		promptClassifierClient: services.NewPromptClassifierClient(),
		metrics:                sharedMetrics,
		providerConstraint:     provider,
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

	provider, err := providers.NewLLMProvider(modelInfo.Provider, nil, h.minionRegistry)
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

	modelInfo, cacheType, err := h.promptClassifierClient.SelectModelWithCache(selectReq, apiKey, requestID)
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
