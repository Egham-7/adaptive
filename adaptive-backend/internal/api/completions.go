package api

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/chat/completions"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/metrics"
	"adaptive-backend/internal/services/minions"
	"adaptive-backend/internal/services/model_selection"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// CompletionHandler handles chat completions with model‐selector, orchestration,
// minions, circuit breakers, metrics, and optional racing.
type CompletionHandler struct {
	requestService       *completions.RequestService
	responseService      *completions.ResponseService
	parameterService     *completions.ParameterService
	orchestrationService *completions.OrchestrationService
	metricsService       *completions.MetricsService
	raceService          *completions.RaceService

	circuitBreakers map[string]*circuitbreaker.CircuitBreaker
	cbMutex         sync.RWMutex
}

// NewCompletionHandler wires up all dependencies.
func NewCompletionHandler() *CompletionHandler {
	// Minions
	minionRegistry := minions.NewMinionRegistry(11)
	registerMinions(minionRegistry)

	// Model selector
	modelSelector, err := model_selection.NewModelSelector(0, 0)
	if err != nil {
		fiberlog.Fatalf("failed init modelSelector: %v", err)
	}

	// Metrics
	chatMetrics := metrics.NewChatMetrics()

	// Services
	reqSvc := completions.NewRequestService()
	respSvc := completions.NewResponseService()
	paramSvc := completions.NewParameterService()
	orchSvc := completions.NewOrchestrationService(modelSelector, minionRegistry)
	metSvc := completions.NewMetricsService(chatMetrics)
	raceSvc := completions.NewRaceService(minionRegistry)

	return &CompletionHandler{
		requestService:       reqSvc,
		responseService:      respSvc,
		parameterService:     paramSvc,
		orchestrationService: orchSvc,
		metricsService:       metSvc,
		raceService:          raceSvc,
		circuitBreakers:      make(map[string]*circuitbreaker.CircuitBreaker),
	}
}

// ChatCompletion is the main HTTP handler.
func (h *CompletionHandler) ChatCompletion(c *fiber.Ctx) error {
	start := time.Now()
	requestID := h.requestService.GetRequestID(c)
	userID := h.requestService.GetAPIKey(c) // using API key as userID

	fiberlog.Infof("[%s] ChatCompletion start", requestID)

	// parse & validate
	req, err := h.requestService.ParseChatCompletionRequest(c)
	if err != nil {
		h.metricsService.RecordError(start, "400", req.Stream, requestID, "unknown")
		return h.responseService.HandleBadRequest(c, "Invalid request: "+err.Error(), requestID)
	}
	isStream := req.Stream
	h.metricsService.RecordRequestStart(requestID, isStream)

	// copy breakers snapshot
	h.cbMutex.RLock()
	cbs := make(map[string]*circuitbreaker.CircuitBreaker)
	for k, v := range h.circuitBreakers {
		cbs[k] = v
	}
	h.cbMutex.RUnlock()

	// orchestrate selection + provider
	provider, resp, err := h.orchestrationService.SelectAndConfigureProvider(
		c.Context(), req, userID, requestID, cbs,
	)
	if err != nil {
		h.metricsService.RecordError(start, "500", isStream, requestID, "unknown")
		return h.responseService.HandleInternalError(c, err.Error(), requestID)
	}

	// determine providerName, modelName, alternatives, parameters
	var providerName, modelName string
	var alts []models.Alternative
	var autoParams models.OpenAIParameters

	switch resp.Protocol {
	case models.ProtocolStandardLLM:
		info := resp.Standard
		providerName = info.Provider
		modelName = info.Model
		alts = info.Alternatives
		autoParams = info.Parameters

	case models.ProtocolMinion:
		info := resp.Minion
		providerName = "minion"
		modelName = info.TaskType
		alts = info.Alternatives
		autoParams = info.Parameters

	default:
		return h.responseService.HandleInternalError(c,
			"unknown protocol "+string(resp.Protocol), requestID)
	}

	// metrics: model selection
	h.metricsService.RecordModelSelection(modelName, requestID, providerName)

	fiberlog.Infof("[%s] Protocol=%s Model=%s Provider=%s",
		requestID, resp.Protocol, modelName, providerName)

	// apply parameters
	if err := h.parameterService.ApplyModelParameters(req, autoParams, requestID); err != nil {
		h.metricsService.RecordError(start, "500", isStream, requestID, providerName)
		return h.responseService.HandleInternalError(c,
			"parameter apply failed: "+err.Error(), requestID)
	}

	// respond
	if isStream {
		return h.handleResponse(c, provider, providerName, req,
			start, requestID, alts, true)
	}
	return h.handleResponse(c, provider, providerName, req,
		start, requestID, alts, false)
}

// handleResponse attempts primary provider, then optional failover.
func (h *CompletionHandler) handleResponse(
	c *fiber.Ctx,
	provider provider_interfaces.LLMProvider,
	providerName string,
	req *models.ChatCompletionRequest,
	start time.Time,
	requestID string,
	alts []models.Alternative,
	isStreaming bool,
) error {
	cb := h.getOrCreateCircuitBreaker(providerName)

	// if circuit open, skip to alternatives
	if cb.GetState() == circuitbreaker.Open {
		if len(alts) > 0 {
			fiberlog.Warnf("[%s] %s circuit open, racing alternatives", requestID, providerName)
			return h.raceAndRespond(c, alts, req, start, requestID, isStreaming)
		}
		cb.RecordFailure()
		h.metricsService.RecordError(start, "503", isStreaming, requestID, providerName)
		return h.responseService.HandleInternalError(c,
			"provider circuit open", requestID)
	}

	// primary
	if !cb.CanExecute() {
		h.metricsService.RecordError(start, "503", isStreaming, requestID, providerName)
		return h.responseService.HandleInternalError(c,
			"provider unavailable", requestID)
	}

	var err error
	if isStreaming {
		err = h.responseService.HandleStreamResponse(c, provider, req, requestID)
	} else {
		err = h.responseService.HandleRegularResponse(c, provider, req, requestID)
	}

	if err != nil {
		cb.RecordFailure()
		h.metricsService.RecordError(start, "500", isStreaming, requestID, providerName)
		if len(alts) > 0 {
			fiberlog.Warnf("[%s] primary %s failed, racing %d alternatives",
				requestID, providerName, len(alts))
			return h.raceAndRespond(c, alts, req, start, requestID, isStreaming)
		}
		return err
	}

	cb.RecordSuccess()
	h.metricsService.RecordSuccess(start, isStreaming, requestID, providerName)
	return nil
}

// raceAndRespond races alternatives and sends first success.
func (h *CompletionHandler) raceAndRespond(
	c *fiber.Ctx,
	alts []models.Alternative,
	req *models.ChatCompletionRequest,
	start time.Time,
	requestID string,
	isStreaming bool,
) error {
	primary := alts[0]
	others := alts[1:]

	raceResult, err := h.raceService.RaceProviders(
		c.Context(), primary, others, requestID,
	)
	if err != nil {
		h.metricsService.RecordError(start, "500", isStreaming, requestID, "alternatives")
		return h.responseService.HandleInternalError(c,
			fmt.Sprintf("all providers failed: %v", err), requestID)
	}

	fiberlog.Infof("[%s] alternative %s won", requestID, raceResult.ProviderName)
	if isStreaming {
		return h.responseService.HandleStreamResponse(c, raceResult.Provider, req, requestID)
	}
	return h.responseService.HandleRegularResponse(c, raceResult.Provider, req, requestID)
}

// getOrCreateCircuitBreaker returns or creates a breaker for provider.
func (h *CompletionHandler) getOrCreateCircuitBreaker(
	providerName string,
) *circuitbreaker.CircuitBreaker {
	h.cbMutex.RLock()
	if cb, ok := h.circuitBreakers[providerName]; ok {
		h.cbMutex.RUnlock()
		return cb
	}
	h.cbMutex.RUnlock()

	h.cbMutex.Lock()
	defer h.cbMutex.Unlock()
	if cb, ok := h.circuitBreakers[providerName]; ok {
		return cb
	}
	cfg := circuitbreaker.Config{
		FailureThreshold: 3,
		SuccessThreshold: 2,
		Timeout:          10 * time.Second,
		ResetAfter:       1 * time.Minute,
	}
	cb := circuitbreaker.NewWithConfig(cfg)
	h.circuitBreakers[providerName] = cb
	return cb
}

// registerMinions configures all available minions in the registry.
func registerMinions(registry *minions.MinionRegistry) {
	registry.RegisterMinion("Open QA", getEnvOrDefault("OPEN_QA_MINION_URL", ""))
	registry.RegisterMinion("Closed QA", getEnvOrDefault("CLOSED_QA_MINION_URL", ""))
	registry.RegisterMinion("Summarization", getEnvOrDefault("SUMMARIZATION_MINION_URL", ""))
	registry.RegisterMinion("Text Generation", getEnvOrDefault("TEXT_GENERATION_MINION_URL", ""))
	registry.RegisterMinion("Classification", getEnvOrDefault("CLASSIFICATION_MINION_URL", ""))
	registry.RegisterMinion("Code Generation", getEnvOrDefault("CODE_GENERATION_MINION_URL", ""))
	registry.RegisterMinion("Chatbot", getEnvOrDefault("CHATBOT_MINION_URL", ""))
	registry.RegisterMinion("Rewrite", getEnvOrDefault("REWRITE_MINION_URL", ""))
	registry.RegisterMinion("Brainstorming", getEnvOrDefault("BRAINSTORMING_MINION_URL", ""))
	registry.RegisterMinion("Extraction", getEnvOrDefault("EXTRACTION_MINION_URL", ""))
	registry.RegisterMinion("Other", getEnvOrDefault("OTHER_MINION_URL", ""))
}

// getEnvOrDefault returns environment variable value or default if not set.
func getEnvOrDefault(key, defaultValue string) string {
	if value, exists := os.LookupEnv(key); exists && value != "" {
		return value
	}
	return defaultValue
}

// Health checks dependencies.
func (h *CompletionHandler) Health() error {
	return h.orchestrationService.ValidateOrchestrationContext()
}
