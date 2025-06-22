package api

import (
	"fmt"
	"maps"
	"os"
	"sync"
	"time"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/chat/completions"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/metrics"
	"adaptive-backend/internal/services/minions"
	"adaptive-backend/internal/services/model_selection"
	"adaptive-backend/internal/services/providers/provider_interfaces"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// CompletionHandler handles chat completion requests with clean separation of concerns
// Features automatic failover racing - tries primary provider first, only races alternatives if primary fails
type CompletionHandler struct {
	// Core services
	requestService       *completions.RequestService
	responseService      *completions.ResponseService
	parameterService     *completions.ParameterService
	orchestrationService *completions.OrchestrationService
	metricsService       *completions.MetricsService
	raceService          *completions.RaceService // Used for automatic failover when primary provider fails

	// Circuit breakers for provider health tracking
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker
	cbMutex         sync.RWMutex
}

// NewCompletionHandler creates a new completion handler with all required services
func NewCompletionHandler() *CompletionHandler {
	// Initialize minion registry
	minionRegistry := minions.NewMinionRegistry(11)
	registerMinions(minionRegistry)

	// Initialize model selector
	modelSelector, err := model_selection.NewModelSelector(0, 0, 0)
	if err != nil {
		fiberlog.Fatalf("failed to initialize ModelSelector: %v", err)
	}

	// Initialize metrics
	chatMetrics := metrics.NewChatMetrics()

	// Initialize services
	requestService := completions.NewRequestService()
	responseService := completions.NewResponseService()
	parameterService := completions.NewParameterService()
	orchestrationService := completions.NewOrchestrationService(modelSelector, minionRegistry)
	metricsService := completions.NewMetricsService(chatMetrics)
	raceService := completions.NewRaceService(minionRegistry)

	return &CompletionHandler{
		requestService:       requestService,
		responseService:      responseService,
		parameterService:     parameterService,
		orchestrationService: orchestrationService,
		metricsService:       metricsService,
		raceService:          raceService,
		circuitBreakers:      make(map[string]*circuitbreaker.CircuitBreaker),
	}
}

// ChatCompletion handles the main chat completion endpoint
func (h *CompletionHandler) ChatCompletion(c *fiber.Ctx) error {
	start := time.Now()
	requestID := h.requestService.GetRequestID(c)
	apiKey := h.requestService.GetAPIKey(c)

	fiberlog.Infof("[%s] Starting chat completion request", requestID)

	// Parse and validate request
	req, err := h.requestService.ParseChatCompletionRequest(c)
	if err != nil {
		h.metricsService.RecordError(start, "400", req != nil && req.Stream, requestID, "unknown")
		return h.responseService.HandleBadRequest(c, "Invalid request: "+err.Error(), requestID)
	}

	isStream := req.Stream
	h.metricsService.RecordRequestStart(requestID, isStream)

	// Orchestrate model selection and provider configuration with automatic racing
	ctx := c.Context()

	// Pass circuit breaker information for health-aware model selection
	h.cbMutex.RLock()
	circuitBreakers := make(map[string]*circuitbreaker.CircuitBreaker)
	maps.Copy(circuitBreakers, h.circuitBreakers)
	h.cbMutex.RUnlock()

	orchestratorResult, err := h.orchestrationService.SelectAndConfigureProvider(ctx, req, apiKey, requestID, circuitBreakers)
	if err != nil {
		h.metricsService.RecordError(start, "500", isStream, requestID, "unknown")
		return h.responseService.HandleInternalError(c, err.Error(), requestID)
	}

	// Record cache metrics if applicable
	if orchestratorResult.CacheType == "user" || orchestratorResult.CacheType == "global" {
		h.metricsService.RecordCacheHit(orchestratorResult.CacheType, requestID, orchestratorResult.ProviderName)
	}

	// Record model selection
	h.metricsService.RecordModelSelection(orchestratorResult.ModelName, requestID, orchestratorResult.ProviderName)

	// Log protocol and model information
	fiberlog.Infof("[%s] Protocol: %s, Model: %s",
		requestID, orchestratorResult.ProtocolType, orchestratorResult.ModelName)

	// Apply parameters based on protocol type
	if err := h.applyParametersFromOrchestrator(req, orchestratorResult.Parameters, requestID); err != nil {
		h.metricsService.RecordError(start, "500", isStream, requestID, orchestratorResult.ProviderName)
		return h.responseService.HandleInternalError(c, "Parameter application failed: "+err.Error(), requestID)
	}

	// Handle response based on streaming preference
	if isStream {
		return h.handleStreamingResponse(c, orchestratorResult.Provider, orchestratorResult.ProviderName, req, start, requestID, orchestratorResult)
	}
	return h.handleRegularResponse(c, orchestratorResult.Provider, orchestratorResult.ProviderName, req, start, requestID, orchestratorResult)
}

// applyParametersFromOrchestrator applies parameters from the orchestrator result
func (h *CompletionHandler) applyParametersFromOrchestrator(
	req *models.ChatCompletionRequest,
	autoParameters models.OpenAIParameters,
	requestID string,
) error {
	return h.parameterService.ApplyModelParameters(req, autoParameters, requestID)
}

func (h *CompletionHandler) handleStreamingResponse(
	c *fiber.Ctx,
	provider provider_interfaces.LLMProvider,
	providerName string,
	req *models.ChatCompletionRequest,
	start time.Time,
	requestID string,
	orchestratorResult *models.OrchestratorResult,
) error {
	return h.handleResponse(c, provider, providerName, req, start, requestID, orchestratorResult, true)
}

func (h *CompletionHandler) handleRegularResponse(
	c *fiber.Ctx,
	provider provider_interfaces.LLMProvider,
	providerName string,
	req *models.ChatCompletionRequest,
	start time.Time,
	requestID string,
	orchestratorResult *models.OrchestratorResult,
) error {
	return h.handleResponse(c, provider, providerName, req, start, requestID, orchestratorResult, false)
}

// handleResponse unified handler for both streaming and regular responses
func (h *CompletionHandler) handleResponse(
	c *fiber.Ctx,
	provider provider_interfaces.LLMProvider,
	providerName string,
	req *models.ChatCompletionRequest,
	start time.Time,
	requestID string,
	orchestratorResult *models.OrchestratorResult,
	isStreaming bool,
) error {
	fiberlog.Infof("[%s] Processing %s response with provider %s", requestID,
		map[bool]string{true: "streaming", false: "regular"}[isStreaming], providerName)

	cb := h.getOrCreateCircuitBreaker(providerName)

	// Circuit breaker is open - skip to alternatives immediately
	if cb.GetState() == circuitbreaker.Open {
		if len(orchestratorResult.Alternatives) > 0 {
			fiberlog.Infof("[%s] Provider %s circuit is OPEN, trying alternatives", requestID, providerName)
			return h.tryAlternatives(c, orchestratorResult, req, start, requestID, isStreaming)
		}
		cb.RecordFailure()
		h.metricsService.RecordError(start, "503", isStreaming, requestID, providerName)
		return h.responseService.HandleInternalError(c, "Provider circuit breaker is open and no alternatives available", requestID)
	}

	// Circuit breaker allows execution - try primary provider
	if !cb.CanExecute() {
		h.metricsService.RecordError(start, "503", isStreaming, requestID, providerName)
		return h.responseService.HandleInternalError(c, "Provider circuit breaker rejected request", requestID)
	}

	// Execute primary provider
	var err error
	if isStreaming {
		err = h.responseService.HandleStreamResponse(c, provider, req, requestID)
	} else {
		err = h.responseService.HandleRegularResponse(c, provider, req, requestID)
	}

	if err != nil {
		cb.RecordFailure()
		// Try alternatives if available
		if len(orchestratorResult.Alternatives) > 0 {
			fiberlog.Warnf("[%s] Primary provider %s failed, trying %d alternatives: %v",
				requestID, providerName, len(orchestratorResult.Alternatives), err)
			return h.tryAlternatives(c, orchestratorResult, req, start, requestID, isStreaming)
		}
		h.metricsService.RecordError(start, "500", isStreaming, requestID, providerName)
		return err
	}

	cb.RecordSuccess()
	h.metricsService.RecordSuccess(start, isStreaming, requestID, providerName)
	return nil
}

// tryAlternatives unified method for trying alternatives for both streaming and regular responses
func (h *CompletionHandler) tryAlternatives(
	c *fiber.Ctx,
	orchestratorResult *models.OrchestratorResult,
	req *models.ChatCompletionRequest,
	start time.Time,
	requestID string,
	isStreaming bool,
) error {
	// Record primary provider failure
	primaryCB := h.getOrCreateCircuitBreaker(orchestratorResult.ProviderName)
	primaryCB.RecordFailure()

	// Create primary alternative for racing
	primary := h.createPrimaryAlternative(orchestratorResult)

	// Race for fastest available provider
	ctx := c.Context()
	raceResult, err := h.raceService.RaceProviders(ctx, primary, orchestratorResult.Alternatives, requestID)
	if err != nil {
		h.metricsService.RecordError(start, "500", isStreaming, requestID, "all_alternatives")
		return h.responseService.HandleInternalError(c, fmt.Sprintf("all providers failed: %v", err), requestID)
	}

	fiberlog.Infof("[%s] Alternative provider %s won the race", requestID, raceResult.ProviderName)

	// Create new orchestrator result with no alternatives to prevent infinite recursion
	fallbackResult := &models.OrchestratorResult{
		Provider:     raceResult.Provider,
		ProviderName: raceResult.ProviderName,
		Alternatives: nil,
	}
	return h.handleResponse(c, raceResult.Provider, raceResult.ProviderName, req, start, requestID, fallbackResult, isStreaming)
}

// createPrimaryAlternative creates the primary alternative for racing
func (h *CompletionHandler) createPrimaryAlternative(orchestratorResult *models.OrchestratorResult) models.Alternative {
	if orchestratorResult.ProtocolType == "Minion" {
		return models.Alternative{
			Provider: "minion",
			Model:    orchestratorResult.TaskType,
		}
	}
	return models.Alternative{
		Provider: orchestratorResult.ProviderName,
		Model:    orchestratorResult.ModelName,
	}
}

// registerMinions configures all available minions in the registry
func registerMinions(registry *minions.MinionRegistry) {
	registry.RegisterMinion("Open QA", getEnvOrDefault("OPEN_QA_MINION_URL", ""))
	registry.RegisterMinion("Closed QA", getEnvOrDefault("CLOSED_QA_MINION_URL", ""))
	registry.RegisterMinion("Summarization", getEnvOrDefault("OPEN_QA_MINION_URL", ""))
	registry.RegisterMinion("Text Generation", getEnvOrDefault("OPEN_QA_MINION_URL", ""))
	registry.RegisterMinion("Classification", getEnvOrDefault("CLASSIFICATION_MINION_URL", ""))
	registry.RegisterMinion("Code Generation", getEnvOrDefault("CODE_GENERATION_MINION_URL", ""))
	registry.RegisterMinion("Chatbot", getEnvOrDefault("CHATBOT_MINION_URL", ""))
	registry.RegisterMinion("Rewrite", getEnvOrDefault("REWRITE_MINION_URL", ""))
	registry.RegisterMinion("Brainstorming", getEnvOrDefault("BRAINSTORMING_MINION_URL", ""))
	registry.RegisterMinion("Extraction", getEnvOrDefault("EXTRACTION_MINION_URL", ""))
	registry.RegisterMinion("Other", getEnvOrDefault("CHATBOT_MINION_URL", ""))
}

// getOrCreateCircuitBreaker returns or creates a circuit breaker for the given provider
func (h *CompletionHandler) getOrCreateCircuitBreaker(providerName string) *circuitbreaker.CircuitBreaker {
	h.cbMutex.RLock()
	if cb, exists := h.circuitBreakers[providerName]; exists {
		h.cbMutex.RUnlock()
		return cb
	}
	h.cbMutex.RUnlock()

	h.cbMutex.Lock()
	defer h.cbMutex.Unlock()

	// Double-check after acquiring write lock
	if cb, exists := h.circuitBreakers[providerName]; exists {
		return cb
	}

	// Create provider-specific circuit breaker with aggressive settings for API providers
	config := circuitbreaker.Config{
		FailureThreshold: 3,                // Fail after 3 consecutive failures
		SuccessThreshold: 2,                // Recover after 2 consecutive successes
		Timeout:          10 * time.Second, // Shorter timeout for API calls
		ResetAfter:       1 * time.Minute,  // Reset failure count after 1 minute
	}

	cb := circuitbreaker.NewWithConfig(config)
	h.circuitBreakers[providerName] = cb
	return cb
}

// getEnvOrDefault returns environment variable value or default if not set
func getEnvOrDefault(key, defaultValue string) string {
	if value, exists := os.LookupEnv(key); exists && value != "" {
		return value
	}
	return defaultValue
}

// GetMetrics returns the metrics service for external access (e.g., health checks)
func (h *CompletionHandler) GetMetrics() *completions.MetricsService {
	return h.metricsService
}

// Health performs a health check on all services
func (h *CompletionHandler) Health() error {
	if err := h.orchestrationService.ValidateOrchestrationContext(); err != nil {
		return err
	}
	// Add other health checks as needed
	return nil
}

// GetCircuitBreakerStatus returns the status of all circuit breakers
func (h *CompletionHandler) GetCircuitBreakerStatus() map[string]map[string]any {
	h.cbMutex.RLock()
	defer h.cbMutex.RUnlock()

	status := make(map[string]map[string]any)
	for providerName, cb := range h.circuitBreakers {
		metrics := cb.GetMetrics()
		state := cb.GetState()
		successRate := cb.GetSuccessRate()

		status[providerName] = map[string]any{
			"state":               h.stateToString(state),
			"total_requests":      metrics.TotalRequests,
			"successful_requests": metrics.SuccessfulRequests,
			"failed_requests":     metrics.FailedRequests,
			"circuit_opens":       metrics.CircuitOpens,
			"circuit_closes":      metrics.CircuitCloses,
			"success_rate":        successRate,
		}
	}
	return status
}

// stateToString converts circuit breaker state to string
func (h *CompletionHandler) stateToString(state circuitbreaker.State) string {
	switch state {
	case circuitbreaker.Closed:
		return "closed"
	case circuitbreaker.Open:
		return "open"
	case circuitbreaker.HalfOpen:
		return "half_open"
	default:
		return "unknown"
	}
}
