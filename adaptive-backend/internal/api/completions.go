package api

import (
	"os"
	"time"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/chat/completions"
	"adaptive-backend/internal/services/metrics"
	"adaptive-backend/internal/services/minions"
	"adaptive-backend/internal/services/model_selection"
	"adaptive-backend/internal/services/providers/provider_interfaces"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// CompletionHandler handles chat completion requests with clean separation of concerns
type CompletionHandler struct {
	// Core services
	requestService       *completions.RequestService
	responseService      *completions.ResponseService
	parameterService     *completions.ParameterService
	orchestrationService *completions.OrchestrationService
	metricsService       *completions.MetricsService

	// Configuration
	providerConstraint string
}

// NewCompletionHandler creates a new completion handler with all required services
func NewCompletionHandler() *CompletionHandler {
	// Initialize minion registry
	minionRegistry := minions.NewMinionRegistry(11)
	registerMinions(minionRegistry)

	// Initialize model selector
	modelSelector, err := model_selection.NewModelSelector()
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

	return &CompletionHandler{
		requestService:       requestService,
		responseService:      responseService,
		parameterService:     parameterService,
		orchestrationService: orchestrationService,
		metricsService:       metricsService,
		providerConstraint:   "",
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

	// Orchestrate model selection and provider configuration
	orchestratorResult, err := h.orchestrationService.SelectAndConfigureProvider(req, apiKey, requestID)
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
		return h.handleStreamingResponse(c, orchestratorResult.Provider, orchestratorResult.ProviderName, req, start, requestID)
	}
	return h.handleRegularResponse(c, orchestratorResult.Provider, orchestratorResult.ProviderName, req, start, requestID)
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
) error {
	fiberlog.Infof("[%s] Processing streaming response", requestID)

	err := h.responseService.HandleStreamResponse(c, provider, req, requestID)
	if err != nil {
		h.metricsService.RecordError(start, "500", true, requestID, providerName)
		return err
	}

	h.metricsService.RecordSuccess(start, true, requestID, providerName)
	return nil
}

func (h *CompletionHandler) handleRegularResponse(
	c *fiber.Ctx,
	provider provider_interfaces.LLMProvider,
	providerName string,
	req *models.ChatCompletionRequest,
	start time.Time,
	requestID string,
) error {
	fiberlog.Infof("[%s] Processing regular response", requestID)

	err := h.responseService.HandleRegularResponse(c, provider, req, requestID)
	if err != nil {
		h.metricsService.RecordError(start, "500", false, requestID, providerName)
		return err
	}

	h.metricsService.RecordSuccess(start, false, requestID, providerName)
	return nil
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
