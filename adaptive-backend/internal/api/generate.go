package api

import (
	"fmt"

	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/fallback"
	"adaptive-backend/internal/services/gemini/generate"
	"adaptive-backend/internal/services/model_router"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// GenerateHandler handles Gemini GenerateContent API requests using dedicated Gemini services
type GenerateHandler struct {
	cfg             *config.Config
	requestSvc      *generate.RequestService
	generateSvc     *generate.GenerateService
	responseSvc     *generate.ResponseService
	modelRouter     *model_router.ModelRouter
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker
	fallbackService *fallback.FallbackService
}

// NewGenerateHandler creates a new GenerateHandler with Gemini-specific services
func NewGenerateHandler(
	cfg *config.Config,
	modelRouter *model_router.ModelRouter,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
) *GenerateHandler {
	return &GenerateHandler{
		cfg:             cfg,
		requestSvc:      generate.NewRequestService(),
		generateSvc:     generate.NewGenerateService(),
		responseSvc:     generate.NewResponseService(),
		modelRouter:     modelRouter,
		circuitBreakers: circuitBreakers,
		fallbackService: fallback.NewFallbackService(cfg),
	}
}

// Generate handles the Gemini GenerateContent API HTTP request (non-streaming)
func (h *GenerateHandler) Generate(c *fiber.Ctx) error {
	requestID := h.requestSvc.GetRequestID(c)
	fiberlog.Infof("[%s] Starting Gemini GenerateContent API request from %s", requestID, c.IP())

	return h.handleRequest(c, requestID, false)
}

// StreamGenerate handles the Gemini GenerateContent API HTTP request (streaming)
func (h *GenerateHandler) StreamGenerate(c *fiber.Ctx) error {
	requestID := h.requestSvc.GetRequestID(c)
	fiberlog.Infof("[%s] Starting Gemini StreamGenerateContent API request from %s", requestID, c.IP())

	return h.handleRequest(c, requestID, true)
}

// handleRequest processes both streaming and non-streaming requests
func (h *GenerateHandler) handleRequest(c *fiber.Ctx, requestID string, isStreaming bool) error {
	// Parse Gemini GenerateContent request
	req, err := h.requestSvc.ParseRequest(c)
	if err != nil {
		fiberlog.Warnf("[%s] Request parsing failed: %v", requestID, err)
		return h.responseSvc.HandleError(c, models.NewValidationError("invalid request", err), requestID)
	}
	fiberlog.Debugf("[%s] Request parsed successfully - model: %s", requestID, req.Model)

	// Resolve config by merging YAML config with request overrides
	resolvedConfig, err := h.cfg.ResolveConfigFromGeminiRequest(req)
	if err != nil {
		fiberlog.Errorf("[%s] Config resolution failed: %v", requestID, err)
		return h.responseSvc.HandleError(c, fmt.Errorf("failed to resolve config: %w", err), requestID)
	}
	fiberlog.Debugf("[%s] Configuration resolved successfully", requestID)

	fiberlog.Debugf("[%s] Request type: streaming=%t", requestID, isStreaming)

	// For now, use direct provider selection - TODO: integrate with model router
	provider := "gemini"
	providerConfig, exists := resolvedConfig.GetProviderConfig(provider, "generate")
	if !exists {
		fiberlog.Errorf("[%s] Provider %s not configured", requestID, provider)
		return h.responseSvc.HandleError(c, models.NewValidationError("provider not configured", nil), requestID)
	}

	// Check circuit breaker
	cb, exists := h.circuitBreakers[provider]
	if exists && cb.IsOpen() {
		fiberlog.Warnf("[%s] Circuit breaker is open for provider %s", requestID, provider)
		return h.responseSvc.HandleError(c, models.NewProviderError(provider, "circuit breaker open", nil), requestID)
	}

	// Execute the request
	cacheSource := "none" // TODO: implement caching
	err = h.generateSvc.HandleProviderRequest(c, req, provider, providerConfig, isStreaming, requestID, h.responseSvc, cacheSource)
	if err != nil {
		// Update circuit breaker on failure
		if cb != nil {
			cb.RecordFailure()
		}
		fiberlog.Errorf("[%s] Provider request failed: %v", requestID, err)
		return h.responseSvc.HandleError(c, err, requestID)
	}

	// Update circuit breaker on success
	if cb != nil {
		cb.RecordSuccess()
	}

	fiberlog.Infof("[%s] Gemini GenerateContent request completed successfully", requestID)
	return nil
}