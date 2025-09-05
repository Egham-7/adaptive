package api

import (
	"fmt"

	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/select_model"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// SelectModelHandler handles model selection requests.
// It determines which model/provider would be selected for a given provider-agnostic request
// without actually executing the completion.
type SelectModelHandler struct {
	cfg             *config.Config
	requestSvc      *select_model.RequestService
	selectModelSvc  *select_model.Service
	responseSvc     *select_model.ResponseService
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker
}

// NewSelectModelHandler initializes the select model handler with injected dependencies.
func NewSelectModelHandler(
	cfg *config.Config,
	requestSvc *select_model.RequestService,
	selectModelSvc *select_model.Service,
	responseSvc *select_model.ResponseService,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
) *SelectModelHandler {
	return &SelectModelHandler{
		cfg:             cfg,
		requestSvc:      requestSvc,
		selectModelSvc:  selectModelSvc,
		responseSvc:     responseSvc,
		circuitBreakers: circuitBreakers,
	}
}

// SelectModel handles the model selection HTTP request.
// It processes a provider-agnostic model selection request and returns the selected model/provider
// without actually executing the completion.
func (h *SelectModelHandler) SelectModel(c *fiber.Ctx) error {
	reqID := h.requestSvc.GetRequestID(c)
	fiberlog.Infof("[%s] starting model selection request", reqID)

	// Parse request using specialized request service
	selectReq, err := h.requestSvc.ParseSelectModelRequest(c)
	if err != nil {
		return h.responseSvc.BadRequest(c, fmt.Sprintf("Invalid request body: %s", err.Error()))
	}

	// Extract user ID from request body (use "anonymous" if not provided)
	userID := "anonymous"
	if selectReq.User != nil && *selectReq.User != "" {
		userID = *selectReq.User
	}

	// Validate request using specialized request service
	if err := h.requestSvc.ValidateSelectModelRequest(selectReq); err != nil {
		return h.responseSvc.BadRequest(c, err.Error())
	}

	// Build model router config from select model request fields
	requestConfig := &models.ModelRouterConfig{
		Models: selectReq.Models,
	}

	// Set cost bias if provided
	if selectReq.CostBias != nil {
		requestConfig.CostBias = *selectReq.CostBias
	}

	// Resolve config by merging YAML config with request overrides
	mergedConfig := h.cfg.MergeModelRouterConfig(requestConfig)

	// Perform model selection using the service
	resp, err := h.selectModelSvc.SelectModel(c.UserContext(), selectReq, userID, reqID, h.circuitBreakers, mergedConfig)
	if err != nil {
		return h.responseSvc.InternalError(c, fmt.Sprintf("Model selection failed: %s", err.Error()))
	}

	return h.responseSvc.Success(c, resp)
}
