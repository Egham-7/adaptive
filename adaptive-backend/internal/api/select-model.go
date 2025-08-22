package api

import (
	"adaptive-backend/internal/services/select_model"
	"fmt"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// SelectModelHandler handles model selection requests.
// It determines which model/provider would be selected for a given provider-agnostic request
// without actually executing the completion.
type SelectModelHandler struct {
	requestSvc     *select_model.RequestService
	selectModelSvc *select_model.Service
	responseSvc    *select_model.ResponseService
}

// NewSelectModelHandler initializes the select model handler with injected dependencies.
func NewSelectModelHandler(
	requestSvc *select_model.RequestService,
	selectModelSvc *select_model.Service,
	responseSvc *select_model.ResponseService,
) *SelectModelHandler {
	return &SelectModelHandler{
		requestSvc:     requestSvc,
		selectModelSvc: selectModelSvc,
		responseSvc:    responseSvc,
	}
}

// SelectModel handles the model selection HTTP request.
// It processes a provider-agnostic model selection request and returns the selected model/provider
// without actually executing the completion.
func (h *SelectModelHandler) SelectModel(c *fiber.Ctx) error {
	reqID := h.requestSvc.GetRequestID(c)
	userID := h.requestSvc.GetUserID(c)
	fiberlog.Infof("[%s] starting model selection request", reqID)

	// Parse request using specialized request service
	selectReq, err := h.requestSvc.ParseSelectModelRequest(c)
	if err != nil {
		return h.responseSvc.BadRequest(c, fmt.Sprintf("Invalid request body: %s", err.Error()))
	}

	// Validate request using specialized request service
	if err := h.requestSvc.ValidateSelectModelRequest(selectReq); err != nil {
		return h.responseSvc.BadRequest(c, err.Error())
	}

	// Perform model selection using the service
	resp, err := h.selectModelSvc.SelectModel(selectReq, userID, reqID)
	if err != nil {
		return h.responseSvc.InternalError(c, fmt.Sprintf("Model selection failed: %s", err.Error()))
	}

	return h.responseSvc.Success(c, resp)
}
