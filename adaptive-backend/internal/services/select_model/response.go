package select_model

import (
	"adaptive-backend/internal/services/response"

	"github.com/gofiber/fiber/v2"
)

// ResponseService handles HTTP responses for select model operations
// It embeds the base response service and specializes it for model selection
type ResponseService struct {
	*response.BaseService
}

// NewResponseService creates a new response service
func NewResponseService() *ResponseService {
	return &ResponseService{
		BaseService: response.NewBaseService(),
	}
}

// BadRequest sends a bad request error response specific to model selection
func (rs *ResponseService) BadRequest(c *fiber.Ctx, message string) error {
	return rs.Error(c, fiber.StatusBadRequest, message, "invalid_request_error", "bad_request")
}

// InternalError sends an internal server error response specific to model selection
func (rs *ResponseService) InternalError(c *fiber.Ctx, message string) error {
	return rs.Error(c, fiber.StatusInternalServerError, message, "internal_error", "model_selection_failed")
}