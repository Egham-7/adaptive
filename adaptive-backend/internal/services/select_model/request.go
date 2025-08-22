package select_model

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/request"

	"github.com/gofiber/fiber/v2"
)

// RequestService handles request parsing and validation for select model operations
// It embeds the base request service and specializes it for model selection
type RequestService struct {
	*request.BaseService
}

// NewRequestService creates a new request service for select model
func NewRequestService() *RequestService {
	return &RequestService{
		BaseService: request.NewBaseService(),
	}
}

// ParseSelectModelRequest parses and validates the select model request body
func (rs *RequestService) ParseSelectModelRequest(c *fiber.Ctx) (*models.SelectModelRequest, error) {
	var req models.SelectModelRequest
	if err := c.BodyParser(&req); err != nil {
		return nil, err
	}
	return &req, nil
}

// ValidateSelectModelRequest validates the parsed select model request
func (rs *RequestService) ValidateSelectModelRequest(req *models.SelectModelRequest) error {
	if len(req.Models) == 0 {
		return &ValidationError{Field: "models", Message: "Models slice cannot be empty"}
	}

	if req.Prompt == "" {
		return &ValidationError{Field: "prompt", Message: "Prompt cannot be empty"}
	}

	return nil
}

// ValidationError represents a request validation error
type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	return e.Message
}
