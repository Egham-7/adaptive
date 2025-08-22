package response

import (
	"github.com/gofiber/fiber/v2"
)

// BaseService provides common HTTP response utilities that can be embedded and specialized
type BaseService struct{}

// NewBaseService creates a new base response service
func NewBaseService() *BaseService {
	return &BaseService{}
}

// ErrorResponse represents a standard API error response
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail contains error information
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// Error sends an error response with specified status, type, and code
func (s *BaseService) Error(c *fiber.Ctx, status int, message, errorType, code string) error {
	return c.Status(status).JSON(ErrorResponse{
		Error: ErrorDetail{
			Message: message,
			Type:    errorType,
			Code:    code,
		},
	})
}

// Success sends a 200 OK response with the provided data
func (s *BaseService) Success(c *fiber.Ctx, data any) error {
	return c.JSON(data)
}
