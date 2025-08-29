package completions

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/format_adapter"
	"adaptive-backend/internal/services/model_router"
	"adaptive-backend/internal/services/response"
	"adaptive-backend/internal/utils"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// ResponseService handles HTTP responses for chat completion operations
// It embeds the base response service and specializes it for completions
type ResponseService struct {
	*response.BaseService
	modelRouter *model_router.ModelRouter
}

func NewResponseService(modelRouter *model_router.ModelRouter) *ResponseService {
	if modelRouter == nil {
		panic("NewResponseService: modelRouter is nil")
	}

	return &ResponseService{
		BaseService: response.NewBaseService(),
		modelRouter: modelRouter,
	}
}

// BadRequest sends a bad request error response specific to completions
func (rs *ResponseService) BadRequest(c *fiber.Ctx, message string) error {
	return rs.Error(c, fiber.StatusBadRequest, message, "invalid_request_error", "bad_request")
}

// Unauthorized sends an unauthorized error response specific to completions
func (rs *ResponseService) Unauthorized(c *fiber.Ctx, message string) error {
	return rs.Error(c, fiber.StatusUnauthorized, message, "authentication_error", "unauthorized")
}

// RateLimited sends a rate limit error response specific to completions
func (rs *ResponseService) RateLimited(c *fiber.Ctx, message string) error {
	return rs.Error(c, fiber.StatusTooManyRequests, message, "rate_limit_error", "rate_limit_exceeded")
}

// InternalError sends an internal server error response specific to completions
func (rs *ResponseService) InternalError(c *fiber.Ctx, message string) error {
	return rs.Error(c, fiber.StatusInternalServerError, message, "internal_error", "completion_failed")
}

// HandleError sends a standardized error response
func (rs *ResponseService) HandleError(
	c *fiber.Ctx,
	statusCode int,
	message string,
	requestID string,
) error {
	fiberlog.Errorf("[%s] Error %d: %s", requestID, statusCode, message)
	// Map to standardized error codes used by BaseService
	var code, subcode string
	switch statusCode {
	case fiber.StatusBadRequest:
		code, subcode = "invalid_request_error", "bad_request"
	case fiber.StatusUnauthorized:
		code, subcode = "authentication_error", "unauthorized"
	case fiber.StatusTooManyRequests:
		code, subcode = "rate_limit_error", "rate_limit_exceeded"
	default:
		code, subcode = "internal_error", "completion_failed"
	}
	return rs.Error(c, statusCode, message, code, subcode)
}

// HandleBadRequest handles 400 errors
func (rs *ResponseService) HandleBadRequest(
	c *fiber.Ctx,
	message, requestID string,
) error {
	return rs.HandleError(c, fiber.StatusBadRequest, message, requestID)
}

// HandleInternalError handles 500 errors
func (rs *ResponseService) HandleInternalError(
	c *fiber.Ctx,
	message, requestID string,
) error {
	return rs.HandleError(c, fiber.StatusInternalServerError, message, requestID)
}

// SetStreamHeaders sets SSE headers
func (rs *ResponseService) SetStreamHeaders(c *fiber.Ctx) {
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Transfer-Encoding", "chunked")
	c.Set("Access-Control-Allow-Origin", "*")
	c.Set("Access-Control-Allow-Headers", "Cache-Control")
}

// StoreSuccessfulSemanticCache stores the protocol response in semantic cache after successful completion
func (rs *ResponseService) StoreSuccessfulSemanticCache(
	req *models.ChatCompletionRequest,
	resp *models.ProtocolResponse,
	requestID string,
) {
	if rs.modelRouter == nil {
		fiberlog.Debugf("[%s] Protocol manager not available for semantic cache storage", requestID)
		return
	}

	// Extract prompt for cache storage
	openAIParams, err := format_adapter.AdaptiveToOpenAI.ConvertRequest(req)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to convert request to OpenAI parameters for semantic cache: %v", requestID, err)
		return
	}

	// Extract prompt from messages
	prompt, err := utils.ExtractLastMessage(openAIParams.Messages)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to extract prompt for semantic cache: %v", requestID, err)
		return
	}

	// Store in semantic cache
	if err := rs.modelRouter.StoreSuccessfulProtocol(prompt, *resp, requestID, req.ModelRouterConfig); err != nil {
		fiberlog.Warnf("[%s] Failed to store successful response in semantic cache: %v", requestID, err)
	}
}
