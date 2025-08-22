package completions

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/protocol_manager"
	"adaptive-backend/internal/services/response"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

const (
	protocolStandard   = "standard"
	protocolMinion     = "minion"
	protocolMinions    = "minions_protocol"
	errUnknownProtocol = "unknown protocol"
)

// ResponseService handles HTTP responses for chat completion operations
// It embeds the base response service and specializes it for completions
type ResponseService struct {
	*response.BaseService
	completionService *CompletionService
	protocolMgr       *protocol_manager.ProtocolManager
}

// NewResponseService creates a new response service for completions
func NewResponseService(cfg *config.Config, protocolMgr *protocol_manager.ProtocolManager, fallbackService *FallbackService) *ResponseService {
	if cfg == nil {
		panic("NewResponseService: cfg is nil")
	}
	if protocolMgr == nil {
		panic("NewResponseService: protocolMgr is nil")
	}
	if fallbackService == nil {
		panic("NewResponseService: fallbackService is nil")
	}

	return &ResponseService{
		BaseService:       response.NewBaseService(),
		completionService: NewCompletionService(cfg, fallbackService),
		protocolMgr:       protocolMgr,
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

// HandleProtocol routes to the correct response flow based on protocol
func (rs *ResponseService) HandleProtocol(
	c *fiber.Ctx,
	protocol models.ProtocolType,
	req *models.ChatCompletionRequest,
	resp *models.ProtocolResponse,
	requestID string,
	isStream bool,
	cacheSource string,
) error {
	if isStream {
		rs.setStreamHeaders(c)
	}

	switch protocol {
	case models.ProtocolStandardLLM:
		if err := rs.completionService.HandleStandardCompletion(c, req, resp.Standard, requestID, isStream, cacheSource); err != nil {
			return rs.HandleError(c, fiber.StatusInternalServerError, err.Error(), requestID)
		}
		// Store successful response in semantic cache
		rs.storeSuccessfulSemanticCache(req, resp, requestID)
		return nil

	case models.ProtocolMinion:
		if err := rs.completionService.HandleMinionCompletion(c, req, resp.Minion, requestID, isStream, cacheSource); err != nil {
			return rs.HandleError(c, fiber.StatusInternalServerError, err.Error(), requestID)
		}
		// Store successful response in semantic cache
		rs.storeSuccessfulSemanticCache(req, resp, requestID)
		return nil

	case models.ProtocolMinionsProtocol:
		if err := rs.completionService.HandleMinionsProtocolCompletion(c, req, resp, requestID, isStream, cacheSource); err != nil {
			return rs.HandleError(c, fiber.StatusInternalServerError, err.Error(), requestID)
		}
		// Store successful response in semantic cache
		rs.storeSuccessfulSemanticCache(req, resp, requestID)
		return nil

	default:
		return rs.HandleError(c, fiber.StatusInternalServerError,
			errUnknownProtocol+" "+string(protocol), requestID)
	}
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

// setStreamHeaders sets SSE headers
func (rs *ResponseService) setStreamHeaders(c *fiber.Ctx) {
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Transfer-Encoding", "chunked")
	c.Set("Access-Control-Allow-Origin", "*")
	c.Set("Access-Control-Allow-Headers", "Cache-Control")
}

// storeSuccessfulSemanticCache stores the protocol response in semantic cache after successful completion
func (rs *ResponseService) storeSuccessfulSemanticCache(
	req *models.ChatCompletionRequest,
	resp *models.ProtocolResponse,
	requestID string,
) {
	if rs.protocolMgr == nil {
		fiberlog.Debugf("[%s] Protocol manager not available for semantic cache storage", requestID)
		return
	}

	// Create ModelSelectionRequest for cache storage
	openAIParams := req.ToOpenAIParams()
	if openAIParams == nil {
		fiberlog.Errorf("[%s] Failed to convert request to OpenAI parameters for semantic cache", requestID)
		return
	}

	selReq := models.ModelSelectionRequest{
		ChatCompletionRequest: *openAIParams,
		ProtocolManagerConfig: req.ProtocolManagerConfig,
	}

	// Store in semantic cache
	if err := rs.protocolMgr.StoreSuccessfulProtocol(selReq, *resp, requestID, req.SemanticCache); err != nil {
		fiberlog.Warnf("[%s] Failed to store successful response in semantic cache: %v", requestID, err)
	}
}
