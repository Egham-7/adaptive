package completions

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/protocol_manager"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

const (
	protocolStandard   = "standard"
	protocolMinion     = "minion"
	protocolMinions    = "minions_protocol"
	errUnknownProtocol = "unknown protocol"
)

// ResponseService handles HTTP responses for all protocols.
type ResponseService struct {
	completionService *CompletionService
	protocolMgr       *protocol_manager.ProtocolManager
}

// NewResponseService creates a new response service.
func NewResponseService(cfg *config.Config, protocolMgr *protocol_manager.ProtocolManager, fallbackService *FallbackService) *ResponseService {
	return &ResponseService{
		completionService: NewCompletionService(cfg, fallbackService),
		protocolMgr:       protocolMgr,
	}
}

// HandleProtocol routes to the correct response flow based on protocol.
func (s *ResponseService) HandleProtocol(
	c *fiber.Ctx,
	protocol models.ProtocolType,
	req *models.ChatCompletionRequest,
	resp *models.ProtocolResponse,
	requestID string,
	isStream bool,
	cacheSource string,
) error {
	if isStream {
		s.setStreamHeaders(c)
	}

	switch protocol {
	case models.ProtocolStandardLLM:
		if err := s.completionService.HandleStandardCompletion(c, req, resp.Standard, requestID, isStream, cacheSource); err != nil {
			return s.HandleError(c, fiber.StatusInternalServerError, err.Error(), requestID)
		}
		// Store successful response in semantic cache
		s.storeSuccessfulSemanticCache(req, resp, requestID)
		return nil

	case models.ProtocolMinion:
		if err := s.completionService.HandleMinionCompletion(c, req, resp.Minion, requestID, isStream, cacheSource); err != nil {
			return s.HandleError(c, fiber.StatusInternalServerError, err.Error(), requestID)
		}
		// Store successful response in semantic cache
		s.storeSuccessfulSemanticCache(req, resp, requestID)
		return nil

	case models.ProtocolMinionsProtocol:
		if err := s.completionService.HandleMinionsProtocolCompletion(c, req, resp, requestID, isStream, cacheSource); err != nil {
			return s.HandleError(c, fiber.StatusInternalServerError, err.Error(), requestID)
		}
		// Store successful response in semantic cache
		s.storeSuccessfulSemanticCache(req, resp, requestID)
		return nil

	default:
		return s.HandleError(c, fiber.StatusInternalServerError,
			errUnknownProtocol+" "+string(protocol), requestID)
	}
}

// HandleError sends a standardized error response.
func (s *ResponseService) HandleError(
	c *fiber.Ctx,
	statusCode int,
	message string,
	requestID string,
) error {
	fiberlog.Errorf("[%s] Error %d: %s", requestID, statusCode, message)
	return c.Status(statusCode).JSON(fiber.Map{
		"error": message,
	})
}

// HandleBadRequest handles 400 errors.
func (s *ResponseService) HandleBadRequest(
	c *fiber.Ctx,
	message, requestID string,
) error {
	return s.HandleError(c, fiber.StatusBadRequest, message, requestID)
}

// HandleInternalError handles 500 errors.
func (s *ResponseService) HandleInternalError(
	c *fiber.Ctx,
	message, requestID string,
) error {
	return s.HandleError(c, fiber.StatusInternalServerError, message, requestID)
}

// setStreamHeaders sets SSE headers.
func (s *ResponseService) setStreamHeaders(c *fiber.Ctx) {
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Transfer-Encoding", "chunked")
	c.Set("Access-Control-Allow-Origin", "*")
	c.Set("Access-Control-Allow-Headers", "Cache-Control")
}

// storeSuccessfulSemanticCache stores the protocol response in semantic cache after successful completion
func (s *ResponseService) storeSuccessfulSemanticCache(
	req *models.ChatCompletionRequest,
	resp *models.ProtocolResponse,
	requestID string,
) {
	if s.protocolMgr == nil {
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
	if err := s.protocolMgr.StoreSuccessfulProtocol(selReq, *resp, requestID, req.SemanticCache); err != nil {
		fiberlog.Warnf("[%s] Failed to store successful response in semantic cache: %v", requestID, err)
	}
}
