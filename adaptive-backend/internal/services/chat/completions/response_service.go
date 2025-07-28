package completions

import (
	"adaptive-backend/internal/models"

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
}

// NewResponseService creates a new response service.
func NewResponseService() *ResponseService {
	return &ResponseService{
		completionService: NewCompletionService(),
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
) error {
	if isStream {
		s.setStreamHeaders(c)
	}

	switch protocol {
	case models.ProtocolStandardLLM:
		if err := s.completionService.HandleStandardCompletion(c, req, resp.Standard, requestID, isStream); err != nil {
			return s.HandleError(c, fiber.StatusInternalServerError, err.Error(), requestID)
		}
		return nil

	case models.ProtocolMinion:
		if err := s.completionService.HandleMinionCompletion(c, req, resp.Minion, requestID, isStream); err != nil {
			return s.HandleError(c, fiber.StatusInternalServerError, err.Error(), requestID)
		}
		return nil

	case models.ProtocolMinionsProtocol:
		if err := s.completionService.HandleMinionsProtocolCompletion(c, req, resp, requestID, isStream); err != nil {
			return s.HandleError(c, fiber.StatusInternalServerError, err.Error(), requestID)
		}
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
