package completions

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"adaptive-backend/internal/services/stream_readers/stream"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// ResponseService handles HTTP responses for all protocols
type ResponseService struct{}

// NewResponseService creates a new response service
func NewResponseService() *ResponseService {
	return &ResponseService{}
}

// HandleProtocol routes to the correct response flow based on protocol.
// remoteProv is the standard‐LLM provider, minionProv is used for minion or
// MinionS protocols. req is the original ChatCompletionRequest.
func (s *ResponseService) HandleProtocol(
	c *fiber.Ctx,
	resp *models.OrchestratorResponse,
	remoteProv provider_interfaces.LLMProvider,
	minionProv provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	requestID string,
	isStream bool,
) error {
	switch resp.Protocol {
	case models.ProtocolStandardLLM:
		return s.handleStandard(c, remoteProv, req, requestID, isStream)

	case models.ProtocolMinion:
		return s.handleMinion(c, minionProv, req, requestID, isStream)

	default:
		return s.HandleError(c, fiber.StatusInternalServerError,
			"unknown protocol "+string(resp.Protocol), requestID)
	}
}

// handleStandard handles both streaming and regular standard‐LLM flows.
func (s *ResponseService) handleStandard(
	c *fiber.Ctx,
	prov provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	requestID string,
	isStream bool,
) error {
	if isStream {
		fiberlog.Infof("[%s] streaming standard response", requestID)
		streamResp, err := prov.Chat().
			Completions().
			StreamCompletion(req.ToOpenAIParams())
		if err != nil {
			fiberlog.Errorf("[%s] stream failed: %v", requestID, err)
			return s.HandleError(c, fiber.StatusInternalServerError,
				"stream failed: "+err.Error(), requestID)
		}
		s.setStreamHeaders(c)
		return stream.HandleStream(c, streamResp, requestID)
	}
	fiberlog.Infof("[%s] generating standard completion", requestID)
	regResp, err := prov.Chat().
		Completions().
		CreateCompletion(req.ToOpenAIParams())
	if err != nil {
		fiberlog.Errorf("[%s] create failed: %v", requestID, err)
		return s.HandleError(c, fiber.StatusInternalServerError,
			"create failed: "+err.Error(), requestID)
	}
	return c.JSON(regResp)
}

// handleMinion handles both streaming and regular minion flows.
func (s *ResponseService) handleMinion(
	c *fiber.Ctx,
	prov provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	requestID string,
	isStream bool,
) error {
	if isStream {
		fiberlog.Infof("[%s] streaming minion response", requestID)
		streamResp, err := prov.Chat().
			Completions().
			StreamCompletion(req.ToOpenAIParams())
		if err != nil {
			fiberlog.Errorf("[%s] minion stream failed: %v", requestID, err)
			return s.HandleError(c, fiber.StatusInternalServerError,
				"minion stream failed: "+err.Error(), requestID)
		}
		s.setStreamHeaders(c)
		return stream.HandleStream(c, streamResp, requestID)
	}
	fiberlog.Infof("[%s] generating minion completion", requestID)
	regResp, err := prov.Chat().
		Completions().
		CreateCompletion(req.ToOpenAIParams())
	if err != nil {
		fiberlog.Errorf("[%s] minion create failed: %v", requestID, err)
		return s.HandleError(c, fiber.StatusInternalServerError,
			"minion create failed: "+err.Error(), requestID)
	}
	return c.JSON(regResp)
}

// HandleError sends a standardized error response
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

// HandleBadRequest handles 400 errors
func (s *ResponseService) HandleBadRequest(
	c *fiber.Ctx,
	message, requestID string,
) error {
	return s.HandleError(c, fiber.StatusBadRequest, message, requestID)
}

// HandleInternalError handles 500 errors
func (s *ResponseService) HandleInternalError(
	c *fiber.Ctx,
	message, requestID string,
) error {
	return s.HandleError(c, fiber.StatusInternalServerError, message, requestID)
}

// setStreamHeaders sets SSE headers
func (s *ResponseService) setStreamHeaders(c *fiber.Ctx) {
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Transfer-Encoding", "chunked")
	c.Set("Access-Control-Allow-Origin", "*")
	c.Set("Access-Control-Allow-Headers", "Cache-Control")
}
