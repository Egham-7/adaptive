package completions

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/minions"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"adaptive-backend/internal/services/stream_readers/stream"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go/shared"
)

const (
	protocolStandard   = "standard"
	protocolMinion     = "minion"
	protocolMinions    = "minions_protocol"
	errUnknownProtocol = "unknown protocol"
)

// ResponseService handles HTTP responses for all protocols.
type ResponseService struct{}

// NewResponseService creates a new response service.
func NewResponseService() *ResponseService {
	return &ResponseService{}
}

// HandleProtocol routes to the correct response flow based on protocol.
// remoteProv is the standard-LLM provider, minionProv is used for minion or
// MinionS protocols. req is the original ChatCompletionRequest.
func (s *ResponseService) HandleProtocol(
	c *fiber.Ctx,
	protocol models.ProtocolType,
	remoteProv *provider_interfaces.LLMProvider,
	minionProv *provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	resp *models.ProtocolResponse,
	requestID string,
	isStream bool,
) error {
	switch protocol {
	case models.ProtocolStandardLLM:
		if remoteProv == nil {
			return s.HandleError(c, fiber.StatusInternalServerError,
				"Failed to get remote provider: remoteProv is nil", requestID)
		}
		if resp.Standard != nil {
			req.Model = shared.ChatModel(resp.Standard.Model)
			fiberlog.Infof("[%s] Set standard model to: %s", requestID, resp.Standard.Model)
		}
		return s.handleStandard(c, *remoteProv, req, requestID, isStream)

	case models.ProtocolMinion:
		if minionProv == nil {
			return s.HandleError(c, fiber.StatusInternalServerError,
				"Failed to get minion provider: minionProv is nil", requestID)
		}
		// For HuggingFace providers, the model is embedded in the BaseURL, so don't set req.Model
		if resp.Minion != nil {
			fiberlog.Infof("[%s] Using minion model: %s with BaseURL: %s", requestID, resp.Minion.Model, resp.Minion.BaseURL)
		}
		return s.handleMinion(c, *minionProv, req, requestID, isStream)

	case models.ProtocolMinionsProtocol:
		if remoteProv == nil || minionProv == nil {
			msg := "Failed to get providers for MinionsProtocol: "
			if remoteProv == nil {
				msg += "remoteProv is nil; "
			}
			if minionProv == nil {
				msg += "minionProv is nil; "
			}
			return s.HandleError(c, fiber.StatusInternalServerError, msg, requestID)
		}
		return s.handleMinionsProtocol(
			c, *remoteProv, *minionProv, req, resp, requestID, isStream,
		)

	default:
		return s.HandleError(c, fiber.StatusInternalServerError,
			errUnknownProtocol+" "+string(protocol), requestID)
	}
}

// handleProtocolGeneric handles both streaming and regular flows for any protocol.
func (s *ResponseService) handleProtocolGeneric(
	c *fiber.Ctx,
	prov provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	requestID string,
	isStream bool,
	protocolName string,
) error {
	if isStream {
		fiberlog.Infof("[%s] streaming %s response", requestID, protocolName)
		streamResp, err := prov.Chat().
			Completions().
			StreamCompletion(c.Context(), req.ToOpenAIParams())
		if err != nil {
			fiberlog.Errorf("[%s] %s stream failed: %v", requestID, protocolName, err)
			return s.HandleError(c, fiber.StatusInternalServerError,
				protocolName+" stream failed: "+err.Error(), requestID)
		}
		s.setStreamHeaders(c)
		return stream.HandleStream(c, streamResp, requestID)
	}
	fiberlog.Infof("[%s] generating %s completion", requestID, protocolName)
	regResp, err := prov.Chat().
		Completions().
		CreateCompletion(c.Context(), req.ToOpenAIParams())
	if err != nil {
		fiberlog.Errorf("[%s] %s create failed: %v", requestID, protocolName, err)
		return s.HandleError(c, fiber.StatusInternalServerError,
			protocolName+" create failed: "+err.Error(), requestID)
	}
	return c.JSON(regResp)
}

// handleStandard handles both streaming and regular standard-LLM flows.
func (s *ResponseService) handleStandard(
	c *fiber.Ctx,
	prov provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	requestID string,
	isStream bool,
) error {
	return s.handleProtocolGeneric(c, prov, req, requestID, isStream, protocolStandard)
}

// handleMinion handles both streaming and regular minion flows.
func (s *ResponseService) handleMinion(
	c *fiber.Ctx,
	prov provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	requestID string,
	isStream bool,
) error {
	return s.handleProtocolGeneric(c, prov, req, requestID, isStream, protocolMinion)
}

// handleMinionsProtocol handles the MinionS protocol, both stream and non-stream.
func (s *ResponseService) handleMinionsProtocol(
	c *fiber.Ctx,
	remoteProv provider_interfaces.LLMProvider,
	minionProv provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	resp *models.ProtocolResponse,
	requestID string,
	isStream bool,
) error {
	orchestrator := minions.NewMinionsOrchestrationService()

	// Set the remote model for remote provider calls
	if resp.Standard != nil {
		req.Model = shared.ChatModel(resp.Standard.Model)
		fiberlog.Infof("[%s] Set remote model to: %s", requestID, resp.Standard.Model)
	}

	// For HuggingFace minions, don't pass model name since it's embedded in BaseURL
	if resp.Minion != nil {
		fiberlog.Infof("[%s] Using minion model: %s with BaseURL: %s", requestID, resp.Minion.Model, resp.Minion.BaseURL)
	}

	if isStream {
		fiberlog.Infof("[%s] streaming MinionS response", requestID)
		s.setStreamHeaders(c)
		streamResp, err := orchestrator.OrchestrateMinionSStream(
			c.Context(), remoteProv, minionProv, req, "",
		)
		if err != nil {
			fiberlog.Errorf("[%s] MinionS stream failed: %v", requestID, err)
			return s.HandleError(c, fiber.StatusInternalServerError,
				"MinionS streaming failed: "+err.Error(), requestID)
		}
		return stream.HandleStream(c, streamResp, requestID)
	}

	fiberlog.Infof("[%s] generating MinionS completion", requestID)
	result, err := orchestrator.OrchestrateMinionS(
		c.Context(), remoteProv, minionProv, req, "",
	)
	if err != nil {
		fiberlog.Errorf("[%s] MinionS create failed: %v", requestID, err)
		return s.HandleError(c, fiber.StatusInternalServerError,
			"MinionS protocol failed: "+err.Error(), requestID)
	}
	return c.JSON(result)
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
