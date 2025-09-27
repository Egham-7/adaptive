package generate

import (
	"context"
	"iter"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/model_router"
	"adaptive-backend/internal/services/stream/handlers"
	"adaptive-backend/internal/utils"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"google.golang.org/genai"
)

// ResponseService handles Gemini response processing
type ResponseService struct {
	modelRouter *model_router.ModelRouter
}

// NewResponseService creates a new ResponseService
func NewResponseService(modelRouter *model_router.ModelRouter) *ResponseService {
	return &ResponseService{
		modelRouter: modelRouter,
	}
}

// HandleNonStreamingResponse processes a non-streaming Gemini response
func (rs *ResponseService) HandleNonStreamingResponse(
	c *fiber.Ctx,
	response *genai.GenerateContentResponse,
	requestID string,
	cacheSource string,
) error {
	fiberlog.Debugf("[%s] Processing non-streaming Gemini response", requestID)

	// Convert to our adaptive response format
	adaptiveResp := &models.GeminiGenerateContentResponse{
		Candidates:     response.Candidates,
		CreateTime:     response.CreateTime,
		ModelVersion:   response.ModelVersion,
		PromptFeedback: response.PromptFeedback,
		ResponseID:     response.ResponseID,
		UsageMetadata:  response.UsageMetadata,
		Provider:       "gemini",
	}

	// Add cache source metadata if available
	if cacheSource != "" {
		fiberlog.Infof("[%s] Response served from cache: %s", requestID, cacheSource)
	}

	fiberlog.Infof("[%s] Non-streaming response processed successfully", requestID)
	return c.JSON(adaptiveResp)
}

// HandleStreamingResponse processes a streaming Gemini response using the streaming pipeline
func (rs *ResponseService) HandleStreamingResponse(
	c *fiber.Ctx,
	streamIter iter.Seq2[*genai.GenerateContentResponse, error],
	requestID string,
	provider string,
	cacheSource string,
) error {
	fiberlog.Infof("[%s] Starting streaming response processing", requestID)

	// Use the proper Gemini streaming handler from the stream package
	return handlers.HandleGemini(c, streamIter, requestID, provider, cacheSource)
}

// HandleError processes and returns error responses
func (rs *ResponseService) HandleError(c *fiber.Ctx, err error, requestID string) error {
	fiberlog.Errorf("[%s] Handling error: %v", requestID, err)

	var appErr *models.AppError
	if e, ok := err.(*models.AppError); ok {
		appErr = e
	} else {
		appErr = models.NewInternalError("internal server error", err)
	}

	errorResponse := map[string]any{
		"error": map[string]any{
			"message":    appErr.Message,
			"type":       string(appErr.Type),
			"code":       appErr.StatusCode,
			"request_id": requestID,
		},
	}

	return c.Status(appErr.StatusCode).JSON(errorResponse)
}

// StoreSuccessfulSemanticCache stores the model response in semantic cache after successful completion
func (rs *ResponseService) StoreSuccessfulSemanticCache(
	ctx context.Context,
	req *models.GeminiGenerateRequest,
	resp *models.ModelSelectionResponse,
	requestID string,
) {
	if rs.modelRouter == nil {
		fiberlog.Debugf("[%s] Model router not available for semantic cache storage", requestID)
		return
	}

	// Extract prompt for cache storage from Gemini contents
	prompt, err := utils.ExtractPromptFromGeminiContents(req.Contents)
	if err != nil {
		fiberlog.Errorf("[%s] Failed to extract prompt for semantic cache: %v", requestID, err)
		return
	}

	// Store in semantic cache
	if err := rs.modelRouter.StoreSuccessfulModel(ctx, prompt, *resp, requestID, nil); err != nil {
		fiberlog.Warnf("[%s] Failed to store successful response in semantic cache: %v", requestID, err)
	}

	fiberlog.Debugf("[%s] Successfully stored model response in semantic cache - provider: %s, model: %s",
		requestID, resp.Provider, resp.Model)
}
