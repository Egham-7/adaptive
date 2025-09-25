package generate

import (
	"adaptive-backend/internal/models"
	"context"
	"fmt"
	"iter"
	"time"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"google.golang.org/genai"
)

// GenerateService handles Gemini GenerateContent API calls using the Gemini SDK
type GenerateService struct{}

// NewGenerateService creates a new GenerateService
func NewGenerateService() *GenerateService {
	return &GenerateService{}
}

// CreateClient creates a Gemini client with the given provider configuration
func (gs *GenerateService) CreateClient(ctx context.Context, providerConfig models.ProviderConfig) (*genai.Client, error) {
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  providerConfig.APIKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}

	return client, nil
}

// SendRequest sends a non-streaming generate request to Gemini
func (gs *GenerateService) SendRequest(
	ctx context.Context,
	client *genai.Client,
	req *models.GeminiGenerateRequest,
	requestID string,
) (*genai.GenerateContentResponse, error) {
	fiberlog.Infof("[%s] Making non-streaming Gemini API request - model: %s", requestID, req.Model)

	startTime := time.Now()
	resp, err := client.Models.GenerateContent(ctx, req.Model, req.Contents, req.GenerationConfig)
	duration := time.Since(startTime)

	if err != nil {
		fiberlog.Errorf("[%s] Gemini API request failed after %v: %v", requestID, duration, err)
		return nil, models.NewProviderError("gemini", "generate request failed", err)
	}

	fiberlog.Infof("[%s] Gemini API request completed successfully in %v", requestID, duration)
	return resp, nil
}

// SendStreamingRequest sends a streaming generate request to Gemini
func (gs *GenerateService) SendStreamingRequest(
	ctx context.Context,
	client *genai.Client,
	req *models.GeminiGenerateRequest,
	requestID string,
) (iter.Seq2[*genai.GenerateContentResponse, error], error) {
	fiberlog.Infof("[%s] Making streaming Gemini API request - model: %s", requestID, req.Model)

	streamIter := client.Models.GenerateContentStream(ctx, req.Model, req.Contents, req.GenerationConfig)

	fiberlog.Debugf("[%s] Streaming request initiated successfully", requestID)
	return streamIter, nil
}

// HandleNonStreamingRequest handles non-streaming requests and returns concrete response
func (gs *GenerateService) HandleNonStreamingRequest(
	c *fiber.Ctx,
	req *models.GeminiGenerateRequest,
	provider string,
	providerConfig models.ProviderConfig,
	requestID string,
) (*genai.GenerateContentResponse, error) {
	// Check provider's native format to determine if conversion is needed
	if providerConfig.NativeFormat == "gemini" || providerConfig.NativeFormat == "" || provider == "gemini" {
		// Native Gemini format - use directly
		return gs.handleGeminiNonStreamingProvider(c, req, providerConfig, requestID)
	}

	// Provider uses different native format - not supported yet for Gemini endpoint
	return nil, fmt.Errorf("native format '%s' not supported for Gemini endpoint", providerConfig.NativeFormat)
}

// HandleStreamingRequest handles streaming requests and returns concrete iterator
func (gs *GenerateService) HandleStreamingRequest(
	c *fiber.Ctx,
	req *models.GeminiGenerateRequest,
	provider string,
	providerConfig models.ProviderConfig,
	requestID string,
) (iter.Seq2[*genai.GenerateContentResponse, error], error) {
	// Check provider's native format to determine if conversion is needed
	if providerConfig.NativeFormat == "gemini" || providerConfig.NativeFormat == "" || provider == "gemini" {
		// Native Gemini format - use directly
		return gs.handleGeminiStreamingProvider(c, req, providerConfig, requestID)
	}

	// Provider uses different native format - not supported yet for Gemini endpoint
	return nil, fmt.Errorf("native format '%s' not supported for Gemini endpoint", providerConfig.NativeFormat)
}

// handleGeminiNonStreamingProvider handles non-streaming requests using native Gemini client
func (gs *GenerateService) handleGeminiNonStreamingProvider(
	c *fiber.Ctx,
	req *models.GeminiGenerateRequest,
	providerConfig models.ProviderConfig,
	requestID string,
) (*genai.GenerateContentResponse, error) {
	fiberlog.Debugf("[%s] Using native Gemini provider for non-streaming request", requestID)

	client, err := gs.CreateClient(c.Context(), providerConfig)
	if err != nil {
		return nil, err
	}

	response, err := gs.SendRequest(c.Context(), client, req, requestID)
	if err != nil {
		return nil, err
	}
	return response, nil
}

// handleGeminiStreamingProvider handles streaming requests using native Gemini client
func (gs *GenerateService) handleGeminiStreamingProvider(
	c *fiber.Ctx,
	req *models.GeminiGenerateRequest,
	providerConfig models.ProviderConfig,
	requestID string,
) (iter.Seq2[*genai.GenerateContentResponse, error], error) {
	fiberlog.Debugf("[%s] Using native Gemini provider for streaming request", requestID)

	client, err := gs.CreateClient(c.Context(), providerConfig)
	if err != nil {
		return nil, err
	}

	streamIter, err := gs.SendStreamingRequest(c.Context(), client, req, requestID)
	if err != nil {
		return nil, err
	}
	return streamIter, nil
}
