package count_tokens

import (
	"context"
	"fmt"
	"time"

	"adaptive-backend/internal/models"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"google.golang.org/genai"
)

// CountTokensService handles Gemini CountTokens API calls using the Gemini SDK
type CountTokensService struct{}

// NewCountTokensService creates a new CountTokensService
func NewCountTokensService() *CountTokensService {
	return &CountTokensService{}
}

// CreateClient creates a Gemini client with the given provider configuration
func (cts *CountTokensService) CreateClient(ctx context.Context, providerConfig models.ProviderConfig) (*genai.Client, error) {
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  providerConfig.APIKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}

	return client, nil
}

// SendRequest sends a count tokens request to Gemini
func (cts *CountTokensService) SendRequest(
	ctx context.Context,
	client *genai.Client,
	contents []*genai.Content,
	model string,
	requestID string,
) (*genai.CountTokensResponse, error) {
	fiberlog.Infof("[%s] Making Gemini CountTokens API request - model: %s", requestID, model)

	startTime := time.Now()
	resp, err := client.Models.CountTokens(ctx, model, contents, nil)
	duration := time.Since(startTime)

	if err != nil {
		fiberlog.Errorf("[%s] Gemini CountTokens API request failed after %v: %v", requestID, duration, err)
		return nil, models.NewProviderError("gemini", "count tokens request failed", err)
	}

	fiberlog.Infof("[%s] Gemini CountTokens API request completed successfully in %v - tokens: %d", requestID, duration, resp.TotalTokens)
	return resp, nil
}

// HandleGeminiCountTokensProvider handles count tokens requests using native Gemini client
func (cts *CountTokensService) HandleGeminiCountTokensProvider(
	c *fiber.Ctx,
	contents []*genai.Content,
	model string,
	providerConfig models.ProviderConfig,
	requestID string,
) (*genai.CountTokensResponse, error) {
	fiberlog.Debugf("[%s] Using native Gemini provider for count tokens request", requestID)

	client, err := cts.CreateClient(c.Context(), providerConfig)
	if err != nil {
		return nil, err
	}

	response, err := cts.SendRequest(c.Context(), client, contents, model, requestID)
	if err != nil {
		return nil, err
	}
	return response, nil
}
