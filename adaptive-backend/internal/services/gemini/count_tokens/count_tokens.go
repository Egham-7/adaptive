package count_tokens

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"time"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/utils/clientcache"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"google.golang.org/genai"
)

// CountTokensService handles Gemini CountTokens API calls using the Gemini SDK
type CountTokensService struct {
	clientCache *clientcache.Cache[*genai.Client]
}

// NewCountTokensService creates a new CountTokensService
func NewCountTokensService() *CountTokensService {
	return &CountTokensService{
		clientCache: clientcache.NewCache[*genai.Client](),
	}
}

// generateConfigHash creates a hash of the provider config to detect changes
func (cts *CountTokensService) generateConfigHash(providerConfig models.ProviderConfig) (string, error) {
	type configForHash struct {
		BaseURL    string
		APIKeyHash string
	}

	apiKeyHash := sha256.Sum256([]byte(providerConfig.APIKey))
	hashConfig := configForHash{
		BaseURL:    providerConfig.BaseURL,
		APIKeyHash: fmt.Sprintf("%x", apiKeyHash[:8]),
	}

	configJSON, err := json.Marshal(hashConfig)
	if err != nil {
		return "", err
	}

	hash := sha256.Sum256(configJSON)
	return fmt.Sprintf("%x", hash[:16]), nil
}

// CreateClient creates or retrieves a cached Gemini client
func (cts *CountTokensService) CreateClient(ctx context.Context, providerConfig models.ProviderConfig) (*genai.Client, error) {
	// Generate cache key based on provider config hash
	configHash, err := cts.generateConfigHash(providerConfig)
	if err != nil {
		fiberlog.Warnf("Failed to generate config hash: %v, creating new client without caching", err)
		return cts.buildClient(ctx, providerConfig)
	}

	// Use type-safe cache with singleflight to prevent duplicate client creation
	client, err := cts.clientCache.GetOrCreate(configHash, func() (*genai.Client, error) {
		fiberlog.Debugf("Creating new Gemini client (config hash: %s)", configHash[:8])
		return cts.buildClient(ctx, providerConfig)
	})
	if err != nil {
		return nil, err
	}

	fiberlog.Debugf("Using Gemini client (config hash: %s)", configHash[:8])
	return client, nil
}

// buildClient creates a new Gemini client with the given configuration
func (cts *CountTokensService) buildClient(ctx context.Context, providerConfig models.ProviderConfig) (*genai.Client, error) {
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
