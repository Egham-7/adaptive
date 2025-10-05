package generate

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"iter"
	"time"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/utils/clientcache"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"google.golang.org/genai"
)

// GenerateService handles Gemini GenerateContent API calls using the Gemini SDK
type GenerateService struct {
	clientCache *clientcache.Cache[*genai.Client]
}

// NewGenerateService creates a new GenerateService
func NewGenerateService() *GenerateService {
	return &GenerateService{
		clientCache: clientcache.NewCache[*genai.Client](),
	}
}

// generateConfigHash creates a hash of the provider config to detect changes
func (gs *GenerateService) generateConfigHash(providerConfig models.ProviderConfig) (string, error) {
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
func (gs *GenerateService) CreateClient(ctx context.Context, providerConfig models.ProviderConfig) (*genai.Client, error) {
	// Generate cache key based on provider config hash
	configHash, err := gs.generateConfigHash(providerConfig)
	if err != nil {
		fiberlog.Warnf("Failed to generate config hash: %v, creating new client without caching", err)
		return gs.buildClient(ctx, providerConfig)
	}

	// Use type-safe cache with singleflight to prevent duplicate client creation
	client, err := gs.clientCache.GetOrCreate(configHash, func() (*genai.Client, error) {
		fiberlog.Debugf("Creating new Gemini client (config hash: %s)", configHash[:8])
		return gs.buildClient(ctx, providerConfig)
	})
	if err != nil {
		return nil, err
	}

	fiberlog.Debugf("Using Gemini client (config hash: %s)", configHash[:8])
	return client, nil
}

// buildClient creates a new Gemini client with the given configuration
func (gs *GenerateService) buildClient(ctx context.Context, providerConfig models.ProviderConfig) (*genai.Client, error) {
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

// handleGeminiNonStreamingProvider handles non-streaming requests using native Gemini client
func (gs *GenerateService) HandleGeminiNonStreamingProvider(
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
func (gs *GenerateService) HandleGeminiStreamingProvider(
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
