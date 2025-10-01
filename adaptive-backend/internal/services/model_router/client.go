package model_router

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"time"

	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services"
	"adaptive-backend/internal/services/circuitbreaker"

	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/redis/go-redis/v9"
)

type ModelRouterClient struct {
	client         *services.Client
	timeout        time.Duration
	circuitBreaker *circuitbreaker.CircuitBreaker
}

func DefaultModelRouterClientConfig() ModelRouterClientConfig {
	return ModelRouterClientConfig{
		BaseURL:        "http://localhost:8000",
		RequestTimeout: 5 * time.Second,
		CircuitBreakerConfig: circuitbreaker.Config{
			FailureThreshold: 3,
			SuccessThreshold: 2,
			Timeout:          10 * time.Second,
			ResetAfter:       30 * time.Second,
		},
	}
}

type ModelRouterClientConfig struct {
	BaseURL              string
	RequestTimeout       time.Duration
	CircuitBreakerConfig circuitbreaker.Config
}

func NewModelRouterClient(cfg *config.Config, redisClient *redis.Client) *ModelRouterClient {
	config := DefaultModelRouterClientConfig()

	if cfg.Services.ModelRouter.Client.BaseURL != "" {
		config.BaseURL = cfg.Services.ModelRouter.Client.BaseURL
	}

	return NewModelRouterClientWithConfig(config, redisClient)
}

func NewModelRouterClientWithConfig(config ModelRouterClientConfig, redisClient *redis.Client) *ModelRouterClient {
	return &ModelRouterClient{
		client:         services.NewClient(config.BaseURL),
		timeout:        config.RequestTimeout,
		circuitBreaker: circuitbreaker.NewWithConfig(redisClient, "model_router", config.CircuitBreakerConfig),
	}
}

func (c *ModelRouterClient) SelectModel(
	ctx context.Context,
	req models.ModelSelectionRequest,
) models.ModelSelectionResponse {
	start := time.Now()

	// Log the select model request details (non-PII at info level)
	fiberlog.Infof("[MODEL_SELECTION] Making request to model_router service - prompt_length: %d, valid_models: %d",
		len(req.Prompt), len(req.Models))

	// Debug-level log with hashed user identifier
	if req.UserID != "" {
		hash := sha256.Sum256([]byte(req.UserID))
		hashedUserID := hex.EncodeToString(hash[:])
		fiberlog.Debugf("[MODEL_SELECTION] Request details - user_id_hash: %s", hashedUserID)
	}
	if req.CostBias != nil {
		fiberlog.Debugf("[MODEL_SELECTION] Request config - cost_bias: %.2f, valid_models: %d",
			*req.CostBias, len(req.Models))
	}

	if !c.circuitBreaker.CanExecute() {
		fiberlog.Warnf("[CIRCUIT_BREAKER] Model Router service unavailable (Open state). Using fallback.")
		// Log circuit breaker error but continue with fallback
		circuitErr := models.NewCircuitBreakerError("model_router")
		fiberlog.Debugf("[CIRCUIT_BREAKER] %v", circuitErr)
		return c.getFallbackModelResponse(req.Models)
	}

	var out models.ModelSelectionResponse
	opts := &services.RequestOptions{Timeout: c.timeout, Context: ctx}
	fiberlog.Debugf("[SELECT_MODEL] Sending POST request to /predict endpoint")
	err := c.client.Post("/predict", req, &out, opts)
	if err != nil {
		c.circuitBreaker.RecordFailure()
		// Log provider error but continue with fallback
		providerErr := models.NewProviderError("model_router", "prediction request failed", err)
		fiberlog.Warnf("[PROVIDER_ERROR] %v", providerErr)
		fiberlog.Warnf("[SELECT_MODEL] Request failed, using fallback model")
		return c.getFallbackModelResponse(req.Models)
	}

	// Validate the response from model router - use fallback if invalid
	if !out.IsValid() {
		c.circuitBreaker.RecordFailure()
		fiberlog.Warnf("[SELECT_MODEL] Model router returned invalid response (provider: '%s', model: '%s'), using fallback",
			out.Provider, out.Model)
		return c.getFallbackModelResponse(req.Models)
	}

	duration := time.Since(start)
	c.circuitBreaker.RecordSuccess()
	fiberlog.Infof("[SELECT_MODEL] Request successful in %v - model: %s/%s",
		duration, out.Provider, out.Model)
	return out
}

func (c *ModelRouterClient) getFallbackModelResponse(availableModels []models.ModelCapability) models.ModelSelectionResponse {
	// Filter valid models (both provider and model name required)
	var validModels []models.ModelCapability
	for _, model := range availableModels {
		if model.Provider != "" && model.ModelName != "" {
			validModels = append(validModels, model)
		}
	}

	// If we have valid models, choose the first one
	if len(validModels) > 0 {
		firstModel := validModels[0]
		response := models.ModelSelectionResponse{
			Provider: firstModel.Provider,
			Model:    firstModel.ModelName,
		}

		// Add alternatives from remaining models (up to 3 alternatives)
		for i := 1; i < len(validModels); i++ {
			response.Alternatives = append(response.Alternatives, models.Alternative{
				Provider: validModels[i].Provider,
				Model:    validModels[i].ModelName,
			})
		}

		return response
	}

	// Simple fallback: route to gemini-2.5-flash when no models provided, with gpt-4o as alternative
	return models.ModelSelectionResponse{
		Provider: "gemini",
		Model:    "gemini-2.5-flash",
		Alternatives: []models.Alternative{
			{
				Provider: "openai",
				Model:    "gpt-4o",
			},
		},
	}
}
