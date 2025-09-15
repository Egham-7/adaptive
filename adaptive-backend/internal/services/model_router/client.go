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

	if cfg.Services.ModelRouter.BaseURL != "" {
		config.BaseURL = cfg.Services.ModelRouter.BaseURL
	}

	return NewModelRouterClientWithConfig(config, redisClient)
}

func NewModelRouterClientWithConfig(config ModelRouterClientConfig, redisClient *redis.Client) *ModelRouterClient {
	return &ModelRouterClient{
		client:         services.NewClient(config.BaseURL),
		timeout:        config.RequestTimeout,
		circuitBreaker: circuitbreaker.NewWithConfig(redisClient, config.CircuitBreakerConfig),
	}
}

func (c *ModelRouterClient) SelectModel(
	ctx context.Context,
	req models.ModelSelectionRequest,
) models.ModelSelectionResponse {
	start := time.Now()

	// Log the select model request details (non-PII at info level)
	fiberlog.Infof("[MODEL_SELECTION] Making request to model_router service - prompt_length: %d",
		len(req.Prompt))

	// Debug-level log with hashed user identifier
	if req.UserID != "" {
		hash := sha256.Sum256([]byte(req.UserID))
		hashedUserID := hex.EncodeToString(hash[:])
		fiberlog.Debugf("[MODEL_SELECTION] Request details - user_id_hash: %s", hashedUserID)
	}
	if req.CostBias != nil {
		fiberlog.Debugf("[MODEL_SELECTION] Request config - cost_bias: %.2f, available_models: %d",
			*req.CostBias, len(req.Models))
	}

	if !c.circuitBreaker.CanExecute() {
		fiberlog.Warnf("[CIRCUIT_BREAKER] Model Router service unavailable (Open state). Using fallback.")
		c.circuitBreaker.RecordRequestDuration(time.Since(start), false)
		// Log circuit breaker error but continue with fallback
		circuitErr := models.NewCircuitBreakerError("model_router")
		fiberlog.Debugf("[CIRCUIT_BREAKER] %v", circuitErr)
		return c.getFallbackModelResponse()
	}

	var out models.ModelSelectionResponse
	opts := &services.RequestOptions{Timeout: c.timeout, Context: ctx}
	fiberlog.Debugf("[SELECT_MODEL] Sending POST request to /predict endpoint")
	err := c.client.Post("/predict", req, &out, opts)
	if err != nil {
		c.circuitBreaker.RecordFailure()
		c.circuitBreaker.RecordRequestDuration(time.Since(start), false)
		// Log provider error but continue with fallback
		providerErr := models.NewProviderError("model_router", "prediction request failed", err)
		fiberlog.Warnf("[PROVIDER_ERROR] %v", providerErr)
		fiberlog.Warnf("[SELECT_MODEL] Request failed, using fallback model")
		return c.getFallbackModelResponse()
	}

	duration := time.Since(start)
	c.circuitBreaker.RecordSuccess()
	c.circuitBreaker.RecordRequestDuration(duration, true)
	fiberlog.Infof("[SELECT_MODEL] Request successful in %v - model: %s/%s",
		duration, out.Provider, out.Model)
	return out
}

func (c *ModelRouterClient) getFallbackModelResponse() models.ModelSelectionResponse {
	// Simple fallback: always route to gpt-4o-mini
	return models.ModelSelectionResponse{
		Provider: "openai",
		Model:    "gpt-4o-mini",
		Alternatives: []models.Alternative{
			{
				Provider: "openai",
				Model:    "gpt-4o",
			},
		},
	}
}
