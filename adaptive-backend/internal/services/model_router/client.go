package model_router

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"

	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services"
	"adaptive-backend/internal/services/circuitbreaker"

	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/golang-jwt/jwt/v5"
	"github.com/redis/go-redis/v9"
)

type ModelRouterClient struct {
	adaptiveRouterURL string
	jwtSecret         string
	timeout           time.Duration
	circuitBreaker    *circuitbreaker.CircuitBreaker
}

func DefaultModelRouterClientConfig() ModelRouterClientConfig {
	return ModelRouterClientConfig{
		AdaptiveRouterURL: "",
		JWTSecret:         "",
		RequestTimeout:    5 * time.Second,
		CircuitBreakerConfig: circuitbreaker.Config{
			FailureThreshold: 3,
			SuccessThreshold: 2,
			Timeout:          10 * time.Second,
			ResetAfter:       30 * time.Second,
		},
	}
}

type ModelRouterClientConfig struct {
	AdaptiveRouterURL    string
	JWTSecret            string
	RequestTimeout       time.Duration
	CircuitBreakerConfig circuitbreaker.Config
}

func NewModelRouterClient(cfg *config.Config, redisClient *redis.Client) *ModelRouterClient {
	config := DefaultModelRouterClientConfig()

	if cfg.Services.ModelRouter.Client.AdaptiveRouterURL != "" {
		config.AdaptiveRouterURL = cfg.Services.ModelRouter.Client.AdaptiveRouterURL
	}

	if cfg.Services.ModelRouter.Client.JWTSecret != "" {
		config.JWTSecret = cfg.Services.ModelRouter.Client.JWTSecret
	}

	// Map timeout from config (milliseconds to duration)
	if cfg.Services.ModelRouter.Client.TimeoutMs > 0 {
		config.RequestTimeout = time.Duration(cfg.Services.ModelRouter.Client.TimeoutMs) * time.Millisecond
	}

	// Map circuit breaker configuration
	cbCfg := &cfg.Services.ModelRouter.Client.CircuitBreaker
	if cbCfg.FailureThreshold > 0 {
		config.CircuitBreakerConfig.FailureThreshold = cbCfg.FailureThreshold
	}
	if cbCfg.SuccessThreshold > 0 {
		config.CircuitBreakerConfig.SuccessThreshold = cbCfg.SuccessThreshold
	}
	if cbCfg.TimeoutMs > 0 {
		config.CircuitBreakerConfig.Timeout = time.Duration(cbCfg.TimeoutMs) * time.Millisecond
	}
	if cbCfg.ResetAfterMs > 0 {
		config.CircuitBreakerConfig.ResetAfter = time.Duration(cbCfg.ResetAfterMs) * time.Millisecond
	}

	return NewModelRouterClientWithConfig(config, redisClient)
}

func NewModelRouterClientWithConfig(config ModelRouterClientConfig, redisClient *redis.Client) *ModelRouterClient {
	return &ModelRouterClient{
		adaptiveRouterURL: config.AdaptiveRouterURL,
		jwtSecret:         config.JWTSecret,
		timeout:           config.RequestTimeout,
		circuitBreaker:    circuitbreaker.NewWithConfig(redisClient, "adaptive_router", config.CircuitBreakerConfig),
	}
}

func (c *ModelRouterClient) generateJWT() (string, error) {
	if c.jwtSecret == "" {
		return "", fmt.Errorf("JWT secret not configured")
	}

	now := time.Now()
	claims := jwt.MapClaims{
		"sub": "adaptive-backend",
		"iat": now.Unix(),
		"exp": now.Add(5 * time.Minute).Unix(),
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(c.jwtSecret))
}

func (c *ModelRouterClient) SelectModel(
	ctx context.Context,
	req models.ModelSelectionRequest,
) models.ModelSelectionResponse {
	start := time.Now()

	// Log the select model request details (non-PII at info level)
	fiberlog.Infof("[MODEL_SELECTION] Making request to adaptive_router service - prompt_length: %d, valid_models: %d",
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
		fiberlog.Warnf("[CIRCUIT_BREAKER] Adaptive Router service unavailable (Open state). Using fallback.")
		// Log circuit breaker error but continue with fallback
		circuitErr := models.NewCircuitBreakerError("adaptive_router")
		fiberlog.Debugf("[CIRCUIT_BREAKER] %v", circuitErr)
		return c.getFallbackModelResponse(req.Models)
	}

	// Generate JWT token
	jwtToken, err := c.generateJWT()
	if err != nil {
		fiberlog.Warnf("[JWT_ERROR] Failed to generate JWT token: %v. Using fallback.", err)
		return c.getFallbackModelResponse(req.Models)
	}

	var out models.ModelSelectionResponse
	opts := &services.RequestOptions{
		Timeout: c.timeout,
		Context: ctx,
		Headers: map[string]string{
			"Authorization": fmt.Sprintf("Bearer %s", jwtToken),
		},
	}
	fiberlog.Debugf("[SELECT_MODEL] Sending POST request to Modal function: %s", c.adaptiveRouterURL)

	client := services.NewClient(c.adaptiveRouterURL)
	err = client.Post("", req, &out, opts)
	if err != nil {
		c.circuitBreaker.RecordFailure()
		// Log provider error but continue with fallback
		providerErr := models.NewProviderError("adaptive_router", "prediction request failed", err)
		fiberlog.Warnf("[PROVIDER_ERROR] %v", providerErr)
		fiberlog.Warnf("[SELECT_MODEL] Request failed, using fallback model")
		return c.getFallbackModelResponse(req.Models)
	}

	// Validate the response from model router - use fallback if invalid
	if !out.IsValid() {
		c.circuitBreaker.RecordFailure()
		fiberlog.Warnf("[SELECT_MODEL] Adaptive router returned invalid response (provider: '%s', model: '%s'), using fallback",
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
