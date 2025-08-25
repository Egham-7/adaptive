package model_router

import (
	"time"

	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services"
	"adaptive-backend/internal/services/circuitbreaker"

	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go/packages/param"
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

	if cfg.Services.AdaptiveAI.BaseURL != "" {
		config.BaseURL = cfg.Services.AdaptiveAI.BaseURL
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

func (c *ModelRouterClient) SelectProtocol(
	req models.ProtocolSelectionRequest,
) models.ProtocolResponse {
	start := time.Now()

	if !c.circuitBreaker.CanExecute() {
		fiberlog.Warnf("[CIRCUIT_BREAKER] Protocol Manager service unavailable (Open state). Using fallback.")
		c.circuitBreaker.RecordRequestDuration(time.Since(start), false)
		// Log circuit breaker error but continue with fallback
		circuitErr := models.NewCircuitBreakerError("adaptive_ai")
		fiberlog.Debugf("[CIRCUIT_BREAKER] %v", circuitErr)
		return c.getFallbackProtocolResponse()
	}

	var out models.ProtocolResponse
	opts := &services.RequestOptions{Timeout: c.timeout}
	err := c.client.Post("/predict", req, &out, opts)
	if err != nil {
		c.circuitBreaker.RecordFailure()
		c.circuitBreaker.RecordRequestDuration(time.Since(start), false)
		// Log provider error but continue with fallback
		providerErr := models.NewProviderError("adaptive_ai", "prediction request failed", err)
		fiberlog.Warnf("[PROVIDER_ERROR] %v", providerErr)
		return c.getFallbackProtocolResponse()
	}

	c.circuitBreaker.RecordSuccess()
	c.circuitBreaker.RecordRequestDuration(time.Since(start), true)
	return out
}

func (c *ModelRouterClient) getFallbackProtocolResponse() models.ProtocolResponse {
	// Simple fallback: always route to standard LLM with basic parameters
	return models.ProtocolResponse{
		Protocol: models.ProtocolStandardLLM,
		Standard: &models.StandardLLMInfo{
			Provider: string(models.ProviderOpenAI),
			Model:    "gpt-4o-mini",
			Parameters: models.OpenAIParameters{
				Temperature:      param.Opt[float64]{Value: 0.7},
				TopP:             param.Opt[float64]{Value: 0.9},
				MaxTokens:        param.Opt[int64]{Value: 1000},
				N:                param.Opt[int64]{Value: 1},
				FrequencyPenalty: param.Opt[float64]{Value: 0},
				PresencePenalty:  param.Opt[float64]{Value: 0},
			},
			Alternatives: []models.Alternative{
				{
					Provider: string(models.ProviderOpenAI),
					Model:    "gpt-4o",
				},
			},
		},
	}
}
