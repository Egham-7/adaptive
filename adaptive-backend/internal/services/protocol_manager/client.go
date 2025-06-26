package protocol_manager

import (
	"fmt"
	"log"
	"os"
	"time"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services"
	"adaptive-backend/internal/services/circuitbreaker"

	"github.com/openai/openai-go/packages/param"
)

type ProtocolManagerClient struct {
	client         *services.Client
	timeout        time.Duration
	circuitBreaker *circuitbreaker.CircuitBreaker
}

func DefaultProtocolManagerConfig() ProtocolManagerConfig {
	return ProtocolManagerConfig{
		BaseURL:        "http://localhost:8000",
		RequestTimeout: 15 * time.Second,
		CircuitBreakerConfig: circuitbreaker.Config{
			FailureThreshold: 3,
			SuccessThreshold: 2,
			Timeout:          30 * time.Second,
			ResetAfter:       2 * time.Minute,
		},
	}
}

type ProtocolManagerConfig struct {
	BaseURL              string
	RequestTimeout       time.Duration
	CircuitBreakerConfig circuitbreaker.Config
}

func NewProtocolManagerClient() *ProtocolManagerClient {
	config := DefaultProtocolManagerConfig()

	if baseURL := os.Getenv("ADAPTIVE_AI_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	return NewProtocolManagerClientWithConfig(config)
}

func NewProtocolManagerClientWithConfig(config ProtocolManagerConfig) *ProtocolManagerClient {
	return &ProtocolManagerClient{
		client:         services.NewClient(config.BaseURL),
		timeout:        config.RequestTimeout,
		circuitBreaker: circuitbreaker.NewWithConfig(config.CircuitBreakerConfig),
	}
}

func (c *ProtocolManagerClient) SelectProtocol(
	req models.ModelSelectionRequest,
) (models.ProtocolResponse, error) {
	start := time.Now()

	if !c.circuitBreaker.CanExecute() {
		log.Printf("[CIRCUIT_BREAKER] Protocol Manager service unavailable (Open state). Using fallback.")
		c.circuitBreaker.RecordRequestDuration(time.Since(start), false)
		return c.getFallbackProtocolResponse(req),
			fmt.Errorf("circuit breaker open for protocol manager service")
	}

	var out models.ProtocolResponse
	opts := &services.RequestOptions{Timeout: c.timeout}
	err := c.client.Post("/predict", req, &out, opts)
	if err != nil {
		c.circuitBreaker.RecordFailure()
		c.circuitBreaker.RecordRequestDuration(time.Since(start), false)
		return c.getFallbackProtocolResponse(req),
			fmt.Errorf("protocol selection request failed: %w", err)
	}

	c.circuitBreaker.RecordSuccess()
	c.circuitBreaker.RecordRequestDuration(time.Since(start), true)
	return out, nil
}

func (c *ProtocolManagerClient) getFallbackProtocolResponse(req models.ModelSelectionRequest) models.ProtocolResponse {
	// Simple fallback: always route to standard LLM with basic parameters
	return models.ProtocolResponse{
		Protocol: models.ProtocolStandardLLM,
		Standard: &models.StandardLLMInfo{
			Provider:   string(models.ProviderOpenAI),
			Model:      "gpt-4o-mini",
			Confidence: 0.5,
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

func (c *ProtocolManagerClient) GetCircuitBreakerMetrics() circuitbreaker.LocalMetrics {
	return c.circuitBreaker.GetMetrics()
}
