package model_selection

import (
	"fmt"
	"log"
	"os"
	"time"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services"
	"adaptive-backend/internal/services/circuitbreaker"
)

type PromptClassifierClient struct {
	client         *services.Client
	timeout        time.Duration
	circuitBreaker *circuitbreaker.CircuitBreaker
}

func DefaultPromptClassifierConfig() PromptClassifierConfig {
	return PromptClassifierConfig{
		BaseURL:        "http://localhost:8000",
		RequestTimeout: 10 * time.Second,
		CircuitBreakerConfig: circuitbreaker.Config{
			FailureThreshold: 3,
			SuccessThreshold: 2,
			Timeout:          20 * time.Second,
			ResetAfter:       2 * time.Minute,
		},
	}
}

type PromptClassifierConfig struct {
	BaseURL              string
	RequestTimeout       time.Duration
	CircuitBreakerConfig circuitbreaker.Config
}

func NewPromptClassifierClient() *PromptClassifierClient {
	config := DefaultPromptClassifierConfig()

	if baseURL := os.Getenv("ADAPTIVE_AI_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	return NewPromptClassifierClientWithConfig(config)
}

func NewPromptClassifierClientWithConfig(config PromptClassifierConfig) *PromptClassifierClient {
	return &PromptClassifierClient{
		client:         services.NewClient(config.BaseURL),
		timeout:        config.RequestTimeout,
		circuitBreaker: circuitbreaker.NewWithConfig(config.CircuitBreakerConfig),
	}
}

func (c *PromptClassifierClient) Classify(
	req models.ModelSelectionRequest,
) (models.ClassificationResult, error) {
	start := time.Now()

	if !c.circuitBreaker.CanExecute() {
		log.Printf("[CIRCUIT_BREAKER] PromptClassifier service unavailable (Open state). Skipping classification.")
		c.circuitBreaker.RecordRequestDuration(time.Since(start), false)
		return c.getFallbackClassificationResult(),
			fmt.Errorf("circuit breaker open for prompt classifier service")
	}

	var out models.ClassificationResult
	opts := &services.RequestOptions{Timeout: c.timeout}
	err := c.client.Post("/predict", req, &out, opts)
	if err != nil {
		c.circuitBreaker.RecordFailure()
		c.circuitBreaker.RecordRequestDuration(time.Since(start), false)
		return c.getFallbackClassificationResult(),
			fmt.Errorf("classification request failed: %w", err)
	}

	c.circuitBreaker.RecordSuccess()
	c.circuitBreaker.RecordRequestDuration(time.Since(start), true)
	return out, nil
}

func (c *PromptClassifierClient) getFallbackClassificationResult() models.ClassificationResult {
	return models.ClassificationResult{
		TaskType1:             []string{string(models.TaskOther)},
		CreativityScope:       []float64{0.5},
		Reasoning:             []float64{0.5},
		ContextualKnowledge:   []float64{0.5},
		PromptComplexityScore: []float64{0.5},
		DomainKnowledge:       []float64{0.5},
	}
}

func (c *PromptClassifierClient) GetCircuitBreakerMetrics() circuitbreaker.LocalMetrics {
	return c.circuitBreaker.GetMetrics()
}
