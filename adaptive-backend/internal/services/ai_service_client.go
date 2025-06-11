package services

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/circuitbreaker"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/botirk38/semanticcache"
	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/openai/openai-go"
)

type PromptClassifierClient struct {
	client                 *Client
	circuitBreaker         *circuitbreaker.CircuitBreaker
	globalPromptModelCache *semanticcache.SemanticCache[string, models.SelectModelResponse]
	userPromptModelCache   *lru.Cache[string, *semanticcache.SemanticCache[string, models.SelectModelResponse]]
	embeddingProvider      semanticcache.EmbeddingProvider
	config                 PromptClassifierConfig
}

type PromptClassifierConfig struct {
	BaseURL              string
	GlobalCacheSize      int
	UserCachePoolSize    int
	UserCacheSize        int
	SemanticThreshold    float64
	RequestTimeout       time.Duration
	EnableFallback       bool
	CircuitBreakerConfig circuitbreaker.Config
}

func DefaultPromptClassifierConfig() PromptClassifierConfig {
	return PromptClassifierConfig{
		BaseURL:           "http://localhost:8000",
		GlobalCacheSize:   2000,
		UserCachePoolSize: 200,
		UserCacheSize:     150,
		SemanticThreshold: 0.85,
		RequestTimeout:    10 * time.Second,
		EnableFallback:    true,
		CircuitBreakerConfig: circuitbreaker.Config{
			FailureThreshold: 3,
			SuccessThreshold: 2,
			Timeout:          20 * time.Second,
			ResetAfter:       2 * time.Minute,
		},
	}
}

func NewPromptClassifierClient() *PromptClassifierClient {
	config := DefaultPromptClassifierConfig()

	if baseURL := os.Getenv("ADAPTIVE_AI_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	return NewPromptClassifierClientWithConfig(config)
}

func NewPromptClassifierClientWithConfig(config PromptClassifierConfig) *PromptClassifierClient {
	embeddingProvider, err := semanticcache.NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "")
	if err != nil {
		log.Fatalf("Failed to create embedding provider: %v", err)
	}

	globalCache, err := semanticcache.NewSemanticCache[string, models.SelectModelResponse](
		config.GlobalCacheSize,
		embeddingProvider,
		nil,
	)
	if err != nil {
		log.Fatalf("Failed to create global semantic cache: %v", err)
	}

	userCache, err := lru.New[string, *semanticcache.SemanticCache[string, models.SelectModelResponse]](
		config.UserCachePoolSize,
	)
	if err != nil {
		log.Fatalf("Failed to create user LRU cache: %v", err)
	}

	client := &PromptClassifierClient{
		client:                 NewClient(config.BaseURL),
		circuitBreaker:         circuitbreaker.NewWithConfig(config.CircuitBreakerConfig),
		globalPromptModelCache: globalCache,
		userPromptModelCache:   userCache,
		embeddingProvider:      embeddingProvider,
		config:                 config,
	}

	go client.startCacheMaintenance()

	return client
}

func (c *PromptClassifierClient) getUserCache(userID string) *semanticcache.SemanticCache[string, models.SelectModelResponse] {
	if cache, ok := c.userPromptModelCache.Get(userID); ok {
		return cache
	}

	newCache, err := semanticcache.NewSemanticCache[string, models.SelectModelResponse](
		c.config.UserCacheSize,
		c.embeddingProvider,
		nil,
	)
	if err != nil {
		log.Printf("Warning: Failed to create user cache for %s: %v", userID, err)
		return nil
	}

	c.userPromptModelCache.Add(userID, newCache)
	return newCache
}

func (c *PromptClassifierClient) getFallbackModel(provider string) *models.SelectModelResponse {
	var selectedModel string
	var fallbackProvider string

	if provider != "" {
		// Provider-specific fallbacks
		switch provider {
		case "openai":
			selectedModel = "gpt-4o"
			fallbackProvider = "openai"
		case "anthropic":
			selectedModel = "claude-3-5-sonnet-20241022"
			fallbackProvider = "anthropic"
		case "groq":
			selectedModel = "llama-3.2-3b-preview"
			fallbackProvider = "groq"
		case "deepseek":
			selectedModel = "deepseek-chat"
			fallbackProvider = "deepseek"
		case "gemini":
			selectedModel = "gemini-pro"
			fallbackProvider = "gemini"
		default:
			selectedModel = "gpt-4o"
			fallbackProvider = "openai"
		}
		log.Printf("[FALLBACK] Using fallback model: %s from %s", selectedModel, fallbackProvider)
	} else {
		selectedModel = "gpt-4o"
		fallbackProvider = "openai"
		log.Printf("[FALLBACK] Using fallback model: %s", selectedModel)
	}

	return &models.SelectModelResponse{
		SelectedModel: selectedModel,
		Provider:      fallbackProvider,
		Parameters: openai.ChatCompletionNewParams{
			MaxTokens:        openai.Int(4096),
			Temperature:      openai.Float(0.7),
			TopP:             openai.Float(1.0),
			FrequencyPenalty: openai.Float(0.0),
			PresencePenalty:  openai.Float(0.0),
		},
	}
}

func (c *PromptClassifierClient) SelectModel(req models.SelectModelRequest) (*models.SelectModelResponse, error) {
	start := time.Now()

	if !c.circuitBreaker.CanExecute() {
		log.Printf("[CIRCUIT_BREAKER] Service unavailable, using fallback")
		c.circuitBreaker.RecordRequestDuration(time.Since(start), false)
		return c.getFallbackModel(req.Provider), nil
	}

	var result models.SelectModelResponse
	err := c.client.Post("/predict", req, &result, &RequestOptions{Timeout: 50 * time.Second})
	if err != nil {
		c.circuitBreaker.RecordFailure()
		c.circuitBreaker.RecordRequestDuration(time.Since(start), false)
		log.Printf("[CIRCUIT_BREAKER] AI service call failed, using fallback: %v", err)
		return c.getFallbackModel(req.Provider), nil
	}

	c.circuitBreaker.RecordSuccess()
	c.circuitBreaker.RecordRequestDuration(time.Since(start), true)

	return &result, nil
}

func (c *PromptClassifierClient) SelectModelWithCache(req models.SelectModelRequest, userID string, requestID string) (*models.SelectModelResponse, string, error) {
	threshold := c.config.SemanticThreshold
	userCache := c.getUserCache(userID)

	// Create cache key that includes provider constraint
	cacheKey := req.Prompt
	if req.Provider != "" {
		cacheKey = fmt.Sprintf("%s:provider:%s", req.Prompt, req.Provider)
	}

	// Check user-specific cache
	if userCache != nil {
		if val, found, err := userCache.Lookup(cacheKey, float32(threshold)); err == nil && found {
			log.Printf("[%s] Semantic cache hit (user)", requestID)
			return &val, "user", nil
		}
	}

	// Check global cache
	if val, found, err := c.globalPromptModelCache.Lookup(cacheKey, float32(threshold)); err == nil && found {
		log.Printf("[%s] Semantic cache hit (global)", requestID)

		if userCache != nil {
			_ = userCache.Set(cacheKey, cacheKey, val)
		}

		return &val, "global", nil
	}

	// Cache miss - call AI service
	modelInfo, err := c.SelectModel(req)
	if err != nil {
		return nil, "miss", err
	}

	cacheType := "miss"
	if c.circuitBreaker.GetState() != circuitbreaker.Closed {
		cacheType = "fallback"
		log.Printf("[%s] Cache miss - using fallback model %s (circuit: %v)", requestID, modelInfo.SelectedModel, c.circuitBreaker.GetState())
	} else {
		providerText := "all providers"
		if req.Provider != "" {
			providerText = req.Provider
		}
		log.Printf("[%s] Cache miss - selected model %s from %s", requestID, modelInfo.SelectedModel, providerText)
	}

	// Cache the result if circuit breaker is closed
	if c.circuitBreaker.GetState() == circuitbreaker.Closed {
		if err := c.globalPromptModelCache.Set(cacheKey, cacheKey, *modelInfo); err != nil {
			log.Printf("[%s] Failed to cache in global cache: %v", requestID, err)
		}

		if userCache != nil {
			if err := userCache.Set(cacheKey, cacheKey, *modelInfo); err != nil {
				log.Printf("[%s] Failed to cache in user cache: %v", requestID, err)
			}
		}
	}

	return modelInfo, cacheType, nil
}

func (c *PromptClassifierClient) GetCircuitBreakerMetrics() circuitbreaker.LocalMetrics {
	return c.circuitBreaker.GetMetrics()
}

func (c *PromptClassifierClient) startCacheMaintenance() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		c.performCacheMaintenance()
	}
}

func (c *PromptClassifierClient) performCacheMaintenance() {
	log.Printf("Cache maintenance completed")
}

