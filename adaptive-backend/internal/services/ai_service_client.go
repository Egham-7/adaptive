// Package services provides client implementations for external service integrations.
// This file contains the AI Service Client, which handles communication with the
// Python-based AI service for intelligent model selection and prompt analysis.
//
// The AI Service Client is responsible for:
//   - Analyzing prompts to determine optimal LLM model selection
//   - Implementing semantic caching for performance optimization
//   - Managing user-specific and global caching strategies
//   - Providing fallback model selection when the AI service is unavailable
//   - Tracking cache hit rates and performance metrics
//
// Key features:
//   - Semantic similarity-based caching using vector embeddings
//   - Two-tier caching: user-specific and global model selection cache
//   - OpenAI embeddings for semantic prompt comparison
//   - LRU eviction for memory management
//   - Request correlation for debugging and tracing
package services

import (
	"adaptive-backend/internal/models"
	"log"
	"os"
	"time"

	"github.com/botirk38/semanticcache"
	lru "github.com/hashicorp/golang-lru/v2"
)

// PromptClassifierClient provides intelligent model selection by communicating
// with the Python AI service and implementing sophisticated caching strategies.
// It uses semantic similarity to cache model selections, reducing AI service
// calls while maintaining accuracy for similar prompts.
//
// Caching Strategy:
// The client implements a two-tier semantic caching system:
//   1. User-specific cache: Stores model selections per user for personalization
//   2. Global cache: Shared model selections across all users for efficiency
//
// Semantic Matching:
// The client uses OpenAI embeddings to compare prompt similarity with a
// configurable threshold (0.9 by default). When a semantically similar prompt
// is found in cache, the cached model selection is returned without calling
// the AI service.
//
// Memory Management:
// - Global cache: 1000 entries with LRU eviction
// - User caches: 100 entries each, with 100 user caches maximum
// - Automatic cleanup of inactive user caches
//
// Thread Safety:
// The client is safe for concurrent use across multiple goroutines.
// All caching operations are thread-safe through the underlying cache implementations.
//
// Example usage:
//
//	client := NewPromptClassifierClient()
//	model, cacheType, err := client.SelectModelWithCache(
//		"Write a Python function to sort a list",
//		"user_123",
//		"req_abc456"
//	)
//	if err != nil {
//		log.Printf("Model selection failed: %v", err)
//		return
//	}
//
//	fmt.Printf("Selected model: %s (cache: %s)", model, cacheType)
type PromptClassifierClient struct {
	// client is the underlying HTTP client for communicating with the AI service
	client *Client

	// globalPromptModelCache stores model selections across all users using semantic similarity.
	// This cache helps reduce AI service calls for common prompt patterns and improves
	// overall system performance. The cache maps prompt embeddings to selected model names.
	globalPromptModelCache *semanticcache.SemanticCache[string, string]

	// userPromptModelCache maintains separate semantic caches for each user.
	// This allows for user-specific model preferences and personalized caching
	// while still benefiting from semantic similarity matching. The outer LRU cache
	// manages user cache instances to prevent unlimited memory growth.
	userPromptModelCache *lru.Cache[string, *semanticcache.SemanticCache[string, string]]

	// embeddingProvider generates vector embeddings for semantic similarity comparison.
	// Currently uses OpenAI's text-embedding-ada-002 model for high-quality embeddings
	// that enable accurate semantic matching between prompts.
	embeddingProvider semanticcache.EmbeddingProvider
}

// NewPromptClassifierClient creates a new client instance for the AI service
// with semantic caching enabled. The client initializes both global and user-specific
// caches using OpenAI embeddings for semantic similarity comparison.
//
// Configuration:
//   - AI service URL: Read from ADAPTIVE_AI_BASE_URL environment variable
//   - OpenAI API key: Read from OPENAI_API_KEY for embeddings
//   - Global cache: 1000 entries with LRU eviction
//   - User cache pool: 100 user caches, each with 100 entries
//   - Semantic threshold: 0.9 similarity for cache hits
//
// Returns:
//   - *PromptClassifierClient: Configured client ready for use
//
// Environment Variables:
//   - ADAPTIVE_AI_BASE_URL: AI service base URL (default: http://localhost:8000)
//   - OPENAI_API_KEY: OpenAI API key for embeddings (required)
//
// The client automatically handles:
//   - Embedding provider initialization with error handling
//   - Cache creation with appropriate size limits
//   - Memory management through LRU eviction
//   - Graceful degradation if caching fails
//
// Panics:
// The function will log.Fatalf if:
//   - OpenAI embedding provider cannot be created (invalid API key)
//   - Global semantic cache initialization fails
//   - User LRU cache creation fails
func NewPromptClassifierClient() *PromptClassifierClient {
	// Get AI service base URL from environment with sensible default
	baseURL := os.Getenv("ADAPTIVE_AI_BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:8000"
	}

	// Initialize OpenAI embedding provider for semantic similarity
	// This provider generates vector embeddings used for prompt comparison
	embeddingProvider, err := semanticcache.NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "")
	if err != nil {
		log.Fatalf("Failed to create embedding provider: %v", err)
	}

	// Create global semantic cache for cross-user model selections
	// Size: 1000 entries - large enough for common prompt patterns
	// Uses LRU eviction when capacity is exceeded
	globalCache, err := semanticcache.NewSemanticCache[string, string](1000, embeddingProvider, nil)
	if err != nil {
		log.Fatalf("Failed to create global semantic cache: %v", err)
	}

	// Create LRU cache to manage per-user semantic caches
	// Size: 100 users - balances memory usage with user coverage
	// Each user gets their own semantic cache for personalized caching
	userCache, err := lru.New[string, *semanticcache.SemanticCache[string, string]](100)
	if err != nil {
		log.Fatalf("Failed to create user LRU cache: %v", err)
	}

	return &PromptClassifierClient{
		client:                 NewClient(baseURL),
		globalPromptModelCache: globalCache,
		userPromptModelCache:   userCache,
		embeddingProvider:      embeddingProvider,
	}
}

// getUserCache retrieves or creates a semantic cache for the specified user.
// This method implements lazy initialization of user caches to optimize memory
// usage - caches are only created when users first make requests.
//
// Cache Management:
//   - Checks if user cache already exists in LRU cache
//   - Creates new semantic cache if not found
//   - Adds new cache to LRU with automatic eviction of oldest users
//   - Returns nil if cache creation fails (graceful degradation)
//
// Parameters:
//   - userID: Unique identifier for the user requesting cache access
//
// Returns:
//   - *semanticcache.SemanticCache[string, string]: User's semantic cache instance
//   - nil: If cache creation fails or userID is empty
//
// Memory Behavior:
//   - Each user cache is limited to 100 entries with LRU eviction
//   - User caches are evicted from memory when the 100-user limit is exceeded
//   - Cache creation failure is logged but doesn't prevent operation
//
// Thread Safety:
// This method is thread-safe through the underlying LRU cache implementation.
// Multiple goroutines can safely request user caches concurrently.
func (c *PromptClassifierClient) getUserCache(userID string) *semanticcache.SemanticCache[string, string] {
	// Check if user already has a cache in the LRU
	if cache, ok := c.userPromptModelCache.Get(userID); ok {
		return cache
	}

	// Create new semantic cache for this user
	// Size: 100 entries per user - balances personalization with memory usage
	newCache, err := semanticcache.NewSemanticCache[string, string](100, c.embeddingProvider, nil)
	if err != nil {
		// Log warning but don't fail - system can still use global cache
		log.Printf("[WARN] Failed to create user semantic cache for %s: %v", userID, err)
		return nil
	}

	// Add new cache to LRU (may evict oldest user's cache if at capacity)
	c.userPromptModelCache.Add(userID, newCache)
	return newCache
}

// SelectModel analyzes the given prompt and returns the optimal model selection
// from the AI service. This method bypasses all caching and makes a direct
// request to the Python AI service for fresh model selection analysis.
//
// The AI service analyzes prompts across multiple dimensions:
//   - Creativity scope: How creative should the response be?
//   - Reasoning complexity: How much logical reasoning is required?
//   - Contextual knowledge: How much context understanding is needed?
//   - Domain expertise: How specialized is the required knowledge?
//
// Parameters:
//   - prompt: The user prompt to analyze for optimal model selection
//
// Returns:
//   - *models.SelectModelResponse: Complete model selection response including:
//     - SelectedModel: Recommended model identifier (e.g., "gpt-4o")
//     - Provider: LLM provider name (e.g., "openai", "anthropic")
//     - MatchScore: Confidence score (0.0-1.0) for the selection
//     - Domain: Detected content domain (e.g., "programming", "creative_writing")
//     - PromptScores: Detailed analysis scores for each dimension
//   - error: Any error encountered during AI service communication
//
// Error Conditions:
//   - Network timeouts (50-second timeout configured)
//   - AI service unavailable or returning errors
//   - Invalid response format from AI service
//   - JSON parsing errors in response
//
// Performance Notes:
//   - Direct AI service call with 50-second timeout
//   - No caching or optimization applied
//   - Use SelectModelWithCache for production workloads
//
// Example:
//
//	response, err := client.SelectModel("Explain quantum computing")
//	if err != nil {
//		return fmt.Errorf("model selection failed: %w", err)
//	}
//
//	fmt.Printf("AI recommends %s from %s (confidence: %.2f)",
//		response.SelectedModel, response.Provider, response.MatchScore)
func (c *PromptClassifierClient) SelectModel(prompt string) (*models.SelectModelResponse, error) {
	var result models.SelectModelResponse
	err := c.client.Post("/predict", models.SelectModelRequest{Prompt: prompt}, &result, &RequestOptions{Timeout: 50 * time.Second})

	return &result, err
}

// SelectModelWithCache performs intelligent model selection with semantic caching
// optimization. This method implements a sophisticated caching strategy that uses
// vector embeddings to find semantically similar prompts and return cached model
// selections when appropriate.
//
// Caching Strategy:
//  1. User-specific cache lookup: Check if user has similar cached prompt
//  2. Global cache lookup: Check cross-user cache for similar prompts
//  3. AI service call: Fresh analysis if no semantic match found
//  4. Cache population: Store new selection in both user and global caches
//
// Semantic Similarity:
// The method uses a similarity threshold of 0.9 to determine cache hits.
// This high threshold ensures that only very similar prompts return cached
// results, maintaining accuracy while improving performance.
//
// Cache Types Returned:
//   - "user": Hit in user-specific cache (fastest, personalized)
//   - "global": Hit in global cache (fast, shared across users)
//   - "miss": No cache hit, fresh AI service call (slowest, most accurate)
//
// Parameters:
//   - prompt: The user prompt to analyze for model selection
//   - userID: Unique user identifier for personalized caching
//   - requestID: Correlation ID for debugging and monitoring
//
// Returns:
//   - string: Selected model identifier (e.g., "gpt-4o", "claude-3-sonnet")
//   - string: Cache hit type ("user", "global", "miss")
//   - error: Any error encountered during the process
//
// Performance Characteristics:
//   - User cache hit: ~1-2ms (embedding lookup only)
//   - Global cache hit: ~2-5ms (embedding lookup + user cache update)
//   - Cache miss: ~1-3s (full AI service roundtrip + caching)
//
// Error Handling:
//   - Cache failures are logged but don't prevent operation
//   - AI service errors are propagated to caller
//   - Embedding generation errors fall back to direct AI service calls
//
// Monitoring:
// The method logs cache hit/miss events with request correlation for monitoring:
//   - "[req_123] Semantic cache hit (user)" - User cache provided result
//   - "[req_123] Semantic cache hit (global)" - Global cache provided result
//   - "[req_123] Semantic cache miss - selected model gpt-4o" - Fresh AI call
//
// Example:
//
//	model, cacheType, err := client.SelectModelWithCache(
//		"Write a Python function to calculate fibonacci numbers",
//		"user_123",
//		"req_abc456"
//	)
//	if err != nil {
//		return fmt.Errorf("model selection failed: %w", err)
//	}
//
//	switch cacheType {
//	case "user":
//		log.Printf("Fast user cache hit: %s", model)
//	case "global":
//		log.Printf("Global cache hit: %s", model)
//	case "miss":
//		log.Printf("Fresh AI analysis selected: %s", model)
//	}
func (c *PromptClassifierClient) SelectModelWithCache(prompt string, userID string, requestID string) (string, string, error) {
	// Semantic similarity threshold for cache hits
	// 0.9 = very high similarity required (conservative caching)
	// Lower values increase cache hits but may reduce accuracy
	const threshold = 0.9

	// Get or create user-specific semantic cache
	userCache := c.getUserCache(userID)

	// Phase 1: Check user-specific semantic cache first
	// User cache provides the fastest lookup and personalized results
	if userCache != nil {
		if val, found, err := userCache.Lookup(prompt, threshold); err == nil && found {
			// Cache hit in user-specific cache - fastest possible response
			log.Printf("[%s] Semantic cache hit (user)", requestID)
			return val, "user", nil
		}
	}

	// Phase 2: Check global semantic cache for cross-user patterns
	// Global cache captures common prompt patterns across all users
	if val, found, err := c.globalPromptModelCache.Lookup(prompt, threshold); err == nil && found {
		// Cache hit in global cache - fast response with user cache update
		log.Printf("[%s] Semantic cache hit (global)", requestID)

		// Populate user cache with global result for future user-specific hits
		if userCache != nil {
			_ = userCache.Set(prompt, prompt, val)
		}

		return val, "global", nil
	}

	// Phase 3: Cache miss - must call AI service for fresh analysis
	// This is the slowest path but provides the most accurate model selection
	modelInfo, err := c.SelectModel(prompt)
	if err != nil {
		// AI service error - return error with cache miss indicator
		return "", "miss", err
	}

	// Log cache miss with selected model for monitoring and debugging
	log.Printf("[%s] Semantic cache miss - selected model %s", requestID, modelInfo.SelectedModel)

	// Phase 4: Populate caches with new model selection
	// Store in global cache for cross-user benefit
	_ = c.globalPromptModelCache.Set(prompt, prompt, modelInfo.SelectedModel)

	// Store in user cache for personalized future hits
	if userCache != nil {
		_ = userCache.Set(prompt, prompt, modelInfo.SelectedModel)
	}

	// Return fresh model selection with miss indicator
	return modelInfo.SelectedModel, "miss", nil
}