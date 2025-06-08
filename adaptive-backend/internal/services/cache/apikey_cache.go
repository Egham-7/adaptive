package cache

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/metrics"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"

	"github.com/hashicorp/golang-lru/v2"
)

// APIKeyCacheEntry represents a cached API key with metadata
type APIKeyCacheEntry struct {
	APIKey    models.APIKey
	Valid     bool
	CachedAt  time.Time
	ExpiresAt time.Time
}

// APIKeyCache provides high-performance caching for API key validation
type APIKeyCache struct {
	cache       *lru.Cache[string, APIKeyCacheEntry]
	cacheTTL    time.Duration
	mu          sync.RWMutex
	stats       CacheStats
	promMetrics *metrics.APIKeyMetrics
}

// CacheStats tracks cache performance metrics
type CacheStats struct {
	Hits         int64
	Misses       int64
	Evictions    int64
	Size         int
	LastClearAt  time.Time
}

// APIKeyCacheConfig holds configuration for the API key cache
type APIKeyCacheConfig struct {
	MaxEntries int
	TTL        time.Duration
}

// DefaultAPIKeyCacheConfig returns sensible defaults for API key caching
func DefaultAPIKeyCacheConfig() *APIKeyCacheConfig {
	return &APIKeyCacheConfig{
		MaxEntries: 10000,        // Support 10k active API keys
		TTL:        5 * time.Minute, // 5 minute cache TTL
	}
}

// NewAPIKeyCache creates a new API key cache with the given configuration
func NewAPIKeyCache(config *APIKeyCacheConfig) *APIKeyCache {
	return NewAPIKeyCacheWithMetrics(config, nil)
}

// NewAPIKeyCacheWithMetrics creates a new API key cache with the given configuration and metrics
func NewAPIKeyCacheWithMetrics(config *APIKeyCacheConfig, promMetrics *metrics.APIKeyMetrics) *APIKeyCache {
	if config == nil {
		config = DefaultAPIKeyCacheConfig()
	}

	cache := &APIKeyCache{
		cacheTTL:    config.TTL,
		promMetrics: promMetrics,
	}

	// Create LRU cache with eviction callback
	lruCache, err := lru.NewWithEvict[string, APIKeyCacheEntry](
		config.MaxEntries,
		func(key string, value APIKeyCacheEntry) {
			cache.mu.Lock()
			cache.stats.Evictions++
			cache.mu.Unlock()
			
			// Record eviction in Prometheus
			if cache.promMetrics != nil {
				cache.promMetrics.CacheEvictions.Inc()
			}
		},
	)
	if err != nil {
		return nil
	}

	cache.cache = lruCache
	
	// Initialize cache size metric
	if cache.promMetrics != nil {
		cache.promMetrics.CacheSize.Set(0)
	}
	
	return cache
}

// hashAPIKey creates a consistent hash of the API key for cache keying
func (c *APIKeyCache) hashAPIKey(apiKey string) string {
	hash := sha256.Sum256([]byte(apiKey))
	return hex.EncodeToString(hash[:])
}

// Get retrieves an API key from cache if it exists and is valid
func (c *APIKeyCache) Get(apiKey string) (models.APIKey, bool, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	key := c.hashAPIKey(apiKey)
	entry, found := c.cache.Get(key)

	if !found {
		c.stats.Misses++
		return models.APIKey{}, false, false
	}

	// Check if cache entry has expired based on TTL
	if time.Now().After(entry.ExpiresAt) {
		c.cache.Remove(key)
		c.stats.Misses++
		return models.APIKey{}, false, false
	}

	c.stats.Hits++

	// Check if the cached entry indicates the API key is valid
	if !entry.Valid {
		return models.APIKey{}, true, false
	}

	// Check if the API key has expired since caching
	if entry.APIKey.ExpiresAt != nil && time.Now().After(*entry.APIKey.ExpiresAt) {
		// Mark as invalid and return
		entry.Valid = false
		c.cache.Add(key, entry)
		return models.APIKey{}, true, false
	}

	return entry.APIKey, true, true
}

// Set stores an API key in the cache with its validation status
func (c *APIKeyCache) Set(apiKey string, apiKeyModel models.APIKey, valid bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := c.hashAPIKey(apiKey)
	entry := APIKeyCacheEntry{
		APIKey:    apiKeyModel,
		Valid:     valid,
		CachedAt:  time.Now(),
		ExpiresAt: time.Now().Add(c.cacheTTL),
	}

	c.cache.Add(key, entry)
	c.stats.Size = c.cache.Len()
	
	// Update Prometheus metrics
	if c.promMetrics != nil {
		c.promMetrics.CacheSize.Set(float64(c.cache.Len()))
	}
}

// SetInvalid marks an API key as invalid in the cache
func (c *APIKeyCache) SetInvalid(apiKey string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := c.hashAPIKey(apiKey)
	entry := APIKeyCacheEntry{
		APIKey:    models.APIKey{},
		Valid:     false,
		CachedAt:  time.Now(),
		ExpiresAt: time.Now().Add(c.cacheTTL),
	}

	c.cache.Add(key, entry)
	c.stats.Size = c.cache.Len()
}

// Delete removes an API key from the cache
func (c *APIKeyCache) Delete(apiKey string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := c.hashAPIKey(apiKey)
	c.cache.Remove(key)
	c.stats.Size = c.cache.Len()
	
	// Update Prometheus metrics
	if c.promMetrics != nil {
		c.promMetrics.CacheSize.Set(float64(c.cache.Len()))
	}
}

// Clear empties the entire cache
func (c *APIKeyCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache.Purge()
	c.stats.Size = 0
	c.stats.LastClearAt = time.Now()
	
	// Update Prometheus metrics
	if c.promMetrics != nil {
		c.promMetrics.CacheSize.Set(0)
	}
}

// GetStats returns current cache statistics
func (c *APIKeyCache) GetStats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	stats := c.stats
	stats.Size = c.cache.Len()
	return stats
}

// GetHitRatio returns the cache hit ratio as a percentage
func (c *APIKeyCache) GetHitRatio() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := c.stats.Hits + c.stats.Misses
	if total == 0 {
		return 0.0
	}
	return float64(c.stats.Hits) / float64(total) * 100.0
}

// Cleanup removes expired entries from the cache
func (c *APIKeyCache) Cleanup() int {
	c.mu.Lock()
	defer c.mu.Unlock()

	initialSize := c.cache.Len()
	
	// Get all keys to check for expiration
	keys := c.cache.Keys()
	now := time.Now()
	
	for _, key := range keys {
		if entry, found := c.cache.Peek(key); found {
			// Remove expired entries
			if now.After(entry.ExpiresAt) {
				c.cache.Remove(key)
			}
			// Remove invalid API keys that have been cached long enough
			if !entry.Valid && now.Sub(entry.CachedAt) > time.Minute {
				c.cache.Remove(key)
			}
		}
	}
	
	c.stats.Size = c.cache.Len()
	return initialSize - c.cache.Len()
}

// StartPeriodicCleanup starts a goroutine that periodically cleans up expired entries
func (c *APIKeyCache) StartPeriodicCleanup(interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		
		for range ticker.C {
			cleaned := c.Cleanup()
			if cleaned > 0 {
				fmt.Printf("API key cache cleanup: removed %d expired entries\n", cleaned)
			}
		}
	}()
}

// Warmup pre-populates the cache with frequently used API keys
func (c *APIKeyCache) Warmup(apiKeys []models.APIKey) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, apiKey := range apiKeys {
		// Use a placeholder key since we don't have the original API key string
		// This is mainly for internal warming when we have the model but not the raw key
		if apiKey.KeyHash != "" {
			entry := APIKeyCacheEntry{
				APIKey:    apiKey,
				Valid:     apiKey.Status == "active",
				CachedAt:  time.Now(),
				ExpiresAt: time.Now().Add(c.cacheTTL),
			}
			c.cache.Add(apiKey.KeyHash, entry)
		}
	}
	
	c.stats.Size = c.cache.Len()
}

// GetCacheKeys returns all cache keys for debugging purposes
func (c *APIKeyCache) GetCacheKeys() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	return c.cache.Keys()
}

// Size returns the current number of entries in the cache
func (c *APIKeyCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	return c.cache.Len()
}