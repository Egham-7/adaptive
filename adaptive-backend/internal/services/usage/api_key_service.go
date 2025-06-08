package usage

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/repositories/usage"
	"adaptive-backend/internal/services/cache"
	"adaptive-backend/internal/services/metrics"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
)

type APIKeyService struct {
	repo        *usage.APIKeyRepository
	cache       *cache.APIKeyCache
	promMetrics *metrics.APIKeyMetrics
}

func NewAPIKeyService() *APIKeyService {
	return &APIKeyService{
		repo:        usage.NewAPIKeyRepository(),
		cache:       cache.NewAPIKeyCache(cache.DefaultAPIKeyCacheConfig()),
		promMetrics: metrics.NewAPIKeyMetrics(),
	}
}

func NewAPIKeyServiceWithCache(cacheConfig *cache.APIKeyCacheConfig) *APIKeyService {
	service := &APIKeyService{
		repo:        usage.NewAPIKeyRepository(),
		cache:       cache.NewAPIKeyCache(cacheConfig),
		promMetrics: metrics.NewAPIKeyMetrics(),
	}
	
	// Start periodic cache cleanup
	service.cache.StartPeriodicCleanup(10 * time.Minute)
	
	return service
}

// CreateAPIKey creates a new API key in the database
func (s *APIKeyService) CreateAPIKey(apiKey *models.APIKey) error {
	start := time.Now()
	err := s.repo.Create(apiKey)
	
	if err != nil {
		s.promMetrics.VerificationDuration.WithLabelValues("creation_failed").Observe(time.Since(start).Seconds())
	} else {
		s.promMetrics.KeyCreationTotal.Inc()
		s.promMetrics.VerificationDuration.WithLabelValues("creation_success").Observe(time.Since(start).Seconds())
	}
	
	return err
}

// GenerateAPIKey creates a new API key
// Returns the API key model and the full API key string
func (s *APIKeyService) GenerateAPIKey(userId, name, status string, expiresAt *time.Time) (models.APIKey, string, error) {
	// Generate a random key with crypto-secure randomness
	keyBytes := make([]byte, 32) // 256 bits
	if _, err := rand.Read(keyBytes); err != nil {
		return models.APIKey{}, "", fmt.Errorf("failed to generate random key: %w", err)
	}

	fullKey := base64.StdEncoding.EncodeToString(keyBytes)
	prefix := fullKey[:8]
	keyHash := s.HashAPIKey(fullKey)

	apiKey := models.APIKey{
		ID:         uuid.New(),
		UserID:     userId,
		Name:       name,
		KeyPrefix:  prefix,
		KeyHash:    keyHash,
		Status:     status,
		CreatedAt:  time.Now(),
		ExpiresAt:  expiresAt,
		LastUsedAt: nil,
	}

	if err := s.CreateAPIKey(&apiKey); err != nil {
		return models.APIKey{}, "", fmt.Errorf("failed to save API key: %w", err)
	}

	formattedKey := fmt.Sprintf("%s.%s", prefix, fullKey[8:])
	
	// Pre-populate cache with the new API key
	s.cache.Set(formattedKey, apiKey, status == "active")
	
	return apiKey, formattedKey, nil
}

func (s *APIKeyService) GetAllAPIKeysByUserId(userId string) ([]models.APIKey, error) {
	return s.repo.GetAllByUserId(userId)
}

func (s *APIKeyService) GetAPIKeyById(id uuid.UUID) (models.APIKey, error) {
	return s.repo.GetById(id)
}

func (s *APIKeyService) UpdateAPIKey(apiKey *models.APIKey) error {
	start := time.Now()
	err := s.repo.Update(apiKey)
	
	if err != nil {
		s.promMetrics.VerificationDuration.WithLabelValues("update_failed").Observe(time.Since(start).Seconds())
	} else {
		s.promMetrics.KeyUpdateTotal.Inc()
		s.promMetrics.VerificationDuration.WithLabelValues("update_success").Observe(time.Since(start).Seconds())
	}
	
	return err
}

func (s *APIKeyService) DeleteAPIKey(id uuid.UUID) error {
	start := time.Now()
	err := s.repo.Delete(id)
	
	if err != nil {
		s.promMetrics.VerificationDuration.WithLabelValues("deletion_failed").Observe(time.Since(start).Seconds())
	} else {
		s.promMetrics.KeyDeletionTotal.Inc()
		s.promMetrics.VerificationDuration.WithLabelValues("deletion_success").Observe(time.Since(start).Seconds())
	}
	
	return err
}

// HashAPIKey securely hashes an API key for storage
func (s *APIKeyService) HashAPIKey(apiKey string) string {
	apiKey = strings.ReplaceAll(apiKey, ".", "")
	hasher := sha256.New()
	hasher.Write([]byte(apiKey))
	return hex.EncodeToString(hasher.Sum(nil))
}

// VerifyAPIKey checks if a provided API key is valid with caching support
func (s *APIKeyService) VerifyAPIKey(providedKey string) (models.APIKey, bool, error) {
	start := time.Now()

	// Step 1: Check cache first
	if cachedKey, found, valid := s.cache.Get(providedKey); found {
		duration := time.Since(start).Seconds()
		
		if valid {
			s.promMetrics.CacheHits.WithLabelValues("valid").Inc()
			s.promMetrics.VerificationTotal.WithLabelValues("success", "cache").Inc()
			s.promMetrics.VerificationDuration.WithLabelValues("success").Observe(duration)
			
			// Update last used time asynchronously
			go s.updateLastUsedAsync(cachedKey.ID)
			
			return cachedKey, true, nil
		} else {
			s.promMetrics.CacheHits.WithLabelValues("invalid").Inc()
			s.promMetrics.VerificationTotal.WithLabelValues("failed", "cache").Inc()
			s.promMetrics.VerificationDuration.WithLabelValues("failed").Observe(duration)
			
			return cachedKey, false, nil
		}
	}

	// Step 2: Cache miss - verify against database
	s.promMetrics.CacheMisses.WithLabelValues("not_found").Inc()

	// Validate API key format
	parts := strings.Split(providedKey, ".")
	if len(parts) != 2 {
		s.cache.SetInvalid(providedKey)
		duration := time.Since(start).Seconds()
		
		s.promMetrics.VerificationTotal.WithLabelValues("failed", "database").Inc()
		s.promMetrics.VerificationDuration.WithLabelValues("invalid").Observe(duration)
		
		return models.APIKey{}, false, fmt.Errorf("invalid API key format")
	}

	prefix := parts[0]
	apiKeys, err := s.repo.GetByPrefix(prefix)
	if err != nil {
		duration := time.Since(start).Seconds()
		
		s.promMetrics.VerificationTotal.WithLabelValues("failed", "database").Inc()
		s.promMetrics.VerificationDuration.WithLabelValues("error").Observe(duration)
		
		return models.APIKey{}, false, err
	}

	if len(apiKeys) == 0 {
		s.cache.SetInvalid(providedKey)
		duration := time.Since(start).Seconds()
		
		s.promMetrics.VerificationTotal.WithLabelValues("failed", "database").Inc()
		s.promMetrics.VerificationDuration.WithLabelValues("not_found").Observe(duration)
		
		return models.APIKey{}, false, nil
	}

	fullKey := strings.ReplaceAll(providedKey, ".", "")
	keyHash := s.HashAPIKey(fullKey)

	for _, key := range apiKeys {
		if key.KeyHash == keyHash {
			duration := time.Since(start).Seconds()
			
			// Check expiration
			if key.ExpiresAt != nil && key.ExpiresAt.Before(time.Now()) {
				s.cache.Set(providedKey, key, false)
				s.promMetrics.ExpiredKeys.Inc()
				s.promMetrics.VerificationTotal.WithLabelValues("failed", "database").Inc()
				s.promMetrics.VerificationDuration.WithLabelValues("expired").Observe(duration)
				
				return key, false, nil
			}

			// Check status
			if key.Status != "active" {
				s.cache.Set(providedKey, key, false)
				s.promMetrics.VerificationTotal.WithLabelValues("failed", "database").Inc()
				s.promMetrics.VerificationDuration.WithLabelValues("inactive").Observe(duration)
				
				return key, false, nil
			}

			// Valid key - update last used time asynchronously
			go s.updateLastUsedAsync(key.ID)

			// Cache the valid result
			s.cache.Set(providedKey, key, true)
			s.promMetrics.VerificationTotal.WithLabelValues("success", "database").Inc()
			s.promMetrics.VerificationDuration.WithLabelValues("success").Observe(duration)
			
			return key, true, nil
		}
	}

	// No matching key found
	s.cache.SetInvalid(providedKey)
	duration := time.Since(start).Seconds()
	
	s.promMetrics.VerificationTotal.WithLabelValues("failed", "database").Inc()
	s.promMetrics.VerificationDuration.WithLabelValues("not_found").Observe(duration)
	
	return models.APIKey{}, false, nil
}

// updateLastUsedAsync updates the last used time for an API key asynchronously
func (s *APIKeyService) updateLastUsedAsync(keyID uuid.UUID) {
	// Use optimized repository method that updates only the last_used_at field
	if err := s.repo.UpdateLastUsed(keyID, time.Now()); err != nil {
		// Log error but don't fail the request
		fmt.Printf("Failed to update last used time for API key %s: %v\n", keyID, err)
	}
}

// UpdateCacheMetrics updates Prometheus cache metrics
func (s *APIKeyService) UpdateCacheMetrics() {
	s.promMetrics.CacheSize.Set(float64(s.cache.Size()))
}

// ClearCache clears all cached API keys
func (s *APIKeyService) ClearCache() {
	s.cache.Clear()
}

// GetActiveKeyCount returns the number of active API keys
func (s *APIKeyService) GetActiveKeyCount() (int64, error) {
	return s.repo.GetActiveCount()
}

// CleanupExpiredKeys marks expired keys as inactive
func (s *APIKeyService) CleanupExpiredKeys() error {
	expiredKeys, err := s.repo.GetExpiredKeys()
	if err != nil {
		return err
	}
	
	if len(expiredKeys) == 0 {
		return nil
	}
	
	// Extract IDs for bulk update
	ids := make([]uuid.UUID, len(expiredKeys))
	for i, key := range expiredKeys {
		ids[i] = key.ID
		// Clear from cache if present
		s.cache.Delete(key.KeyHash)
	}
	
	// Bulk update status to inactive
	err = s.repo.BulkUpdateStatus(ids, "inactive")
	if err == nil {
		s.promMetrics.ExpiredKeys.Add(float64(len(expiredKeys)))
	}
	
	return err
}

// StartPeriodicCleanup starts a goroutine that periodically cleans up expired keys
func (s *APIKeyService) StartPeriodicCleanup(interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		
		for range ticker.C {
			if err := s.CleanupExpiredKeys(); err != nil {
				fmt.Printf("Error during periodic cleanup: %v\n", err)
			}
		}
	}()
}

// HealthCheck performs a health check of the service
func (s *APIKeyService) HealthCheck() error {
	// Check if we can perform a basic repository operation
	_, err := s.repo.GetActiveCount()
	return err
}