package usage

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/repositories/usage"
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
	repo *usage.APIKeyRepository
}

func NewAPIKeyService() *APIKeyService {
	return &APIKeyService{
		repo: usage.NewAPIKeyRepository(),
	}
}

// CreateAPIKey creates a new API key in the database
func (s *APIKeyService) CreateAPIKey(apiKey *models.APIKey) error {
	return s.repo.Create(apiKey)
}

// GenerateAPIKey creates a new API key with a secure random value
// It returns the API key model and the full API key string
func (s *APIKeyService) GenerateAPIKey(userId uuid.UUID, name, status string, expiresAt *time.Time) (models.APIKey, string, error) {
	// Generate a random key
	keyBytes := make([]byte, 32) // 256 bits
	if _, err := rand.Read(keyBytes); err != nil {
		return models.APIKey{}, "", fmt.Errorf("failed to generate random key: %w", err)
	}

	// Encode the key to base64 for user-friendly representation
	fullKey := base64.StdEncoding.EncodeToString(keyBytes)

	// Create a prefix (first 8 characters)
	prefix := fullKey[:8]

	// Hash the key for storage
	keyHash := s.HashAPIKey(fullKey)

	// Create the API key model
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

	// Save to database
	if err := s.CreateAPIKey(&apiKey); err != nil {
		return models.APIKey{}, "", fmt.Errorf("failed to save API key: %w", err)
	}

	// Format the key for display (prefix.rest_of_key)
	formattedKey := fmt.Sprintf("%s.%s", prefix, fullKey[8:])

	return apiKey, formattedKey, nil
}

func (s *APIKeyService) GetAllAPIKeysByUserId(userId uint) ([]models.APIKey, error) {
	return s.repo.GetAllByUserId(userId)
}

func (s *APIKeyService) GetAPIKeyById(id uint) (models.APIKey, error) {
	return s.repo.GetById(id)
}

func (s *APIKeyService) UpdateAPIKey(apiKey *models.APIKey) error {
	return s.repo.Update(apiKey)
}

func (s *APIKeyService) DeleteAPIKey(id uint) error {
	return s.repo.Delete(id)
}

// HashAPIKey securely hashes an API key for storage
func (s *APIKeyService) HashAPIKey(apiKey string) string {
	// Remove any formatting (like dots) if present
	apiKey = strings.ReplaceAll(apiKey, ".", "")

	// Create a SHA-256 hash
	hasher := sha256.New()
	hasher.Write([]byte(apiKey))

	// Return the hex-encoded hash
	return hex.EncodeToString(hasher.Sum(nil))
}

// VerifyAPIKey checks if a provided API key is valid
func (s *APIKeyService) VerifyAPIKey(providedKey string) (models.APIKey, bool, error) {
	// Extract the prefix from the key
	parts := strings.Split(providedKey, ".")
	if len(parts) != 2 {
		return models.APIKey{}, false, fmt.Errorf("invalid API key format")
	}

	prefix := parts[0]

	// Find API keys with this prefix
	apiKeys, err := s.repo.GetByPrefix(prefix)
	if err != nil {
		return models.APIKey{}, false, err
	}

	if len(apiKeys) == 0 {
		return models.APIKey{}, false, nil
	}

	// Hash the provided key
	fullKey := strings.ReplaceAll(providedKey, ".", "")
	keyHash := s.HashAPIKey(fullKey)

	// Check if any of the keys match
	for _, key := range apiKeys {
		if key.KeyHash == keyHash {
			// Update last used time
			key.LastUsedAt = new(time.Time)
			*key.LastUsedAt = time.Now()
			s.repo.Update(&key)

			// Check if key is expired
			if key.ExpiresAt != nil && key.ExpiresAt.Before(time.Now()) {
				return key, false, nil
			}

			// Check if key is active
			if key.Status != "active" {
				return key, false, nil
			}

			return key, true, nil
		}
	}

	return models.APIKey{}, false, nil
}
