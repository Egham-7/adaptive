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

// GenerateAPIKey creates a new API key, storing the Stripe customer ID if provided
// Returns the API key model and the full API key string
func (s *APIKeyService) GenerateAPIKey(userId, name, status string, expiresAt *time.Time, stripeCustomerID string) (models.APIKey, string, error) {
	// Generate a random key
	keyBytes := make([]byte, 32) // 256 bits
	if _, err := rand.Read(keyBytes); err != nil {
		return models.APIKey{}, "", fmt.Errorf("failed to generate random key: %w", err)
	}

	fullKey := base64.StdEncoding.EncodeToString(keyBytes)
	prefix := fullKey[:8]
	keyHash := s.HashAPIKey(fullKey)

	apiKey := models.APIKey{
		ID:               uuid.New(),
		UserID:           userId,
		Name:             name,
		KeyPrefix:        prefix,
		KeyHash:          keyHash,
		Status:           status,
		StripeCustomerID: stripeCustomerID, // Set Stripe Customer ID
		CreatedAt:        time.Now(),
		ExpiresAt:        expiresAt,
		LastUsedAt:       nil,
	}

	if err := s.CreateAPIKey(&apiKey); err != nil {
		return models.APIKey{}, "", fmt.Errorf("failed to save API key: %w", err)
	}

	formattedKey := fmt.Sprintf("%s.%s", prefix, fullKey[8:])
	return apiKey, formattedKey, nil
}

func (s *APIKeyService) GetAllAPIKeysByUserId(userId string) ([]models.APIKey, error) {
	return s.repo.GetAllByUserId(userId)
}

func (s *APIKeyService) GetAPIKeyById(id uuid.UUID) (models.APIKey, error) {
	return s.repo.GetById(id)
}

func (s *APIKeyService) UpdateAPIKey(apiKey *models.APIKey) error {
	return s.repo.Update(apiKey)
}

func (s *APIKeyService) DeleteAPIKey(id uuid.UUID) error {
	return s.repo.Delete(id)
}

// HashAPIKey securely hashes an API key for storage
func (s *APIKeyService) HashAPIKey(apiKey string) string {
	apiKey = strings.ReplaceAll(apiKey, ".", "")
	hasher := sha256.New()
	hasher.Write([]byte(apiKey))
	return hex.EncodeToString(hasher.Sum(nil))
}

// VerifyAPIKey checks if a provided API key is valid and returns the key and valid status
func (s *APIKeyService) VerifyAPIKey(providedKey string) (models.APIKey, bool, error) {
	parts := strings.Split(providedKey, ".")
	if len(parts) != 2 {
		return models.APIKey{}, false, fmt.Errorf("invalid API key format")
	}
	prefix := parts[0]
	apiKeys, err := s.repo.GetByPrefix(prefix)
	if err != nil {
		return models.APIKey{}, false, err
	}
	if len(apiKeys) == 0 {
		return models.APIKey{}, false, nil
	}
	fullKey := strings.ReplaceAll(providedKey, ".", "")
	keyHash := s.HashAPIKey(fullKey)
	for _, key := range apiKeys {
		if key.KeyHash == keyHash {
			key.LastUsedAt = new(time.Time)
			*key.LastUsedAt = time.Now()
			err := s.repo.Update(&key)
			if err != nil {
				return key, false, err
			}
			if key.ExpiresAt != nil && key.ExpiresAt.Before(time.Now()) {
				return key, false, nil
			}
			if key.Status != "active" {
				return key, false, nil
			}
			return key, true, nil
		}
	}
	return models.APIKey{}, false, nil
}
