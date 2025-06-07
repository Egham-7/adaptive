package usage

import (
	"adaptive-backend/config"
	"adaptive-backend/internal/models"

	"github.com/google/uuid"
)

type APIKeyRepository struct{}

func NewAPIKeyRepository() *APIKeyRepository {
	return &APIKeyRepository{}
}

func (r *APIKeyRepository) Create(apiKey *models.APIKey) error {
	return config.DB.Create(apiKey).Error
}

func (r *APIKeyRepository) GetAllByUserId(userId string) ([]models.APIKey, error) {
	var keys []models.APIKey
	err := config.DB.Where("user_id = ?", userId).Find(&keys).Error
	return keys, err
}

func (r *APIKeyRepository) GetById(id uuid.UUID) (models.APIKey, error) {
	var key models.APIKey
	err := config.DB.First(&key, id).Error
	return key, err
}

func (r *APIKeyRepository) Update(apiKey *models.APIKey) error {
	return config.DB.Save(apiKey).Error
}

func (r *APIKeyRepository) Delete(id uuid.UUID) error {
	return config.DB.Delete(&models.APIKey{}, id).Error
}

func (r *APIKeyRepository) GetByPrefix(prefix string) ([]models.APIKey, error) {
	var apiKeys []models.APIKey
	result := config.DB.Where("key_prefix = ?", prefix).Find(&apiKeys)
	return apiKeys, result.Error
}

// New: Find an API key by Stripe customer ID (useful for usage billing flows)
func (r *APIKeyRepository) GetByStripeCustomerID(stripeCustomerID string) ([]models.APIKey, error) {
	var apiKeys []models.APIKey
	result := config.DB.Where("stripe_customer_id = ?", stripeCustomerID).Find(&apiKeys)
	return apiKeys, result.Error
}
