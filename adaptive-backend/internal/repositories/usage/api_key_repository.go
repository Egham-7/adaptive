package usage

import (
	"adaptive-backend/config"
	"adaptive-backend/internal/models"
	"time"

	"github.com/google/uuid"
)

type APIKeyRepository struct{}

func NewAPIKeyRepository() *APIKeyRepository {
	return &APIKeyRepository{}
}

func (r *APIKeyRepository) Create(apiKey *models.APIKey) error {
	start := time.Now()
	err := config.DB.Create(apiKey).Error
	config.RecordQueryMetrics("create", "api_keys", time.Since(start), err)
	return err
}

func (r *APIKeyRepository) GetAllByUserId(userId string) ([]models.APIKey, error) {
	start := time.Now()
	var keys []models.APIKey
	
	// Optimized query with index hint and selective fields
	err := config.DB.
		Select("id, user_id, name, key_prefix, status, created_at, expires_at, last_used_at").
		Where("user_id = ? AND status = ?", userId, "active").
		Order("created_at DESC").
		Find(&keys).Error
	
	config.RecordQueryMetrics("select", "api_keys", time.Since(start), err)
	return keys, err
}

func (r *APIKeyRepository) GetById(id uuid.UUID) (models.APIKey, error) {
	start := time.Now()
	var key models.APIKey
	
	err := config.DB.
		Select("id, user_id, name, key_prefix, key_hash, status, created_at, expires_at, last_used_at").
		First(&key, "id = ?", id).Error
	
	config.RecordQueryMetrics("select", "api_keys", time.Since(start), err)
	return key, err
}

func (r *APIKeyRepository) Update(apiKey *models.APIKey) error {
	start := time.Now()
	
	// Optimized update - only update specific fields to avoid unnecessary writes
	err := config.DB.Model(apiKey).
		Select("name", "status", "last_used_at", "expires_at").
		Where("id = ?", apiKey.ID).
		Updates(apiKey).Error
	
	config.RecordQueryMetrics("update", "api_keys", time.Since(start), err)
	return err
}

func (r *APIKeyRepository) UpdateLastUsed(id uuid.UUID, lastUsed time.Time) error {
	start := time.Now()
	
	// Highly optimized update for just last_used_at field
	err := config.DB.Model(&models.APIKey{}).
		Where("id = ?", id).
		Update("last_used_at", lastUsed).Error
	
	config.RecordQueryMetrics("update", "api_keys", time.Since(start), err)
	return err
}

func (r *APIKeyRepository) Delete(id uuid.UUID) error {
	start := time.Now()
	err := config.DB.Delete(&models.APIKey{}, "id = ?", id).Error
	config.RecordQueryMetrics("delete", "api_keys", time.Since(start), err)
	return err
}

func (r *APIKeyRepository) GetByPrefix(prefix string) ([]models.APIKey, error) {
	start := time.Now()
	var apiKeys []models.APIKey
	
	// Optimized query using the prefix index
	err := config.DB.
		Select("id, user_id, key_hash, status, expires_at").
		Where("key_prefix = ? AND status = ?", prefix, "active").
		Find(&apiKeys).Error
	
	config.RecordQueryMetrics("select", "api_keys", time.Since(start), err)
	return apiKeys, err
}



func (r *APIKeyRepository) GetActiveCount() (int64, error) {
	start := time.Now()
	var count int64
	
	err := config.DB.Model(&models.APIKey{}).
		Where("status = ? AND (expires_at IS NULL OR expires_at > ?)", "active", time.Now()).
		Count(&count).Error
	
	config.RecordQueryMetrics("count", "api_keys", time.Since(start), err)
	return count, err
}

func (r *APIKeyRepository) GetExpiredKeys() ([]models.APIKey, error) {
	start := time.Now()
	var apiKeys []models.APIKey
	
	err := config.DB.
		Select("id, user_id, expires_at").
		Where("status = ? AND expires_at IS NOT NULL AND expires_at <= ?", "active", time.Now()).
		Find(&apiKeys).Error
	
	config.RecordQueryMetrics("select", "api_keys", time.Since(start), err)
	return apiKeys, err
}

func (r *APIKeyRepository) BulkUpdateStatus(ids []uuid.UUID, status string) error {
	start := time.Now()
	
	err := config.DB.Model(&models.APIKey{}).
		Where("id IN ?", ids).
		Update("status", status).Error
	
	config.RecordQueryMetrics("bulk_update", "api_keys", time.Since(start), err)
	return err
}