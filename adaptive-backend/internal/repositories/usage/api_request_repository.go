package usage

import (
	"adaptive-backend/config"
	"adaptive-backend/internal/models"
	"time"

	"github.com/google/uuid"
)

type APIRequestRepository struct{}

func NewAPIRequestRepository() *APIRequestRepository {
	return &APIRequestRepository{}
}

func (*APIRequestRepository) Create(apiRequest *models.APIRequest) error {
	return config.DB.Create(apiRequest).Error
}

func (*APIRequestRepository) GetById(id uuid.UUID) (models.APIRequest, error) {
	var request models.APIRequest
	err := config.DB.First(&request, id).Error
	return request, err
}

func (*APIRequestRepository) GetAllByUserId(userId uuid.UUID) ([]models.APIRequest, error) {
	var requests []models.APIRequest
	err := config.DB.Where("user_id = ?", userId).Order("created_at DESC").Find(&requests).Error
	return requests, err
}

func (*APIRequestRepository) GetAllByAPIKeyId(apiKeyId uuid.UUID) ([]models.APIRequest, error) {
	var requests []models.APIRequest
	err := config.DB.Where("api_key_id = ?", apiKeyId).Order("created_at DESC").Find(&requests).Error
	return requests, err
}

func (*APIRequestRepository) GetByProvider(providerName string) ([]models.APIRequest, error) {
	var requests []models.APIRequest
	err := config.DB.Where("provider_name = ?", providerName).Order("created_at DESC").Find(&requests).Error
	return requests, err
}

func (*APIRequestRepository) GetByDateRange(userId uuid.UUID, startDate, endDate time.Time) ([]models.APIRequest, error) {
	var requests []models.APIRequest
	err := config.DB.Where("user_id = ? AND created_at BETWEEN ? AND ?", userId, startDate, endDate).
		Order("created_at DESC").Find(&requests).Error
	return requests, err
}

func (*APIRequestRepository) GetByStatus(status string) ([]models.APIRequest, error) {
	var requests []models.APIRequest
	err := config.DB.Where("status = ?", status).Order("created_at DESC").Find(&requests).Error
	return requests, err
}

func (*APIRequestRepository) GetByRequestId(requestId string) (models.APIRequest, error) {
	var request models.APIRequest
	err := config.DB.Where("request_id = ?", requestId).First(&request).Error
	return request, err
}

func (*APIRequestRepository) Update(apiRequest *models.APIRequest) error {
	return config.DB.Save(apiRequest).Error
}

func (*APIRequestRepository) Delete(id uuid.UUID) error {
	return config.DB.Delete(&models.APIRequest{}, id).Error
}

func (*APIRequestRepository) GetUsageStatsByUserId(userId uuid.UUID, startDate, endDate time.Time) (map[string]interface{}, error) {
	var result map[string]any
	err := config.DB.Model(&models.APIRequest{}).
		Select("COUNT(*) as total_requests, AVG(latency_ms) as avg_latency, SUM(latency_ms) as total_latency").
		Where("user_id = ? AND created_at BETWEEN ? AND ?", userId, startDate, endDate).
		Scan(&result).Error
	return result, err
}
