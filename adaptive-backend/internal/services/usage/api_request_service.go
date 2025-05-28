package usage

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/repositories/usage"
	"errors"
	"time"

	"github.com/google/uuid"
)

type APIRequestService struct {
	repo *usage.APIRequestRepository
}

func NewAPIRequestService() *APIRequestService {
	return &APIRequestService{
		repo: usage.NewAPIRequestRepository(),
	}
}

// CreateAPIRequest creates a new API request record
func (s *APIRequestService) CreateAPIRequest(request *models.APIRequest) error {
	if request == nil {
		return errors.New("api request cannot be nil")
	}

	// Set creation timestamp if not already set
	if request.CreatedAt.IsZero() {
		request.CreatedAt = time.Now()
	}

	return s.repo.Create(request)
}

// GetAPIRequestById retrieves an API request by ID
func (s *APIRequestService) GetAPIRequestById(id uuid.UUID) (models.APIRequest, error) {
	if id == uuid.Nil {
		return models.APIRequest{}, errors.New("invalid request ID")
	}

	return s.repo.GetById(id)
}

// GetUserAPIRequests retrieves all API requests for a user
func (s *APIRequestService) GetUserAPIRequests(userId uuid.UUID) ([]models.APIRequest, error) {
	if userId == uuid.Nil {
		return nil, errors.New("invalid user ID")
	}

	return s.repo.GetAllByUserId(userId)
}

// GetAPIKeyRequests retrieves all API requests for a specific API key
func (s *APIRequestService) GetAPIKeyRequests(apiKeyId uuid.UUID) ([]models.APIRequest, error) {
	if apiKeyId == uuid.Nil {
		return nil, errors.New("invalid API key ID")
	}

	return s.repo.GetAllByAPIKeyId(apiKeyId)
}

// GetProviderRequests retrieves all API requests for a specific provider
func (s *APIRequestService) GetProviderRequests(providerName string) ([]models.APIRequest, error) {
	if providerName == "" {
		return nil, errors.New("provider name cannot be empty")
	}

	return s.repo.GetByProvider(providerName)
}

// GetRequestsByDateRange retrieves API requests within a date range for a user
func (s *APIRequestService) GetRequestsByDateRange(userId uuid.UUID, startDate, endDate time.Time) ([]models.APIRequest, error) {
	if userId == uuid.Nil {
		return nil, errors.New("invalid user ID")
	}

	if startDate.After(endDate) {
		return nil, errors.New("start date cannot be after end date")
	}

	return s.repo.GetByDateRange(userId, startDate, endDate)
}

// GetRequestsByStatus retrieves API requests by status
func (s *APIRequestService) GetRequestsByStatus(status string) ([]models.APIRequest, error) {
	if status == "" {
		return nil, errors.New("status cannot be empty")
	}

	return s.repo.GetByStatus(status)
}

// GetRequestByRequestId retrieves an API request by request ID
func (s *APIRequestService) GetRequestByRequestId(requestId string) (models.APIRequest, error) {
	if requestId == "" {
		return models.APIRequest{}, errors.New("request ID cannot be empty")
	}

	return s.repo.GetByRequestId(requestId)
}

// UpdateAPIRequest updates an existing API request
func (s *APIRequestService) UpdateAPIRequest(request *models.APIRequest) error {
	if request == nil {
		return errors.New("api request cannot be nil")
	}

	// Update the modification timestamp
	request.UpdatedAt = time.Now()

	return s.repo.Update(request)
}

// DeleteAPIRequest deletes an API request by ID
func (s *APIRequestService) DeleteAPIRequest(id uuid.UUID) error {
	if id == uuid.Nil {
		return errors.New("invalid request ID")
	}

	return s.repo.Delete(id)
}

// GetUsageStats retrieves usage statistics for a user within a date range
func (s *APIRequestService) GetUsageStats(userId uuid.UUID, startDate, endDate time.Time) (map[string]any, error) {
	if userId == uuid.Nil {
		return nil, errors.New("invalid user ID")
	}

	if startDate.After(endDate) {
		return nil, errors.New("start date cannot be after end date")
	}

	return s.repo.GetUsageStatsByUserId(userId, startDate, endDate)
}

// MarkRequestCompleted marks a request as completed and records latency
func (s *APIRequestService) MarkRequestCompleted(requestId string, latencyMs int) error {
	request, err := s.repo.GetByRequestId(requestId)
	if err != nil {
		return err
	}

	request.Status = "completed"
	request.LatencyMs = latencyMs
	request.UpdatedAt = time.Now()

	return s.repo.Update(&request)
}

// MarkRequestFailed marks a request as failed
func (s *APIRequestService) MarkRequestFailed(requestId string, errorMessage string) error {
	request, err := s.repo.GetByRequestId(requestId)
	if err != nil {
		return err
	}

	request.Status = "failed"
	request.ErrorMessage = errorMessage
	request.UpdatedAt = time.Now()

	return s.repo.Update(&request)
}
