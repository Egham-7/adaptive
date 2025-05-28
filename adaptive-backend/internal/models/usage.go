package models

import (
	"time"

	"github.com/google/uuid"
)

type APIKey struct {
	ID         uuid.UUID  `json:"id" gorm:"type:uniqueidentifier;primary_key"`
	UserID     string     `json:"user_id" gorm:"type:uniqueidentifier;not null"`
	Name       string     `json:"name" gorm:"type:varchar(255);not null"`
	KeyPrefix  string     `json:"key_prefix" gorm:"type:varchar(10);not null"`
	KeyHash    string     `json:"key_hash" gorm:"type:varchar(255);not null"`
	Status     string     `json:"status" gorm:"type:varchar(20);not null"`
	CreatedAt  time.Time  `json:"created_at" gorm:"not null;default:CURRENT_TIMESTAMP"`
	ExpiresAt  *time.Time `json:"expires_at" gorm:"null"`
	LastUsedAt *time.Time `json:"last_used_at" gorm:"null"`
}

type APIRequest struct {
	ID           uuid.UUID `json:"id" gorm:"primaryKey;type:uniqueidentifier"`
	UserID       uuid.UUID `json:"user_id" gorm:"type:uniqueidentifier;not null"`
	APIKey       string    `json:"api_key_id" gorm:"type:text;not null"`
	ProviderName string    `json:"provider_name" gorm:"not null"`
	ModelName    string    `json:"model_name" gorm:"not null"`
	RequestType  string    `json:"request_type" gorm:"not null"`
	Status       string    `json:"status" gorm:"not null"`
	LatencyMs    int       `json:"latency_ms" gorm:"not null"`
	ErrorMessage string    `json:"error_message" gorm:"type:text;null"`
	RequestID    string    `json:"request_id" gorm:"not null"`
	CreatedAt    time.Time `json:"created_at" gorm:"not null;default:CURRENT_TIMESTAMP"`
	UpdatedAt    time.Time `json:"updated_at" gorm:"not null;default:CURRENT_TIMESTAMP"`
	Metadata     string    `json:"metadata" gorm:"type:text;serializer:json"`
}

type ModelInfo struct {
	ID                         uuid.UUID `json:"id" gorm:"primaryKey;type:uniqueidentifier"`
	ProviderName               string    `json:"provider_name" gorm:"not null"`
	ModelName                  string    `json:"model_name" gorm:"type:varchar(255);not null"`
	IsEnabled                  bool      `json:"is_enabled" gorm:"not null;default:true"`
	CostPerMillionInputTokens  float64   `json:"cost_per_million_input_tokens" gorm:"not null"`
	CostPerMillionOutputTokens float64   `json:"cost_per_million_output_tokens" gorm:"not null"`
	CreatedAt                  time.Time `json:"created_at" gorm:"not null;default:CURRENT_TIMESTAMP"`
	UpdatedAt                  time.Time `json:"updated_at" gorm:"not null;default:CURRENT_TIMESTAMP"`
}

type TokenUsage struct {
	ID               uuid.UUID `json:"id" gorm:"primaryKey;type:uniqueidentifier"`
	APIRequestID     uuid.UUID `json:"api_request_id" gorm:"type:uniqueidentifier;not null"`
	PromptTokens     int       `json:"prompt_tokens" gorm:"not null;default:0"`
	CompletionTokens int       `json:"completion_tokens" gorm:"not null;default:0"`
	TotalTokens      int       `json:"total_tokens" gorm:"not null;default:0"`
	CreatedAt        time.Time `json:"created_at" gorm:"not null;default:CURRENT_TIMESTAMP"`
}

type CostEntry struct {
	ID             uuid.UUID `json:"id" gorm:"primaryKey;type:uniqueidentifier"`
	APIRequestID   uuid.UUID `json:"api_request_id" gorm:"type:uniqueidentifier;not null"`
	PromptCost     float64   `json:"prompt_cost" gorm:"not null;default:0"`
	CompletionCost float64   `json:"completion_cost" gorm:"not null;default:0"`
	TotalCost      float64   `json:"total_cost" gorm:"not null;default:0"`
	OriginalCost   float64   `json:"original_cost" gorm:"not null;default:0"`
	Savings        float64   `json:"savings" gorm:"not null;default:0"`
	Currency       string    `json:"currency" gorm:"not null;default:'USD'"`
	CreatedAt      time.Time `json:"created_at" gorm:"not null;default:CURRENT_TIMESTAMP"`
}
