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
