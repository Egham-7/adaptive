package models

import (
	"time"

	"gorm.io/gorm"
)

// Conversation represents a chat conversation
type Conversation struct {
	ID        uint           `json:"id" gorm:"primaryKey"`
	Title     string         `json:"title"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
	DeletedAt gorm.DeletedAt `json:"deleted_at" gorm:"index"`
	Messages  []DBMessage    `json:"messages,omitempty" gorm:"foreignKey:ConversationID"`
}

// DBMessage represents a message in the database
type DBMessage struct {
	ID             uint           `json:"id" gorm:"primaryKey"`
	ConversationID uint           `json:"conversation_id"`
	Role           string         `json:"role"`
	Content        string         `json:"content"`
	Provider       *string        `json:"provider"`
	Model          *string        `json:"model"`
	CreatedAt      time.Time      `json:"created_at"`
	UpdatedAt      time.Time      `json:"updated_at"`
	DeletedAt      gorm.DeletedAt `json:"deleted_at" gorm:"index"`
}
