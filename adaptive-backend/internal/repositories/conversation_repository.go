package repositories

import (
	"adaptive-backend/config"
	"adaptive-backend/internal/models"
)

// ConversationRepository handles database operations for conversations
type ConversationRepository struct{}

// NewConversationRepository creates a new conversation repository
func NewConversationRepository() *ConversationRepository {
	return &ConversationRepository{}
}

// Create adds a new conversation
func (r *ConversationRepository) Create(conversation *models.Conversation) error {
	return config.DB.Create(conversation).Error
}

// GetAll returns all conversations
func (r *ConversationRepository) GetAll() ([]models.Conversation, error) {
	var conversations []models.Conversation
	err := config.DB.Find(&conversations).Error
	return conversations, err
}

// GetByID returns a conversation by ID
func (r *ConversationRepository) GetByID(id uint) (models.Conversation, error) {
	var conversation models.Conversation
	err := config.DB.Preload("Messages").First(&conversation, id).Error
	return conversation, err
}

// Update updates a conversation
func (r *ConversationRepository) Update(conversation *models.Conversation) error {
	return config.DB.Save(conversation).Error
}

// Delete removes a conversation
func (r *ConversationRepository) Delete(id uint) error {
	return config.DB.Delete(&models.Conversation{}, id).Error
}
