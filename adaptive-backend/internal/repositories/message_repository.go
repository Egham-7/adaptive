package repositories

import (
	"adaptive-backend/config"
	"adaptive-backend/internal/models"
)

// MessageRepository handles database operations for messages
type MessageRepository struct{}

// NewMessageRepository creates a new message repository
func NewMessageRepository() *MessageRepository {
	return &MessageRepository{}
}

// Create adds a new message to a conversation
func (r *MessageRepository) Create(message *models.DBMessage) error {
	return config.DB.Create(message).Error
}

// GetByConversationID returns all messages for a conversation
func (r *MessageRepository) GetByConversationID(conversationID uint) ([]models.DBMessage, error) {
	var messages []models.DBMessage
	err := config.DB.Where("conversation_id = ?", conversationID).Order("created_at").Find(&messages).Error
	return messages, err
}

// GetByID returns a message by ID
func (r *MessageRepository) GetByID(id uint) (models.DBMessage, error) {
	var message models.DBMessage
	err := config.DB.First(&message, id).Error
	return message, err
}

// Update updates a message
func (r *MessageRepository) Update(message *models.DBMessage) error {
	return config.DB.Save(message).Error
}

// Delete removes a message
func (r *MessageRepository) Delete(id uint) error {
	return config.DB.Delete(&models.DBMessage{}, id).Error
}

func (r *MessageRepository) DeleteByConversationId(id uint) error {
	err := config.DB.Where("conversation_id = ?", id).Delete(&models.DBMessage{}).Error

	return err
}
