package conversations

import (
	"adaptive-backend/internal/models"
	repositories "adaptive-backend/internal/repositories/conversations"
)

// MessageService handles business logic for messages
type MessageService struct {
	messageRepo      *repositories.MessageRepository
	conversationRepo *repositories.ConversationRepository
}

// NewMessageService creates a new message service
func NewMessageService() *MessageService {
	return &MessageService{
		messageRepo:      repositories.NewMessageRepository(),
		conversationRepo: repositories.NewConversationRepository(),
	}
}

// CreateMessage adds a new message to a conversation
func (s *MessageService) CreateMessage(conversationID uint, role, content string) (*models.DBMessage, error) {
	// First check if the conversation exists
	_, err := s.conversationRepo.GetByID(conversationID)
	if err != nil {
		return nil, err
	}

	message := &models.DBMessage{
		ConversationID: conversationID,
		Role:           role,
		Content:        content,
	}

	err = s.messageRepo.Create(message)
	if err != nil {
		return nil, err
	}

	return message, nil
}

// GetMessagesByConversation returns all messages for a conversation
func (s *MessageService) GetMessagesByConversation(conversationID uint) ([]models.DBMessage, error) {
	return s.messageRepo.GetByConversationID(conversationID)
}

// GetMessage returns a message by ID
func (s *MessageService) GetMessage(id uint) (*models.DBMessage, error) {
	message, err := s.messageRepo.GetByID(id)
	if err != nil {
		return nil, err
	}

	return &message, nil
}

// UpdateMessage updates a message
func (s *MessageService) UpdateMessage(id uint, role, content string) (*models.DBMessage, error) {
	message, err := s.messageRepo.GetByID(id)
	if err != nil {
		return nil, err
	}

	message.Role = role
	message.Content = content

	err = s.messageRepo.Update(&message)
	if err != nil {
		return nil, err
	}

	return &message, nil
}

// DeleteMessage deletes a message
func (s *MessageService) DeleteMessage(id uint) error {
	return s.messageRepo.Delete(id)
}

func (s *MessageService) DeleteAllMessages(conversationID uint) error {
	return s.messageRepo.DeleteAllByConversationId(conversationID)
}

func (s *MessageService) BatchDeleteMessages(ids []uint) error {
	return s.messageRepo.BatchDelete(ids)
}
