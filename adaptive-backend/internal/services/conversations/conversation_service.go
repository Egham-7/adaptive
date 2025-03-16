package conversations

import (
	"adaptive-backend/internal/models"
	repositories "adaptive-backend/internal/repositories/conversations"
)

// ConversationService handles business logic for conversations
type ConversationService struct {
	repo *repositories.ConversationRepository
}

// NewConversationService creates a new conversation service
func NewConversationService() *ConversationService {
	return &ConversationService{
		repo: repositories.NewConversationRepository(),
	}
}

// CreateConversation creates a new conversation
func (s *ConversationService) CreateConversation(title string) (*models.Conversation, error) {
	conversation := &models.Conversation{
		Title: title,
	}

	err := s.repo.Create(conversation)
	if err != nil {
		return nil, err
	}

	return conversation, nil
}

// GetAllConversations returns all conversations
func (s *ConversationService) GetAllConversations() ([]models.Conversation, error) {
	return s.repo.GetAll()
}

// GetConversation returns a conversation by ID
func (s *ConversationService) GetConversation(id uint) (*models.Conversation, error) {
	conversation, err := s.repo.GetByID(id)
	if err != nil {
		return nil, err
	}

	return &conversation, nil
}

// UpdateConversation updates a conversation
func (s *ConversationService) UpdateConversation(id uint, title string) (*models.Conversation, error) {
	conversation, err := s.repo.GetByID(id)
	if err != nil {
		return nil, err
	}

	conversation.Title = title

	err = s.repo.Update(&conversation)
	if err != nil {
		return nil, err
	}

	return &conversation, nil
}

// DeleteConversation deletes a conversation
func (s *ConversationService) DeleteConversation(id uint) error {
	return s.repo.Delete(id)
}
