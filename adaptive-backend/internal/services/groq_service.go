package services

import (
	"adaptive-backend/internal/models"
	"context"
	"github.com/conneroisu/groq-go"
	"os"
)

// GroqService handles Groq API interactions
type GroqService struct {
	client *groq.Client
}

// NewGroqService creates a new Groq service
func NewGroqService() (*GroqService, error) {
	client, err := groq.NewClient(os.Getenv("GROQ_KEY"))
	if err != nil {
		return nil, err
	}
	return &GroqService{client: client}, nil
}

// CreateChatCompletion processes a chat completion request with Groq
func (s *GroqService) CreateChatCompletion(req *models.ChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	// Convert our unified message format to Groq's format
	messages := make([]groq.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		// Convert string role to groq.Role type
		var role groq.Role
		switch msg.Role {
		case "user":
			role = groq.RoleUser
		case "assistant":
			role = groq.RoleAssistant
		case "system":
			role = groq.RoleSystem
		default:
			role = groq.RoleUser // Default to user if unknown
		}

		messages[i] = groq.ChatCompletionMessage{
			Role:    role,
			Content: msg.Content,
		}
	}

	// Default to Llama-3 if no model is specified
	var model groq.ChatModel
	if req.Model != "" {
		switch req.Model {
		case "llama-3.1-70b":
			model = groq.ModelLlama3Groq70B8192ToolUsePreview
		default:
			model = groq.ModelLlama3Groq70B8192ToolUsePreview // Default model
		}
	} else {
		model = groq.ModelLlama3Groq70B8192ToolUsePreview
	}

	groqReq := groq.ChatCompletionRequest{
		Model:     model,
		Messages:  messages,
		MaxTokens: req.MaxTokens,
	}

	chatCompletionsResponse, err := s.client.ChatCompletion(context.Background(), groqReq)
	if err != nil {
		return &models.ChatCompletionResponse{
			Provider: "groq",
			Error:    err.Error(),
		}, err
	}

	return &models.ChatCompletionResponse{
		Provider: "groq",
		Response: chatCompletionsResponse,
	}, nil
}

