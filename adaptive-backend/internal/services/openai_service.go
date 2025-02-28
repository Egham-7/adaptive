package services

import (
	"adaptive-backend/internal/models"
	"context"
	"github.com/sashabaranov/go-openai"
	"os"
)

// OpenAIService handles OpenAI API interactions
type OpenAIService struct {
	client *openai.Client
}

// NewOpenAIService creates a new OpenAI service
func NewOpenAIService() *OpenAIService {
	return &OpenAIService{
		client: openai.NewClient(os.Getenv("OPENAI_API_KEY")),
	}
}

// CreateChatCompletion processes a chat completion request with OpenAI
func (s *OpenAIService) CreateChatCompletion(req *models.ChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	// Convert our unified message format to OpenAI's format
	messages := make([]openai.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	// Default to GPT-3.5 if no model is specified
	model := openai.GPT3Dot5Turbo
	if req.Model != "" {
		model = req.Model
	}

	// Create OpenAI request
	openaiReq := openai.ChatCompletionRequest{
		Model:     model,
		Messages:  messages,
		MaxTokens: req.MaxTokens,
	}

	// Call OpenAI API
	resp, err := s.client.CreateChatCompletion(context.Background(), openaiReq)
	if err != nil {
		return &models.ChatCompletionResponse{
			Provider: "openai",
			Error:    err.Error(),
		}, err
	}

	// Return the content from the response
	return &models.ChatCompletionResponse{
		Provider: "openai",
		Response: resp,
	}, nil
}
