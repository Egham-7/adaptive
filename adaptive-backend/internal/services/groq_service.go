package services

import (
	"adaptive-backend/internal/models"
	"context"
	"fmt"
	"os"

	"github.com/conneroisu/groq-go"
)

// GroqService handles Groq API interactions
type GroqService struct {
	client *groq.Client
}

// NewGroqService creates a new Groq service
func NewGroqService() (*GroqService, error) {
	apiKey := os.Getenv("GROQ_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("GROQ_API_KEY environment variable not set")
	}

	client, err := groq.NewClient(apiKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create groq client: %w", err)
	}

	return &GroqService{client: client}, nil
}

// CreateChatCompletion processes a chat completion request with Groq
func (s *GroqService) CreateChatCompletion(req *models.ProviderChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Convert our unified message format to Groq's format
	messages := make([]groq.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = convertToGroqMessage(msg)
	}

	// Determine which model to use
	model := determineGroqModel(req.Model)

	groqReq := groq.ChatCompletionRequest{
		Model:            model,
		Messages:         messages,
		Temperature:      req.Temperature,
		TopP:             req.TopP,
		MaxTokens:        req.MaxTokens,
		PresencePenalty:  req.PresencePenalty,
		FrequencyPenalty: req.FrequencyPenalty,
	}

	chatCompletionsResponse, err := s.client.ChatCompletion(context.Background(), groqReq)
	if err != nil {
		return &models.ChatCompletionResponse{
			Provider: "groq",
			Error:    err.Error(),
		}, fmt.Errorf("groq chat completion failed: %w", err)
	}

	return &models.ChatCompletionResponse{
		Provider: "groq",
		Response: chatCompletionsResponse,
	}, nil
}

// convertToGroqMessage converts our message model to Groq's format
func convertToGroqMessage(msg models.Message) groq.ChatCompletionMessage {
	role := convertRole(msg.Role)

	var multiContent []groq.ChatMessagePart
	if len(msg.MultiContent) > 0 {
		multiContent = make([]groq.ChatMessagePart, len(msg.MultiContent))
		for i, part := range msg.MultiContent {
			multiContent[i] = convertToGroqMessagePart(part)
		}
	}

	return groq.ChatCompletionMessage{
		Role:         role,
		Content:      msg.Content,
		MultiContent: multiContent,
		FunctionCall: msg.FunctionCall,
		ToolCalls:    msg.ToolCalls,
		ToolCallID:   msg.ToolCallID,
	}
}

// convertRole maps string roles to Groq role types
func convertRole(role string) groq.Role {
	switch role {
	case "user":
		return groq.RoleUser
	case "assistant":
		return groq.RoleAssistant
	case "system":
		return groq.RoleSystem
	default:
		return groq.RoleUser // Default to user if unknown
	}
}

// convertToGroqMessagePart converts our content part to Groq's format
func convertToGroqMessagePart(part models.ChatMessagePart) groq.ChatMessagePart {
	messagePart := groq.ChatMessagePart{
		Text: part.Text,
		Type: groq.ChatMessagePartType(part.Type),
	}

	// Only set ImageURL if it's valid
	if part.ImageURL != nil && part.ImageURL.URL != "" {
		messagePart.ImageURL = &groq.ChatMessageImageURL{
			URL:    part.ImageURL.URL,
			Detail: groq.ImageURLDetail(part.ImageURL.Detail),
		}
	}

	return messagePart
}

func determineGroqModel(requestedModel string) groq.ChatModel {
	if requestedModel == "" {
		return groq.ModelLlama3Groq70B8192ToolUsePreview // Default model
	}

	switch requestedModel {
	case "llama-3.1-70b", "llama3-1-70b":
		return groq.ModelLlama3170BVersatile
	case "llama-3.1-8b", "llama-3-8b", "llama3-1-8b":
		return groq.ModelLlama318BInstant
	case "llama-3.2-11b-vision", "llama3-2-11b-vision":
		return groq.ModelLlama3211BVisionPreview
	case "llama-3.2-1b", "llama3-2-1b":
		return groq.ModelLlama321BPreview
	case "llama-3.2-3b", "llama3-2-3b":
		return groq.ModelLlama323BPreview
	case "llama-3.2-90b-vision", "llama3-2-90b-vision":
		return groq.ModelLlama3290BVisionPreview
	case "llama-3.3-70b-specdec", "llama3-3-70b-specdec":
		return groq.ModelLlama3370BSpecdec
	case "llama-3.3-70b", "llama3-3-70b":
		return groq.ModelLlama3370BVersatile
	case "llama3-70b":
		return groq.ModelLlama370B8192
	case "llama3-8b":
		return groq.ModelLlama38B8192
	case "llama3-70b-tool-use", "llama3-70b-tools":
		return groq.ModelLlama3Groq70B8192ToolUsePreview
	case "llama3-8b-tool-use", "llama3-8b-tools":
		return groq.ModelLlama3Groq8B8192ToolUsePreview
	case "mixtral-8x7b", "mixtral":
		return groq.ModelMixtral8X7B32768
	case "gemma-7b":
		return groq.ModelGemma7BIt
	case "gemma-2-9b", "gemma2-9b":
		return groq.ModelGemma29BIt
	default:
		return groq.ModelLlama3Groq70B8192ToolUsePreview // Default for unknown models
	}
}
