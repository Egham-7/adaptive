package providers

import (
	"adaptive-backend/internal/models"
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/conneroisu/groq-go"
	"github.com/conneroisu/groq-go/pkg/schema"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
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

// convertResponseFormat maps the OpenAI union into a *groq.ChatResponseFormat
func convertResponseFormat(
	u *openai.ChatCompletionNewParamsResponseFormatUnion,
) *groq.ChatResponseFormat {
	if u == nil {
		return nil
	}

	// 1) Text?
	if u.OfText != nil && !param.IsOmitted(u.OfText) {
		return &groq.ChatResponseFormat{Type: groq.FormatText}
	}

	// 2) json_object?
	if u.OfJSONObject != nil && !param.IsOmitted(u.OfJSONObject) {
		return &groq.ChatResponseFormat{Type: groq.FormatJSONObject}
	}

	// 3) json_schema?
	if u.OfJSONSchema != nil && !param.IsOmitted(u.OfJSONSchema) {
		// a) try round‐trip via JSON
		if b, err := json.Marshal(u.OfJSONSchema); err == nil {
			var gf groq.ChatResponseFormat
			if err2 := json.Unmarshal(b, &gf); err2 == nil {
				return &gf
			}
		}
		// b) fallback to manual mapping
		p := u.OfJSONSchema
		return &groq.ChatResponseFormat{
			Type: groq.FormatJSONSchema,
			JSONSchema: &groq.JSONSchema{
				Name:        p.JSONSchema.Name,
				Description: p.JSONSchema.Description.Value,
				// Here we assert that the param’s Schema (type any) matches
				// groq/pkg/schema.Schema
				Schema: p.JSONSchema.Schema.(schema.Schema),
				Strict: p.JSONSchema.Strict.Value,
			},
		}
	}

	return nil
}

func (s *GroqService) StreamChatCompletion(req *models.ProviderChatCompletionRequest) (*models.ChatCompletionResponse, error) {
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
		Stream:           req.Stream,
	}

	stream, err := s.client.ChatCompletionStream(context.Background(), groqReq)
	if err != nil {
		return &models.ChatCompletionResponse{
			Provider: "groq",
			Error:    err.Error(),
		}, fmt.Errorf("groq chat stream completion failed: %w", err)
	}

	return &models.ChatCompletionResponse{
		Response: stream,
		Provider: "groq",
	}, nil
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
		ResponseFormat:   convertResponseFormat(req.ResponseFormat),
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
		return groq.ModelLlama323BPreview
	}

	switch requestedModel {
	case "llama-3.1-70b-versatile":
		return groq.ModelLlama3170BVersatile
	case "llama-3.1-8b-instant":
		return groq.ModelLlama318BInstant
	case "llama-guard-3-8b":
		return groq.ChatModel(groq.ModelLlamaGuard38B)
	case "llama-3.2-1b":
		return groq.ModelLlama321BPreview
	case "llama-3.2-3b":
		return groq.ModelLlama323BPreview
	case "llama-3.2-90b-vision":
		return groq.ModelLlama3290BVisionPreview
	case "llama-3.3-70b-specdec":
		return groq.ModelLlama3370BSpecdec
	case "llama3-70b-8192":
		return groq.ModelLlama370B8192
	case "llama3-8b-8192":
		return groq.ModelLlama38B8192
	case "gemma-7b":
		return groq.ModelGemma7BIt
	case "gemma-2-9b-it":
		return groq.ModelGemma29BIt
	default:
		return groq.ModelLlama323BPreview // Default model
	}
}
