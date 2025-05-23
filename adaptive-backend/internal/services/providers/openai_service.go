package providers

import (
	"adaptive-backend/internal/models"
	"context"
	"fmt"
	"os"

	"github.com/conneroisu/groq-go/pkg/tools"
	"github.com/sashabaranov/go-openai"
)

// OpenAIService handles OpenAI API interactions
type OpenAIService struct {
	client *openai.Client
}

// NewOpenAIService creates a new OpenAI service
func NewOpenAIService() *OpenAIService {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		// We're not returning an error here to maintain compatibility with the original function signature
		// But it's a good practice to log this error elsewhere
		fmt.Println("Warning: OPENAI_API_KEY environment variable not set")
	}

	return &OpenAIService{
		client: openai.NewClient(apiKey),
	}
}

// StreamChatCompletion processes a streaming chat completion request with OpenAI

func (s *OpenAIService) StreamChatCompletion(req *models.ProviderChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Convert our messages to OpenAI format
	messages := make([]openai.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = convertToOpenAIMessage(msg)
	}

	// Determine which model to use
	model := determineOpenAIModel(req.Model)

	// Create OpenAI request
	openaiReq := openai.ChatCompletionRequest{
		Model:               model,
		Messages:            messages,
		Temperature:         req.Temperature,
		TopP:                req.TopP,
		MaxCompletionTokens: req.MaxTokens,
		FrequencyPenalty:    req.FrequencyPenalty,
		PresencePenalty:     req.PresencePenalty,
	}

	stream, err := s.client.CreateChatCompletionStream(context.Background(), openaiReq)
	if err != nil {
		return &models.ChatCompletionResponse{
			Provider: "openai",
			Error:    err.Error(),
		}, fmt.Errorf("openai streaming chat completion failed: %w", err)
	}

	return &models.ChatCompletionResponse{
		Provider: "openai",
		Response: stream,
	}, nil
}

// CreateChatCompletion processes a chat completion request with OpenAI
func (s *OpenAIService) CreateChatCompletion(req *models.ProviderChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Convert our messages to OpenAI format
	messages := make([]openai.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = convertToOpenAIMessage(msg)
	}

	// Determine which model to use
	model := determineOpenAIModel(req.Model)

	// Create OpenAI request
	openaiReq := openai.ChatCompletionRequest{
		Model:               model,
		Messages:            messages,
		Temperature:         req.Temperature,
		TopP:                req.TopP,
		MaxCompletionTokens: req.MaxTokens,
		FrequencyPenalty:    req.FrequencyPenalty,
		PresencePenalty:     req.PresencePenalty,
	}

	// Call OpenAI API
	resp, err := s.client.CreateChatCompletion(context.Background(), openaiReq)
	if err != nil {
		return &models.ChatCompletionResponse{
			Provider: "openai",
			Error:    err.Error(),
		}, fmt.Errorf("openai chat completion failed: %w", err)
	}

	// Return the content from the response
	return &models.ChatCompletionResponse{
		Provider: "openai",
		Response: resp,
	}, nil
}

// convertToOpenAIMessage converts our message model to OpenAI's format
func convertToOpenAIMessage(msg models.Message) openai.ChatCompletionMessage {
	// Convert the role
	role := msg.Role

	// Handle messages with multiple content parts (like images)
	if len(msg.MultiContent) > 0 {
		var multiContent []openai.ChatMessagePart
		for _, part := range msg.MultiContent {
			multiContent = append(multiContent, convertToOpenAIMessagePart(part))
		}

		return openai.ChatCompletionMessage{
			Role:         role,
			MultiContent: multiContent,
			FunctionCall: convertFunctionCall(msg.FunctionCall),
			ToolCalls:    convertToolCalls(msg.ToolCalls),
			ToolCallID:   msg.ToolCallID,
		}
	}

	// Handle standard text-only messages
	return openai.ChatCompletionMessage{
		Role:         role,
		Content:      msg.Content,
		FunctionCall: convertFunctionCall(msg.FunctionCall),
		ToolCalls:    convertToolCalls(msg.ToolCalls),
		ToolCallID:   msg.ToolCallID,
	}
}

// convertToOpenAIMessagePart converts our content part to OpenAI's format
func convertToOpenAIMessagePart(part models.ChatMessagePart) openai.ChatMessagePart {
	messagePart := openai.ChatMessagePart{
		Type: openai.ChatMessagePartType(part.Type),
		Text: part.Text,
	}

	// Only set ImageURL if it's valid
	if part.ImageURL != nil && part.ImageURL.URL != "" {
		messagePart.ImageURL = &openai.ChatMessageImageURL{
			URL:    part.ImageURL.URL,
			Detail: openai.ImageURLDetail(part.ImageURL.Detail),
		}
	}

	return messagePart
}

// convertFunctionCall converts our function call to OpenAI's format
func convertFunctionCall(functionCall *tools.FunctionCall) *openai.FunctionCall {
	if functionCall == nil {
		return nil
	}

	return &openai.FunctionCall{
		Name:      functionCall.Name,
		Arguments: functionCall.Arguments,
	}
}

// convertToolCalls converts our tool calls to OpenAI's format
func convertToolCalls(toolCalls []tools.ToolCall) []openai.ToolCall {
	if toolCalls == nil {
		return nil
	}

	result := make([]openai.ToolCall, len(toolCalls))
	for i, tc := range toolCalls {
		result[i] = openai.ToolCall{
			ID:   tc.ID,
			Type: openai.ToolType(tc.Type),
			Function: openai.FunctionCall{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		}
	}

	return result
}

func determineOpenAIModel(requestedModel string) string {
	if requestedModel == "" {
		return openai.GPT4oMini // Default model
	}

	// Map of supported models - one entry per model family
	supportedModels := map[string]bool{
		// O1 models
		openai.O1:     true,
		openai.O1Mini: true,

		// O3 Models
		openai.O3Mini: true,

		// GPT-4 models
		openai.GPT4o:     true,
		openai.GPT4oMini: true,
		openai.GPT4Turbo: true,

		// GPT-3.5 models
		openai.GPT3Dot5Turbo: true,
	}

	// If the requested model is directly supported, use it
	if _, ok := supportedModels[requestedModel]; ok {
		return requestedModel
	}

	// For unknown models, fall back to default
	return openai.GPT4oMini
}
