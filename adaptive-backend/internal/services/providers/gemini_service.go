package providers

import (
	"adaptive-backend/internal/models"
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/openai/openai-go/packages/param"
	"google.golang.org/genai"
)

// GeminiService handles Gemini API interactions.
type GeminiService struct {
	client *genai.Client
}

// NewGeminiService initializes the GeminiService with the provided API key.
func NewGeminiService() (*GeminiService, error) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("GOOGLE_API_KEY environment variable not set")
	}
	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create genai client: %w", err)
	}
	return &GeminiService{client: client}, nil
}

// CreateChatCompletion calls Gemini’s non‐streaming API.
func (s *GeminiService) CreateChatCompletion(
	req *models.ProviderChatCompletionRequest,
) (*models.ChatCompletionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}
	ctx := context.Background()
	parts := make([]*genai.Part, len(req.Messages))
	for i, msg := range req.Messages {
		parts[i] = &genai.Part{Text: msg.Content}
	}
	content := &genai.Content{Parts: parts}

	opts := convertGeminiOptions(req)

	resp, err := s.client.Models.GenerateContent(
		ctx,
		req.Model,
		[]*genai.Content{content},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("gemini chat completion failed: %w", err)
	}
	return &models.ChatCompletionResponse{
		Provider: "gemini",
		Response: resp,
	}, nil
}

// StreamChatCompletion calls Gemini’s streaming API.
func (s *GeminiService) StreamChatCompletion(
	req *models.ProviderChatCompletionRequest,
) (*models.ChatCompletionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}
	ctx := context.Background()
	parts := make([]*genai.Part, len(req.Messages))
	for i, msg := range req.Messages {
		parts[i] = &genai.Part{Text: msg.Content}
	}
	content := &genai.Content{Parts: parts}

	opts := convertGeminiOptions(req)

	stream := s.client.Models.GenerateContentStream(
		ctx,
		req.Model,
		[]*genai.Content{content},
		opts,
	)
	return &models.ChatCompletionResponse{
		Provider: "gemini",
		Response: stream,
	}, nil
}

// convertGeminiOptions maps your internal request into Gemini’s GenerateContentConfig.
func convertGeminiOptions(
	req *models.ProviderChatCompletionRequest,
) *genai.GenerateContentConfig {
	cfg := &genai.GenerateContentConfig{
		Temperature:     &req.Temperature,     // *float32
		TopP:            &req.TopP,            // *float32
		MaxOutputTokens: int32(req.MaxTokens), // int32
	}

	// Map OpenAI-style union for response_format → Gemini fields
	if u := req.ResponseFormat; u != nil {
		switch {
		case u.OfText != nil && !param.IsOmitted(u.OfText):
			cfg.ResponseMIMEType = "text/plain"

		case u.OfJSONObject != nil && !param.IsOmitted(u.OfJSONObject):
			cfg.ResponseMIMEType = "application/json"

		case u.OfJSONSchema != nil && !param.IsOmitted(u.OfJSONSchema):
			// Set JSON output
			cfg.ResponseMIMEType = "application/json"

			// Now convert the OpenAI JSON‐Schema param into a genai.Schema
			// by marshaling then unmarshaling it:
			schemaParam := u.OfJSONSchema.JSONSchema
			raw, err := json.Marshal(schemaParam)
			if err != nil {
				fmt.Printf("Error marshaling JSON schema: %v\n", err)
				break
			}

			var gs genai.Schema
			if err2 := json.Unmarshal(raw, &gs); err2 != nil {
				fmt.Printf("Error unmarshaling JSON schema into genai.Schema: %v\n", err2)
				break
			}

			cfg.ResponseSchema = &gs
		}
	}

	return cfg
}
