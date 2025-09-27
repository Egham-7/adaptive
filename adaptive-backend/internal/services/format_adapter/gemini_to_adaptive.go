package format_adapter

import (
	"fmt"

	"adaptive-backend/internal/models"

	"google.golang.org/genai"
)

// GeminiToAdaptiveConverter handles conversion from pure genai types to our adaptive extended types
type GeminiToAdaptiveConverter struct{}

// ConvertResponse converts pure genai.GenerateContentResponse to our adaptive GeminiGenerateContentResponse
func (c *GeminiToAdaptiveConverter) ConvertResponse(resp *genai.GenerateContentResponse, provider string) (*models.GeminiGenerateContentResponse, error) {
	if resp == nil {
		return nil, fmt.Errorf("genai generate content response cannot be nil")
	}

	return &models.GeminiGenerateContentResponse{
		Candidates:     resp.Candidates,
		CreateTime:     resp.CreateTime,
		ModelVersion:   resp.ModelVersion,
		PromptFeedback: resp.PromptFeedback,
		ResponseID:     resp.ResponseID,
		UsageMetadata:  resp.UsageMetadata,
		Provider:       provider,
	}, nil
}

// ConvertRequest converts our adaptive GeminiGenerateContentResponse back to pure genai.GenerateContentResponse
func (c *GeminiToAdaptiveConverter) ConvertRequest(resp *models.GeminiGenerateContentResponse) (*genai.GenerateContentResponse, error) {
	if resp == nil {
		return nil, fmt.Errorf("adaptive gemini generate response cannot be nil")
	}

	return &genai.GenerateContentResponse{
		Candidates:     resp.Candidates,
		CreateTime:     resp.CreateTime,
		ModelVersion:   resp.ModelVersion,
		PromptFeedback: resp.PromptFeedback,
		ResponseID:     resp.ResponseID,
		UsageMetadata:  resp.UsageMetadata,
	}, nil
}
