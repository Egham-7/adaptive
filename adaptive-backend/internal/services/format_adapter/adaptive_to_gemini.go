package format_adapter

import (
	"adaptive-backend/internal/models"
	"fmt"

	"google.golang.org/genai"
)

// AdaptiveToGeminiConverter handles conversion from our adaptive types to pure Gemini types
type AdaptiveToGeminiConverter struct{}

// ConvertResponse converts pure genai response to our adaptive response (adding provider info)
func (c *AdaptiveToGeminiConverter) ConvertResponse(resp *genai.GenerateContentResponse, provider string) (*models.GeminiGenerateContentResponse, error) {
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

