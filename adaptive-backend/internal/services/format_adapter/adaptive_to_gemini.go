package format_adapter

import (
	"fmt"

	"adaptive-backend/internal/models"

	"google.golang.org/genai"
)

// AdaptiveToGeminiConverter handles conversion from our adaptive types to pure Gemini types
type AdaptiveToGeminiConverter struct{}

// convertUsageMetadata converts genai.GenerateContentResponseUsageMetadata to our AdaptiveGeminiUsage
func (c *AdaptiveToGeminiConverter) convertUsageMetadata(usage *genai.GenerateContentResponseUsageMetadata) *models.AdaptiveGeminiUsage {
	if usage == nil {
		return nil
	}

	return &models.AdaptiveGeminiUsage{
		CacheTokensDetails:         usage.CacheTokensDetails,
		CachedContentTokenCount:    usage.CachedContentTokenCount,
		CandidatesTokenCount:       usage.CandidatesTokenCount,
		CandidatesTokensDetails:    usage.CandidatesTokensDetails,
		PromptTokenCount:           usage.PromptTokenCount,
		PromptTokensDetails:        usage.PromptTokensDetails,
		ThoughtsTokenCount:         usage.ThoughtsTokenCount,
		ToolUsePromptTokenCount:    usage.ToolUsePromptTokenCount,
		ToolUsePromptTokensDetails: usage.ToolUsePromptTokensDetails,
		TotalTokenCount:            usage.TotalTokenCount,
		TrafficType:                usage.TrafficType,
	}
}

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
		UsageMetadata:  c.convertUsageMetadata(resp.UsageMetadata),
		Provider:       provider,
	}, nil
}
