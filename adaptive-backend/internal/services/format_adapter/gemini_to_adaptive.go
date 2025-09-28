package format_adapter

import (
	"fmt"

	"adaptive-backend/internal/models"

	"google.golang.org/genai"
)

// GeminiToAdaptiveConverter handles conversion from pure genai types to our adaptive extended types
type GeminiToAdaptiveConverter struct{}

// ConvertResponse converts pure genai.GenerateContentResponse to our adaptive GeminiGenerateContentResponse
func (c *GeminiToAdaptiveConverter) ConvertResponse(resp *genai.GenerateContentResponse, provider, cacheTier string) (*models.GeminiGenerateContentResponse, error) {
	if resp == nil {
		return nil, fmt.Errorf("genai generate content response cannot be nil")
	}
	usage := c.ConvertUsage(resp.UsageMetadata, cacheTier)
	return &models.GeminiGenerateContentResponse{
		Candidates:     resp.Candidates,
		CreateTime:     resp.CreateTime,
		ModelVersion:   resp.ModelVersion,
		PromptFeedback: resp.PromptFeedback,
		ResponseID:     resp.ResponseID,
		UsageMetadata:  usage,
		Provider:       provider,
	}, nil
}

func (c *GeminiToAdaptiveConverter) ConvertUsage(usage *genai.GenerateContentResponseUsageMetadata, cacheTier string) *models.AdaptiveGeminiUsage {
	adaptiveUsage := &models.AdaptiveGeminiUsage{
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
		CacheTier:                  cacheTier,
	}
	return adaptiveUsage
}

func (c *GeminiToAdaptiveConverter) ConvertGeminiUsage(usage *models.AdaptiveGeminiUsage) *genai.GenerateContentResponseUsageMetadata {
	if usage == nil {
		return nil
	}
	return &genai.GenerateContentResponseUsageMetadata{
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

// ConvertRequest converts our adaptive GeminiGenerateContentResponse back to pure genai.GenerateContentResponse
func (c *GeminiToAdaptiveConverter) ConvertRequest(resp *models.GeminiGenerateContentResponse) (*genai.GenerateContentResponse, error) {
	if resp == nil {
		return nil, fmt.Errorf("adaptive gemini generate response cannot be nil")
	}

	usage := c.ConvertGeminiUsage(resp.UsageMetadata)

	return &genai.GenerateContentResponse{
		Candidates:     resp.Candidates,
		CreateTime:     resp.CreateTime,
		ModelVersion:   resp.ModelVersion,
		PromptFeedback: resp.PromptFeedback,
		ResponseID:     resp.ResponseID,
		UsageMetadata:  usage,
	}, nil
}
