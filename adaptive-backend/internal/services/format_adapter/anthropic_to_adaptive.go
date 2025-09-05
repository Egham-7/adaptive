package format_adapter

import (
	"fmt"

	"adaptive-backend/internal/models"

	"github.com/anthropics/anthropic-sdk-go"
)

// AnthropicToAdaptiveConverter handles conversion from standard Anthropic types to our adaptive types
type AnthropicToAdaptiveConverter struct{}

// ConvertRequest converts standard Anthropic MessageNewParams to our AnthropicMessageRequest
func (c *AnthropicToAdaptiveConverter) ConvertRequest(req *anthropic.MessageNewParams) (*models.AnthropicMessageRequest, error) {
	if req == nil {
		return nil, fmt.Errorf("anthropic message new params cannot be nil")
	}

	// Create our enhanced request with the standard params copied
	return &models.AnthropicMessageRequest{
		MaxTokens:     req.MaxTokens,
		Messages:      req.Messages,
		Model:         req.Model,
		Temperature:   req.Temperature,
		TopK:          req.TopK,
		TopP:          req.TopP,
		Metadata:      req.Metadata,
		ServiceTier:   req.ServiceTier,
		StopSequences: req.StopSequences,
		System:        req.System,
		Thinking:      req.Thinking,
		ToolChoice:    req.ToolChoice,
		Tools:         req.Tools,
		// Custom fields are left as nil/defaults - caller can set them as needed
		ModelRouterConfig:   nil,
		PromptResponseCache: nil,
		PromptCache:         nil,
		Fallback:            nil,
		ProviderConfigs:     nil,
	}, nil
}

// ConvertResponse converts standard Anthropic Message to our AdaptiveAnthropicMessage
func (c *AnthropicToAdaptiveConverter) ConvertResponse(resp *anthropic.Message, provider string) (*models.AnthropicMessage, error) {
	if resp == nil {
		return nil, fmt.Errorf("anthropic message cannot be nil")
	}

	return &models.AnthropicMessage{
		ID:           resp.ID,
		Content:      c.convertContentBlocks(resp.Content),
		Model:        string(resp.Model),
		Role:         string(resp.Role),
		StopReason:   string(resp.StopReason),
		StopSequence: resp.StopSequence,
		Type:         string(resp.Type),
		Usage:        *c.convertUsage(resp.Usage),
		Provider:     provider,
	}, nil
}

// ConvertStreamingChunk converts standard Anthropic MessageStreamEventUnion to our AdaptiveAnthropicMessageChunk
func (c *AnthropicToAdaptiveConverter) ConvertStreamingChunk(chunk *anthropic.MessageStreamEventUnion, provider string) (*models.AnthropicMessageChunk, error) {
	if chunk == nil {
		return nil, fmt.Errorf("anthropic message stream event cannot be nil")
	}

	// Use typed accessors for all event kinds instead of direct field access
	switch eventVariant := chunk.AsAny().(type) {
	case anthropic.MessageStartEvent:
		convertedMessage, err := c.ConvertResponse(&eventVariant.Message, provider)
		if err != nil {
			return nil, fmt.Errorf("failed to convert message in chunk: %w", err)
		}
		return &models.AnthropicMessageChunk{
			Type:     "message_start",
			Message:  convertedMessage,
			Provider: provider,
		}, nil

	case anthropic.MessageDeltaEvent:
		adaptive := &models.AnthropicMessageChunk{
			Type: "message_delta",
			Delta: &models.AdaptiveDelta{
				StopReason:   string(eventVariant.Delta.StopReason),
				StopSequence: eventVariant.Delta.StopSequence,
			},
			Provider: provider,
		}
		if eventVariant.Usage.OutputTokens != 0 || eventVariant.Usage.InputTokens != 0 {
			adaptive.Usage = &models.AdaptiveAnthropicUsage{
				InputTokens:  eventVariant.Usage.InputTokens,
				OutputTokens: eventVariant.Usage.OutputTokens,
			}
		}
		return adaptive, nil

	case anthropic.MessageStopEvent:
		return &models.AnthropicMessageChunk{
			Type:     "message_stop",
			Provider: provider,
		}, nil

	case anthropic.ContentBlockStartEvent:
		return &models.AnthropicMessageChunk{
			Type:         "content_block_start",
			ContentBlock: &eventVariant.ContentBlock,
			Index:        &eventVariant.Index,
			Provider:     provider,
		}, nil

	case anthropic.ContentBlockDeltaEvent:
		adaptive := &models.AnthropicMessageChunk{
			Type: "content_block_delta",
			Delta: &models.AdaptiveDelta{
				Type: eventVariant.Delta.Type,
			},
			Index:    &eventVariant.Index,
			Provider: provider,
		}

		// Populate only the relevant fields based on delta type
		switch eventVariant.Delta.Type {
		case "text_delta":
			adaptive.Delta.Text = eventVariant.Delta.Text
		case "input_json_delta":
			adaptive.Delta.PartialJSON = eventVariant.Delta.PartialJSON
			// Add other delta types as needed
		}
		return adaptive, nil

	case anthropic.ContentBlockStopEvent:
		return &models.AnthropicMessageChunk{
			Type:     "content_block_stop",
			Index:    &eventVariant.Index,
			Provider: provider,
		}, nil

	default:
		// Handle unknown event types gracefully
		return nil, fmt.Errorf("unknown stream event type: %T", eventVariant)
	}
}

// convertUsage creates AdaptiveAnthropicUsage from Anthropic's Usage
func (c *AnthropicToAdaptiveConverter) convertUsage(usage anthropic.Usage) *models.AdaptiveAnthropicUsage {
	return &models.AdaptiveAnthropicUsage{
		CacheCreationInputTokens: usage.CacheCreationInputTokens,
		CacheReadInputTokens:     usage.CacheReadInputTokens,
		InputTokens:              usage.InputTokens,
		OutputTokens:             usage.OutputTokens,
		ServiceTier:              string(usage.ServiceTier),
	}
}

// convertContentBlocks converts Anthropic ContentBlockUnion to our custom ContentBlockUnion
func (c *AnthropicToAdaptiveConverter) convertContentBlocks(content []anthropic.ContentBlockUnion) []models.ContentBlockUnion {
	if content == nil {
		return nil
	}

	result := make([]models.ContentBlockUnion, len(content))
	for i, block := range content {
		switch b := block.AsAny().(type) {
		case anthropic.TextBlock:
			result[i] = models.ContentBlockUnion{
				Type:      "text",
				Text:      b.Text,
				Citations: c.convertTextCitations(b.Citations),
			}
		case anthropic.ThinkingBlock:
			result[i] = models.ContentBlockUnion{
				Type:      "thinking",
				Thinking:  b.Thinking,
				Signature: b.Signature,
			}
		case anthropic.RedactedThinkingBlock:
			result[i] = models.ContentBlockUnion{
				Type: "redacted_thinking",
				Data: b.Data,
			}
		case anthropic.ToolUseBlock:
			result[i] = models.ContentBlockUnion{
				Type:  "tool_use",
				ID:    b.ID,
				Name:  b.Name,
				Input: b.Input,
			}
		case anthropic.ServerToolUseBlock:
			result[i] = models.ContentBlockUnion{
				Type:  "server_tool_use",
				ID:    b.ID,
				Name:  string(b.Name),
				Input: b.Input,
			}
		case anthropic.WebSearchToolResultBlock:
			result[i] = models.ContentBlockUnion{
				Type:      "web_search_tool_result",
				ToolUseID: b.ToolUseID,
				Content:   b.Content,
			}
		default:
			// Fallback for unknown types
			result[i] = models.ContentBlockUnion{
				Type: "text",
				Text: "",
			}
		}
	}
	return result
}

// convertTextCitations converts Anthropic TextCitationUnion to our custom TextCitationUnion
func (c *AnthropicToAdaptiveConverter) convertTextCitations(citations []anthropic.TextCitationUnion) []models.TextCitationUnion {
	if citations == nil {
		return nil
	}

	result := make([]models.TextCitationUnion, len(citations))
	for i, citation := range citations {
		switch c := citation.AsAny().(type) {
		case anthropic.CitationCharLocation:
			result[i] = models.TextCitationUnion{
				Type:           "char_location",
				CitedText:      c.CitedText,
				DocumentIndex:  c.DocumentIndex,
				DocumentTitle:  c.DocumentTitle,
				StartCharIndex: c.StartCharIndex,
				EndCharIndex:   c.EndCharIndex,
				FileID:         c.FileID,
			}
		case anthropic.CitationPageLocation:
			result[i] = models.TextCitationUnion{
				Type:            "page_location",
				CitedText:       c.CitedText,
				DocumentIndex:   c.DocumentIndex,
				DocumentTitle:   c.DocumentTitle,
				StartPageNumber: c.StartPageNumber,
				EndPageNumber:   c.EndPageNumber,
				FileID:          c.FileID,
			}
		case anthropic.CitationContentBlockLocation:
			result[i] = models.TextCitationUnion{
				Type:            "content_block_location",
				CitedText:       c.CitedText,
				DocumentIndex:   c.DocumentIndex,
				DocumentTitle:   c.DocumentTitle,
				StartBlockIndex: c.StartBlockIndex,
				EndBlockIndex:   c.EndBlockIndex,
				FileID:          c.FileID,
			}
		case anthropic.CitationsWebSearchResultLocation:
			result[i] = models.TextCitationUnion{
				Type:           "web_search_result_location",
				CitedText:      c.CitedText,
				EncryptedIndex: c.EncryptedIndex,
				Title:          c.Title,
				URL:            c.URL,
			}
		case anthropic.CitationsSearchResultLocation:
			result[i] = models.TextCitationUnion{
				Type:              "search_result_location",
				CitedText:         c.CitedText,
				SearchResultIndex: c.SearchResultIndex,
				Source:            c.Source,
			}
		default:
			// Fallback for unknown types
			result[i] = models.TextCitationUnion{
				Type:      "char_location",
				CitedText: "",
			}
		}
	}
	return result
}
