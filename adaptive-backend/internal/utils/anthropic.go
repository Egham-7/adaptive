package utils

import (
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
)

// ExtractPromptFromAnthropicMessages extracts the prompt text from Anthropic MessageParam array for routing decisions
// Gets the last user message with text content
func ExtractPromptFromAnthropicMessages(messages []anthropic.MessageParam) (string, error) {
	if len(messages) == 0 {
		return "", fmt.Errorf("no messages found in request")
	}

	// Iterate backwards to find the last user message
	for i := len(messages) - 1; i >= 0; i-- {
		message := messages[i]

		if message.Role == anthropic.MessageParamRoleUser {
			// Extract text content from the message content blocks
			for _, contentBlock := range message.Content {
				// Check all possible content block types
				if contentBlock.OfText != nil && contentBlock.OfText.Text != "" {
					return contentBlock.OfText.Text, nil
				}
				if contentBlock.OfImage != nil {
					// For image blocks, we could return a description or skip
					continue
				}
				if contentBlock.OfDocument != nil {
					// For document blocks, we could extract text or skip
					continue
				}
				if contentBlock.OfToolResult != nil && len(contentBlock.OfToolResult.Content) > 0 {
					// Tool result content is an array, extract text from first block
					for _, resultContent := range contentBlock.OfToolResult.Content {
						if resultContent.OfText != nil && resultContent.OfText.Text != "" {
							return resultContent.OfText.Text, nil
						}
					}
				}
				if contentBlock.OfToolUse != nil {
					// Tool use blocks don't typically contain routing text
					continue
				}
			}
		}
	}

	return "", fmt.Errorf("no user message with text content found for routing")
}
