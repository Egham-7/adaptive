package utils

import (
	"fmt"
	"strings"

	"github.com/openai/openai-go"
)

// FindLastUserMessage safely finds the last user message in a conversation.
func FindLastUserMessage(messages []openai.ChatCompletionMessageParamUnion) (string, error) {
	if len(messages) == 0 {
		return "", fmt.Errorf("no messages provided")
	}

	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		if msg.OfUser == nil {
			continue
		}

		// Handle string content
		if msg.OfUser.Content.OfString.Value != "" {
			content := msg.OfUser.Content.OfString.Value
			if content != "" {
				return content, nil
			}
		}

		// Handle multi-modal content (text + images)
		if msg.OfUser.Content.OfArrayOfContentParts != nil {
			text := extractTextFromParts(msg.OfUser.Content.OfArrayOfContentParts)
			if text != "" {
				return text, nil
			}
		}
	}

	return "", fmt.Errorf("no user message found")
}

// extractTextFromParts extracts text content from multi-modal message parts.
func extractTextFromParts(parts []openai.ChatCompletionContentPartUnionParam) string {
	var texts []string
	for _, part := range parts {
		texts = append(texts, part.OfText.Text)
	}
	return strings.Join(texts, " ")
}

