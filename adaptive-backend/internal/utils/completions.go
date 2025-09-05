package utils

import (
	"fmt"
	"strings"

	"github.com/openai/openai-go/v2"
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

// ExtractLastMessage extracts content from the last message for cache key generation.
// This supports all message roles (user, assistant, system, developer, tool).
func ExtractLastMessage(messages []openai.ChatCompletionMessageParamUnion) (string, error) {
	if len(messages) == 0 {
		return "", fmt.Errorf("no messages provided")
	}

	// Get the last message
	lastMsg := messages[len(messages)-1]

	var role string
	var content string

	// Handle different message types based on the union structure
	if lastMsg.OfUser != nil {
		role = "user"
		content = extractContentFromUser(lastMsg.OfUser)
	} else if lastMsg.OfAssistant != nil {
		role = "assistant"
		content = extractContentFromAssistant(lastMsg.OfAssistant)
	} else if lastMsg.OfSystem != nil {
		role = "system"
		content = extractContentFromSystem(lastMsg.OfSystem)
	} else if lastMsg.OfDeveloper != nil {
		role = "developer"
		content = extractContentFromDeveloper(lastMsg.OfDeveloper)
	} else if lastMsg.OfTool != nil {
		role = "tool"
		content = extractContentFromTool(lastMsg.OfTool)
	}
	if content == "" {
		return "", fmt.Errorf("no valid content found in last message")
	}

	return fmt.Sprintf("[%s]: %s", role, content), nil
}

// ExtractToolCallsFromLastMessage extracts tool calls from the last message if it's an assistant message.
func ExtractToolCallsFromLastMessage(messages []openai.ChatCompletionMessageParamUnion) any {
	if len(messages) == 0 {
		return nil
	}

	lastMsg := messages[len(messages)-1]

	// Only assistant messages can have tool calls
	if lastMsg.OfAssistant != nil && len(lastMsg.OfAssistant.ToolCalls) > 0 {
		return lastMsg.OfAssistant.ToolCalls
	}

	return nil
}

// extractContentFromUser extracts content from a user message
func extractContentFromUser(msg *openai.ChatCompletionUserMessageParam) string {
	if msg.Content.OfString.Valid() && msg.Content.OfString.Value != "" {
		return msg.Content.OfString.Value
	}
	if msg.Content.OfArrayOfContentParts != nil {
		return extractTextFromUserContentParts(msg.Content.OfArrayOfContentParts)
	}
	return ""
}

// extractContentFromAssistant extracts content from an assistant message
func extractContentFromAssistant(msg *openai.ChatCompletionAssistantMessageParam) string {
	if msg.Content.OfString.Valid() && msg.Content.OfString.Value != "" {
		return msg.Content.OfString.Value
	}
	if msg.Content.OfArrayOfContentParts != nil {
		return extractTextFromAssistantContentParts(msg.Content.OfArrayOfContentParts)
	}

	// Handle messages with tool calls but empty content
	if len(msg.ToolCalls) > 0 {
		var toolCallSummaries []string
		for _, toolCall := range msg.ToolCalls {
			toolCallSummaries = append(toolCallSummaries, fmt.Sprintf("tool_call:%s, args: %s", toolCall.OfFunction.Function.Name, toolCall.OfFunction.Function.Arguments))
		}
		return strings.Join(toolCallSummaries, ",")
	}

	return ""
}

// extractContentFromSystem extracts content from a system message
func extractContentFromSystem(msg *openai.ChatCompletionSystemMessageParam) string {
	if msg.Content.OfString.Valid() && msg.Content.OfString.Value != "" {
		return msg.Content.OfString.Value
	}
	if msg.Content.OfArrayOfContentParts != nil {
		return extractTextFromSystemContentParts(msg.Content.OfArrayOfContentParts)
	}
	return ""
}

// extractContentFromDeveloper extracts content from a developer message
func extractContentFromDeveloper(msg *openai.ChatCompletionDeveloperMessageParam) string {
	if msg.Content.OfString.Valid() && msg.Content.OfString.Value != "" {
		return msg.Content.OfString.Value
	}
	if msg.Content.OfArrayOfContentParts != nil {
		return extractTextFromDeveloperContentParts(msg.Content.OfArrayOfContentParts)
	}
	return ""
}

// extractContentFromTool extracts content from a tool message
func extractContentFromTool(msg *openai.ChatCompletionToolMessageParam) string {
	if msg.Content.OfString.Valid() && msg.Content.OfString.Value != "" {
		return msg.Content.OfString.Value
	}
	if msg.Content.OfArrayOfContentParts != nil {
		return extractTextFromToolContentParts(msg.Content.OfArrayOfContentParts)
	}
	return ""
}

// extractTextFromUserContentParts extracts text from user content parts
func extractTextFromUserContentParts(parts []openai.ChatCompletionContentPartUnionParam) string {
	var texts []string
	for _, part := range parts {
		if part.OfText != nil {
			texts = append(texts, part.OfText.Text)
		}
	}
	return strings.Join(texts, " ")
}

// extractTextFromAssistantContentParts extracts text from assistant content parts
func extractTextFromAssistantContentParts(parts []openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion) string {
	var texts []string
	for _, part := range parts {
		if part.OfText != nil {
			texts = append(texts, part.OfText.Text)
		}
	}
	return strings.Join(texts, " ")
}

// extractTextFromSystemContentParts extracts text from system content parts
func extractTextFromSystemContentParts(parts []openai.ChatCompletionContentPartTextParam) string {
	var texts []string
	for _, part := range parts {
		texts = append(texts, part.Text)
	}
	return strings.Join(texts, " ")
}

// extractTextFromDeveloperContentParts extracts text from developer content parts
func extractTextFromDeveloperContentParts(parts []openai.ChatCompletionContentPartTextParam) string {
	var texts []string
	for _, part := range parts {
		texts = append(texts, part.Text)
	}
	return strings.Join(texts, " ")
}

// extractTextFromToolContentParts extracts text from tool content parts
func extractTextFromToolContentParts(parts []openai.ChatCompletionContentPartTextParam) string {
	var texts []string
	for _, part := range parts {
		texts = append(texts, part.Text)
	}
	return strings.Join(texts, " ")
}
