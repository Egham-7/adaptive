package utils

import (
	"fmt"
	"strings"

	"google.golang.org/genai"
)

// ExtractPromptFromGeminiContents extracts the prompt text from Gemini Content array for routing decisions
// Intelligently processes different part types and combines them into a coherent prompt
func ExtractPromptFromGeminiContents(contents []*genai.Content) (string, error) {
	if len(contents) == 0 {
		return "", fmt.Errorf("no contents found in request")
	}

	var promptBuilder strings.Builder
	var hasUserContent bool

	// Process all contents to build comprehensive prompt context
	for _, content := range contents {
		if content == nil {
			continue
		}

		// Focus on user content for routing, but include context from other roles
		if content.Role == "user" {
			hasUserContent = true
			if promptBuilder.Len() > 0 {
				promptBuilder.WriteString(" ")
			}

			// Process each part in the content
			for _, part := range content.Parts {
				if part == nil {
					continue
				}

				partText := extractTextFromPart(part)
				if partText != "" {
					if promptBuilder.Len() > 0 {
						promptBuilder.WriteString(" ")
					}
					promptBuilder.WriteString(partText)
				}
			}
		} else if content.Role == "model" && promptBuilder.Len() == 0 {
			// Include model context if no user content found yet (for multi-turn conversations)
			for _, part := range content.Parts {
				if part == nil {
					continue
				}

				partText := extractTextFromPart(part)
				if partText != "" {
					if promptBuilder.Len() > 0 {
						promptBuilder.WriteString(" ")
					}
					promptBuilder.WriteString(partText)
				}
			}
		}
	}

	if !hasUserContent && promptBuilder.Len() == 0 {
		return "", fmt.Errorf("no user content with text found for routing")
	}

	prompt := strings.TrimSpace(promptBuilder.String())
	if prompt == "" {
		return "", fmt.Errorf("no extractable text found in content for routing")
	}

	return prompt, nil
}

// extractTextFromPart extracts text from different types of genai.Part
func extractTextFromPart(part *genai.Part) string {
	if part == nil {
		return ""
	}

	var textParts []string

	// Handle direct text content
	if part.Text != "" {
		textParts = append(textParts, part.Text)
	}

	// Handle function calls - extract function name and purpose for routing context
	if part.FunctionCall != nil {
		functionContext := fmt.Sprintf("function:%s", part.FunctionCall.Name)
		textParts = append(textParts, functionContext)
	}

	// Handle function responses - they might contain relevant context
	if part.FunctionResponse != nil {
		if part.FunctionResponse.Name != "" {
			functionContext := fmt.Sprintf("function_result:%s", part.FunctionResponse.Name)
			textParts = append(textParts, functionContext)
		}
	}

	// Handle executable code - include language context for routing
	if part.ExecutableCode != nil {
		codeContext := fmt.Sprintf("code:%s", part.ExecutableCode.Language)
		if part.ExecutableCode.Code != "" {
			// Include a snippet of code for context (first 100 chars)
			codeSnippet := part.ExecutableCode.Code
			if len(codeSnippet) > 100 {
				codeSnippet = codeSnippet[:100] + "..."
			}
			codeContext += " " + codeSnippet
		}
		textParts = append(textParts, codeContext)
	}

	// Handle code execution results
	if part.CodeExecutionResult != nil {
		if part.CodeExecutionResult.Output != "" {
			// Include a snippet of output for context
			outputSnippet := part.CodeExecutionResult.Output
			if len(outputSnippet) > 100 {
				outputSnippet = outputSnippet[:100] + "..."
			}
			textParts = append(textParts, "code_output: "+outputSnippet)
		}
	}

	// Handle media content - provide context about media type for routing
	if part.InlineData != nil {
		mediaContext := fmt.Sprintf("media:%s", part.InlineData.MIMEType)
		textParts = append(textParts, mediaContext)
	}

	if part.FileData != nil {
		mediaContext := fmt.Sprintf("file:%s", part.FileData.MIMEType)
		if part.FileData.DisplayName != "" {
			mediaContext += ":" + part.FileData.DisplayName
		}
		textParts = append(textParts, mediaContext)
	}

	return strings.Join(textParts, " ")
}

// ExtractToolCallsFromGeminiContents extracts tool calls from the last content if it's a model content
func ExtractToolCallsFromGeminiContents(contents []*genai.Content) any {
	if len(contents) == 0 {
		return nil
	}

	lastContent := contents[len(contents)-1]

	// Only model/assistant content can have function calls
	if lastContent.Role == "model" {
		for _, part := range lastContent.Parts {
			// Check if this part represents a function call
			if part != nil && part.FunctionCall != nil {
				return part
			}
		}
	}

	return nil
}
