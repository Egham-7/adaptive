package format_adapter

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicparam "github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/openai/openai-go/v2"
)

// OpenAIToAnthropicConverter handles conversion from OpenAI format to Anthropic format
type OpenAIToAnthropicConverter struct{}

// NewOpenAIToAnthropicConverter creates a new OpenAI to Anthropic converter
func NewOpenAIToAnthropicConverter() *OpenAIToAnthropicConverter {
	return &OpenAIToAnthropicConverter{}
}

// ConvertRequest converts OpenAI ChatCompletionNewParams to Anthropic MessageNewParams
func (c *OpenAIToAnthropicConverter) ConvertRequest(req *openai.ChatCompletionNewParams) (*anthropic.MessageNewParams, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Convert messages from OpenAI to Anthropic format
	anthropicMessages, systemMessage, err := c.convertMessagesFromOpenAI(req.Messages)
	if err != nil {
		return nil, fmt.Errorf("failed to convert messages: %w", err)
	}

	// Create base Anthropic request
	anthropicReq := &anthropic.MessageNewParams{
		Model:    anthropic.Model(req.Model),
		Messages: anthropicMessages,
	}

	// Set system message if present (Anthropic uses separate system field)
	if systemMessage != "" {
		anthropicReq.System = []anthropic.TextBlockParam{
			{
				Text: systemMessage,
			},
		}
	}

	// Convert parameters
	if err := c.convertParameters(req, anthropicReq); err != nil {
		return nil, fmt.Errorf("failed to convert parameters: %w", err)
	}

	// Convert tool choice if present (check if any ToolChoice variant is set)
	if req.ToolChoice.OfAuto.Valid() ||
		req.ToolChoice.OfFunctionToolChoice != nil || req.ToolChoice.OfCustomToolChoice != nil {
		toolChoice, err := c.convertToolChoice(req.ToolChoice)
		if err != nil {
			return nil, fmt.Errorf("failed to convert tool choice: %w", err)
		}
		anthropicReq.ToolChoice = toolChoice
	}

	// Convert service tier if present
	if req.ServiceTier != "" {
		serviceTier, err := c.convertServiceTier(req.ServiceTier)
		if err != nil {
			return nil, fmt.Errorf("failed to convert service tier: %w", err)
		}
		anthropicReq.ServiceTier = serviceTier
	}

	// Convert metadata if present - convert OpenAI param types to Anthropic param types
	if req.SafetyIdentifier.Valid() || req.User.Valid() {
		var safetyID, user anthropicparam.Opt[string]
		if req.SafetyIdentifier.Valid() {
			safetyID = anthropicparam.Opt[string]{Value: req.SafetyIdentifier.Value}
		}
		if req.User.Valid() {
			user = anthropicparam.Opt[string]{Value: req.User.Value}
		}
		metadata := c.convertMetadata(safetyID, user)
		anthropicReq.Metadata = metadata
	}

	// Convert tools if present
	if len(req.Tools) > 0 {
		anthropicTools, err := c.convertTools(req.Tools)
		if err != nil {
			return nil, fmt.Errorf("failed to convert tools: %w", err)
		}
		// Convert to ToolUnionParam slice
		var toolUnions []anthropic.ToolUnionParam
		for i := range anthropicTools {
			toolUnions = append(toolUnions, anthropic.ToolUnionParam{
				OfTool: &anthropicTools[i],
			})
		}
		anthropicReq.Tools = toolUnions
	}

	// Convert stop sequences if present
	if req.Stop.OfString.Value != "" || len(req.Stop.OfStringArray) > 0 {
		stopSequences, err := c.convertStopSequences(req.Stop)
		if err != nil {
			return nil, fmt.Errorf("failed to convert stop sequences: %w", err)
		}
		if len(stopSequences) > 0 {
			anthropicReq.StopSequences = stopSequences
		}
	}

	return anthropicReq, nil
}

// convertMessagesFromOpenAI converts OpenAI messages to Anthropic format
func (c *OpenAIToAnthropicConverter) convertMessagesFromOpenAI(messages []openai.ChatCompletionMessageParamUnion) ([]anthropic.MessageParam, string, error) {
	var anthropicMessages []anthropic.MessageParam
	var systemMessage string

	for _, msg := range messages {
		// OpenAI SDK uses union structs with Of* fields, not interface switches
		if msg.OfUser != nil {
			// Convert user message
			content, err := c.convertUserContent(msg.OfUser.Content)
			if err != nil {
				return nil, "", fmt.Errorf("failed to convert user message: %w", err)
			}

			anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
				Role:    anthropic.MessageParamRoleUser,
				Content: content,
			})

		} else if msg.OfAssistant != nil {
			// Convert assistant message
			content, err := c.convertAssistantContent(msg.OfAssistant.Content, msg.OfAssistant.ToolCalls)
			if err != nil {
				return nil, "", fmt.Errorf("failed to convert assistant message: %w", err)
			}

			anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
				Role:    anthropic.MessageParamRoleAssistant,
				Content: content,
			})

		} else if msg.OfSystem != nil {
			// Anthropic uses separate system field - collect all system messages
			systemContent := c.extractSystemContent(msg.OfSystem.Content)
			if systemMessage != "" {
				systemMessage += "\n\n" + systemContent
			} else {
				systemMessage = systemContent
			}

		} else if msg.OfDeveloper != nil {
			// Handle developer messages (convert to system in Anthropic)
			developerContent := c.extractDeveloperContent(msg.OfDeveloper.Content)
			if systemMessage != "" {
				systemMessage += "\n\n" + developerContent
			} else {
				systemMessage = developerContent
			}

		} else if msg.OfTool != nil {
			// Convert tool messages to user messages with tool result content
			toolContent, err := c.convertToolContent(msg.OfTool.Content)
			if err != nil {
				return nil, "", fmt.Errorf("failed to convert tool content: %w", err)
			}

			isError := false
			// Check if this is an error response based on content or status
			if msg.OfTool.Content.OfString.Valid() {
				content := strings.ToLower(msg.OfTool.Content.OfString.Value)
				if strings.Contains(content, "error") || strings.Contains(content, "failed") {
					isError = true
				}
			}

			anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
				Role: anthropic.MessageParamRoleUser,
				Content: []anthropic.ContentBlockParamUnion{
					{
						OfToolResult: &anthropic.ToolResultBlockParam{
							Type:      "tool_result",
							ToolUseID: msg.OfTool.ToolCallID,
							Content:   toolContent,
							IsError:   anthropicparam.Opt[bool]{Value: isError},
						},
					},
				},
			})
		} else if msg.OfFunction != nil {
			// Convert deprecated function messages to tool result format
			functionContent := msg.OfFunction.Content.Value
			if functionContent == "" {
				functionContent = "Function executed successfully"
			}

			isError := false
			if strings.Contains(strings.ToLower(functionContent), "error") {
				isError = true
			}

			anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
				Role: anthropic.MessageParamRoleUser,
				Content: []anthropic.ContentBlockParamUnion{
					{
						OfToolResult: &anthropic.ToolResultBlockParam{
							Type:      "tool_result",
							ToolUseID: msg.OfFunction.Name, // Use function name as tool ID
							Content:   []anthropic.ToolResultBlockParamContentUnion{{OfText: &anthropic.TextBlockParam{Type: "text", Text: functionContent}}},
							IsError:   anthropicparam.Opt[bool]{Value: isError},
						},
					},
				},
			})
		}
	}

	return anthropicMessages, systemMessage, nil
}

// convertUserContent converts OpenAI user content to Anthropic format
func (c *OpenAIToAnthropicConverter) convertUserContent(content openai.ChatCompletionUserMessageParamContentUnion) ([]anthropic.ContentBlockParamUnion, error) {
	// OpenAI SDK uses union structs with Of* fields
	if content.OfString.Valid() {
		// Simple text content
		return []anthropic.ContentBlockParamUnion{
			{
				OfText: &anthropic.TextBlockParam{
					Type: "text",
					Text: content.OfString.Value,
				},
			},
		}, nil
	} else if len(content.OfArrayOfContentParts) > 0 {
		// Multi-part content (text + images, etc.)
		var parts []anthropic.ContentBlockParamUnion

		for _, part := range content.OfArrayOfContentParts {
			if part.OfText != nil {
				parts = append(parts, anthropic.ContentBlockParamUnion{
					OfText: &anthropic.TextBlockParam{
						Type: "text",
						Text: part.OfText.Text,
					},
				})
			} else if part.OfImageURL != nil {
				imageBlock, err := c.convertImageContent(*part.OfImageURL)
				if err != nil {
					return nil, fmt.Errorf("failed to convert image: %w", err)
				}
				parts = append(parts, *imageBlock)
			} else if part.OfFile != nil {
				fileBlock, err := c.convertFileContent(*part.OfFile)
				if err != nil {
					return nil, fmt.Errorf("failed to convert file: %w", err)
				}
				parts = append(parts, *fileBlock)
			}
		}

		return parts, nil

	} else {
		// Fallback for unsupported content types
		return []anthropic.ContentBlockParamUnion{
			{
				OfText: &anthropic.TextBlockParam{
					Type: "text",
					Text: "Unsupported content type converted to text",
				},
			},
		}, nil
	}
}

// convertAssistantContent converts OpenAI assistant content to Anthropic format
func (c *OpenAIToAnthropicConverter) convertAssistantContent(content openai.ChatCompletionAssistantMessageParamContentUnion, toolCalls []openai.ChatCompletionMessageToolCallUnionParam) ([]anthropic.ContentBlockParamUnion, error) {
	var contentParts []anthropic.ContentBlockParamUnion

	// Add text content if present (based on Context7 docs, assistant content can be string or null)
	if content.OfString.Valid() {
		contentParts = append(contentParts, anthropic.ContentBlockParamUnion{
			OfText: &anthropic.TextBlockParam{
				Type: "text",
				Text: content.OfString.Value,
			},
		})
	}

	// Convert tool calls to Anthropic tool_use blocks
	for _, toolCall := range toolCalls {
		// Based on Context7 docs, tool calls are ChatCompletionMessageToolCallUnionParam
		if toolCall.OfFunction != nil && toolCall.OfFunction.Function.Arguments != "" {
			// Parse function arguments
			var args map[string]any
			if err := json.Unmarshal([]byte(toolCall.OfFunction.Function.Arguments), &args); err != nil {
				// If JSON parsing fails, use the raw string in a simple map
				args = map[string]any{"arguments": toolCall.OfFunction.Function.Arguments}
			}

			contentParts = append(contentParts, anthropic.ContentBlockParamUnion{
				OfToolUse: &anthropic.ToolUseBlockParam{
					Type:  "tool_use",
					ID:    toolCall.OfFunction.ID,
					Name:  toolCall.OfFunction.Function.Name,
					Input: args,
				},
			})
		}
	}

	// If no content at all, add empty text block
	if len(contentParts) == 0 {
		contentParts = append(contentParts, anthropic.ContentBlockParamUnion{
			OfText: &anthropic.TextBlockParam{
				Type: "text",
				Text: "",
			},
		})
	}

	return contentParts, nil
}

// convertImageContent converts OpenAI image content to Anthropic format
func (c *OpenAIToAnthropicConverter) convertImageContent(img openai.ChatCompletionContentPartImageParam) (*anthropic.ContentBlockParamUnion, error) {
	if img.ImageURL.URL == "" {
		return nil, fmt.Errorf("image URL is required")
	}

	url := img.ImageURL.URL

	// Handle base64 data URLs (data:image/jpeg;base64,...)
	if strings.HasPrefix(url, "data:") {
		parts := strings.Split(url, ",")
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid data URL format")
		}

		// Extract media type from data URL header
		headerParts := strings.Split(parts[0], ":")
		if len(headerParts) < 2 {
			return nil, fmt.Errorf("invalid data URL header")
		}

		mediaTypeParts := strings.Split(headerParts[1], ";")
		mediaType := mediaTypeParts[0]

		return &anthropic.ContentBlockParamUnion{
			OfImage: &anthropic.ImageBlockParam{
				Type: "image",
				Source: anthropic.ImageBlockParamSourceUnion{
					OfBase64: &anthropic.Base64ImageSourceParam{
						Type:      "base64",
						MediaType: anthropic.Base64ImageSourceMediaType(mediaType),
						Data:      parts[1], // base64 data without data URL prefix
					},
				},
			},
		}, nil
	}

	// Handle regular URL references
	return &anthropic.ContentBlockParamUnion{
		OfImage: &anthropic.ImageBlockParam{
			Type: "image",
			Source: anthropic.ImageBlockParamSourceUnion{
				OfURL: &anthropic.URLImageSourceParam{
					Type: "url",
					URL:  url,
				},
			},
		},
	}, nil
}

// convertParameters converts OpenAI parameters to Anthropic equivalents
func (c *OpenAIToAnthropicConverter) convertParameters(openaiReq *openai.ChatCompletionNewParams, anthropicReq *anthropic.MessageNewParams) error {
	// Temperature (both use 0-1 range)
	if openaiReq.Temperature.Valid() {
		anthropicReq.Temperature = anthropicparam.Opt[float64]{Value: openaiReq.Temperature.Value}
	}

	// Max tokens (Anthropic requires this field)
	if openaiReq.MaxTokens.Valid() {
		anthropicReq.MaxTokens = openaiReq.MaxTokens.Value
	} else if openaiReq.MaxCompletionTokens.Valid() {
		anthropicReq.MaxTokens = openaiReq.MaxCompletionTokens.Value
	} else {
		// Anthropic requires max_tokens, set reasonable default
		anthropicReq.MaxTokens = 4096
	}

	// Top-p (both use 0-1 range)
	if openaiReq.TopP.Valid() {
		anthropicReq.TopP = anthropicparam.Opt[float64]{Value: openaiReq.TopP.Value}
	}

	// Top-k (OpenAI has this as topLogprobs, Anthropic as top_k)
	if openaiReq.TopLogprobs.Valid() {
		// Convert top_logprobs to top_k (approximate mapping)
		topK := openaiReq.TopLogprobs.Value
		if topK > 0 && topK <= 500 { // Anthropic's valid range
			anthropicReq.TopK = anthropicparam.Opt[int64]{Value: topK}
		}
	}

	// Note: OpenAI doesn't have a separate system field in ChatCompletionNewParams
	// System messages are included in the Messages array and converted separately

	// Convert reasoning effort to thinking budget if present
	if openaiReq.ReasoningEffort != "" {
		thinkingBudget, err := c.convertReasoningEffort(string(openaiReq.ReasoningEffort))
		if err == nil {
			anthropicReq.Thinking = anthropic.ThinkingConfigParamUnion{
				OfEnabled: &anthropic.ThinkingConfigEnabledParam{
					Type:         "enabled",
					BudgetTokens: int64(thinkingBudget),
				},
			}
		}
	}

	// Convert modalities and audio parameters (OpenAI audio to Anthropic text)
	if len(openaiReq.Modalities) > 0 {
		// Anthropic doesn't support audio output directly
		// Audio parameters are present but will be ignored in Anthropic conversion
		_ = openaiReq.Audio.Format // Acknowledge audio format but ignore for Anthropic
	}

	// Note: Some OpenAI parameters don't have Anthropic equivalents:
	// - frequency_penalty, presence_penalty, logit_bias, n, seed, etc.
	// These are silently ignored during conversion

	return nil
}

// convertTools converts OpenAI tools to Anthropic format
func (c *OpenAIToAnthropicConverter) convertTools(tools []openai.ChatCompletionToolUnionParam) ([]anthropic.ToolParam, error) {
	var anthropicTools []anthropic.ToolParam
	var errors []string

	for i, tool := range tools {
		var anthropicTool *anthropic.ToolParam
		var err error

		if tool.OfFunction != nil {
			anthropicTool, err = c.convertFunctionTool(&tool)
		} else if tool.OfCustom != nil {
			anthropicTool, err = c.convertCustomTool(&tool)
		} else {
			// Skip unsupported tool types with warning
			errors = append(errors, fmt.Sprintf("unsupported tool type at index %d", i))
			continue
		}

		if err != nil {
			errors = append(errors, fmt.Sprintf("failed to convert tool at index %d: %v", i, err))
			continue
		}

		if anthropicTool != nil {
			anthropicTools = append(anthropicTools, *anthropicTool)
		}
	}

	// Return error if we couldn't convert any tools
	if len(anthropicTools) == 0 && len(tools) > 0 {
		return nil, fmt.Errorf("failed to convert any tools: %s", strings.Join(errors, "; "))
	}

	return anthropicTools, nil
}

// convertFunctionTool converts OpenAI function tool to Anthropic format
func (c *OpenAIToAnthropicConverter) convertFunctionTool(funcTool *openai.ChatCompletionToolUnionParam) (*anthropic.ToolParam, error) {
	if funcTool == nil || funcTool.OfFunction == nil || funcTool.OfFunction.Function.Name == "" {
		return nil, fmt.Errorf("function tool name cannot be empty")
	}

	// Validate tool name (Anthropic has restrictions)
	if err := c.validateToolName(funcTool.OfFunction.Function.Name); err != nil {
		return nil, fmt.Errorf("invalid function tool name '%s': %w", funcTool.OfFunction.Function.Name, err)
	}

	anthropicTool := anthropic.ToolParam{
		Name: funcTool.OfFunction.Function.Name,
		Type: "custom",
	}

	// Set description if present
	if funcTool.OfFunction.Function.Description.Valid() {
		anthropicTool.Description = anthropicparam.Opt[string]{Value: funcTool.OfFunction.Function.Description.Value}
	}

	// Convert function parameters with enhanced validation
	if funcTool.OfFunction.Function.Parameters != nil {
		inputSchema, err := c.convertJSONSchemaToInputSchema(funcTool.OfFunction.Function.Parameters)
		if err != nil {
			return nil, fmt.Errorf("failed to convert function parameters: %w", err)
		}
		anthropicTool.InputSchema = inputSchema
	} else {
		// Default empty schema for parameterless functions
		anthropicTool.InputSchema = anthropic.ToolInputSchemaParam{
			Type:       "object",
			Properties: map[string]any{},
			Required:   []string{},
		}
	}

	return &anthropicTool, nil
}

// convertCustomTool converts OpenAI custom tool to Anthropic format
func (c *OpenAIToAnthropicConverter) convertCustomTool(customTool *openai.ChatCompletionToolUnionParam) (*anthropic.ToolParam, error) {
	if customTool == nil || customTool.OfCustom == nil || customTool.OfCustom.Custom.Name == "" {
		return nil, fmt.Errorf("custom tool name cannot be empty")
	}

	// Validate tool name
	if err := c.validateToolName(customTool.OfCustom.Custom.Name); err != nil {
		return nil, fmt.Errorf("invalid custom tool name '%s': %w", customTool.OfCustom.Custom.Name, err)
	}

	anthropicTool := anthropic.ToolParam{
		Name: customTool.OfCustom.Custom.Name,
		Type: "custom",
	}

	// Set description if present
	if customTool.OfCustom.Custom.Description.Valid() {
		anthropicTool.Description = anthropicparam.Opt[string]{Value: customTool.OfCustom.Custom.Description.Value}
	}

	// Convert custom tool format to input schema with enhanced handling
	if customTool.OfCustom.Custom.Format.OfGrammar != nil {
		// Handle grammar format with validation
		grammar := customTool.OfCustom.Custom.Format.OfGrammar
		if grammar.Grammar.Syntax == "" || grammar.Grammar.Definition == "" {
			return nil, fmt.Errorf("grammar format requires both syntax and definition")
		}

		anthropicTool.InputSchema = anthropic.ToolInputSchemaParam{
			Type: "object",
			Properties: map[string]any{
				"input": map[string]any{
					"type": "string",
					"description": fmt.Sprintf("Input following %s grammar: %s",
						grammar.Grammar.Syntax, grammar.Grammar.Definition),
					"pattern": c.generatePatternFromGrammar(grammar.Grammar.Syntax, grammar.Grammar.Definition),
				},
			},
			Required: []string{"input"},
		}
	} else if customTool.OfCustom.Custom.Format.OfText != nil {
		// Handle text format
		anthropicTool.InputSchema = anthropic.ToolInputSchemaParam{
			Type: "object",
			Properties: map[string]any{
				"input": map[string]any{
					"type":        "string",
					"description": "Free text input for the custom tool",
				},
			},
			Required: []string{"input"},
		}
	} else {
		// Default to text format
		anthropicTool.InputSchema = anthropic.ToolInputSchemaParam{
			Type: "object",
			Properties: map[string]any{
				"input": map[string]any{
					"type":        "string",
					"description": "Input for the custom tool",
				},
			},
			Required: []string{"input"},
		}
	}

	return &anthropicTool, nil
}

// validateToolName validates tool names according to Anthropic requirements
func (c *OpenAIToAnthropicConverter) validateToolName(name string) error {
	if name == "" {
		return fmt.Errorf("tool name cannot be empty")
	}

	// Anthropic tool names must be alphanumeric with underscores, 1-64 chars
	if len(name) > 64 {
		return fmt.Errorf("tool name too long (max 64 characters)")
	}

	for i, char := range name {
		if (char < 'a' || char > 'z') && (char < 'A' || char > 'Z') &&
			(char < '0' || char > '9') && char != '_' {
			return fmt.Errorf("invalid character '%c' at position %d (only alphanumeric and underscore allowed)", char, i)
		}
	}

	return nil
}

// convertJSONSchemaToInputSchema converts OpenAI JSON Schema to Anthropic InputSchema
func (c *OpenAIToAnthropicConverter) convertJSONSchemaToInputSchema(parameters openai.FunctionParameters) (anthropic.ToolInputSchemaParam, error) {
	// Cast to map for processing
	m := map[string]any(parameters)

	// Validate required fields
	schemaType, hasType := m["type"]
	if !hasType {
		// Default to object if no type specified
		schemaType = "object"
	}

	if schemaType != "object" {
		return anthropic.ToolInputSchemaParam{}, fmt.Errorf("only object type schemas are supported, got: %v", schemaType)
	}

	// Extract properties with validation
	var props any
	if p, ok := m["properties"]; ok {
		if propsMap, isMap := p.(map[string]any); isMap {
			props = propsMap
		} else {
			return anthropic.ToolInputSchemaParam{}, fmt.Errorf("properties must be an object")
		}
	} else {
		// Empty properties for schemas without explicit properties
		props = map[string]any{}
	}

	// Extract and validate required fields
	var reqFields []string
	if r, ok := m["required"]; ok {
		switch reqArray := r.(type) {
		case []any:
			reqFields = make([]string, 0, len(reqArray))
			for i, v := range reqArray {
				if s, ok := v.(string); ok {
					reqFields = append(reqFields, s)
				} else {
					return anthropic.ToolInputSchemaParam{}, fmt.Errorf("required[%d] must be a string, got: %T", i, v)
				}
			}
		case []string:
			reqFields = reqArray
		default:
			return anthropic.ToolInputSchemaParam{}, fmt.Errorf("required must be an array of strings, got: %T", r)
		}
	}

	return anthropic.ToolInputSchemaParam{
		Type:       "object",
		Properties: props,
		Required:   reqFields,
	}, nil
}

// generatePatternFromGrammar generates a regex pattern hint from grammar definition
func (c *OpenAIToAnthropicConverter) generatePatternFromGrammar(syntax, definition string) string {
	// This is a simplified pattern generator - in practice, you'd want more sophisticated grammar parsing
	switch strings.ToLower(syntax) {
	case "json":
		return `^[\s]*\{.*\}[\s]*$`
	case "xml":
		return `^[\s]*<.*>.*</.*>[\s]*$`
	case "regex", "regexp":
		// Use the definition as the pattern if it's a regex
		return definition
	default:
		// Generic pattern for structured text
		return `^.+$`
	}
}

// convertStopSequences converts OpenAI stop sequences to Anthropic format
func (c *OpenAIToAnthropicConverter) convertStopSequences(stop openai.ChatCompletionNewParamsStopUnion) ([]string, error) {
	// OpenAI SDK uses union structs with Of* fields
	if stop.OfString.Value != "" {
		return []string{stop.OfString.Value}, nil
	} else if len(stop.OfStringArray) > 0 {
		return stop.OfStringArray, nil
	}
	return nil, fmt.Errorf("no stop sequences found in union")
}

// extractSystemContent extracts string content from OpenAI system message
func (c *OpenAIToAnthropicConverter) extractSystemContent(content openai.ChatCompletionSystemMessageParamContentUnion) string {
	// OpenAI SDK uses union structs with Of* fields
	if content.OfString.Valid() {
		return content.OfString.Value
	} else if len(content.OfArrayOfContentParts) > 0 {
		var text strings.Builder
		for _, part := range content.OfArrayOfContentParts {
			if part.Text != "" {
				if text.Len() > 0 {
					text.WriteString("\n")
				}
				text.WriteString(part.Text)
			}
		}
		return text.String()
	}
	return ""
}

// convertFinishReason converts OpenAI finish reason to Anthropic stop reason
func (c *OpenAIToAnthropicConverter) convertFinishReason(finishReason string) anthropic.StopReason {
	switch finishReason {
	case "stop":
		return anthropic.StopReasonEndTurn
	case "length":
		return anthropic.StopReasonMaxTokens
	case "tool_calls":
		return anthropic.StopReasonToolUse
	case "content_filter":
		return anthropic.StopReasonRefusal
	default:
		return anthropic.StopReasonEndTurn
	}
}

// convertUsage converts OpenAI usage to Anthropic usage
func (c *OpenAIToAnthropicConverter) convertUsage(usage openai.CompletionUsage) anthropic.Usage {
	return anthropic.Usage{
		InputTokens:              usage.PromptTokens,
		OutputTokens:             usage.CompletionTokens,
		CacheCreationInputTokens: 0,
		CacheReadInputTokens:     0,
		ServerToolUse: anthropic.ServerToolUsage{
			WebSearchRequests: 0,
		},
		ServiceTier: anthropic.UsageServiceTierStandard,
	}
}

// convertResponseContent converts OpenAI message content to Anthropic content blocks
func (c *OpenAIToAnthropicConverter) convertResponseContent(content string, toolCalls []openai.ChatCompletionMessageToolCallUnion) ([]anthropic.ContentBlockUnion, error) {
	var contentBlocks []anthropic.ContentBlockUnion

	// Add text content if present
	if content != "" {
		contentBlocks = append(contentBlocks, anthropic.ContentBlockUnion{
			Type: "text",
			Text: content,
		})
	}

	// Add tool calls if present
	for _, toolCall := range toolCalls {

		// Parse function arguments as raw JSON
		var input json.RawMessage
		if toolCall.Function.Arguments != "" {
			input = json.RawMessage(toolCall.Function.Arguments)
		} else {
			input = json.RawMessage("{}")
		}

		contentBlocks = append(contentBlocks, anthropic.ContentBlockUnion{
			Type:  "tool_use",
			ID:    toolCall.ID,
			Name:  toolCall.Function.Name,
			Input: input,
		})
	}

	return contentBlocks, nil
}

// ConvertResponse converts OpenAI ChatCompletion to Anthropic Message format
func (c *OpenAIToAnthropicConverter) ConvertResponse(resp *openai.ChatCompletion, provider string) (*anthropic.Message, error) {
	if resp == nil {
		return nil, fmt.Errorf("chat completion cannot be nil")
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in OpenAI response")
	}

	// Use the first choice (OpenAI can return multiple, Anthropic returns one)
	choice := resp.Choices[0]

	// Use helper function to convert response content
	contentBlocks, err := c.convertResponseContent(choice.Message.Content, choice.Message.ToolCalls)
	if err != nil {
		return nil, fmt.Errorf("failed to convert response content: %w", err)
	}

	// Use helper to convert stop reason
	stopReason := c.convertFinishReason(choice.FinishReason)

	// Use helper to convert model name
	modelName := anthropicparam.Opt[string]{Value: resp.Model}

	// Use helper to convert usage
	usage := c.convertUsage(resp.Usage)

	// Create Anthropic Message
	message := &anthropic.Message{
		ID:           resp.ID,
		Content:      contentBlocks,
		Model:        anthropic.Model(modelName.Value),
		Role:         "assistant",
		StopReason:   stopReason,
		StopSequence: "",
		Type:         "message",
		Usage:        usage,
	}

	return message, nil
}

// ConvertStreamingChunk converts OpenAI streaming chunk to Anthropic streaming event format
// Note: This function should be called as part of a stateful streaming process that tracks
// whether content blocks have been started
func (c *OpenAIToAnthropicConverter) ConvertStreamingChunk(chunk *openai.ChatCompletionChunk, provider string) (*anthropic.MessageStreamEventUnion, error) {
	if chunk == nil {
		return nil, fmt.Errorf("chat completion chunk cannot be nil")
	}

	if len(chunk.Choices) == 0 {
		return nil, fmt.Errorf("no choices in OpenAI streaming chunk")
	}

	choice := chunk.Choices[0]

	// Handle different types of streaming events based on OpenAI chunk content

	// Check if this is the start of streaming (when delta role appears)
	if choice.Delta.Role != "" && choice.Delta.Content == "" && len(choice.Delta.ToolCalls) == 0 {
		// This is a message start event - convert usage properly
		usage := c.convertUsage(chunk.Usage)
		return &anthropic.MessageStreamEventUnion{
			Type: "message_start",
			Message: anthropic.Message{
				ID:      chunk.ID,
				Content: []anthropic.ContentBlockUnion{},
				Model:   anthropic.Model(chunk.Model),
				Role:    "assistant",
				Type:    "message",
				Usage:   usage,
			},
		}, nil
	}

	// Handle content delta (text content) - caller should emit content_block_start first
	if choice.Delta.Content != "" {
		return &anthropic.MessageStreamEventUnion{
			Type:  "content_block_delta",
			Index: choice.Index,
			Delta: anthropic.MessageStreamEventUnionDelta{
				Type: "text_delta",
				Text: choice.Delta.Content,
			},
		}, nil
	}

	// Handle tool call delta
	if len(choice.Delta.ToolCalls) > 0 {
		toolCall := choice.Delta.ToolCalls[0]

		// Check if this is a new tool use block starting
		if toolCall.ID != "" && toolCall.Function.Name != "" {
			return &anthropic.MessageStreamEventUnion{
				Type:  "content_block_start",
				Index: choice.Index,
				ContentBlock: anthropic.ContentBlockStartEventContentBlockUnion{
					Type:  "tool_use",
					ID:    toolCall.ID,
					Name:  toolCall.Function.Name,
					Input: json.RawMessage("{}"), // Initially empty
				},
			}, nil
		}

		// Handle function arguments delta
		if toolCall.Function.Arguments != "" {
			return &anthropic.MessageStreamEventUnion{
				Type:  "content_block_delta",
				Index: choice.Index,
				Delta: anthropic.MessageStreamEventUnionDelta{
					Type:        "input_json_delta",
					PartialJSON: toolCall.Function.Arguments,
				},
			}, nil
		}
	}

	// Handle finish reason (end of streaming)
	if choice.FinishReason != "" {
		stopReason := c.convertFinishReason(choice.FinishReason)
		event := &anthropic.MessageStreamEventUnion{
			Type: "message_delta",
			Delta: anthropic.MessageStreamEventUnionDelta{
				StopReason: stopReason,
			},
		}

		// Add usage only if present in the chunk
		if chunk.Usage.CompletionTokens != 0 || chunk.Usage.PromptTokens != 0 {
			event.Usage = anthropic.MessageDeltaUsage{
				OutputTokens: chunk.Usage.CompletionTokens,
				InputTokens:  chunk.Usage.PromptTokens,
			}
		}

		return event, nil
	}

	// Return nil for chunks that don't contain meaningful updates
	// This allows the caller to skip empty chunks
	return nil, nil
}

// convertToolChoice converts OpenAI tool choice to Anthropic format
func (c *OpenAIToAnthropicConverter) convertToolChoice(toolChoice openai.ChatCompletionToolChoiceOptionUnionParam) (anthropic.ToolChoiceUnionParam, error) {
	if toolChoice.OfAuto.Valid() {
		return anthropic.ToolChoiceUnionParam{
			OfAuto: &anthropic.ToolChoiceAutoParam{
				Type:                   "auto",
				DisableParallelToolUse: anthropicparam.Opt[bool]{Value: false},
			},
		}, nil
	}

	if toolChoice.OfFunctionToolChoice != nil {
		return anthropic.ToolChoiceUnionParam{
			OfTool: &anthropic.ToolChoiceToolParam{
				Type:                   "tool",
				Name:                   toolChoice.OfFunctionToolChoice.Function.Name,
				DisableParallelToolUse: anthropicparam.Opt[bool]{Value: true},
			},
		}, nil
	}

	// Default to auto
	return anthropic.ToolChoiceUnionParam{
		OfAuto: &anthropic.ToolChoiceAutoParam{
			Type:                   "auto",
			DisableParallelToolUse: anthropicparam.Opt[bool]{Value: false},
		},
	}, nil
}

// convertServiceTier converts OpenAI service tier to Anthropic format
func (c *OpenAIToAnthropicConverter) convertServiceTier(serviceTier openai.ChatCompletionNewParamsServiceTier) (anthropic.MessageNewParamsServiceTier, error) {
	switch serviceTier {
	case "auto":
		return anthropic.MessageNewParamsServiceTierAuto, nil
	case "default":
		return anthropic.MessageNewParamsServiceTierStandardOnly, nil
	default:
		return anthropic.MessageNewParamsServiceTierAuto, nil
	}
}

// convertMetadata converts OpenAI metadata fields to Anthropic format
func (c *OpenAIToAnthropicConverter) convertMetadata(safetyIdentifier, user anthropicparam.Opt[string]) anthropic.MetadataParam {
	metadata := anthropic.MetadataParam{}

	// Use safety_identifier as user_id, fallback to user field
	if safetyIdentifier.Valid() {
		metadata.UserID = safetyIdentifier
	} else if user.Valid() {
		metadata.UserID = user
	}

	return metadata
}

// extractDeveloperContent extracts string content from OpenAI developer message
func (c *OpenAIToAnthropicConverter) extractDeveloperContent(content openai.ChatCompletionDeveloperMessageParamContentUnion) string {
	if content.OfString.Valid() {
		return content.OfString.Value
	} else if len(content.OfArrayOfContentParts) > 0 {
		var text strings.Builder
		for _, part := range content.OfArrayOfContentParts {
			if part.Text != "" {
				if text.Len() > 0 {
					text.WriteString("\n")
				}
				text.WriteString(part.Text)
			}
		}
		return text.String()
	}
	return ""
}

// convertToolContent converts OpenAI tool content to Anthropic format
func (c *OpenAIToAnthropicConverter) convertToolContent(content openai.ChatCompletionToolMessageParamContentUnion) ([]anthropic.ToolResultBlockParamContentUnion, error) {
	if content.OfString.Valid() {
		return []anthropic.ToolResultBlockParamContentUnion{
			{OfText: &anthropic.TextBlockParam{Type: "text", Text: content.OfString.Value}},
		}, nil
	} else if len(content.OfArrayOfContentParts) > 0 {
		var parts []anthropic.ToolResultBlockParamContentUnion
		for _, part := range content.OfArrayOfContentParts {
			if part.Text != "" {
				parts = append(parts, anthropic.ToolResultBlockParamContentUnion{
					OfText: &anthropic.TextBlockParam{Type: "text", Text: part.Text},
				})
			}
		}
		return parts, nil
	}

	// Fallback to empty text
	return []anthropic.ToolResultBlockParamContentUnion{
		{OfText: &anthropic.TextBlockParam{Type: "text", Text: ""}},
	}, nil
}

// convertReasoningEffort converts OpenAI reasoning effort to Anthropic thinking budget
func (c *OpenAIToAnthropicConverter) convertReasoningEffort(effort string) (int, error) {
	// Convert reasoning effort levels to token budgets
	switch effort {
	case "low":
		return 5000, nil
	case "medium":
		return 15000, nil
	case "high":
		return 50000, nil
	default:
		return 15000, nil // Default to medium
	}
}

// Note: Audio content conversion not available - ChatCompletionContentPartAudioParam
// doesn't exist in current OpenAI SDK. Audio handling is done through modalities.

// convertFileContent converts OpenAI file content to Anthropic format
func (c *OpenAIToAnthropicConverter) convertFileContent(file openai.ChatCompletionContentPartFileParam) (*anthropic.ContentBlockParamUnion, error) {
	if file.File.FileData.Valid() {
		// Handle base64 file data
		data := file.File.FileData.Value
		filename := "file"
		if file.File.Filename.Valid() {
			filename = file.File.Filename.Value
		}

		// Try to determine if it's a PDF based on filename or content
		if strings.HasSuffix(strings.ToLower(filename), ".pdf") {
			return &anthropic.ContentBlockParamUnion{
				OfDocument: &anthropic.DocumentBlockParam{
					Type: "document",
					Source: anthropic.DocumentBlockParamSourceUnion{
						OfBase64: &anthropic.Base64PDFSourceParam{
							Type:      "base64",
							MediaType: "application/pdf",
							Data:      data,
						},
					},
					Title: anthropicparam.Opt[string]{Value: filename},
				},
			}, nil
		}

		// For other file types, decode and treat as text
		decodedData, err := base64.StdEncoding.DecodeString(data)
		if err != nil {
			// If not base64, treat as plain text
			return &anthropic.ContentBlockParamUnion{
				OfText: &anthropic.TextBlockParam{
					Type: "text",
					Text: fmt.Sprintf("[File: %s]\n%s", filename, data),
				},
			}, nil
		}

		return &anthropic.ContentBlockParamUnion{
			OfText: &anthropic.TextBlockParam{
				Type: "text",
				Text: fmt.Sprintf("[File: %s]\n%s", filename, string(decodedData)),
			},
		}, nil
	}

	if file.File.FileID.Valid() {
		// Handle file ID (treat as text reference)
		return &anthropic.ContentBlockParamUnion{
			OfText: &anthropic.TextBlockParam{
				Type: "text",
				Text: fmt.Sprintf("[File ID: %s]", file.File.FileID.Value),
			},
		}, nil
	}

	return &anthropic.ContentBlockParamUnion{
		OfText: &anthropic.TextBlockParam{
			Type: "text",
			Text: "[Unknown file content]",
		},
	}, nil
}
