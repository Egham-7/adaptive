package models

type ChatCompletionRequest struct {
	Messages []Message `json:"messages"`
}

type ProviderChatCompletionRequest struct {
	Provider string    `json:"provider"`
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatCompletionResponse struct {
	Provider string `json:"provider"`
	Response any    `json:"response"`
	Error    string `json:"error,omitempty"`
}

// PromptRequest represents the prompt request body
type SelectModelRequest struct {
	Prompt string `json:"prompt"`
}

// ModelResponse represents the response from the select-model endpoint
type SelectModelResponse struct {
	SelectedModel string `json:"selected_model"`
	Provider      string `json:"provider"`
}
