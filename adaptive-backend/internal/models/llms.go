package models

// PromptRequest represents the prompt request body
type SelectModelRequest struct {
	Prompt string `json:"prompt"`
}

// SelectModelResponse represents the response from the select-model endpoint
type SelectModelResponse struct {
	SelectedModel string           `json:"selected_model"`
	Provider      string           `json:"provider"`
	Parameters    OpenAIParameters `json:"parameters"`
}

type OpenAIParameters struct {
	MaxTokens        int64   `json:"max_tokens,omitempty"`
	Temperature      float64 `json:"temperature,omitempty"`
	TopP             float64 `json:"top_p,omitempty"`
	PresencePenalty  float64 `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64 `json:"frequency_penalty,omitempty"`
	N                int64   `json:"n,omitempty"`
}
