package models

type ChatCompletionRequest struct {
	Messages  []Message `json:"messages"`
	Model     string    `json:"model,omitempty"`
	MaxTokens int       `json:"max_tokens,omitempty"`
}

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatCompletionResponse struct {
	Provider string      `json:"provider"`
	Response interface{} `json:"response"`
	Error    string      `json:"error,omitempty"`
}
