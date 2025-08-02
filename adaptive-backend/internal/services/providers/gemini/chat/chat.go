package chat

import (
	"adaptive-backend/internal/services/providers/gemini/chat/completions"
	"adaptive-backend/internal/services/providers/provider_interfaces"

	"github.com/openai/openai-go"
)

// GeminiChat implements the Chat interface for Gemini using OpenAI Go SDK
type GeminiChat struct {
	client      *openai.Client
	completions *completions.GeminiCompletions
}

func NewGeminiChat(client *openai.Client) *GeminiChat {
	completionsService := completions.NewGeminiCompletions(client)

	return &GeminiChat{
		client:      client,
		completions: completionsService,
	}
}

// Completions returns the Completions interface implementation
func (c *GeminiChat) Completions() provider_interfaces.Completions {
	return c.completions
}
