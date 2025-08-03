package chat

import (
	"adaptive-backend/internal/services/providers/anthropic/chat/completions"
	"adaptive-backend/internal/services/providers/provider_interfaces"

	"github.com/openai/openai-go"
)

// AnthropicChat implements the Chat interface for Anthropic using OpenAI Go SDK
type AnthropicChat struct {
	client      *openai.Client
	completions *completions.AnthropicCompletions
}

func NewAnthropicChat(client *openai.Client) *AnthropicChat {
	completionsService := completions.NewAnthropicCompletions(client)

	return &AnthropicChat{
		client:      client,
		completions: completionsService,
	}
}

// Completions returns the Completions interface implementation
func (c *AnthropicChat) Completions() provider_interfaces.Completions {
	return c.completions
}
