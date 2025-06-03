package chat

import (
	"adaptive-backend/internal/services/providers/openai/chat/completions"
	"adaptive-backend/internal/services/providers/provider_interfaces"

	"github.com/openai/openai-go"
)

// OpenAIChat implements the Chat interface for OpenAI
type OpenAIChat struct {
	client      *openai.Client
	completions *completions.OpenAICompletions
}

func NewOpenAIChat(client *openai.Client) *OpenAIChat {
	completionsService := completions.NewOpenAICompletions(client)

	return &OpenAIChat{ // Return pointer
		client:      client,
		completions: completionsService, // Type assertion since NewOpenAICompletions returns interface
	}
}

// Completions returns the Completions interface implementation - Fixed receiver
func (c *OpenAIChat) Completions() provider_interfaces.Completions {
	return c.completions
}
