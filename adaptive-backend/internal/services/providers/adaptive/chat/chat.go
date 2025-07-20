package chat

import (
	"adaptive-backend/internal/services/providers/adaptive/chat/completions"
	"adaptive-backend/internal/services/providers/provider_interfaces"

	"github.com/openai/openai-go"
)

// AdaptiveChat implements the Chat interface for Adaptive
type AdaptiveChat struct {
	client      *openai.Client
	completions *completions.AdaptiveCompletions
}

func NewAdaptiveChat(client *openai.Client) *AdaptiveChat {
	completionsService := completions.NewAdaptiveCompletions(client)

	return &AdaptiveChat{
		client:      client,
		completions: completionsService,
	}
}

// Completions returns the Completions interface implementation
func (c *AdaptiveChat) Completions() provider_interfaces.Completions {
	return c.completions
}
