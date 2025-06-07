package anthropic

import (
	"adaptive-backend/internal/services/providers/provider_interfaces"
)

// AnthropicChat implements the Chat interface for Anthropic
type AnthropicChat struct {
	completions *AnthropicCompletions
}

// Completions implements Chat interface
func (c *AnthropicChat) Completions() provider_interfaces.Completions {
	return c.completions
}
