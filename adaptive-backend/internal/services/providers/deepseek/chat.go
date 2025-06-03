package deepseek

import "adaptive-backend/internal/services/providers/provider_interfaces"

// DeepSeekChat implements the Chat interface
type DeepSeekChat struct {
	service *DeepSeekService
}

// Completions returns the completions interface
func (c *DeepSeekChat) Completions() provider_interfaces.Completions {
	return &DeepSeekCompletions{chat: c}
}
