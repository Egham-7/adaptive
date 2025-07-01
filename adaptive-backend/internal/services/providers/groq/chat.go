package groq

import "adaptive-backend/internal/services/providers/provider_interfaces"

// GroqChat implements the Chat interface
type GroqChat struct {
	service *GroqService
}

// Completions returns the completions interface
func (c *GroqChat) Completions() provider_interfaces.Completions {
	return &GroqCompletions{chat: c}
}