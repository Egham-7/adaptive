package gemini

import "adaptive-backend/internal/services/providers/provider_interfaces"

// GeminiChat implements the Chat interface
type GeminiChat struct {
	service *GeminiService
}

// Completions returns the completions interface
func (c *GeminiChat) Completions() provider_interfaces.Completions {
	return &GeminiCompletions{chat: c}
}
