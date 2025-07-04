package grok

import "adaptive-backend/internal/services/providers/provider_interfaces"

// GrokChat implements the Chat interface
type GrokChat struct {
	service *GrokService
}

// Completions returns the completions interface
func (c *GrokChat) Completions() provider_interfaces.Completions {
	return &GrokCompletions{chat: c}
}
