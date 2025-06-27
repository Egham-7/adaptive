package chat

import (
	"adaptive-backend/internal/services/providers/huggingface/chat/completions"
	"adaptive-backend/internal/services/providers/provider_interfaces"

	"github.com/openai/openai-go"
)

// HuggingFaceChat implements the Chat interface for HuggingFace
type HuggingFaceChat struct {
	client      *openai.Client
	completions *completions.HuggingFaceCompletions
}

func NewHuggingFaceChat(client *openai.Client) *HuggingFaceChat {
	completionsService := completions.NewHuggingFaceCompletions(client)

	return &HuggingFaceChat{
		client:      client,
		completions: completionsService,
	}
}

// Completions returns the Completions interface implementation
func (c *HuggingFaceChat) Completions() provider_interfaces.Completions {
	return c.completions
}