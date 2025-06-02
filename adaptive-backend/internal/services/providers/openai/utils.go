package openai

import (
	"fmt"

	"adaptive-backend/internal/models"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
)

// convertToOpenAIMessageParams converts your messages to the new SDK's union types.
// This is a simple implementation: supports text system/user/assistant messages.
// Expand as needed for tools/function calls/multimodal.
func ConvertToOpenAIMessageParams(msgs []models.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	out := make([]openai.ChatCompletionMessageParamUnion, len(msgs))
	for i, m := range msgs {
		switch m.Role {
		case "user":
			out[i] = openai.UserMessage(m.Content)
		case "assistant":
			out[i] = openai.AssistantMessage(m.Content)
		case "system":
			out[i] = openai.SystemMessage(m.Content)
		default:
			return nil, fmt.Errorf("unknown message role: %s", m.Role)
		}
		// If you use tool/function/multimodal, add handling here.
	}
	return out, nil
}

// DetermineOpenAIModel uses OpenAI's official model enums; fallback to GPT-4o if unknown.
func DetermineOpenAIModel(requestedModel string) string {
	switch requestedModel {
	case shared.ChatModelGPT4o, shared.ChatModelGPT4_1, shared.ChatModelGPT4_1Mini, shared.ChatModelO3, shared.ChatModelO4Mini, shared.ChatModelGPT4_1Nano:
		return requestedModel
	default:
		return shared.ChatModelGPT4o // default fallback
	}
}