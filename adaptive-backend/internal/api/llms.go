package api

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services"

	"github.com/gofiber/fiber/v2"
)

func ChatCompletion(c *fiber.Ctx) error {
	// Parse request body
	var req models.ChatCompletionRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body: " + err.Error(),
		})
	}

	prompt_classifier_client := services.NewPromptClassifierClient()

	prompt := req.Messages[len(req.Messages)-1]

	selected_model, err := prompt_classifier_client.SelectModel(prompt.Content)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	full_chat_completion_req := models.ProviderChatCompletionRequest{
		Provider:         selected_model.Provider,
		Model:            selected_model.SelectedModel,
		Messages:         req.Messages,
		Temperature:      selected_model.Parameters.Temperature,
		N:                selected_model.Parameters.N,
		MaxTokens:        selected_model.Parameters.MaxTokens,
		PresencePenalty:  selected_model.Parameters.PresencePenalty,
		FrequencyPenalty: selected_model.Parameters.FrequencyPenalty,
	}

	// Get the appropriate LLM provider
	provider, err := services.NewLLMProvider(full_chat_completion_req.Provider)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	// Call the provider's chat completion
	resp, err := provider.CreateChatCompletion(&full_chat_completion_req)
	if err != nil {
		// If there's an error but we have a response with error details
		if resp != nil && resp.Error != "" {
			return c.Status(fiber.StatusInternalServerError).JSON(resp)
		}
		// Generic error
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to generate completion: " + err.Error(),
		})
	}

	// Return successful response
	return c.JSON(resp)
}
