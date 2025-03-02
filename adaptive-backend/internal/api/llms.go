package api

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services"

	"github.com/gofiber/fiber/v2"
)

// ChatCompletionRequest is the request body for the chat completion endpoint
type ChatCompletionRequest struct {
	Request models.ChatCompletionRequest `json:"request"`
}

// ChatCompletion handles chat completion requests
func ChatCompletion(c *fiber.Ctx) error {
	// Parse request body
	var req ChatCompletionRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body: " + err.Error(),
		})
	}

	// Validate request
	if req.Provider == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Provider is required",
		})
	}

	// Get the appropriate LLM provider
	provider, err := services.NewLLMProvider(req.Provider)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	// Call the provider's chat completion
	resp, err := provider.CreateChatCompletion(&req.Request)
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
