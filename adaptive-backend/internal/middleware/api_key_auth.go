package middleware

import (
	"adaptive-backend/internal/api"

	"github.com/gofiber/fiber/v2"
)

// APIKeyMiddleware creates a middleware that authenticates requests using API keys
// from the X-API-Key header
func APIKeyMiddleware(apiKeyHandler *api.APIKeyHandler) fiber.Handler {
	return func(c *fiber.Ctx) error {
		// Use the existing VerifyAPIKey handler method
		err := apiKeyHandler.VerifyAPIKey(c)
		if err != nil {
			// The VerifyAPIKey method already sets the appropriate status code and error message
			return err
		}

		// If verification passed, continue to the next handler
		return c.Next()
	}
}
