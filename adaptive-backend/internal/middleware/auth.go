package middleware

import (
	"log"
	"os"
	"strings"

	"adaptive-backend/internal/services/auth"

	"github.com/gofiber/fiber/v2"
)

// APIKeyAuth creates middleware for API key authentication
func APIKeyAuth() fiber.Handler {
	apiKeyService := auth.NewAPIKeyService()

	return func(c *fiber.Ctx) error {
		// Skip auth in development environment
		env := os.Getenv("ENV")
		if env != "production" {
			return c.Next()
		}

		// Get API key from X-Stainless-API-Key header
		apiKey := c.Get("X-Stainless-API-Key")
		if apiKey == "" {
			return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
				"error": "Missing API key. Please provide X-Stainless-API-Key header.",
				"code":  fiber.StatusUnauthorized,
			})
		}

		// Validate API key format (basic validation)
		apiKey = strings.TrimSpace(apiKey)
		if len(apiKey) < 10 {
			return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
				"error": "Invalid API key format.",
				"code":  fiber.StatusUnauthorized,
			})
		}

		// Verify API key with frontend service
		valid, err := apiKeyService.VerifyAPIKey(apiKey)
		if err != nil {
			log.Printf("API key verification error: %v", err)
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"error": "Internal server error during authentication.",
				"code":  fiber.StatusInternalServerError,
			})
		}

		if !valid {
			return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
				"error": "Invalid API key.",
				"code":  fiber.StatusUnauthorized,
			})
		}

		// Store API key in context for potential use in handlers
		c.Locals("api_key", apiKey)

		return c.Next()
	}
}
