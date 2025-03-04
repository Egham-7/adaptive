package main

import (
	"adaptive-backend/internal/api"
	"fmt"
	"log"
	"os"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
)

func main() {
	// Check required environment variables
	port := os.Getenv("ADDR")
	if port == "" {
		log.Fatal("ADDR environment variable is required but not set")
	}

	allowedOrigins := os.Getenv("ALLOWED_ORIGINS")
	if allowedOrigins == "" {
		log.Fatal("ALLOWED_ORIGINS environment variable is required but not set")
	}

	app := fiber.New(fiber.Config{
		AppName:           "Adaptive v1.0",
		EnablePrintRoutes: true,
	})

	// Middleware
	app.Use(logger.New())
	app.Use(recover.New())

	// Add CORS middleware with required environment variable
	app.Use(cors.New(cors.Config{
		AllowOrigins:     allowedOrigins,
		AllowHeaders:     "Origin, Content-Type, Accept, Authorization",
		AllowMethods:     "GET, POST, PUT, DELETE, OPTIONS",
		AllowCredentials: true,
	}))

	// Routes
	app.Get("/", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"message": "Welcome to Adaptive!",
		})
	})

	app.Post("/api/chat/completion", api.ChatCompletion)

	fmt.Printf("Server starting on %s with allowed origins: %s\n", port, allowedOrigins)
	log.Fatal(app.Listen(port))
}
