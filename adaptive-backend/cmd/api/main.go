package main

import (
	"adaptive-backend/config"
	"adaptive-backend/internal/api"
	"fmt"
	"log"
	"os"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
)

// SetupRoutes configures all the application routes
func SetupRoutes(app *fiber.App) {
	// Create handler instances
	conversationHandler := api.NewConversationHandler()
	messageHandler := api.NewMessageHandler()

	// API group
	apiGroup := app.Group("/api")

	// Chat completion endpoint
	apiGroup.Post("/chat/completion", api.ChatCompletion)

	// Conversation routes
	conversations := apiGroup.Group("/conversations")
	conversations.Get("/", conversationHandler.GetAllConversations)
	conversations.Get("/:id", conversationHandler.GetConversation)
	conversations.Post("/", conversationHandler.CreateConversation)
	conversations.Put("/:id", conversationHandler.UpdateConversation)
	conversations.Delete("/:id", conversationHandler.DeleteConversation)

	// Message routes related to conversations
	conversations.Get("/:id/messages", messageHandler.GetMessagesByConversation)
	conversations.Post("/:id/messages", messageHandler.CreateMessage)
	conversations.Delete("/:id/messages/", messageHandler.DeleteAllConversationMessages)

	// Individual message routes
	messages := apiGroup.Group("/messages")
	messages.Get("/:id", messageHandler.GetMessage)
	messages.Put("/:id", messageHandler.UpdateMessage)
	messages.Delete("/:id", messageHandler.DeleteMessage)
	messages.Delete("/batch", messageHandler.BatchDeleteMessages)
}

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

	// Initialize Fiber app
	app := fiber.New(fiber.Config{
		AppName:           "Adaptive v1.0",
		EnablePrintRoutes: true,
	})
	config.Initialize("adaptive.db")

	// Setup middleware
	setupMiddleware(app, allowedOrigins)

	// Setup routes
	SetupRoutes(app)

	// Add welcome route
	app.Get("/", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"message": "Welcome to Adaptive!",
		})
	})

	// Start server
	fmt.Printf("Server starting on %s with allowed origins: %s\n", port, allowedOrigins)
	log.Fatal(app.Listen(port))
}

// setupMiddleware configures all the application middleware
func setupMiddleware(app *fiber.App, allowedOrigins string) {
	app.Use(logger.New())
	app.Use(recover.New())

	// Add CORS middleware with required environment variable
	app.Use(cors.New(cors.Config{
		AllowOrigins:     allowedOrigins,
		AllowHeaders:     "Origin, Content-Type, Accept, Authorization",
		AllowMethods:     "GET, POST, PUT, DELETE, OPTIONS",
		AllowCredentials: true,
	}))
}
