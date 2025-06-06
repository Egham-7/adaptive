package main

import (
	"adaptive-backend/config"
	"adaptive-backend/internal/api"
	"adaptive-backend/internal/middleware"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/ansrivas/fiberprometheus/v2"
	"github.com/joho/godotenv"
	"github.com/stripe/stripe-go/v82"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
)

// SetupRoutes configures all the application routes
func SetupRoutes(app *fiber.App) {
	// Create handler instances
	apiKeyHandler := api.NewAPIKeyHandler()
	chatCompletionHandler := api.NewChatCompletionHandler()

	authMiddleware := middleware.AuthMiddleware()
	apiKeyMiddleware := middleware.APIKeyMiddleware(apiKeyHandler)

	// API group
	apiGroup := app.Group("/api")

	chatCompletions := apiGroup.Group("/chat/completions", apiKeyMiddleware)

	chatCompletions.Post("/", chatCompletionHandler.ChatCompletion)

	// API key routes
	apiKeys := apiGroup.Group("/api_keys", authMiddleware)
	apiKeys.Get("/:userId", apiKeyHandler.GetAllAPIKeysByUserId)
	apiKeys.Get("/:id", apiKeyHandler.GetAPIKeyById)
	apiKeys.Post("/", apiKeyHandler.CreateAPIKey)
	apiKeys.Put("/:id", apiKeyHandler.UpdateAPIKey)
	apiKeys.Delete("/:id", apiKeyHandler.DeleteAPIKey)
}

func main() {
	err := godotenv.Load(".env.local")
	if err != nil {
		log.Println("No .env.local file found, proceeding with environment variables")
	}

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
		ReadTimeout:       2 * time.Minute,
		WriteTimeout:      2 * time.Minute,
		IdleTimeout:       5 * time.Minute,
	})

	db_err := config.Initialize(os.Getenv("DB_SERVER"), os.Getenv("DB_NAME"), os.Getenv("DB_USER"), os.Getenv("DB_PASSWORD"))
	if db_err != nil {
		log.Fatal(db_err)
	}

	stripe.Key = os.Getenv("STRIPE_SECRET_KEY")

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

	allAllowedHeaders := []string{
		"Origin",
		"Content-Type",
		"Accept",
		"Authorization",
		"User-Agent",
		"X-Stainless-API-Key",
		"X-Stainless-Arch",
		"X-Stainless-OS",
		"X-Stainless-Runtime",
		"X-Stainless-Runtime-Version",
		"X-Stainless-Package-Version",
		"X-Stainless-Lang",
		"X-Stainless-Retry-Count",
		"X-Stainless-Read-Timeout",
		"X-Stainless-Async",
		"X-Stainless-Raw-Response",
		"X-Stainless-Helper-Method",
		"X-Stainless-Timeout",
	}

	// Join the headers into a comma-separated string, as required by rs/cors Config.AllowHeaders
	allowedHeadersString := strings.Join(allAllowedHeaders, ", ")

	// Add CORS middleware with required environment variable
	app.Use(cors.New(cors.Config{
		AllowOrigins: allowedOrigins,
		AllowHeaders: allowedHeadersString,

		AllowMethods:     "GET, POST, PUT, DELETE, OPTIONS",
		AllowCredentials: true,
		MaxAge:           86400, // Preflight requests can be cached for 24 hours
		ExposeHeaders:    "Content-Length, Content-Type",
	}))

	prometheus := fiberprometheus.New("adaptive-backend")
	prometheus.RegisterAt(app, "/metrics")
	app.Use(prometheus.Middleware)
}
