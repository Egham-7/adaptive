package main

import (
	"adaptive-backend/config"
	"adaptive-backend/internal/api"
	"adaptive-backend/internal/middleware"
	"adaptive-backend/internal/services/metrics"
	"context"
	"fmt"
	"log"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/ansrivas/fiberprometheus/v2"
	"github.com/joho/godotenv"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/compress"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/limiter"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/pprof"
	"github.com/gofiber/fiber/v2/middleware/recover"
)

// SetupRoutes configures all the application routes
func SetupRoutes(app *fiber.App) {
	// Create handler instances
	apiKeyHandler := api.NewAPIKeyHandler()
	chatCompletionHandler := api.NewChatCompletionHandler()

	authMiddleware := middleware.AuthMiddleware()
	apiKeyMiddleware := middleware.APIKeyMiddleware(apiKeyHandler)

	// OpenAI-compatible API routes
	v1Group := app.Group("/v1")
	v1Group.Post("/chat/completions", apiKeyMiddleware, chatCompletionHandler.ChatCompletion)

	// API group for internal management
	apiGroup := app.Group("/api")

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

	// Initialize system metrics and start periodic collection
	systemMetrics := metrics.NewSystemMetrics()
	systemMetrics.StartPeriodicUpdates(30 * time.Second)
	log.Println("System metrics initialized and collection started")

	// Check required environment variables
	port := os.Getenv("ADDR")
	if port == "" {
		log.Fatal("ADDR environment variable is required but not set")
	}

	allowedOrigins := os.Getenv("ALLOWED_ORIGINS")
	if allowedOrigins == "" {
		log.Fatal("ALLOWED_ORIGINS environment variable is required but not set")
	}

	// Get environment-specific configuration
	isProd := os.Getenv("ENV") == "production"

	// Initialize Fiber app with optimized configuration
	app := fiber.New(fiber.Config{
		AppName:              "Adaptive v1.0",
		EnablePrintRoutes:    !isProd, // Disable route printing in production
		ReadTimeout:          2 * time.Minute,
		WriteTimeout:         2 * time.Minute,
		IdleTimeout:          5 * time.Minute,
		ReadBufferSize:       8192,  // Increased buffer size
		WriteBufferSize:      8192,  // Increased buffer size
		CompressedFileSuffix: ".gz", // Enable compression
		Prefork:              false, // Disable prefork for now
		CaseSensitive:        true,
		StrictRouting:        false,
		ServerHeader:         "Adaptive",
		ErrorHandler: func(c *fiber.Ctx, err error) error {
			code := fiber.StatusInternalServerError
			if e, ok := err.(*fiber.Error); ok {
				code = e.Code
			}

			// Log error for monitoring
			log.Printf("Request error: %v (status: %d, path: %s)", err, code, c.Path())

			return c.Status(code).JSON(fiber.Map{
				"error": err.Error(),
				"code":  code,
			})
		},
	})

	// Initialize database with optimized configuration
	dbConfig := config.DefaultDatabaseConfig()
	if isProd {
		// Production optimizations
		dbConfig.MaxOpenConns = 50
		dbConfig.MaxIdleConns = 20
		dbConfig.ConnMaxLifetime = 10 * time.Minute
	}

	db_err := config.InitializeWithConfig(
		os.Getenv("DB_SERVER"),
		os.Getenv("DB_NAME"),
		os.Getenv("DB_USER"),
		os.Getenv("DB_PASSWORD"),
		dbConfig,
	)
	if db_err != nil {
		log.Fatal(db_err)
	}

	// Setup middleware
	setupMiddleware(app, allowedOrigins)

	// Setup routes
	SetupRoutes(app)

	// Add welcome route
	app.Get("/", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"message":    "Welcome to Adaptive!",
			"version":    "1.0.0",
			"go_version": runtime.Version(),
			"status":     "running",
			"endpoints": map[string]string{
				"metrics":  "/metrics",
				"chat":     "/v1/chat/completions",
				"api_keys": "/api/api_keys",
			},
		})
	})

	// Add health check endpoint
	app.Get("/health", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"status":     "healthy",
			"timestamp":  time.Now(),
			"uptime":     time.Since(time.Now()).String(),
			"go_version": runtime.Version(),
			"goroutines": runtime.NumGoroutine(),
		})
	})

	// Graceful shutdown setup
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start server with graceful shutdown
	fmt.Printf("Server starting on %s with allowed origins: %s\n", port, allowedOrigins)
	fmt.Printf("Environment: %s\n", os.Getenv("ENV"))
	fmt.Printf("Go version: %s\n", runtime.Version())
	fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))

	// Start the server
	go func() {
		if err := app.Listen(port); err != nil {
			log.Printf("Server startup error: %v", err)
			cancel()
		}
	}()

	// Wait for shutdown signal
	<-ctx.Done()
	log.Println("Server shutting down...")
	if err := app.Shutdown(); err != nil {
		log.Printf("Server shutdown error: %v", err)
	}
}

// setupMiddleware configures all the application middleware
func setupMiddleware(app *fiber.App, allowedOrigins string) {
	isProd := os.Getenv("ENV") == "production"

	// Recovery middleware (must be first)
	app.Use(recover.New(recover.Config{
		EnableStackTrace: !isProd,
	}))

	// Compression middleware for better performance
	app.Use(compress.New(compress.Config{
		Level: compress.LevelBestSpeed, // Balance between speed and compression
	}))

	// Rate limiting to prevent abuse
	app.Use(limiter.New(limiter.Config{
		Max:               1000, // requests
		Expiration:        1 * time.Minute,
		LimiterMiddleware: limiter.SlidingWindow{},
		KeyGenerator: func(c *fiber.Ctx) string {
			// Use API key if available, otherwise IP
			apiKey := c.Get("X-Stainless-API-Key")
			if apiKey != "" {
				return apiKey
			}
			return c.IP()
		},
		LimitReached: func(c *fiber.Ctx) error {
			return c.Status(fiber.StatusTooManyRequests).JSON(fiber.Map{
				"error":       "Rate limit exceeded",
				"retry_after": "60 seconds",
			})
		},
	}))

	// Logger middleware with conditional configuration
	if isProd {
		app.Use(logger.New(logger.Config{
			Format: "${time} ${status} ${method} ${path} ${latency} ${bytesSent}b\n",
			Output: os.Stdout,
		}))
	} else {
		app.Use(logger.New(logger.Config{
			Format: "[${time}] ${status} - ${latency} ${method} ${path} ${error}\n",
			Output: os.Stdout,
		}))
	}

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

	// Join the headers into a comma-separated string
	allowedHeadersString := strings.Join(allAllowedHeaders, ", ")

	// CORS middleware with optimized configuration
	app.Use(cors.New(cors.Config{
		AllowOrigins:     allowedOrigins,
		AllowHeaders:     allowedHeadersString,
		AllowMethods:     "GET, POST, PUT, DELETE, OPTIONS",
		AllowCredentials: true,
		MaxAge:           86400, // 24 hours cache for preflight
		ExposeHeaders:    "Content-Length, Content-Type, X-Request-ID",
	}))

	// Prometheus metrics
	prometheus := fiberprometheus.New("adaptive-backend")
	prometheus.RegisterAt(app, "/metrics")
	app.Use(prometheus.Middleware)

	// pprof for performance profiling (development only)
	if !isProd {
		app.Use(pprof.New())
	}
}
