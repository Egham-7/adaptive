package main

import (
	"adaptive-backend/internal/api"
	"adaptive-backend/internal/middleware"
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/joho/godotenv"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/gofiber/fiber/v2/middleware/compress"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/limiter"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/pprof"
	"github.com/gofiber/fiber/v2/middleware/recover"
)

// SetupRoutes configures all the application routes for the Fiber app.
func SetupRoutes(app *fiber.App, healthHandler *api.HealthHandler) {
	chatCompletionHandler := api.NewCompletionHandler()

	// Health endpoint (no auth required)
	app.Get("/health", healthHandler.Health)

	// Apply JWT authentication to all v1 routes
	v1Group := app.Group("/v1", middleware.JWTAuth())
	v1Group.Post("/chat/completions", chatCompletionHandler.ChatCompletion)
}

const (
	defaultAppName    = "Adaptive v1.0"
	defaultVersion    = "1.0.0"
	healthEndpoint    = "/health"
	chatEndpoint      = "/v1/chat/completions"
	allowedMethods    = "GET, POST, PUT, DELETE, OPTIONS"
	allowedHeadersKey = "ALLOWED_ORIGINS"
	addrKey           = "ADDR"
	envKey            = "ENV"
	logLevelKey       = "LOG_LEVEL"
)

// main is the entry point for the Adaptive backend server.
func main() {
	if err := godotenv.Load(".env.local"); err != nil {
		fiberlog.Info("No .env.local file found, proceeding with environment variables")
	}

	// Set log level based on environment variable
	setupLogLevel()

	port := os.Getenv(addrKey)
	if port == "" {
		fiberlog.Fatal("ADDR environment variable is required but not set")
	}

	allowedOrigins := os.Getenv(allowedHeadersKey)
	if allowedOrigins == "" {
		fiberlog.Fatal("ALLOWED_ORIGINS environment variable is required but not set")
	}

	isProd := os.Getenv(envKey) == "production"

	app := fiber.New(fiber.Config{
		AppName:              defaultAppName,
		EnablePrintRoutes:    !isProd,
		ReadTimeout:          2 * time.Minute,
		WriteTimeout:         2 * time.Minute,
		IdleTimeout:          5 * time.Minute,
		ReadBufferSize:       8192,
		WriteBufferSize:      8192,
		CompressedFileSuffix: ".gz",
		Prefork:              false,
		CaseSensitive:        true,
		StrictRouting:        false,
		ServerHeader:         "Adaptive",
		ErrorHandler: func(c *fiber.Ctx, err error) error {
			code := fiber.StatusInternalServerError
			if e, ok := err.(*fiber.Error); ok {
				code = e.Code
			}
			fiberlog.Errorf("Request error: %v (status: %d, path: %s)", err, code, c.Path())
			return c.Status(code).JSON(fiber.Map{
				"error": err.Error(),
				"code":  code,
			})
		},
	})

	setupMiddleware(app, allowedOrigins)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Wait for services to become healthy before starting server
	healthHandler := api.NewHealthHandler()
	if err := healthHandler.WaitForServices(ctx, 10*time.Minute); err != nil {
		fiberlog.Errorf("Failed to wait for services: %v", err)
		return
	}

	SetupRoutes(app, healthHandler)

	app.Get("/", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"message":    "Welcome to Adaptive!",
			"version":    defaultVersion,
			"go_version": runtime.Version(),
			"status":     "running",
			"endpoints": map[string]string{
				"chat": chatEndpoint,
			},
		})
	})

	fmt.Printf("Server starting on %s with allowed origins: %s\n", port, allowedOrigins)
	fmt.Printf("Environment: %s\n", os.Getenv(envKey))
	fmt.Printf("Go version: %s\n", runtime.Version())
	fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))

	go func() {
		if err := app.Listen(port); err != nil {
			fiberlog.Errorf("Server startup error: %v", err)
			cancel()
		}
	}()

	<-ctx.Done()
	fiberlog.Info("Server shutting down...")
	if err := app.Shutdown(); err != nil {
		fiberlog.Errorf("Server shutdown error: %v", err)
	}
}

// setupMiddleware configures all the application middleware for the Fiber app.
func setupMiddleware(app *fiber.App, allowedOrigins string) {
	isProd := os.Getenv(envKey) == "production"

	app.Use(recover.New(recover.Config{
		EnableStackTrace: !isProd,
	}))

	app.Use(compress.New(compress.Config{
		Level: compress.LevelBestSpeed,
	}))

	app.Use(limiter.New(limiter.Config{
		Max:               1000,
		Expiration:        1 * time.Minute,
		LimiterMiddleware: limiter.SlidingWindow{},
		KeyGenerator: func(c *fiber.Ctx) string {
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

	allowedHeadersString := strings.Join(allAllowedHeaders, ", ")

	app.Use(cors.New(cors.Config{
		AllowOrigins:     allowedOrigins,
		AllowHeaders:     allowedHeadersString,
		AllowMethods:     allowedMethods,
		AllowCredentials: true,
		MaxAge:           86400,
		ExposeHeaders:    "Content-Length, Content-Type, X-Request-ID",
	}))

	if !isProd {
		app.Use(pprof.New())
	}
}

// setupLogLevel configures the Fiber log level based on environment variable
func setupLogLevel() {
	logLevel := strings.ToLower(os.Getenv(logLevelKey))
	if logLevel == "" {
		logLevel = "info" // default to info if not set
	}

	switch logLevel {
	case "trace":
		fiberlog.SetLevel(fiberlog.LevelTrace)
	case "debug":
		fiberlog.SetLevel(fiberlog.LevelDebug)
	case "info":
		fiberlog.SetLevel(fiberlog.LevelInfo)
	case "warn", "warning":
		fiberlog.SetLevel(fiberlog.LevelWarn)
	case "error":
		fiberlog.SetLevel(fiberlog.LevelError)
	case "fatal":
		fiberlog.SetLevel(fiberlog.LevelFatal)
	case "panic":
		fiberlog.SetLevel(fiberlog.LevelPanic)
	default:
		fiberlog.SetLevel(fiberlog.LevelInfo)
		fiberlog.Warnf("Unknown log level '%s', defaulting to 'info'", logLevel)
	}

	fiberlog.Infof("Log level set to: %s", logLevel)
}
