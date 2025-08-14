package main

import (
	"adaptive-backend/internal/api"
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/middleware"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/chat/completions"
	"adaptive-backend/internal/services/protocol_manager"
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

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
func SetupRoutes(app *fiber.App, cfg *config.Config, healthHandler *api.HealthHandler) {
	// Create shared services once
	reqSvc := completions.NewRequestService()
	paramSvc := completions.NewParameterService()
	fallbackSvc := completions.NewFallbackService(cfg)

	// Create protocol manager (shared between handlers)
	protocolMgr, err := protocol_manager.NewProtocolManager(cfg)
	if err != nil {
		fiberlog.Fatalf("protocol manager initialization failed: %v", err)
	}

	// Create response service (depends on protocol manager)
	respSvc := completions.NewResponseService(cfg, protocolMgr)

	// Initialize handlers with shared dependencies
	chatCompletionHandler := api.NewCompletionHandler(reqSvc, respSvc, paramSvc, protocolMgr, fallbackSvc)
	selectModelHandler := api.NewSelectModelHandler(reqSvc, protocolMgr)

	// Health endpoint (no auth required)
	app.Get("/health", healthHandler.Health)

	// Apply JWT authentication to all v1 routes
	v1Group := app.Group("/v1", middleware.JWTAuth(cfg))
	v1Group.Post("/chat/completions", chatCompletionHandler.ChatCompletion)
	v1Group.Post("/select-model", selectModelHandler.SelectModel)
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
	// Load configuration
	cfg, err := config.New()
	if err != nil {
		fiberlog.Fatal("Failed to load configuration: " + err.Error())
	}

	// Validate required configuration
	if err := cfg.Validate(); err != nil {
		fiberlog.Fatal(err.Error())
	}

	// Set log level based on configuration
	setupLogLevel(cfg)

	port := cfg.Server.Addr
	allowedOrigins := cfg.Server.AllowedOrigins
	isProd := cfg.IsProduction()

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
			// Sanitize error for external consumption
			sanitized := models.SanitizeError(err)
			statusCode := sanitized.GetStatusCode()
			
			// Log internal error details (but don't expose them)
			if isProd {
				fiberlog.Errorf("Request error: path=%s, type=%s, retryable=%v", 
					c.Path(), sanitized.Type, sanitized.Retryable)
			} else {
				fiberlog.Errorf("Request error: %v (status: %d, path: %s)", err, statusCode, c.Path())
			}
			
			// Return sanitized error response
			response := fiber.Map{
				"error": sanitized.Message,
				"type":  sanitized.Type,
				"code":  statusCode,
			}
			
			// Add retry info for retryable errors
			if sanitized.Retryable {
				response["retryable"] = true
				if sanitized.Type == models.ErrorTypeRateLimit {
					response["retry_after"] = "60s"
				}
			}
			
			// Add error code if available
			if sanitized.Code != "" {
				response["error_code"] = sanitized.Code
			}
			
			return c.Status(statusCode).JSON(response)
		},
	})

	setupMiddleware(app, cfg, allowedOrigins)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Wait for services to become healthy before starting server
	healthHandler := api.NewHealthHandler()
	if err := healthHandler.WaitForServices(ctx, 10*time.Minute); err != nil {
		fiberlog.Errorf("Failed to wait for services: %v", err)
		return
	}

	SetupRoutes(app, cfg, healthHandler)

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
	fmt.Printf("Environment: %s\n", cfg.Server.Environment)
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
func setupMiddleware(app *fiber.App, cfg *config.Config, allowedOrigins string) {
	isProd := cfg.IsProduction()

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
			err := models.NewRateLimitError("1000 requests per minute")
			return err
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

// setupLogLevel configures the Fiber log level based on configuration
func setupLogLevel(cfg *config.Config) {
	logLevel := cfg.GetNormalizedLogLevel()

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
