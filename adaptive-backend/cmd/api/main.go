package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"

	"adaptive-backend/internal/api"
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/cache"
	"adaptive-backend/internal/services/chat/completions"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/model_router"
	"adaptive-backend/internal/services/select_model"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/gofiber/fiber/v2/middleware/compress"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/limiter"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/pprof"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/redis/go-redis/v9"
)

// SetupRoutes configures all the application routes for the Fiber app.
func SetupRoutes(app *fiber.App, cfg *config.Config, redisClient *redis.Client) error {
	// Create shared services once
	reqSvc := completions.NewRequestService()

	// Create protocol manager (shared between handlers)
	modelRouter, err := model_router.NewModelRouter(cfg, redisClient)
	if err != nil {
		return fmt.Errorf("protocol manager initialization failed: %w", err)
	}

	// Create prompt caches (shared services using Redis client)
	openaiPromptCache, err := cache.NewOpenAIPromptCache(redisClient, cfg.PromptCache, cfg.Services.Redis.URL)
	if err != nil {
		return fmt.Errorf("openAI prompt cache initialization failed: %w", err)
	}

	anthropicPromptCache, err := cache.NewAnthropicPromptCache(redisClient, cfg.PromptCache, cfg.Services.Redis.URL)
	if err != nil {
		return fmt.Errorf("anthropic prompt cache initialization failed: %w", err)
	}

	// Initialize Gemini prompt cache with Redis DB 2
	geminiPromptCache, err := cache.NewGeminiPromptCache(redisClient, cfg.PromptCache, cfg.Services.Redis.URL)
	if err != nil {
		return fmt.Errorf("gemini prompt cache initialization failed: %w", err)
	}

	// Create response service (depends on model router)
	respSvc := completions.NewResponseService(modelRouter)

	// Create completion service (depends on response service)
	completionSvc := completions.NewCompletionService(cfg, respSvc)

	// Create shared circuit breakers for all providers across all services
	circuitBreakers := make(map[string]*circuitbreaker.CircuitBreaker)
	providerTypes := []string{"chat_completions", "messages", "generate"}

	for _, serviceType := range providerTypes {
		for providerName := range cfg.GetProviders(serviceType) {
			if _, exists := circuitBreakers[providerName]; !exists {
				circuitBreakers[providerName] = circuitbreaker.NewForProvider(redisClient, providerName)
			}
		}
	}

	// Create select model services
	selectModelReqSvc := select_model.NewRequestService()
	selectModelSvc := select_model.NewService(modelRouter)
	selectModelRespSvc := select_model.NewResponseService()

	// Initialize handlers with shared dependencies
	chatCompletionHandler := api.NewCompletionHandler(cfg, reqSvc, respSvc, completionSvc, modelRouter, openaiPromptCache, circuitBreakers)
	selectModelHandler := api.NewSelectModelHandler(cfg, selectModelReqSvc, selectModelSvc, selectModelRespSvc, circuitBreakers)
	messagesHandler := api.NewMessagesHandler(cfg, modelRouter, anthropicPromptCache, circuitBreakers)
	generateHandler := api.NewGenerateHandler(cfg, modelRouter, geminiPromptCache, circuitBreakers)

	// Setup v1 routes for internal communication (no authentication needed)
	v1Group := app.Group("/v1")
	v1Group.Post("/chat/completions", chatCompletionHandler.ChatCompletion)
	v1Group.Post("/messages", messagesHandler.Messages)
	v1Group.Post("/select-model", selectModelHandler.SelectModel)
	v1Group.Post("/generate", generateHandler.Generate)
	v1Group.Post("/generate/stream", generateHandler.StreamGenerate)

	return nil
}

const (
	defaultAppName         = "Adaptive v1.0"
	defaultVersion         = "1.0.0"
	chatEndpoint           = "/v1/chat/completions"
	messagesEndpoint       = "/v1/messages"
	selectModelEndpoint    = "/v1/select-model"
	generateEndpoint       = "/v1/generate"
	generateStreamEndpoint = "/v1/generate/stream"
	allowedMethods         = "GET, POST, PUT, DELETE, OPTIONS"
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

	port := cfg.Server.Port
	if port == "" {
		port = "8080" // Default port
	}
	listenAddr := ":" + port // Dual-stack IPv4/IPv6 binding
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
		Network:              "tcp",
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

	// Create shared Redis client for distributed services
	redisClient, err := createRedisClient(cfg)
	if err != nil {
		fiberlog.Fatalf("Failed to create Redis client: %v", err)
	}
	defer func() {
		if err := redisClient.Close(); err != nil {
			fiberlog.Errorf("Failed to close Redis client: %v", err)
		}
	}()
	fiberlog.Info("Redis client initialized successfully")

	if err := SetupRoutes(app, cfg, redisClient); err != nil {
		fiberlog.Fatalf("Failed to setup routes: %v", err)
	}

	app.Get("/", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"message":    "Welcome to Adaptive!",
			"version":    defaultVersion,
			"go_version": runtime.Version(),
			"status":     "running",
			"endpoints": map[string]string{
				"chat":            chatEndpoint,
				"messages":        messagesEndpoint,
				"select-model":    selectModelEndpoint,
				"generate":        generateEndpoint,
				"generate-stream": generateStreamEndpoint,
			},
		})
	})

	fmt.Printf("Server starting on %s (dual-stack IPv4/IPv6) with allowed origins: %s\n", listenAddr, allowedOrigins)
	fmt.Printf("Environment: %s\n", cfg.Server.Environment)
	fmt.Printf("Go version: %s\n", runtime.Version())
	fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))

	// Create a channel to listen for interrupt signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM, syscall.SIGINT)

	// Start server in a goroutine
	serverErrChan := make(chan error, 1)
	go func() {
		if err := app.Listen(listenAddr); err != nil {
			serverErrChan <- err
		}
	}()

	// Block until we receive a signal or server error
	select {
	case sig := <-sigChan:
		fiberlog.Infof("Received signal: %v. Starting graceful shutdown...", sig)
	case err := <-serverErrChan:
		fiberlog.Errorf("Server startup error: %v", err)
		cancel()
		return
	case <-ctx.Done():
		fiberlog.Info("Context cancelled, starting shutdown...")
	}

	// Graceful shutdown with timeout
	fiberlog.Info("Server shutting down gracefully...")
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	// Shutdown server with timeout
	shutdownErrChan := make(chan error, 1)
	go func() {
		shutdownErrChan <- app.ShutdownWithTimeout(30 * time.Second)
	}()

	select {
	case err := <-shutdownErrChan:
		if err != nil {
			fiberlog.Errorf("Server shutdown error: %v", err)
		} else {
			fiberlog.Info("Server shutdown completed successfully")
		}
	case <-shutdownCtx.Done():
		fiberlog.Error("Server shutdown timeout exceeded")
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

// createRedisClient creates a shared Redis client for distributed services with optimized pooling
func createRedisClient(cfg *config.Config) (*redis.Client, error) {
	// Get Redis connection configuration
	redisURL := cfg.Services.Redis.URL
	if redisURL == "" {
		return nil, fmt.Errorf("redis URL not set in configuration (services.redis.url)")
	}

	// Parse Redis URL
	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Redis URL: %w", err)
	}

	// Configure connection pool settings for high-throughput scenarios
	opt.PoolSize = 20                      // Maximum number of socket connections
	opt.MinIdleConns = 5                   // Minimum idle connections to maintain
	opt.PoolTimeout = 4 * time.Second      // Connection pool timeout
	opt.ConnMaxIdleTime = 5 * time.Minute  // Close connections after remaining idle
	opt.ConnMaxLifetime = 30 * time.Minute // Maximum connection lifetime

	// Connection timeout settings
	opt.DialTimeout = 10 * time.Second // Timeout for establishing new connections
	opt.ReadTimeout = 3 * time.Second  // Socket read timeout
	opt.WriteTimeout = 3 * time.Second // Socket write timeout

	// Retry settings for connection failures
	opt.MaxRetries = 3                           // Maximum number of retries for failed commands
	opt.MinRetryBackoff = 8 * time.Millisecond   // Minimum backoff between retries
	opt.MaxRetryBackoff = 512 * time.Millisecond // Maximum backoff between retries

	fiberlog.Debugf("Redis client configuration: PoolSize=%d, MinIdle=%d, MaxRetries=%d",
		opt.PoolSize, opt.MinIdleConns, opt.MaxRetries)

	// Create Redis client with optimized settings
	client := redis.NewClient(opt)

	// Test connection with retries
	return testRedisConnectionWithRetry(client)
}

// testRedisConnectionWithRetry tests Redis connection with retry logic
func testRedisConnectionWithRetry(client *redis.Client) (*redis.Client, error) {
	const maxAttempts = 3
	const baseDelay = 1 * time.Second

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)

		err := client.Ping(ctx).Err()
		cancel()

		if err == nil {
			fiberlog.Infof("Redis connection established successfully (attempt %d/%d)", attempt, maxAttempts)

			// Log pool stats for monitoring
			stats := client.PoolStats()
			fiberlog.Debugf("Redis pool initialized: Hits=%d, Misses=%d, Timeouts=%d, TotalConns=%d, IdleConns=%d",
				stats.Hits, stats.Misses, stats.Timeouts, stats.TotalConns, stats.IdleConns)

			return client, nil
		}

		fiberlog.Warnf("Redis connection failed (attempt %d/%d): %v", attempt, maxAttempts, err)

		if attempt < maxAttempts {
			delay := time.Duration(attempt) * baseDelay
			fiberlog.Infof("Retrying Redis connection in %v...", delay)
			time.Sleep(delay)
		}
	}

	// Close the client if all attempts failed
	if err := client.Close(); err != nil {
		fiberlog.Errorf("Failed to close Redis client after connection failures: %v", err)
	}

	return nil, fmt.Errorf("failed to connect to Redis after %d attempts", maxAttempts)
}
