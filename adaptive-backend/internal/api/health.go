package api

import (
	"context"
	"runtime"
	"time"

	"adaptive-backend/internal/config"
	"adaptive-backend/internal/services"

	"github.com/gofiber/fiber/v2"
	"github.com/redis/go-redis/v9"
)

// HealthHandler handles health check requests
type HealthHandler struct {
	cfg         *config.Config
	redisClient *redis.Client
	apiClient   *services.Client
}

// NewHealthHandler creates a new health check handler
func NewHealthHandler(cfg *config.Config, redisClient *redis.Client) *HealthHandler {
	var apiClient *services.Client
	if cfg.Services.ModelRouter.Client.AdaptiveRouterURL != "" {
		apiClient = services.NewClient(cfg.Services.ModelRouter.Client.AdaptiveRouterURL)
	}

	return &HealthHandler{
		cfg:         cfg,
		redisClient: redisClient,
		apiClient:   apiClient,
	}
}

// HealthCheck returns the health status of the service and its dependencies
func (h *HealthHandler) HealthCheck(c *fiber.Ctx) error {
	startTime := time.Now()

	// Check Redis connectivity
	redisStatus := h.checkRedis()

	// Check adaptive_ai service
	aiServiceStatus := h.checkAIService()

	responseTime := time.Since(startTime)

	// Determine overall health status
	overallStatus := "healthy"
	statusCode := fiber.StatusOK

	if redisStatus != "healthy" || aiServiceStatus != "healthy" {
		overallStatus = "degraded"
		statusCode = fiber.StatusServiceUnavailable
	}

	response := fiber.Map{
		"status":    overallStatus,
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"service":   "adaptive-backend",
		"version":   "1.0.0",
		"checks": fiber.Map{
			"redis":      redisStatus,
			"ai_service": aiServiceStatus,
		},
		"uptime":       time.Since(startTime).Seconds(),
		"responseTime": responseTime.String(),
		"runtime": fiber.Map{
			"go_version": runtime.Version(),
			"num_cpu":    runtime.NumCPU(),
			"goroutines": runtime.NumGoroutine(),
		},
	}

	return c.Status(statusCode).JSON(response)
}

// checkRedis verifies Redis connectivity
func (h *HealthHandler) checkRedis() string {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := h.redisClient.Ping(ctx).Err(); err != nil {
		return "unhealthy"
	}

	return "healthy"
}

// checkAIService verifies adaptive_ai service connectivity
func (h *HealthHandler) checkAIService() string {
	if h.apiClient == nil {
		return "unknown"
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	var healthResp map[string]any
	err := h.apiClient.Get("/health", &healthResp, &services.RequestOptions{
		Context: ctx,
		Retries: 0,
	})

	if err != nil {
		return "unhealthy"
	}

	return "healthy"
}
