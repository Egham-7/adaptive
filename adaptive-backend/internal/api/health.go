package api

import (
	"adaptive-backend/internal/services"
	"context"
	"time"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// HealthHandler handles health check requests
type HealthHandler struct {
	healthService *services.HealthService
}

// NewHealthHandler creates a new health handler
func NewHealthHandler() *HealthHandler {
	healthService := services.NewHealthService()
	return &HealthHandler{
		healthService: healthService,
	}
}

// Health handles GET /health requests
func (h *HealthHandler) Health(c *fiber.Ctx) error {
	ctx, cancel := context.WithTimeout(c.Context(), 30*time.Second)
	defer cancel()

	// Check if services are ready
	if !h.healthService.IsReady() {
		return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{
			"status":  "unhealthy",
			"message": "Services are still starting up (loading models and classifiers)",
			"ready":   false,
		})
	}

	// Perform health check
	health := h.healthService.CheckHealth(ctx)

	if health.Healthy {
		return c.JSON(fiber.Map{
			"status":    "healthy",
			"message":   health.Message,
			"services":  health.Services,
			"timestamp": time.Now(),
		})
	}

	// Return unhealthy status with service details
	fiberlog.Warnf("Health check failed: %s", health.Message)
	return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{
		"status":    "unhealthy",
		"message":   health.Message,
		"services":  health.Services,
		"timestamp": time.Now(),
	})
}
