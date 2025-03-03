package main

import (
	"adaptive-backend/internal/api"
	"log"
	"os"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
)

func main() {
	app := fiber.New(fiber.Config{
		AppName:           "Adaptive v1.0",
		EnablePrintRoutes: true,
	})

	// Middleware
	app.Use(logger.New())
	app.Use(recover.New())

	// Routes
	app.Get("/", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"message": "Welcome to Adaptive!",
		})
	})

	app.Post("/api/chat/completion", api.ChatCompletion)

	port := os.Getenv("ADDR")

	log.Fatal(app.Listen(port))
}
