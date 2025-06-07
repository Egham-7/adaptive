package api

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/usage"
	"context"
	"os"
	"time"

	"github.com/clerk/clerk-sdk-go/v2"
	"github.com/clerk/clerk-sdk-go/v2/user"
	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

type APIKeyHandler struct {
	service *usage.APIKeyService
}

type CreateAPIKeyRequest struct {
	Name      string `json:"name"`
	Status    string `json:"status"`
	ExpiresAt string `json:"expires_at,omitempty"`
}

type CreateAPIKeyResponse struct {
	APIKey     models.APIKey `json:"api_key"`
	FullAPIKey string        `json:"full_api_key"`
}

type UpdateAPIKeyRequest struct {
	Name   string `json:"name"`
	Status string `json:"status"`
}

type VerifyAPIKeyResponse struct {
	Valid  bool          `json:"valid"`
	APIKey models.APIKey `json:"api_key"`
}

func NewAPIKeyHandler() *APIKeyHandler {
	return &APIKeyHandler{
		service: usage.NewAPIKeyService(),
	}
}

func (h *APIKeyHandler) GetAllAPIKeysByUserId(c *fiber.Ctx) error {
	userId := c.Params("userId")
	apiKeys, err := h.service.GetAllAPIKeysByUserId(userId)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(apiKeys)
}

func (h *APIKeyHandler) GetAPIKeyById(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid API key ID"})
	}
	apiKey, err := h.service.GetAPIKeyById(id)
	if err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "API key not found"})
	}
	return c.JSON(apiKey)
}

func (h *APIKeyHandler) CreateAPIKey(c *fiber.Ctx) error {
	var request CreateAPIKeyRequest
	if err := c.BodyParser(&request); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid request body"})
	}

	var expiresAt *time.Time
	if request.ExpiresAt != "" {
		parsedTime, err := time.Parse(time.RFC3339, request.ExpiresAt)
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": "Invalid expires_at format. Use RFC3339 format (e.g., 2023-01-01T00:00:00Z)",
			})
		}
		expiresAt = &parsedTime
	}

	userId, ok := c.Locals("userID").(string)
	if !ok || userId == "" {
		return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{"error": "User ID missing from context"})
	}

	// Validate user exists in Clerk
	ctx := context.Background()
	config := &clerk.ClientConfig{}
	config.Key = clerk.String(os.Getenv("CLERK_SECRET_KEY"))
	userClient := user.NewClient(config)

	_, err := userClient.Get(ctx, userId)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to validate user: " + err.Error(),
		})
	}

	apiKey, fullAPIKey, err := h.service.GenerateAPIKey(
		userId,
		request.Name,
		request.Status,
		expiresAt,
	)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}

	return c.Status(fiber.StatusCreated).JSON(CreateAPIKeyResponse{
		APIKey:     apiKey,
		FullAPIKey: fullAPIKey,
	})
}

func (h *APIKeyHandler) UpdateAPIKey(c *fiber.Ctx) error {
	idStr := c.Params("id")
	id, err := uuid.Parse(idStr)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid API key ID format"})
	}
	var request UpdateAPIKeyRequest
	if err := c.BodyParser(&request); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid request body"})
	}
	apiKey, err := h.service.GetAPIKeyById(id)
	if err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "API key not found"})
	}
	apiKey.Name = request.Name
	apiKey.Status = request.Status

	if err := h.service.UpdateAPIKey(&apiKey); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(apiKey)
}

func (h *APIKeyHandler) DeleteAPIKey(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid API key ID"})
	}
	if err := h.service.DeleteAPIKey(id); err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "API key not found"})
	}
	return c.SendStatus(fiber.StatusNoContent)
}

func (h *APIKeyHandler) VerifyAPIKey(c *fiber.Ctx) error {
	apiKey := c.Request().Header.Peek("X-Stainless-API-Key")
	if len(apiKey) == 0 {
		return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{"error": "API key is missing"})
	}
	_, verified, err := h.service.VerifyAPIKey(string(apiKey))
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "Failed to verify API key"})
	}
	if !verified {
		return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{"error": "Invalid API key"})
	}
	return c.SendStatus(fiber.StatusOK)
}