package api

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/usage"
	"strconv"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

type APIKeyHandler struct {
	service *usage.APIKeyService
}

type CreateAPIKeyRequest struct {
	Name      string    `json:"name"`
	UserID    uuid.UUID `json:"user_id"`
	Status    string    `json:"status"`
	ExpiresAt string    `json:"expires_at,omitempty"`
}

type CreateAPIKeyResponse struct {
	APIKey     models.APIKey `json:"api_key"`
	FullAPIKey string        `json:"full_api_key"`
}

type UpdateAPIKeyRequest struct {
	Name   string `json:"name"`
	Status string `json:"status"`
}

type VerifyAPIKeyRequest struct {
	APIKey string `json:"api_key"`
}

type VerifyAPIKeyResponse struct {
	Valid  bool          `json:"valid"`
	APIKey models.APIKey `json:"api_key,omitempty"`
}

func NewAPIKeyHandler() *APIKeyHandler {
	return &APIKeyHandler{
		service: usage.NewAPIKeyService(),
	}
}

// GetAllAPIKeysByUserId handles GET /api-keys/user/:userId
func (h *APIKeyHandler) GetAllAPIKeysByUserId(c *fiber.Ctx) error {
	userId, err := strconv.ParseUint(c.Params("userId"), 10, 64)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid user ID",
		})
	}

	apiKeys, err := h.service.GetAllAPIKeysByUserId(uint(userId))
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return c.JSON(apiKeys)
}

// GetAPIKeyById handles GET /api-keys/:id
func (h *APIKeyHandler) GetAPIKeyById(c *fiber.Ctx) error {
	id, err := strconv.ParseUint(c.Params("id"), 10, 64)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid API key ID",
		})
	}

	apiKey, err := h.service.GetAPIKeyById(uint(id))
	if err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": "API key not found",
		})
	}

	return c.JSON(apiKey)
}

// CreateAPIKey handles POST /api-keys
func (h *APIKeyHandler) CreateAPIKey(c *fiber.Ctx) error {
	var request CreateAPIKeyRequest
	if err := c.BodyParser(&request); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body",
		})
	}

	// Parse expires_at if provided
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

	// Generate the API key
	apiKey, fullAPIKey, err := h.service.GenerateAPIKey(request.UserID, request.Name, request.Status, expiresAt)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return c.Status(fiber.StatusCreated).JSON(CreateAPIKeyResponse{
		APIKey:     apiKey,
		FullAPIKey: fullAPIKey,
	})
}

// VerifyAPIKey handles POST /api-keys/verify
func (h *APIKeyHandler) VerifyAPIKey(c *fiber.Ctx) error {
	var request VerifyAPIKeyRequest
	if err := c.BodyParser(&request); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body",
		})
	}

	apiKey, valid, err := h.service.VerifyAPIKey(request.APIKey)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	response := VerifyAPIKeyResponse{
		Valid: valid,
	}

	if valid {
		response.APIKey = apiKey
	}

	return c.JSON(response)
}

// UpdateAPIKey handles PUT /api-keys/:id
func (h *APIKeyHandler) UpdateAPIKey(c *fiber.Ctx) error {
	id, err := strconv.ParseUint(c.Params("id"), 10, 64)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid API key ID",
		})
	}

	var request UpdateAPIKeyRequest
	if err := c.BodyParser(&request); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body",
		})
	}

	// First get the existing API key
	apiKey, err := h.service.GetAPIKeyById(uint(id))
	if err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": "API key not found",
		})
	}

	// Update fields
	apiKey.Name = request.Name
	apiKey.Status = request.Status

	if err := h.service.UpdateAPIKey(&apiKey); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return c.JSON(apiKey)
}

// DeleteAPIKey handles DELETE /api-keys/:id
func (h *APIKeyHandler) DeleteAPIKey(c *fiber.Ctx) error {
	id, err := strconv.ParseUint(c.Params("id"), 10, 64)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid API key ID",
		})
	}

	if err := h.service.DeleteAPIKey(uint(id)); err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": "API key not found",
		})
	}

	return c.SendStatus(fiber.StatusNoContent)
}
