package api

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/usage"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

type APIKeyHandler struct {
	service *usage.APIKeyService
}

type CreateAPIKeyRequest struct {
	Name             string  `json:"name"`
	UserID           string  `json:"user_id"`
	Status           string  `json:"status"`
	ExpiresAt        string  `json:"expires_at,omitempty"`
	StripeCustomerID *string `json:"stripe_customer_id,omitempty"` // Optional for now
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
	APIKey models.APIKey `json:"api_key,omitempty"`
}

func NewAPIKeyHandler() *APIKeyHandler {
	return &APIKeyHandler{
		service: usage.NewAPIKeyService(),
	}
}

// GetAllAPIKeysByUserId handles GET /api-keys/user/:userId
func (h *APIKeyHandler) GetAllAPIKeysByUserId(c *fiber.Ctx) error {
	userId := c.Params("userId")
	apiKeys, err := h.service.GetAllAPIKeysByUserId(userId)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}
	return c.JSON(apiKeys)
}

// GetAPIKeyById handles GET /api-keys/:id
func (h *APIKeyHandler) GetAPIKeyById(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid API key ID",
		})
	}
	apiKey, err := h.service.GetAPIKeyById(id)
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

	// For the first API key, there is no Stripe customer yet.
	// Generate the API key without Stripe customer logic.
	apiKey, fullAPIKey, err := h.service.GenerateAPIKey(
		request.UserID,
		request.Name,
		request.Status,
		expiresAt,
		"", // Empty StripeCustomerID for now
	)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	// TODO: After creating the API key, check if this is the first for the user.
	// If so, create a Stripe customer and update the APIKey record with the StripeCustomerID.

	return c.Status(fiber.StatusCreated).JSON(CreateAPIKeyResponse{
		APIKey:     apiKey,
		FullAPIKey: fullAPIKey,
	})
}

// UpdateAPIKey handles PUT /api-keys/:id
func (h *APIKeyHandler) UpdateAPIKey(c *fiber.Ctx) error {
	idStr := c.Params("id")
	id, err := uuid.Parse(idStr)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid API key ID format",
		})
	}
	var request UpdateAPIKeyRequest
	if err := c.BodyParser(&request); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body",
		})
	}
	apiKey, err := h.service.GetAPIKeyById(id)
	if err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": "API key not found",
		})
	}
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
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid API key ID",
		})
	}
	if err := h.service.DeleteAPIKey(id); err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": "API key not found",
		})
	}

	return c.SendStatus(fiber.StatusNoContent)
}

func (h *APIKeyHandler) VerifyAPIKey(c *fiber.Ctx) error {
	apiKey := c.Request().Header.Peek("X-API-Key")
	if len(apiKey) == 0 {
		return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
			"error": "API key is missing",
		})
	}
	_, verified, err := h.service.VerifyAPIKey(string(apiKey))
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to verify api key",
		})
	}
	if !verified {
		return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
			"error": "Invalid",
		})
	}
	return nil
}
