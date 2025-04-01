package api

import (
	"adaptive-backend/internal/services/conversations"
	"strconv"

	"github.com/gofiber/fiber/v2"
)

type ConversationHandler struct {
	service *conversations.ConversationService
}

type CreateConversationRequest struct {
	Title string `json:"title"`
}

func NewConversationHandler() *ConversationHandler {
	return &ConversationHandler{
		service: conversations.NewConversationService(),
	}
}

// GetAllConversations handles GET /conversations
func (h *ConversationHandler) GetAllConversations(c *fiber.Ctx) error {
	conversations, err := h.service.GetAllConversations()
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return c.JSON(conversations)
}

// GetConversation handles GET /conversations/:id
func (h *ConversationHandler) GetConversation(c *fiber.Ctx) error {
	id, err := strconv.ParseUint(c.Params("id"), 10, 64)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid conversation ID",
		})
	}

	conversation, err := h.service.GetConversation(uint(id))
	if err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": "Conversation not found",
		})
	}

	return c.JSON(conversation)
}

func (h *ConversationHandler) CreateConversation(c *fiber.Ctx) error {
	var request CreateConversationRequest

	if err := c.BodyParser(&request); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body",
		})
	}

	conversation, err := h.service.CreateConversation(request.Title)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	return c.Status(fiber.StatusCreated).JSON(conversation)
}

func (h *ConversationHandler) UpdateConversation(c *fiber.Ctx) error {
	id, err := strconv.ParseUint(c.Params("id"), 10, 64)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid conversation ID",
		})
	}

	var request CreateConversationRequest

	if err := c.BodyParser(&request); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body",
		})
	}

	conversation, err := h.service.UpdateConversation(uint(id), request.Title)
	if err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": "Conversation not found",
		})
	}

	return c.JSON(conversation)
}

func (h *ConversationHandler) DeleteConversation(c *fiber.Ctx) error {
	id, err := strconv.ParseUint(c.Params("id"), 10, 64)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid conversation ID",
		})
	}

	if err := h.service.DeleteConversation(uint(id)); err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": "Conversation not found",
		})
	}

	return c.SendStatus(fiber.StatusNoContent)
}

func (h *ConversationHandler) PinConversation(c *fiber.Ctx) error {
	id, err := strconv.ParseUint(c.Params("id"), 10, 64)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid conversation ID",
		})
	}

	if err := h.service.PinConversation(uint(id)); err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": "Conversation not found",
		})
	}
	return c.SendStatus(fiber.StatusNoContent)
}
