package api

import (
	"adaptive-backend/internal/services/conversations"
	"strconv"

	"github.com/gofiber/fiber/v2"
)

// MessageRequest represents the request body for creating or updating a message
type MessageRequest struct {
	Role     string  `json:"role"`
	Content  string  `json:"content"`
	Provider *string `json:"provider"`
	Model    *string `json:"model"`
}

// BatchDeleteRequest represents the request body for batch deleting messages
type BatchDeleteRequest struct {
	MessageIDs []uint `json:"messageIds"`
}

// MessageHandler handles HTTP requests related to messages
type MessageHandler struct {
	service *conversations.MessageService
}

// NewMessageHandler creates a new instance of MessageHandler
func NewMessageHandler() *MessageHandler {
	return &MessageHandler{
		service: conversations.NewMessageService(),
	}
}

// GetMessagesByConversation handles GET /conversations/:id/messages
func (h *MessageHandler) GetMessagesByConversation(c *fiber.Ctx) error {
	conversationID, err := strconv.ParseUint(c.Params("id"), 10, 64)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid conversation ID",
		})
	}
	messages, err := h.service.GetMessagesByConversation(uint(conversationID))
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}
	return c.JSON(messages)
}

// GetMessage handles GET /messages/:id
func (h *MessageHandler) GetMessage(c *fiber.Ctx) error {
	id, err := strconv.ParseUint(c.Params("id"), 10, 64)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid message ID",
		})
	}
	message, err := h.service.GetMessage(uint(id))
	if err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": "Message not found",
		})
	}
	return c.JSON(message)
}

// CreateMessage handles POST /conversations/:id/messages
func (h *MessageHandler) CreateMessage(c *fiber.Ctx) error {
	conversationID, err := strconv.ParseUint(c.Params("id"), 10, 64)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid conversation ID",
		})
	}
	var request MessageRequest
	if err := c.BodyParser(&request); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body",
		})
	}
	message, err := h.service.CreateMessage(uint(conversationID), request.Role, request.Content, request.Provider, request.Model)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}
	return c.Status(fiber.StatusCreated).JSON(message)
}

// UpdateMessage handles PUT /messages/:id
func (h *MessageHandler) UpdateMessage(c *fiber.Ctx) error {
	id, err := strconv.ParseUint(c.Params("id"), 10, 64)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid message ID",
		})
	}
	var request MessageRequest
	if err := c.BodyParser(&request); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body",
		})
	}
	message, err := h.service.UpdateMessage(uint(id), request.Role, request.Content, request.Provider, request.Model)
	if err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": "Message not found",
		})
	}
	return c.JSON(message)
}

// DeleteMessage handles DELETE /messages/:id
func (h *MessageHandler) DeleteMessage(c *fiber.Ctx) error {
	id, err := strconv.ParseUint(c.Params("id"), 10, 64)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid message ID",
		})
	}
	if err := h.service.DeleteMessage(uint(id)); err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": "Message not found",
		})
	}
	return c.SendStatus(fiber.StatusNoContent)
}

// DeleteAllConversationMessages handles DELETE /conversations/:id/messages
func (h *MessageHandler) DeleteAllConversationMessages(c *fiber.Ctx) error {
	conversationID, err := strconv.ParseUint(c.Params("id"), 10, 64)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid conversation ID",
		})
	}
	if err := h.service.DeleteAllMessages(uint(conversationID)); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}
	return c.SendStatus(fiber.StatusNoContent)
}

// BatchDeleteMessages handles batch deletion of messages
func (h *MessageHandler) BatchDeleteMessages(c *fiber.Ctx) error {
	var request BatchDeleteRequest
	if err := c.BodyParser(&request); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body",
		})
	}
	if err := h.service.BatchDeleteMessages(request.MessageIDs); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}
	return c.SendStatus(fiber.StatusNoContent)
}
