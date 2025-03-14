package api

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services"
	"bufio"
	"fmt"
	"io"
	"log"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/valyala/fasthttp"
)

func StreamChatCompletion(c *fiber.Ctx) error {
	requestID := c.Get("X-Request-ID", time.Now().String())
	var req models.ChatCompletionRequest
	if err := c.BodyParser(&req); err != nil {
		log.Printf("[%s] Error parsing request body: %v", requestID, err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body: " + err.Error(),
		})
	}

	log.Printf("[%s] Processing chat completion with %d messages", requestID, len(req.Messages))
	prompt_classifier_client := services.NewPromptClassifierClient()
	prompt := req.Messages[len(req.Messages)-1]

	log.Printf("[%s] Selecting model for prompt", requestID)
	selected_model, err := prompt_classifier_client.SelectModel(prompt.Content)
	if err != nil {
		log.Printf("[%s] Model selection failed: %v", requestID, err)
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	log.Printf("[%s] Selected model: %s from provider: %s", requestID, selected_model.SelectedModel, selected_model.Provider)
	full_chat_completion_req := models.ProviderChatCompletionRequest{
		Provider:         selected_model.Provider,
		Model:            selected_model.SelectedModel,
		Messages:         req.Messages,
		Temperature:      selected_model.Parameters.Temperature,
		N:                selected_model.Parameters.N,
		MaxTokens:        selected_model.Parameters.MaxTokens,
		PresencePenalty:  selected_model.Parameters.PresencePenalty,
		FrequencyPenalty: selected_model.Parameters.FrequencyPenalty,
		Stream:           true,
	}

	log.Printf("[%s] Initializing LLM provider: %s", requestID, full_chat_completion_req.Provider)
	provider, err := services.NewLLMProvider(full_chat_completion_req.Provider)
	if err != nil {
		log.Printf("[%s] Failed to initialize provider: %v", requestID, err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	log.Printf("[%s] Sending request to provider for chat completion", requestID)
	startTime := time.Now()
	resp, err := provider.StreamChatCompletion(&full_chat_completion_req)
	duration := time.Since(startTime)
	if err != nil {
		log.Printf("[%s] Chat completion failed after %v: %v", requestID, duration, err)
		if resp != nil && resp.Error != "" {
			return c.Status(fiber.StatusInternalServerError).JSON(resp)
		}
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to generate completion: " + err.Error(),
		})
	}

	log.Printf("[%s] Chat completion successful in %v", requestID, duration)

	// Set appropriate headers for SSE
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("Transfer-Encoding", "chunked")

	// Follow the Fiber docs pattern using StreamWriter
	c.Context().SetBodyStreamWriter(fasthttp.StreamWriter(func(w *bufio.Writer) {
		streamReader, err := services.GetStreamReader(resp, resp.Provider, requestID)
		if err != nil {
			log.Printf("[%s] Failed to create stream reader: %v", requestID, err)
			fmt.Fprintf(w, "data: {\"error\": \"%s\"}\n\n", err.Error())
			w.Flush()
			return
		}

		defer func() {
			streamReader.Close()
			log.Printf("[%s] Stream completed", requestID)
		}()

		buffer := make([]byte, 1024)
		startTime := time.Now()

		for {
			n, err := streamReader.Read(buffer)
			if n > 0 {
				// Write the buffer contents to the response
				_, writeErr := w.Write(buffer[:n])
				if writeErr != nil {
					log.Printf("[%s] Error writing to response: %v", requestID, writeErr)
					break
				}

				// Flush to send data immediately to client
				if flushErr := w.Flush(); flushErr != nil {
					log.Printf("[%s] Error flushing data: %v", requestID, flushErr)
					break
				}
			}

			// Check for EOF (end of stream)
			if err == io.EOF {
				log.Printf("[%s] Stream completed after %v", requestID, time.Since(startTime))
				break
			}

			// Handle other errors
			if err != nil {
				log.Printf("[%s] Error reading from stream: %v", requestID, err)
				fmt.Fprintf(w, "data: {\"error\": \"%s\"}\n\n", err.Error())
				w.Flush()
				break
			}
		}
	}))

	return nil
}

func ChatCompletion(c *fiber.Ctx) error {
	requestID := c.Get("X-Request-ID", time.Now().String())
	log.Printf("[%s] Received chat completion request", requestID)

	// Parse request body
	var req models.ChatCompletionRequest
	if err := c.BodyParser(&req); err != nil {
		log.Printf("[%s] Error parsing request body: %v", requestID, err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "Invalid request body: " + err.Error(),
		})
	}

	log.Printf("[%s] Processing chat completion with %d messages", requestID, len(req.Messages))

	prompt_classifier_client := services.NewPromptClassifierClient()
	prompt := req.Messages[len(req.Messages)-1]

	log.Printf("[%s] Selecting model for prompt", requestID)
	selected_model, err := prompt_classifier_client.SelectModel(prompt.Content)
	if err != nil {
		log.Printf("[%s] Model selection failed: %v", requestID, err)
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	log.Printf("[%s] Selected model: %s from provider: %s", requestID, selected_model.SelectedModel, selected_model.Provider)

	full_chat_completion_req := models.ProviderChatCompletionRequest{
		Provider:         selected_model.Provider,
		Model:            selected_model.SelectedModel,
		Messages:         req.Messages,
		Temperature:      selected_model.Parameters.Temperature,
		N:                selected_model.Parameters.N,
		MaxTokens:        selected_model.Parameters.MaxTokens,
		PresencePenalty:  selected_model.Parameters.PresencePenalty,
		FrequencyPenalty: selected_model.Parameters.FrequencyPenalty,
	}

	// Get the appropriate LLM provider
	log.Printf("[%s] Initializing LLM provider: %s", requestID, full_chat_completion_req.Provider)
	provider, err := services.NewLLMProvider(full_chat_completion_req.Provider)
	if err != nil {
		log.Printf("[%s] Failed to initialize provider: %v", requestID, err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	// Call the provider's chat completion
	log.Printf("[%s] Sending request to provider for chat completion", requestID)
	startTime := time.Now()
	resp, err := provider.CreateChatCompletion(&full_chat_completion_req)
	duration := time.Since(startTime)

	if err != nil {
		log.Printf("[%s] Chat completion failed after %v: %v", requestID, duration, err)
		// If there's an error but we have a response with error details
		if resp != nil && resp.Error != "" {
			return c.Status(fiber.StatusInternalServerError).JSON(resp)
		}
		// Generic error
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": "Failed to generate completion: " + err.Error(),
		})
	}

	log.Printf("[%s] Chat completion successful in %v", requestID, duration)
	// Return successful response
	return c.JSON(resp)
}
