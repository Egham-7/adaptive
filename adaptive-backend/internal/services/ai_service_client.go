package services

import (
	"adaptive-backend/internal/models"
	"os"
	"time"
)

// PromptClassifierClient is a client for the prompt classifier API
type PromptClassifierClient struct {
	client *Client
}

// NewPromptClassifierClient creates a new client for the prompt classifier API
func NewPromptClassifierClient() *PromptClassifierClient {
	baseURL := os.Getenv("ADAPTIVE_AI_BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:8000"
	}

	return &PromptClassifierClient{
		client: NewClient(baseURL),
	}
}

func (c *PromptClassifierClient) SelectModel(prompt string) (*models.SelectModelResponse, error) {
	var result models.SelectModelResponse
	err := c.client.Post("/api/model/select-model", models.SelectModelRequest{Prompt: prompt}, &result, &RequestOptions{Timeout: 50 * time.Second})

	return &result, err
}
