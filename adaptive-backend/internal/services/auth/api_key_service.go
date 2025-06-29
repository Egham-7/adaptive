package auth

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

type APIKeyVerificationRequest struct {
	APIKey string `json:"apiKey"`
}

type APIKeyVerificationResponse struct {
	Valid bool `json:"valid"`
}

type APIKeyService struct {
	frontendURL string
	httpClient  *http.Client
}

func NewAPIKeyService() *APIKeyService {
	frontendURL := os.Getenv("FRONTEND_URL")
	if frontendURL == "" {
		frontendURL = "http://localhost:3000" // default for dev
	}

	return &APIKeyService{
		frontendURL: frontendURL,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

func (s *APIKeyService) VerifyAPIKey(apiKey string) (bool, error) {
	// Skip verification in development environment
	env := os.Getenv("ENV")
	if env != "production" {
		return true, nil
	}

	// Prepare the request payload
	payload := APIKeyVerificationRequest{
		APIKey: apiKey,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return false, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Make the request to the frontend tRPC endpoint
	url := fmt.Sprintf("%s/api/trpc/apiKeys.verify", s.frontendURL)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return false, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return false, fmt.Errorf("failed to make request: %w", err)
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			// Log the error but don't override the main function's return value
			fmt.Printf("Warning: failed to close response body: %v\n", closeErr)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("verification request failed with status: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return false, fmt.Errorf("failed to read response body: %w", err)
	}

	var verificationResp APIKeyVerificationResponse
	if err := json.Unmarshal(body, &verificationResp); err != nil {
		return false, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return verificationResp.Valid, nil
}
