package middleware

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// ModelValidationRequest represents the request to validate models
type ModelValidationRequest struct {
	Models []string `json:"models"`
}

// ModelValidationResponse represents the response from model validation
type ModelValidationResponse struct {
	ValidModels   []string `json:"valid_models"`
	InvalidModels []string `json:"invalid_models"`
	Error         string   `json:"error,omitempty"`
}

// ModelConversionRequest represents the request to convert model names to capabilities
type ModelConversionRequest struct {
	ModelNames []string `json:"model_names"`
}

// ModelConversionResponse represents the response from model conversion
type ModelConversionResponse struct {
	ModelCapabilities []map[string]interface{} `json:"model_capabilities"`
	InvalidModels     []string                 `json:"invalid_models"`
	Error             string                   `json:"error,omitempty"`
}

// CompletionRequest represents the structure we need to extract models from
type CompletionRequest struct {
	Model  string                 `json:"model,omitempty"`
	Models []string               `json:"models,omitempty"` // New: support for model arrays
	Other  map[string]interface{} `json:"-"`                // Capture other fields to preserve in response
}

// ModelValidator creates middleware for validating model names
func ModelValidator() fiber.Handler {
	// Get AI service base URL
	aiServiceURL := os.Getenv("ADAPTIVE_AI_BASE_URL")
	if aiServiceURL == "" {
		aiServiceURL = "http://localhost:8000"
	}

	return func(c *fiber.Ctx) error {
		// Only validate for chat completions endpoint
		if c.Path() != "/v1/chat/completions" {
			return c.Next()
		}

		// Read and parse the request body
		body := c.Body()
		if len(body) == 0 {
			return c.Next()
		}

		var reqData CompletionRequest
		if err := json.Unmarshal(body, &reqData); err != nil {
			// If we can't parse, let the request continue (other validation will catch it)
			return c.Next()
		}

		// Extract models to validate
		var modelsToValidate []string
		if reqData.Model != "" {
			modelsToValidate = append(modelsToValidate, reqData.Model)
		}
		if len(reqData.Models) > 0 {
			modelsToValidate = append(modelsToValidate, reqData.Models...)
		}

		// Skip validation if no models to validate
		if len(modelsToValidate) == 0 {
			return c.Next()
		}

		// If models array is provided, convert to full capabilities and inject
		if len(reqData.Models) > 0 {
			err := handleModelArrayConversion(c, body, aiServiceURL, reqData.Models)
			if err != nil {
				fiberlog.Errorf("Model array conversion failed: %v", err)
				return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
					"error": map[string]interface{}{
						"message": err.Error(),
						"type":    "invalid_request_error",
						"param":   "models",
						"code":    "model_conversion_failed",
					},
				})
			}
			return c.Next()
		}

		// Regular single model validation
		validModels, invalidModels, err := validateModelsWithAIService(aiServiceURL, modelsToValidate)
		if err != nil {
			fiberlog.Warnf("Model validation failed: %v", err)
			// Continue without validation if service is unavailable
			return c.Next()
		}

		// If all models are invalid, return error
		if len(validModels) == 0 && len(invalidModels) > 0 {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": map[string]interface{}{
					"message": fmt.Sprintf("Invalid model(s): %v", invalidModels),
					"type":    "invalid_request_error",
					"param":   "model",
					"code":    "model_not_found",
				},
			})
		}

		// If some models are invalid, log warning but continue
		if len(invalidModels) > 0 {
			fiberlog.Warnf("Some invalid models found: %v. Continuing with valid models: %v", invalidModels, validModels)
		}

		// Store validation results in context for potential use by handlers
		c.Locals("valid_models", validModels)
		c.Locals("invalid_models", invalidModels)

		return c.Next()
	}
}

// validateModelsWithAIService calls the AI service to validate models
func validateModelsWithAIService(baseURL string, models []string) ([]string, []string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Prepare request
	reqData := ModelValidationRequest{
		Models: models,
	}

	jsonData, err := json.Marshal(reqData)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	url := fmt.Sprintf("%s/validate-models", baseURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	// Make request
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			fiberlog.Warnf("Failed to close response body: %v", err)
		}
	}()

	// Read response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Parse response
	var validationResp ModelValidationResponse
	if err := json.Unmarshal(respBody, &validationResp); err != nil {
		return nil, nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if validationResp.Error != "" {
		return nil, nil, fmt.Errorf("validation error: %s", validationResp.Error)
	}

	return validationResp.ValidModels, validationResp.InvalidModels, nil
}

// handleModelArrayConversion converts model names to capabilities and modifies the request
func handleModelArrayConversion(c *fiber.Ctx, originalBody []byte, baseURL string, modelNames []string) error {
	// Convert model names to capabilities
	capabilities, invalidModels, err := convertModelNamesWithAIService(baseURL, modelNames)
	if err != nil {
		return fmt.Errorf("failed to convert model names: %w", err)
	}

	// If any models are invalid, return error
	if len(invalidModels) > 0 {
		return fmt.Errorf("invalid model(s): %v", invalidModels)
	}

	// Parse the original request body to modify it
	var requestData map[string]interface{}
	if err := json.Unmarshal(originalBody, &requestData); err != nil {
		return fmt.Errorf("failed to parse request body: %w", err)
	}

	// Remove the "models" array and set model to "adaptive"
	delete(requestData, "models")
	requestData["model"] = "adaptive"

	// Add protocol_manager_config with the converted capabilities
	requestData["protocol_manager_config"] = map[string]interface{}{
		"models": capabilities,
	}

	// Marshal the modified request
	modifiedBody, err := json.Marshal(requestData)
	if err != nil {
		return fmt.Errorf("failed to marshal modified request: %w", err)
	}

	// Replace the request body
	c.Request().SetBody(modifiedBody)

	return nil
}

// convertModelNamesWithAIService calls the AI service to convert model names to capabilities
func convertModelNamesWithAIService(baseURL string, modelNames []string) ([]map[string]interface{}, []string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Prepare request
	reqData := ModelConversionRequest{
		ModelNames: modelNames,
	}

	jsonData, err := json.Marshal(reqData)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	url := fmt.Sprintf("%s/convert-model-names", baseURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	// Make request
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			fiberlog.Warnf("Failed to close response body: %v", err)
		}
	}()

	// Read response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Parse response
	var conversionResp ModelConversionResponse
	if err := json.Unmarshal(respBody, &conversionResp); err != nil {
		return nil, nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if conversionResp.Error != "" {
		return nil, nil, fmt.Errorf("conversion error: %s", conversionResp.Error)
	}

	return conversionResp.ModelCapabilities, conversionResp.InvalidModels, nil
}
