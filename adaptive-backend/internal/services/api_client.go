package services

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client represents a generic API client
type Client struct {
	BaseURL    string
	HTTPClient *http.Client
	Headers    map[string]string
}

// RequestOptions provides options for API requests
type RequestOptions struct {
	Headers      map[string]string
	QueryParams  map[string]string
	Timeout      time.Duration
	Context      context.Context
	ResponseType string // "json", "text", "binary"
}

// NewClient creates a new API client
func NewClient(baseURL string) *Client {
	return &Client{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		Headers: map[string]string{
			"Content-Type": "application/json",
			"Accept":       "application/json",
		},
	}
}

// Get performs a GET request
func (c *Client) Get(path string, result any, opts *RequestOptions) error {
	return c.doRequest(http.MethodGet, path, nil, result, opts)
}

// Post performs a POST request
func (c *Client) Post(path string, body any, result any, opts *RequestOptions) error {
	return c.doRequest(http.MethodPost, path, body, result, opts)
}

// Put performs a PUT request
func (c *Client) Put(path string, body any, result any, opts *RequestOptions) error {
	return c.doRequest(http.MethodPut, path, body, result, opts)
}

// Delete performs a DELETE request
func (c *Client) Delete(path string, result any, opts *RequestOptions) error {
	return c.doRequest(http.MethodDelete, path, nil, result, opts)
}

// Patch performs a PATCH request
func (c *Client) Patch(path string, body any, result any, opts *RequestOptions) error {
	return c.doRequest(http.MethodPatch, path, body, result, opts)
}

// doRequest performs an HTTP request
func (c *Client) doRequest(method, path string, body any, result any, opts *RequestOptions) error {
	url := c.BaseURL + path

	// Create request context
	ctx := context.Background()
	if opts != nil && opts.Context != nil {
		ctx = opts.Context
	} else if opts != nil && opts.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, opts.Timeout)
		defer cancel()
	}

	// Prepare request body
	var reqBody io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("error marshaling request body: %w", err)
		}
		reqBody = bytes.NewBuffer(jsonBody)
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return fmt.Errorf("error creating request: %w", err)
	}

	// Set default headers
	for k, v := range c.Headers {
		req.Header.Set(k, v)
	}

	// Set custom headers if provided
	if opts != nil && len(opts.Headers) > 0 {
		for k, v := range opts.Headers {
			req.Header.Set(k, v)
		}
	}

	// Add query parameters if provided
	if opts != nil && len(opts.QueryParams) > 0 {
		q := req.URL.Query()
		for k, v := range opts.QueryParams {
			q.Add(k, v)
		}
		req.URL.RawQuery = q.Encode()
	}

	// Execute request
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("error executing request: %w", err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API request failed with status code %d: %s", resp.StatusCode, string(bodyBytes))
	}

	// Handle response based on expected type
	responseType := "json"
	if opts != nil && opts.ResponseType != "" {
		responseType = opts.ResponseType
	}

	switch responseType {
	case "json":
		if result != nil {
			// Read the response body
			bodyBytes, err := io.ReadAll(resp.Body)
			if err != nil {
				return fmt.Errorf("error reading response body: %w", err)
			}

			// Unmarshal the response
			if err := json.Unmarshal(bodyBytes, result); err != nil {
				return fmt.Errorf("error unmarshaling response: %w", err)
			}
		}
	case "text":
		if stringResult, ok := result.(*string); ok {
			bodyBytes, err := io.ReadAll(resp.Body)
			if err != nil {
				return fmt.Errorf("error reading response body: %w", err)
			}
			*stringResult = string(bodyBytes)
		}
	case "binary":
		if bytesResult, ok := result.(*[]byte); ok {
			bodyBytes, err := io.ReadAll(resp.Body)
			if err != nil {
				return fmt.Errorf("error reading response body: %w", err)
			}
			*bytesResult = bodyBytes
		}
	}

	return nil
}
