package services

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	fiberlog "github.com/gofiber/fiber/v2/log"
)

// Client represents an optimized API client with connection pooling
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
	Retries      int
	RetryDelay   time.Duration
}

// ClientConfig holds configuration for the API client
type ClientConfig struct {
	BaseURL             string
	Timeout             time.Duration
	MaxIdleConns        int
	MaxIdleConnsPerHost int
	IdleConnTimeout     time.Duration
	DialTimeout         time.Duration
	KeepAlive           time.Duration
	TLSHandshakeTimeout time.Duration
}

// DefaultClientConfig returns optimized defaults for the API client
func DefaultClientConfig(baseURL string) *ClientConfig {
	return &ClientConfig{
		BaseURL:             baseURL,
		Timeout:             30 * time.Second,
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 10,
		IdleConnTimeout:     90 * time.Second,
		DialTimeout:         10 * time.Second,
		KeepAlive:           30 * time.Second,
		TLSHandshakeTimeout: 10 * time.Second,
	}
}

// NewClient creates a new optimized API client
func NewClient(baseURL string) *Client {
	config := DefaultClientConfig(baseURL)
	return NewClientWithConfig(config)
}

// NewClientWithConfig creates a new API client with custom configuration
func NewClientWithConfig(config *ClientConfig) *Client {
	// Create optimized transport with connection pooling
	transport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   config.DialTimeout,
			KeepAlive: config.KeepAlive,
		}).DialContext,
		MaxIdleConns:        config.MaxIdleConns,
		MaxIdleConnsPerHost: config.MaxIdleConnsPerHost,
		IdleConnTimeout:     config.IdleConnTimeout,
		TLSHandshakeTimeout: config.TLSHandshakeTimeout,
		ForceAttemptHTTP2:   true,
		DisableCompression:  false,
	}

	client := &Client{
		BaseURL: config.BaseURL,
		HTTPClient: &http.Client{
			Timeout:   config.Timeout,
			Transport: transport,
		},
		Headers: map[string]string{
			"Content-Type": "application/json",
			"Accept":       "application/json",
			"User-Agent":   "adaptive-backend/1.0",
		},
	}

	return client
}

// Get performs a GET request
func (c *Client) Get(path string, result any, opts *RequestOptions) error {
	return c.doRequest(http.MethodGet, path, nil, result, opts)
}

// Post performs a POST request
func (c *Client) Post(path string, body, result any, opts *RequestOptions) error {
	return c.doRequest(http.MethodPost, path, body, result, opts)
}

// Put performs a PUT request
func (c *Client) Put(path string, body, result any, opts *RequestOptions) error {
	return c.doRequest(http.MethodPut, path, body, result, opts)
}

// Delete performs a DELETE request
func (c *Client) Delete(path string, result any, opts *RequestOptions) error {
	return c.doRequest(http.MethodDelete, path, nil, result, opts)
}

// Patch performs a PATCH request
func (c *Client) Patch(path string, body, result any, opts *RequestOptions) error {
	return c.doRequest(http.MethodPatch, path, body, result, opts)
}

// doRequest performs an HTTP request with retries
func (c *Client) doRequest(method, path string, body, result any, opts *RequestOptions) error {
	url := c.BaseURL + path

	// Set default options
	if opts == nil {
		opts = &RequestOptions{}
	}
	if opts.Retries == 0 {
		opts.Retries = 3
	}
	if opts.RetryDelay == 0 {
		opts.RetryDelay = 1 * time.Second
	}

	var lastErr error
	for attempt := 0; attempt <= opts.Retries; attempt++ {
		if attempt > 0 {
			// Wait before retry with exponential backoff
			delay := time.Duration(attempt) * opts.RetryDelay
			time.Sleep(delay)
		}

		err := c.executeRequest(method, url, body, result, opts)
		if err == nil {
			return nil
		}

		lastErr = err

		// Check if error is retryable
		if !c.isRetryableError(err) {
			break
		}
	}

	return fmt.Errorf("request failed after %d attempts: %w", opts.Retries+1, lastErr)
}

// executeRequest performs a single HTTP request
func (c *Client) executeRequest(method, url string, body, result any, opts *RequestOptions) error {
	// Create request context
	ctx := context.Background()
	if opts.Context != nil {
		ctx = opts.Context
	} else if opts.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, opts.Timeout)
		defer cancel()
	}

	// Prepare request body
	var reqBody io.Reader
	var bodySize int64
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("error marshaling request body: %w", err)
		}
		reqBody = bytes.NewBuffer(jsonBody)
		bodySize = int64(len(jsonBody))
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return fmt.Errorf("error creating request: %w", err)
	}

	// Set content length for POST/PUT requests
	if bodySize > 0 {
		req.ContentLength = bodySize
	}

	// Set default headers
	for k, v := range c.Headers {
		req.Header.Set(k, v)
	}

	// Set custom headers if provided
	if len(opts.Headers) > 0 {
		for k, v := range opts.Headers {
			req.Header.Set(k, v)
		}
	}

	// Add query parameters if provided
	if len(opts.QueryParams) > 0 {
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
	defer func() {
		if err := resp.Body.Close(); err != nil {
			fiberlog.Errorf("Error closing response body: %v", err)
		}
	}()

	// Check response status
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API request failed with status code %d: %s", resp.StatusCode, string(bodyBytes))
	}

	// Handle response based on expected type
	return c.handleResponse(resp, result, opts)
}

// handleResponse processes the HTTP response based on the expected type
func (c *Client) handleResponse(resp *http.Response, result any, opts *RequestOptions) error {
	responseType := "json"
	if opts.ResponseType != "" {
		responseType = opts.ResponseType
	}

	switch responseType {
	case "json":
		return c.handleJSONResponse(resp, result)
	case "text":
		return c.handleTextResponse(resp, result)
	case "binary":
		return c.handleBinaryResponse(resp, result)
	default:
		return fmt.Errorf("unsupported response type: %s", responseType)
	}
}

// handleJSONResponse processes JSON responses
func (c *Client) handleJSONResponse(resp *http.Response, result any) error {
	if result == nil {
		// Discard response body
		_, err := io.Copy(io.Discard, resp.Body)
		return err
	}

	// Read the response body
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("error reading response body: %w", err)
	}

	// Unmarshal the response
	if err := json.Unmarshal(bodyBytes, result); err != nil {
		return fmt.Errorf("error unmarshaling response: %w", err)
	}

	return nil
}

// handleTextResponse processes text responses
func (c *Client) handleTextResponse(resp *http.Response, result any) error {
	stringResult, ok := result.(*string)
	if !ok {
		return fmt.Errorf("result must be *string for text response")
	}

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("error reading response body: %w", err)
	}

	*stringResult = string(bodyBytes)
	return nil
}

// handleBinaryResponse processes binary responses
func (c *Client) handleBinaryResponse(resp *http.Response, result any) error {
	bytesResult, ok := result.(*[]byte)
	if !ok {
		return fmt.Errorf("result must be *[]byte for binary response")
	}

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("error reading response body: %w", err)
	}

	*bytesResult = bodyBytes
	return nil
}

// isRetryableError determines if an error is retryable
func (c *Client) isRetryableError(err error) bool {
	if err == nil {
		return false
	}

	// Check for network errors
	if netErr, ok := err.(net.Error); ok {
		return netErr.Timeout()
	}

	// Check for context timeout
	if err == context.DeadlineExceeded {
		return true
	}

	// Check for specific HTTP status codes (5xx errors)
	errStr := err.Error()
	retryableStatusCodes := []string{"500", "502", "503", "504", "520", "521", "522", "523", "524"}
	for _, code := range retryableStatusCodes {
		if containsStatusCode(errStr, code) {
			return true
		}
	}

	return false
}

// containsStatusCode checks if error message contains a specific status code
func containsStatusCode(errStr, statusCode string) bool {
	return fmt.Sprintf("status code %s", statusCode) == errStr ||
		fmt.Sprintf("status %s", statusCode) == errStr
}

// Close closes the underlying HTTP client
func (c *Client) Close() {
	if transport, ok := c.HTTPClient.Transport.(*http.Transport); ok {
		transport.CloseIdleConnections()
	}
}
