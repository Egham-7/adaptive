package models

import (
	"fmt"
	"net/http"
)

// ErrorType represents the category of error
type ErrorType string

const (
	// ErrorTypeValidation represents validation errors (4xx)
	ErrorTypeValidation ErrorType = "validation"
	// ErrorTypeAuthentication represents authentication errors (401)
	ErrorTypeAuthentication ErrorType = "authentication"
	// ErrorTypeAuthorization represents authorization errors (403)
	ErrorTypeAuthorization ErrorType = "authorization"
	// ErrorTypeNotFound represents resource not found errors (404)
	ErrorTypeNotFound ErrorType = "not_found"
	// ErrorTypeRateLimit represents rate limiting errors (429)
	ErrorTypeRateLimit ErrorType = "rate_limit"
	// ErrorTypeProvider represents provider-specific errors (502/503)
	ErrorTypeProvider ErrorType = "provider"
	// ErrorTypeTimeout represents timeout errors (504)
	ErrorTypeTimeout ErrorType = "timeout"
	// ErrorTypeInternal represents internal server errors (500)
	ErrorTypeInternal ErrorType = "internal"
	// ErrorTypeCircuitBreaker represents circuit breaker errors (503)
	ErrorTypeCircuitBreaker ErrorType = "circuit_breaker"
)

// AppError represents a structured application error
type AppError struct {
	Type       ErrorType `json:"type"`
	Message    string    `json:"message"`
	Code       string    `json:"code,omitempty"`
	StatusCode int       `json:"-"`
	Retryable  bool      `json:"retryable"`
	Cause      error     `json:"-"`
}

// Error implements the error interface
func (e *AppError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%s: %v", e.Message, e.Cause)
	}
	return e.Message
}

// Unwrap allows error unwrapping
func (e *AppError) Unwrap() error {
	return e.Cause
}

// IsRetryable returns whether the error is retryable
func (e *AppError) IsRetryable() bool {
	return e.Retryable
}

// GetStatusCode returns the HTTP status code for the error
func (e *AppError) GetStatusCode() int {
	if e.StatusCode > 0 {
		return e.StatusCode
	}
	
	// Default status codes based on error type
	switch e.Type {
	case ErrorTypeValidation:
		return http.StatusBadRequest
	case ErrorTypeAuthentication:
		return http.StatusUnauthorized
	case ErrorTypeAuthorization:
		return http.StatusForbidden
	case ErrorTypeNotFound:
		return http.StatusNotFound
	case ErrorTypeRateLimit:
		return http.StatusTooManyRequests
	case ErrorTypeProvider, ErrorTypeCircuitBreaker:
		return http.StatusBadGateway
	case ErrorTypeTimeout:
		return http.StatusGatewayTimeout
	default:
		return http.StatusInternalServerError
	}
}

// NewValidationError creates a validation error
func NewValidationError(message string, cause error) *AppError {
	return &AppError{
		Type:       ErrorTypeValidation,
		Message:    message,
		StatusCode: http.StatusBadRequest,
		Retryable:  false,
		Cause:      cause,
	}
}

// NewProviderError creates a provider error
func NewProviderError(provider, message string, cause error) *AppError {
	return &AppError{
		Type:       ErrorTypeProvider,
		Message:    fmt.Sprintf("provider %s error: %s", provider, message),
		Code:       fmt.Sprintf("PROVIDER_%s_ERROR", provider),
		StatusCode: http.StatusBadGateway,
		Retryable:  true,
		Cause:      cause,
	}
}

// NewTimeoutError creates a timeout error
func NewTimeoutError(operation string, cause error) *AppError {
	return &AppError{
		Type:       ErrorTypeTimeout,
		Message:    fmt.Sprintf("operation %s timed out", operation),
		StatusCode: http.StatusGatewayTimeout,
		Retryable:  true,
		Cause:      cause,
	}
}

// NewCircuitBreakerError creates a circuit breaker error
func NewCircuitBreakerError(service string) *AppError {
	return &AppError{
		Type:       ErrorTypeCircuitBreaker,
		Message:    fmt.Sprintf("service %s is currently unavailable (circuit breaker open)", service),
		Code:       "CIRCUIT_BREAKER_OPEN",
		StatusCode: http.StatusServiceUnavailable,
		Retryable:  true,
	}
}

// NewRateLimitError creates a rate limit error
func NewRateLimitError(limit string) *AppError {
	return &AppError{
		Type:       ErrorTypeRateLimit,
		Message:    "rate limit exceeded",
		Code:       "RATE_LIMIT_EXCEEDED",
		StatusCode: http.StatusTooManyRequests,
		Retryable:  true,
	}
}

// NewInternalError creates an internal server error
func NewInternalError(message string, cause error) *AppError {
	return &AppError{
		Type:       ErrorTypeInternal,
		Message:    "internal server error",
		StatusCode: http.StatusInternalServerError,
		Retryable:  false,
		Cause:      cause,
	}
}

// SanitizeError sanitizes an error for external consumption
func SanitizeError(err error) *AppError {
	if appErr, ok := err.(*AppError); ok {
		// Return a copy without internal details
		return &AppError{
			Type:       appErr.Type,
			Message:    appErr.Message,
			Code:       appErr.Code,
			StatusCode: appErr.GetStatusCode(),
			Retryable:  appErr.Retryable,
			// Don't expose internal cause
		}
	}
	
	// For unknown errors, return a generic internal error
	return NewInternalError("an unexpected error occurred", err)
}