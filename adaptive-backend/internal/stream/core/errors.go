package core

import (
	"errors"
	"fmt"
	"strings"
)

// StreamErrorType categorizes different types of streaming errors
type StreamErrorType int

const (
	// Expected errors - not logged as errors
	ClientDisconnect StreamErrorType = iota
	StreamComplete

	// Unexpected errors - logged as errors
	ProviderError
	InternalError
)

// StreamError provides structured error handling
type StreamError struct {
	Type      StreamErrorType
	Message   string
	Cause     error
	RequestID string
}

func (e *StreamError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%s: %v", e.Message, e.Cause)
	}
	return e.Message
}

func (e *StreamError) Unwrap() error {
	return e.Cause
}

// IsExpected returns true if this error type is expected (client disconnect, etc.)
func (e *StreamError) IsExpected() bool {
	return e.Type == ClientDisconnect || e.Type == StreamComplete
}

// Error constructors
func NewClientDisconnectError(requestID string) *StreamError {
	return &StreamError{
		Type:      ClientDisconnect,
		Message:   "Client disconnected",
		RequestID: requestID,
	}
}

func NewStreamCompleteError(requestID string) *StreamError {
	return &StreamError{
		Type:      StreamComplete,
		Message:   "Stream completed normally",
		RequestID: requestID,
	}
}

func NewProviderError(requestID, provider string, cause error) *StreamError {
	return &StreamError{
		Type:      ProviderError,
		Message:   fmt.Sprintf("Provider %s error", provider),
		Cause:     cause,
		RequestID: requestID,
	}
}

func NewInternalError(requestID, message string, cause error) *StreamError {
	return &StreamError{
		Type:      InternalError,
		Message:   message,
		Cause:     cause,
		RequestID: requestID,
	}
}

// IsClientDisconnect checks if error is a client disconnect
func IsClientDisconnect(err error) bool {
	var streamErr *StreamError
	if errors.As(err, &streamErr) {
		return streamErr.Type == ClientDisconnect
	}
	return false
}

// IsExpectedError checks if error is expected (not a real error)
func IsExpectedError(err error) bool {
	var streamErr *StreamError
	if errors.As(err, &streamErr) {
		return streamErr.IsExpected()
	}
	return false
}

// IsConnectionClosed checks if error indicates closed connection
func IsConnectionClosed(err error) bool {
	if err == nil {
		return false
	}
	errStr := strings.ToLower(err.Error())
	return strings.Contains(errStr, "connection closed") ||
		strings.Contains(errStr, "broken pipe") ||
		strings.Contains(errStr, "connection reset") ||
		strings.Contains(errStr, "write: broken pipe") ||
		strings.Contains(errStr, "use of closed network connection")
}
