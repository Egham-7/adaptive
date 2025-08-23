package models

import (
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"time"
)

// RaceResult represents a parallel provider race outcome.
type RaceResult struct {
	Provider     provider_interfaces.LLMProvider
	ProviderName string
	ModelName    string
	TaskType     string
	Duration     time.Duration
	Error        error
}
