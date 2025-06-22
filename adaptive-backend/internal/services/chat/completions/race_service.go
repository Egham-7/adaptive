package completions

import (
	"context"
	"fmt"
	"sync"
	"time"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/minions"
	"adaptive-backend/internal/services/providers"
	"adaptive-backend/internal/services/providers/provider_interfaces"

	fiberlog "github.com/gofiber/fiber/v2/log"
)

// RaceResult represents the result of a parallel request race
type RaceResult struct {
	Provider     provider_interfaces.LLMProvider
	ProviderName string
	ModelName    string
	TaskType     string
	Duration     time.Duration
	Error        error
}

// RaceService handles provider selection with automatic failover
// Tries primary provider first, then races alternatives if primary fails
type RaceService struct {
	minionRegistry *minions.MinionRegistry
	timeout        time.Duration
}

// NewRaceService creates a new race service
func NewRaceService(minionRegistry *minions.MinionRegistry) *RaceService {
	return &RaceService{
		minionRegistry: minionRegistry,
		timeout:        10 * time.Second, // Default timeout
	}
}

// RaceProviders tries primary first, then races alternatives if primary fails
// Returns the fastest available provider ready to use for requests
// This provides automatic failover with minimal latency overhead
func (rs *RaceService) RaceProviders(
	ctx context.Context,
	primary models.Alternative,
	alternatives []models.Alternative,
	requestID string,
) (*RaceResult, error) {
	fiberlog.Infof("[%s] Trying primary provider %s (%s) first", requestID, primary.Provider, primary.Model)

	// Try primary provider first (just test connectivity)
	primaryResult := rs.tryProviderConnection(primary, requestID)

	if primaryResult.Error == nil {
		fiberlog.Infof("[%s] Primary provider %s (%s) available in %v",
			requestID, primary.Provider, primary.Model, primaryResult.Duration)
		return primaryResult, nil
	}

	fiberlog.Warnf("[%s] Primary provider %s (%s) failed: %v",
		requestID, primary.Provider, primary.Model, primaryResult.Error)

	// If no alternatives, return primary failure
	if len(alternatives) == 0 {
		return nil, fmt.Errorf("primary provider failed and no alternatives available: %w", primaryResult.Error)
	}

	// Primary failed, race all providers (including primary retry)
	allOptions := append([]models.Alternative{primary}, alternatives...)
	fiberlog.Infof("[%s] Primary failed, racing %d providers for fastest available", requestID, len(allOptions))

	// Create context with timeout
	raceCtx, cancel := context.WithTimeout(ctx, rs.timeout)
	defer cancel()

	// Channel to collect results
	resultCh := make(chan *RaceResult, len(allOptions))
	var wg sync.WaitGroup

	// Start parallel requests
	for i, option := range allOptions {
		wg.Add(1)
		go func(idx int, opt models.Alternative) {
			defer wg.Done()
			result := rs.tryProviderConnection(opt, requestID)

			select {
			case resultCh <- result:
			case <-raceCtx.Done():
				fiberlog.Debugf("[%s] Context cancelled for provider %s (%s)", requestID, opt.Provider, opt.Model)
			}
		}(i, option)
	}

	// Close result channel when all goroutines are done
	go func() {
		wg.Wait()
		close(resultCh)
	}()

	// Wait for first successful result or all failures
	var allErrors []error
	resultsReceived := 0

	fiberlog.Debugf("[%s] Waiting for race results from %d providers", requestID, len(allOptions))

	for result := range resultCh {
		resultsReceived++
		fiberlog.Debugf("[%s] Received result %d/%d from %s (%s): success=%v",
			requestID, resultsReceived, len(allOptions), result.ProviderName, result.ModelName, result.Error == nil)

		if result.Error == nil {
			// First successful result wins
			fiberlog.Infof("[%s] Winner: %s (%s) in %v",
				requestID, result.ProviderName, result.ModelName, result.Duration)

			// Cancel remaining requests
			cancel()
			return result, nil
		} else {
			allErrors = append(allErrors, fmt.Errorf("%s (%s): %w",
				result.ProviderName, result.ModelName, result.Error))
		}

		// If we've received all results and none succeeded
		if resultsReceived == len(allOptions) {
			fiberlog.Warnf("[%s] All %d providers failed", requestID, len(allOptions))
			break
		}
	}

	// All providers failed
	return nil, fmt.Errorf("all %d providers failed: %v", len(allOptions), allErrors)
}

// tryProviderConnection tests if a provider is available and creates it
// This is a lightweight check that only creates the provider instance
func (rs *RaceService) tryProviderConnection(
	option models.Alternative,
	requestID string,
) *RaceResult {
	fiberlog.Debugf("[%s] Testing connection to provider %s (%s)", requestID, option.Provider, option.Model)
	start := time.Now()
	result := &RaceResult{
		ProviderName: option.Provider,
		ModelName:    option.Model,
		Duration:     0,
	}

	// Create provider
	var provider provider_interfaces.LLMProvider
	var err error

	if option.Provider == "minion" {
		// For minions, the model field contains the task type
		taskType := option.Model
		fiberlog.Debugf("[%s] Creating minion provider for task: %s", requestID, taskType)
		provider, err = providers.NewLLMProvider("minion", &taskType, rs.minionRegistry)
		result.TaskType = taskType
	} else {
		fiberlog.Debugf("[%s] Creating LLM provider: %s", requestID, option.Provider)
		provider, err = providers.NewLLMProvider(option.Provider, nil, rs.minionRegistry)
	}

	if err != nil {
		result.Error = fmt.Errorf("failed to create provider %s: %w", option.Provider, err)
		result.Duration = time.Since(start)
		fiberlog.Errorf("[%s] Provider creation failed for %s: %v", requestID, option.Provider, err)
		return result
	}

	result.Provider = provider
	result.Duration = time.Since(start)

	fiberlog.Debugf("[%s] Provider %s (%s) available after %v",
		requestID, option.Provider, option.Model, result.Duration)

	return result
}

// SetTimeout configures the timeout for racing requests
func (rs *RaceService) SetTimeout(timeout time.Duration) {
	rs.timeout = timeout
}

// GetTimeout returns the current timeout setting
func (rs *RaceService) GetTimeout() time.Duration {
	return rs.timeout
}
