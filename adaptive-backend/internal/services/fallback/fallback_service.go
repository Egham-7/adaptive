package fallback

import (
	"context"
	"fmt"
	"sync"
	"time"

	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// FallbackService provides reusable fallback logic for any API endpoint
type FallbackService struct {
	cfg *config.Config
}

// NewFallbackService creates a new fallback service
func NewFallbackService(cfg *config.Config) *FallbackService {
	return &FallbackService{
		cfg: cfg,
	}
}

// Execute runs the providers with the specified fallback configuration
func (fs *FallbackService) Execute(
	c *fiber.Ctx,
	providers []models.Alternative,
	fallbackConfig models.FallbackConfig,
	executeFunc models.ExecutionFunc,
	requestID string,
	isStream bool,
) error {
	if c == nil || executeFunc == nil || requestID == "" {
		return models.NewValidationError("invalid input parameters", nil)
	}
	if len(providers) == 0 {
		return models.NewValidationError("no providers available", nil)
	}

	if len(providers) == 1 || fallbackConfig.Mode == "" {
		// Only one provider or fallback disabled (empty mode), execute directly
		fiberlog.Infof("[%s] Using single provider: %s (%s)", requestID, providers[0].Provider, providers[0].Model)
		return executeFunc(c, providers[0], requestID)
	}

	switch fallbackConfig.Mode {
	case models.FallbackModeSequential:
		return fs.executeSequential(c, providers, executeFunc, requestID)
	case models.FallbackModeRace:
		return fs.executeRace(c, providers, fallbackConfig, executeFunc, requestID, isStream)
	default:
		fiberlog.Warnf("[%s] Unknown fallback mode %s, using sequential", requestID, fallbackConfig.Mode)
		return fs.executeSequential(c, providers, executeFunc, requestID)
	}
}

// executeSequential tries providers one by one until one succeeds
func (fs *FallbackService) executeSequential(
	c *fiber.Ctx,
	providers []models.Alternative,
	executeFunc models.ExecutionFunc,
	requestID string,
) error {
	for i, provider := range providers {
		providerType := "alternative"
		if i == 0 {
			providerType = "primary"
		}

		fiberlog.Infof("[%s] Using %s provider: %s (%s)", requestID, providerType, provider.Provider, provider.Model)

		if err := executeFunc(c, provider, requestID); err == nil {
			return nil
		} else {
			fiberlog.Warnf("[%s] %s provider %s (%s) failed: %v", requestID, providerType, provider.Provider, provider.Model, err)
		}
	}

	return fmt.Errorf("all providers failed")
}

// executeRace tries all providers in parallel and returns the first successful result
func (fs *FallbackService) executeRace(
	c *fiber.Ctx,
	providers []models.Alternative,
	fallbackConfig models.FallbackConfig,
	executeFunc models.ExecutionFunc,
	requestID string,
	isStream bool,
) error {
	// For streaming, race to first response then cancel others
	if isStream {
		fiberlog.Infof("[%s] Racing %d providers for streaming (first to respond wins)", requestID, len(providers))
		return fs.executeStreamingRace(c, providers, executeFunc, requestID)
	}

	fiberlog.Infof("[%s] Racing %d providers", requestID, len(providers))

	resultCh := make(chan models.FallbackResult, len(providers))

	// Create context with timeout if specified
	ctx := context.Background()
	if fallbackConfig.TimeoutMs > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(fallbackConfig.TimeoutMs)*time.Millisecond)
		defer cancel()
	}

	// Start all providers in parallel
	var wg sync.WaitGroup
	for _, provider := range providers {
		wg.Add(1)
		go func(prov models.Alternative) {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					fiberlog.Errorf("[%s] Panic in race provider %s: %v", requestID, prov.Provider, r)
					resultCh <- models.FallbackResult{
						Success:  false,
						Provider: prov,
						Error:    fmt.Errorf("panic: %v", r),
					}
				}
			}()

			start := time.Now()
			fiberlog.Debugf("[%s] Racing provider %s (%s)", requestID, prov.Provider, prov.Model)

			if err := executeFunc(c, prov, requestID); err == nil {
				fiberlog.Infof("[%s] Race winner: %s (%s)", requestID, prov.Provider, prov.Model)
				resultCh <- models.FallbackResult{
					Success:  true,
					Provider: prov,
					Error:    nil,
					Duration: time.Since(start),
				}
			} else {
				fiberlog.Warnf("[%s] Race provider %s (%s) failed: %v", requestID, prov.Provider, prov.Model, err)
				resultCh <- models.FallbackResult{
					Success:  false,
					Provider: prov,
					Error:    err,
					Duration: time.Since(start),
				}
			}
		}(provider)
	}

	// Context-aware cleanup with done channel coordination
	done := make(chan struct{})
	var closeOnce sync.Once

	// Goroutine 1: Wait for all provider goroutines to finish
	go func() {
		wg.Wait()
		close(done)
	}()

	// Goroutine 2: Coordinate cleanup respecting context cancellation
	go func() {
		select {
		case <-done:
			// All provider goroutines completed normally
			closeOnce.Do(func() { close(resultCh) })
		case <-ctx.Done():
			// Context cancelled, cleanup and return early
			closeOnce.Do(func() { close(resultCh) })
		}
	}()

	// Wait for results with proper context handling
	var errors []string
	failureCount := 0

	for {
		select {
		case result, ok := <-resultCh:
			if !ok {
				// Channel closed, all goroutines finished
				goto raceComplete
			}

			if result.Success {
				// First successful result wins
				fiberlog.Infof("[%s] Race completed successfully with %s in %v", requestID, result.Provider.Provider, result.Duration)
				return nil
			}

			failureCount++
			errors = append(errors, fmt.Sprintf("%s(%s): %v", result.Provider.Provider, result.Provider.Model, result.Error))

			// Check if we've received all results
			if failureCount == len(providers) {
				goto raceComplete
			}

		case <-ctx.Done():
			return fmt.Errorf("race cancelled or timed out: %w", ctx.Err())
		}
	}

raceComplete:
	// All providers failed
	return fmt.Errorf("all providers failed in race: %v", errors)
}

// executeStreamingRace races providers for streaming - first to respond wins
func (fs *FallbackService) executeStreamingRace(
	c *fiber.Ctx,
	providers []models.Alternative,
	executeFunc models.ExecutionFunc,
	requestID string,
) error {
	fiberlog.Infof("[%s] Starting streaming race with %d providers", requestID, len(providers))

	// Channel to receive the first successful result
	resultCh := make(chan models.FallbackResult, len(providers))

	// Context to cancel other providers when one succeeds
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start all providers in goroutines
	var wg sync.WaitGroup
	for _, provider := range providers {
		wg.Add(1)
		go func(prov models.Alternative) {
			defer wg.Done()

			// Check if context is already cancelled
			select {
			case <-ctx.Done():
				fiberlog.Debugf("[%s] Provider %s cancelled before starting", requestID, prov.Provider)
				return
			default:
			}

			start := time.Now()
			fiberlog.Debugf("[%s] Racing streaming provider %s (%s)", requestID, prov.Provider, prov.Model)

			// Execute the provider
			err := executeFunc(c, prov, requestID)
			duration := time.Since(start)

			if err == nil {
				// Success - this provider wins the race
				fiberlog.Infof("[%s] Streaming race winner: %s (%s) in %v", requestID, prov.Provider, prov.Model, duration)
				resultCh <- models.FallbackResult{
					Success:  true,
					Provider: prov,
					Duration: duration,
				}
			} else {
				// Failure - try next provider
				fiberlog.Warnf("[%s] Streaming provider %s failed in %v: %v", requestID, prov.Provider, duration, err)
				resultCh <- models.FallbackResult{
					Success:  false,
					Provider: prov,
					Error:    err,
					Duration: duration,
				}
			}
		}(provider)
	}

	// Wait for results in a separate goroutine
	go func() {
		wg.Wait()
		close(resultCh)
	}()

	// Process results as they come in
	var errors []error
	for result := range resultCh {
		if result.Success {
			// First successful provider wins - cancel the rest
			fiberlog.Infof("[%s] Streaming race complete, cancelling remaining providers", requestID)
			cancel()
			return nil
		}
		errors = append(errors, result.Error)
	}

	// All providers failed
	fiberlog.Errorf("[%s] All streaming providers failed in race: %v", requestID, errors)
	return fmt.Errorf("all streaming providers failed in race: %v", errors)
}

// GetFallbackConfig gets the merged fallback configuration from config and request
func (fs *FallbackService) GetFallbackConfig(requestFallback *models.FallbackConfig) models.FallbackConfig {
	merged := fs.cfg.MergeFallbackConfig(requestFallback)
	if merged == nil {
		return models.FallbackConfig{}
	}
	return *merged
}
