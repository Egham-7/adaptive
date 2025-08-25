package fallback

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

// FallbackService provides reusable fallback logic for any API endpoint
type FallbackService struct {
	cfg *config.Config
}

// NewFallbackService creates a new fallback service
func NewFallbackService(cfg *config.Config) *FallbackService {
	if cfg == nil {
		panic("NewFallbackService: cfg cannot be nil")
	}
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
	protocolType string,
	isStream bool,
) error {
	if c == nil || executeFunc == nil || requestID == "" || protocolType == "" {
		return models.NewValidationError("invalid input parameters", nil)
	}
	if len(providers) == 0 {
		return models.NewValidationError("no providers available", nil)
	}

	if len(providers) == 1 || fallbackConfig.Mode == "" {
		// Only one provider or fallback disabled (empty mode), execute directly
		fiberlog.Infof("[%s] Using single %s provider: %s (%s)", requestID, protocolType, providers[0].Provider, providers[0].Model)
		return executeFunc(c, providers[0], requestID)
	}

	switch fallbackConfig.Mode {
	case models.FallbackModeSequential:
		return fs.executeSequential(c, providers, executeFunc, requestID, protocolType)
	case models.FallbackModeRace:
		return fs.executeRace(c, providers, fallbackConfig, executeFunc, requestID, protocolType, isStream)
	default:
		fiberlog.Warnf("[%s] Unknown fallback mode %s, using sequential", requestID, fallbackConfig.Mode)
		return fs.executeSequential(c, providers, executeFunc, requestID, protocolType)
	}
}

// executeSequential tries providers one by one until one succeeds
func (fs *FallbackService) executeSequential(
	c *fiber.Ctx,
	providers []models.Alternative,
	executeFunc models.ExecutionFunc,
	requestID string,
	protocolType string,
) error {
	for i, provider := range providers {
		providerType := "alternative"
		if i == 0 {
			providerType = "primary"
		}

		fiberlog.Infof("[%s] Using %s %s provider: %s (%s)", requestID, providerType, protocolType, provider.Provider, provider.Model)

		if err := executeFunc(c, provider, requestID); err == nil {
			return nil
		} else {
			fiberlog.Warnf("[%s] %s %s provider %s (%s) failed: %v", requestID, providerType, protocolType, provider.Provider, provider.Model, err)
		}
	}

	return fmt.Errorf("all %s providers failed", protocolType)
}

// executeRace tries all providers in parallel and returns the first successful result
func (fs *FallbackService) executeRace(
	c *fiber.Ctx,
	providers []models.Alternative,
	fallbackConfig models.FallbackConfig,
	executeFunc models.ExecutionFunc,
	requestID string,
	protocolType string,
	isStream bool,
) error {
	// For streaming requests, fall back to sequential since we can't race streams
	if isStream {
		fiberlog.Infof("[%s] Race mode not supported for streaming, using sequential fallback", requestID)
		return fs.executeSequential(c, providers, executeFunc, requestID, protocolType)
	}

	fiberlog.Infof("[%s] Racing %d %s providers", requestID, len(providers), protocolType)

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
	return fmt.Errorf("all %s providers failed in race: %v", protocolType, errors)
}

// GetFallbackConfig gets the merged fallback configuration from config and request
func (fs *FallbackService) GetFallbackConfig(requestFallback *models.FallbackConfig) models.FallbackConfig {
	merged := fs.cfg.MergeFallbackConfig(requestFallback)
	if merged == nil {
		return models.FallbackConfig{}
	}
	return *merged
}
