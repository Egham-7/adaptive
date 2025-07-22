package services

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	fiberlog "github.com/gofiber/fiber/v2/log"
)

// HealthService manages health checks for dependent services
type HealthService struct {
	protocolManagerClient *Client
	adaptiveClient        *Client
	checkInterval         time.Duration
	timeout               time.Duration
	servicesReady         bool
	mu                    sync.RWMutex
}

// HealthStatus represents the health status of a service
type HealthStatus struct {
	Service   string    `json:"service"`
	Healthy   bool      `json:"healthy"`
	Error     string    `json:"error,omitempty"`
	CheckedAt time.Time `json:"checked_at"`
}

// OverallHealth represents the overall health status
type OverallHealth struct {
	Healthy  bool           `json:"healthy"`
	Services []HealthStatus `json:"services"`
	Message  string         `json:"message,omitempty"`
}

// NewHealthService creates a new health service with extended timeouts for model loading
func NewHealthService() *HealthService {
	// Get protocol manager base URL
	protocolManagerURL := os.Getenv("ADAPTIVE_AI_BASE_URL")
	if protocolManagerURL == "" {
		protocolManagerURL = "http://localhost:8000"
	}

	// Get adaptive provider base URL
	adaptiveURL := os.Getenv("ADAPTIVE_BASE_URL")
	if adaptiveURL == "" {
		adaptiveURL = "https://api.adaptive.ai/v1"
	}

	return &HealthService{
		protocolManagerClient: NewClient(protocolManagerURL),
		adaptiveClient:        NewClient(adaptiveURL),
		checkInterval:         10 * time.Second, // Longer interval for model loading
		timeout:               30 * time.Second, // Longer timeout for model loading
		servicesReady:         false,
	}
}

// CheckHealth performs a health check on both services
func (hs *HealthService) CheckHealth(ctx context.Context) *OverallHealth {
	hs.mu.RLock()
	ready := hs.servicesReady
	hs.mu.RUnlock()

	// If services haven't been marked as ready yet, do a lightweight check
	if !ready {
		return &OverallHealth{
			Healthy: false,
			Message: "Services are still starting up (loading models and classifiers)",
			Services: []HealthStatus{
				{
					Service:   "protocol_manager",
					Healthy:   false,
					Error:     "Starting up",
					CheckedAt: time.Now(),
				},
				{
					Service:   "adaptive_provider",
					Healthy:   false,
					Error:     "Starting up",
					CheckedAt: time.Now(),
				},
			},
		}
	}

	// Perform actual health checks
	var wg sync.WaitGroup
	results := make(chan HealthStatus, 2)

	// Check protocol manager health
	wg.Add(1)
	go func() {
		defer wg.Done()
		results <- hs.checkProtocolManagerHealth(ctx)
	}()

	// Check adaptive provider health
	wg.Add(1)
	go func() {
		defer wg.Done()
		results <- hs.checkAdaptiveHealth(ctx)
	}()

	// Wait for all checks to complete
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	var services []HealthStatus
	allHealthy := true

	for status := range results {
		services = append(services, status)
		if !status.Healthy {
			allHealthy = false
		}
	}

	message := "All services are healthy"
	if !allHealthy {
		message = "Some services are unhealthy"
	}

	return &OverallHealth{
		Healthy:  allHealthy,
		Services: services,
		Message:  message,
	}
}

// checkProtocolManagerHealth checks the protocol manager /health endpoint
func (hs *HealthService) checkProtocolManagerHealth(ctx context.Context) HealthStatus {
	status := HealthStatus{
		Service:   "protocol_manager",
		CheckedAt: time.Now(),
	}

	opts := &RequestOptions{
		Context: ctx,
		Timeout: hs.timeout,
		Retries: 1, // Fewer retries since models take time to load
	}

	err := hs.protocolManagerClient.Get("/health", nil, opts)

	if err != nil {
		status.Healthy = false
		status.Error = err.Error()
		fiberlog.Debugf("Protocol manager health check failed: %v", err)
	} else {
		status.Healthy = true
		fiberlog.Debugf("Protocol manager health check passed")
	}

	return status
}

// checkAdaptiveHealth checks the adaptive provider /health endpoint
func (hs *HealthService) checkAdaptiveHealth(ctx context.Context) HealthStatus {
	status := HealthStatus{
		Service:   "adaptive_provider",
		CheckedAt: time.Now(),
	}

	opts := &RequestOptions{
		Context: ctx,
		Timeout: hs.timeout,
		Retries: 1, // Fewer retries since models take time to load
	}

	err := hs.adaptiveClient.Get("/health", nil, opts)

	if err != nil {
		status.Healthy = false
		status.Error = err.Error()
		fiberlog.Debugf("Adaptive provider health check failed: %v", err)
	} else {
		status.Healthy = true
		fiberlog.Debugf("Adaptive provider health check passed")
	}

	return status
}

// WaitForServices waits for all services to become healthy before returning
// This supports very long startup times for model loading
func (hs *HealthService) WaitForServices(ctx context.Context, maxWaitTime time.Duration) error {
	fiberlog.Info("Waiting for dependent services to become healthy (this may take several minutes for model loading)...")

	deadline := time.Now().Add(maxWaitTime)
	ticker := time.NewTicker(hs.checkInterval)
	defer ticker.Stop()

	checkCount := 0

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			checkCount++

			if time.Now().After(deadline) {
				return fmt.Errorf("timeout waiting for services to become healthy after %v", maxWaitTime)
			}

			// Create a context with timeout for this check
			checkCtx, cancel := context.WithTimeout(ctx, hs.timeout)

			// Do the actual health check
			var wg sync.WaitGroup
			results := make(chan HealthStatus, 2)

			wg.Add(1)
			go func() {
				defer wg.Done()
				results <- hs.checkProtocolManagerHealth(checkCtx)
			}()

			wg.Add(1)
			go func() {
				defer wg.Done()
				results <- hs.checkAdaptiveHealth(checkCtx)
			}()

			go func() {
				wg.Wait()
				close(results)
			}()

			allHealthy := true
			var services []HealthStatus
			for status := range results {
				services = append(services, status)
				if !status.Healthy {
					allHealthy = false
				}
			}

			cancel()

			if allHealthy {
				hs.mu.Lock()
				hs.servicesReady = true
				hs.mu.Unlock()
				fiberlog.Info("All services are healthy and ready!")
				return nil
			}

			// Log progress every 30 seconds
			if checkCount%3 == 0 {
				var unhealthyServices []string
				for _, service := range services {
					if !service.Healthy {
						unhealthyServices = append(unhealthyServices, fmt.Sprintf("%s: %s", service.Service, service.Error))
					}
				}

				remaining := time.Until(deadline).Round(time.Second)
				fiberlog.Infof("Still waiting for services (%.0f%% of max wait time used). Unhealthy: %v",
					float64(maxWaitTime-remaining)/float64(maxWaitTime)*100, unhealthyServices)
			}
		}
	}
}

// IsReady returns whether all services have been confirmed healthy at least once
func (hs *HealthService) IsReady() bool {
	hs.mu.RLock()
	defer hs.mu.RUnlock()
	return hs.servicesReady
}

// Close closes the underlying HTTP clients
func (hs *HealthService) Close() {
	if hs.protocolManagerClient != nil {
		hs.protocolManagerClient.Close()
	}
	if hs.adaptiveClient != nil {
		hs.adaptiveClient.Close()
	}
}
