package completions

import (
	"adaptive-backend/internal/config"
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/alitto/pond/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

const (
	fallbackDefaultTimeout    = 10 * time.Second
	fallbackDefaultMaxRetries = 3
)

// Candidate represents a model/provider/protocol candidate for completion.
type Candidate struct {
	Name     string
	Provider provider_interfaces.LLMProvider
	Protocol models.ProtocolType
}

// FallbackService handles provider selection with configurable fallback strategies
type FallbackService struct {
	cfg        *config.Config
	mode       models.FallbackMode
	timeout    time.Duration
	maxRetries int
	workerPool pond.Pool
}

// NewFallbackService creates a new fallback service reading config values
func NewFallbackService(cfg *config.Config) *FallbackService {
	// Use the FallbackMode from config directly
	mode := cfg.Fallback.Mode

	// Parse timeout from config (with default if absent or invalid)
	timeout := fallbackDefaultTimeout
	if cfg.Fallback.TimeoutMs > 0 {
		timeout = time.Duration(cfg.Fallback.TimeoutMs) * time.Millisecond
	}

	// Parse max retries from config (with default if absent or invalid)
	maxRetries := fallbackDefaultMaxRetries
	if cfg.Fallback.MaxRetries > 0 {
		maxRetries = cfg.Fallback.MaxRetries
	}

	// Set default worker pool configuration if not specified
	workers := 10
	queueSize := 100
	if cfg.Fallback.WorkerPool.Workers > 0 {
		workers = cfg.Fallback.WorkerPool.Workers
	}
	if cfg.Fallback.WorkerPool.QueueSize > 0 {
		queueSize = cfg.Fallback.WorkerPool.QueueSize
	}

	// Create worker pool with queue size option
	var workerPool pond.Pool
	if queueSize > 0 {
		workerPool = pond.NewPool(workers, pond.WithQueueSize(queueSize))
	} else {
		workerPool = pond.NewPool(workers)
	}

	return &FallbackService{
		cfg:        cfg,
		mode:       mode,
		timeout:    timeout,
		maxRetries: maxRetries,
		workerPool: workerPool,
	}
}

// SetMode changes the fallback mode
func (fs *FallbackService) SetMode(mode models.FallbackMode) {
	fs.mode = mode
}

// SetTimeout configures the timeout for racing requests
func (fs *FallbackService) SetTimeout(timeout time.Duration) {
	fs.timeout = timeout
}

// GetMaxRetries returns the configured max retries
func (fs *FallbackService) GetMaxRetries() int {
	return fs.maxRetries
}

// SelectAlternative selects the best alternative provider when primary fails
// It modifies the alternatives slice by removing tried providers to avoid retries
func (fs *FallbackService) SelectAlternative(
	ctx context.Context,
	alternatives *[]models.Alternative,
	providerConfigs map[string]*models.ProviderConfig,
	requestID string,
) (*models.RaceResult, error) {
	if len(*alternatives) == 0 {
		return nil, fmt.Errorf("no alternatives available")
	}

	switch fs.mode {
	case models.FallbackModeRace:
		return fs.raceAlternatives(ctx, alternatives, providerConfigs, requestID)
	case models.FallbackModeSequential:
		return fs.selectSequentialAlternative(ctx, alternatives, providerConfigs, requestID)
	default:
		return nil, fmt.Errorf("unsupported fallback mode: %s", fs.mode)
	}
}

// raceAlternatives races alternative providers in parallel
func (fs *FallbackService) raceAlternatives(
	ctx context.Context,
	alternatives *[]models.Alternative,
	providerConfigs map[string]*models.ProviderConfig,
	requestID string,
) (*models.RaceResult, error) {
	fiberlog.Infof("[%s] Racing %d alternative providers", requestID, len(*alternatives))

	// For race mode, we try all alternatives at once
	result, err := fs.raceAllProviders(ctx, *alternatives, providerConfigs, requestID)

	// Always clear the slice since we tried all providers
	alternativeCount := len(*alternatives)
	*alternatives = (*alternatives)[:0]

	if err == nil {
		// Success - we found a winner
		fiberlog.Infof("[%s] Race winner found: %s (%s), cleared all alternatives", requestID, result.ProviderName, result.ModelName)
	} else {
		// All failed - we tried them all
		fiberlog.Warnf("[%s] All %d alternatives failed in race, cleared all alternatives", requestID, alternativeCount)
	}

	return result, err
}

// selectSequentialAlternative tries the first alternative and removes it from the slice
func (fs *FallbackService) selectSequentialAlternative(
	ctx context.Context,
	alternatives *[]models.Alternative,
	providerConfigs map[string]*models.ProviderConfig,
	requestID string,
) (*models.RaceResult, error) {
	if len(*alternatives) == 0 {
		return nil, fmt.Errorf("no alternatives available")
	}

	fiberlog.Infof("[%s] Sequential fallback: trying next alternative from %d remaining", requestID, len(*alternatives))

	// Try the first alternative
	alt := (*alternatives)[0]
	fiberlog.Infof("[%s] Trying alternative: %s (%s)", requestID, alt.Provider, alt.Model)
	result := fs.tryProviderConnection(alt, providerConfigs, requestID)

	if result.Error == nil {
		// Success - remove this alternative from the slice and return it
		*alternatives = (*alternatives)[1:]
		fiberlog.Infof("[%s] Alternative provider %s (%s) available, %d alternatives remaining", requestID, alt.Provider, alt.Model, len(*alternatives))
		return result, nil
	}

	// Failed - remove this alternative from the slice
	*alternatives = (*alternatives)[1:]
	fiberlog.Warnf("[%s] Alternative provider %s (%s) failed: %v, %d alternatives remaining", requestID, alt.Provider, alt.Model, result.Error, len(*alternatives))

	return nil, fmt.Errorf("alternative %s (%s) failed: %w", alt.Provider, alt.Model, result.Error)
}

// raceAllProviders races all providers in parallel using worker pool
func (fs *FallbackService) raceAllProviders(
	ctx context.Context,
	options []models.Alternative,
	providerConfigs map[string]*models.ProviderConfig,
	requestID string,
) (*models.RaceResult, error) {
	// Create context with timeout
	raceCtx, cancel := context.WithTimeout(ctx, fs.timeout)
	defer cancel()

	// Use fresh channel per request (never pool channels - unsafe!)
	resultCh := make(chan *models.RaceResult, len(options))

	// Submit tasks to worker pool instead of creating unlimited goroutines
	var wg sync.WaitGroup
	for _, option := range options {
		wg.Add(1)
		task := fs.workerPool.SubmitErr(func() error {
			defer func() {
				wg.Done()
				// Ensure we don't leak if panic occurs
				if r := recover(); r != nil {
					fiberlog.Errorf("[%s] Panic in provider %s (%s): %v", requestID, option.Provider, option.Model, r)
				}
			}()

			// Pass the race context to provider connection
			result := fs.tryProviderConnectionWithContext(raceCtx, option, providerConfigs, requestID)

			// Non-blocking send to avoid goroutine leak if context is cancelled
			select {
			case resultCh <- result:
				fiberlog.Debugf("[%s] Result sent for provider %s (%s)", requestID, option.Provider, option.Model)
			case <-raceCtx.Done():
				fiberlog.Debugf("[%s] Context cancelled for provider %s (%s), result discarded", requestID, option.Provider, option.Model)
			}
			return nil
		})
		// Ignore the task result since we're handling results via the channel
		_ = task
	}

	// Close result channel when all tasks are done
	go func() {
		wg.Wait()
		close(resultCh)
	}()

	// Wait for first successful result or all failures
	var allErrors []error
	resultsReceived := 0

	for {
		select {
		case result, ok := <-resultCh:
			if !ok {
				// Channel closed, all tasks finished
				fiberlog.Warnf("[%s] All %d providers failed", requestID, len(options))
				return nil, fmt.Errorf("all %d providers failed: %v", len(options), allErrors)
			}

			resultsReceived++
			fiberlog.Debugf("[%s] Received result %d/%d from %s (%s): success=%v",
				requestID, resultsReceived, len(options), result.ProviderName, result.ModelName, result.Error == nil)

			if result.Error == nil {
				// First successful result wins
				fiberlog.Infof("[%s] Winner: %s (%s) in %v",
					requestID, result.ProviderName, result.ModelName, result.Duration)
				return result, nil
			} else {
				allErrors = append(allErrors, fmt.Errorf("%s (%s): %w",
					result.ProviderName, result.ModelName, result.Error))
			}

		case <-raceCtx.Done():
			// Context timeout or cancellation
			fiberlog.Warnf("[%s] Race timeout after %v with %d/%d results received",
				requestID, fs.timeout, resultsReceived, len(options))
			return nil, fmt.Errorf("race timeout after %v: %v", fs.timeout, allErrors)
		}
	}
}

// tryProviderConnectionWithContext tests if a provider is available and creates it with context
func (fs *FallbackService) tryProviderConnectionWithContext(
	ctx context.Context,
	option models.Alternative,
	providerConfigs map[string]*models.ProviderConfig,
	requestID string,
) *models.RaceResult {
	// Check if context is already cancelled
	select {
	case <-ctx.Done():
		return &models.RaceResult{
			ProviderName: option.Provider,
			ModelName:    option.Model,
			Duration:     0,
			Error:        fmt.Errorf("context cancelled before provider connection"),
		}
	default:
	}

	return fs.tryProviderConnection(option, providerConfigs, requestID)
}

// tryProviderConnection tests if a provider is available and creates it
func (fs *FallbackService) tryProviderConnection(
	option models.Alternative,
	providerConfigs map[string]*models.ProviderConfig,
	requestID string,
) *models.RaceResult {
	fiberlog.Debugf("[%s] Testing connection to provider %s (%s)", requestID, option.Provider, option.Model)
	start := time.Now()
	result := &models.RaceResult{
		ProviderName: option.Provider,
		ModelName:    option.Model,
		Duration:     0,
	}

	// Create provider
	var provider provider_interfaces.LLMProvider
	var err error

	fiberlog.Debugf("[%s] Creating LLM provider: %s", requestID, option.Provider)
	provider, err = providers.NewLLMProvider(fs.cfg, option.Provider, providerConfigs)
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

// BuildCandidates builds candidates from protocol response (for backward compatibility)
func (fs *FallbackService) BuildCandidates(resp *models.ProtocolResponse) ([]Candidate, error) {
	return fs.BuildCandidatesWithCustomConfig(resp, nil)
}

// BuildCandidatesWithCustomConfig builds candidates from protocol response with custom config support
func (fs *FallbackService) BuildCandidatesWithCustomConfig(resp *models.ProtocolResponse, providerConfigs map[string]*models.ProviderConfig) ([]Candidate, error) {
	switch resp.Protocol {
	case models.ProtocolStandardLLM:
		return fs.buildStandardCandidates(resp.Standard, providerConfigs)
	case models.ProtocolMinion:
		return fs.buildMinionCandidates(resp.Minion, providerConfigs)
	case models.ProtocolMinionsProtocol:
		stds, err := fs.buildStandardCandidates(resp.Standard, providerConfigs)
		if err != nil {
			return nil, err
		}
		mins, err := fs.buildMinionCandidates(resp.Minion, providerConfigs)
		if err != nil {
			return nil, err
		}
		if len(stds) > 0 && len(mins) > 0 {
			return []Candidate{
				{stds[0].Name, stds[0].Provider, models.ProtocolStandardLLM},
				{mins[0].Name, mins[0].Provider, models.ProtocolMinion},
			}, nil
		}
		return append(stds, mins...), nil
	default:
		return nil, fmt.Errorf("unsupported protocol %s", resp.Protocol)
	}
}

// buildStandardCandidates returns primary + fallback LLMs.
func (fs *FallbackService) buildStandardCandidates(std *models.StandardLLMInfo, providerConfigs map[string]*models.ProviderConfig) ([]Candidate, error) {
	if std == nil {
		return nil, fmt.Errorf("standard info is nil")
	}

	var out []Candidate

	svc, err := providers.NewLLMProvider(fs.cfg, std.Provider, providerConfigs)
	if err != nil {
		return nil, fmt.Errorf("standard provider %s: %w", std.Provider, err)
	}
	out = append(out, Candidate{std.Provider, svc, models.ProtocolStandardLLM})

	for _, alt := range std.Alternatives {
		svc, err := providers.NewLLMProvider(fs.cfg, alt.Provider, providerConfigs)
		if err != nil {
			return nil, fmt.Errorf("standard alternative provider %s: %w", alt.Provider, err)
		}
		out = append(out, Candidate{alt.Provider, svc, models.ProtocolStandardLLM})
	}
	return out, nil
}

// buildMinionCandidates returns primary + fallback minions.
func (fs *FallbackService) buildMinionCandidates(min *models.MinionInfo, providerConfigs map[string]*models.ProviderConfig) ([]Candidate, error) {
	if min == nil {
		return nil, fmt.Errorf("minion info is nil")
	}

	var out []Candidate

	svc, err := providers.NewLLMProvider(fs.cfg, min.Provider, providerConfigs)
	if err != nil {
		return nil, fmt.Errorf("%s model %s: %w", min.Provider, min.Model, err)
	}
	out = append(out, Candidate{min.Provider, svc, models.ProtocolMinion})

	for _, alt := range min.Alternatives {
		svc, err := providers.NewLLMProvider(fs.cfg, alt.Provider, providerConfigs)
		if err != nil {
			return nil, fmt.Errorf("%s alternative model %s: %w", alt.Provider, alt.Model, err)
		}
		out = append(out, Candidate{alt.Provider, svc, models.ProtocolMinion})
	}
	return out, nil
}
