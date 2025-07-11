package completions

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"context"
	"fmt"
	"sync"
	"time"

	fiberlog "github.com/gofiber/fiber/v2/log"
)

// FallbackMode defines the strategy for handling provider failures
type FallbackMode int

const (
	// FallbackModeRace tries primary first, then races all alternatives in parallel
	FallbackModeRace FallbackMode = iota
	// FallbackModeSequential tries providers one by one in order
	FallbackModeSequential
)

const (
	fallbackDefaultTimeout = 10 * time.Second
)

// Candidate represents a model/provider/protocol candidate for completion.
type Candidate struct {
	Name     string
	Provider provider_interfaces.LLMProvider
	Protocol models.ProtocolType
}

// FallbackService handles provider selection with configurable fallback strategies
type FallbackService struct {
	mode    FallbackMode
	timeout time.Duration
}

// NewFallbackService creates a new fallback service with race mode by default
func NewFallbackService() *FallbackService {
	return &FallbackService{
		mode:    FallbackModeRace,
		timeout: fallbackDefaultTimeout,
	}
}

// NewFallbackServiceWithMode creates a new fallback service with specified mode
func NewFallbackServiceWithMode(mode FallbackMode) *FallbackService {
	return &FallbackService{
		mode:    mode,
		timeout: fallbackDefaultTimeout,
	}
}

// SetMode changes the fallback mode
func (fs *FallbackService) SetMode(mode FallbackMode) {
	fs.mode = mode
}

// SetTimeout configures the timeout for racing requests
func (fs *FallbackService) SetTimeout(timeout time.Duration) {
	fs.timeout = timeout
}

// SelectAlternative selects the best alternative provider when primary fails
func (fs *FallbackService) SelectAlternative(
	ctx context.Context,
	alternatives []models.Alternative,
	requestID string,
) (*models.RaceResult, error) {
	if len(alternatives) == 0 {
		return nil, fmt.Errorf("no alternatives available")
	}

	switch fs.mode {
	case FallbackModeRace:
		return fs.raceAlternatives(ctx, alternatives, requestID)
	case FallbackModeSequential:
		return fs.selectSequentialAlternative(ctx, alternatives, requestID)
	default:
		return nil, fmt.Errorf("unsupported fallback mode: %d", fs.mode)
	}
}

// raceAlternatives races alternative providers in parallel
func (fs *FallbackService) raceAlternatives(
	ctx context.Context,
	alternatives []models.Alternative,
	requestID string,
) (*models.RaceResult, error) {
	fiberlog.Infof("[%s] Racing %d alternative providers", requestID, len(alternatives))
	return fs.raceAllProviders(ctx, alternatives, requestID)
}

// selectSequentialAlternative tries alternative providers one by one in order
func (fs *FallbackService) selectSequentialAlternative(
	ctx context.Context,
	alternatives []models.Alternative,
	requestID string,
) (*models.RaceResult, error) {
	fiberlog.Infof("[%s] Sequential fallback: trying %d alternatives", requestID, len(alternatives))

	// Try alternatives one by one
	for i, alt := range alternatives {
		fiberlog.Infof("[%s] Trying alternative %d: %s (%s)", requestID, i+1, alt.Provider, alt.Model)
		result := fs.tryProviderConnection(alt, requestID)
		if result.Error == nil {
			fiberlog.Infof("[%s] Alternative %d provider %s (%s) available", requestID, i+1, alt.Provider, alt.Model)
			return result, nil
		}
		fiberlog.Warnf("[%s] Alternative %d provider %s (%s) failed: %v", requestID, i+1, alt.Provider, alt.Model, result.Error)
	}

	return nil, fmt.Errorf("all %d alternatives failed", len(alternatives))
}

// raceAllProviders races all providers in parallel
func (fs *FallbackService) raceAllProviders(
	ctx context.Context,
	options []models.Alternative,
	requestID string,
) (*models.RaceResult, error) {
	// Create context with timeout
	raceCtx, cancel := context.WithTimeout(ctx, fs.timeout)
	defer cancel()

	// Channel to collect results
	resultCh := make(chan *models.RaceResult, len(options))
	var wg sync.WaitGroup

	// Start parallel requests
	for i, option := range options {
		wg.Add(1)
		go func(idx int, opt models.Alternative) {
			defer wg.Done()
			result := fs.tryProviderConnection(opt, requestID)

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

	for result := range resultCh {
		resultsReceived++
		fiberlog.Debugf("[%s] Received result %d/%d from %s (%s): success=%v",
			requestID, resultsReceived, len(options), result.ProviderName, result.ModelName, result.Error == nil)

		if result.Error == nil {
			// First successful result wins
			fiberlog.Infof("[%s] Winner: %s (%s) in %v",
				requestID, result.ProviderName, result.ModelName, result.Duration)
			cancel()
			return result, nil
		} else {
			allErrors = append(allErrors, fmt.Errorf("%s (%s): %w",
				result.ProviderName, result.ModelName, result.Error))
		}

		// If we've received all results and none succeeded
		if resultsReceived == len(options) {
			fiberlog.Warnf("[%s] All %d providers failed", requestID, len(options))
			break
		}
	}

	// All providers failed
	return nil, fmt.Errorf("all %d providers failed: %v", len(options), allErrors)
}

// tryProviderConnection tests if a provider is available and creates it
func (fs *FallbackService) tryProviderConnection(
	option models.Alternative,
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
	provider, err = providers.NewLLMProvider(option.Provider)
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
	switch resp.Protocol {
	case models.ProtocolStandardLLM:
		return fs.buildStandardCandidates(resp.Standard)
	case models.ProtocolMinion:
		return fs.buildMinionCandidates(resp.Minion)
	case models.ProtocolMinionsProtocol:
		stds, err := fs.buildStandardCandidates(resp.Standard)
		if err != nil {
			return nil, err
		}
		mins, err := fs.buildMinionCandidates(resp.Minion)
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
func (fs *FallbackService) buildStandardCandidates(std *models.StandardLLMInfo) ([]Candidate, error) {
	if std == nil {
		return nil, fmt.Errorf("standard info is nil")
	}

	var out []Candidate

	svc, err := providers.NewLLMProvider(std.Provider)
	if err != nil {
		return nil, fmt.Errorf("standard provider %s: %w", std.Provider, err)
	}
	out = append(out, Candidate{std.Provider, svc, models.ProtocolStandardLLM})

	for _, alt := range std.Alternatives {
		svc, err := providers.NewLLMProvider(alt.Provider)
		if err != nil {
			return nil, fmt.Errorf("standard alternative provider %s: %w", alt.Provider, err)
		}
		out = append(out, Candidate{alt.Provider, svc, models.ProtocolStandardLLM})
	}
	return out, nil
}

// buildMinionCandidates returns primary + fallback minions.
func (fs *FallbackService) buildMinionCandidates(min *models.MinionInfo) ([]Candidate, error) {
	if min == nil {
		return nil, fmt.Errorf("minion info is nil")
	}

	var out []Candidate

	svc, err := providers.NewLLMProviderWithBaseURL(min.Provider, nil)
	if err != nil {
		return nil, fmt.Errorf("%s model %s: %w", min.Provider, min.Model, err)
	}
	out = append(out, Candidate{min.Provider, svc, models.ProtocolMinion})

	for _, alt := range min.Alternatives {
		svc, err := providers.NewLLMProviderWithBaseURL(alt.Provider, nil)
		if err != nil {
			return nil, fmt.Errorf("%s alternative model %s: %w", alt.Provider, alt.Model, err)
		}
		out = append(out, Candidate{alt.Provider, svc, models.ProtocolMinion})
	}
	return out, nil
}

