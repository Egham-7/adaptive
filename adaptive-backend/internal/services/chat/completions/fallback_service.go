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
	case FallbackModeRace:
		return fs.raceAlternatives(ctx, alternatives, providerConfigs, requestID)
	case FallbackModeSequential:
		return fs.selectSequentialAlternative(ctx, alternatives, providerConfigs, requestID)
	default:
		return nil, fmt.Errorf("unsupported fallback mode: %d", fs.mode)
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

// raceAllProviders races all providers in parallel
func (fs *FallbackService) raceAllProviders(
	ctx context.Context,
	options []models.Alternative,
	providerConfigs map[string]*models.ProviderConfig,
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
			result := fs.tryProviderConnection(opt, providerConfigs, requestID)

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
	provider, err = providers.NewLLMProvider(option.Provider, providerConfigs)
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

	svc, err := providers.NewLLMProvider(std.Provider, nil)
	if err != nil {
		return nil, fmt.Errorf("standard provider %s: %w", std.Provider, err)
	}
	out = append(out, Candidate{std.Provider, svc, models.ProtocolStandardLLM})

	for _, alt := range std.Alternatives {
		svc, err := providers.NewLLMProvider(alt.Provider, nil)
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

	svc, err := providers.NewLLMProvider(min.Provider, nil)
	if err != nil {
		return nil, fmt.Errorf("%s model %s: %w", min.Provider, min.Model, err)
	}
	out = append(out, Candidate{min.Provider, svc, models.ProtocolMinion})

	for _, alt := range min.Alternatives {
		svc, err := providers.NewLLMProvider(alt.Provider, nil)
		if err != nil {
			return nil, fmt.Errorf("%s alternative model %s: %w", alt.Provider, alt.Model, err)
		}
		out = append(out, Candidate{alt.Provider, svc, models.ProtocolMinion})
	}
	return out, nil
}
