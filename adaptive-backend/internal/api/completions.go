package api

import (
	"context"
	"fmt"
	"maps"
	"sync"
	"time"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/chat/completions"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/metrics"
	"adaptive-backend/internal/services/protocol_manager"
	"adaptive-backend/internal/services/providers"
	"adaptive-backend/internal/services/providers/provider_interfaces"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

const (
	// Circuit breaker configuration
	defaultFailureThreshold = 3
	defaultSuccessThreshold = 2
	defaultTimeout          = 10 * time.Second
	defaultResetAfter       = time.Minute

	// HTTP status codes
	statusBadRequest  = 400
	statusServerError = 503
)

// candidate represents a model/provider/protocol candidate for completion.
type candidate struct {
	Name     string
	Provider provider_interfaces.LLMProvider
	Protocol models.ProtocolType
}

// CompletionHandler handles chat completions end-to-end.
// It manages the lifecycle of chat completion requests, including provider selection,
// circuit breaking, and response handling.
type CompletionHandler struct {
	reqSvc      *completions.RequestService
	respSvc     *completions.ResponseService
	paramSvc    *completions.ParameterService
	protocolMgr *protocol_manager.ProtocolManager
	metricsSvc  *completions.MetricsService
	raceSvc     *completions.RaceService

	cbMu sync.RWMutex
	cbs  map[string]*circuitbreaker.CircuitBreaker
}

// NewCompletionHandler wires up dependencies and initializes the completion handler.
func NewCompletionHandler() *CompletionHandler {
	protocolMgr, err := protocol_manager.NewProtocolManager(0, 0)
	if err != nil {
		fiberlog.Fatalf("protocol manager initialization failed: %v", err)
	}
	chatMetrics := metrics.NewChatMetrics()

	return &CompletionHandler{
		reqSvc:      completions.NewRequestService(),
		respSvc:     completions.NewResponseService(),
		paramSvc:    completions.NewParameterService(),
		protocolMgr: protocolMgr,
		metricsSvc:  completions.NewMetricsService(chatMetrics),
		raceSvc:     completions.NewRaceService(),
		cbs:         make(map[string]*circuitbreaker.CircuitBreaker),
	}
}

// ChatCompletion handles the chat completion HTTP request.
// It processes the request through provider selection, parameter configuration,
// and response handling with circuit breaking for reliability.
func (h *CompletionHandler) ChatCompletion(c *fiber.Ctx) error {
	start := time.Now()
	reqID := h.reqSvc.GetRequestID(c)
	userID := h.reqSvc.GetAPIKey(c)
	fiberlog.Infof("[%s] starting chat completion request", reqID)

	req, err := h.reqSvc.ParseChatCompletionRequest(c)
	if err != nil {
		h.metricsSvc.RecordError(start, fmt.Sprint(statusBadRequest), false, reqID, "")
		return h.respSvc.HandleBadRequest(c, err.Error(), reqID)
	}
	isStream := req.Stream
	h.metricsSvc.RecordRequestStart(reqID, isStream)

	resp, err := h.selectProtocol(
		c.Context(), req, userID, reqID, h.copyCBs(),
	)
	if err != nil {
		h.metricsSvc.RecordError(start, fmt.Sprint(statusServerError), isStream, reqID, "")
		return h.respSvc.HandleInternalError(c, err.Error(), reqID)
	}

	// Pick parameters from the "standard" branch on MinionsProtocol
	var params models.OpenAIParameters
	switch resp.Protocol {
	case models.ProtocolStandardLLM:
		params = resp.Standard.Parameters
	case models.ProtocolMinion:
		params = resp.Minion.Parameters
	case models.ProtocolMinionsProtocol:
		params = resp.Standard.Parameters
	default:
		return h.respSvc.HandleInternalError(
			c, fmt.Sprintf("unknown protocol: %s", resp.Protocol), reqID,
		)
	}

	if err := h.paramSvc.ApplyModelParameters(req, params, reqID); err != nil {
		return h.respSvc.HandleInternalError(c, err.Error(), reqID)
	}

	return h.handleResponse(c, resp, req, start, reqID, isStream)
}

// selectProtocol runs protocol selection and returns the chosen protocol response.
func (h *CompletionHandler) selectProtocol(
	ctx context.Context,
	req *models.ChatCompletionRequest,
	userID, requestID string,
	circuitBreakers map[string]*circuitbreaker.CircuitBreaker,
) (
	resp *models.ProtocolResponse,
	err error,
) {
	fiberlog.Infof("[%s] Starting protocol selection for user: %s", requestID, userID)

	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("messages array must contain at least one element")
	}
	prompt := req.Messages[len(req.Messages)-1].OfUser.Content.OfString.Value
	if prompt == "" {
		return nil, fmt.Errorf("last message content cannot be empty")
	}

	var costBias *float32
	if req.CostBias != 0 {
		costBias = &req.CostBias
	}

	selReq := models.ModelSelectionRequest{
		Prompt:             prompt,
		ProviderConstraint: req.ProviderConstraint,
		CostBias:           costBias,
	}

	resp, _, err = h.protocolMgr.SelectProtocolWithCache(
		selReq, userID, requestID, circuitBreakers,
	)
	if err != nil {
		fiberlog.Errorf("[%s] Protocol selection error: %v", requestID, err)
		return nil, fmt.Errorf("protocol selection failed: %w", err)
	}

	return resp, nil
}

// copyCBs safely copies the circuit breaker map.
func (h *CompletionHandler) copyCBs() map[string]*circuitbreaker.CircuitBreaker {
	h.cbMu.RLock()
	defer h.cbMu.RUnlock()
	m := make(map[string]*circuitbreaker.CircuitBreaker, len(h.cbs))
	maps.Copy(m, h.cbs)
	return m
}

// buildStandardCandidates returns primary + fallback LLMs.
func (h *CompletionHandler) buildStandardCandidates(
	std *models.StandardLLMInfo,
) ([]candidate, error) {
	var out []candidate

	svc, err := providers.NewLLMProvider(std.Provider)
	if err != nil {
		return nil, fmt.Errorf("standard provider %s: %w", std.Provider, err)
	}
	out = append(out, candidate{std.Provider, svc, models.ProtocolStandardLLM})

	for _, alt := range std.Alternatives {
		svc, err := providers.NewLLMProvider(alt.Provider)
		if err != nil {
			return nil, fmt.Errorf("standard alternative provider %s: %w", alt.Provider, err)
		}
		out = append(out, candidate{alt.Provider, svc, models.ProtocolStandardLLM})
	}
	return out, nil
}

// buildMinionCandidates returns the HuggingFace + fallback LLMs.
func (h *CompletionHandler) buildMinionCandidates(
	min *models.MinionInfo,
) ([]candidate, error) {
	var out []candidate

	var baseURL *string
	if min.BaseURL != "" {
		baseURL = &min.BaseURL
	}
	
	svc, err := providers.NewLLMProviderWithBaseURL("huggingface", baseURL)
	if err != nil {
		return nil, fmt.Errorf("huggingface model %s: %w", min.Model, err)
	}
	out = append(out, candidate{"huggingface", svc, models.ProtocolMinion})

	for _, alt := range min.Alternatives {
		var altBaseURL *string
		if alt.BaseURL != "" {
			altBaseURL = &alt.BaseURL
		}
		svc, err := providers.NewLLMProviderWithBaseURL("huggingface", altBaseURL)
		if err != nil {
			return nil, fmt.Errorf("huggingface alternative model %s: %w", alt.Model, err)
		}
		out = append(out, candidate{alt.Model, svc, models.ProtocolMinion})
	}
	return out, nil
}

// buildCandidates flattens resp into an ordered slice.
// For ProtocolMinionsProtocol, returns a pair: [remote, minion]
func (h *CompletionHandler) buildCandidates(
	resp *models.ProtocolResponse,
) ([]candidate, error) {
	switch resp.Protocol {
	case models.ProtocolStandardLLM:
		return h.buildStandardCandidates(resp.Standard)
	case models.ProtocolMinion:
		return h.buildMinionCandidates(resp.Minion)
	case models.ProtocolMinionsProtocol:
		stds, err := h.buildStandardCandidates(resp.Standard)
		if err != nil {
			return nil, err
		}
		mins, err := h.buildMinionCandidates(resp.Minion)
		if err != nil {
			return nil, err
		}
		if len(stds) > 0 && len(mins) > 0 {
			return []candidate{
				{stds[0].Name, stds[0].Provider, models.ProtocolStandardLLM},
				{mins[0].Name, mins[0].Provider, models.ProtocolMinion},
			}, nil
		}
		return append(stds, mins...), nil
	default:
		return nil, fmt.Errorf("unsupported protocol %s", resp.Protocol)
	}
}

// handleResponse tries each candidate under its circuit-breaker.
func (h *CompletionHandler) handleResponse(
	c *fiber.Ctx,
	resp *models.ProtocolResponse,
	req *models.ChatCompletionRequest,
	start time.Time,
	reqID string,
	isStream bool,
) error {
	cands, err := h.buildCandidates(resp)
	if err != nil {
		return h.respSvc.HandleInternalError(c, err.Error(), reqID)
	}

	var lastErr error
	for _, cand := range cands {
		cb := h.getOrCreateCB(cand.Name)
		if cb.GetState() == circuitbreaker.Open || !cb.CanExecute() {
			cb.RecordFailure()
			continue
		}

		var remoteProv, minionProv provider_interfaces.LLMProvider
		if resp.Protocol == models.ProtocolMinionsProtocol {
			for _, candidate := range cands {
				switch candidate.Protocol {
				case models.ProtocolStandardLLM:
					remoteProv = candidate.Provider
				case models.ProtocolMinion:
					minionProv = candidate.Provider
				}
			}
			if remoteProv == nil || minionProv == nil {
				cb.RecordFailure()
				lastErr = fmt.Errorf("MinionsProtocol: missing remote or minion provider")
				continue
			}
		} else if cand.Protocol == models.ProtocolMinion {
			minionProv = cand.Provider
		} else {
			remoteProv = cand.Provider
		}

		if err := h.respSvc.HandleProtocol(
			c,
			resp.Protocol,
			&remoteProv,
			&minionProv,
			req,
			resp,
			reqID,
			isStream,
		); err != nil {
			cb.RecordFailure()
			lastErr = err
			continue
		}

		cb.RecordSuccess()
		h.metricsSvc.RecordSuccess(start, isStream, reqID, cand.Name)
		return nil
	}

	h.metricsSvc.RecordError(start, fmt.Sprint(statusServerError), isStream, reqID, "all")
	if lastErr != nil {
		return lastErr
	}
	return h.respSvc.HandleInternalError(c, "all providers failed", reqID)
}

// getOrCreateCB returns or initializes a circuit-breaker.
// It uses double-checked locking pattern for thread safety and efficiency.
func (h *CompletionHandler) getOrCreateCB(name string) *circuitbreaker.CircuitBreaker {
	h.cbMu.RLock()
	cb, ok := h.cbs[name]
	h.cbMu.RUnlock()
	if ok {
		return cb
	}

	h.cbMu.Lock()
	defer h.cbMu.Unlock()
	if cb, ok = h.cbs[name]; ok {
		return cb
	}

	cfg := circuitbreaker.Config{
		FailureThreshold: defaultFailureThreshold,
		SuccessThreshold: defaultSuccessThreshold,
		Timeout:          defaultTimeout,
		ResetAfter:       defaultResetAfter,
	}
	cb = circuitbreaker.NewWithConfig(cfg)
	h.cbs[name] = cb
	return cb
}

// Health returns protocol manager health.
func (h *CompletionHandler) Health() error {
	return h.protocolMgr.ValidateContext()
}
