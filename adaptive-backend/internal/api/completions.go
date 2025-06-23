package api

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/chat/completions"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/metrics"
	"adaptive-backend/internal/services/minions"
	"adaptive-backend/internal/services/model_selection"
	"adaptive-backend/internal/services/providers"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"fmt"
	"maps"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gofiber/fiber/v2"
	fiberlog "github.com/gofiber/fiber/v2/log"
)

type candidate struct {
	Name     string
	Provider provider_interfaces.LLMProvider
	Protocol models.ProtocolType
}

// CompletionHandler handles chat completions end-to-end.
type CompletionHandler struct {
	reqSvc         *completions.RequestService
	respSvc        *completions.ResponseService
	paramSvc       *completions.ParameterService
	orchSvc        *completions.OrchestrationService
	metricsSvc     *completions.MetricsService
	raceSvc        *completions.RaceService
	minionRegistry *minions.MinionRegistry

	cbMu sync.RWMutex
	cbs  map[string]*circuitbreaker.CircuitBreaker
}

// NewCompletionHandler wires up dependencies.
func NewCompletionHandler() *CompletionHandler {
	mr := minions.NewMinionRegistry(11)
	registerMinions(mr)

	modelSel, err := model_selection.NewModelSelector(0, 0)
	if err != nil {
		fiberlog.Fatalf("model selector init: %v", err)
	}
	chatMetrics := metrics.NewChatMetrics()

	return &CompletionHandler{
		reqSvc:         completions.NewRequestService(),
		respSvc:        completions.NewResponseService(),
		paramSvc:       completions.NewParameterService(),
		orchSvc:        completions.NewOrchestrationService(modelSel, mr),
		metricsSvc:     completions.NewMetricsService(chatMetrics),
		raceSvc:        completions.NewRaceService(mr),
		minionRegistry: mr,
		cbs:            make(map[string]*circuitbreaker.CircuitBreaker),
	}
}

// ChatCompletion is the HTTP handler.
func (h *CompletionHandler) ChatCompletion(c *fiber.Ctx) error {
	start := time.Now()
	reqID := h.reqSvc.GetRequestID(c)
	userID := h.reqSvc.GetAPIKey(c)
	fiberlog.Infof("[%s] start", reqID)

	req, err := h.reqSvc.ParseChatCompletionRequest(c)
	if err != nil {
		h.metricsSvc.RecordError(start, "400", false, reqID, "")
		return h.respSvc.HandleBadRequest(c, err.Error(), reqID)
	}
	isStream := req.Stream
	h.metricsSvc.RecordRequestStart(reqID, isStream)

	resp, err := h.orchSvc.SelectAndConfigureProvider(
		c.Context(), req, userID, reqID, h.copyCBs(),
	)
	if err != nil {
		h.metricsSvc.RecordError(start, "500", isStream, reqID, "")
		return h.respSvc.HandleInternalError(c, err.Error(), reqID)
	}

	// pick parameters from the "standard" branch on MinionsProtocol
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
			c, "unknown protocol "+string(resp.Protocol), reqID,
		)
	}

	if err := h.paramSvc.ApplyModelParameters(req, params, reqID); err != nil {
		return h.respSvc.HandleInternalError(c, err.Error(), reqID)
	}

	return h.handleResponse(c, resp, req, start, reqID, isStream)
}

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
	var cleanup []func() error
	defer func() {
		if r := recover(); r != nil {
			for _, fn := range cleanup {
				fn()
			}
			panic(r)
		}
	}()

	svc, err := providers.NewLLMProvider(std.Provider, nil, h.minionRegistry)
	if err != nil {
		for _, fn := range cleanup {
			fn()
		}
		return nil, fmt.Errorf("standard %s: %w", std.Provider, err)
	}
	if closer, ok := svc.(interface{ Close() error }); ok {
		cleanup = append(cleanup, closer.Close)
	}
	out = append(out, candidate{std.Provider, svc, models.ProtocolStandardLLM})
	for _, alt := range std.Alternatives {
		svc, err := providers.NewLLMProvider(alt.Provider, nil, h.minionRegistry)
		if err != nil {
			for _, fn := range cleanup {
				fn()
			}
			return nil, fmt.Errorf("standard alt %s: %w", alt.Provider, err)
		}
		if closer, ok := svc.(interface{ Close() error }); ok {
			cleanup = append(cleanup, closer.Close)
		}
		out = append(out, candidate{alt.Provider, svc, models.ProtocolStandardLLM})
	}
	return out, nil
}

// buildMinionCandidates returns the minion + fallback LLMs.
func (h *CompletionHandler) buildMinionCandidates(
	min *models.MinionInfo,
) ([]candidate, error) {
	var out []candidate
	var cleanup []func() error

	defer func() {
		if r := recover(); r != nil {
			for _, fn := range cleanup {
				fn()
			}
			panic(r)
		}
	}()

	tt := min.TaskType
	svc, err := providers.NewLLMProvider("minion", &tt, h.minionRegistry)
	if err != nil {
		for _, fn := range cleanup {
			fn()
		}
		return nil, fmt.Errorf("minion %s: %w", tt, err)
	}
	if closer, ok := svc.(interface{ Close() error }); ok {
		cleanup = append(cleanup, closer.Close)
	}
	out = append(out, candidate{"minion", svc, models.ProtocolMinion})
	for _, alt := range min.Alternatives {
		svc, err := providers.NewLLMProvider(alt.Provider, nil, h.minionRegistry)
		if err != nil {
			for _, fn := range cleanup {
				fn()
			}
			return nil, fmt.Errorf("minion alt %s: %w", alt.Provider, err)
		}
		if closer, ok := svc.(interface{ Close() error }); ok {
			cleanup = append(cleanup, closer.Close)
		}
		out = append(out, candidate{alt.Provider, svc, models.ProtocolStandardLLM})
	}
	return out, nil
}

// buildCandidates flattens resp into an ordered slice.
func (h *CompletionHandler) buildCandidates(
	resp *models.OrchestratorResponse,
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
		return append(stds, mins...), nil
	default:
		return nil, fmt.Errorf("unsupported protocol %s", resp.Protocol)
	}
}

// handleResponse tries each candidate under its circuit-breaker.
func (h *CompletionHandler) handleResponse(
	c *fiber.Ctx,
	resp *models.OrchestratorResponse,
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

		// we always pass both slots; only one is non-nil per protocol
		var remoteProv, minionProv provider_interfaces.LLMProvider
		if cand.Protocol == models.ProtocolMinion {
			minionProv = cand.Provider
		} else {
			remoteProv = cand.Provider
		}

		if err := h.respSvc.HandleProtocol(
			c,
			cand.Protocol,
			&remoteProv,
			&minionProv,
			req,
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

	h.metricsSvc.RecordError(start, "503", isStream, reqID, "all")
	if lastErr != nil {
		return lastErr
	}
	return h.respSvc.HandleInternalError(c, "all providers failed", reqID)
}

// getOrCreateCB returns or initializes a circuit-breaker.
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
		FailureThreshold: 3,
		SuccessThreshold: 2,
		Timeout:          10 * time.Second,
		ResetAfter:       time.Minute,
	}
	cb = circuitbreaker.NewWithConfig(cfg)
	h.cbs[name] = cb
	return cb
}

func registerMinions(reg *minions.MinionRegistry) {
	for _, t := range models.ValidTaskTypes() {
		key := strings.ToUpper(string(t)) + "_MINION_URL"
		url := os.Getenv(key)
		reg.RegisterMinion(string(t), url)
	}
}

// Health returns orchestration service health.
func (h *CompletionHandler) Health() error {
	return h.orchSvc.ValidateOrchestrationContext()
}
