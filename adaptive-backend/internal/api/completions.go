package api

import (
	"fmt"
	"maps"
	"os"
	"sync"
	"time"

	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/chat/completions"
	"adaptive-backend/internal/services/circuitbreaker"
	"adaptive-backend/internal/services/metrics"
	"adaptive-backend/internal/services/minions"
	"adaptive-backend/internal/services/protocol_manager"
	"adaptive-backend/internal/services/providers"
	"adaptive-backend/internal/services/providers/provider_interfaces"

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

	protocolMgr, err := protocol_manager.NewProtocolManager(0, 0)
	if err != nil {
		fiberlog.Fatalf("protocol manager init: %v", err)
	}
	chatMetrics := metrics.NewChatMetrics()

	return &CompletionHandler{
		reqSvc:         completions.NewRequestService(),
		respSvc:        completions.NewResponseService(),
		paramSvc:       completions.NewParameterService(),
		orchSvc:        completions.NewOrchestrationService(protocolMgr, mr),
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

	svc, err := providers.NewLLMProvider(std.Provider, nil, h.minionRegistry)
	if err != nil {
		return nil, fmt.Errorf("standard %s: %w", std.Provider, err)
	}
	out = append(out, candidate{std.Provider, svc, models.ProtocolStandardLLM})
	for _, alt := range std.Alternatives {
		svc, err := providers.NewLLMProvider(alt.Provider, nil, h.minionRegistry)
		if err != nil {
			return nil, fmt.Errorf("standard alt %s: %w", alt.Provider, err)
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

	tt := min.TaskType
	svc, err := providers.NewLLMProvider("minion", &tt, h.minionRegistry)
	if err != nil {
		return nil, fmt.Errorf("minion %s: %w", tt, err)
	}
	out = append(out, candidate{"minion", svc, models.ProtocolMinion})
	for _, alt := range min.Alternatives {
		svc, err := providers.NewLLMProvider(alt.Provider, nil, h.minionRegistry)
		if err != nil {
			return nil, fmt.Errorf("minion alt %s: %w", alt.Provider, err)
		}
		out = append(out, candidate{alt.Provider, svc, models.ProtocolStandardLLM})
	}
	return out, nil
}

// buildCandidates flattens resp into an ordered slice.
// For ProtocolMinionsProtocol, returns a pair: [remote, minion]
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

func getEnvOrDefault(key, defaultValue string) string {
	if value, exists := os.LookupEnv(key); exists && value != "" {
		return value
	}
	return defaultValue
}

func registerMinions(registry *minions.MinionRegistry) {
	registry.RegisterMinion("Open QA", getEnvOrDefault("OPEN_QA_MINION_URL", ""))
	registry.RegisterMinion("Closed QA", getEnvOrDefault("CLOSED_QA_MINION_URL", ""))
	registry.RegisterMinion("Summarization", getEnvOrDefault("OPEN_QA_MINION_URL", ""))
	registry.RegisterMinion("Text Generation", getEnvOrDefault("OPEN_QA_MINION_URL", ""))
	registry.RegisterMinion("Classification", getEnvOrDefault("CLASSIFICATION_MINION_URL", ""))
	registry.RegisterMinion("Code Generation", getEnvOrDefault("CODE_GENERATION_MINION_URL", ""))
	registry.RegisterMinion("Chatbot", getEnvOrDefault("CHATBOT_MINION_URL", ""))
	registry.RegisterMinion("Rewrite", getEnvOrDefault("REWRITE_MINION_URL", ""))
	registry.RegisterMinion("Brainstorming", getEnvOrDefault("BRAINSTORMING_MINION_URL", ""))
	registry.RegisterMinion("Extraction", getEnvOrDefault("EXTRACTION_MINION_URL", ""))
	registry.RegisterMinion("Other", getEnvOrDefault("CHATBOT_MINION_URL", ""))
}

// Health returns orchestration service health.
func (h *CompletionHandler) Health() error {
	return h.orchSvc.ValidateOrchestrationContext()
}
