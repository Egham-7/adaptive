package completions

import (
	"fmt"
	"time"

	"adaptive-backend/internal/services/metrics"

	fiberlog "github.com/gofiber/fiber/v2/log"
)

const (
	methodCompletion = "completion"
	methodStream     = "stream"
	statusOK         = "200"
)

// MetricsService handles metrics recording for chat completions
type MetricsService struct {
	chatMetrics *metrics.ChatMetrics
}

// NewMetricsService creates a new metrics service
func NewMetricsService(chatMetrics *metrics.ChatMetrics) *MetricsService {
	return &MetricsService{
		chatMetrics: chatMetrics,
	}
}

// RecordRequestStart records the start of a request
func (s *MetricsService) RecordRequestStart(requestID string, isStream bool) {
	method := methodCompletion
	if isStream {
		method = methodStream
	}
	fiberlog.Debugf("[%s] Recording request start for method: %s", requestID, method)
}

// RecordSuccess records a successful request completion
func (s *MetricsService) RecordSuccess(start time.Time, isStream bool, requestID, provider string) {
	if s.chatMetrics == nil {
		return
	}

	method := methodCompletion
	if isStream {
		method = methodStream
	}

	duration := time.Since(start).Seconds()

	s.chatMetrics.RequestDuration.
		WithLabelValues(method, statusOK, provider).
		Observe(duration)

	fiberlog.Infof("[%s] Recorded success: method=%s, provider=%s, duration=%.3fs",
		requestID, method, provider, duration)
}

// RecordError records an error during request processing
func (s *MetricsService) RecordError(start time.Time, statusCode string, isStream bool, requestID, provider string) {
	if s.chatMetrics == nil {
		return
	}

	method := methodCompletion
	if isStream {
		method = methodStream
	}

	duration := time.Since(start).Seconds()

	s.chatMetrics.RequestDuration.
		WithLabelValues(method, statusCode, provider).
		Observe(duration)

	fiberlog.Warnf("[%s] Recorded error: method=%s, status=%s, provider=%s, duration=%.3fs",
		requestID, method, statusCode, provider, duration)
}

// RecordCacheHit records a cache hit event
func (s *MetricsService) RecordCacheHit(cacheType, requestID, provider string) {
	if s.chatMetrics == nil {
		return
	}

	s.chatMetrics.CacheHits.
		WithLabelValues(cacheType, provider).
		Inc()

	fiberlog.Infof("[%s] Recorded cache hit: type=%s, provider=%s", requestID, cacheType, provider)
}

// RecordProtocolSelection records which protocol was selected for the request
func (s *MetricsService) RecordProtocolSelection(model, requestID, provider string) {
	if s.chatMetrics == nil {
		return
	}

	s.chatMetrics.ProtocolSelections.
		WithLabelValues(model, provider).
		Inc()

	fiberlog.Infof("[%s] Recorded protocol selection: model=%s, provider=%s", requestID, model, provider)
}

// RecordMinionSelection records when a minion is selected for the request
func (s *MetricsService) RecordMinionSelection(taskType, requestID string) {
	s.RecordProtocolSelection(fmt.Sprintf("Minion-%s", taskType), requestID, "minion")
}

// RecordParameterApplication records parameter application metrics
func (s *MetricsService) RecordParameterApplication(paramType, requestID string) {
	fiberlog.Debugf("[%s] Applied parameter: %s", requestID, paramType)
}

// RecordValidationError records validation errors
func (s *MetricsService) RecordValidationError(validationType, requestID string) {
	fiberlog.Warnf("[%s] Validation error: %s", requestID, validationType)
}

// RecordProviderLatency records the latency of provider calls
func (s *MetricsService) RecordProviderLatency(provider, operation string, duration time.Duration, requestID string) {
	if s.chatMetrics == nil {
		return
	}

	fiberlog.Debugf("[%s] Provider latency: provider=%s, operation=%s, duration=%.3fs",
		requestID, provider, operation, duration.Seconds())
}

// GetMetricsSnapshot returns a snapshot of current metrics (for debugging/monitoring)
func (s *MetricsService) GetMetricsSnapshot(requestID string) map[string]any {
	snapshot := map[string]any{
		"metrics_enabled": s.chatMetrics != nil,
	}

	fiberlog.Debugf("[%s] Metrics snapshot: %+v", requestID, snapshot)
	return snapshot
}
