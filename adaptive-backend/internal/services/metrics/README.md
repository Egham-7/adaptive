# Metrics Module

A comprehensive metrics collection system for monitoring chat completion performance, caching efficiency, and model usage patterns in the Adaptive Backend. This module uses Prometheus for metrics collection and provides observability into the LLM provider interactions.

## Overview

The metrics module provides instrumentation for tracking key performance indicators (KPIs) and operational metrics across the chat completion system. It enables monitoring, alerting, and performance optimization through detailed telemetry data.

## Architecture

```
metrics/
├── chat-metrics.go     # Chat completion metrics definitions
└── README.md          # This documentation
```

## Metrics Categories

### Chat Completion Metrics

The module tracks four primary categories of metrics:

1. **Performance Metrics** - Request duration and latency
2. **Cache Metrics** - Cache hit rates and lookup patterns
3. **Model Usage** - Model selection frequency
4. **System Health** - Overall system performance indicators

## Metrics Definitions

### ChatMetrics Structure

```go
type ChatMetrics struct {
    RequestDuration *prometheus.HistogramVec
    CacheHits       *prometheus.CounterVec
    ModelSelections *prometheus.CounterVec
    CacheLookups    *prometheus.CounterVec
}
```

### Individual Metrics

#### Request Duration Histogram
- **Name**: `chat_completion_duration_seconds`
- **Type**: Histogram
- **Labels**: `endpoint`, `status`
- **Purpose**: Tracks the time taken for chat completion requests
- **Buckets**: Default Prometheus histogram buckets (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

```go
RequestDuration: promauto.NewHistogramVec(prometheus.HistogramOpts{
    Name: "chat_completion_duration_seconds",
    Help: "Duration of chat completion requests",
}, []string{"endpoint", "status"})
```

#### Cache Hits Counter
- **Name**: `chat_completion_cache_hits_total`
- **Type**: Counter
- **Labels**: `cache_type`
- **Purpose**: Counts successful cache hits by cache type

```go
CacheHits: promauto.NewCounterVec(prometheus.CounterOpts{
    Name: "chat_completion_cache_hits_total",
    Help: "Number of cache hits",
}, []string{"cache_type"})
```

#### Model Selections Counter
- **Name**: `chat_completion_model_selections_total`
- **Type**: Counter
- **Labels**: `model`
- **Purpose**: Tracks frequency of model usage

```go
ModelSelections: promauto.NewCounterVec(prometheus.CounterOpts{
    Name: "chat_completion_model_selections_total",
    Help: "Number of times each model was selected",
}, []string{"model"})
```

#### Cache Lookups Counter
- **Name**: `chat_completion_cache_lookups_total`
- **Type**: Counter
- **Labels**: `cache_type`
- **Purpose**: Tracks all cache lookup attempts by type

```go
CacheLookups: promauto.NewCounterVec(prometheus.CounterOpts{
    Name: "chat_completion_cache_lookups_total",
    Help: "Number of cache lookups by type (user/global/miss)",
}, []string{"cache_type"})
```

## Usage Examples

### Initialization

```go
import "adaptive-backend/internal/services/metrics"

// Initialize metrics
chatMetrics := metrics.NewChatMetrics()
```

### Recording Request Duration

```go
import "time"

// Start timer
start := time.Now()

// Process request...
// (chat completion logic)

// Record duration
duration := time.Since(start)
chatMetrics.RequestDuration.WithLabelValues("chat", "success").Observe(duration.Seconds())
```

### Recording Cache Operations

```go
// Cache hit
chatMetrics.CacheHits.WithLabelValues("user").Inc()

// Cache lookup
chatMetrics.CacheLookups.WithLabelValues("user").Inc()

// Cache miss
chatMetrics.CacheLookups.WithLabelValues("miss").Inc()
```

### Recording Model Usage

```go
// Track model selection
chatMetrics.ModelSelections.WithLabelValues("gpt-4o").Inc()
chatMetrics.ModelSelections.WithLabelValues("claude-3-sonnet").Inc()
```

## Label Values

### Endpoint Labels
- `chat` - Standard chat completion endpoint
- `stream` - Streaming chat completion endpoint
- `function` - Function calling endpoint

### Status Labels
- `success` - Successful request
- `error` - Failed request
- `timeout` - Request timeout
- `rate_limited` - Rate limited request

### Cache Type Labels
- `user` - User-specific cache
- `global` - Global cache
- `miss` - Cache miss
- `redis` - Redis cache
- `memory` - In-memory cache

### Model Labels
Common model identifiers:
- `gpt-4o`
- `gpt-4`
- `claude-3-sonnet`
- `claude-3-haiku`
- `llama-3.1-70b`
- `deepseek-chat`
- `gemini-pro`

## Integration with Monitoring Stack

### Prometheus Configuration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'adaptive-backend'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana Dashboard Queries

#### Request Rate
```promql
rate(chat_completion_duration_seconds_count[5m])
```

#### Average Response Time
```promql
rate(chat_completion_duration_seconds_sum[5m]) / rate(chat_completion_duration_seconds_count[5m])
```

#### Cache Hit Rate
```promql
rate(chat_completion_cache_hits_total[5m]) / rate(chat_completion_cache_lookups_total[5m])
```

#### Model Usage Distribution
```promql
topk(10, rate(chat_completion_model_selections_total[1h]))
```

#### Error Rate
```promql
rate(chat_completion_duration_seconds_count{status="error"}[5m]) / rate(chat_completion_duration_seconds_count[5m])
```

### Alerting Rules

#### High Error Rate
```yaml
- alert: HighChatCompletionErrorRate
  expr: rate(chat_completion_duration_seconds_count{status="error"}[5m]) / rate(chat_completion_duration_seconds_count[5m]) > 0.05
  for: 2m
  annotations:
    summary: High error rate in chat completions
    description: Error rate is {{ $value | humanizePercentage }}
```

#### High Response Time
```yaml
- alert: HighChatCompletionLatency
  expr: histogram_quantile(0.95, rate(chat_completion_duration_seconds_bucket[5m])) > 5
  for: 5m
  annotations:
    summary: High chat completion latency
    description: 95th percentile latency is {{ $value }}s
```

#### Low Cache Hit Rate
```yaml
- alert: LowCacheHitRate
  expr: rate(chat_completion_cache_hits_total[5m]) / rate(chat_completion_cache_lookups_total[5m]) < 0.3
  for: 10m
  annotations:
    summary: Low cache hit rate
    description: Cache hit rate is {{ $value | humanizePercentage }}
```

## Metric Collection Patterns

### Request Lifecycle Tracking

```go
func (h *ChatHandler) HandleChatCompletion(w http.ResponseWriter, r *http.Request) {
    start := time.Now()
    status := "success"
    
    defer func() {
        duration := time.Since(start)
        h.metrics.RequestDuration.WithLabelValues("chat", status).Observe(duration.Seconds())
    }()
    
    // Handle request...
    if err != nil {
        status = "error"
        return
    }
}
```

### Cache Performance Tracking

```go
func (c *ChatCache) Get(key string) (interface{}, bool) {
    c.metrics.CacheLookups.WithLabelValues("user").Inc()
    
    if value, exists := c.store[key]; exists {
        c.metrics.CacheHits.WithLabelValues("user").Inc()
        return value, true
    }
    
    c.metrics.CacheLookups.WithLabelValues("miss").Inc()
    return nil, false
}
```

### Model Selection Tracking

```go
func (p *ProviderRouter) SelectModel(request *ChatRequest) string {
    model := p.selectBestModel(request)
    p.metrics.ModelSelections.WithLabelValues(model).Inc()
    return model
}
```

## Performance Considerations

### Metric Collection Overhead

- **Counters**: Minimal overhead (~5ns per increment)
- **Histograms**: Low overhead (~100ns per observation)
- **Label Cardinality**: Keep label combinations under 10,000 per metric

### Memory Usage

- Each metric series consumes ~1KB of memory
- High cardinality labels can cause memory issues
- Use label aggregation for high-volume metrics

### Best Practices

1. **Avoid High Cardinality**: Don't use user IDs or request IDs as labels
2. **Use Meaningful Labels**: Labels should be useful for aggregation
3. **Consistent Naming**: Follow Prometheus naming conventions
4. **Documentation**: Document all custom metrics

## Testing

### Unit Tests

```go
func TestChatMetrics_RequestDuration(t *testing.T) {
    metrics := NewChatMetrics()
    
    // Record a duration
    metrics.RequestDuration.WithLabelValues("chat", "success").Observe(0.1)
    
    // Verify metric was recorded
    metric := &dto.Metric{}
    err := metrics.RequestDuration.WithLabelValues("chat", "success").Write(metric)
    assert.NoError(t, err)
    assert.Equal(t, uint64(1), metric.GetHistogram().GetSampleCount())
}
```

### Integration Tests

```go
func TestMetricsEndpoint(t *testing.T) {
    server := setupTestServer()
    defer server.Close()
    
    resp, err := http.Get(server.URL + "/metrics")
    require.NoError(t, err)
    defer resp.Body.Close()
    
    body, err := io.ReadAll(resp.Body)
    require.NoError(t, err)
    
    assert.Contains(t, string(body), "chat_completion_duration_seconds")
}
```

## Future Enhancements

### Planned Metrics

- **Token Usage**: Track input/output tokens per request
- **Cost Metrics**: Track estimated costs per provider
- **Provider Performance**: Response times by provider
- **User Metrics**: Request patterns by user
- **Quality Metrics**: Response quality scores

### Advanced Features

- **Custom Dashboards**: Pre-built Grafana dashboards
- **Metric Aggregation**: Automatic metric rollups
- **Real-time Alerts**: Integration with alerting systems
- **Cost Analytics**: Cost tracking and optimization metrics

## Dependencies

```go
import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)
```

## Contributing

When adding new metrics:

1. Follow Prometheus naming conventions
2. Use appropriate metric types (Counter, Gauge, Histogram, Summary)
3. Add comprehensive documentation
4. Include usage examples
5. Consider label cardinality impact
6. Add unit tests for new metrics

## License

This module is part of the Adaptive Backend project and follows the same licensing terms.