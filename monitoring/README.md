# Adaptive Monitoring

Monitoring and observability setup for the Adaptive AI platform using Prometheus and Grafana.

## Overview

Complete monitoring solution for collecting metrics, alerting, and visualization across all Adaptive services.

### Components

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards  
- **Alert Manager**: Alert routing and notifications (optional)

## Quick Start

```bash
# Start monitoring stack
docker-compose up -d prometheus grafana

# Access dashboards
open http://localhost:3001  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus
```

## Metrics Collected

### Backend API (Go)
- HTTP request duration and count
- Cache hit/miss rates
- Model selection frequency
- Database connection pool stats

### AI Service (Python)
- Model inference latency
- Prompt classification accuracy
- Memory and CPU usage
- Request throughput

### Frontend (React)
- Web vitals (LCP, FID, CLS)
- User interactions and error rates
- Bundle size metrics

## Grafana Dashboards

### System Overview
- Service health status
- Request rates and latency
- Error rates and resource utilization

### Cost Analytics
- Token usage by model/provider
- Cost per request
- Monthly spending trends

### Business Metrics
- Active users and conversations
- Model selection distribution
- Feature usage patterns

## Configuration

```
monitoring/
├── prometheus.dev.yml      # Development config
├── prometheus.prod.yml     # Production config
├── grafana/               # Grafana configuration
│   └── plugins/          # Pre-installed plugins
└── alerts/               # Alert rules
```

## Custom Metrics

### Backend (Go)
```go
var customCounter = prometheus.NewCounterVec(
    prometheus.CounterOpts{
        Name: "adaptive_custom_total",
        Help: "Custom metric description",
    },
    []string{"label1", "label2"},
)

// Increment metric
customCounter.WithLabelValues("value1", "value2").Inc()
```

### Frontend (JavaScript)
```typescript
// Send custom metric to backend
await fetch('/api/metrics', {
  method: 'POST',
  body: JSON.stringify({
    metric: 'user_action',
    value: 1,
    labels: { action: 'button_click', page: 'dashboard' }
  })
});
```

## Alerting

### Alert Rules

Create alert rules in `alerts/` directory:

```yaml
# alerts/application.yml
groups:
  - name: adaptive_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
```

### Notification Channels
- Slack notifications
- Email alerts
- PagerDuty integration
- Discord webhooks

## Key Performance Indicators

- **Response Time**: API endpoint latency percentiles
- **Throughput**: Requests per second
- **Error Rate**: 4xx/5xx response percentage
- **Availability**: Service uptime percentage
- **Cost Efficiency**: Cost per successful request

## Deployment

### Development
```bash
docker-compose -f docker-compose.yml up prometheus grafana
```

### Production
```bash
# With persistent storage and backup
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes
```bash
kubectl apply -f k8s/monitoring/
```

## Data Retention

- **Prometheus**: 30 days local storage
- **Long-term**: Export to cloud storage
- **Grafana**: Dashboard configs in version control

## Security

- Grafana admin password via environment variables
- Prometheus scrape authentication
- Network policies for service isolation
- SSL/TLS for external access

## Health Checks

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check Grafana health
curl http://localhost:3001/api/health

# View metrics endpoint
curl http://localhost:8080/metrics
```

## Backup and Recovery

```bash
# Backup Prometheus data
docker exec prometheus tar -czf /backup/prometheus-$(date +%Y%m%d).tar.gz /prometheus

# Backup Grafana dashboards
docker exec grafana tar -czf /backup/grafana-$(date +%Y%m%d).tar.gz /var/lib/grafana
```